"""The in-loop Adam step's capture-safe prerequisites: what the eager step
skips, and what it may sync on."""

import pytest
import torch
from torch.optim import Adam
from torch.profiler import ProfilerActivity, profile

from cutamp.config import TAMPConfiguration
from cutamp.costs import _sphere_to_sphere_overlap, get_aabb_from_spheres
from cutamp.tests.conftest import (
    SpyVis, clone_particles, make_plan_info, needs_cuda, run_optimizer, sync_debug_error,
)
from cutamp.tests.test_step_lockstep import _head_upper_sum

pytestmark = pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")


@needs_cuda
@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_noop_visualizer_skips_vis_block(blocks_t1_plan, cuda_graphs):
    world, skeleton, base = blocks_t1_plan
    config = TAMPConfiguration(num_particles=64, num_opt_steps=12, early_stop_patience=0,
                               cuda_graphs=cuda_graphs, enable_visualizer=False)
    _, timer = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config)
    assert "visualize_opt_rollout" not in timer.get_summaries()  # the no-op visualizer skips the block
    vis = SpyVis(config)
    run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config, visualizer=vis)
    assert vis.opt_steps == [0, 10, 11]


@needs_cuda
def test_rollout_viewer_takes_the_conf_and_the_gripper_state(blocks_t1_plan):
    # The rollout_* timeline shows each conf in the model's cspace, with each hand's gripper set on the rerun robot:
    # both open at the start, then a pick conf's hand closed and a place conf's open, the other hand as it was.
    from types import SimpleNamespace

    from cutamp.robots.t1 import GRIPPER_CLOSED, GRIPPER_OPEN
    world, skeleton, base = blocks_t1_plan
    config = TAMPConfiguration(num_particles=64, num_opt_steps=2, early_stop_patience=0, enable_visualizer=False)

    class Vis(SpyVis):
        def __init__(self, config):
            super().__init__(config)
            self.timeline, self.grippers, self.shown = "", None, []
            self.robot = SimpleNamespace(set_grippers=lambda left, right: setattr(self, "grippers", (left, right)))

        def set_time_sequence(self, timeline: str, val: int) -> None:
            super().set_time_sequence(timeline, val)
            self.timeline = timeline

        def set_joint_positions(self, q) -> None:
            if self.timeline.startswith("rollout_"):
                self.shown.append((len(q), self.grippers))

    vis = Vis(config)
    run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config, visualizer=vis)
    dof, state = len(world.model.joint_names), {"left": GRIPPER_OPEN, "right": GRIPPER_OPEN}
    want = [(dof, (GRIPPER_OPEN, GRIPPER_OPEN))]
    for op in skeleton:
        metadata = op.operator.metadata
        if metadata.is_actionable:
            state[metadata.arm] = GRIPPER_CLOSED if metadata.action_type == "pick" else GRIPPER_OPEN
            want.append((dof, (state["left"], state["right"])))
    assert len(want) > 2 and vis.shown == want


# cfa7a46's sphere_to_sphere_overlap, frozen: the masked AABB form, which syncs on intersect.any().
def _head_sphere_to_sphere_overlap(
    spheres_1: torch.Tensor,
    spheres_2: torch.Tensor,
    activation_distance: float | None = None,
    use_aabb_check: bool = False,
) -> torch.Tensor:
    if not use_aabb_check:
        return _sphere_to_sphere_overlap(spheres_1, spheres_2, activation_distance)

    aabb_1 = get_aabb_from_spheres(spheres_1)  # [b, *h, 2, 3]
    aabb_2 = get_aabb_from_spheres(spheres_2)  # [b, *h, 2, 3]

    min_1, max_1 = aabb_1.unbind(-2)
    min_2, max_2 = aabb_2.unbind(-2)
    intersect = (min_1 <= max_2).all(-1) & (max_1 >= min_2).all(-1)  # [b, *h]

    output = torch.zeros_like(intersect, dtype=torch.float32)  # [b, *h]
    if intersect.any():
        output[intersect] = _sphere_to_sphere_overlap(spheres_1[intersect], spheres_2[intersect], activation_distance)
    return output


def _spheres(k=None, seed=0):
    """[64, 3, 50, 4] sphere pairs, and the [64, 3] rows whose AABBs intersect: the first ``k`` of the
    192 rows in row-major order, or about half of them at random when ``k`` is None."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def make():
        return torch.cat([0.1 * torch.randn(64, 3, 50, 3, device="cuda", generator=g),
                          0.02 + 0.03 * torch.rand(64, 3, 50, 1, device="cuda", generator=g)], -1)
    s1, s2 = make(), make()
    if k is None:
        near = torch.rand(64, 3, 1, device="cuda", generator=g) < 0.5
    else:
        near = (torch.arange(64 * 3, device="cuda") < k).view(64, 3, 1)
    s2[..., 0] += torch.where(near, 0.05, 3.0)  # the far rows' AABBs are metres apart
    return s1, s2, near.squeeze(-1)


@needs_cuda
def test_dense_aabb_overlap_matches_masked(monkeypatch):
    # Not bitwise: the masked form ran the overlap on only the intersecting rows, so TF32 cdist and the
    # reduction batch differently (grads ~1e-4 relative). Non-intersecting rows are exactly 0.
    from cutamp.costs import sphere_to_sphere_overlap

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)   # the loop's regime
    # Own generator: the global one's state depends on the tests run before, and some weights fail rtol.
    gen = torch.Generator(device="cuda").manual_seed(0)
    for k in (None, 1, 2, 5):
        s1, s2, near = _spheres(k)
        if k is None:
            assert 0.3 < near.float().mean() < 0.7                  # mixed rows, not the trivial case
        for act in (None, 0.0, 0.02):
            case = f"k={k}, activation {act}"
            a, b = s1.clone().requires_grad_(), s2.clone().requires_grad_()
            want = _head_sphere_to_sphere_overlap(a, b, act, use_aabb_check=True)
            assert torch.equal(want > 0, near), case                # the near rows overlap, the far ones don't
            w = torch.rand(want.shape, device=want.device, generator=gen)
            gw = torch.autograd.grad((want * w).sum(), (a, b))
            with sync_debug_error():
                got = sphere_to_sphere_overlap(a, b, act, use_aabb_check=True)
                gg = torch.autograd.grad((got * w).sum(), (a, b))
            assert (got[~near] == 0).all() and all((g[~near] == 0).all() for g in gg), case
            for name, x, y in (("out", got, want), ("grad 1", gg[0], gw[0]), ("grad 2", gg[1], gw[1])):
                torch.testing.assert_close(x, y, rtol=1e-3, atol=1e-3 * y.abs().max().item(),
                                           msg=lambda m: f"{case}, {name}: {m}")
    s1, s2, _ = _spheres()
    a = s1.clone().requires_grad_()
    b = (s2 + torch.tensor([10.0, 0, 0, 0], device="cuda")).requires_grad_()
    miss = sphere_to_sphere_overlap(a, b, use_aabb_check=True)
    assert torch.equal(miss, torch.zeros_like(miss))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(miss.sum(), (a, b)))


@needs_cuda
def test_com_fk_skips_identity_reorder(blocks_t1_plan):
    from curobo.types import JointState
    from cutamp.com_polygon_cost import _fk_state

    world = blocks_t1_plan[0]
    torch.manual_seed(0)
    q = world.q_init + 0.3 * torch.randn(640, world.q_init.numel(), device="cuda")
    for kin in (world.kinematics, world.site_kinematics("com")):
        js = JointState.from_position(q, joint_names=list(kin.joint_names))
        active = kin.get_active_js(js)
        assert list(active.joint_names) == list(js.joint_names) and torch.equal(active.position, js.position)
    _fk_state(world, q, kin=world.site_kinematics("com"))           # sizes the site
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _fk_state(world, q, kin=world.site_kinematics("com"))
    names = [e.name for e in prof.events()]
    assert not [n for n in names if "HtoD" in n or "DtoH" in n]
    assert [n for n in names if "Synchronize" in n] == ["cudaDeviceSynchronize"]   # the profiler's own


@needs_cuda
def test_pairwise_upper_sum_matches_triu_mask():
    from cutamp.cost_function import _pairwise_upper_sum

    torch.manual_seed(0)
    for n in (2, 3, 4):
        x = torch.randn(64, n, n, device="cuda", requires_grad=True)
        w = torch.rand(64, device="cuda")
        want = _head_upper_sum(x)
        gw, = torch.autograd.grad((want * w).sum(), x)
        with sync_debug_error():
            got = _pairwise_upper_sum(x)
            gg, = torch.autograd.grad((got * w).sum(), x)
        assert torch.equal(got, want) and torch.equal(gg, gw), n


def test_constant_builders_never_evict():
    from cutamp.com_polygon_cost import _center, _half_extents
    from cutamp.cost_function import _device_const
    from cutamp.gn_refresh import _row_layout

    assert [f.cache_info().maxsize for f in (_row_layout, _device_const, _half_extents, _center)] == [None] * 4


@needs_cuda
def test_merged_adam_groups_match_per_param():
    from cutamp.optimize_plan import _adam_param_groups

    torch.manual_seed(0)
    spec = [((64, 4), False), ((64, 4), False), ((64, 21), True), ((64, 21), True), ((64, 4), False), ((64, 21), True)]
    init = [torch.randn(s, device="cuda") for s, _ in spec]
    a = [t.clone().requires_grad_() for t in init]
    b = [t.clone().requires_grad_() for t in init]
    head = Adam([{"params": p, **({"lr": 2.226e-2} if conf else {})} for p, (_, conf) in zip(a, spec)], lr=7e-3)
    merged = Adam(_adam_param_groups([(p, conf) for p, (_, conf) in zip(b, spec)], 2.226e-2), lr=7e-3)
    assert len(merged.param_groups) == 2
    for step in range(1, 201):
        for pa, pb in zip(a, b):
            g = torch.randn_like(pa)
            pa.grad, pb.grad = g.clone(), g.clone()
        head.step()
        merged.step()
        if step in (1, 10, 50, 200):
            assert all(torch.equal(x, y) for x, y in zip(a, b)), step
    assert [len(g["params"]) for g in _adam_param_groups([(a[0], False), (a[1], False)], 0.1)] == [2]
    assert _adam_param_groups([(a[2], True)], 0.1) == [{"params": [a[2]], "lr": 0.1}]


@needs_cuda
def test_align_yaw_syncs_in_roma(blocks_t1_plan):
    # Why align_yaw runs stay eager under cuda_graphs: its sync is roma's.
    from cutamp.cost_function import CostFunction
    from cutamp.rollout import RolloutFunction

    world, skeleton, base = blocks_t1_plan
    config = TAMPConfiguration(soft_cost=["align_yaw"])
    # Built first: the constructor's own host-to-device copies would raise first.
    cf = CostFunction(skeleton, world, config)
    with torch.no_grad():
        rollout = RolloutFunction(skeleton, world, config)(clone_particles(base))
    with pytest.raises(RuntimeError, match="synchroniz") as excinfo, sync_debug_error():
        cf._compute_soft_cost("align_yaw", rollout)
    assert any("roma/mappings.py" in str(e.path) for e in excinfo.traceback)   # roma's sync, not cutamp's
