"""The in-loop Adam step's capture-safe prerequisites: what the eager step
skips, and what it may sync on."""

import pytest
import torch
from torch.optim import Adam
from torch.profiler import ProfilerActivity, profile

from cutamp.config import SUPPORTED_SOFT_COSTS, TAMPConfiguration
from cutamp.tests.conftest import (
    SpyVis, assert_bitwise, clone_particles, cupti_sync_events, make_plan_info, needs_cuda, run_optimizer,
    sync_debug_error,
)
from cutamp.tests.test_step_lockstep import _head_sphere_to_sphere_overlap, _head_upper_sum

pytestmark = pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")


@needs_cuda
def test_noop_visualizer_skips_vis_block(blocks_t1_plan):
    world, skeleton, base = blocks_t1_plan
    config = TAMPConfiguration(num_particles=64, num_opt_steps=12, early_stop_patience=0,
                               reik_balance_first=True, enable_visualizer=False)
    _, timer = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config)
    assert "visualize_opt_rollout" not in timer.get_summaries()  # P13: the no-op visualizer skips the block
    vis = SpyVis(config)
    run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config, visualizer=vis)
    assert vis.opt_steps == [0, 10, 11]


def _spheres(seed=0):
    """[64, 3, 50, 4] sphere pairs; about half of the [64, 3] rows have intersecting AABBs."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def make():
        return torch.cat([0.1 * torch.randn(64, 3, 50, 3, device="cuda", generator=g),
                          0.02 + 0.03 * torch.rand(64, 3, 50, 1, device="cuda", generator=g)], -1)
    s1, s2 = make(), make()
    s2[..., 0] += torch.where(torch.rand(64, 3, 1, device="cuda", generator=g) < 0.5, 0.05, 3.0)
    return s1, s2


@needs_cuda
def test_dense_aabb_overlap_matches_masked(monkeypatch):            # P1
    from cutamp.costs import sphere_to_sphere_overlap

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)   # the loop's regime
    s1, s2 = _spheres()
    for act in (None, 0.0, 0.02):
        a, b = s1.clone().requires_grad_(), s2.clone().requires_grad_()
        want = _head_sphere_to_sphere_overlap(a, b, act, use_aabb_check=True)
        assert 0.3 < (want > 0).float().mean() < 0.7                # mixed rows, not the trivial case
        w = torch.rand_like(want)
        gw = torch.autograd.grad((want * w).sum(), (a, b))
        with sync_debug_error():                                    # RED: costs.py:85 intersect.any()
            got = sphere_to_sphere_overlap(a, b, act, use_aabb_check=True)
            gg = torch.autograd.grad((got * w).sum(), (a, b))
        assert_bitwise({"out": got, "grads": gg}, {"out": want, "grads": gw}, f"activation {act}")
    a = s1.clone().requires_grad_()
    b = (s2 + torch.tensor([10.0, 0, 0, 0], device="cuda")).requires_grad_()
    miss = sphere_to_sphere_overlap(a, b, use_aabb_check=True)
    assert torch.equal(miss, torch.zeros_like(miss))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(miss.sum(), (a, b)))


@needs_cuda
def test_com_fk_skips_identity_reorder(blocks_t1_plan):             # P2
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
    assert not [n for n in names if "HtoD" in n or "DtoH" in n]     # RED: the reorder index's H2D
    assert [n for n in names if "Synchronize" in n] == ["cudaDeviceSynchronize"]   # the profiler's own


@needs_cuda
def test_pairwise_upper_sum_matches_triu_mask():                    # P12
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


def test_constant_builders_never_evict():                           # P8, CPU
    from cutamp.com_polygon_cost import _half_extents
    from cutamp.cost_function import _device_const
    from cutamp.gn_refresh import _limit_margins, _row_layout

    assert [f.cache_info().maxsize for f in (_row_layout, _limit_margins, _device_const, _half_extents)] == [None] * 4


@needs_cuda
def test_merged_adam_groups_match_per_param():                      # P14
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
def test_eager_step_makes_one_host_read(blocks_t1_plan, monkeypatch):
    # The warm eager step, every capturable soft cost on, profiled with sync debug mode
    # "error": its one host read is the loop's packed.tolist(). align_yaw (D1) is the
    # exception, and its sync is roma's.
    import cutamp.optimize_plan as optimize_plan
    from cutamp.cost_function import CostFunction
    from cutamp.rollout import RolloutFunction

    world, skeleton, base = blocks_t1_plan
    config = TAMPConfiguration(num_particles=64, num_opt_steps=3, early_stop_patience=0, reik_balance_first=True,
                               enable_visualizer=False,
                               soft_cost=[s for s in SUPPORTED_SOFT_COSTS if s != "align_yaw"])
    orig, names = optimize_plan._StepBody.__call__, []
    calls = 0

    def profiled(self):  # calls 1-2 pass through and warm everything up; call 3 is profiled
        nonlocal calls
        calls += 1
        if calls < 3:
            return orig(self)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], **cupti_sync_events()) as prof:
            with sync_debug_error():
                packed, vis = orig(self)
            packed.tolist()
        names.extend(e.name for e in prof.events())
        return packed, vis

    monkeypatch.setattr(optimize_plan._StepBody, "__call__", profiled)
    run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config)
    assert calls == 3
    # The profiler's own sync, and the tolist's.
    assert sorted(n for n in names if "Synchronize" in n) == ["cudaDeviceSynchronize", "cudaStreamSynchronize"]
    assert sum("DtoH" in n for n in names) == 1
    assert not [n for n in names if "HtoD" in n]
    if cupti_sync_events():  # driver-API waits too
        assert sorted(n for n in names if n.endswith(" Sync")) == ["Context Sync", "Stream Sync"]

    # Built first: the constructor's own host-to-device copies would raise first.
    cf = CostFunction(skeleton, world, config)
    with torch.no_grad():
        rollout = RolloutFunction(skeleton, world, config)(clone_particles(base))
    with pytest.raises(RuntimeError, match="synchroniz") as excinfo, sync_debug_error():
        cf._compute_soft_cost("align_yaw", rollout)
    assert any("roma/mappings.py" in str(e.path) for e in excinfo.traceback)   # roma's sync, not cutamp's
