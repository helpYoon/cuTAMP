"""Tests for CoM-aware seed-IK (LM residual fork). Spec:
docs/superpowers/specs/2026-06-09-com-aware-ik-design.md"""
from contextlib import contextmanager

import pytest
import torch

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda

#: The fork's seed-IK CoM box, (its support link, its half-extents), worked out by hand. The box is centred on the
#: support frame: the largest such box inside the keyframe rectangle about the support centre. Each sole is the vendor
#: foot's collision box, 0.224868 x 0.1 m (T1_29dof.urdf), taken as 0.2249 x 0.100 to design 7.2's 0.1 mm
#: (build_planner_model.SOLE_BOX), the sole centres 0.10625 m to either side of the mid-sole at home (t1_29dof.yml's
#: sole_y). The soles' hull is |x| <= 0.2249 / 2 = 0.11245 and |y| <= 0.10625 + 0.1 / 2 = 0.15625. The load guard
#: (a = 0.1) narrows y by 2 * 0.10625 * 0.1 = 0.02125, to 0.135. The 2 cm margin takes x to 0.09245 and leaves y at
#: min(0.135, 0.15625 - 0.02) = 0.135, and the 5 mm reserve gives (0.08745, 0.130). The stance frame sits at the home
#: mid-sole, the rectangle's centre (0, 0), so the box is the rectangle itself.
SEED_COM_BOX = ("stance", [0.08745, 0.130])


@contextmanager
def _com_blind(world):
    """``world``'s IK solver, CoM-blind until the block ends: the fork's seed CoM weights at 0, the IK config's (which
    also gate the per-problem merge of the seed solutions) and the seed solver's (which scale its CoM residual). The
    baseline the CoM-aware solve is measured against; only the in-memory solver changes."""
    ik_cfg, seed = world.ik_solver.config, world.ik_solver.seed_ik_solver
    assert seed.error_calculator.config is seed.config  # the residual reads the weights the patch sets
    assert ik_cfg.seed_com_support_weight > 0.0 and seed.config.com_support_weight > 0.0  # premise: CoM-aware
    with pytest.MonkeyPatch.context() as m:
        for cfg, prefix in ((ik_cfg, "seed_com_"), (seed.config, "com_")):
            m.setattr(cfg, prefix + "support_weight", 0.0)
            m.setattr(cfg, prefix + "center_weight", 0.0)
        yield


def _home_tool_matrix(world, arm="left"):
    """4x4 world pose of the arm's tool frame at the home configuration."""
    from curobo.types import JointState
    names = list(world.kinematics.joint_names)
    qh = world.q_init
    af = world.tool_frame_for_arm(arm)
    return world.kinematics.compute_kinematics(
        world.kinematics.get_active_js(
            JointState.from_position(qh.unsqueeze(0), joint_names=names))
    ).tool_poses.get_link_pose(af, make_contiguous=True).get_matrix().reshape(4, 4)


@needs_cuda
def test_seed_cfg_carries_com_params():
    from cutamp.com_polygon_cost import COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT
    from cutamp.robots.t1 import T1_COM_IK_WEIGHT
    world = make_blocks_t1_world()
    cfg = world.ik_solver.seed_ik_solver.config
    link, half = SEED_COM_BOX
    assert cfg.com_support_weight == T1_COM_IK_WEIGHT > 0.0
    assert list(cfg.com_half_extents) == pytest.approx(half)
    assert cfg.com_inside_margin == COM_INSIDE_MARGIN
    assert cfg.com_inside_weight == COM_INSIDE_WEIGHT
    from cutamp.robots.t1 import T1_COM_IK_CENTER_WEIGHT
    assert cfg.com_center_weight == T1_COM_IK_CENTER_WEIGHT
    assert cfg.com_base_link_name == link


@needs_cuda
def test_seed_ik_robot_com_is_real():
    from curobo.types import JointState
    world = make_blocks_t1_world()
    model = world.ik_solver.seed_ik_solver._robot_model
    assert model.compute_com is True
    js = JointState.from_position(
        torch.zeros(1, model.get_dof(), device=model.device_cfg.device),
        joint_names=list(model.joint_names),
    )
    ks = model.compute_kinematics(js)
    assert float(ks.robot_com[..., :3].abs().max()) > 0.01, "robot_com xyz must be populated, not zeros"


def test_fork_penalty_parity_with_cutamp_reference():
    # The fork's com_support_penalty must agree with cuTAMP's com_polygon_penalty
    # (single source of truth enforced by test, since curobo cannot import cutamp).
    # Random rotated/translated base frames catch quaternion-projection sign errors.
    import roma
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import com_support_penalty
    from cutamp.com_polygon_cost import _project_com_to_base, com_polygon_penalty
    torch.manual_seed(0)
    B = 256
    com_world = torch.randn(B, 3) * 0.3
    base_pos = torch.randn(B, 3) * 0.2
    quat_xyzw = roma.random_unitquat(B)
    quat_wxyz = roma.quat_xyzw_to_wxyz(quat_xyzw)
    R = roma.unitquat_to_rotmat(quat_xyzw)
    base_T = torch.eye(4).repeat(B, 1, 1)
    base_T[:, :3, :3] = R
    base_T[:, :3, 3] = base_pos
    half = torch.tensor([0.1115, 0.156])
    # cuTAMP's rectangle sits at a support centre in the base frame, the fork's at the base origin: shift cuTAMP's CoM
    # by the centre so both see the same offset. The real legs' centre is (0, 0); a nonzero one keeps the shift tested.
    center = (0.01, 0.0)
    com_shifted = com_world + (R @ torch.tensor([*center, 0.0])).view(B, 3)
    com_in_base = _project_com_to_base(com_shifted, base_T, center)
    for cw in (0.0, 0.01):
        ref = com_polygon_penalty(com_shifted, base_T, half, 0.02, 1.0, center)
        ref = ref + cw * ((com_in_base / half) ** 2).sum(-1)
        got = com_support_penalty(com_world, base_pos, quat_wxyz, half, 0.02, 1.0, cw)
        assert torch.allclose(got, ref, atol=1e-5), f"max diff {(got - ref).abs().max()}"


def test_fork_penalty_gradient_matches_analytic_value():
    # At com=[0.15, 0, 0.3], half=[0.1115, 0.156], margin=0.02, inside_weight=1:
    # offset_x = 0.0385 (outside), inside_x = 0.0585, y-terms inactive.
    # d(pen)/d(com_x) = 2*0.0385 + 2*0.0585 = 0.194 exactly; y and z grads = 0.
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import com_support_penalty
    com = torch.tensor([[0.15, 0.0, 0.3]], requires_grad=True)
    half = torch.tensor([0.1115, 0.156])
    pos = torch.zeros(1, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pen = com_support_penalty(com, pos, quat, half, 0.02, 1.0, 0.0)
    pen.sum().backward()
    grad = com.grad[0]
    assert grad[0].item() == pytest.approx(0.194, rel=1e-5)
    assert grad[1].item() == pytest.approx(0.0, abs=1e-9)
    assert grad[2].item() == pytest.approx(0.0, abs=1e-9)
    # Center term gradient: pen = cw*((x/hx)^2 + (y/hy)^2) inside the inset
    # -> d/dx = 2*cw*x/hx^2. At x=0.05: 2*0.01*0.05/0.1115^2 = 0.0804366...
    com2 = torch.tensor([[0.05, 0.0, 0.3]], requires_grad=True)
    pen2 = com_support_penalty(com2, pos, quat, half, 0.02, 1.0, 0.01)
    pen2.sum().backward()
    expected = 2 * 0.01 * 0.05 / (0.1115 ** 2)
    assert com2.grad[0][0].item() == pytest.approx(expected, rel=1e-4)


@needs_cuda
def test_residual_fires():
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import SeedIKErrorCalculator
    from cutamp.particle_initialization import _ik_for_pose
    calls = {"n": 0}
    orig = SeedIKErrorCalculator._compute_com_penalty
    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)
    SeedIKErrorCalculator._compute_com_penalty = counting
    try:
        world = make_blocks_t1_world()
        hm = _home_tool_matrix(world)
        T = hm.unsqueeze(0).repeat(4, 1, 1).clone()
        T[:, 0, 3] += 0.2
        _ik_for_pose(world, T, "left")
        assert calls["n"] > 0, "residual must fire"
    finally:
        SeedIKErrorCalculator._compute_com_penalty = orig


@needs_cuda
def test_robot_com_gradient_flows_to_joints_with_fd_check():
    # The residual's core chain: joint_position -> fused-FK robot_com -> penalty
    # -> backward -> joint_position.grad. Verify against finite differences on
    # every joint the solver moves: the arms, the legs and trunk the residual is
    # meant to recruit (the model's body) and the floating base.
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import com_support_penalty
    from curobo.types import JointState
    from cutamp.com_polygon_cost import COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT

    world = make_blocks_t1_world()
    solver = world.ik_solver.seed_ik_solver
    model = solver._robot_model
    device = model.device_cfg.device
    names = list(model.joint_names)
    half = torch.tensor(list(solver.config.com_half_extents), device=device)
    base_idx = list(model.tool_frames).index(world.model.support.link)
    # At home with the box moved 15 cm behind its frame, the CoM is past the box's front edge, so the penalty is
    # active (the support frame is unrotated at home).
    behind = torch.tensor([0.15, 0.0, 0.0], device=device)

    def penalty_at(q):
        ks = model.compute_kinematics(
            JointState.from_position(q, joint_names=names))
        com = ks.robot_com.view(1, -1)[:, :3]
        pos = ks.tool_poses.position.view(1, len(model.tool_frames), 3)[:, base_idx, :] - behind
        quat = ks.tool_poses.quaternion.view(1, len(model.tool_frames), 4)[:, base_idx, :]
        return com_support_penalty(com, pos, quat, half, COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT, 0.0).sum()

    home = dict(zip(world.model.joint_names, world.model.home))
    q0 = torch.tensor([[home[n] for n in names]], device=device)

    q = q0.clone().requires_grad_(True)
    pen = penalty_at(q)
    assert pen.item() > 0, "premise: the CoM must be outside the moved box"
    pen.backward()
    grad = q.grad[0]
    assert torch.isfinite(grad).all()

    eps = 1e-3
    for j in range(len(names)):  # every column of the solver's model
        qp = q0.clone(); qp[0, j] += eps
        qm = q0.clone(); qm[0, j] -= eps
        with torch.no_grad():
            fd = (penalty_at(qp) - penalty_at(qm)) / (2 * eps)
        assert grad[j].item() == pytest.approx(fd.item(), rel=0.05, abs=1e-4), \
            f"{names[j]}: autograd {grad[j].item():.6f} vs FD {fd.item():.6f}"
    # The chain must actually recruit the body: a leg or trunk gradient clearly nonzero. The solver moves the floating
    # base, the main CoM lever: a base gradient clearly nonzero too.
    body = [names.index(n) for n in world.model.joint_names[world.model.body]]
    assert grad[body].abs().max().item() > 1e-3
    base = [names.index(n) for n in world.model.joint_names[world.model.base]]
    assert grad[base].abs().max().item() > 1e-3


def _forward_reach_solutions(world, n_fwd=4, n_lat=3, unreachable=False, stand_in=False):
    """IK a forward-reach grid for the left arm; return (q_full[B, dof], success[B], result).
    ``unreachable`` appends a pose 3 m ahead, which no IK stage can solve. ``stand_in`` appends the grid's first pose
    instead, the problem the solver pads its batch with: the grid alone at the batch size of a solve with the
    unreachable pose (the fork's IK raises when one solver sees a second batch size). Each solve starts from the
    solver's reset seed, so the solves on one world see the same seeds: without it a solve's result moves with how
    many came before it."""
    from cutamp.particle_initialization import (
        _ik_for_pose, _ik_solution_to_full_q, _ik_success_mask,
    )
    world.ik_solver.reset_seed()
    qh = world.q_init
    hm = _home_tool_matrix(world)
    Ts = []
    for f in torch.linspace(0.15, 0.33, n_fwd, device=qh.device):
        for lat in torch.linspace(-0.05, 0.15, n_lat, device=qh.device):
            # Absolute world-Y on purpose: left tool home Y is ~+0.3, so
            # sweeping Y in [-0.05, 0.15] creates demanding cross-body reaches.
            m = hm.clone(); m[0, 3] += float(f); m[1, 3] = float(lat); Ts.append(m)
    if unreachable:
        m = hm.clone(); m[0, 3] += 3.0; Ts.append(m)
    if stand_in:
        Ts.append(Ts[0].clone())
    res = _ik_for_pose(world, torch.stack(Ts, 0), "left")
    return _ik_solution_to_full_q(res, world), _ik_success_mask(res), res


def _com_abs_x(world, q_full):
    from cutamp.com_polygon_cost import _fk_com_and_base, _project_com_to_base
    com_world, base_T = _fk_com_and_base(world, q_full)
    return _project_com_to_base(com_world, base_T, world.model.support.center)[:, 0].abs()


@needs_cuda
def test_centering_ab_legs_within_limits_pose_preserved():
    world = make_blocks_t1_world()
    q_on, s_on, res_on = _forward_reach_solutions(world)
    assert int(s_on.sum()) > 0, "CoM-aware IK produced no successful solutions"
    absx_on = _com_abs_x(world, q_on)[s_on]
    with _com_blind(world):  # the baseline: the same solver, CoM-blind
        q_off, s_off, _ = _forward_reach_solutions(world)
    absx_off = _com_abs_x(world, q_off)[s_off]
    assert int(s_off.sum()) > 0, "the CoM-blind baseline produced no successes: the A/B comparison is blind"

    # (1) centering: with the center pull on (T1_COM_IK_CENTER_WEIGHT=1e-3)
    # the COM must be actively pulled toward center, not just kept in-band.
    # Measured on the GPU 2026-11-05: mean |CoM x| 0.29 cm CoM-aware, 8.34 cm
    # CoM-blind, 12 of 12 solved each, a ratio of 28.6; assert half of it.
    ratio = float(absx_off.mean()) / float(absx_on.mean())
    assert ratio >= 14.3, \
        f"ratio {ratio:.3f}: on={float(absx_on.mean()):.4f} blind={float(absx_off.mean()):.4f}"

    # (2) joint limits: every SUCCESSFUL solution's joints (the base, the arms,
    #     the legs and trunk) within the solver's bounds (+1e-3 slack). Limits
    #     are a soft residual + success filter in the seed-IK solver — failed
    #     rows may legitimately sit slightly out of bounds and are masked out
    #     downstream, but a successful out-of-limit solution would be a fork bug.
    body = world.model.body
    lower, upper = world.kinematics.get_joint_limits().position
    assert len(world.model.joint_names) == q_on.shape[1] == lower.numel() == upper.numel()
    for jn, col, lo, hi in zip(world.model.joint_names, q_on[s_on].T, lower.tolist(), upper.tolist()):
        assert float(col.min()) >= lo - 1e-3 and float(col.max()) <= hi + 1e-3, \
            f"{jn} out of [{lo:.3f},{hi:.3f}]: [{float(col.min()):.3f},{float(col.max()):.3f}]"

    # (3) pose accuracy on successes: solver tolerances (5mm / 0.05rad)
    pe = res_on.position_error
    pe = (pe.reshape(pe.shape[0], -1)[:, 0] if pe.ndim > 1 else pe)[s_on]
    assert float(pe.max()) <= 0.005 + 1e-4, f"pose err {float(pe.max()):.4f}"

    # (4) soft recruitment signal (informational)
    body_delta = float((q_on[s_on][:, body] - world.q_init[body]).abs().max())
    print(f"recruitment: max body-joint move from home {body_delta:.3f} rad; "
          f"absx on/blind mean {float(absx_on.mean()):.4f}/{float(absx_off.mean()):.4f}")


@needs_cuda
def test_unsolved_problem_keeps_the_others_centred():
    # N7: cuRobo's IK kept the seed LM's (CoM-aware) solutions only when every problem in the batch
    # succeeded; otherwise its CoM-blind optimizer re-solved them all. A pose no stage can reach
    # forces that path every time; the fork's per-problem merge keeps each problem's seed solution
    # where it succeeded. One world solves the grid with the unreachable pose and without it.
    world = make_blocks_t1_world()
    q_alone, s_alone, _ = _forward_reach_solutions(world, stand_in=True)
    q_with, s_with, _ = _forward_reach_solutions(world, unreachable=True)
    with _com_blind(world):
        q_blind, s_blind, _ = _forward_reach_solutions(world, unreachable=True)
    assert bool(s_alone.all()), "premise: the grid alone must solve, its stand-in too"
    assert not bool(s_with[-1]), "premise: the pose 3 m ahead must fail"
    assert torch.equal(s_with[:-1], s_alone[:-1]) and bool(s_blind[:-1].any())
    absx_alone = _com_abs_x(world, q_alone[:-1])[s_alone[:-1]]
    absx_with = _com_abs_x(world, q_with[:-1])[s_with[:-1]]
    absx_blind = _com_abs_x(world, q_blind[:-1])[s_blind[:-1]]
    # Measured on the GPU 2026-11-05: mean |CoM x| over the 12 reachable rows 0.29 cm with the unreachable pose and
    # without it, the rows bitwise equal; the CoM-blind solve 8.34 cm, a ratio of 28.6. Equal, so equality. On the
    # real legs every solve here takes the optimizer and the merge, the grid alone too (some seed of its batch
    # fails), so equality alone passes without the merge (3.23 cm both, seen): the ratio to the CoM-blind solve,
    # asserted at half the measured one as in the centring test, is what fails then.
    assert float(absx_with.mean()) == float(absx_alone.mean()), \
        f"with={float(absx_with.mean()):.6f} alone={float(absx_alone.mean()):.6f}"
    ratio = float(absx_blind.mean()) / float(absx_with.mean())
    assert ratio >= 14.3, f"ratio {ratio:.3f}: with={float(absx_with.mean()):.4f} blind={float(absx_blind.mean()):.4f}"


@needs_cuda
def test_store_ik_q_masks_failed_solutions_to_home():
    from cutamp.particle_initialization import _store_ik_q
    world = make_blocks_t1_world()
    q_full, succ, res = _forward_reach_solutions(world, n_fwd=2, n_lat=2)
    # Forge a failure on problem 0 and store.
    if res.success.ndim > 1:
        res.success[0, :] = False
    else:
        res.success[0] = False
    particles = {}
    _store_ik_q(particles, "left_q1", res, world, "left")
    stored = particles["left_q1"]
    assert torch.allclose(stored[0], world.q_init), \
        "failed-IK row must fall back to q_init, not keep an unvetted conf"

