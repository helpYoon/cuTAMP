"""Tests for CoM-aware seed-IK (LM residual fork). Spec:
docs/superpowers/specs/2026-06-09-com-aware-ik-design.md"""
import os
import pytest
import torch

needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _world(enable_com_aware_ik: bool):
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from cutamp.robots.t1 import t1_home
    from curobo.types import DeviceCfg
    dc = DeviceCfg()
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    robot = load_robot_container("t1", dc)
    qh = torch.as_tensor(t1_home, dtype=torch.float32, device=dc.device)
    return TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=qh,
                     enable_com_polygon=True, enable_com_aware_ik=enable_com_aware_ik)


@needs_cuda
def test_flag_off_seed_cfg_weight_zero_by_default():
    world = _world(enable_com_aware_ik=False)
    cfg = world.ik_solver.seed_ik_solver.config
    assert cfg.com_support_weight == 0.0


@needs_cuda
def test_flag_on_seed_cfg_carries_com_params():
    from cutamp.com_polygon_cost import COM_HALF_EXTENTS, COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT
    from cutamp.robots.t1 import T1_COM_IK_WEIGHT
    world = _world(enable_com_aware_ik=True)
    cfg = world.ik_solver.seed_ik_solver.config
    assert cfg.com_support_weight == T1_COM_IK_WEIGHT > 0.0
    assert list(cfg.com_half_extents) == list(COM_HALF_EXTENTS)
    assert cfg.com_inside_margin == COM_INSIDE_MARGIN
    assert cfg.com_inside_weight == COM_INSIDE_WEIGHT
    from cutamp.robots.t1 import T1_COM_IK_CENTER_WEIGHT
    assert cfg.com_center_weight == T1_COM_IK_CENTER_WEIGHT
    assert cfg.com_base_link_name == "mobile_base_link"


@needs_cuda
def test_flag_on_seed_ik_robot_com_is_real():
    from curobo.types import JointState
    world = _world(enable_com_aware_ik=True)
    model = world.ik_solver.seed_ik_solver._robot_model
    assert model.compute_com is True
    js = JointState.from_position(
        torch.zeros(1, model.get_dof(), device=model.device_cfg.device),
        joint_names=list(model.joint_names),
    )
    ks = model.compute_kinematics(js)
    assert float(ks.robot_com[..., :3].abs().max()) > 0.01, "robot_com xyz must be populated, not zeros"


@needs_cuda
def test_flag_off_seed_ik_robot_com_stays_off():
    world = _world(enable_com_aware_ik=False)
    assert world.ik_solver.seed_ik_solver._robot_model.compute_com is False


def test_fork_penalty_parity_with_cutamp_reference():
    # The fork's com_support_penalty must agree with cuTAMP's com_polygon_penalty
    # (single source of truth enforced by test, since curobo cannot import cutamp).
    # Random rotated/translated base frames catch quaternion-projection sign errors.
    import roma
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import com_support_penalty
    from cutamp.com_polygon_cost import com_polygon_penalty
    torch.manual_seed(0)
    B = 256
    com_world = torch.randn(B, 3) * 0.3
    base_pos = torch.randn(B, 3) * 0.2
    quat_xyzw = roma.random_unitquat(B)
    quat_wxyz = torch.cat([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], dim=-1)
    R = roma.unitquat_to_rotmat(quat_xyzw)
    base_T = torch.eye(4).repeat(B, 1, 1)
    base_T[:, :3, :3] = R
    base_T[:, :3, 3] = base_pos
    half = torch.tensor([0.1115, 0.156])
    for cw in (0.0, 0.01):
        ref = com_polygon_penalty(com_world, base_T, half, 0.02, 1.0, center_weight=cw)
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
def test_residual_fires_iff_enabled():
    import gc
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import SeedIKErrorCalculator
    from curobo.types import JointState
    from cutamp.particle_initialization import _ik_for_pose
    calls = {"n": 0}
    orig = SeedIKErrorCalculator._compute_com_penalty
    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)
    SeedIKErrorCalculator._compute_com_penalty = counting
    try:
        for flag, expect_calls in ((False, False), (True, True)):
            calls["n"] = 0
            world = _world(enable_com_aware_ik=flag)
            names = list(world.kinematics.joint_names)
            qh = world.q_init
            af = world.tool_frame_for_arm("left")
            hm = world.kinematics.compute_kinematics(
                world.kinematics.get_active_js(
                    JointState.from_position(qh.unsqueeze(0), joint_names=names))
            ).tool_poses.get_link_pose(af, make_contiguous=True).get_matrix().reshape(4, 4)
            T = hm.unsqueeze(0).repeat(4, 1, 1).clone()
            T[:, 0, 3] += 0.2
            _ik_for_pose(world, T, "left")
            if expect_calls:
                assert calls["n"] > 0, "residual must fire when enabled"
            else:
                assert calls["n"] == 0, "residual must not fire when disabled"
            del world
            gc.collect(); torch.cuda.empty_cache()
    finally:
        SeedIKErrorCalculator._compute_com_penalty = orig


@needs_cuda
def test_robot_com_gradient_flows_to_joints_with_fd_check():
    # The residual's core chain: joint_position -> fused-FK robot_com -> penalty
    # -> backward -> joint_position.grad. Verify against finite differences on
    # the leg/torso joints the residual is meant to recruit.
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import com_support_penalty
    from curobo.types import JointState

    world = _world(enable_com_aware_ik=True)
    model = world.ik_solver.seed_ik_solver._robot_model
    device = model.device_cfg.device
    names = list(model.joint_names)
    half = torch.tensor([0.1115, 0.156], device=device)
    base_idx = list(model.tool_frames).index("mobile_base_link")

    def penalty_at(q):
        ks = model.compute_kinematics(
            JointState.from_position(q, joint_names=names))
        com = ks.robot_com.view(1, -1)[:, :3]
        pos = ks.tool_poses.position.view(1, len(model.tool_frames), 3)[:, base_idx, :]
        quat = ks.tool_poses.quaternion.view(1, len(model.tool_frames), 4)[:, base_idx, :]
        return com_support_penalty(com, pos, quat, half, 0.02, 1.0, 0.0).sum()

    # A leaning config so the penalty is active: Torso_Pitch forward.
    q0 = torch.zeros(1, model.get_dof(), device=device)
    q0[0, names.index("Torso_Pitch")] = -1.5

    q = q0.clone().requires_grad_(True)
    penalty_at(q).backward()
    grad = q.grad[0]
    assert torch.isfinite(grad).all()

    eps = 1e-3
    for jn in ("Torso_Pitch", "knee_pitch", "ankle_pitch"):
        j = names.index(jn)
        qp = q0.clone(); qp[0, j] += eps
        qm = q0.clone(); qm[0, j] -= eps
        with torch.no_grad():
            fd = (penalty_at(qp) - penalty_at(qm)) / (2 * eps)
        assert grad[j].item() == pytest.approx(fd.item(), rel=0.05, abs=1e-4), \
            f"{jn}: autograd {grad[j].item():.6f} vs FD {fd.item():.6f}"
    # The chain must actually recruit the legs: torso gradient clearly nonzero.
    assert abs(grad[names.index("Torso_Pitch")].item()) > 1e-3


def _forward_reach_solutions(world, n_fwd=4, n_lat=3):
    """IK a forward-reach grid for the left arm; return (q_full[B,21], success[B], result)."""
    from curobo.types import JointState
    from cutamp.particle_initialization import _ik_for_pose, _ik_solution_to_full_q
    names = list(world.kinematics.joint_names)
    qh = world.q_init
    af = world.tool_frame_for_arm("left")
    hm = world.kinematics.compute_kinematics(
        world.kinematics.get_active_js(
            JointState.from_position(qh.unsqueeze(0), joint_names=names))
    ).tool_poses.get_link_pose(af, make_contiguous=True).get_matrix().reshape(4, 4)
    Ts = []
    for f in torch.linspace(0.15, 0.33, n_fwd, device=qh.device):
        for lat in torch.linspace(-0.05, 0.15, n_lat, device=qh.device):
            # Absolute world-Y on purpose: left tool home Y is ~+0.3, so
            # sweeping Y in [-0.05, 0.15] creates demanding cross-body reaches.
            m = hm.clone(); m[0, 3] += float(f); m[1, 3] = float(lat); Ts.append(m)
    res = _ik_for_pose(world, torch.stack(Ts, 0), "left")
    s = res.success
    succ = (s.reshape(s.shape[0], -1)[:, 0] if s.ndim > 1 else s).bool().reshape(-1)
    return _ik_solution_to_full_q(res, world), succ, res


def _com_abs_x(world, q_full):
    from curobo.types import JointState
    kin = world.kinematics_with_com
    names = list(world.kinematics.joint_names)
    ks = kin.compute_kinematics(kin.get_active_js(
        JointState.from_position(q_full.to(kin.device_cfg.device), joint_names=names)))
    cw = ks.robot_com[..., :3].reshape(-1, 3)
    bT = ks.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True) \
        .get_matrix().reshape(-1, 4, 4)
    cib = (bT[:, :3, :3].transpose(-1, -2) @ (cw - bT[:, :3, 3]).unsqueeze(-1)).squeeze(-1)
    return cib[:, 0].abs()


@needs_cuda
def test_centering_ab_legs_within_limits_pose_preserved():
    import gc
    from cutamp.com_polygon_cost import COM_HALF_EXTENTS, COM_INSIDE_MARGIN
    # OFF baseline
    world_off = _world(enable_com_aware_ik=False)
    q_off, s_off, _ = _forward_reach_solutions(world_off)
    absx_off = _com_abs_x(world_off, q_off)[s_off]
    assert int(s_off.sum()) > 0, "OFF baseline produced no successes — A/B comparison is blind"
    del world_off; gc.collect(); torch.cuda.empty_cache()
    # ON
    world_on = _world(enable_com_aware_ik=True)
    q_on, s_on, res_on = _forward_reach_solutions(world_on)
    assert int(s_on.sum()) > 0, "CoM-aware IK produced no successful solutions"
    absx_on = _com_abs_x(world_on, q_on)[s_on]

    # (1) centering: with the center pull on (T1_COM_IK_CENTER_WEIGHT=1e-3)
    # the COM must be actively pulled toward center, not just kept in-band.
    # Sweep showed ~30x reduction; assert a conservative 2x to absorb solver
    # nondeterminism.
    assert float(absx_on.mean()) < 0.5 * float(absx_off.mean()), \
        f"on={float(absx_on.mean()):.4f} off={float(absx_off.mean()):.4f}"

    # (2) joint limits: every SUCCESSFUL solution within URDF bounds (+1e-3
    #     slack). Limits are a soft residual + success filter in the seed-IK
    #     solver — failed rows may legitimately sit slightly out of bounds and
    #     are masked out downstream, but a successful out-of-limit solution
    #     would be a fork bug.
    names = list(world_on.kinematics.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    bounds = {"ankle_pitch": (-0.87, 0.0), "knee_pitch": (0.0, 2.34),
              "Torso_Pitch": (-1.8, 0.0)}
    for jn, (lo, hi) in bounds.items():
        col = q_on[s_on][:, idx[jn]]
        assert float(col.min()) >= lo - 1e-3 and float(col.max()) <= hi + 1e-3, \
            f"{jn} out of [{lo},{hi}]: [{float(col.min()):.3f},{float(col.max()):.3f}]"

    # (3) pose accuracy on successes: solver tolerances (5mm / 0.05rad)
    pe = res_on.position_error
    pe = (pe.reshape(pe.shape[0], -1)[:, 0] if pe.ndim > 1 else pe)[s_on]
    assert float(pe.max()) <= 0.005 + 1e-4, f"pose err {float(pe.max()):.4f}"

    # (4) soft recruitment signal (informational; generous threshold)
    knee_delta = float((q_on[s_on][:, idx["knee_pitch"]]).abs().max())
    print(f"recruitment: max knee_pitch={knee_delta:.3f} rad; "
          f"absx on/off mean {float(absx_on.mean()):.4f}/{float(absx_off.mean()):.4f}")


@needs_cuda
def test_store_ik_q_masks_failed_solutions_to_home():
    from cutamp.particle_initialization import _store_ik_q
    world = _world(enable_com_aware_ik=False)
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


def test_ik_for_pose_has_no_dormant_num_seeds_kwarg():
    import inspect
    from cutamp.particle_initialization import _ik_for_pose
    assert "num_seeds" not in inspect.signature(_ik_for_pose).parameters, \
        "num_seeds forwarded to solve_pose would TypeError (no such param)"
