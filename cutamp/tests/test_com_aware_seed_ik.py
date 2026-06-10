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
    assert cfg.com_center_weight == 0.0
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
