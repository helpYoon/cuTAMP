"""Tests for the Gauss-Newton balance-first IK projection (cutamp/gn_refresh.py).

CUDA-only: exercises the real T1 kinematics kernel — the Jacobian convention
check is the load-bearing regression (GN silently diverges if cuRobo ever
changes row order or frame). The tests run on the real legs.
"""
import inspect

import pytest
import roma
import torch

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def _ik(world, target, arm):
    """Confs for ``target`` ``[B, 4, 4]`` on ``arm``: the balance-first IK projection from home with a seeded 0.1 rad of
    noise on the arm's joints, no objects to clear; every row passes the IK check (``ik_check``)."""
    from cutamp.gn_refresh import gn_project_balanced
    from cutamp.particle_initialization import IKTol
    from cutamp.tests.conftest import default_checker, ik_check
    q = world.q_init.expand(len(target), -1).clone()
    arm_cols = getattr(world.model, f"{arm}_arm")
    noise = torch.randn(q[:, arm_cols].shape, device=q.device, generator=torch.Generator(q.device).manual_seed(0))
    q[:, arm_cols] += 0.1 * noise
    movables = target.new_zeros(len(target), 0, 4)
    q = gn_project_balanced(world, q, target, arm, iters=30, movables=movables)
    ok, _, _ = ik_check(world, q, target, arm, movables, IKTol.from_checker(default_checker()))
    assert bool(ok.all()), (arm, int(ok.sum()))
    return q


def _four_dep_stack(world):
    """``(q [32, dof], T [32, 4, 4], arms)``: four arm-sorted 8-row IK deps. Per arm, one IK
    solution for a target at (0.35, +-0.25, 0.25) feeds a warm dep (target drifted by mm) and a
    far dep (x = 0.9, out of balanced reach)."""
    B = 8
    qs, targets = [], []
    for arm, y in (("left", 0.25), ("right", -0.25)):
        target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
        target[:, :3, 3] = torch.tensor([0.35, y, 0.25], device="cuda")
        q0 = _ik(world, target, arm)
        moved = target.clone()
        moved[:, 0, 3] += 0.003
        moved[:, 2, 3] += 0.0025
        far = target.clone()
        far[:, 0, 3] = 0.9
        qs += [q0, q0]
        targets += [moved, far]
    return torch.cat(qs), torch.cat(targets), ("left", "left", "right", "right")


def _without_com_level(monkeypatch):
    """Until ``monkeypatch`` undoes it, GN without its CoM level: ``_balanced_step`` sees a zero CoM and CoM Jacobian,
    its positional arguments 3 and 4, so both CoM active sets stay empty and level 1 keeps only the sole rows. A
    reorder of those arguments fails here, by name or by shape, not silently."""
    import cutamp.gn_refresh as gn_refresh
    step = gn_refresh._balanced_step
    assert list(inspect.signature(step).parameters)[3:5] == ["jac_com", "com"]

    def no_com(q, r, jac, jac_com, com, *args, **kwargs):
        assert jac_com.shape == (q.shape[0], 2, q.shape[1]) and com.shape == (q.shape[0], 2)
        return step(q, r, jac, torch.zeros_like(jac_com), torch.zeros_like(com), *args, **kwargs)

    monkeypatch.setattr(gn_refresh, "_balanced_step", no_com)


def _pose_errors(world, q, target, arm):
    """``([B] pos_err, [B] rot_err)`` of ``q`` ``[B, dof]`` against ``target`` ``[B, 4, 4]`` on ``arm``'s tool frame,
    from one FK on ``world.kinematics``, in the constraint checker's metric (``curobo_pose_error``)."""
    from curobo.types import JointState, Pose
    from cutamp.costs import curobo_pose_error
    kin = world.kinematics
    state = kin.compute_kinematics(JointState.from_position(q, joint_names=list(kin.joint_names)))
    ee = state.tool_poses.get_link_pose(world.tool_frame_for_arm(arm))
    # get_link_pose returns strided views; get_matrix requires contiguous.
    ee = Pose(position=ee.position.contiguous(), quaternion=ee.quaternion.contiguous()).get_matrix()
    return curobo_pose_error(ee, target)


def _quat_to_mat(q_wxyz):
    return roma.unitquat_to_rotmat(q_wxyz[..., [1, 2, 3, 0]].contiguous())


def _fk_pose(world, q, frame):
    from curobo.types import JointState
    kin = world.kinematics_jac
    js = JointState.from_position(q, joint_names=list(kin.config.kinematics_config.joint_names))
    state = kin.compute_kinematics(js)
    pose = state.tool_poses.get_link_pose(frame)
    B = q.shape[0]
    # Clone: cuRobo reuses its FK buffers in place, so the views go stale on
    # the next compute_kinematics call.
    return pose.position.reshape(B, 3).clone(), pose.quaternion.reshape(B, 4).clone(), state


@needs_cuda
def test_tool_jacobian_matches_finite_differences():
    # Rows 0-2 = world-frame linear, rows 3-5 = world-frame angular. GN's
    # update law assumes exactly this; fail loudly if cuRobo changes it.
    world = make_blocks_t1_world()
    frame = world.tool_frame_for_arm("left")
    kin = world.kinematics_jac
    frame_idx = list(kin.tool_frames).index(frame)

    torch.manual_seed(0)
    q0 = world.q_init.unsqueeze(0).clone() + 0.05 * torch.randn(1, world.q_init.numel(), device="cuda")
    pos0, quat0, state = _fk_pose(world, q0, frame)
    J = state.tool_jacobians[0, 0, frame_idx].clone()  # [6, dof]
    R0 = _quat_to_mat(quat0)[0]

    eps = 1e-3
    for k in range(q0.shape[1]):  # every column: base, arms, legs and trunk
        dq = torch.zeros_like(q0)
        dq[0, k] = eps
        pos1, quat1, _ = _fk_pose(world, q0 + dq, frame)
        fd_lin = (pos1 - pos0)[0] / eps
        dR = _quat_to_mat(quat1)[0] @ R0.T
        fd_ang = 0.5 * torch.tensor([
            dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1],
        ], device="cuda") / eps
        assert torch.allclose(J[:3, k], fd_lin, atol=2e-3), f"linear rows, joint {k}"
        assert torch.allclose(J[3:, k], fd_ang, atol=2e-3), f"angular rows, joint {k}"


@needs_cuda
def test_gn_project_balanced_keeps_com_in_and_closes_pose():
    # Balance-first GN: from a warm start it converges; from a target well out
    # of balanced reach it ends CoM-inside and closer to the target than it
    # started (closest balanced pose, not an unbalanced exact one).
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project_balanced

    world = make_blocks_t1_world()
    B = 8
    target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
    target[:, :3, 3] = torch.tensor([0.35, 0.25, 0.25], device="cuda")
    q0 = _ik(world, target, "left")
    moved = target.clone()
    moved[:, 0, 3] += 0.003
    moved[:, 2, 3] += 0.0025
    q1 = gn_project_balanced(world, q0, moved, "left")
    pos, rot = _pose_errors(world, q1, moved, "left")
    assert bool((pos < 0.005).all() & (rot < 0.05).all()), f"pos {pos.max().item():.4f}"
    assert bool(compute_com_polygon_mask(world, q1).all())

    far = target.clone()
    far[:, 0, 3] = 0.9
    pos0, _ = _pose_errors(world, q0, far, "left")
    q2 = gn_project_balanced(world, q0, far, "left", iters=30)
    pos2, _ = _pose_errors(world, q2, far, "left")
    assert bool(compute_com_polygon_mask(world, q2).all())
    assert bool((pos2 < pos0 - 0.05).all()), f"{pos0.min().item():.3f} -> {pos2.max().item():.3f}"

    # Without its CoM level, GN trades the CoM for the pose, so it ends closer to
    # the far target and, on most rows, outside the polygon: level 1 is what holds the CoM. Under the ankle cap
    # (shins never pitch back past home) one row of the 8 ends inside without it too, at the pose error it has with
    # it (0.349 m, seen 2026-11-06): its joint limits stop it first. So half the rows, and the ones that leave.
    with pytest.MonkeyPatch.context() as m:
        _without_com_level(m)
        q3 = gn_project_balanced(world, q0, far, "left", iters=30)
    pos3, _ = _pose_errors(world, q3, far, "left")
    out = ~compute_com_polygon_mask(world, q3)
    assert int(out.sum()) >= B // 2
    assert bool((pos3[out] < pos2[out]).all()), f"{pos3[out].max().item():.3f} vs {pos2[out].min().item():.3f}"


@needs_cuda
def test_gn_project_balanced_pins_per_row():
    # Each row keeps its own dep's frozen columns and tool frame (a right row on the left frame
    # would chase a target half a metre away).
    from cutamp.gn_refresh import gn_project_balanced

    world = make_blocks_t1_world()
    model = world.model
    q, T, arms = _four_dep_stack(world)
    q1 = gn_project_balanced(world, q, T, arms)
    per_dep = torch.cat([gn_project_balanced(world, qi, Ti, a) for qi, Ti, a in zip(q.split(8), T.split(8), arms)])
    assert (q1 - per_dep).abs().max().item() <= 1e-5  # bitwise as measured; slack for batch-dependent GPU reductions
    left_only = gn_project_balanced(world, q[:16], T[:16], arms[:2])  # a stack with no right-arm rows
    assert (left_only - per_dep[:16]).abs().max().item() <= 1e-5
    assert torch.equal(q1[:16, model.right_arm], q[:16, model.right_arm])
    assert torch.equal(q1[16:, model.left_arm], q[16:, model.left_arm])
    assert not torch.equal(q1[:16, model.left_arm], q[:16, model.left_arm])
    for rows, arm in ((slice(0, 8), "left"), (slice(16, 24), "right")):
        pos, rot = _pose_errors(world, q1[rows], T[rows], arm)
        assert bool((pos < 2e-3).all() & (rot < 2e-2).all()), (
            f"{arm}: pos {pos.max().item():.4f} rot {rot.max().item():.4f}"
        )
    # A real blocks_t1 refresh interleaves the arms (left_q1, right_q1, left_q3, right_q3).
    perm = torch.cat([torch.arange(8, device="cuda") + 8 * b for b in (0, 2, 1, 3)])
    inter = ("left", "right", "left", "right")
    q_int = gn_project_balanced(world, q[perm], T[perm], inter)
    assert (q_int - q1[perm]).abs().max().item() <= 1e-5


def test_pose_error_resolves_sub_mrad_rotations_in_float32():
    # The checker's atan2 angle must see a 4e-4 rad rotation (an acos form collapses to 0 below ~5e-4 rad in float32).
    from cutamp.costs import curobo_pose_error

    a = torch.eye(4).expand(2, 4, 4).clone()
    b = a.clone()
    b[:, :3, :3] = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, 4e-4]))
    assert curobo_pose_error(a, b)[1].min().item() > 1e-4
