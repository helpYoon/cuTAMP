"""Tests for the Gauss-Newton reIK refresh (cutamp/gn_refresh.py).

CUDA-only: exercises the real T1 kinematics kernel — the Jacobian convention
check is the load-bearing regression (GN silently diverges if cuRobo ever
changes row order or frame).
"""
import math

import torch

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def _quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


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
    q0 = world.q_init.unsqueeze(0).clone() + 0.05 * torch.randn(1, 21, device="cuda")
    pos0, quat0, state = _fk_pose(world, q0, frame)
    J = state.tool_jacobians[0, 0, frame_idx].clone()  # [6, 21]
    R0 = _quat_to_mat(quat0)[0]

    eps = 1e-3
    for k in [3, 5, 8, 12]:  # ankle, torso, left arm joints
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
def test_gn_project_converges_from_warm_start():
    # The refresh scenario: an exact IK solution whose target then moves by
    # mm/mrad-scale Adam drift. GN must recover checker tolerance (5mm/0.05rad).
    from cutamp.gn_refresh import gn_project
    from cutamp.particle_initialization import (
        _achieved_pose_errors, _ik_for_pose, _ik_solution_to_full_q, _ik_success_mask,
    )

    world = make_blocks_t1_world()
    B = 8
    target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
    target[:, :3, 3] = torch.tensor([0.35, 0.25, 0.25], device="cuda")
    ik_result = _ik_for_pose(world, target, "left")
    assert bool(_ik_success_mask(ik_result).all())
    q0 = _ik_solution_to_full_q(ik_result, world)

    # Drift: 4 mm translation + 0.02 rad yaw (2x the mean measured Adam drift).
    moved = target.clone()
    moved[:, 0, 3] += 0.003
    moved[:, 2, 3] += 0.0025
    c, s = math.cos(0.02), math.sin(0.02)
    yaw = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device="cuda")
    moved[:, :3, :3] = yaw @ moved[:, :3, :3]

    pre_pos, pre_rot = _achieved_pose_errors(world, q0, moved, "left")
    assert bool((pre_pos > 0.002).all())  # drift meaningfully displaced the target

    q1 = gn_project(world, q0, moved, "left")
    post_pos, post_rot = _achieved_pose_errors(world, q1, moved, "left")
    assert bool((post_pos < 0.005).all()), f"pos {post_pos.max().item():.4f}"
    assert bool((post_rot < 0.05).all()), f"rot {post_rot.max().item():.4f}"
    # And it should be a small correction, not a re-solve to a distant basin.
    assert (q1 - q0).abs().max().item() < 0.2


@needs_cuda
def test_gn_com_row_inactive_matches_pose_only():
    # When every particle's CoM is inside the rectangle, the CoM rows are
    # zeroed (active set) and the augmented solve must equal the pose-only one.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project
    from cutamp.particle_initialization import _ik_for_pose, _ik_solution_to_full_q

    world = make_blocks_t1_world()
    B = 8
    target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
    target[:, :3, 3] = torch.tensor([0.35, 0.25, 0.25], device="cuda")
    q0 = _ik_solution_to_full_q(_ik_for_pose(world, target, "left"), world)
    assert bool(compute_com_polygon_mask(world, q0).all())
    moved = target.clone()
    moved[:, 0, 3] += 0.003
    q_pose = gn_project(world, q0, moved, "left", com_row=False)
    q_com = gn_project(world, q0, moved, "left", com_row=True)
    assert torch.allclose(q_pose, q_com, atol=1e-5)


@needs_cuda
def test_gn_com_row_recenters_lean():
    # A pose-converged but leaned-out conf: the CoM row must pull the CoM back
    # into the support rectangle while (re)converging the hand pose. Without
    # the row, GN has no balance knowledge and leaves the lean in place.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project
    from cutamp.particle_initialization import (
        _achieved_pose_errors, _ik_for_pose, _ik_solution_to_full_q,
    )

    world = make_blocks_t1_world()
    B = 8
    target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
    target[:, :3, 3] = torch.tensor([0.35, 0.25, 0.25], device="cuda")
    q0 = _ik_solution_to_full_q(_ik_for_pose(world, target, "left"), world)

    # Lean the ankle forward hard: big CoM displacement, hand dragged along.
    names = list(world.kinematics.joint_names)
    q_bad = q0.clone()
    q_bad[:, names.index("ankle_pitch")] += 0.45
    assert not bool(compute_com_polygon_mask(world, q_bad).any())

    # 12 iters: this correction is ~40x the refresh-scale drift GN_ITERS=3 is
    # budgeted for, and the step clamp (a crude trust region — unclamped
    # steps overshoot badly here) makes giant corrections converge linearly
    # at ~0.7/iter. CoM itself recovers in ONE iteration; in production a
    # row this far gone would go to the cuRobo fallback, by design.
    q_fixed = gn_project(world, q_bad, target, "left", com_row=True, iters=12)
    pos_err, rot_err = _achieved_pose_errors(world, q_fixed, target, "left")
    assert bool(compute_com_polygon_mask(world, q_fixed).all()), "CoM not recovered"
    assert bool((pos_err < 0.005).all()) and bool((rot_err < 0.05).all()), (
        f"pose lost: {pos_err.max().item():.4f} m, {rot_err.max().item():.4f} rad"
    )


@needs_cuda
def test_gn_project_respects_locked_dofs():
    from cutamp.gn_refresh import gn_project
    from cutamp.robots.t1 import BASE_INDICES, RIGHT_ARM_INDICES

    world = make_blocks_t1_world()
    B = 4
    q0 = world.q_init.unsqueeze(0).expand(B, 21).clone()
    target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
    target[:, :3, 3] = torch.tensor([0.35, 0.25, 0.25], device="cuda")
    q1 = gn_project(world, q0, target, "left")
    assert torch.equal(q1[:, BASE_INDICES], q0[:, BASE_INDICES])
    assert torch.equal(q1[:, RIGHT_ARM_INDICES], q0[:, RIGHT_ARM_INDICES])
    # ...and it did actually move the unlocked DOFs toward the target.
    assert not torch.equal(q1, q0)
