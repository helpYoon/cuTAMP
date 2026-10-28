"""Tests for the Gauss-Newton reIK refresh (cutamp/gn_refresh.py).

CUDA-only: exercises the real T1 kinematics kernel — the Jacobian convention
check is the load-bearing regression (GN silently diverges if cuRobo ever
changes row order or frame).
"""
import math
from unittest import mock

import pytest
import roma
import torch

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def _four_dep_stack(world):
    """A refresh's dep-major stack of four 8-row IK deps: ``(q [32, 21],
    T [32, 4, 4], arms)``. Per arm, one IK solution for an identity-rotation
    target at (0.35, +-0.25, 0.25) feeds a warm dep (its target drifted by mm,
    as between refreshes) and a far dep (x = 0.9, out of balanced reach)."""
    from cutamp.particle_initialization import _ik_for_pose, _ik_solution_to_full_q, _ik_success_mask

    B = 8
    qs, targets = [], []
    for arm, y in (("left", 0.25), ("right", -0.25)):
        target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
        target[:, :3, 3] = torch.tensor([0.35, y, 0.25], device="cuda")
        ik_result = _ik_for_pose(world, target, arm)
        assert bool(_ik_success_mask(ik_result).all()), arm
        q0 = _ik_solution_to_full_q(ik_result, world)
        moved = target.clone()
        moved[:, 0, 3] += 0.003
        moved[:, 2, 3] += 0.0025
        far = target.clone()
        far[:, 0, 3] = 0.9
        qs += [q0, q0]
        targets += [moved, far]
    return torch.cat(qs), torch.cat(targets), ("left", "left", "right", "right")


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


@needs_cuda
def test_gn_project_balanced_keeps_com_in_and_closes_pose():
    # Balance-first GN: from a warm start it converges like gn_project; from a
    # target well out of balanced reach it ends CoM-inside and closer to the
    # target than it started (closest balanced pose, not an unbalanced exact one).
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project_balanced
    from cutamp.particle_initialization import (
        _achieved_pose_errors, _ik_for_pose, _ik_solution_to_full_q,
    )

    world = make_blocks_t1_world()
    B = 8
    target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
    target[:, :3, 3] = torch.tensor([0.35, 0.25, 0.25], device="cuda")
    q0 = _ik_solution_to_full_q(_ik_for_pose(world, target, "left"), world)
    moved = target.clone()
    moved[:, 0, 3] += 0.003
    moved[:, 2, 3] += 0.0025
    q1 = gn_project_balanced(world, q0, moved, "left")
    pos, rot = _achieved_pose_errors(world, q1, moved, "left")
    assert bool((pos < 0.005).all() & (rot < 0.05).all()), f"pos {pos.max().item():.4f}"
    assert bool(compute_com_polygon_mask(world, q1).all())

    far = target.clone()
    far[:, 0, 3] = 0.9
    pos0, _ = _achieved_pose_errors(world, q0, far, "left")
    q2 = gn_project_balanced(world, q0, far, "left", iters=30)
    pos2, _ = _achieved_pose_errors(world, q2, far, "left")
    assert bool(compute_com_polygon_mask(world, q2).all())
    assert bool((pos2 < pos0 - 0.05).all()), f"{pos0.min().item():.3f} -> {pos2.max().item():.3f}"


@needs_cuda
def test_gn_project_balanced_leaves_margin_bands_holding_pose_and_com():
    # Level 3: Torso_Pitch 3 mrad above its hard lower limit (the pick confs
    # behind the 9.35 soft-cost mode) climbs out of its joint_limit_margin
    # band while the pose and CoM levels hold. This posture can't leave the
    # band entirely (it stalls ~0.04 rad deep with the CoM at the 5 mm lock),
    # so it checks steady progress: two-level GN would not move it at all.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project_balanced
    from cutamp.particle_initialization import (
        _achieved_pose_errors, _ik_for_pose, _ik_solution_to_full_q,
    )
    from cutamp.robots.t1 import T1_LIMIT_MARGIN_LOWER

    world = make_blocks_t1_world()
    B = 8
    target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
    target[:, :3, 3] = torch.tensor([0.35, 0.25, 0.25], device="cuda")
    q0 = _ik_solution_to_full_q(_ik_for_pose(world, target, "left"), world)
    tp = list(world.kinematics.joint_names).index("Torso_Pitch")
    lower = world.kinematics.get_joint_limits().position[0]
    q0[:, tp] = lower[tp] + 0.003
    assert bool(compute_com_polygon_mask(world, q0).all())
    pos, quat, _ = _fk_pose(world, q0, world.tool_frame_for_arm("left"))
    held = target.clone()
    held[:, :3, 3] = pos
    held[:, :3, :3] = _quat_to_mat(quat)

    q = q0
    for _ in range(8):  # one call per refresh
        q = gn_project_balanced(world, q, held, "left")
    band_edge = lower[tp] + T1_LIMIT_MARGIN_LOWER[tp]
    depth = (band_edge - q[:, tp]).clamp_min(0)
    assert bool((depth < 0.6 * (band_edge - q0[:, tp])).all()), f"depth {depth.max().item():.4f}"
    pos_err, rot_err = _achieved_pose_errors(world, q, held, "left")
    assert bool((pos_err < 0.005).all() & (rot_err < 0.05).all()), f"pos {pos_err.max().item():.4f}"
    assert bool(compute_com_polygon_mask(world, q).all())


@needs_cuda
def test_gn_project_balanced_stack_matches_per_dep():
    # One call on a refresh's whole dep-major stack matches one call per dep,
    # and so does a stack with no right-arm rows (measured bitwise; the bound
    # leaves room for GPU kernels whose reduction order depends on the batch).
    from cutamp.gn_refresh import gn_project_balanced

    world = make_blocks_t1_world()
    q, T, arms = _four_dep_stack(world)
    per_dep = torch.cat([gn_project_balanced(world, qi, Ti, a)
                         for qi, Ti, a in zip(q.split(8), T.split(8), arms)])
    assert (gn_project_balanced(world, q, T, arms) - per_dep).abs().max().item() <= 1e-5
    left_only = gn_project_balanced(world, q[:16], T[:16], arms[:2])
    assert (left_only - per_dep[:16]).abs().max().item() <= 1e-5


@needs_cuda
def test_gn_project_balanced_accepts_bare_arm():
    # Today's call style: a bare arm is one dep, not ("l", "e", "f", "t").
    # Rows that don't split into equal dep blocks, or an arm that is neither
    # left nor right, raise instead of picking a frame.
    from cutamp.gn_refresh import gn_project_balanced

    world = make_blocks_t1_world()
    q, T, arms = _four_dep_stack(world)
    assert torch.equal(gn_project_balanced(world, q[:8], T[:8], "left"),
                       gn_project_balanced(world, q[:8], T[:8], ("left",)))
    with pytest.raises(ValueError):
        gn_project_balanced(world, q[:10], T[:10], arms)
    with pytest.raises(ValueError):
        gn_project_balanced(world, q[:8], T[:8], "up")


@needs_cuda
def test_gn_project_balanced_pins_per_row():
    # Each row keeps its own dep's cspace pin and tool frame: the base never
    # moves, each arm moves only in its own deps' rows, and both warm deps
    # close their 3.9 mm drift on their own arm's frame (a right row on the
    # left frame would chase a target half a metre away).
    from cutamp.gn_refresh import gn_project_balanced
    from cutamp.particle_initialization import _achieved_pose_errors
    from cutamp.robots.t1 import BASE_INDICES, LEFT_ARM_INDICES, RIGHT_ARM_INDICES

    world = make_blocks_t1_world()
    q, T, arms = _four_dep_stack(world)
    q1 = gn_project_balanced(world, q, T, arms)
    assert torch.equal(q1[:, BASE_INDICES], q[:, BASE_INDICES])
    assert torch.equal(q1[:16, RIGHT_ARM_INDICES], q[:16, RIGHT_ARM_INDICES])
    assert torch.equal(q1[16:, LEFT_ARM_INDICES], q[16:, LEFT_ARM_INDICES])
    assert not torch.equal(q1[:16, LEFT_ARM_INDICES], q[:16, LEFT_ARM_INDICES])
    for rows, arm in ((slice(0, 8), "left"), (slice(16, 24), "right")):
        pos, rot = _achieved_pose_errors(world, q1[rows], T[rows], arm)
        assert bool((pos < 2e-3).all() & (rot < 2e-2).all()), (
            f"{arm}: pos {pos.max().item():.4f} rot {rot.max().item():.4f}"
        )


@needs_cuda
def test_gn_gate_matches_checker_helpers():
    # One FK over the whole stack gives the checker helpers' masks: per-dep
    # _achieved_pose_errors on each row's own arm, and compute_com_polygon_mask
    # on the stack and per dep. The lean is forward: a 0.45 rad back-lean
    # leaves every leaned CoM inside on this seeded draw; forward, every leaned
    # CoM was out on each of 20 draws (114 mm out on this one).
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_gate
    from cutamp.particle_initialization import _achieved_pose_errors

    world = make_blocks_t1_world()
    q, T, arms = _four_dep_stack(world)
    q = q.clone()
    q[::3, 3] -= 0.45  # lean the ankle on every third row: CoM leaves the polygon
    # Make every term of the masks matter: unleaned warm rows 1 and 2 (3.9 mm
    # off) get rotation-only errors between and beyond the tolerances, and
    # leaned row 0 aims at its own tool pose, so it is pose-ok but CoM-out.
    T = T.clone()
    for row, ang in ((1, 0.02), (2, 0.1)):
        T[row, :3, :3] = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, ang], device="cuda")) @ T[row, :3, :3]
    pos, quat, _ = _fk_pose(world, q[:8], world.tool_frame_for_arm("left"))
    T[0, :3, :3] = _quat_to_mat(quat[0])
    T[0, :3, 3] = pos[0]
    kin = world.kinematics
    with mock.patch.object(kin, "compute_kinematics", wraps=kin.compute_kinematics) as fk:
        pose_ok, com_ok = gn_gate(world, q, T, arms, 0.005, 0.05)
    assert fk.call_count == 1  # both masks from one FK
    ref = [_achieved_pose_errors(world, qi, Ti, a) for qi, Ti, a in zip(q.split(8), T.split(8), arms)]
    assert torch.equal(pose_ok, torch.cat([(p < 0.005) & (r < 0.05) for p, r in ref]))
    assert torch.equal(com_ok, compute_com_polygon_mask(world, q))
    assert torch.equal(com_ok, torch.cat([compute_com_polygon_mask(world, qi) for qi in q.split(8)]))
    assert pose_ok.any() and (~pose_ok).any() and com_ok.any() and (~com_ok).any()
    assert (pose_ok & ~com_ok).any()
