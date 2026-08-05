# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Batched Gauss-Newton projection for coupled-reIK refreshes.

Between refreshes, each covered Conf already holds a near-answer for a pose
target that Adam has moved by millimetres. Instead of re-running cuRobo's
fixed-schedule IK pipeline (~170 ms per dep regardless of how converged the
warm start is), project the stored q back onto the moved constraint manifold
with damped Gauss-Newton on the 6-DOF pose residual:

    dq = J^T (J J^T + lambda I)^-1 [t* - t(q); rotvec(R* R(q)^T)]

Cost: ``GN_ITERS`` batched FK+Jacobian kernel calls plus a [B, 6, 6] solve —
about 2-5 ms per dep at 512 particles.

What this deliberately does NOT do (the full cuRobo solver covers these at
particle init, on per-refresh fallback, and at the final pre-check refresh):

- no self/world-collision vetting inside the update (the rollout's hard
  constraints still score every conf every step, so violations surface in
  the satisfying mask rather than silently passing),
- no CoM residual — the CALLER must AND ``compute_com_polygon_mask`` into
  its merge gate so out-of-polygon rows route to the CoM-aware cuRobo
  fallback (a pose-only gate demonstrably locks in CoM decay: GN keeps pose
  converged while the conf drifts out of the support polygon, and nothing
  else re-centers covered Confs during Adam). An in-GN CoM residual row was
  A/B-rejected 2026-08-04 (null runtime win, quality parity) — see git
  history if the K=1 future ever revives it,
- no basin hopping (strictly local; rows that don't converge to checker
  tolerance are handed to the cuRobo fallback by the caller).
"""

import roma
import torch

from curobo.types import JointState

from cutamp.robots.t1 import BASE_INDICES, LEFT_ARM_INDICES, RIGHT_ARM_INDICES

# ponytail: fixed schedule tuned for mm-scale drift; revisit only if the
# fallback-rate instrumentation shows a fat non-converged band.
GN_ITERS = 3
# Levenberg damping on JJ^T. Units mix metres/radians; at mm/mrad residuals
# the conditioning effect dominates and the bias is negligible.
GN_DAMPING = 1e-4
# Per-iteration |dq_i| cap (rad or metre for the virtual base — base is
# masked anyway). Acts as a crude trust region: measured on a 0.45 rad lean
# (18 cm pose error), UNclamped steps overshoot to 45+ cm while clamped
# descent converges linearly at ~0.7x error/iter. mm-scale refresh drift
# never saturates it, so warm-start convergence stays effectively quadratic;
# corrections big enough to ride the clamp belong to the cuRobo fallback.
GN_STEP_CLAMP = 0.15


def gn_project(
    world,
    q_full: torch.Tensor,
    world_from_ee: torch.Tensor,
    arm: str,
    iters: int = GN_ITERS,
) -> torch.Tensor:
    """Project ``q_full`` toward ``FK(q) == world_from_ee`` on ``arm``'s tool frame.

    Args:
        world: TAMPWorld (uses ``kinematics_jac``, the Jacobian-enabled FK).
        q_full: ``[B, dof]`` full-cspace joint positions (not modified).
        world_from_ee: ``[B, 4, 4]`` target tool poses.
        arm: "left" / "right" — the OTHER arm and the virtual base are locked,
            mirroring the IK solver's cspace pin.

    Returns:
        ``[B, dof]`` projected joint positions, clamped to joint limits.
        Convergence is NOT guaranteed — callers must measure the achieved
        error (``_achieved_pose_errors``) and fall back for rows over
        tolerance.
    """
    kin = world.kinematics_jac
    frame = world.tool_frame_for_arm(arm)
    frame_idx = list(kin.tool_frames).index(frame)
    joint_names = list(kin.config.kinematics_config.joint_names)
    B, dof = q_full.shape
    q = q_full.detach().clone()

    limits = world.kinematics.get_joint_limits().position
    lower = limits[0].to(q.device, q.dtype)
    upper = limits[1].to(q.device, q.dtype)

    # Column mask = the IK solver's cspace pin: base + inactive arm immobile.
    col_mask = torch.ones(dof, device=q.device, dtype=q.dtype)
    col_mask[BASE_INDICES] = 0.0
    if arm == "left":
        col_mask[RIGHT_ARM_INDICES] = 0.0
    elif arm == "right":
        col_mask[LEFT_ARM_INDICES] = 0.0

    target_pos = world_from_ee[:, :3, 3]
    target_rot = world_from_ee[:, :3, :3]
    eye6 = GN_DAMPING * torch.eye(6, device=q.device, dtype=q.dtype)

    with torch.no_grad():
        for _ in range(iters):
            js = JointState.from_position(q, joint_names=joint_names)
            state = kin.compute_kinematics(js)
            pose = state.tool_poses.get_link_pose(frame)
            pos = pose.position.reshape(B, 3)
            # cuRobo quaternions are wxyz; roma expects xyzw.
            rot = roma.unitquat_to_rotmat(
                pose.quaternion.reshape(B, 4)[:, [1, 2, 3, 0]].contiguous()
            )

            # World-frame residual [Δpos; rotvec(R* R^T)].
            r_pos = target_pos - pos
            r_rot = roma.rotmat_to_rotvec(target_rot @ rot.transpose(-1, -2))
            r = torch.cat([r_pos, r_rot], dim=-1)                   # [B, 6]

            # Rows 0-2 linear, 3-5 angular, world frame (finite-diff verified;
            # see test_gn_refresh.py).
            jac = state.tool_jacobians[:, 0, frame_idx] * col_mask  # [B, 6, dof]
            jjt = jac @ jac.transpose(-1, -2) + eye6                # [B, 6, 6]
            y = torch.linalg.solve(jjt, r.unsqueeze(-1))
            dq = (jac.transpose(-1, -2) @ y).squeeze(-1)            # [B, dof]
            q = (q + dq.clamp(-GN_STEP_CLAMP, GN_STEP_CLAMP)).clamp(lower, upper)

    return q
