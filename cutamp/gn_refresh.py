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
fixed-schedule IK pipeline (~280 ms per dep regardless of how converged the
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
- CoM handling is opt-in (``com_row=True``): two extra one-sided residual
  rows pull an out-of-rectangle CoM back inside (see below). With
  ``com_row=False`` the CALLER must AND ``compute_com_polygon_mask`` into its
  merge gate so out-of-polygon rows route to the CoM-aware cuRobo fallback
  (a pose-only gate demonstrably locks in CoM decay: GN keeps pose converged
  while the conf drifts out of the support polygon, and nothing else
  re-centers covered Confs during Adam),
- no basin hopping (strictly local; rows that don't converge to checker
  tolerance are handed to the cuRobo fallback by the caller).

CoM residual row (``com_row=True``): the projected CoM in the
``mobile_base_link`` frame gets, per violated axis, a residual pulling it
``COM_INSIDE_MARGIN`` inside the support-rectangle boundary (the same
geometry ``compute_com_polygon_mask`` gates on). Rows are ACTIVE-SET
one-sided: residual AND Jacobian rows are zeroed for in-rectangle axes, so
balanced particles solve the exact pose-only system. The CoM Jacobian rows
come from autograd through the same FK call (the fused kernel carries CoM
gradients — this is what the cuRobo fork's seed-IK residual uses); with 11
free DOFs against 6 pose + <=2 CoM rows, damped least squares recruits the
leverage joints (ankle/torso) while the arm compensates to hold the tool.
"""

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
# CoM row scale vs the pose rows (both in metres; the fork's validated
# seed-IK CoM weight is likewise 1.0 pose-equivalent).
GN_COM_WEIGHT = 1.0


def _quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """``[B, 4]`` wxyz quaternions -> ``[B, 3, 3]`` rotation matrices."""
    w, x, y, z = quat.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def gn_project(
    world,
    q_full: torch.Tensor,
    world_from_ee: torch.Tensor,
    arm: str,
    iters: int = GN_ITERS,
    com_row: bool = False,
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
    if com_row:
        from cutamp.com_polygon_cost import (
            COM_HALF_EXTENTS, COM_INSIDE_MARGIN, _project_com_to_base,
        )
        half = torch.tensor(COM_HALF_EXTENTS, device=q.device, dtype=q.dtype)
        inner = half - COM_INSIDE_MARGIN
        # Hysteresis (Schmitt) active set per axis, per particle: a row
        # activates when the CoM crosses the gate boundary (|d| > half) and
        # releases only once pulled comfortably inside (|d| < release). The
        # band kills both failure modes of the naive alternatives: pure
        # boundary switching chatters (pose steps re-lean, the re-activated
        # pull undoes pose progress), and a permanently-sticky row taxes
        # pose convergence forever via its J_com dq ~ 0 side constraint.
        release = half - 0.5 * COM_INSIDE_MARGIN
        sticky = torch.zeros(B, 2, dtype=torch.bool, device=q.device)

    for _ in range(iters):
        jac_com = None
        if com_row:
            # FK with a live graph so autograd can produce the base-frame CoM
            # Jacobian rows. Building the projected CoM in-graph means frame
            # handling (base translation + yaw) is differentiated for us.
            q_g = q.detach().clone().requires_grad_(True)
            js = JointState.from_position(q_g, joint_names=joint_names)
            state = kin.compute_kinematics(js)
            # base_T is CONSTANT w.r.t. the free DOFs (base joints are locked
            # by col_mask), so build it outside the graph: gradients then flow
            # only through robot_com (the fused kernel's CoM backward), never
            # through the pose-transform autograd, whose reused quaternion
            # buffers trip "modified by an inplace operation" on re-backward.
            with torch.no_grad():
                base_T = (
                    state.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True)
                    .get_matrix().reshape(B, 4, 4)
                )
            com_in_base = _project_com_to_base(
                state.robot_com[..., :3].reshape(B, 3), base_T,
            )
            d = com_in_base[:, :2]
            # Activation boundary = the |d| <= half rectangle
            # compute_com_polygon_mask gates on; release inside the band.
            d_abs = d.abs().detach()
            sticky = (sticky | (d_abs > half)) & ~(d_abs < release)
            active = sticky
            if bool(active.any()):
                jc_x = torch.autograd.grad(com_in_base[:, 0].sum(), q_g, retain_graph=True)[0]
                jc_y = torch.autograd.grad(com_in_base[:, 1].sum(), q_g)[0]
                jac_com = torch.stack([jc_x, jc_y], dim=1)       # [B, 2, dof]
            d = d.detach()
        else:
            with torch.no_grad():
                js = JointState.from_position(q, joint_names=joint_names)
                state = kin.compute_kinematics(js)

        with torch.no_grad():
            pose = state.tool_poses.get_link_pose(frame)
            pos = pose.position.reshape(B, 3)
            rot = _quat_to_rotmat(pose.quaternion.reshape(B, 4).contiguous())

            # World-frame residual [Δpos; rotvec(R* R^T)]; atan2 form is exact
            # near identity (same rationale as curobo_pose_error).
            r_pos = target_pos - pos
            d_rot = target_rot @ rot.transpose(-1, -2)
            skew = 0.5 * torch.stack([
                d_rot[:, 2, 1] - d_rot[:, 1, 2],
                d_rot[:, 0, 2] - d_rot[:, 2, 0],
                d_rot[:, 1, 0] - d_rot[:, 0, 1],
            ], dim=-1)                                           # sin(theta) * axis
            sin_norm = skew.norm(dim=-1, keepdim=True)
            cos_part = (d_rot.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) - 1.0) * 0.5
            angle = torch.atan2(sin_norm, cos_part)
            r_rot = skew / sin_norm.clamp_min(1e-9) * angle
            r = torch.cat([r_pos, r_rot], dim=-1)                # [B, 6]

            # Rows 0-2 linear, 3-5 angular, world frame (finite-diff verified;
            # see test_gn_refresh.py).
            jac = state.tool_jacobians[:, 0, frame_idx] * col_mask  # [B, 6, dof]
            if jac_com is not None:
                # Pull each violated axis COM_INSIDE_MARGIN inside the
                # boundary. Zero residual AND Jacobian rows for inactive axes
                # — a zero residual with a live row would wrongly constrain
                # the CoM to not move at all.
                r_com = GN_COM_WEIGHT * torch.where(
                    active, d.clamp(-inner, inner) - d, torch.zeros_like(d),
                )
                jac_c = GN_COM_WEIGHT * jac_com.detach() * col_mask * active.unsqueeze(-1)
                r = torch.cat([r, r_com], dim=-1)                # [B, <=8]
                jac = torch.cat([jac, jac_c], dim=1)             # [B, <=8, dof]
            n_rows = jac.shape[1]
            jjt = jac @ jac.transpose(-1, -2) + GN_DAMPING * torch.eye(
                n_rows, device=q.device, dtype=q.dtype,
            )
            y = torch.linalg.solve(jjt, r.unsqueeze(-1))
            dq = (jac.transpose(-1, -2) @ y).squeeze(-1)            # [B, dof]
            q = (q + dq.clamp(-GN_STEP_CLAMP, GN_STEP_CLAMP)).clamp(lower, upper)

    return q.detach()
