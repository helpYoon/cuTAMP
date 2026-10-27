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
  A/B-rejected 2026-08-04 (null runtime win, quality parity), but that
  version's CoM Jacobian had its x row overwritten by the y row (see the
  clone in ``gn_project_balanced``), so the rejection says nothing about a
  correct CoM row,
- no basin hopping (strictly local; rows that don't converge to checker
  tolerance are handed to the cuRobo fallback by the caller).

``gn_project_balanced`` is the balance-first variant behind
``config.reik_balance_first``: CoM-in-polygon is a hard first level, the
pose is least squares inside it, and joint-limit margins use what is left.
"""

import roma
import torch

from curobo.types import JointState

from cutamp.robots.t1 import (
    BASE_INDICES, LEFT_ARM_INDICES, RIGHT_ARM_INDICES, T1_LIMIT_MARGIN_LOWER, T1_LIMIT_MARGIN_UPPER,
)

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
# Balance-first only: a CoM axis outside the support rectangle (or that a step
# would push out) is driven this far inside the edge, absorbing ~2-4 mm of
# linearization error. Axes already inside are left alone: optimized confs
# cluster within 1 cm of the edge, and a 1 cm activation band knocked 27/240
# of them off their pose on the captured rows.
GN_COM_MARGIN = 0.005
# Balance-first only, level 3 (joint margins). Damping: a joint with little
# freedom left after balance and pose would otherwise demand a huge
# compensating motion from the others; this caps it near 5x its own error.
GN_MARGIN_DAMPING = 1e-2
# Per-iteration |dq| cap on level 3. Its null-space steps hold levels 1-2 to
# first order only: at the full GN_STEP_CLAMP box their second-order drift
# walked edge-hugging place confs' CoM out of the rectangle, and level 1's
# correction then clamped joints at hard limits (9-20/512 place confs lost
# their pose per refresh; 0-2 at this cap). Margins need ~0.1 rad and get a
# refresh every 5 Adam steps, so small steps cost little.
GN_MARGIN_STEP = 0.03


def _setup(world, q_full: torch.Tensor, arm: str):
    """Kinematics handles, joint limits and the cspace-pin column mask."""
    kin = world.kinematics_jac
    frame = world.tool_frame_for_arm(arm)
    frame_idx = list(kin.tool_frames).index(frame)
    joint_names = list(kin.config.kinematics_config.joint_names)
    q = q_full.detach().clone()

    limits = world.kinematics.get_joint_limits().position
    lower = limits[0].to(q.device, q.dtype)
    upper = limits[1].to(q.device, q.dtype)

    # Column mask = the IK solver's cspace pin: base + inactive arm immobile.
    col_mask = torch.ones(q.shape[1], device=q.device, dtype=q.dtype)
    col_mask[BASE_INDICES] = 0.0
    if arm == "left":
        col_mask[RIGHT_ARM_INDICES] = 0.0
    elif arm == "right":
        col_mask[LEFT_ARM_INDICES] = 0.0
    return kin, frame, frame_idx, joint_names, q, lower, upper, col_mask


def _pose_residual_and_jacobian(state, frame, frame_idx, world_from_ee, col_mask):
    """World-frame residual ``[Δpos; rotvec(R* R^T)]`` ``[B, 6]`` and the
    column-masked tool Jacobian ``[B, 6, dof]``."""
    B = world_from_ee.shape[0]
    pose = state.tool_poses.get_link_pose(frame)
    pos = pose.position.reshape(B, 3)
    # cuRobo quaternions are wxyz; roma expects xyzw.
    rot = roma.unitquat_to_rotmat(
        pose.quaternion.reshape(B, 4)[:, [1, 2, 3, 0]].contiguous()
    )
    r_pos = world_from_ee[:, :3, 3] - pos
    r_rot = roma.rotmat_to_rotvec(world_from_ee[:, :3, :3] @ rot.transpose(-1, -2))
    # Rows 0-2 linear, 3-5 angular, world frame (finite-diff verified;
    # see test_gn_refresh.py).
    jac = state.tool_jacobians[:, 0, frame_idx] * col_mask
    return torch.cat([r_pos, r_rot], dim=-1), jac


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
    kin, frame, frame_idx, joint_names, q, lower, upper, col_mask = _setup(world, q_full, arm)
    eye6 = GN_DAMPING * torch.eye(6, device=q.device, dtype=q.dtype)

    with torch.no_grad():
        for _ in range(iters):
            js = JointState.from_position(q, joint_names=joint_names)
            state = kin.compute_kinematics(js)
            r, jac = _pose_residual_and_jacobian(state, frame, frame_idx, world_from_ee, col_mask)
            jjt = jac @ jac.transpose(-1, -2) + eye6                # [B, 6, 6]
            y = torch.linalg.solve(jjt, r.unsqueeze(-1))
            dq = (jac.transpose(-1, -2) @ y).squeeze(-1)            # [B, dof]
            q = (q + dq.clamp(-GN_STEP_CLAMP, GN_STEP_CLAMP)).clamp(lower, upper)

    return q


def _prioritized_step(jac_com, com, active, com_target, jac, r, damping):
    """One two-level step (Siciliano-Slotine form). Level 1 moves the active
    CoM axes to ``com_target`` with minimum norm; level 2 is damped GN on the
    pose residual inside the null space of the active CoM rows. Inactive CoM
    rows are zeroed, so they drop out of both levels."""
    a = active.to(jac.dtype)
    ja = jac_com * a.unsqueeze(-1)                                  # [B, 2, dof]
    gram = ja @ ja.transpose(-1, -2) + 1e-6 * torch.eye(2, device=jac.device, dtype=jac.dtype)
    ja_pinv = ja.transpose(-1, -2) @ torch.linalg.inv(gram)         # [B, dof, 2]
    dq1 = (ja_pinv @ ((com_target - com) * a).unsqueeze(-1)).squeeze(-1)
    eye_dof = torch.eye(jac.shape[-1], device=jac.device, dtype=jac.dtype)
    jn = jac @ (eye_dof - ja_pinv @ ja)                             # [B, 6, dof]
    eye6 = torch.eye(6, device=jac.device, dtype=jac.dtype)
    rhs = r - (jac @ dq1.unsqueeze(-1)).squeeze(-1)
    y = torch.linalg.solve(jn @ jn.transpose(-1, -2) + damping * eye6, rhs.unsqueeze(-1))
    dq2 = (jn.transpose(-1, -2) @ y).squeeze(-1)
    return dq1, dq2


def _fraction_that_fits(base, extra):
    """Largest ``f`` in [0, 1] with ``|base + f * extra| <= GN_STEP_CLAMP`` per joint."""
    room = (GN_STEP_CLAMP - base * extra.sign()) / extra.abs().clamp_min(1e-12)
    return room.amin(-1, keepdim=True).clamp(0.0, 1.0)


def _margin_step(q, step, margin_err, jac, jac_com, com, locked, inner, free, lower, upper):
    """Level 3: move joints with nonzero ``margin_err`` (rad to their margin
    band's edge) by at most ``GN_MARGIN_STEP``, on top of the levels 1-2
    ``step``, without moving the pose or the ``locked`` CoM axes to first
    order. Its task Jacobian is a joint selector, so its projected Jacobian is
    the selected rows of the null-space projector.

    Active sets, re-solved up to three times like levels 1-2: a CoM axis this
    step would push past ``inner`` is locked, and a joint it would push past
    a hard limit is frozen (joints frozen by levels 1-2 stay frozen).
    Float64 because the explicit projector is ill-conditioned (cond(A A^T)
    ~5e5 on place confs): in float32 it leaked ~5e-3 of each step into the
    pose, the damped solve amplified that ~15x, and ~80/512 place confs lost
    their pose per refresh.
    """
    f64 = torch.float64
    eye_dof = torch.eye(q.shape[-1], device=q.device, dtype=f64)
    eye8 = torch.eye(8, device=q.device, dtype=f64)
    for _ in range(3):
        a = (torch.cat([jac_com * locked.to(q.dtype).unsqueeze(-1), jac], dim=1) * free.unsqueeze(1)).to(f64)
        proj = eye_dof - a.transpose(-1, -2) @ torch.linalg.solve(a @ a.transpose(-1, -2) + 1e-12 * eye8, a)
        sel = (margin_err * free != 0).to(f64)
        m = sel.unsqueeze(-1) * proj                                # [B, dof, dof]
        rhs = sel * (margin_err - step).to(f64)
        y = torch.linalg.solve(m @ m.transpose(-1, -2) + GN_MARGIN_DAMPING * eye_dof, rhs.unsqueeze(-1))
        dq3 = (m.transpose(-1, -2) @ y).squeeze(-1).to(q.dtype)
        s3 = _fraction_that_fits(step, dq3) * dq3
        s3 = s3 * (GN_MARGIN_STEP / s3.abs().amax(-1, keepdim=True).clamp_min(GN_MARGIN_STEP))
        q_lin = q + step + s3
        crossed = (((q_lin > upper) & (s3 > 0)) | ((q_lin < lower) & (s3 < 0))) & free.bool()
        com_lin = com + (jac_com @ (step + s3).unsqueeze(-1)).squeeze(-1)
        pushed = ~locked & (com_lin.abs() > inner) & (com_lin.sign() * (jac_com @ s3.unsqueeze(-1)).squeeze(-1) > 0)
        if not bool(crossed.any() or pushed.any()):
            break
        free = free * ~crossed
        locked = locked | pushed
    return s3


def gn_project_balanced(
    world,
    q_full: torch.Tensor,
    world_from_ee: torch.Tensor,
    arm: str,
    iters: int = GN_ITERS,
) -> torch.Tensor:
    """Balance-first variant of :func:`gn_project` (``config.reik_balance_first``).

    Each step is a three-level hierarchy (Kanoun et al. T-RO 2011; Escande et
    al. IJRR 2014): (1) keep the base-frame CoM inside the support rectangle
    shrunk by ``GN_COM_MARGIN``; (2) least squares on the pose residual in
    whatever freedom level 1 leaves; (3) move joints out of the
    ``joint_limit_margin`` bands (robots/t1.py margins) in whatever freedom
    is left, in small steps (:func:`_margin_step`). An unreachable target
    therefore returns the closest balanced pose, not an unbalanced exact one,
    and its leftover pose error is what the kinematic cost hands to the
    placement. Level 3 exists because covered confs are outside Adam, so the
    margin soft cost never reaches them: on blocks_t1 some runs initialized
    every particle's pick-conf Torso_Pitch ~3 mrad from its limit and kept a
    soft cost near 9.35 all run.

    Differences from :func:`gn_project` beyond the hierarchy:
    - steps are scaled into the ``GN_STEP_CLAMP`` box (level 1 first, then
      the largest fractions of levels 2 and 3 that fit), not clipped per
      joint, so they keep their direction;
    - damping grows with the squared residual (Levenberg-Marquardt), which
      keeps steps sane where the two levels conflict;
    - the CoM Jacobian comes from autograd through cuRobo's FK, so each
      iteration costs about two pose-only iterations.

    Same contract as :func:`gn_project`: ``[B, dof]`` in, ``[B, dof]`` out,
    convergence not guaranteed; callers measure the achieved error.
    """
    from cutamp.com_polygon_cost import COM_HALF_EXTENTS, _project_com_to_base

    kin, frame, frame_idx, joint_names, q, lower, upper, col_mask = _setup(world, q_full, arm)
    B = q.shape[0]
    half = torch.tensor(COM_HALF_EXTENTS, device=q.device, dtype=q.dtype)
    inner = half - GN_COM_MARGIN
    band_lo = lower + torch.tensor(T1_LIMIT_MARGIN_LOWER, device=q.device, dtype=q.dtype)
    band_hi = upper - torch.tensor(T1_LIMIT_MARGIN_UPPER, device=q.device, dtype=q.dtype)

    for _ in range(iters):
        q_g = q.detach().requires_grad_(True)
        with torch.enable_grad():
            state = kin.compute_kinematics(JointState.from_position(q_g, joint_names=joint_names))
            with torch.no_grad():  # base joints are pinned, so the base pose is constant
                base_T = (
                    state.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True)
                    .get_matrix().reshape(B, 4, 4)
                )
            com_g = _project_com_to_base(state.robot_com[..., :3].reshape(B, 3), base_T)
            # Clone: cuRobo's backward writes into a reused buffer, so without
            # it the second grad call overwrites the x row with the y row.
            jcx = torch.autograd.grad(com_g[:, 0].sum(), q_g, retain_graph=True)[0].clone()
            jcy = torch.autograd.grad(com_g[:, 1].sum(), q_g)[0]
        with torch.no_grad():
            jac_com = torch.stack([jcx, jcy], dim=1) * col_mask    # [B, 2, dof]
            com = com_g.detach()
            r, jac = _pose_residual_and_jacobian(state, frame, frame_idx, world_from_ee, col_mask)
            damping = (GN_DAMPING + (r * r).sum(-1)).view(B, 1, 1)

            # Two active sets, re-solved up to three times: CoM axes outside
            # the rectangle or that the step's linearization would push out
            # (driven to ``inner``), and joints at a limit that the step
            # pushes further (columns frozen, as in clamping-loop / SNS IK).
            # Without the second, the final limit clamp discards the
            # CoM-correcting part of the step and the pose part walks the
            # CoM out.
            active = com.abs() > half
            com_target = com.sign() * inner
            free = torch.ones_like(q)
            for _ in range(3):
                fc = free.unsqueeze(1)
                dq1, dq2 = _prioritized_step(jac_com * fc, com, active, com_target, jac * fc, r, damping)
                dq = dq1 + dq2
                com_lin = com + (jac_com @ dq.unsqueeze(-1)).squeeze(-1)
                pushed_out = ~active & (com_lin.abs() > half)
                blocked = ((q >= upper - 1e-5) & (dq > 0)) | ((q <= lower + 1e-5) & (dq < 0))
                blocked &= free.bool()
                if not bool(pushed_out.any() or blocked.any()):
                    break
                com_target = torch.where(pushed_out, com_lin.sign() * inner, com_target)
                active = active | pushed_out
                free = free * ~blocked

            # Trust region in priority order: scale dq1 into the box, then add
            # the largest fraction of dq2 that still fits; level 3 takes what
            # room is left.
            step = _fraction_that_fits(torch.zeros_like(dq1), dq1) * dq1
            step = step + _fraction_that_fits(step, dq2) * dq2
            margin_err = ((band_lo - q).clamp_min(0) - (q - band_hi).clamp_min(0)) * col_mask
            step = step + _margin_step(q, step, margin_err, jac, jac_com, com, active, inner, free, lower, upper)
            q = (q + step).clamp(lower, upper)

    return q.detach()
