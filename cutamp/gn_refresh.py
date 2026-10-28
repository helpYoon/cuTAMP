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
about 2-5 ms per dep at 512 particles. ``gn_project_balanced`` projects all
of a refresh's deps in one call, as a dep-major stack whose rows each keep
their own arm's tool frame and cspace pin, and makes no host syncs: the Adam
loop is host-launch-bound, and one call per dep cost hundreds of syncs per
refresh. So its active-set rounds always run in full, its solves are the
``*_ex`` kernels without the error check, its level-3 SPD system goes
through Cholesky (batched 21x21 f64 LU dispatches to MAGMA, which syncs),
and its constants are cached on the device.

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

from functools import lru_cache
from typing import Sequence

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


@lru_cache
def _row_layout(arms: tuple[str, ...], rows: int, dof: int, device, dtype):
    """Per-row layout of ``len(arms)`` dep-major blocks of ``rows`` rows each,
    block ``i`` on ``arms[i]``: ``(col_mask [N, dof], is_left [N] bool)`` with
    ``N = len(arms) * rows``.

    The column mask is the IK solver's cspace pin (base + inactive arm
    immobile); ``is_left`` picks each row's tool frame. Made once per layout:
    building it from Python data is a blocking host-to-device copy. Shared;
    never mutate.
    """
    if not set(arms) <= {"left", "right"}:
        raise ValueError(f"arms must each be 'left' or 'right', got {arms}")
    col_mask = torch.ones(len(arms), dof, dtype=dtype)
    col_mask[:, BASE_INDICES] = 0.0
    for i, arm in enumerate(arms):
        col_mask[i, RIGHT_ARM_INDICES if arm == "left" else LEFT_ARM_INDICES] = 0.0
    is_left = torch.tensor([arm == "left" for arm in arms])
    return (
        col_mask.repeat_interleave(rows, dim=0).to(device),
        is_left.repeat_interleave(rows).to(device),
    )


@lru_cache
def _limit_margins(device, dtype):
    """``T1_LIMIT_MARGIN_LOWER/UPPER`` on ``device``, made once (a blocking
    host-to-device copy otherwise). Shared; never mutate."""
    return (
        torch.tensor(T1_LIMIT_MARGIN_LOWER, device=device, dtype=dtype),
        torch.tensor(T1_LIMIT_MARGIN_UPPER, device=device, dtype=dtype),
    )


def _setup(world, q_full: torch.Tensor, arms: tuple[str, ...]):
    """Kinematics handles, joint limits, both tool frames as ``((left, jac
    index), (right, jac index))``, and the :func:`_row_layout` of ``arms``."""
    kin = world.kinematics_jac
    frames = tuple(
        (frame, list(kin.tool_frames).index(frame))
        for frame in (world.tool_frame_for_arm("left"), world.tool_frame_for_arm("right"))
    )
    joint_names = list(kin.config.kinematics_config.joint_names)
    q = q_full.detach().clone()

    limits = world.kinematics.get_joint_limits().position
    lower = limits[0].to(q.device, q.dtype)
    upper = limits[1].to(q.device, q.dtype)
    col_mask, is_left = _row_layout(arms, q.shape[0] // len(arms), q.shape[1], q.device, q.dtype)
    return kin, frames, joint_names, q, lower, upper, col_mask, is_left


def _rotmat_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """``roma.rotmat_to_rotvec`` (roma 1.5.4: ``rotmat_to_unitquat``, then
    shortest-arc ``unitquat_to_rotvec``), bit for bit, without its host syncs:
    roma splits rows by ``nonzero`` twice, so this builds all four quaternion
    candidates and gathers each row's ``argmax`` choice, and picks the
    small/large-angle scale with ``torch.where``."""
    m = R.reshape(-1, 3, 3)
    diag = m.diagonal(dim1=-2, dim2=-1)
    tr = diag.sum(-1)
    choice = torch.cat([diag, tr[:, None]], dim=-1).argmax(-1)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.reshape(-1, 9).unbind(-1)
    cand = torch.stack([                                            # [n, choice, xyzw]
        torch.stack([1 - tr + 2 * m00, m10 + m01, m20 + m02, m21 - m12], -1),
        torch.stack([m01 + m10, 1 - tr + 2 * m11, m21 + m12, m02 - m20], -1),
        torch.stack([m02 + m20, m12 + m21, 1 - tr + 2 * m22, m10 - m01], -1),
        torch.stack([m21 - m12, m02 - m20, m10 - m01, 1 + tr], -1),
    ], dim=1)
    quat = cand.gather(1, choice.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)
    quat = quat / torch.norm(quat, dim=1)[:, None]

    quat = torch.where(quat[:, 3:] < 0, -quat, quat)  # w >= 0: angle in [0, pi]
    half_angle = torch.atan2(torch.norm(quat[:, :3], dim=1), quat[:, 3])
    angle = 2 * half_angle
    scale = torch.where(
        torch.abs(angle) <= 1e-3,
        2 + angle ** 2 / 12 + 7 * angle ** 4 / 2880,
        angle / torch.sin(half_angle),  # 0/0 at angle 0, but that row takes the series
    )
    return (scale[:, None] * quat[:, :3]).reshape(R.shape[:-2] + (3,))


def _pose_residual_and_jacobian(state, frames, world_from_ee, col_mask, is_left):
    """World-frame residual ``[Δpos; rotvec(R* R^T)]`` ``[N, 6]`` and the
    column-masked tool Jacobian ``[N, 6, dof]``, each row on its own arm's
    tool frame (``frames``, ``is_left`` and ``col_mask`` from :func:`_setup`)."""
    N = world_from_ee.shape[0]
    (left, left_idx), (right, right_idx) = frames
    pose_l = state.tool_poses.get_link_pose(left)
    pose_r = state.tool_poses.get_link_pose(right)
    row = is_left.view(N, 1)
    pos = torch.where(row, pose_l.position.reshape(N, 3), pose_r.position.reshape(N, 3))
    quat = torch.where(row, pose_l.quaternion.reshape(N, 4), pose_r.quaternion.reshape(N, 4))
    # cuRobo quaternions are wxyz; roma expects xyzw.
    rot = roma.unitquat_to_rotmat(quat.roll(-1, dims=-1))
    r_pos = world_from_ee[:, :3, 3] - pos
    r_rot = _rotmat_to_rotvec(world_from_ee[:, :3, :3] @ rot.transpose(-1, -2))
    # Rows 0-2 linear, 3-5 angular, world frame (finite-diff verified;
    # see test_gn_refresh.py).
    jacs = state.tool_jacobians[:, 0]
    jac = torch.where(is_left.view(N, 1, 1), jacs[:, left_idx], jacs[:, right_idx])
    return torch.cat([r_pos, r_rot], dim=-1), jac * col_mask.unsqueeze(1)


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
    kin, frames, joint_names, q, lower, upper, col_mask, is_left = _setup(world, q_full, (arm,))
    eye6 = GN_DAMPING * torch.eye(6, device=q.device, dtype=q.dtype)

    with torch.no_grad():
        for _ in range(iters):
            js = JointState.from_position(q, joint_names=joint_names)
            state = kin.compute_kinematics(js)
            r, jac = _pose_residual_and_jacobian(state, frames, world_from_ee, col_mask, is_left)
            jjt = jac @ jac.transpose(-1, -2) + eye6                # [B, 6, 6]
            y = torch.linalg.solve_ex(jjt, r.unsqueeze(-1))[0]
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
    ja_pinv = ja.transpose(-1, -2) @ torch.linalg.inv_ex(gram)[0]   # [B, dof, 2]
    dq1 = (ja_pinv @ ((com_target - com) * a).unsqueeze(-1)).squeeze(-1)
    eye_dof = torch.eye(jac.shape[-1], device=jac.device, dtype=jac.dtype)
    jn = jac @ (eye_dof - ja_pinv @ ja)                             # [B, 6, dof]
    eye6 = torch.eye(6, device=jac.device, dtype=jac.dtype)
    rhs = r - (jac @ dq1.unsqueeze(-1)).squeeze(-1)
    y = torch.linalg.solve_ex(jn @ jn.transpose(-1, -2) + damping * eye6, rhs.unsqueeze(-1))[0]
    dq2 = (jn.transpose(-1, -2) @ y).squeeze(-1)
    return dq1, dq2


def _fraction_that_fits(base, extra):
    """Largest ``f`` in [0, 1] with ``|base + f * extra| <= GN_STEP_CLAMP`` per joint."""
    room = (GN_STEP_CLAMP - base * extra.sign()) / extra.abs().clamp_min(1e-12)
    return room.amin(-1, keepdim=True).clamp(0.0, 1.0)


def _spd_solve(M, b):
    """``M^-1 b`` for a batch of SPD ``M``, by Cholesky and two triangular
    solves: batched 21x21 f64 LU dispatches to MAGMA, which syncs."""
    L = torch.linalg.cholesky_ex(M)[0]
    y = torch.linalg.solve_triangular(L, b, upper=False)
    return torch.linalg.solve_triangular(L.mT, y, upper=True)


def _margin_step(q, step, margin_err, jac, jac_com, com, locked, inner, free, lower, upper):
    """Level 3: move joints with nonzero ``margin_err`` (rad to their margin
    band's edge) by at most ``GN_MARGIN_STEP``, on top of the levels 1-2
    ``step``, without moving the pose or the ``locked`` CoM axes to first
    order. Its task Jacobian is a joint selector, so its projected Jacobian is
    the selected rows of the null-space projector.

    Active sets, re-solved three times like levels 1-2: a CoM axis this step
    would push past ``inner`` is locked, and a joint it would push past a
    hard limit is frozen (joints frozen by levels 1-2 stay frozen). The
    rounds always run in full: a round whose sets didn't change recomputes
    the same step, and stopping early would need a host read.
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
        proj = eye_dof - a.transpose(-1, -2) @ torch.linalg.solve_ex(a @ a.transpose(-1, -2) + 1e-12 * eye8, a)[0]
        sel = (margin_err * free != 0).to(f64)
        m = sel.unsqueeze(-1) * proj                                # [B, dof, dof]
        rhs = sel * (margin_err - step).to(f64)
        # m m^T + damping I is SPD.
        y = _spd_solve(m @ m.transpose(-1, -2) + GN_MARGIN_DAMPING * eye_dof, rhs.unsqueeze(-1))
        dq3 = (m.transpose(-1, -2) @ y).squeeze(-1).to(q.dtype)
        s3 = _fraction_that_fits(step, dq3) * dq3
        s3 = s3 * (GN_MARGIN_STEP / s3.abs().amax(-1, keepdim=True).clamp_min(GN_MARGIN_STEP))
        q_lin = q + step + s3
        crossed = (((q_lin > upper) & (s3 > 0)) | ((q_lin < lower) & (s3 < 0))) & free.bool()
        com_lin = com + (jac_com @ (step + s3).unsqueeze(-1)).squeeze(-1)
        pushed = ~locked & (com_lin.abs() > inner) & (com_lin.sign() * (jac_com @ s3.unsqueeze(-1)).squeeze(-1) > 0)
        free = free * ~crossed
        locked = locked | pushed
    return s3


def _balanced_step(q, r, jac, jac_com, com, lower, upper, band_lo, band_hi, col_mask):
    """One balance-first step ``[N, dof]`` (levels 1-3, see
    :func:`gn_project_balanced`), which the caller adds and clamps to the
    joint limits. ``jac`` and ``jac_com`` arrive column-masked by
    ``col_mask [N, dof]``, each row's cspace pin. Every op is row-wise, so a
    row's step doesn't depend on which deps share the call."""
    from cutamp.com_polygon_cost import _half_extents

    half = _half_extents(q.device, q.dtype)
    inner = half - GN_COM_MARGIN
    damping = (GN_DAMPING + (r * r).sum(-1)).view(-1, 1, 1)

    # Two active sets, re-solved three times: CoM axes outside the rectangle
    # or that the step's linearization would push out (driven to ``inner``),
    # and joints at a limit that the step pushes further (columns frozen, as
    # in clamping-loop / SNS IK). Without the second, the final limit clamp
    # discards the CoM-correcting part of the step and the pose part walks
    # the CoM out. The rounds always run in full: a round whose sets didn't
    # change recomputes the same step, and stopping early needs a host read.
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
        com_target = torch.where(pushed_out, com_lin.sign() * inner, com_target)
        active = active | pushed_out
        free = free * ~blocked

    # Trust region in priority order: scale dq1 into the box, then add
    # the largest fraction of dq2 that still fits; level 3 takes what
    # room is left.
    step = _fraction_that_fits(torch.zeros_like(dq1), dq1) * dq1
    step = step + _fraction_that_fits(step, dq2) * dq2
    margin_err = ((band_lo - q).clamp_min(0) - (q - band_hi).clamp_min(0)) * col_mask
    return step + _margin_step(q, step, margin_err, jac, jac_com, com, active, inner, free, lower, upper)


def gn_project_balanced(
    world,
    q_full: torch.Tensor,
    world_from_ee: torch.Tensor,
    arms: str | Sequence[str],
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

    Otherwise the contract is :func:`gn_project`'s: convergence is not
    guaranteed; callers measure the achieved error.

    Args:
        world: TAMPWorld (uses ``kinematics_jac``, the Jacobian-enabled FK).
        q_full: ``[N, dof]`` full-cspace joint positions (not modified).
        world_from_ee: ``[N, 4, 4]`` target tool poses.
        arms: one arm ("left" / "right") per IK dep, so a refresh projects
            all its deps in one call. Rows are dep-major blocks of
            ``N // len(arms)`` rows; block ``i`` targets ``arms[i]``'s tool
            frame with the other arm and the virtual base locked. A bare
            ``str`` is one dep. ``ValueError`` if ``N`` doesn't split evenly.

    Returns:
        ``[N, dof]`` projected joint positions, clamped to joint limits.
    """
    from cutamp.com_polygon_cost import _project_com_to_base

    arms = (arms,) if isinstance(arms, str) else tuple(arms)
    N = q_full.shape[0]
    if not arms or N % len(arms):
        raise ValueError(f"{N} rows don't split into {len(arms)} equal dep blocks")
    kin, frames, joint_names, q, lower, upper, col_mask, is_left = _setup(world, q_full, arms)
    margin_lower, margin_upper = _limit_margins(q.device, q.dtype)
    band_lo = lower + margin_lower
    band_hi = upper - margin_upper

    for _ in range(iters):
        q_g = q.detach().requires_grad_(True)
        with torch.enable_grad():
            state = kin.compute_kinematics(JointState.from_position(q_g, joint_names=joint_names))
            with torch.no_grad():  # base joints are pinned, so the base pose is constant
                base_T = (
                    state.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True)
                    .get_matrix().reshape(N, 4, 4)
                )
            com_g = _project_com_to_base(state.robot_com[..., :3].reshape(N, 3), base_T)
            # Clone: cuRobo's backward writes into a reused buffer, so without
            # it the second grad call overwrites the x row with the y row.
            jcx = torch.autograd.grad(com_g[:, 0].sum(), q_g, retain_graph=True)[0].clone()
            jcy = torch.autograd.grad(com_g[:, 1].sum(), q_g)[0]
        with torch.no_grad():
            jac_com = torch.stack([jcx, jcy], dim=1) * col_mask.unsqueeze(1)    # [N, 2, dof]
            com = com_g.detach()
            r, jac = _pose_residual_and_jacobian(state, frames, world_from_ee, col_mask, is_left)
            step = _balanced_step(q, r, jac, jac_com, com, lower, upper, band_lo, band_hi, col_mask)
            q = (q + step).clamp(lower, upper)

    return q.detach()
