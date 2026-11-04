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
of a refresh's deps in one call and makes no host syncs (the Adam loop is
host-launch-bound), so its solves are the unchecked ``*_ex`` kernels.

What this deliberately does NOT do (the full cuRobo solver covers these at
particle init, at the final pre-check refresh, and in pose-only mode on the
per-refresh fallback):

- no self/world-collision vetting inside the update (the rollout's hard
  constraints still score every conf every step, so violations surface in
  the satisfying mask rather than silently passing),
- no CoM residual in ``gn_project`` — its pose-only-mode CALLER must AND
  ``compute_com_polygon_mask`` into its merge gate so out-of-polygon rows
  route to the CoM-aware cuRobo fallback (a pose-only gate demonstrably
  locks in CoM decay: GN keeps pose converged while the conf drifts out of
  the support polygon, and nothing else re-centers covered Confs during
  Adam). An in-GN CoM residual row was A/B-rejected 2026-08-04 (null
  runtime win, quality parity), but that version's CoM Jacobian had its x
  row overwritten by the y row (see the clone in ``gn_project_balanced``),
  so the rejection says nothing about a correct CoM row,
- no basin hopping (strictly local; in pose-only mode, rows that don't
  converge to checker tolerance are handed to the cuRobo fallback by the
  caller).

``gn_project_balanced`` is the balance-first variant behind
``config.reik_balance_first``: CoM-in-polygon is a hard first level, the
pose is least squares inside it, and joint-limit margins use what is left.
On the real legs, whose base columns are free, level 1 also holds both soles
on the stance (12 rows, design 7.4), and ``gn_gate`` checks them.
"""

from functools import lru_cache
from typing import Sequence, Tuple

import roma
import torch

from curobo.types import JointState, Pose

# ponytail: fixed schedule tuned for mm-scale drift; revisit only if the
# fallback-rate or gn_fail instrumentation shows a fat non-converged band.
GN_ITERS = 3
# Levenberg damping on JJ^T. Units mix metres/radians; at mm/mrad residuals
# the conditioning effect dominates and the bias is negligible.
GN_DAMPING = 1e-4
# Per-iteration |dq_i| cap (rad or metre for the virtual base — base is
# masked anyway). Acts as a crude trust region: measured on a 0.45 rad lean
# (18 cm pose error), UNclamped steps overshoot to 45+ cm while clamped
# descent converges linearly at ~0.7x error/iter. mm-scale refresh drift
# never saturates it, so warm-start convergence stays effectively quadratic;
# corrections big enough to ride the clamp belong to cuRobo or a later refresh.
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


@lru_cache(maxsize=None)
def _row_layout(model, arms: tuple[str, ...], n: int, dof: int, device, dtype):
    """``(col_mask [n, dof], is_left [n])`` for ``n`` rows in ``len(arms)`` equal
    dep-major blocks, block ``i`` on ``arms[i]``: the columns GN may move in
    ``model``'s cspace (the inactive arm stays frozen, so that hand rides with
    the trunk, known-bugs B15, and so does the base where ``model.lock_base``)
    and each row's tool frame. Cached, since building it is a host-to-device
    copy; never mutate or evict it: a captured CUDA graph keeps reading its
    addresses."""
    if not arms or n % len(arms) or not set(arms) <= {"left", "right"}:
        raise ValueError(f"{n} rows over arms {arms}: need equal dep blocks, each 'left' or 'right'")
    col_mask = torch.ones(len(arms), dof, dtype=dtype)
    if model.lock_base:
        col_mask[:, model.base] = 0.0
    for i, arm in enumerate(arms):
        col_mask[i, model.right_arm if arm == "left" else model.left_arm] = 0.0
    is_left = torch.tensor([arm == "left" for arm in arms])
    return (
        col_mask.repeat_interleave(n // len(arms), dim=0).to(device),
        is_left.repeat_interleave(n // len(arms)).to(device),
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
    col_mask, is_left = _row_layout(world.model, arms, q.shape[0], q.shape[1], q.device, q.dtype)
    return kin, frames, joint_names, q, lower, upper, col_mask, is_left


def _rotmat_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """``roma.rotmat_to_rotvec`` (roma 1.5.4), bit for bit, without the host syncs
    of its ``nonzero`` row splits: every candidate is built, then gathered."""
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


def _row_tool_pose(state, left: str, right: str, is_left):
    """Each row's own arm's tool pose ``(position [N, 3], quaternion [N, 4]
    wxyz)``: frame ``left`` where ``is_left``, else ``right``."""
    N = is_left.shape[0]
    pose_l = state.tool_poses.get_link_pose(left)
    pose_r = state.tool_poses.get_link_pose(right)
    row = is_left.view(N, 1)
    return (
        torch.where(row, pose_l.position.reshape(N, 3), pose_r.position.reshape(N, 3)),
        torch.where(row, pose_l.quaternion.reshape(N, 4), pose_r.quaternion.reshape(N, 4)),
    )


def _pose_residual_and_jacobian(state, frames, world_from_ee, col_mask, is_left):
    """World-frame residual ``[Δpos; rotvec(R* R^T)]`` ``[N, 6]`` and the
    column-masked tool Jacobian ``[N, 6, dof]``, each row on its own arm's frame."""
    (left, left_idx), (right, right_idx) = frames
    pos, quat = _row_tool_pose(state, left, right, is_left)
    # cuRobo quaternions are wxyz; roma expects xyzw.
    rot = roma.unitquat_to_rotmat(quat.roll(-1, dims=-1))
    r_pos = world_from_ee[:, :3, 3] - pos
    r_rot = _rotmat_to_rotvec(world_from_ee[:, :3, :3] @ rot.transpose(-1, -2))
    return torch.cat([r_pos, r_rot], dim=-1), _row_tool_jacobian(state, frames, col_mask, is_left)


def _row_tool_jacobian(state, frames, col_mask, is_left):
    """Each row's own arm's column-masked tool Jacobian ``[N, 6, dof]``: rows 0-2
    linear, 3-5 angular, world frame (finite-diff verified; see test_gn_refresh.py)."""
    (_, left_idx), (_, right_idx) = frames
    jacs = state.tool_jacobians[:, 0]
    jac = torch.where(is_left.view(-1, 1, 1), jacs[:, left_idx], jacs[:, right_idx])
    return jac * col_mask.unsqueeze(1)


def _sole_residual_and_jacobian(state, sole_idx: Tuple[int, int], goals: torch.Tensor, col_mask: torch.Tensor):
    """Both soles' world-frame residual ``[N, 12]`` against ``goals`` ``[2, 4, 4]``, rows [left Δpos, left
    rotvec, right Δpos, right rotvec] (:func:`_pose_residual_and_jacobian`'s metric), and their column-masked
    Jacobian ``[N, 12, dof]``. ``sole_idx`` holds the soles' indices among the tool frames."""
    N = col_mask.shape[0]
    poses = state.tool_poses
    pos = torch.stack([poses.position[:, 0, i] for i in sole_idx], dim=1)        # [N, 2, 3]
    quat = torch.stack([poses.quaternion[:, 0, i] for i in sole_idx], dim=1)     # [N, 2, 4], wxyz
    rot = roma.unitquat_to_rotmat(quat.roll(-1, dims=-1))
    r_pos = goals[:, :3, 3] - pos
    r_rot = _rotmat_to_rotvec(goals[:, :3, :3] @ rot.transpose(-1, -2))
    jacs = state.tool_jacobians[:, 0]
    jac = torch.stack([jacs[:, i] for i in sole_idx], dim=1).reshape(N, 12, -1)
    return torch.cat([r_pos, r_rot], dim=-1).reshape(N, 12), jac * col_mask.unsqueeze(1)


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
        arm: "left" / "right" — the OTHER arm and the virtual base are locked.
            (cuRobo IK instead holds the other hand in the world; known-bugs B15.)

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


def _prioritized_step(jac_com, com, active, com_target, jac, r, damping, jac_sole=None, r_sole=None):
    """One two-level step (Siciliano-Slotine form). Level 1 moves the active
    CoM axes to ``com_target`` with minimum norm and, given the sole rows
    ``jac_sole`` ``[B, 12, dof]`` and ``r_sole`` ``[B, 12]``, closes the soles'
    residual, whose rows are always active; level 2 is damped GN on the pose
    residual inside the null space of level 1. Inactive CoM rows are zeroed,
    so they drop out of both levels.

    With the sole rows both levels run in float64: cuRobo turns TF32 matmuls
    on, and the 14-row Gram (condition number ~1e3) amplified their ~1e-3
    relative error into level 2's null space. On 64 balanced blocks_t1 pick
    confs reaching 3 cm, float32 left 3 with a sole 1.5-12 mm off the stance
    after a call, float64 none over 0.35 mm."""
    dtype = jac.dtype
    a = active.to(dtype)
    ja = jac_com * a.unsqueeze(-1)                                  # [B, 2, dof]
    t1 = (com_target - com) * a
    if jac_sole is not None:
        f64 = torch.float64
        ja = torch.cat([ja, jac_sole], dim=1).to(f64)               # [B, 14, dof]
        t1 = torch.cat([t1, r_sole], dim=1).to(f64)
        jac, r, damping = jac.to(f64), r.to(f64), damping.to(f64)
    # Batched LU stays in cuBLAS's sync-free kernels up to 16 x 16, so these inverses capture.
    gram = ja @ ja.transpose(-1, -2) + 1e-6 * torch.eye(ja.shape[1], device=jac.device, dtype=jac.dtype)
    ja_pinv = ja.transpose(-1, -2) @ torch.linalg.inv_ex(gram)[0]   # [B, dof, 2 | 14]
    dq1 = (ja_pinv @ t1.unsqueeze(-1)).squeeze(-1)
    eye_dof = torch.eye(jac.shape[-1], device=jac.device, dtype=jac.dtype)
    jn = jac @ (eye_dof - ja_pinv @ ja)                             # [B, 6, dof]
    eye6 = torch.eye(6, device=jac.device, dtype=jac.dtype)
    rhs = r - (jac @ dq1.unsqueeze(-1)).squeeze(-1)
    y = torch.linalg.solve_ex(jn @ jn.transpose(-1, -2) + damping * eye6, rhs.unsqueeze(-1))[0]
    dq2 = (jn.transpose(-1, -2) @ y).squeeze(-1)
    return dq1.to(dtype), dq2.to(dtype)


def _fraction_that_fits(base, extra):
    """Largest ``f`` in [0, 1] with ``|base + f * extra| <= GN_STEP_CLAMP`` per joint."""
    room = (GN_STEP_CLAMP - base * extra.sign()) / extra.abs().clamp_min(1e-12)
    return room.amin(-1, keepdim=True).clamp(0.0, 1.0)


def _fraction_in_limits(pos, extra, lower, upper):
    """Largest ``f`` in [0, 1] with ``lower <= pos + f * extra <= upper`` on every joint that ``extra`` moves,
    ``[B, 1]``: 0 where it pushes a joint already at a limit further."""
    room = torch.where(extra > 0, upper - pos, pos - lower).clamp_min(0.0)
    frac = torch.where(extra == 0, 1.0, room / extra.abs().clamp_min(1e-12))
    return frac.amin(-1, keepdim=True).clamp(0.0, 1.0)


def _scaled_step(q, dq1, dq2, lower, upper, limits: bool):
    """Levels 1-2's step, trust region in priority order: ``dq1`` scaled into the ``GN_STEP_CLAMP`` box, plus the
    largest fraction of ``dq2`` that still fits. With ``limits`` (the sole rows), each fraction also keeps every
    joint inside [``lower``, ``upper``]: the caller's final clamp would cut a crossing joint's motion, and that cut
    moves the soles by the joint's column of the sole Jacobian. Scaling a level keeps it in level 1's null space;
    cutting one joint's motion does not."""
    f1 = _fraction_that_fits(torch.zeros_like(dq1), dq1)
    if limits:
        f1 = torch.minimum(f1, _fraction_in_limits(q, dq1, lower, upper))
    step = f1 * dq1
    f2 = _fraction_that_fits(step, dq2)
    if limits:
        f2 = torch.minimum(f2, _fraction_in_limits(q + step, dq2, lower, upper))
    return step + f2 * dq2


def _crossing_joints(q, dq1, dq2, lower, upper):
    """``[B, dof]`` bool: the joints that levels 1-2's box-scaled step (as :func:`_balanced_step` scales it) would
    carry past a limit, either level pushing them outward."""
    s1 = _fraction_that_fits(torch.zeros_like(dq1), dq1) * dq1
    s2 = _fraction_that_fits(s1, dq2) * dq2
    p1, p2 = q + s1, q + s1 + s2
    return ((p1 > upper) & (s1 > 0)) | ((p1 < lower) & (s1 < 0)) | ((p2 > upper) & (s2 > 0)) | ((p2 < lower) & (s2 < 0))


def _sole_closing_step(state, frames, sole_idx, goals, q, lower, upper, col_mask, is_left):
    """The step that ends a call on a model with soles, from a pose-only FK ``state`` at ``q``: the 12 sole rows
    against ``goals`` as level 1, and the hand rows, held where they are, as level 2 in their null space, so the
    hand does not ride the correction. Each iteration's level 1 holds the soles only to first order, so a
    saturated last step leaves a second-order drift that nothing else in the call would correct.

    No CoM rows (so no autograd) and no margin level. Joints at a limit are frozen, and one re-solve freezes the
    joints the first solve would carry past one; then the step is scaled into the box and the limits as levels
    1-2 are (:func:`_scaled_step`). Without the re-solve, a joint near its limit capped the correction on some
    out-of-reach rows, leaving 2.9 mm. Float64, as :func:`_prioritized_step` with soles, but it projects the hand
    rows off level 1 as ``h - (h a^+) a``, without that function's ``[N, dof, dof]`` projector: 0.6 ms against
    0.95 ms per solve at 2048 rows, captured."""
    f64 = torch.float64
    r_sole, jac_sole = _sole_residual_and_jacobian(state, sole_idx, goals, col_mask)
    a0, h0 = jac_sole.to(f64), _row_tool_jacobian(state, frames, col_mask, is_left).to(f64)
    t1 = r_sole.to(f64).unsqueeze(-1)
    eye12 = 1e-6 * torch.eye(12, device=q.device, dtype=f64)
    eye6 = GN_DAMPING * torch.eye(6, device=q.device, dtype=f64)
    free = col_mask * ((q < upper - 1e-5) & (q > lower + 1e-5))
    for _ in range(2):
        fc = free.to(f64).unsqueeze(1)
        a, h = a0 * fc, h0 * fc
        a_pinv = a.mT @ torch.linalg.inv_ex(a @ a.mT + eye12)[0]          # [N, dof, 12]
        dq1 = a_pinv @ t1
        jn = h - (h @ a_pinv) @ a                                         # [N, 6, dof]
        dq2 = jn.mT @ torch.linalg.solve_ex(jn @ jn.mT + eye6, -(h @ dq1))[0]
        dq1, dq2 = dq1.squeeze(-1).to(q.dtype), dq2.squeeze(-1).to(q.dtype)
        free = free * ~_crossing_joints(q, dq1, dq2, lower, upper)
    return _scaled_step(q, dq1, dq2, lower, upper, limits=True)


def _spd_solve(M, b):
    """``M^-1 b`` for a batch of SPD ``M``, by Cholesky and two triangular
    solves: batched 21x21 f64 LU dispatches to MAGMA, which syncs."""
    L = torch.linalg.cholesky_ex(M)[0]
    y = torch.linalg.solve_triangular(L, b, upper=False)
    return torch.linalg.solve_triangular(L.mT, y, upper=True)


def _margin_step(q, step, margin_err, jac, jac_com, com, locked, inner, free, lower, upper, jac_sole=None):
    """Level 3: move joints with nonzero ``margin_err`` (rad to their margin
    band's edge) by at most ``GN_MARGIN_STEP``, on top of the levels 1-2
    ``step``, without moving the pose, the ``locked`` CoM axes or, given
    ``jac_sole``, the soles to first order. Its task Jacobian is a joint
    selector, so its projected Jacobian is the selected rows of the
    null-space projector.

    Active sets, re-solved three times like levels 1-2: a CoM axis this step
    would push past ``inner`` is locked, and a joint it would push past a
    hard limit is frozen (joints frozen by levels 1-2 stay frozen).
    Float64 because the explicit projector is ill-conditioned (cond(A A^T)
    ~5e5 on place confs): in float32 it leaked ~5e-3 of each step into the
    pose, the damped solve amplified that ~15x, and ~80/512 place confs lost
    their pose per refresh. With the sole rows the projector's Gram is
    20 x 20, past the 16 x 16 up to which batched LU stays sync-free (beyond
    it the solve syncs and fails a capture), so it takes :func:`_spd_solve`.
    """
    f64 = torch.float64
    eye_dof = torch.eye(q.shape[-1], device=q.device, dtype=f64)
    soles = () if jac_sole is None else (jac_sole,)
    eye_a = torch.eye(jac_com.shape[1] + sum(s.shape[1] for s in soles) + jac.shape[1], device=q.device, dtype=f64)
    for _ in range(3):
        a = (torch.cat([jac_com * locked.to(q.dtype).unsqueeze(-1), *soles, jac], dim=1) * free.unsqueeze(1)).to(f64)
        gram = a @ a.transpose(-1, -2) + 1e-12 * eye_a
        proj = eye_dof - a.transpose(-1, -2) @ (_spd_solve(gram, a) if soles else torch.linalg.solve_ex(gram, a)[0])
        sel = (margin_err * free != 0).to(f64)
        m = sel.unsqueeze(-1) * proj                                # [B, dof, dof]
        rhs = sel * (margin_err - step).to(f64)
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
    if soles:  # the last round's crossings get no re-solve: scale it inside the limits, as levels 1-2 are
        s3 = _fraction_in_limits(q + step, s3, lower, upper) * s3
    return s3


def _balanced_step(q, r, jac, jac_com, com, lower, upper, band_lo, band_hi, col_mask, half=None,
                   jac_sole=None, r_sole=None):
    """One balance-first step ``[N, dof]``, levels 1-3 of :func:`gn_project_balanced`,
    on ``jac``, ``jac_com`` and ``jac_sole`` already masked by ``col_mask``. Every
    op is row-wise, so a row's step doesn't depend on which deps share the call.

    ``com`` is the CoM about the support's centre, in its frame, and ``half``
    the rectangle level 1 holds it in: a support's ``keyframe_half``, the
    simplified model's when ``None``. ``jac_sole`` and ``r_sole`` are the real
    legs' 12 sole rows (:func:`_sole_residual_and_jacobian`): level 1 closes
    them beside the active CoM axes, and levels 2-3 move in their null space."""
    from cutamp.com_polygon_cost import _half_extents

    if half is None:
        from cutamp.robots.t1 import SIMPLIFIED
        half = SIMPLIFIED.support.keyframe_half
    half = _half_extents(half, q.device, q.dtype)
    inner = half - GN_COM_MARGIN
    damping = (GN_DAMPING + (r * r).sum(-1)).view(-1, 1, 1)

    # Two active sets, re-solved three times: CoM axes outside the rectangle
    # or that the step's linearization would push out (driven to ``inner``),
    # and joints at a limit that the step pushes further (columns frozen, as
    # in clamping-loop / SNS IK). Without the second, the final limit clamp
    # discards the CoM-correcting part of the step and the pose part walks
    # the CoM out. The rounds always run in full: a round whose sets didn't
    # change recomputes the same step, and stopping early needs a host read.
    # With the sole rows, a joint the box-scaled step would carry past a limit
    # is frozen too, and the step is then scaled inside the limits: scaling
    # alone stalls a level whenever its step pushes a joint at a limit (on
    # blocks_t1's pick confs it left the CoM outside on 16 of 46 rows).
    active = com.abs() > half
    com_target = com.sign() * inner
    free = torch.ones_like(q)
    for _ in range(3):
        fc = free.unsqueeze(1)
        dq1, dq2 = _prioritized_step(jac_com * fc, com, active, com_target, jac * fc, r, damping,
                                     None if jac_sole is None else jac_sole * fc, r_sole)
        dq = dq1 + dq2
        com_lin = com + (jac_com @ dq.unsqueeze(-1)).squeeze(-1)
        pushed_out = ~active & (com_lin.abs() > half)
        blocked = ((q >= upper - 1e-5) & (dq > 0)) | ((q <= lower + 1e-5) & (dq < 0))
        if jac_sole is not None:
            blocked |= _crossing_joints(q, dq1, dq2, lower, upper)
        blocked &= free.bool()
        com_target = torch.where(pushed_out, com_lin.sign() * inner, com_target)
        active = active | pushed_out
        free = free * ~blocked

    # Level 3 takes what room levels 1-2 leave.
    step = _scaled_step(q, dq1, dq2, lower, upper, limits=jac_sole is not None)
    margin_err = ((band_lo - q).clamp_min(0) - (q - band_hi).clamp_min(0)) * col_mask
    return step + _margin_step(q, step, margin_err, jac, jac_com, com, active, inner, free, lower, upper, jac_sole)


def gn_project_balanced(
    world,
    q_full: torch.Tensor,
    world_from_ee: torch.Tensor,
    arms: str | Sequence[str],
    iters: int = GN_ITERS,
    balance: bool = True,
) -> torch.Tensor:
    """Balance-first variant of :func:`gn_project` (``config.reik_balance_first``).

    Each step is a three-level hierarchy (Kanoun et al. T-RO 2011; Escande et
    al. IJRR 2014): (1) keep the CoM, in the support frame of
    ``world.model.support``, inside its keyframe rectangle shrunk by
    ``GN_COM_MARGIN``, and on the real legs both soles on the stance
    (``world.stance_soles``), since their free base and leg columns would
    otherwise slide them; (2) least squares on the pose residual in
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

    On a model with soles, every level's step also stays inside the joint
    limits (crossing joints frozen, then each level scaled), so the final
    clamp cuts nothing that would move the soles, and the call ends with
    :func:`_sole_closing_step`, which closes the soles' residual that the last
    iteration leaves (second-order drift of a saturated step) without moving
    the hand. Together they hold the soles within 0.1 mm after each call on
    blocks_t1's pick confs, out-of-reach targets included.

    Otherwise the contract is :func:`gn_project`'s, except that ``arms`` holds
    one arm per IK dep (a bare ``str`` is one): rows are ``len(arms)`` equal
    dep-major blocks, block ``i`` on ``arms[i]``'s tool frame and frozen columns.

    ``balance=False`` (the no-CoM ablation, ``config.enable_com_aware_ik``
    off) drops level 1's CoM rows: a zero CoM and CoM Jacobian leave both CoM
    active sets empty, so the step is levels 2-3 alone (beside the sole rows,
    which a model with soles keeps), with no CoM autograd.
    """
    from cutamp.com_polygon_cost import _project_com_to_base
    from cutamp.cost_function import _device_const

    N = q_full.shape[0]
    arms = (arms,) if isinstance(arms, str) else tuple(arms)
    support = world.model.support
    kin, frames, joint_names, q, lower, upper, col_mask, is_left = _setup(world, q_full, arms)
    band_lo = lower + _device_const(world.model.margin_lower, q.device, q.dtype)
    band_hi = upper - _device_const(world.model.margin_upper, q.device, q.dtype)
    sole_idx = tuple(list(kin.tool_frames).index(frame) for frame in world.model.sole_frames)  # () when simplified

    for _ in range(iters):
        if balance:
            q_g = q.detach().requires_grad_(True)
            with torch.enable_grad():
                state = kin.compute_kinematics(JointState.from_position(q_g, joint_names=joint_names))
                # The support frame is constant: the simplified base is pinned, and the real legs' stance hangs off
                # the world, not off their free base. Held under no_grad, so CoM gradients never reach it (design 7.4).
                with torch.no_grad():
                    base_T = (
                        state.tool_poses.get_link_pose(support.link, make_contiguous=True)
                        .get_matrix().reshape(N, 4, 4)
                    )
                com_g = _project_com_to_base(state.robot_com[..., :3].reshape(N, 3), base_T, support.center)
                # Clone: cuRobo's backward writes into a reused buffer, so without
                # it the second grad call overwrites the x row with the y row.
                jcx = torch.autograd.grad(com_g[:, 0].sum(), q_g, retain_graph=True)[0].clone()
                jcy = torch.autograd.grad(com_g[:, 1].sum(), q_g)[0]
            with torch.no_grad():
                jac_com = torch.stack([jcx, jcy], dim=1) * col_mask.unsqueeze(1)    # [N, 2, dof]
                com = com_g.detach()
        else:
            with torch.no_grad():
                state = kin.compute_kinematics(JointState.from_position(q, joint_names=joint_names))
            com, jac_com = q.new_zeros(N, 2), q.new_zeros(N, 2, q.shape[-1])
        with torch.no_grad():
            r, jac = _pose_residual_and_jacobian(state, frames, world_from_ee, col_mask, is_left)
            r_sole, jac_sole = (
                _sole_residual_and_jacobian(state, sole_idx, world.stance_soles, col_mask) if sole_idx else (None, None)
            )
            step = _balanced_step(q, r, jac, jac_com, com, lower, upper, band_lo, band_hi, col_mask,
                                  half=support.keyframe_half, jac_sole=jac_sole, r_sole=r_sole)
            q = (q + step).clamp(lower, upper)

    if sole_idx:
        with torch.no_grad():
            state = kin.compute_kinematics(JointState.from_position(q, joint_names=joint_names))
            q = (q + _sole_closing_step(state, frames, sole_idx, world.stance_soles, q, lower, upper, col_mask,
                                        is_left)).clamp(lower, upper)
    return q.detach()


def gn_gate(
    world,
    q_full: torch.Tensor,
    world_from_ee: torch.Tensor,
    arms: str | Sequence[str],
    pos_tol: float,
    rot_tol: float,
    kin=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge gate for :func:`gn_project_balanced`'s output (same ``arms``), from
    one FK pass on ``kin`` (default ``world.kinematics``) and with no host syncs.

    Returns ``(pose_ok [N], com_ok [N])``: ``_achieved_pose_errors`` (the
    checker's metric, on each row's own arm) strictly under ``pos_tol`` and
    ``rot_tol``, on the real legs both soles' errors against the stance too,
    and ``compute_com_polygon_mask``. ``q_full`` must be contiguous float32 on
    the kinematics device, as cuRobo's FK requires.
    """
    from cutamp.com_polygon_cost import _fk_state, com_polygon_mask_from_state
    from cutamp.costs import curobo_pose_error

    arms = (arms,) if isinstance(arms, str) else tuple(arms)
    _, is_left = _row_layout(world.model, arms, q_full.shape[0], q_full.shape[1], q_full.device, q_full.dtype)
    state = _fk_state(world, q_full, kin)
    pos, quat = _row_tool_pose(state, world.tool_frame_for_arm("left"), world.tool_frame_for_arm("right"), is_left)
    ee = Pose(position=pos.contiguous(), quaternion=quat.contiguous()).get_matrix()
    pos_err, rot_err = curobo_pose_error(ee, world_from_ee)
    pose_ok = (pos_err < pos_tol) & (rot_err < rot_tol)
    if world.model.sole_frames:
        # A conf whose soles left the stance passes com_ok on a support polygon it no longer stands on (design 7.4).
        soles = torch.stack([state.tool_poses.get_link_pose(frame, make_contiguous=True).get_matrix()
                             for frame in world.model.sole_frames], dim=1)  # [N, 2, 4, 4]
        sole_pos, sole_rot = curobo_pose_error(soles, world.stance_soles)
        pose_ok = pose_ok & ((sole_pos < pos_tol) & (sole_rot < rot_tol)).all(-1)
    return pose_ok, com_polygon_mask_from_state(state, world.model.support)
