# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Batched Gauss-Newton IK, balance first: the planner's only IK.

Each covered Conf (a pick's or a place's) is the IK of a pose target that Adam moves. Damped Gauss-Newton
projects q onto that target's constraint manifold; on the 6-DOF pose residual alone, its step is

    dq = J^T (J J^T + lambda I)^-1 [t* - t(q); rotvec(R* R(q)^T)]

``gn_project_balanced`` puts that residual at level 2 of a four-level hierarchy whose level 1 holds the CoM in
the support rectangle, both soles on the stance (12 rows, design 7.4: the base columns are free), the robot
out of collision with the scene, the objects and itself (one row); level 3, beside the collision row, pulls the
CoM toward the rectangle's centre, and level 4, below it, the Trunk toward upright. ``gn_gate`` checks the soles
too.

Two callers: the particle init (``particle_initialization.gn_ik``: from home and from noisy stance-rooted
starts, iterations that reach balance first, then iterations with the collision row on, since only the conf
the iterates end at must be clear) and the periodic refresh inside the Adam loop (``GN_ITERS`` iterations from
the stored q, the collision row on in every one, captured as G_refresh). A refresh projects all of its deps in
one call and makes no host syncs (the Adam loop is host-launch-bound), so its solves are the unchecked ``*_ex``
kernels. Strictly local: a row that doesn't converge resumes at the next refresh.
"""

from functools import lru_cache
from typing import Optional, Sequence, Tuple

import roma
import torch

from curobo.types import JointState, Pose

# ponytail: fixed schedule tuned for mm-scale drift; revisit only if the
# refresh records' gn_fail counts (opt_metrics["reik_refresh"]) show a fat
# non-converged band.
GN_ITERS = 3
# Levenberg damping on JJ^T. Units mix metres/radians; at mm/mrad residuals
# the conditioning effect dominates and the bias is negligible.
GN_DAMPING = 1e-4
# Per-iteration |dq_i| cap (rad, or metre on the base's translation columns).
# Acts as a crude trust region: measured on a 0.45 rad lean
# (18 cm pose error), UNclamped steps overshoot to 45+ cm while clamped
# descent converges linearly at ~0.7x error/iter. mm-scale refresh drift
# never saturates it, so warm-start convergence stays effectively quadratic;
# corrections big enough to ride the clamp take more iterations (the init's 30) or a later refresh.
GN_STEP_CLAMP = 0.15
# A CoM axis outside the support rectangle (or that a step
# would push out) is driven this far inside the edge, absorbing ~2-4 mm of
# linearization error. Axes already inside are left alone: optimized confs
# cluster within 1 cm of the edge, and a 1 cm activation band knocked 27/240
# of them off their pose on the captured rows.
GN_COM_MARGIN = 0.005
# A conf in collision (summed penetration c > 0, :func:`collision_penetration`) is driven this far past contact,
# as the CoM is driven GN_COM_MARGIN inside its edge: a solver margin that absorbs the linearization error, not
# a clearance a plan must keep. No band ahead of contact: a pick's palm sits 2-3 mm from its own block.
GN_COLL_MARGIN = 0.005


@lru_cache(maxsize=None)
def _self_pairs(kin):
    """``(i [P], j [P], pad [S])``: the sphere pairs cuRobo's self-collision kernel checks on ``kin`` (long
    indices), kept where both spheres are real (the kernel skips a pair with a negative padded radius; a
    disabled sphere's radius is -100), and the per-sphere padding the kernel adds to each radius. FK moves only
    the centres, and objects attach to the motion planner's own kinematics, never to the world's, so the filter
    holds. Cached, since building it reads the host; never evict it: a captured CUDA graph keeps reading it."""
    cfg = kin.get_self_collision_config()
    pad = cfg.sphere_padding
    radius = kin.robot_spheres[:, 3].to(pad.device) + pad
    pairs = cfg.collision_pairs.long()
    pairs = pairs[(radius[pairs] >= 0).all(-1)]
    return pairs[:, 0].contiguous(), pairs[:, 1].contiguous(), pad


def collision_penetration(world, spheres: torch.Tensor, movables: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``[N]``: the summed linear penetration depth (m) of the robot's ``spheres`` ``[N, S, 4]`` (an FK of
    ``world.kinematics`` or one of its sites; differentiable) with the static scene (``world.collision_fn``),
    with ``movables`` ``[N, M, 4]`` (world-frame object spheres, :func:`cutamp.rollout.movable_spheres_fn`)
    and with itself. Zero exactly where the constraint checker's three terms are (``robot_to_world``,
    ``robot_to_movables`` at the default activation 0 and ``self_collision``): the self term sums
    ``relu(r_i + r_j - d)`` over the kernel's pairs, which is positive where the kernel's ``(r_i + r_j)^2 - d^2``
    is, and has a gradient on every penetrating pair, not only the deepest."""
    spheres = spheres.reshape(spheres.shape[0], -1, 4)
    pen = world.collision_fn(spheres.unsqueeze(1).contiguous()).reshape(-1)
    if movables is not None:
        from cutamp.costs import sphere_to_sphere_overlap
        pen = pen + sphere_to_sphere_overlap(spheres, movables, activation_distance=0.0)
    i, j, pad = _self_pairs(world.kinematics)
    c, r = spheres[..., :3], spheres[..., 3] + pad
    d = (c[:, i] - c[:, j]).norm(dim=-1)
    return pen + torch.relu(r[:, i] + r[:, j] - d).sum(-1)


@lru_cache(maxsize=None)
def _row_layout(model, arms: tuple[str, ...], n: int, dof: int, device, dtype):
    """``(col_mask [n, dof], is_left [n])`` for ``n`` rows in ``len(arms)`` equal
    dep-major blocks, block ``i`` on ``arms[i]``: the columns GN may move in
    ``model``'s cspace (the inactive arm stays frozen, so that hand rides with
    the trunk, known-bugs B15) and each row's tool frame. Cached, since building it is a host-to-device
    copy; never mutate or evict it: a captured CUDA graph keeps reading its
    addresses."""
    if not arms or n % len(arms) or not set(arms) <= {"left", "right"}:
        raise ValueError(f"{n} rows over arms {arms}: need equal dep blocks, each 'left' or 'right'")
    col_mask = torch.ones(len(arms), dof, dtype=dtype)
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


def _prioritized_step(jac_com, com, active, com_target, jac, r, damping, jac_sole, r_sole, jac_coll=None,
                      r_coll=None):
    """One two-level step (Siciliano-Slotine form). Level 1 moves the active
    CoM axes to ``com_target`` with minimum norm, closes the soles' residual
    (``jac_sole`` ``[B, 12, dof]`` and ``r_sole`` ``[B, 12]``), whose rows are
    always active, and, given the collision row ``jac_coll`` ``[B, 1, dof]`` and ``r_coll`` ``[B, 1]`` (zeroed by
    the caller where inactive), moves the penetration by ``r_coll``; level 2 is
    damped GN on the pose residual inside the null space of level 1. Inactive
    CoM rows are zeroed, so they drop out of both levels.

    Both levels run in float64: cuRobo turns TF32 matmuls on, and the 14-row
    Gram (condition number ~1e3) amplified their ~1e-3 relative error into
    level 2's null space. On 64 balanced blocks_t1 pick confs reaching 3 cm,
    float32 left 3 with a sole 1.5-12 mm off the stance after a call, float64
    none over 0.35 mm.

    Returns ``(dq1, dq2, ja)``: ``ja`` is level 1's rows ``[B, 14-15, dof]`` (float64), for
    :func:`_centring_step`."""
    dtype = jac.dtype
    a = active.to(dtype)
    ja = jac_com * a.unsqueeze(-1)                                  # [B, 2, dof]
    t1 = (com_target - com) * a
    extra = [(j, t) for j, t in ((jac_sole, r_sole), (jac_coll, r_coll)) if j is not None]
    f64 = torch.float64
    ja = torch.cat([ja, *(j for j, _ in extra)], dim=1).to(f64)      # [B, 14-15, dof]
    t1 = torch.cat([t1, *(t for _, t in extra)], dim=1).to(f64)
    jac, r, damping = jac.to(f64), r.to(f64), damping.to(f64)
    # Batched LU stays in cuBLAS's sync-free kernels up to 16 x 16, so these inverses capture.
    gram = ja @ ja.transpose(-1, -2) + 1e-6 * torch.eye(ja.shape[1], device=jac.device, dtype=jac.dtype)
    ja_pinv = ja.transpose(-1, -2) @ torch.linalg.inv_ex(gram)[0]   # [B, dof, 14-15]
    dq1 = (ja_pinv @ t1.unsqueeze(-1)).squeeze(-1)
    eye_dof = torch.eye(jac.shape[-1], device=jac.device, dtype=jac.dtype)
    jn = jac @ (eye_dof - ja_pinv @ ja)                             # [B, 6, dof]
    eye6 = torch.eye(6, device=jac.device, dtype=jac.dtype)
    rhs = r - (jac @ dq1.unsqueeze(-1)).squeeze(-1)
    y = torch.linalg.solve_ex(jn @ jn.transpose(-1, -2) + damping * eye6, rhs.unsqueeze(-1))[0]
    dq2 = (jn.transpose(-1, -2) @ y).squeeze(-1)
    return dq1.to(dtype), dq2.to(dtype), ja


def _null_space_step(ja, jac, rows, target, extra=None, own_complement=False):
    """``[B, dof]`` float64: damped least squares of ``rows`` ``[B, m, dof]`` toward ``target`` ``[B, m]`` in the null
    space of level 1's rows ``ja`` ``[B, k, dof]``, of the hand rows ``jac`` ``[B, 6, dof]`` and, given, of
    ``extra`` ``[B, e, dof]``, a level's between the hand and this one, every Jacobian on this level's columns: the
    step of GN's levels 3 (:func:`_centring_step`) and 4 (:func:`_upright_step`). The damping grows with the target's
    residual, as level 2's does.

    The null space is the stacked rows ``A = [ja; jac; extra]``'s, ``I - A^T (A A^T + eps I)^-1 A``, by block
    inversion: ``N1 = I - ja^+ ja``, then each further block ``b`` projected off the ones above it (``b N``) and its
    Schur complement ``b N b^T + eps I``, so no inverse is over 16 x 16 and the call captures, and no
    ``[B, dof, dof]`` projector is formed. Taking level 2's ``jn jn^T`` (``jac N1^2 jac^T``) for the hand's complement
    instead is right only where the regularized ``N1`` is idempotent: on blocks_t1's floor picks, where level 1 is
    near-singular (a singular value ~2e-4) and level 3 has no exact freedom left, that step moved the hand up to
    0.4 m per step. Float64, as levels 1-2, with ``eps`` 1e-9: the leak into any row of ``A`` is at most
    ``sqrt(eps) / 2`` per unit step; there it was under 1.5e-5 against 1.3e-3 at levels 1-2's 1e-6, and at 1e-12 the
    16-row inverse lost 1.5e-2 to rounding (2026-11-06).

    With ``own_complement`` the level's own solve is on its Schur complement too, ``rows N rows^T``: its rows then
    move by ``M (M + damping)^-1`` of ``target``, ``M = rows N rows^T``, whose eigenvalues are below 1, so they never
    pass it. Without it (level 3, bitwise its step before level 4) the solve is on ``rows N^2 rows^T``, whose gain
    can pass 1 where ``N`` is not idempotent; no real-legs replay met that (2026-11-06)."""
    f64, eps = torch.float64, 1e-9

    def eye(n):
        return torch.eye(n, device=jac.device, dtype=f64)

    ja, jac, rows, target = ja.to(f64), jac.to(f64), rows.to(f64), target.to(f64)
    rows0 = rows
    ja_pinv = ja.mT @ torch.linalg.inv_ex(ja @ ja.mT + eps * eye(ja.shape[1]))[0]   # [B, dof, k]
    jn = jac - (jac @ ja_pinv) @ ja                                                 # jac N1
    s_inv = torch.linalg.inv_ex(jn @ jac.mT + eps * eye(6))[0]                      # (jac N1 jac^T + eps I)^-1

    def off(x):                                                                     # x N, off levels 1 and 2
        x = x - (x @ ja_pinv) @ ja
        return x - ((x @ jac.mT) @ s_inv) @ jn

    rows = off(rows)
    if extra is not None:
        extra = extra.to(f64)
        en = off(extra)
        rows = rows - ((rows @ extra.mT) @ torch.linalg.inv_ex(en @ extra.mT + eps * eye(extra.shape[1]))[0]) @ en
    damping = (GN_DAMPING + (target * target).sum(-1)).view(-1, 1, 1)
    y = torch.linalg.solve_ex(rows @ (rows0 if own_complement else rows).mT + damping * eye(rows.shape[1]),
                              target.unsqueeze(-1))[0]
    return (rows.mT @ y).squeeze(-1)


def _centring_step(ja, jac, jac_com, com_lin, active, half):
    """Level 3, below :func:`_prioritized_step`'s two: the CoM axes level 1 leaves free (``~active``) toward the
    support's centre from ``com_lin``, the CoM (about the centre) after levels 1-2's step, linear on every
    column, in the null space of level 1's rows ``ja`` ``[B, k, dof]`` and of the hand rows ``jac``
    ``[B, 6, dof]`` (:func:`_null_space_step`), every Jacobian on level 3's columns. Each axis is over ``half``, as
    in the old seed IK's pull-to-center term ``sum((com / half)^2)``.

    Without the level, GN's minimum-norm steps leave the CoM wherever level 1 stops it, at the rectangle's edge:
    GN IK's blocks_t1 first picks sat at a median 0.95 (x) and 0.96 (|y|) of ``half``, with the base slid
    0.12-0.18 m sideways onto one leg, and trajopt's move there from home needed 1.06-1.37 of the
    Left_Ankle_Roll's effort (2026-11-06)."""
    f64 = torch.float64
    w = (~active).to(f64) / half.to(f64)                                             # [B, 2]
    rows = jac_com.to(f64) * w.unsqueeze(-1)                                         # [B, 2, dof]
    return _null_space_step(ja, jac, rows, -com_lin.to(f64) * w).to(jac_com.dtype)


def _upright_step(ja, jac, jac_com, active, half, q3, cols, free):
    """Level 4, below :func:`_centring_step`: the Trunk toward upright, the base's pitch and roll (``cols``, the two
    columns its tilt reads) toward 0 from ``q3`` ``[B, dof]``, the conf after levels 1-3's box-scaled step, in the
    null space of level 1's rows ``ja``, of the hand rows ``jac`` and of level 3's (the free CoM axes over ``half``,
    as :func:`_centring_step` weighs them) (:func:`_null_space_step`), every Jacobian on level 4's columns, ``free``
    ``[B, dof]``. Home's Trunk is upright and the two rows weigh pitch and roll alike, so the level lowers
    ``pitch^2 + roll^2``, the squared tilt to second order, with no weight to set; a frozen pitch or roll column
    drops its row. Its step moves no CoM axis, to first order, and never raises that sum: the rows are coordinates,
    which move exactly as the own-complement solve (gain below 1) and the box take them, though pitch alone can still
    cross 0 by the two rows' coupling.

    Without it, GN's minimum-norm steps keep whatever tilt a conf's start or reach gave it: gn_ik's noisy starts
    tilt the Trunk by up to 50 deg and that tilt survives (start-to-end correlation 0.3-0.8 on the accepted rows),
    and blocks_t1's first places tilt back past 5 deg on 16 of 47 rows even from home (2026-11-06)."""
    f64 = torch.float64
    pitch, roll = cols
    mask = torch.stack([free[:, pitch], free[:, roll]], -1).to(f64)                  # [B, 2]
    rows = torch.zeros(free.shape[0], 2, free.shape[1], device=free.device, dtype=f64)
    rows[:, 0, pitch] = mask[:, 0]
    rows[:, 1, roll] = mask[:, 1]
    target = -torch.stack([q3[:, pitch], q3[:, roll]], -1).to(f64) * mask
    w = (~active).to(f64) / half.to(f64)
    return _null_space_step(ja, jac, rows, target, extra=jac_com.to(f64) * w.unsqueeze(-1),
                            own_complement=True).to(jac.dtype)


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


def _add_scaled(q, step, dq, lower, upper):
    """``step`` plus the largest fraction of ``dq`` that keeps it in the ``GN_STEP_CLAMP`` box and every joint of
    ``q + step`` inside [``lower``, ``upper``]."""
    f = torch.minimum(_fraction_that_fits(step, dq), _fraction_in_limits(q + step, dq, lower, upper))
    return step + f * dq


def _scaled_step(q, dq1, dq2, lower, upper, *below):
    """Levels 1-2's step (and, given, those of the levels ``below`` them, in order), trust region in priority order:
    ``dq1`` scaled into the ``GN_STEP_CLAMP`` box, plus the largest fraction of ``dq2`` that still fits, then of each
    level below. Each fraction also keeps every joint inside [``lower``, ``upper``]: the caller's final clamp would
    cut a crossing joint's motion, and that cut moves the soles by the joint's column of the sole Jacobian. Scaling a
    level keeps it in the null space of the levels above it; cutting one joint's motion does not."""
    f1 = torch.minimum(_fraction_that_fits(torch.zeros_like(dq1), dq1), _fraction_in_limits(q, dq1, lower, upper))
    step = f1 * dq1
    for dq in (dq2, *below):
        step = _add_scaled(q, step, dq, lower, upper)
    return step


def _crossing_joints(q, dq1, dq2, lower, upper):
    """``[B, dof]`` bool: the joints that levels 1-2's box-scaled step (as :func:`_balanced_step` scales it) would
    carry past a limit, either level pushing them outward."""
    s1 = _fraction_that_fits(torch.zeros_like(dq1), dq1) * dq1
    s2 = _fraction_that_fits(s1, dq2) * dq2
    p1, p2 = q + s1, q + s1 + s2
    return ((p1 > upper) & (s1 > 0)) | ((p1 < lower) & (s1 < 0)) | ((p2 > upper) & (s2 > 0)) | ((p2 < lower) & (s2 < 0))


def _lower_level_blocked(q, above, dq, lower, upper):
    """``[B, dof]`` bool: the joints that a lower level's step ``dq`` (level 3's or 4's) pushes outward from where the
    box-scaled steps of the levels ``above`` it (in priority order, as :func:`_crossing_joints` scales levels 1-2's)
    leave them: at a limit already, or carried past one."""
    s = _fraction_that_fits(torch.zeros_like(above[0]), above[0]) * above[0]
    for d in above[1:]:
        s = s + _fraction_that_fits(s, d) * d
    p = q + s
    p_new = p + _fraction_that_fits(s, dq) * dq
    up = ((p >= upper - 1e-5) | (p_new > upper)) & (dq > 0)
    return up | (((p <= lower + 1e-5) | (p_new < lower)) & (dq < 0))


def _sole_closing_step(state, frames, sole_idx, goals, q, lower, upper, col_mask, is_left):
    """The step that ends a call, from a pose-only FK ``state`` at ``q``: the 12 sole rows
    against ``goals`` as level 1, and the hand rows, held where they are, as level 2 in their null space, so the
    hand does not ride the correction. Each iteration's level 1 holds the soles only to first order, so a
    saturated last step leaves a second-order drift that nothing else in the call would correct.

    No CoM rows (so no autograd). Joints at a limit are frozen, and one re-solve freezes the
    joints the first solve would carry past one; then the step is scaled into the box and the limits as levels
    1-2 are (:func:`_scaled_step`). Without the re-solve, a joint near its limit capped the correction on some
    out-of-reach rows, leaving 2.9 mm. Float64, as :func:`_prioritized_step`, but it projects the hand
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
    return _scaled_step(q, dq1, dq2, lower, upper)


def _balanced_step(q, r, jac, jac_com, com, lower, upper, col_mask, half, jac_sole, r_sole, coll=None, centre=False,
                   upright=None):
    """``(step, step3, step12)``, each ``[N, dof]``: one balance-first step, levels 1-4 of
    :func:`gn_project_balanced` (level 3 only with ``centre``, which needs ``coll``, and level 4 only with level 3
    and ``upright``, the base's pitch and roll columns), levels 1-3's and levels 1-2's alone, on ``jac``,
    ``jac_com``, ``jac_sole`` and ``coll``'s Jacobian already masked by ``col_mask``. Every op is row-wise, so a
    row's step doesn't depend on which deps share the call.

    ``com`` is the CoM about the support's centre, in its frame, and ``half``
    the rectangle level 1 holds it in, a support's ``keyframe_half``.
    ``jac_sole`` and ``r_sole`` are the 12 sole rows
    (:func:`_sole_residual_and_jacobian`): level 1 closes them beside the
    active CoM axes, and level 2 moves in their null space. ``coll`` is
    ``(c [N, 1], jac [N, 1, dof])``, the summed penetration
    (:func:`collision_penetration`) and its Jacobian: level 1 drives a row in collision to ``-GN_COLL_MARGIN``.
    A row out of collision has a zero gradient there, so its row stays off for the call's step: the next
    iteration catches a step that ends in contact."""
    from cutamp.com_polygon_cost import _half_extents

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
    # A joint the box-scaled step would carry past a limit is frozen too, and
    # the step is then scaled inside the limits: scaling alone stalls a level
    # whenever its step pushes a joint at a limit (on blocks_t1's pick confs
    # it left the CoM outside on 16 of 46 rows).
    active = com.abs() > half
    com_target = com.sign() * inner
    if coll is not None:
        pen, jac_pen = coll
        in_contact = pen > 0
        coll_target = (-GN_COLL_MARGIN - pen) * in_contact
        jac_pen = jac_pen * in_contact.unsqueeze(-1)
    # Level 3 (:func:`_centring_step`, with ``centre``) changes none of levels 1-2's active sets, so ``step12``
    # is bitwise the step levels 1-2 take without it: the CoM's pushes and the frozen joints are all tested on
    # levels 1-2's step. Level 3's own step can still cross a bound: it moves one free CoM axis outward while it
    # pulls the other in (a damped least-squares step on both: on blocks_t1's gn_ik, 81-101 of the 2400-3500 free
    # axes it moved, by up to 1.2 cm, past the edge on at most 2 of 2304 rows). So a row takes ``step12`` where
    # level 3's step would carry a CoM axis that levels 1-2's leave inside past the rectangle (linear, as their
    # pushes are tested), as the caller's contact test does where the step would end in contact
    # (:func:`gn_project_balanced`), and where the row is in contact: there
    # the collision row pushes the hand off its target to clear it, and the pull builds on a posture the hand then
    # has to leave. On a noisy self-contact conf the two alternated for a whole 10-iteration call, the clearing step
    # 18-23 mm off the hand target and level 2's step back into the contact, ending 7.5 mm off, where without
    # level 3 it converged (2026-11-06). Level 3's own frozen set (``free3``, inside ``free``) is the joints its
    # step pushes past a limit; freezing them in ``free`` would let the lowest level take a joint from the top
    # two. Level 4 (:func:`_upright_step`, with ``upright``) is to level 3 what level 3 is to levels 1-2: it changes
    # none of their active or frozen sets (its own, ``free4``, is inside ``free3``), so ``step3`` is bitwise the step
    # levels 1-3 take without it; it aims from levels 1-3's box-scaled step, so its own never raises pitch^2 +
    # roll^2 (:func:`_upright_step`); and it rides on level 3's vetoes (a row that takes ``step12`` takes it for
    # both), since it moves no CoM axis (to first order) and never raises that sum. Its own veto is the caller's
    # contact test, where a row takes ``step3``.
    free = torch.ones_like(q)
    free3 = free4 = free
    for _ in range(3):
        fc = free.unsqueeze(1)
        row_coll = (None, None) if coll is None else (jac_pen * fc, coll_target)
        dq1, dq2, ja = _prioritized_step(jac_com * fc, com, active, com_target, jac * fc, r, damping,
                                         jac_sole * fc, r_sole, *row_coll)
        dq = dq1 + dq2
        com_lin = com + (jac_com @ dq.unsqueeze(-1)).squeeze(-1)
        if centre:
            f3c = free3.unsqueeze(1)
            dq3 = _centring_step(ja * f3c, jac * f3c, jac_com * f3c, com_lin, active, half)
            if upright is not None:
                f4c = free4.unsqueeze(1)
                q3 = q + _scaled_step(q, dq1, dq2, lower, upper, dq3)
                dq4 = _upright_step(ja * f4c, jac * f4c, jac_com * f4c, active, half, q3, upright, free4)
        pushed_out = ~active & (com_lin.abs() > half)
        blocked = ((q >= upper - 1e-5) & (dq > 0)) | ((q <= lower + 1e-5) & (dq < 0))
        blocked |= _crossing_joints(q, dq1, dq2, lower, upper)
        blocked &= free.bool()
        com_target = torch.where(pushed_out, com_lin.sign() * inner, com_target)
        active = active | pushed_out
        free = free * ~blocked
        if centre:
            free3 = free3 * free * ~_lower_level_blocked(q, (dq1, dq2), dq3, lower, upper)
            if upright is not None:
                free4 = free4 * free3 * ~_lower_level_blocked(q, (dq1, dq2, dq3), dq4, lower, upper)

    step12 = _scaled_step(q, dq1, dq2, lower, upper)
    if not centre:
        return step12, step12, step12
    step = _scaled_step(q, dq1, dq2, lower, upper, dq3)

    def com_after(s):
        return com + (jac_com @ s.unsqueeze(-1)).squeeze(-1)

    skip = in_contact | ((com_after(step).abs() > half) & (com_after(step12).abs() <= half)).any(-1, keepdim=True)
    step3 = torch.where(skip, step12, step)
    if upright is None:
        return step3, step3, step12
    step4 = _add_scaled(q, step, dq4, lower, upper)
    return torch.where(skip, step3, step4), step3, step12


def gn_project_balanced(
    world,
    q_full: torch.Tensor,
    world_from_ee: torch.Tensor,
    arms: str | Sequence[str],
    iters: int = GN_ITERS,
    movables: Optional[torch.Tensor] = None,
    collision: bool = True,
) -> torch.Tensor:
    """Project ``q_full`` (``[N, dof]`` full-cspace joint positions, not modified) toward
    ``FK(q) == world_from_ee`` (``[N, 4, 4]`` target tool poses), balance first, on ``world.kinematics_jac``.

    Each step is a four-level hierarchy (Kanoun et al. T-RO 2011; Escande et
    al. IJRR 2014): (1) keep the CoM, in the support frame of
    ``world.model.support``, inside its keyframe rectangle shrunk by
    ``GN_COM_MARGIN``, both soles on the stance (``world.stance_soles``),
    since the free base and leg columns would otherwise slide them, and the
    robot out of collision with the static scene,
    with ``movables`` (``[N, M, 4]`` world-frame object spheres per row, the objects where the rollout puts
    them at that conf; ``None``: none) and with itself (one row on :func:`collision_penetration`, active where
    it is positive and driven ``GN_COLL_MARGIN`` past contact), since Adam's collision costs never move a
    covered conf (``collision=False`` drops the row, for :func:`~cutamp.particle_initialization.gn_ik`'s
    reaching iterations); (2) least squares
    on the pose residual in whatever freedom level 1 leaves; (3) with the collision row on, on every iteration
    but the last, the CoM toward the rectangle's centre in whatever freedom levels 1-2 leave
    (:func:`_centring_step`), so a conf doesn't stay edge-parked where level 1 stopped its CoM, except on a row
    in contact, or whose step would end in contact or (linearly) with a CoM axis past the
    rectangle that levels 1-2 left inside, which takes levels 1-2's step alone
    (an FK per iteration tests the step's end): the collision row is blind outside contact, so the pull
    would take a conf back into a contact the row had just cleared. A conf whose
    pull runs into a contact stays where it stopped, off-centre: on one init's blocks_t1 first pick the median row
    stops 0.76 of the half-width out in y, where the pull would lower the Trunk onto the right Shank; (4) with
    level 3, the Trunk toward upright (the base's pitch and roll toward 0) in whatever freedom levels 1-3 leave
    (:func:`_upright_step`), so a conf keeps less of the tilt its start or its reach gave it, on the rows that take
    level 3's step, except where its own would end in contact (a second FK), where they take
    levels 1-3's. Neither
    lower level changes the active or frozen sets of the levels above it, and the last iteration's level 2 closes
    the hand's second-order drift from their last step. An unreachable
    target therefore returns the closest balanced pose, not an unbalanced
    exact one, and its leftover pose error is what the kinematic cost hands
    to the placement.

    Beyond the hierarchy:
    - steps are scaled into the ``GN_STEP_CLAMP`` box (level 1 first, then
      the largest fraction of level 2 that fits, then of levels 3 and 4), not clipped per joint, so
      they keep their direction;
    - damping grows with the squared residual (Levenberg-Marquardt), which
      keeps steps sane where the levels conflict;
    - the CoM's and the penetration's Jacobians come from autograd through cuRobo's FK, so each
      iteration costs about three iterations on the pose residual alone.

    Every level's step also stays inside the joint
    limits (crossing joints frozen, then each level scaled), so the final
    clamp cuts nothing that would move the soles, and the call ends with
    :func:`_sole_closing_step`, which closes the soles' residual that the last
    iteration leaves (second-order drift of a saturated step) without moving
    the hand. Together they hold the soles within 0.1 mm after each call on
    blocks_t1's pick confs, out-of-reach targets included.

    ``arms`` holds one arm per IK dep (a bare ``str`` is one): rows are
    ``len(arms)`` equal dep-major blocks, block ``i`` on ``arms[i]``'s tool
    frame, with the other arm's columns frozen, so that hand rides with the trunk (known-bugs B15).

    Returns ``[N, dof]`` projected joint positions, clamped to the joint
    limits. Convergence is NOT guaranteed: :func:`gn_gate` measures it.
    """
    from cutamp.com_polygon_cost import _project_com_to_base

    N = q_full.shape[0]
    arms = (arms,) if isinstance(arms, str) else tuple(arms)
    support = world.model.support
    kin, frames, joint_names, q, lower, upper, col_mask, is_left = _setup(world, q_full, arms)
    sole_idx = tuple(list(kin.tool_frames).index(frame) for frame in world.model.sole_frames)
    upright = (joint_names.index("base_j_pitch"), joint_names.index("base_j_roll"))

    def in_contact_at(q_end):
        spheres = kin.compute_kinematics(JointState.from_position(q_end, joint_names=joint_names)).robot_spheres
        return (collision_penetration(world, spheres, movables) > 0).unsqueeze(-1)

    for it in range(iters):
        q_g = q.detach().requires_grad_(True)
        with torch.enable_grad():
            state = kin.compute_kinematics(JointState.from_position(q_g, joint_names=joint_names))
            # The support frame is constant: the stance hangs off the world, not off the free base. Held under
            # no_grad, so CoM gradients never reach it (design 7.4).
            with torch.no_grad():
                base_T = (
                    state.tool_poses.get_link_pose(support.link, make_contiguous=True)
                    .get_matrix().reshape(N, 4, 4)
                )
            com_g = _project_com_to_base(state.robot_com[..., :3].reshape(N, 3), base_T)
            # Clone: cuRobo's backward writes into a reused buffer, so without
            # it the next grad call overwrites the x row with the y row.
            jcx = torch.autograd.grad(com_g[:, 0].sum(), q_g, retain_graph=True)[0].clone()
            jcy = torch.autograd.grad(com_g[:, 1].sum(), q_g, retain_graph=collision)[0].clone()
            if collision:
                pen_g = collision_penetration(world, state.robot_spheres, movables)
                jpen = torch.autograd.grad(pen_g.sum(), q_g)[0]
        with torch.no_grad():
            jac_com = torch.stack([jcx, jcy], dim=1) * col_mask.unsqueeze(1)    # [N, 2, dof]
            com = com_g.detach()
            r, jac = _pose_residual_and_jacobian(state, frames, world_from_ee, col_mask, is_left)
            r_sole, jac_sole = _sole_residual_and_jacobian(state, sole_idx, world.stance_soles, col_mask)
            coll = (pen_g.detach().unsqueeze(-1), (jpen * col_mask).unsqueeze(1)) if collision else None
            # Levels 3 and 4 run beside the collision row, whose contact tests below them need, and not on the last
            # iteration: level 3's second-order drift moves the hand (about 3 mm per 0.3 rad step on noisy
            # self-contact confs, 2026-11-06), which the next iteration's level 2 closes and the last would leave.
            centre = collision and it < iters - 1
            step, step3, step12 = _balanced_step(q, r, jac, jac_com, com, lower, upper, col_mask,
                                                 half=support.keyframe_half, jac_sole=jac_sole, r_sole=r_sole,
                                                 coll=coll, centre=centre, upright=upright)
            if centre:
                # Levels 3 and 4 never take a row into contact: where levels 1-3's step ends in contact, levels 1-2's
                # alone, and where only the full step does, levels 1-3's. Each test is its own FK, read before the
                # next one reuses cuRobo's buffers.
                hit3 = in_contact_at((q + step3).clamp(lower, upper))
                hit = hit3 | in_contact_at((q + step).clamp(lower, upper))
                step = torch.where(hit, torch.where(hit3, step12, step3), step)
            q = (q + step).clamp(lower, upper)

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

    Returns ``(pose_ok [N], com_ok [N])``: the hand's pose errors (``curobo_pose_error``, the
    checker's metric, on each row's own arm) strictly under ``pos_tol`` and
    ``rot_tol``, both soles' errors against the stance too
    (:func:`soles_on_stance`, at most the same tolerances), and
    ``compute_com_polygon_mask``. ``q_full`` must be contiguous float32 on
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
    pose_ok = (pos_err < pos_tol) & (rot_err < rot_tol) & soles_on_stance(world, state, pos_tol, rot_tol)
    return pose_ok, com_polygon_mask_from_state(state, world.model.support)


def soles_on_stance(world, state, pos_tol: float, rot_tol: float) -> torch.Tensor:
    """``[N]`` bool: both soles' pose errors against the stance (``world.stance_soles``, the checker's metric)
    within ``pos_tol`` and ``rot_tol`` as the checker accepts them (:func:`~cutamp.com_polygon_cost.soles_planted`),
    on an FK ``state`` of ``world.kinematics`` or one of its sites. A conf whose soles left the stance passes a
    CoM check on a support polygon it no longer stands on (design 7.4)."""
    from cutamp.com_polygon_cost import soles_planted
    from cutamp.costs import curobo_pose_error

    soles = torch.stack([state.tool_poses.get_link_pose(frame, make_contiguous=True).get_matrix()
                         for frame in world.model.sole_frames], dim=1)  # [N, 2, 4, 4]
    sole_pos, sole_rot = curobo_pose_error(soles, world.stance_soles)
    return soles_planted(sole_pos, sole_rot, pos_tol, rot_tol).all(-1)
