# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Particle initialization under cuRobo's single-MotionPlanner architecture.

Each conf parameter (``left_q0``, ``right_q1``, etc.) is a single vector in
the model's full cspace order (its 33 joints). A pick's and a place's conf come from :func:`gn_ik`, the
planner's only IK: balance first, clear of collision. A pool walk first solves each GraspGenX grasp that the hand can
take, from any side, once at its pick and its place; each pick row then takes one of the most upright balanced
ones in turn (:func:`choose_grasps`), and each place row draws its placement, with the yaw following the held grasp;
the rows whose IK fails draw again (design 7.8).
"""

import logging
import math
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import roma
import torch

from curobo.scene import Obstacle
from curobo.types import JointState, Pose

from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.gn_refresh import (
    _row_layout,
    _row_tool_pose,
    collision_penetration,
    gn_project_balanced,
    soles_on_stance,
)
from cutamp.grasps import AZIMUTH_CONE, GRASP_REDRAW_ROUNDS, GRASP_SCORE_CUTOFF, near_vertical
from cutamp.joint_limit_cost import trunk_tilt_cosine
from cutamp.rollout import movable_spheres_fn
from cutamp.samplers import place_4dof_sampler
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import KinematicConstraint
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    grasp_7dof_to_mat4x4,
    transform_spheres,
)

_log = logging.getLogger(__name__)


#: A noisy start's leg noise: each pitch joint draws once per pair, left onto right,
#: and roll and yaw hold q_init's, so the legs stay mirrored.
_MIRRORED_LEG_JOINTS = ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch")
_HELD_LEG_JOINTS = ("Hip_Roll", "Hip_Yaw", "Ankle_Roll")
#: The floating base's joints in ``model.base`` order: x, y, z, then YXZ Euler (build_planner_model.py's ``BASE``).
_BASE_JOINTS = ("base_j_x", "base_j_y", "base_j_z", "base_j_pitch", "base_j_roll", "base_j_yaw")


def _noisy_start(world: TAMPWorld, batch_size: int, arm: Optional[str], noise_std: float) -> torch.Tensor:
    """``[batch_size, dof]``: one of :func:`gn_ik`'s noisy start confs, ``world.q_init``
    plus Gaussian noise of std ``noise_std`` rad on the body and the active
    arm, clamped to the joint limits. The inactive arm keeps ``q_init``.

    The model's root is the trunk, so that noise would swing the soles. The
    stance is the noise's root instead: the legs draw each pitch once per pair
    and hold roll and yaw, and the base is re-solved so that the first sole
    sits on its stance pose (``_stand_on_the_stance``).
    """
    model = world.model
    full_dof = world.q_init.shape[-1]
    device = world.q_init.device
    seed = world.q_init.unsqueeze(0).expand(batch_size, -1).clone()
    noise = torch.randn(batch_size, full_dof, device=device) * noise_std
    noise[:, model.base] = 0.0  # re-solved from the stance below
    if arm == "left":
        noise[:, model.right_arm] = 0.0  # right arm locked
    elif arm == "right":
        noise[:, model.left_arm] = 0.0  # left arm locked
    # ponytail: holds while q_init's legs mirror each other with zero roll and yaw (home, as setup_cutamp sets
    # it); walking (stage 5) revisits it.
    col = list(model.joint_names).index
    for joint in _MIRRORED_LEG_JOINTS:
        noise[:, col(f"Right_{joint}")] = noise[:, col(f"Left_{joint}")]
    for joint in _HELD_LEG_JOINTS:
        for side in ("Left", "Right"):
            noise[:, col(f"{side}_{joint}")] = 0.0
    seed = seed + noise
    # Clamp the perturbed seed to joint limits, which GN keeps every iterate inside. world.joint_limits are the
    # full-cspace kinematics' ([2, full_dof]), in world.q_init's joint order.
    # The left and right pitch limits are equal, so the clamp keeps the legs mirrored.
    _lower, _upper = world.joint_limits.to(seed.device, seed.dtype)
    seed = torch.clamp(seed, min=_lower, max=_upper)
    return _stand_on_the_stance(world, seed, _lower, _upper)


def _stand_on_the_stance(
    world: TAMPWorld, seed: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
) -> torch.Tensor:
    """``seed`` with each row's base re-solved so that the first of
    ``model.sole_frames`` sits on its stance pose: ``B' = S @ inv(W) @ B``,
    with ``S`` the stance pose, ``W`` the sole's FK pose and ``B`` the base
    pose. With mirrored legs (``_noisy_start``) the other sole's pose relative
    to the first is unchanged, so it lands on its stance pose too. A row whose
    ``B'`` leaves the base joints' limits takes home's base and body, keeping
    only its arm's noise.
    """
    model = world.model
    if tuple(model.joint_names[model.base]) != _BASE_JOINTS:
        raise ValueError(f"expected the base joints {_BASE_JOINTS}, got {model.joint_names[model.base]}")
    kin = world.kinematics
    state = kin.compute_kinematics(JointState.from_position(seed, joint_names=list(kin.joint_names)))
    # get_matrix copies the pose out of the FK state: cuRobo reuses its output buffers.
    sole = state.tool_poses.get_link_pose(model.sole_frames[0], make_contiguous=True).get_matrix().reshape(-1, 4, 4)
    stance = world.stance_soles[0]
    q_base = seed[:, model.base]
    # B' = S @ inv(W) @ B, as rotation and translation.
    rot = stance[:3, :3] @ sole[:, :3, :3].transpose(-1, -2)
    new_rot = rot @ roma.euler_to_rotmat("YXZ", q_base[:, 3:])
    new_pos = (rot @ (q_base[:, :3] - sole[:, :3, 3]).unsqueeze(-1)).squeeze(-1) + stance[:3, 3]
    new_base = torch.cat([new_pos, roma.rotmat_to_euler("YXZ", new_rot)], dim=-1)
    inside = ((lower[model.base] <= new_base) & (new_base <= upper[model.base])).all(-1, keepdim=True)
    home = torch.tensor(model.home, device=seed.device, dtype=seed.dtype)
    out = seed.clone()
    out[:, model.base] = torch.where(inside, new_base, home[model.base])
    out[:, model.body] = torch.where(inside, seed[:, model.body], home[model.body])
    return out


#: The GN IK (:func:`gn_ik`): GN_IK_ITERS iterations that reach balance first, then GN_IK_CLEAR_ITERS with the
#: collision row, and the noise std (rad) of the stance-rooted starts that run beside the first (home), one start
#: per value. The iterates are not motions, so only the conf they end at must be clear;
#: with the row on from the start, an arm that met the body on its way pressed against it, and the picks of
#: blocks_t1's first plan fell from 64 to 49 of 64 targets (20 + 10 iterations: 52; 25 + 5 and 30 + 10: 64).
#: Places reached 42 of 64 on two plans at every split; their misses are out of balanced reach (still about 6 cm
#: off at 120 iterations), and the clear iterations take the rest out of contact. Before the collision row, 16
#: starts gained 0-3 targets over one, and four rows per target cost about what one does (a call: 0.2 s at 64
#: targets, 256 rows, and at 256).
GN_IK_ITERS = 30
GN_IK_CLEAR_ITERS = 10
GN_IK_START_SIGMAS = (0.1, 0.2, 0.3)


class IKTol(NamedTuple):
    """The constraint checker's tolerances an IK conf must meet: the hand's and the soles' pose errors
    (KinematicConstraint)."""
    pos: float
    rot: float
    sole_pos: float
    sole_rot: float

    @classmethod
    def from_checker(cls, checker: ConstraintChecker) -> "IKTol":
        kc = KinematicConstraint.type
        return cls(*(checker._get_tol(kc, n) for n in ("pos_err", "rot_err", "sole_pos_err", "sole_rot_err")))


def ik_check(world: TAMPWorld, q: torch.Tensor, world_from_ee: torch.Tensor, arms: Union[str, Sequence[str]],
             movables: torch.Tensor, tol: IKTol):
    """``(ok, safe, hand_err)``, each ``[N]``, from one FK of ``q`` ``[N, dof]`` on ``world.kinematics``, with rows
    in ``arms``' dep-major blocks (:func:`~cutamp.gn_refresh.gn_project_balanced`'s layout). ``safe``: the conf
    stands balanced (the CoM inside the support rectangle, both soles on the stance within ``tol``), clear of the
    scene, of ``movables`` ``[N, M, 4]`` and of itself (:func:`~cutamp.gn_refresh.collision_penetration` is 0).
    ``ok``: safe, with the hand on ``world_from_ee`` within ``tol``, all at
    the checker's ``<=``. ``hand_err``: the hand's position plus rotation error, :func:`gn_ik`'s ranking."""
    from cutamp.com_polygon_cost import _fk_state, com_polygon_mask_from_state
    from cutamp.costs import curobo_pose_error

    arms = (arms,) if isinstance(arms, str) else tuple(arms)
    _, is_left = _row_layout(world.model, arms, q.shape[0], q.shape[1], q.device, q.dtype)
    state = _fk_state(world, q)
    pos, quat = _row_tool_pose(state, world.tool_frame_for_arm("left"), world.tool_frame_for_arm("right"), is_left)
    pos_err, rot_err = curobo_pose_error(
        Pose(position=pos.contiguous(), quaternion=quat.contiguous()).get_matrix(), world_from_ee,
    )
    safe = (com_polygon_mask_from_state(state, world.model.support)
            & soles_on_stance(world, state, tol.sole_pos, tol.sole_rot)
            & (collision_penetration(world, state.robot_spheres, movables) <= 0))
    return safe & (pos_err <= tol.pos) & (rot_err <= tol.rot), safe, pos_err + rot_err


def gn_ik(world: TAMPWorld, world_from_ee: torch.Tensor, arms: Union[str, Sequence[str]], movables: torch.Tensor,
          tol: IKTol):
    """The planner's IK: :func:`~cutamp.gn_refresh.gn_project_balanced` of ``world_from_ee`` ``[N, 4, 4]``, rows
    in ``arms``' dep-major blocks (a bare ``str`` is one dep), from ``1 + len(GN_IK_START_SIGMAS)`` starts per row
    in one batched call, ``world.q_init`` and :func:`_noisy_start`'s stance-rooted noise:
    ``GN_IK_ITERS`` iterations balance first, then ``GN_IK_CLEAR_ITERS`` that also clear the scene, ``movables``
    ``[N, M, 4]`` and the robot itself, and pull the CoM toward the keyframe rectangle's centre and the Trunk toward
    upright (GN's levels 3 and 4).

    Returns ``(q [N, dof], ok [N])``: per row, the accepted start (:func:`ik_check`) whose Trunk tilts least; where
    none is, the safe one closest to the target, else the closest, a balanced-first iterate that the periodic refresh keeps
    projecting and whose pose error the kinematic cost hands to the placement. ``ok`` marks the accepted rows."""
    arms = (arms,) if isinstance(arms, str) else tuple(arms)
    n = world_from_ee.shape[0]
    batch = n // len(arms)
    noisy = [torch.cat([_noisy_start(world, batch, arm, s) for arm in arms]) for s in GN_IK_START_SIGMAS]
    starts = [world.q_init.expand(n, -1)] + noisy
    k = len(starts)
    targets, mov, rows_arms = world_from_ee.repeat(k, 1, 1), movables.repeat(k, 1, 1), arms * k
    q = gn_project_balanced(world, torch.cat(starts), targets, rows_arms, iters=GN_IK_ITERS, collision=False)
    q = gn_project_balanced(world, q, targets, rows_arms, iters=GN_IK_CLEAR_ITERS, movables=mov)
    ok, safe, err = ik_check(world, q, targets, rows_arms, mov, tol)
    # Tiers: accepted, by the Trunk's tilt (1 - its cosine, at most 2), then safe, then the rest, by hand error
    # (under 10: metres plus radians). Accepted starts' hand errors tie to two digits, so by hand error a row took
    # whichever converged a hair closer: on blocks_t1's floor picks the starts that a noisy start tilts back reach a
    # second branch, the Trunk 70-115 deg back, and the first left pick kept it on 2 of 64 rows, 10 with level 4,
    # which shifts the hair (seed 1, 2026-11-06). Level 4 can't bring such a row up: it is local.
    names = list(world.model.joint_names)
    tilt = 1.0 - trunk_tilt_cosine(q, names.index("base_j_pitch"), names.index("base_j_roll"))
    best = (torch.where(ok, tilt, err) + 100.0 * ~ok + 1e4 * ~safe).view(k, n).argmin(0) * n
    best = best + torch.arange(n, device=q.device)
    return q[best], ok[best]


def worst_posture(world: TAMPWorld, confs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per row of ``confs`` (each ``[B, dof]``): the Trunk's tilt in radians and its travel from home in metres, each
    the largest over the confs, and the CoM margin in metres, the smallest."""
    from cutamp.com_polygon_cost import compute_com_margin

    names = list(world.model.joint_names)
    pitch, roll = names.index("base_j_pitch"), names.index("base_j_roll")
    xyz = [names.index(n) for n in _BASE_JOINTS[:3]]
    tilts, travels, margins = [], [], []
    for q in confs:
        tilts.append(torch.acos(trunk_tilt_cosine(q, pitch, roll).clamp(-1, 1)))
        home = torch.as_tensor(world.model.home, dtype=q.dtype, device=q.device)
        travels.append((q[:, xyz] - home[xyz]).norm(dim=-1))
        margins.append(compute_com_margin(world, q))
    return (torch.stack(tilts).amax(0), torch.stack(travels).amax(0), torch.stack(margins).amin(0))


def choose_grasps(ok: torch.Tensor, scores: torch.Tensor, tilt: torch.Tensor, travel: torch.Tensor,
                  margin: torch.Tensor, k: int) -> Optional[torch.Tensor]:
    """Pool indices (a LongTensor on ``scores.device``, at most ``k``, in try order) of the grasps a pick tries, or
    ``None`` when none is balanced. The balanced grasps scoring at least ``GRASP_SCORE_CUTOFF``, upright first (by
    tilt, travel, larger CoM margin, then index), first ``k``; when fewer clear it, the best-scoring balanced grasps
    below it (score ties by index) fill, ordered among themselves by the same key."""
    okl, sc, ti, tr, mg = (t.tolist() for t in (ok, scores, tilt, travel, margin))
    clears = (scores >= GRASP_SCORE_CUTOFF).tolist()  # in the scores' dtype, so a float32 0.7 clears
    balanced = [i for i, o in enumerate(okl) if o]
    key = lambda i: (ti[i], tr[i], -mg[i], i)
    chosen = sorted((i for i in balanced if clears[i]), key=key)[:k]
    if len(chosen) < k:
        below = sorted((i for i in balanced if not clears[i]), key=lambda i: (-sc[i], i))
        chosen += sorted(below[:k - len(chosen)], key=key)
    return torch.tensor(chosen, dtype=torch.long, device=scores.device) if chosen else None


def next_place(plan_skeleton: PlanSkeleton, pick_idx: int) -> Optional[str]:
    """The conf name of the first place after ``plan_skeleton[pick_idx]`` that moves the pick's object, else
    ``None``."""
    obj = plan_skeleton[pick_idx].values[0]
    for op in plan_skeleton[pick_idx + 1:]:
        if op.operator.metadata.action_type == "place" and op.values[0] == obj:
            return op.values[-1]
    return None


def pick_place_confs(plan_skeleton: PlanSkeleton) -> List[str]:
    """The conf name of every pick and place op, in skeleton order."""
    return [op.values[-1] for op in plan_skeleton if op.operator.metadata.action_type in ("pick", "place")]


class PickChoice(NamedTuple):
    """One pick's grasp pool and its pool-walk results; ``ok``, ``tilt``, ``travel`` and ``margin`` are taken at the
    worse of the pick and its place. ``chosen``: pool indices in try order, or ``None`` when none is balanced."""

    obj: str
    grasp: str
    grasps: torch.Tensor  # [m, 7]
    scores: torch.Tensor  # [m]
    ok: torch.Tensor  # [m] bool
    tilt: torch.Tensor  # [m] radians
    travel: torch.Tensor  # [m] metres
    margin: torch.Tensor  # [m] metres
    chosen: Optional[torch.Tensor]


def choice_stats(c: PickChoice) -> dict:
    """``c`` as plain JSON-ready values: pool and balanced counts, and ranges over the chosen grasps."""
    span = lambda x, s=1.0: None if c.chosen is None else [float(x[c.chosen].min()) * s, float(x[c.chosen].max()) * s]
    return {
        "object": c.obj, "pool": int(c.scores.numel()), "balanced": int(c.ok.sum()),
        "balanced_at_cutoff": int((c.ok & (c.scores >= GRASP_SCORE_CUTOFF)).sum()),
        "chosen": 0 if c.chosen is None else int(c.chosen.numel()),
        "tilt_deg": span(c.tilt, math.degrees(1.0)), "travel": span(c.travel), "com_margin": span(c.margin),
    }


class _Step(NamedTuple):
    """One pick or place of a walk, which :func:`_solve_steps` solves with the walk's others. ``name``: its drawn
    parameter (the grasp or the placement), ``q``: its conf, ``ik_inputs(rows)``: the rows' hand targets ``[r, 4, 4]``
    and movables ``[r, M, 4]`` under the particles as they then stand, ``redraw(rows)``: the rows' values when they
    draw again, ``q_start``: start confs ``[n, dof]`` (the rows that :func:`ik_check` accepts keep theirs).
    ``drawn_from``: the earlier steps' parameters its draw reads (a place's grasp); ``checked_against``: those its
    movables read (the earlier placements)."""

    name: str
    q: str
    arm: str
    ik_inputs: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    redraw: Callable[[torch.Tensor], torch.Tensor]
    drawn_from: Tuple[str, ...] = ()
    checked_against: Tuple[str, ...] = ()
    q_start: Optional[torch.Tensor] = None


def _gn_ik_blocks(world: TAMPWorld, blocks: Sequence[Tuple[torch.Tensor, str, torch.Tensor]], tol: IKTol):
    """:func:`gn_ik` of ``blocks`` (each ``(targets [r, 4, 4], arm, movables [r, M, 4])``, ``r >= 1``) in one call:
    ``[(q [r, dof], ok [r])]``. gn_ik's dep blocks are equal, so each block repeats its rows up to the largest's, and
    the repeats are dropped."""
    sizes = [len(t) for t, _, _ in blocks]
    r = max(sizes)
    pad = [torch.arange(r, device=world.device) % size for size in sizes]
    q, ok = gn_ik(world, torch.cat([t[i] for (t, _, _), i in zip(blocks, pad)]), tuple(a for _, a, _ in blocks),
                  torch.cat([m[i] for (_, _, m), i in zip(blocks, pad)]), tol)
    return [(q[i * r:i * r + size], ok[i * r:i * r + size]) for i, size in enumerate(sizes)]


def _solve_steps(world: TAMPWorld, particles: Particles, steps: Sequence[_Step], n: int, rounds: int,
                 tol: IKTol) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]:
    """Solves every step's ``n`` rows, drawn in ``particles``, in one :func:`_gn_ik_blocks` call (a step with
    ``q_start`` solves only the rows whose start :func:`ik_check` rejects), then in rounds, one call each, the rows to
    solve again. A round walks the steps in order: a row draws again (``redraw``) where a parameter in its
    ``drawn_from`` drew again, or where its IK failed and it has redraws left, and is checked again, as it stands,
    where one in its ``checked_against`` drew again; it is solved again if it drew or the check rejects it. A row has
    ``rounds`` redraws after the last change to the values it reads (a redraw or a rejected check restarts them). So
    every row ends solved under the values the particles end with, with the redraws it had when each step finished
    its rounds before the next drew.

    Writes ``particles[step.q]``; returns, per conf, ``(q [n, dof], ok [n], accepted counts after the first solve and
    each round)``."""
    q, ok, todo = {}, {}, {}  # todo: per conf, the rows to solve
    every = torch.arange(n, device=world.device)
    for step in steps:
        if step.q_start is None:
            q[step.q], ok[step.q], todo[step.q] = world.q_init.repeat(n, 1), torch.zeros(n, dtype=torch.bool,
                                                                                         device=world.device), every
        else:
            targets, movables = step.ik_inputs(every)
            q[step.q], ok[step.q] = step.q_start.clone(), ik_check(world, step.q_start, targets, step.arm, movables,
                                                                   tol)[0]
            todo[step.q] = (~ok[step.q]).nonzero().squeeze(-1)
    accepted = {step.q: [] for step in steps}
    left = {step.q: torch.full((n,), rounds, device=world.device) for step in steps}  # each row's redraws left
    # A row's redraws restart when a value it reads changes, so a step at depth d is done by round d * rounds.
    last = len(steps) * rounds
    for round_ in range(last + 1):
        blocks = [(step, todo[step.q]) for step in steps if todo[step.q].numel()]
        if round_ and not blocks:
            break
        if blocks:
            inputs = [step.ik_inputs(rows) for step, rows in blocks]
            solved = _gn_ik_blocks(world, [(t, step.arm, m) for (step, _), (t, m) in zip(blocks, inputs)], tol)
            for (step, rows), (q_rows, ok_rows) in zip(blocks, solved):
                q[step.q][rows], ok[step.q][rows] = q_rows, ok_rows
        for step in steps:
            accepted[step.q].append(int(ok[step.q].sum()))
        if round_ == last:  # no draw may follow the last solve
            break
        drew = {}  # per drawn parameter, the rows that drew again this round
        for step in steps:
            fresh = torch.zeros(n, dtype=torch.bool, device=world.device)
            for name in step.drawn_from:
                fresh = fresh | drew[name]
            own = ~ok[step.q] & (left[step.q] > 0) & ~fresh
            draw = own | fresh
            left[step.q] = torch.where(fresh, rounds, left[step.q] - own.long())
            rows = draw.nonzero().squeeze(-1)
            if rows.numel():
                particles[step.name][rows] = step.redraw(rows)
            drew[step.name] = draw
            moved = torch.zeros_like(draw)
            for name in step.checked_against:
                moved = moved | drew[name]
            check = (moved & ~draw).nonzero().squeeze(-1)
            if check.numel():
                targets, movables = step.ik_inputs(check)
                passed = ik_check(world, q[step.q][check], targets, step.arm, movables, tol)[0]
                ok[step.q][check] = passed
                left[step.q][check[~passed]] = rounds
                rows = torch.cat([rows, check[~passed]])
            todo[step.q] = rows
    for step in steps:
        particles[step.q] = q[step.q]
    return {step.q: (q[step.q], ok[step.q], accepted[step.q]) for step in steps}


#: Placement candidates per row at a place: the row keeps the one that collides least with the scene.
_PLACE_DRAWS = 4


def _draw_placements(
    world: TAMPWorld, obj: Obstacle, obj_spheres: torch.Tensor, surface: Obstacle, held: torch.Tensor,
) -> torch.Tensor:
    """``[n, 4]``: a placement of ``obj`` on ``surface`` for each row of ``held`` ``[n, 4, 4]``, the row's grasp of
    ``obj`` (G in the object frame). Of ``_PLACE_DRAWS`` candidates whose xyz come from ``place_4dof_sampler``, the row
    keeps the one whose spheres ``obj_spheres`` collide least with the scene (``world.collision_fn``).

    Each candidate's yaw follows the grasp: ψ = φ_u - φ_g + U(-AZIMUTH_CONE, AZIMUTH_CONE), with φ_g the azimuth of
    the grasp's approach Z_G in the object frame and φ_u that of the candidate's xy minus the stance's (the mean of
    ``world.stance_soles``). The world approach, ψ + φ_g, then comes from the robot's side, within ``AZIMUTH_CONE`` of
    the direction from the stance to the placement. A near-vertical grasp keeps the sampler's uniform yaw."""
    n = held.shape[0]
    candidates = place_4dof_sampler(n * _PLACE_DRAWS, obj, obj_spheres, surface).view(n, _PLACE_DRAWS, 4)
    approach = held[:, :3, 2]
    away = candidates[..., :2] - world.stance_soles[:, :2, 3].mean(0)
    phi_u, phi_g = torch.atan2(away[..., 1], away[..., 0]), torch.atan2(approach[:, 1:2], approach[:, :1])
    psi = phi_u - phi_g + (2 * torch.rand_like(phi_u) - 1) * AZIMUTH_CONE
    candidates[..., 3] = torch.where(near_vertical(approach)[:, None], candidates[..., 3],
                                     torch.remainder(psi, 2 * math.pi))
    world_from_obj = action_4dof_to_mat4x4(candidates)  # [n, k, 4, 4]
    collision = world.collision_fn(transform_spheres(obj_spheres, world_from_obj).contiguous())  # [n, k]
    return candidates[torch.arange(n, device=candidates.device), collision.argmin(-1)]


class ParticleInitializer:
    def __init__(self, world: TAMPWorld, config: TAMPConfiguration, constraint_checker: ConstraintChecker):
        self.world = world
        self.config = config
        # A conf's IK is accepted on the tolerances the run's checker enforces.
        self.ik_tol = IKTol.from_checker(constraint_checker)
        # The last call's grasp choice, one per pick in skeleton order.
        self.grasp_choice: List[PickChoice] = []

    def _initial_particles(self, n: int) -> Particles:
        return {
            "left_q0": self.world.q_init.repeat(n, 1),
            "right_q0": self.world.q_init.repeat(n, 1),
        }

    def __call__(self, plan_skeleton: PlanSkeleton, verbose: bool = True) -> Optional[Particles]:
        """``config.num_particles`` particles for ``plan_skeleton``. A pool walk solves every eligible grasp once, at
        its pick and its place (G3); :func:`choose_grasps` chooses each pick's grasps from it (``self.grasp_choice``);
        then the particle walk hands each pick's chosen grasps to the particles in turn (G4)."""
        log_debug = _log.debug if verbose else lambda *args, **kwargs: None
        self.grasp_choice = []
        pools = self._pick_pools(plan_skeleton)
        # One row per grasp of the largest pool; a skeleton without a pick walks no rows.
        _, confs = self._walk(plan_skeleton, max((len(scores) for _, scores in pools.values()), default=0), pools)
        chosen, pool_q = {}, {}
        for idx, ground_op in enumerate(plan_skeleton):
            if ground_op.operator.metadata.action_type != "pick":
                continue
            obj, grasp, q = ground_op.values
            grasps, scores = pools[grasp]
            m = len(scores)
            place = next_place(plan_skeleton, idx)
            names = [q] if place is None else [q, place]  # a pick with no place counts its pick alone (G11)
            ok = torch.stack([confs[name][1][:m] for name in names]).all(0)
            tilt, travel, margin = worst_posture(self.world, [confs[name][0][:m] for name in names])
            choice = PickChoice(obj, grasp, grasps, scores, ok, tilt, travel, margin,
                                choose_grasps(ok, scores, tilt, travel, margin, k=self.config.motion_plan_max_retries))
            self.grasp_choice.append(choice)
            chosen[grasp], pool_q[q] = choice.chosen, confs[q][0]
            if choice.chosen is None:
                judged = "its pick" if place is None else "its pick and its place"
                _log.warning(f"{idx + 1}. {ground_op}: no grasp of {obj} is balanced at {judged}, so its particles "
                             f"draw by score. {choice_stats(choice)}")
            else:
                log_debug(f"{idx + 1}. {ground_op}. Grasp choice: {choice_stats(choice)}")
        particles, _ = self._walk(plan_skeleton, self.config.num_particles, pools, chosen, pool_q, log_debug)
        return particles

    def _pick_pools(self, plan_skeleton: PlanSkeleton) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Each pick's grasps ``[m, 7]`` and scores ``[m]``, keyed by its grasp parameter: every grasp its hand can take
        (``world.grasp_candidates``), top or side, from any direction. The posture rule judges them all."""
        world = self.world
        pools = {}
        for idx, ground_op in enumerate(plan_skeleton):
            if ground_op.operator.metadata.action_type != "pick":
                continue
            arm = ground_op.operator.metadata.arm
            obj, grasp, _q = ground_op.values
            if not world.has_object(obj):
                raise ValueError(f"{obj=} not found in world")
            candidates = world.grasp_candidates(obj, arm)
            if not len(candidates.grasps):
                raise ValueError(f"{idx + 1}. {ground_op}: none of {obj}'s {len(world.grasps[obj].grasps)} grasps is "
                                 f"left for the {arm} hand: each reaches below its bottom face or touches it")
            pools[grasp] = candidates.grasps, candidates.scores
        return pools

    def _pick_step(
        self, plan_skeleton: PlanSkeleton, idx: int, particles: Particles, n: int,
        pools: Dict[str, Tuple[torch.Tensor, torch.Tensor]], chosen: Optional[Dict[str, Optional[torch.Tensor]]],
        pool_q: Optional[Dict[str, torch.Tensor]], placements: Sequence[str],
    ) -> _Step:
        """The pick ``plan_skeleton[idx]``'s step, its grasps drawn into ``particles`` as :meth:`_walk` describes."""
        world, device = self.world, self.world.device
        obj, grasp, q = plan_skeleton[idx].values
        arm = plan_skeleton[idx].operator.metadata.arm
        grasps, scores = pools[grasp]
        redraw = q_start = None
        if chosen is None:  # the pool walk
            draw = lambda rows: grasps[rows % len(grasps)]
        elif chosen[grasp] is None:  # no grasp is balanced (G6): by score, as before the choice
            draw = lambda rows: grasps[torch.multinomial(scores, len(rows), replacement=True)]
        else:  # the chosen grasps in turn, each from its pool conf (G4, G5)
            c = chosen[grasp]
            draw = lambda rows: grasps[c[rows % len(c)]]
            redraw = lambda rows: grasps[c[torch.randint(len(c), (len(rows),), device=device)]]
            q_start = pool_q[q][c[torch.arange(n, device=device) % len(c)]]
        world_from_obj = world.get_object_pose(obj)
        tool_from_ee = world.tool_from_ee[world.tool_frame_for_arm(arm)]
        movables = movable_spheres_fn(world, plan_skeleton, idx)  # the earlier placements; the grasp moves none
        particles[grasp] = draw(torch.arange(n, device=device))
        return _Step(
            grasp, q, arm,
            lambda rows: (world_from_obj @ grasp_7dof_to_mat4x4(particles[grasp][rows]) @ tool_from_ee,
                          movables(particles)[rows]),
            draw if redraw is None else redraw, checked_against=tuple(placements), q_start=q_start,
        )

    def _place_step(self, plan_skeleton: PlanSkeleton, idx: int, particles: Particles, n: int,
                    placements: Sequence[str]) -> _Step:
        """The place ``plan_skeleton[idx]``'s step, its placements drawn into ``particles``: each row's follows the
        grasp the row holds as it then stands (``particles[grasp]``), so it draws again when the grasp does."""
        world = self.world
        obj, grasp, placement, surface, q = plan_skeleton[idx].values
        arm = plan_skeleton[idx].operator.metadata.arm
        obj_curobo, obj_spheres = world.get_object(obj), world.get_collision_spheres(obj)
        surface_curobo = world.get_object(surface)
        obj_from_G = lambda rows: grasp_7dof_to_mat4x4(particles[grasp][rows])
        tool_from_ee = world.tool_from_ee[world.tool_frame_for_arm(arm)]
        # The placed object at its placement (particles[placement]), as the rollout has it, and the earlier ones.
        movables = movable_spheres_fn(world, plan_skeleton, idx)
        draw = lambda rows: _draw_placements(world, obj_curobo, obj_spheres, surface_curobo, obj_from_G(rows))
        particles[placement] = draw(torch.arange(n, device=world.device))
        return _Step(
            placement, q, arm,
            lambda rows: (action_4dof_to_mat4x4(particles[placement][rows]) @ obj_from_G(rows) @ tool_from_ee,
                          movables(particles)[rows]),
            draw, drawn_from=(grasp,), checked_against=tuple(placements),
        )

    def _walk(
        self, plan_skeleton: PlanSkeleton, n: int, pools: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        chosen: Optional[Dict[str, Optional[torch.Tensor]]] = None, pool_q: Optional[Dict[str, torch.Tensor]] = None,
        log_debug: Callable[..., None] = lambda *args, **kwargs: None,
    ) -> Tuple[Particles, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """One walk of ``plan_skeleton`` on ``n`` rows: ``(particles, confs)``, with ``confs[q] = (q [n, dof], ok [n])``
        for every pick and place conf ``q``. Without ``chosen``, the pool walk (G3): pick row i takes grasp i mod m of
        its pool ``pools[grasp]``, and every pick and place solves once. With it, the particle walk (G4): at a pick
        whose chosen grasps ``chosen[grasp]`` are ``c``, row i takes grasp ``c[i % len(c)]`` and starts from that
        grasp's row of the pool walk's confs ``pool_q[q]``, and a failed row draws again from ``c``, uniformly; a pick
        with none (G6) draws by score. Places draw their placements and redraw as before. The walk draws every step
        in skeleton order, then solves them all together (:func:`_solve_steps`)."""
        world = self.world
        device = world.device
        particles = self._initial_particles(n)
        steps: List[_Step] = []
        placements: List[str] = []  # the placements drawn so far, which later steps' movables read
        deferred_params: set[str] = set()
        move_free_deferred: dict[str, str] = {}

        for idx, ground_op in enumerate(plan_skeleton):
            metadata = ground_op.operator.metadata
            params = ground_op.values
            header = f"{idx + 1}. {ground_op}"
            arm = metadata.arm

            if metadata.is_motion and metadata.action_type is None:
                if len(params) == 3:
                    q_start, _traj, q_end = params
                    if q_start not in particles:
                        raise ValueError(f"{q_start=} should already be bound")
                    deferred_params.add(q_end)
                    move_free_deferred[q_end] = q_start
                    log_debug(f"{header}. Deferred {q_end}")
                else:
                    obj, grasp, q_start, _traj, q_end = params
                    if not world.has_object(obj):
                        raise ValueError(f"{obj=} not found in world")
                    if grasp not in particles:
                        raise ValueError(f"{grasp=} should already be bound")
                    if q_start not in particles:
                        raise ValueError(f"{q_start=} should already be bound")
                    deferred_params.add(q_end)
                    log_debug(f"{header}. Deferred {q_end}")

            elif metadata.action_type == "pick":
                obj, grasp, q = params
                if grasp in particles:
                    raise ValueError(f"{grasp=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                steps.append(self._pick_step(plan_skeleton, idx, particles, n, pools, chosen, pool_q, placements))
                particles[q] = None  # ponytail: a placeholder, so later ops see q bound; _solve_steps writes it
                log_debug(f"{header}. {len(pools[grasp][1])} grasps pass the drops")
                deferred_params.discard(q)
                move_free_deferred.pop(q, None)

            elif metadata.action_type == "place":
                obj, grasp, placement, surface, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if placement in particles:
                    raise ValueError(f"{placement=} shouldn't already be bound")
                if not world.has_object(surface):
                    raise ValueError(f"{surface=} not found in world")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                steps.append(self._place_step(plan_skeleton, idx, particles, n, placements))
                placements.append(placement)
                particles[q] = None  # as at a pick
                log_debug(f"{header}. Drew {placement}")
                deferred_params.discard(q)
                move_free_deferred.pop(q, None)

            elif metadata.action_type == "retract":
                # RetractHolding: obj, grasp, q_start, traj, q_retract
                # RetractFree: q_start, traj, q_retract
                q_retract = params[-1]
                q_start_param = params[-3]
                if q_start_param not in particles:
                    raise ValueError(f"{q_start_param=} should already be bound")
                if q_retract in particles:
                    raise ValueError(f"{q_retract=} shouldn't already be bound")
                home = torch.tensor(world.model.home, device=device)
                particles[q_retract] = home.expand(n, -1).clone()
                log_debug(f"{header}. Initialized q_retract to the model's home")

            else:
                raise NotImplementedError(f"Unsupported operator {ground_op.operator.name}")

        # Every pick and place solves together, one gn_ik call per round (_solve_steps).
        solved = _solve_steps(world, particles, steps, n, 0 if chosen is None else GRASP_REDRAW_ROUNDS, self.ik_tol)
        confs = {q: (q_rows, ok) for q, (q_rows, ok, _) in solved.items()}
        for q, (_, _, accepted) in solved.items():
            log_debug(f"{q}: IK accepted {accepted[0]}/{n}, after redraws {accepted[1:]}")

        # Resolve trailing MoveFree configs by copying the source.
        unresolved = deferred_params & set(move_free_deferred.keys())
        for q_end in unresolved:
            q_start = move_free_deferred[q_end]
            particles[q_end] = particles[q_start].clone()
            deferred_params.remove(q_end)

        if deferred_params:
            raise RuntimeError(f"Deferred parameters not resolved: {deferred_params}")

        return particles, confs
