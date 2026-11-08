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
planner's only IK: balance first, clear of collision. Each pick row draws its grasp from the object's GraspGenX
grasps that the hand can take from the robot's side, and each place row its placement, with the yaw following the
held grasp; the rows whose IK fails draw again (design 7.8).
"""

import logging
import math
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

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
from cutamp.grasps import AZIMUTH_CONE, GRASP_REDRAW_ROUNDS, approach_in_cone, near_vertical
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


def _draw_and_solve(
    world: TAMPWorld, particles: Particles, name: str, num_particles: int,
    draw: Callable[[torch.Tensor], torch.Tensor],
    ik_inputs: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], arm: str, tol: IKTol,
) -> Tuple[torch.Tensor, List[int]]:
    """Draws ``particles[name]`` (``draw(rows)`` gives the rows' values) and solves the rows' confs with
    :func:`gn_ik` from ``ik_inputs(rows)``: their hand targets ``[r, 4, 4]`` and movables ``[r, M, 4]`` under the
    particles as they then stand. For up to ``GRASP_REDRAW_ROUNDS`` rounds, the rows that ``gn_ik`` failed draw again
    and only those rows are solved again. Returns the confs ``[num_particles, dof]`` and the accepted count after the
    first solve and after each round."""
    rows = torch.arange(num_particles, device=world.device)
    particles[name] = draw(rows)
    targets, movables = ik_inputs(rows)
    q, ok = gn_ik(world, targets, arm, movables, tol)
    accepted = [int(ok.sum())]
    for _ in range(GRASP_REDRAW_ROUNDS):
        rows = (~ok).nonzero().squeeze(-1)
        if rows.numel() == 0:
            break
        particles[name][rows] = draw(rows)
        targets, movables = ik_inputs(rows)
        q[rows], ok[rows] = gn_ik(world, targets, arm, movables, tol)
        accepted.append(int(ok.sum()))
    return q, accepted


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
        self.q_init = world.q_init.repeat(config.num_particles, 1)
        # A conf's IK is accepted on the tolerances the run's checker enforces.
        self.ik_tol = IKTol.from_checker(constraint_checker)

    def _initial_particles(self) -> Particles:
        return {
            "left_q0": self.q_init.clone(),
            "right_q0": self.q_init.clone(),
        }

    def __call__(self, plan_skeleton: PlanSkeleton, verbose: bool = True) -> Optional[Particles]:
        config = self.config
        num_particles = config.num_particles
        world = self.world
        device = world.device

        particles = self._initial_particles()
        deferred_params: set[str] = set()
        move_free_deferred: dict[str, str] = {}
        log_debug = _log.debug if verbose else lambda *args, **kwargs: None

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
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp in particles:
                    raise ValueError(f"{grasp=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                # The grasps this hand can take from the robot's side, at the object's start pose: init picks only
                # from start poses.
                world_from_obj = world.get_object_pose(obj)
                candidates = world.grasp_candidates(obj, arm)
                in_cone = approach_in_cone(world_from_obj @ grasp_7dof_to_mat4x4(candidates.grasps),
                                           world.stance_soles[:, :2, 3].mean(0), world_from_obj[:2, 3])
                if not in_cone.any():
                    below, touching = world.grasp_drops(obj, arm)
                    raise ValueError(
                        f"{header}: no grasp of {obj} is left for the {arm} hand. Of its "
                        f"{len(world.grasps[obj].grasps)} grasps, {below} reach below its bottom face and {touching} "
                        f"touch it (a grasp may do both), and the approach cone drops the {len(candidates.grasps)} "
                        f"others")
                grasps, scores = candidates.grasps[in_cone], candidates.scores[in_cone]
                tool_from_ee = world.tool_from_ee[world.tool_frame_for_arm(arm)]
                movables = movable_spheres_fn(world, plan_skeleton, idx)(particles)  # the grasp moves none of them
                particles[q], accepted = _draw_and_solve(
                    world, particles, grasp, num_particles,
                    lambda rows: grasps[torch.multinomial(scores, len(rows), replacement=True)],
                    lambda rows: (world_from_obj @ grasp_7dof_to_mat4x4(particles[grasp][rows]) @ tool_from_ee,
                                  movables[rows]),
                    arm, self.ik_tol,
                )
                log_debug(f"{header}. {len(grasps)} grasps pass the drops and the cone; IK accepted: "
                          f"{accepted[0]}/{num_particles}, after redraws {accepted[1:]}")
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

                obj_curobo, obj_spheres = world.get_object(obj), world.get_collision_spheres(obj)
                surface_curobo = world.get_object(surface)
                obj_from_G = grasp_7dof_to_mat4x4(particles[grasp])
                tool_from_ee = world.tool_from_ee[world.tool_frame_for_arm(arm)]
                # The placed object at its placement (particles[placement], redrawn per round), as the rollout has it.
                movables = movable_spheres_fn(world, plan_skeleton, idx)
                particles[q], accepted = _draw_and_solve(
                    world, particles, placement, num_particles,
                    lambda rows: _draw_placements(world, obj_curobo, obj_spheres, surface_curobo, obj_from_G[rows]),
                    lambda rows: (action_4dof_to_mat4x4(particles[placement][rows]) @ obj_from_G[rows] @ tool_from_ee,
                                  movables(particles)[rows]),
                    arm, self.ik_tol,
                )
                log_debug(f"{header}. IK accepted: {accepted[0]}/{num_particles}, after redraws {accepted[1:]}")
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
                particles[q_retract] = home.expand(num_particles, -1).clone()
                log_debug(f"{header}. Initialized q_retract to the model's home")

            else:
                raise NotImplementedError(f"Unsupported operator {ground_op.operator.name}")

        # Resolve trailing MoveFree configs by copying the source.
        unresolved = deferred_params & set(move_free_deferred.keys())
        for q_end in unresolved:
            q_start = move_free_deferred[q_end]
            particles[q_end] = particles[q_start].clone()
            deferred_params.remove(q_end)

        if deferred_params:
            raise RuntimeError(f"Deferred parameters not resolved: {deferred_params}")

        return particles
