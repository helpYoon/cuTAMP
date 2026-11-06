# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Particle initialization under cuRobo v0.8 single-MotionPlanner architecture.

Each conf parameter (``left_q0``, ``right_q1``, etc.) is a single vector in
the model's full cspace order (its 33 joints). A pick's and a place's conf come from :func:`gn_ik`, the
planner's only IK: balance first, clear of collision.
"""

import logging
from typing import NamedTuple, Optional, Sequence, Union

import roma
import torch

from curobo.scene import Cuboid
from curobo.types import JointState, Pose

from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.gn_refresh import (
    _row_layout,
    _row_tool_pose,
    collision_penetration,
    gn_project_balanced,
    soles_on_stance,
    tilt_within_bound,
)
from cutamp.rollout import movable_spheres_fn
from cutamp.samplers import (
    grasp_4dof_sampler,
    place_4dof_sampler,
)
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import KinematicConstraint, Motion
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    pose_list_to_mat4x4,
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
    # Clamp the perturbed seed to joint limits, which GN keeps every iterate inside. Limits come from the same
    # full-cspace kinematics that defines world.q_init's joint order
    # (JointLimits.joint_names == kinematics.joint_names, verified [2, full_dof]).
    # The left and right pitch limits are equal, so the clamp keeps the legs mirrored.
    _limits = world.kinematics.get_joint_limits().position  # [2, full_dof]
    _lower = _limits[0].to(seed.device, seed.dtype)
    _upper = _limits[1].to(seed.device, seed.dtype)
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
#: collision row, and the noise std (rad) of the stance-rooted starts that run beside the first (home, or the
#: stored conf), one start per value. The iterates are not motions, so only the conf they end at must be clear;
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
    (KinematicConstraint) and the Trunk tilt past the envelope's bound (Motion)."""
    pos: float
    rot: float
    sole_pos: float
    sole_rot: float
    tilt: float

    @classmethod
    def from_checker(cls, checker: ConstraintChecker) -> "IKTol":
        kc = KinematicConstraint.type
        return cls(*(checker._get_tol(kc, n) for n in ("pos_err", "rot_err", "sole_pos_err", "sole_rot_err")),
                   checker._get_tol(Motion.type, "trunk_tilt"))


def ik_check(world: TAMPWorld, q: torch.Tensor, world_from_ee: torch.Tensor, arms: Union[str, Sequence[str]],
             movables: torch.Tensor, tol: IKTol):
    """``(ok, safe, hand_err)``, each ``[N]``, from one FK of ``q`` ``[N, dof]`` on ``world.kinematics``, with rows
    in ``arms``' dep-major blocks (:func:`~cutamp.gn_refresh.gn_project_balanced`'s layout). ``safe``: the conf
    stands balanced (the CoM inside the support rectangle, both soles on the stance within ``tol``), clear of the
    scene, of ``movables`` ``[N, M, 4]`` and of itself (:func:`~cutamp.gn_refresh.collision_penetration` is 0)
    and inside the envelope's tilt bound. ``ok``: safe, with the hand on ``world_from_ee`` within ``tol``, all at
    the checker's ``<=``. ``hand_err``: the hand's position plus rotation error, cuRobo IK's ranking."""
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
            & (collision_penetration(world, state.robot_spheres, movables) <= 0)
            & tilt_within_bound(world, q, tol.tilt))
    return safe & (pos_err <= tol.pos) & (rot_err <= tol.rot), safe, pos_err + rot_err


def gn_ik(world: TAMPWorld, world_from_ee: torch.Tensor, arms: Union[str, Sequence[str]], movables: torch.Tensor,
          tol: IKTol, q_start: Optional[torch.Tensor] = None):
    """The planner's IK: :func:`~cutamp.gn_refresh.gn_project_balanced` of ``world_from_ee`` ``[N, 4, 4]``, rows
    in ``arms``' dep-major blocks (a bare ``str`` is one dep), from ``1 + len(GN_IK_START_SIGMAS)`` starts per row
    in one batched call, ``q_start`` (default ``world.q_init``) and :func:`_noisy_start`'s stance-rooted noise:
    ``GN_IK_ITERS`` iterations balance first, then ``GN_IK_CLEAR_ITERS`` that also clear the scene, ``movables``
    ``[N, M, 4]`` and the robot itself, and pull the CoM toward the keyframe rectangle's centre (GN's level 3).

    Returns ``(q [N, dof], ok [N])``: per row, the accepted start (:func:`ik_check`) closest to the target; where
    none is, the closest safe one, else the closest, a balanced-first iterate that the periodic refresh keeps
    projecting and whose pose error the kinematic cost hands to the placement. ``ok`` marks the accepted rows."""
    arms = (arms,) if isinstance(arms, str) else tuple(arms)
    n = world_from_ee.shape[0]
    batch = n // len(arms)
    first = world.q_init.expand(n, -1) if q_start is None else q_start
    starts = [first] + [torch.cat([_noisy_start(world, batch, arm, s) for arm in arms]) for s in GN_IK_START_SIGMAS]
    k = len(starts)
    targets, mov, rows_arms = world_from_ee.repeat(k, 1, 1), movables.repeat(k, 1, 1), arms * k
    q = gn_project_balanced(world, torch.cat(starts), targets, rows_arms, iters=GN_IK_ITERS, collision=False)
    q = gn_project_balanced(world, q, targets, rows_arms, iters=GN_IK_CLEAR_ITERS, movables=mov)
    ok, safe, err = ik_check(world, q, targets, rows_arms, mov, tol)
    # Tiers: accepted, then safe, then the rest, each by hand error (under 10: metres plus radians).
    best = (err + 100.0 * ~ok + 1e4 * ~safe).view(k, n).argmin(0) * n + torch.arange(n, device=q.device)
    return q[best], ok[best]


def _achieved_pose_errors(
    world: TAMPWorld, q_full: torch.Tensor, world_from_ee: torch.Tensor, arm: str,
):
    """``([B] pos_err, [B] rot_err)`` of ``q_full`` against ``world_from_ee``
    on ``arm``'s tool frame, in the constraint checker's own metric
    (``curobo_pose_error``).
    """
    from cutamp.costs import curobo_pose_error
    js = JointState.from_position(
        q_full, joint_names=list(world.kinematics.joint_names),
    )
    kin_state = world.kinematics.compute_kinematics(js)
    ee_pose = kin_state.tool_poses.get_link_pose(world.tool_frame_for_arm(arm))
    # get_link_pose returns strided views; get_matrix requires contiguous.
    ee = Pose(
        position=ee_pose.position.contiguous(),
        quaternion=ee_pose.quaternion.contiguous(),
    ).get_matrix()
    return curobo_pose_error(ee, world_from_ee)


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

                tool_frame = world.tool_frame_for_arm(arm)
                gripper_spheres = world.gripper_spheres[tool_frame]
                tool_from_ee = world.tool_from_ee[tool_frame]

                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                num_faces = 4 if isinstance(obj_curobo, Cuboid) else None

                sampled_grasps = grasp_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, num_faces=num_faces)
                obj_from_grasp = action_4dof_to_mat4x4(sampled_grasps)

                grasp_spheres = transform_spheres(gripper_spheres, obj_from_grasp)
                grasp_coll = sphere_to_sphere_overlap(obj_spheres, grasp_spheres)
                good_idxs = grasp_coll.topk(num_particles, largest=False).indices
                particles[grasp] = sampled_grasps[good_idxs]

                obj_from_grasp = obj_from_grasp[good_idxs]
                world_from_obj = pose_list_to_mat4x4(obj_curobo.pose).to(device)
                world_from_ee = world_from_obj @ obj_from_grasp @ tool_from_ee
                movables = movable_spheres_fn(world, plan_skeleton, idx)(particles)
                particles[q], ok = gn_ik(world, world_from_ee, arm, movables, self.ik_tol)
                log_debug(f"{header}. IK accepted: {int(ok.sum())}/{num_particles}")
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

                tool_frame = world.tool_frame_for_arm(arm)
                tool_from_ee = world.tool_from_ee[tool_frame]

                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                surface_curobo = world.get_object(surface)
                sampled_placements = place_4dof_sampler(
                    num_particles * 4, obj_curobo, obj_spheres, surface_curobo,
                )

                world_from_obj = action_4dof_to_mat4x4(sampled_placements)
                obj_place_spheres = transform_spheres(obj_spheres, world_from_obj)
                # WorldCollisionCost returns [B, H] summed across spheres.
                place_coll = world.collision_fn(obj_place_spheres[:, None].contiguous())[:, 0]
                best_idxs = place_coll.topk(num_particles, largest=False).indices
                sampled_placements = sampled_placements[best_idxs]
                world_from_obj = world_from_obj[best_idxs]

                particles[placement] = sampled_placements
                obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                world_from_ee = world_from_obj @ obj_from_grasp @ tool_from_ee
                # The placed object at its new placement (particles[placement], set above), as the rollout has it.
                movables = movable_spheres_fn(world, plan_skeleton, idx)(particles)
                particles[q], ok = gn_ik(world, world_from_ee, arm, movables, self.ik_tol)
                log_debug(f"{header}. IK accepted: {int(ok.sum())}/{num_particles}")
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
