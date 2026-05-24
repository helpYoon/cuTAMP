# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Particle initialization under cuRobo v0.8 single-MotionPlanner architecture.

Each conf parameter (``left_q0``, ``right_q1``, etc.) is a single 21-DOF
vector in T1's full cspace order.
"""

import logging
from typing import Optional

import roma
import torch

from curobo.scene import Cuboid
from curobo.types import GoalToolPose, JointState, Pose

from cutamp._curobo_internals import (
    inactive_arm_cspace_weights,
    restore_cspace_target_dof_weight,
    snapshot_cspace_target_dof_weight,
    write_cspace_target_dof_weight,
)
from cutamp.config import TAMPConfiguration
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.grasp_planning import _build_multi_frame_goal
from cutamp.samplers import (
    grasp_4dof_sampler,
    grasp_6dof_sampler,
    place_4dof_sampler,
    sample_yaw,
)
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    action_6dof_to_mat4x4,
    pose_list_to_mat4x4,
    sample_between_bounds,
    transform_spheres,
)
from cutamp.utils.shapes import MultiSphere

try:
    from cutamp.robots.t1 import t1_home
except ImportError:
    t1_home = None

_log = logging.getLogger(__name__)


def _ik_for_pose(
    world: TAMPWorld,
    world_from_ee: torch.Tensor,
    arm: Optional[str],
    *,
    current_state_q: Optional[torch.Tensor] = None,
    num_seeds: Optional[int] = None,
):
    """Solve IK for a batch of EE poses, targeting the active arm's tool frame.

    Uses the same multi-frame-goal + cspace-pin pattern the motion planner
    applies during arm operators: the inactive frame is targeted at its
    current FK pose, and its joints are pinned via ``cspace_target_dof_weight``.
    Without this, IK leaves the inactive arm unconstrained and it drifts to
    arbitrary joint values, polluting downstream particle optimization.

    ``current_state_q`` (full-cspace ``[B, len(world.kinematics.joint_names)]``)
    seeds IK from a non-home pose — used by the coupled-reIK refresh during
    Adam to warm-start from the current best particle's q. When ``None``,
    seeds from ``world.q_init`` as before.

    ``num_seeds`` overrides the IK solver's seed count. Set to 1 for warm-start
    refresh (~10-20x faster than the default multi-seed); leave ``None`` for
    initial IK from home (which needs the default for reliability).
    """
    ik = world.ik_solver
    active_frame = world.tool_frame_for_arm(arm)
    if active_frame not in ik.kinematics.tool_frames:
        raise RuntimeError(
            f"IK solver does not expose tool frame {active_frame!r}; "
            f"available: {ik.kinematics.tool_frames}"
        )
    target_pose = Pose.from_matrix(world_from_ee)
    batch_size = world_from_ee.shape[0]
    full_names = list(world.kinematics.joint_names)
    if current_state_q is None:
        seed_full = world.q_init.unsqueeze(0).expand(batch_size, -1)
    else:
        # Caller passes full-cspace q; trust their batch shape.
        seed_full = current_state_q
    full_js = JointState.from_position(seed_full, joint_names=full_names)
    current_state = ik.kinematics.get_active_js(full_js)
    goal = _build_multi_frame_goal(ik, active_frame, target_pose, current_state)

    snapshot = snapshot_cspace_target_dof_weight([ik])
    try:
        if arm is not None:
            write_cspace_target_dof_weight(
                [ik], inactive_arm_cspace_weights(ik, arm),
            )
        solve_kwargs = {"goal_tool_poses": goal, "current_state": current_state}
        if num_seeds is not None:
            solve_kwargs["num_seeds"] = num_seeds
        return ik.solve_pose(**solve_kwargs)
    finally:
        restore_cspace_target_dof_weight([ik], snapshot)


def _ik_solution_to_full_q(ik_result, world: TAMPWorld) -> torch.Tensor:
    """Build a full-cspace ``[B, len(world.kinematics.joint_names)]`` tensor
    from an IK result. Locked-out joints (e.g. mobile base) are filled from
    the IK kinematics' own ``lock_jointstate`` — same source the planner uses,
    so values stay consistent without an external ``t1_home`` reference.
    """
    ik = world.ik_solver
    sol_js = JointState.from_position(
        ik_result.solution[:, 0], joint_names=list(ik.kinematics.joint_names),
    )
    return ik.kinematics.get_full_js(sol_js).reorder(list(world.kinematics.joint_names)).position


def _store_ik_q(particles: dict, q_name: str, ik_result, world: TAMPWorld, arm: Optional[str]) -> None:
    """Store the IK result's q under ``q_name``, reordered to the full kin's joint order."""
    particles[q_name] = _ik_solution_to_full_q(ik_result, world)


class ParticleInitializer:
    def __init__(self, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectory initialization not yet supported")
        if config.place_dof != 4:
            raise NotImplementedError(f"Only 4-DOF placement supported, not {config.place_dof}")
        if config.grasp_dof not in (4, 6):
            raise NotImplementedError(f"Only 4-DOF or 6-DOF grasp supported, not {config.grasp_dof}")

        self.world = world
        self.config = config
        self.q_init = world.q_init.repeat(config.num_particles, 1)

        self.pick_cache = {}
        self.place_cache = {}
        self.push_button_cache = {}
        self.push_stick_cache = {}
        self.failed_push = set()

    def _initial_particles(self, num_particles: int) -> Particles:
        return {
            "left_q0": self.q_init.clone(),
            "right_q0": self.q_init.clone(),
        }

    def _apply_cached(
        self, cache_entry: dict, sampled: dict, q: str, arm: Optional[str],
        particles: dict, deferred_params: set, move_free_deferred: dict,
        grasp_check: Optional[tuple] = None,
        log_msg: Optional[str] = None, log_debug=None,
    ) -> None:
        if grasp_check is not None:
            grasp_name, err_msg = grasp_check
            if not (particles[grasp_name] == cache_entry["grasp"]).all():
                raise RuntimeError(err_msg)
        for particle_name, cache_field in sampled.items():
            particles[particle_name] = cache_entry[cache_field].clone()
        ik_result = cache_entry["ik_result"]
        _store_ik_q(particles, q, ik_result, self.world, arm)
        deferred_params.discard(q)
        move_free_deferred.pop(q, None)
        if log_msg is not None and log_debug is not None:
            log_debug(f"{log_msg} {ik_result.success.sum()}/{self.config.num_particles} success")

    def __call__(self, plan_skeleton: PlanSkeleton, verbose: bool = True) -> Optional[Particles]:
        config = self.config
        num_particles = config.num_particles
        world = self.world
        device = world.device

        particles = self._initial_particles(num_particles)
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
                joint_limits = world.joint_limits
                tool_from_ee = world.tool_from_ee[tool_frame]

                cache_key = (obj, arm) if arm else obj
                if cache_key in self.pick_cache:
                    self._apply_cached(
                        self.pick_cache[cache_key],
                        sampled={grasp: "sampled_grasps"},
                        q=q, arm=arm, particles=particles,
                        deferred_params=deferred_params,
                        move_free_deferred=move_free_deferred,
                        log_msg=f"{header}. Using cached grasp poses for {obj}.",
                        log_debug=log_debug,
                    )
                    continue

                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                num_faces = 4 if isinstance(obj_curobo, Cuboid) else None

                if config.grasp_dof == 4:
                    sampled_grasps = grasp_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, num_faces=num_faces)
                    obj_from_grasp = action_4dof_to_mat4x4(sampled_grasps)
                else:
                    sampled_grasps = grasp_6dof_sampler(num_particles * 4, obj_curobo, num_faces=num_faces)
                    obj_from_grasp = action_6dof_to_mat4x4(sampled_grasps)

                grasp_spheres = transform_spheres(gripper_spheres, obj_from_grasp)
                grasp_coll = sphere_to_sphere_overlap(obj_spheres, grasp_spheres)
                good_idxs = grasp_coll.topk(num_particles, largest=False).indices
                particles[grasp] = sampled_grasps[good_idxs]

                if config.random_init:
                    particles[q] = sample_between_bounds(num_particles, joint_limits)
                else:
                    obj_from_grasp = obj_from_grasp[good_idxs]
                    world_from_obj = pose_list_to_mat4x4(obj_curobo.pose).to(device)
                    world_from_ee = world_from_obj @ obj_from_grasp @ tool_from_ee
                    ik_result = _ik_for_pose(world, world_from_ee, arm)
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}"
                    )
                    _store_ik_q(particles, q, ik_result, world, arm)
                deferred_params.discard(q)
                move_free_deferred.pop(q, None)

                if config.cache_subgraphs and not config.random_init:
                    self.pick_cache[cache_key] = {
                        "sampled_grasps": particles[grasp], "ik_result": ik_result,
                    }

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
                joint_limits = world.joint_limits
                tool_from_ee = world.tool_from_ee[tool_frame]

                cache_key = (obj, surface, arm) if arm else (obj, surface)
                if cache_key in self.place_cache:
                    self._apply_cached(
                        self.place_cache[cache_key],
                        sampled={placement: "sampled_placements"},
                        q=q, arm=arm, particles=particles,
                        deferred_params=deferred_params,
                        move_free_deferred=move_free_deferred,
                        grasp_check=(grasp, f"Grasps don't match for {obj} on {surface}"),
                        log_msg=f"{header}. Using cached placement for {obj}.",
                        log_debug=log_debug,
                    )
                    continue

                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                if config.random_init:
                    yaw = sample_yaw(num_particles * 4, None, device)
                    aabb = world.world_aabb.clone()
                    aabb[0, 2] = 0.0
                    aabb[1, 2] = max(aabb[1, 2], 0.2)
                    xyz = sample_between_bounds(num_particles * 4, aabb)
                    sampled_placements = torch.cat([xyz, yaw.unsqueeze(-1)], dim=1)
                else:
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
                if config.random_init:
                    particles[q] = sample_between_bounds(num_particles, joint_limits)
                else:
                    if config.grasp_dof == 4:
                        obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                    else:
                        obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                    world_from_ee = world_from_obj @ obj_from_grasp @ tool_from_ee
                    ik_result = _ik_for_pose(world, world_from_ee, arm)
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}"
                    )
                    _store_ik_q(particles, q, ik_result, world, arm)
                deferred_params.discard(q)
                move_free_deferred.pop(q, None)

                if config.cache_subgraphs and not config.random_init:
                    self.place_cache[cache_key] = {
                        "sampled_placements": sampled_placements,
                        "ik_result": ik_result,
                        "grasp": particles[grasp],
                    }

            elif metadata.action_type == "push":
                button, push_pose, q = params
                assert not config.random_init, "Random initialization not supported for pushing"
                if not world.has_object(button):
                    raise ValueError(f"{button=} not found in world")
                if push_pose in particles:
                    raise ValueError(f"{push_pose=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                tool_frame = world.tool_frame_for_arm(arm)
                tool_from_ee = world.tool_from_ee[tool_frame]

                cache_key = (button, arm) if arm else button
                if cache_key in self.failed_push and config.skip_failed_subgraphs:
                    return None
                if cache_key in self.push_button_cache:
                    self._apply_cached(
                        self.push_button_cache[cache_key],
                        sampled={push_pose: "sampled_push"},
                        q=q, arm=arm, particles=particles,
                        deferred_params=deferred_params,
                        move_free_deferred=move_free_deferred,
                    )
                    continue

                aabb = world.get_aabb(button).clone()
                surface_z = aabb[1, 2]
                lower_xy, upper_xy = aabb[:, :2]
                lower_xy += 0.01
                upper_xy -= 0.01
                sampled_xy = lower_xy + torch.rand(num_particles, 2, device=device) * (upper_xy - lower_xy)
                sampled_z = surface_z.expand(num_particles) + 0.02 + world.collision_activation_distance
                sampled_yaw = sample_yaw(num_particles, num_faces=None, device=device)
                sampled_push = torch.cat([sampled_xy, sampled_z[:, None], sampled_yaw[:, None]], dim=1)
                particles[push_pose] = sampled_push

                world_from_push = action_4dof_to_mat4x4(sampled_push)
                world_from_ee = world_from_push @ tool_from_ee
                ik_result = _ik_for_pose(world, world_from_ee, arm)
                log_debug(f"{header}. IK success: {ik_result.success.sum()}/{num_particles}")
                _store_ik_q(particles, q, ik_result, world, arm)
                deferred_params.discard(q)
                move_free_deferred.pop(q, None)

                if not ik_result.success.any():
                    self.failed_push.add(cache_key)
                if config.cache_subgraphs:
                    self.push_button_cache[cache_key] = {
                        "sampled_push": sampled_push, "ik_result": ik_result,
                    }

            elif metadata.action_type == "push_stick":
                button, stick_name, grasp, push_pose, q = params
                assert not config.random_init, "Random initialization not supported for pushing"
                if not world.has_object(button):
                    raise ValueError(f"{button=} not found in world")
                if not world.has_object(stick_name):
                    raise ValueError(f"{stick_name=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if push_pose in particles:
                    raise ValueError(f"{push_pose=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                tool_frame = world.tool_frame_for_arm(arm)
                tool_from_ee = world.tool_from_ee[tool_frame]

                cache_key = (button, stick_name, arm) if arm else (button, stick_name)
                if cache_key in self.push_stick_cache:
                    self._apply_cached(
                        self.push_stick_cache[cache_key],
                        sampled={push_pose: "sampled_push"},
                        q=q, arm=arm, particles=particles,
                        deferred_params=deferred_params,
                        move_free_deferred=move_free_deferred,
                        grasp_check=(grasp, f"Grasps don't match for {button}/{stick_name}"),
                    )
                    continue

                aabb = world.get_aabb(button).clone()
                surface_z = aabb[1, 2]
                lower_xy, upper_xy = aabb[:, :2]
                lower_xy += 0.01
                upper_xy -= 0.01
                sampled_xy = lower_xy + torch.rand(num_particles, 2, device=device) * (upper_xy - lower_xy)
                sampled_z = surface_z.expand(num_particles) + 0.02 + world.collision_activation_distance
                sampled_yaw = sample_yaw(num_particles, num_faces=None, device=device)
                sampled_push = torch.cat([sampled_xy, sampled_z[:, None], sampled_yaw[:, None]], dim=1)

                stick: MultiSphere = world.get_object("stick")
                spheres = stick.spheres
                if not (spheres[:, 1:3] == 0.0).all():
                    raise ValueError("Expected stick spheres to have y, z = 0")
                sphere_x = spheres[:, 0]
                x_idxs = torch.randint(0, len(sphere_x), (num_particles,), device=spheres.device)
                sampled_x = sphere_x[x_idxs]
                stick_from_tip = torch.eye(4, device=device).repeat(num_particles, 1, 1)
                stick_from_tip[:, 0, 3] = -sampled_x

                world_from_push = action_4dof_to_mat4x4(sampled_push)
                world_from_stick = world_from_push @ stick_from_tip
                rpy = roma.rotmat_to_euler("XYZ", world_from_stick[:, :3, :3])
                assert (rpy[:, :2] == 0.0).all(), "stick roll/pitch should be 0"
                pos = world_from_stick[:, :3, 3]
                yaw = rpy[:, 2]
                particles[push_pose] = torch.cat([pos, yaw[:, None]], dim=1)

                if config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                world_from_ee = world_from_stick @ obj_from_grasp @ tool_from_ee
                ik_result = _ik_for_pose(world, world_from_ee, arm)
                log_debug(f"{header}. IK success: {ik_result.success.sum()}/{num_particles}")
                _store_ik_q(particles, q, ik_result, world, arm)
                deferred_params.discard(q)
                move_free_deferred.pop(q, None)

                if config.cache_subgraphs:
                    self.push_stick_cache[cache_key] = {
                        "sampled_push": sampled_push,
                        "ik_result": ik_result,
                        "grasp": particles[grasp],
                    }

            elif metadata.action_type == "retract":
                # RetractHolding: obj, grasp, q_start, traj, q_retract
                # RetractFree: q_start, traj, q_retract
                q_retract = params[-1]
                q_start_param = params[-3]
                if q_start_param not in particles:
                    raise ValueError(f"{q_start_param=} should already be bound")
                if q_retract in particles:
                    raise ValueError(f"{q_retract=} shouldn't already be bound")
                home = torch.tensor(t1_home, device=device)
                particles[q_retract] = home.expand(num_particles, -1).clone()
                log_debug(f"{header}. Initialized q_retract to t1_home")

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
