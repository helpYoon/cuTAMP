# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from typing import Literal, Optional

import roma
import torch
from curobo.geom.types import Cuboid
from curobo.types.math import Pose

from cutamp.config import TAMPConfiguration
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.samplers import (
    grasp_4dof_sampler,
    grasp_6dof_sampler,
    place_4dof_sampler,
    sample_yaw,
)
from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place, Push, PushStick
from cutamp.t1_domain import (
    LeftMoveFree, LeftMoveHolding, LeftPick, LeftPlace, LeftPush, LeftPushStick, 
    RightMoveFree, RightMoveHolding, RightPick, RightPlace, RightPush, RightPushStick,
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

_log = logging.getLogger(__name__)

# Number of shared joints (lift + torso) for T1 dual-arm robot
NUM_SHARED_JOINTS = 4


def get_arm_from_operator(op_name: str) -> Optional[Literal["left", "right"]]:
    """Extract arm identifier from T1 operator name. Returns None for single-arm operators."""
    if op_name.startswith("Left"):
        return "left"
    elif op_name.startswith("Right"):
        return "right"
    return None


def propagate_shared_joints(
    particles: Particles, 
    active_arm: Literal["left", "right"], 
    solved_q: torch.Tensor,
    solved_configs: set = None,
) -> None:
    """
    Propagate shared joints from active arm's IK solution to unsolved inactive arm's configurations.
    
    T1 robot has 4 shared joints (lift + torso) that appear in both arms' 11-DOF configs.
    When IK solves for one arm, we update ALL of the other arm's configurations to match,
    ensuring shared joint consistency throughout the plan.
    
    Args:
        particles: Current particle dictionary containing configurations like 
                   left_q0, left_q1, right_q0, right_q1, etc.
        active_arm: Which arm just solved IK ("left" or "right")
        solved_q: The IK solution for the active arm, shape (num_particles, 11)
        solved_configs: Set of configuration names that have already been solved by IK
                        which will not be updated.
    """
    if solved_configs is None:
        solved_configs = set()
    
    # Get the shared joint values from the active arm's solution
    shared_joints = solved_q[:, :NUM_SHARED_JOINTS]  # First 4 joints are shared
    
    # Determine the prefix for the inactive arm's configurations
    inactive_prefix = "right_q" if active_arm == "left" else "left_q"
    
    # Update unsolved configurations belonging to the inactive arm
    for key in particles.keys():
        if key.startswith(inactive_prefix) and key not in solved_configs:
            # Clone to avoid modifying in-place
            inactive_q = particles[key].clone()
            # Update shared joints while keeping arm-specific joints unchanged
            inactive_q[:, :NUM_SHARED_JOINTS] = shared_joints
            particles[key] = inactive_q


class ParticleInitializer:
    def __init__(self, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectory initialization not yet supported")
        if config.place_dof != 4:
            raise NotImplementedError(f"Only 4-DOF grasp and placement supported for now, not {config.place_dof}")
        if config.grasp_dof != 4 and config.grasp_dof != 6:
            raise NotImplementedError(f"Only 4-DOF or 6-DOF grasp supported for now, not {config.grasp_dof}")
        self.world = world
        self.config = config
        if world.is_dual_arm:
            self.left_q_init = world.left_q_init.repeat(config.num_particles, 1)
            self.right_q_init = world.right_q_init.repeat(config.num_particles, 1)
        else:
            self.q_init = world.q_init.repeat(config.num_particles, 1)

        # Sampler caching
        self.pick_cache = {}
        self.place_cache = {}
        self.push_button_cache = {}
        self.push_stick_cache = {}
        self.failed_push = set()

    def __call__(self, plan_skeleton: PlanSkeleton, verbose: bool = True) -> Optional[Particles]:
        config = self.config
        num_particles = self.config.num_particles
        world = self.world
        if world.is_dual_arm:
            particles = {"left_q0": self.left_q_init.clone(), "right_q0": self.right_q_init.clone()}
        else:
            particles = {"q0": self.q_init.clone()}
        deferred_params = set()
        # move_free at the end of plan skeleton don't need IK resolution
        move_free_deferred: dict[str, str] = {}
        # track conf solved by IK
        solved_configs = set()
        log_debug = _log.debug if verbose else lambda *args, **kwargs: None

        # Note: we don't consider state after executing earlier samples
        # Iterate through each ground operator in the plan skeleton and initialize and build up particles
        for idx, ground_op in enumerate(plan_skeleton):
            op_name = ground_op.operator.name
            params = ground_op.values
            header = f"{idx + 1}. {ground_op}"

            # MoveFree (single-arm and dual-arm)
            if op_name in (MoveFree.name, LeftMoveFree.name, RightMoveFree.name):
                q_start, _traj, q_end = params
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                move_free_deferred[q_end] = q_start
                log_debug(f"{header}. Deferred {q_end}")

            # MoveHolding (single-arm and dual-arm)
            elif op_name in (MoveHolding.name, LeftMoveHolding.name, RightMoveHolding.name):
                obj, grasp, q_start, _traj, q_end = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # Pick (single-arm and dual-arm)
            elif op_name in (Pick.name, LeftPick.name, RightPick.name):
                obj, grasp, q = params
                arm = get_arm_from_operator(op_name)
                
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp in particles:
                    raise ValueError(f"{grasp=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                # Get arm-specific resources
                gripper_spheres = world.get_gripper_spheres(arm) if arm else world.robot_container.gripper_spheres
                joint_limits = world.get_joint_limits(arm) if arm else world.robot_container.joint_limits
                tool_from_ee = world.get_tool_from_ee(arm) if arm else world.tool_from_ee
                ik_solver = world.get_ik_solver(arm) if arm else world.ik_solver

                # Note: pick cache currently assumes object is at same pose as when sampled
                # For dual-arm, include arm in cache key
                cache_key = (obj, arm) if arm else obj
                if cache_key in self.pick_cache:
                    # important, we need to clone here
                    particles[grasp] = self.pick_cache[cache_key]["sampled_grasps"].clone()
                    ik_result = self.pick_cache[cache_key]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    solved_configs.add(q)
                    deferred_params.remove(q)
                    # clean up deferred params
                    move_free_deferred.pop(q, None)
                    # Propagate shared joints for dual-arm
                    if arm:
                        propagate_shared_joints(particles, arm, particles[q], solved_configs)
                    log_debug(
                        f"{header}. Using cached grasp poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample grasps
                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                num_faces = 4 if isinstance(obj_curobo, Cuboid) else None

                # Sample 4 times as many grasps as particles
                if config.grasp_dof == 4:
                    sampled_grasps = grasp_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, num_faces=num_faces)
                    obj_from_grasp = action_4dof_to_mat4x4(sampled_grasps)
                else:
                    sampled_grasps = grasp_6dof_sampler(num_particles * 4, obj_curobo, num_faces=num_faces)
                    obj_from_grasp = action_6dof_to_mat4x4(sampled_grasps)

                # Select the grasps that are not in collision with the object
                grasp_spheres = transform_spheres(gripper_spheres, obj_from_grasp)
                grasp_coll = sphere_to_sphere_overlap(obj_spheres, grasp_spheres)
                good_idxs = grasp_coll.topk(num_particles, largest=False).indices
                particles[grasp] = sampled_grasps[good_idxs]

                # Transform grasps to hand frame
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, joint_limits)
                    particles[q] = q_sample
                else:
                    obj_from_grasp = obj_from_grasp[good_idxs]
                    world_from_obj = pose_list_to_mat4x4(obj_curobo.pose).to(world.tensor_args.device)
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ tool_from_ee

                    # Solve IK with cuRobo
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    ik_result = ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = ik_result.solution[:, 0]
                solved_configs.add(q)
                deferred_params.remove(q)
                move_free_deferred.pop(q, None)

                # Propagate shared joints for dual-arm
                if arm:
                    propagate_shared_joints(particles, arm, particles[q], solved_configs)

                # Store in cache
                if config.cache_subgraphs:
                    self.pick_cache[cache_key] = {"sampled_grasps": particles[grasp], "ik_result": ik_result}

            # Place (single-arm and dual-arm)
            elif op_name in (Place.name, LeftPlace.name, RightPlace.name):
                obj, grasp, placement, surface, q = params
                arm = get_arm_from_operator(op_name)
                
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

                # Get arm-specific resources
                joint_limits = world.get_joint_limits(arm) if arm else world.robot_container.joint_limits
                tool_from_ee = world.get_tool_from_ee(arm) if arm else world.tool_from_ee
                ik_solver = world.get_ik_solver(arm) if arm else world.ik_solver

                # For dual-arm, include arm in cache key
                cache_key = (obj, surface, arm) if arm else (obj, surface)
                if cache_key in self.place_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.place_cache[cache_key]["grasp"]
                    if not (actual_grasp == cached_grasp).all():
                        raise RuntimeError(f"Grasps don't match for {obj} on {surface}")

                    # important, we need to clone here
                    sampled_placements = self.place_cache[cache_key]["sampled_placements"].clone()
                    particles[placement] = sampled_placements
                    ik_result = self.place_cache[cache_key]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    solved_configs.add(q)
                    deferred_params.remove(q)
                    # clean up deferred params
                    move_free_deferred.pop(q, None)
                    # Propagate shared joints for dual-arm
                    if arm:
                        propagate_shared_joints(particles, arm, particles[q], solved_configs)
                    log_debug(
                        f"{header}. Using cached placement poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample placements pose of object (in world frame)
                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                if config.random_init:
                    yaw = sample_yaw(num_particles * 4, None, self.world.tensor_args.device)
                    aabb = world.world_aabb.clone()
                    aabb[0, 2] = 0.0
                    aabb[1, 2] = max(aabb[1, 2], 0.2)
                    xyz = sample_between_bounds(num_particles * 4, aabb)
                    sampled_placements = torch.cat([xyz, yaw.unsqueeze(-1)], dim=1)
                else:
                    surface_curobo = world.get_object(surface)
                    sampled_placements = place_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, surface_curobo)

                # Select the placements that are not in collision with the object
                world_from_obj = action_4dof_to_mat4x4(sampled_placements)  # desired placement pose
                obj_place_spheres = transform_spheres(obj_spheres, world_from_obj)
                place_coll = world.collision_fn(obj_place_spheres[:, None].contiguous())[:, 0]
                best_idxs = place_coll.topk(num_particles, largest=False).indices
                sampled_placements = sampled_placements[best_idxs]
                world_from_obj = world_from_obj[best_idxs]

                # Set particles and then solve for robot configurations
                particles[placement] = sampled_placements
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, joint_limits)
                    particles[q] = q_sample
                else:
                    # Get the hand pose given the placement pose in world frame.
                    # Need to take grasp into account to transform into hand frame.
                    if config.grasp_dof == 4:
                        obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                    else:
                        obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ tool_from_ee

                    # Solve IK
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    ik_result = ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding?
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = ik_result.solution[:, 0]
                solved_configs.add(q)
                deferred_params.remove(q)
                move_free_deferred.pop(q, None)
                # Propagate shared joints for dual-arm
                if arm:
                    propagate_shared_joints(particles, arm, particles[q], solved_configs)

                # Store in cache
                if config.cache_subgraphs:
                    self.place_cache[cache_key] = {
                        "sampled_placements": sampled_placements,
                        "ik_result": ik_result,
                        "grasp": particles[grasp],
                    }

            # Push Button (without stick) - single-arm and dual-arm
            elif op_name in (Push.name, LeftPush.name, RightPush.name):
                button, push_pose, q = params
                arm = get_arm_from_operator(op_name)
                
                assert not config.random_init, "Random initialization not supported for pushing"
                if not world.has_object(button):
                    raise ValueError(f"{button=} not found in world")
                if push_pose in particles:
                    raise ValueError(f"{push_pose=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                # Get arm-specific resources
                tool_from_ee = world.get_tool_from_ee(arm) if arm else world.tool_from_ee
                ik_solver = world.get_ik_solver(arm) if arm else world.ik_solver

                # For dual-arm, include arm in cache key
                cache_key = (button, arm) if arm else button
                failed_key = (button, arm) if arm else button
                
                # Pruning failed subgraphs (i.e., we couldn't push this button at all)
                if failed_key in self.failed_push and config.skip_failed_subgraphs:
                    return None

                if cache_key in self.push_button_cache:
                    # important, we need to clone here
                    sampled_push = self.push_button_cache[cache_key]["sampled_push"].clone()
                    particles[push_pose] = sampled_push
                    ik_result = self.push_button_cache[cache_key]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    solved_configs.add(q)
                    deferred_params.remove(q)
                    move_free_deferred.pop(q, None)
                    # Propagate shared joints for dual-arm
                    if arm:
                        propagate_shared_joints(particles, arm, particles[q], solved_configs)
                    log_debug(
                        f"{header}. Using cached push poses for {button}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample 4-DOF push poses for the button
                aabb = world.get_aabb(button).clone()
                surface_z = aabb[1, 2]  # top of the button aabb
                # add 1cm buffer
                lower_xy, upper_xy = aabb[:, :2]
                lower_xy += 0.01
                upper_xy -= 0.01

                sampled_xy = lower_xy + torch.rand(num_particles, 2, device=world.tensor_args.device) * (
                    upper_xy - lower_xy
                )
                sampled_z = (
                    surface_z.expand(num_particles) + 0.02 + world.collision_activation_distance
                )  # 2cm above button for now
                sampled_yaw = sample_yaw(num_particles, num_faces=None, device=world.tensor_args.device)
                sampled_push = torch.cat([sampled_xy, sampled_z[:, None], sampled_yaw[:, None]], dim=1)
                particles[push_pose] = sampled_push

                # Transform from tool to hand frame
                world_from_push = action_4dof_to_mat4x4(sampled_push)
                world_from_ee = world_from_push @ tool_from_ee

                # Solve IK with cuRobo
                world_from_ee = Pose.from_matrix(world_from_ee)
                ik_result = ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding
                log_debug(
                    f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                )
                particles[q] = ik_result.solution[:, 0]
                solved_configs.add(q)
                deferred_params.remove(q)
                move_free_deferred.pop(q, None)
                # Propagate shared joints for dual-arm
                if arm:
                    propagate_shared_joints(particles, arm, particles[q], solved_configs)

                # Failed subgraph!
                if not ik_result.success.any():
                    self.failed_push.add(failed_key)

                # Cache the push poses
                if config.cache_subgraphs:
                    self.push_button_cache[cache_key] = {"sampled_push": sampled_push, "ik_result": ik_result}

            # Push Button with Stick - single-arm and dual-arm
            elif op_name in (PushStick.name, LeftPushStick.name, RightPushStick.name):
                button, stick_name, grasp, push_pose, q = params
                arm = get_arm_from_operator(op_name)
                
                assert not config.random_init, "Random initialization not supported for pushing"
                if not world.has_object(button):
                    raise ValueError(f"{button=} not found in world")
                if not world.has_object(stick_name):
                    raise ValueError(f"{stick_name=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be binded")
                if push_pose in particles:
                    raise ValueError(f"{push_pose=} shouldn't already be binded")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be binded")

                # Get arm-specific resources
                tool_from_ee = world.get_tool_from_ee(arm) if arm else world.tool_from_ee
                ik_solver = world.get_ik_solver(arm) if arm else world.ik_solver

                # For dual-arm, include arm in cache key
                cache_key = (button, stick_name, arm) if arm else (button, stick_name)
                if cache_key in self.push_stick_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.push_stick_cache[cache_key]["grasp"]
                    if not (actual_grasp == cached_grasp).all():
                        raise RuntimeError(f"Grasps don't match for {button} with {stick_name}")

                    # important, we need to clone here
                    sampled_push = self.push_stick_cache[cache_key]["sampled_push"].clone()
                    particles[push_pose] = sampled_push
                    ik_result = self.push_stick_cache[cache_key]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    solved_configs.add(q)
                    deferred_params.remove(q)
                    move_free_deferred.pop(q, None)
                    # Propagate shared joints for dual-arm
                    if arm:
                        propagate_shared_joints(particles, arm, particles[q], solved_configs)

                    log_debug(
                        f"{header}. Using cached push for {button} with {stick_name}. "
                        f"{ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample pushes for the button, this will be in the stick frame
                aabb = world.get_aabb(button).clone()
                surface_z = aabb[1, 2]  # top of the button aabb
                # add 1cm buffer
                lower_xy, upper_xy = aabb[:, :2]
                lower_xy += 0.01
                upper_xy -= 0.01

                sampled_xy = lower_xy + torch.rand(num_particles, 2, device=world.tensor_args.device) * (
                    upper_xy - lower_xy
                )
                sampled_z = (
                    surface_z.expand(num_particles) + 0.02 + world.collision_activation_distance
                )  # 2cm above button for now
                sampled_yaw = sample_yaw(num_particles, num_faces=None, device=world.tensor_args.device)
                sampled_push = torch.cat([sampled_xy, sampled_z[:, None], sampled_yaw[:, None]], dim=1)

                # Sample somewhere along the stick for the push
                stick: MultiSphere = world.get_object("stick")
                spheres = stick.spheres
                if not (spheres[:, 1:3] == 0.0).all():
                    raise ValueError("Expected stick spheres to have y and z positions of 0")
                sphere_x = spheres[:, 0]
                x_idxs = torch.randint(0, len(sphere_x), (num_particles,), device=spheres.device)
                sampled_x = sphere_x[x_idxs]

                stick_from_tip = torch.eye(4, device=world.tensor_args.device).repeat(num_particles, 1, 1)
                stick_from_tip[:, 0, 3] = -sampled_x

                # Where we are pushing the button with the stick - i.e., the pose of the stick
                world_from_push = action_4dof_to_mat4x4(sampled_push)
                world_from_stick = world_from_push @ stick_from_tip

                # Push pose is pose of stick in world frame
                rpy = roma.rotmat_to_euler("XYZ", world_from_stick[:, :3, :3])
                assert (rpy[:, :2] == 0.0).all(), "roll and pitch should be 0"
                pos = world_from_stick[:, :3, 3]
                yaw = rpy[:, 2]
                action_4dof = torch.cat([pos, yaw[:, None]], dim=1)
                particles[push_pose] = action_4dof

                # Convert to tool frame
                if config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                world_from_grasp = world_from_stick @ obj_from_grasp
                world_from_ee = world_from_grasp @ tool_from_ee

                # Solve IK with cuRobo
                world_from_ee = Pose.from_matrix(world_from_ee)
                ik_result = ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding
                log_debug(
                    f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                )
                particles[q] = ik_result.solution[:, 0]
                solved_configs.add(q)
                deferred_params.remove(q)
                move_free_deferred.pop(q, None)
                # Propagate shared joints for dual-arm
                if arm:
                    propagate_shared_joints(particles, arm, particles[q], solved_configs)

                # Store in cache
                if config.cache_subgraphs:
                    self.push_stick_cache[cache_key] = {
                        "sampled_push": sampled_push,
                        "ik_result": ik_result,
                        "grasp": particles[grasp],
                    }

            # Unknown
            else:
                raise NotImplementedError(f"Unsupported operator {op_name}")
        
        # Handle unresolved deferred parameters
        # MoveFree params that weren't resolved are trailing moves (no Pick/Place after them)
        # We can use the source config as a fallback since we don't need IK for trailing moves
        unresolved_move_free = deferred_params & set(move_free_deferred.keys())
        for q_end in unresolved_move_free:
            q_start = move_free_deferred[q_end]
            particles[q_end] = particles[q_start].clone()
            deferred_params.remove(q_end)
            log_debug(f"Resolved trailing MoveFree config {q_end} using {q_start}")
            
        # There should not be any deferred parameters left
        if deferred_params:
            raise RuntimeError(f"Deferred parameters not resolved: {deferred_params}")

        return particles
