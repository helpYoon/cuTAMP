# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Solving motions with cuRobo."""

import logging
from typing import List, Optional

import torch
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Sphere
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenPlanConfig

from cutamp.utils.common import Particles, action_6dof_to_mat4x4, action_4dof_to_mat4x4
from cutamp.config import TAMPConfiguration
from cutamp.dual_arm_state import (
    DualArmState,
    update_locked_arm_position,
    log_dual_arm_traj,
    compute_held_obj_poses,
    make_gripper_anim_traj,
)
from cutamp.optimize_plan import PlanContainer
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)

# Motion planning constants
APPROACH_HEIGHT = 0.05  # Height above grasp/place pose for approach/retract (meters)
GRIPPER_ANIM_STEPS = 20  # Number of interpolation steps for gripper animation
GRIPPER_ANIM_DT = 0.02  # Time step for gripper animation (seconds)


def _get_gripper_interp(robot: str, closing: bool) -> torch.Tensor:
    """Get gripper interpolation trajectory for opening/closing."""
    if robot == "ur5":
        if closing:
            return torch.linspace(0.0, 0.4, GRIPPER_ANIM_STEPS)[:, None]
        return torch.linspace(0.4, 0.0, GRIPPER_ANIM_STEPS)[:, None]
    elif robot == "t1":
        gripper_interp = torch.zeros(GRIPPER_ANIM_STEPS, 4)
        if closing:
            for i, (s, e) in enumerate([(0.0, 1.0), (0.0, -1.0), (0.0, -1.0), (0.0, 1.0)]):
                gripper_interp[:, i] = torch.linspace(s, e, GRIPPER_ANIM_STEPS)
        else:
            for i, (s, e) in enumerate([(1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)]):
                gripper_interp[:, i] = torch.linspace(s, e, GRIPPER_ANIM_STEPS)
        return gripper_interp
    else:  # Franka
        if closing:
            return torch.linspace(0.04, 0.02, GRIPPER_ANIM_STEPS)[:, None].repeat(1, 2)
        return torch.linspace(0.02, 0.04, GRIPPER_ANIM_STEPS)[:, None].repeat(1, 2)


def _attach_object_to_robot(
    motion_gen: MotionGen,
    obj_name: str,
    last_js: JointState,
    world: TAMPWorld,
    timer: TorchTimer,
    is_dual_arm: bool = False,
    other_motion_gen: Optional[MotionGen] = None,
):
    """Attach object to robot for collision checking during holding."""
    obstacle = motion_gen.world_model.get_obstacle(obj_name)
    old_get_bounding_spheres = obstacle.get_bounding_spheres
    
    def make_get_bounding_spheres(obj, world_ref, obstacle_ref):
        def get_bounding_spheres(self, *args, **kwargs) -> List[Sphere]:
            spheres = world_ref.get_collision_spheres(obj)
            pts = spheres[:, :3].cpu().numpy()
            n_radius = spheres[:, 3].cpu().numpy()
            obj_pose = Pose.from_list(self.pose, self.tensor_args)
            pre_transform = kwargs.get("pre_transform_pose")
            if pre_transform is not None:
                obj_pose = pre_transform.multiply(obj_pose)
            if pts is None or len(pts) == 0:
                raise ValueError("No sphere points found")
            points_cuda = self.tensor_args.to_device(pts)
            pts = obj_pose.transform_points(points_cuda).cpu().view(-1, 3).numpy()
            return [Sphere(name=f"{self.name}_sph_{i}", pose=[pts[i,0], pts[i,1], pts[i,2], 1,0,0,0], radius=n_radius[i]) 
                    for i in range(pts.shape[0])]
        return get_bounding_spheres
    
    obstacle.get_bounding_spheres = make_get_bounding_spheres(obj_name, world, obstacle).__get__(obstacle)
    
    with timer.time("curobo_planning"):
        motion_gen.attach_objects_to_robot(
            last_js, object_names=[obj_name], surface_sphere_radius=0.005,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE, voxelize_method="subdivide"
        )
    
    if is_dual_arm and other_motion_gen is not None:
        other_motion_gen.world_coll_checker.enable_obstacle(enable=False, name=obj_name)
    
    obstacle.get_bounding_spheres = old_get_bounding_spheres


def solve_curobo(
    plan_info: PlanContainer,
    best_particle: Particles,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer: Visualizer,
    timeline: str = "curobo",
):
    """
    Solve for full motion plan given a plan skeleton and optimized particles.
    
    For dual-arm robots (T1), supports plans that use both arms with automatic
    arm switching and shared joint synchronization.
    """
    plan_skeleton = plan_info["plan_skeleton"]
    is_dual_arm = world.is_dual_arm
    
    # Config for Cartesian planning (pick/place/approach)
    plan_config = MotionGenPlanConfig(
        timeout=20.0,
        max_attempts=120,
        enable_graph=False,
        enable_graph_attempt=5,
        parallel_finetune=True,
        enable_finetune_trajopt=True,
        time_dilation_factor=config.time_dilation_factor,
    )
    # Config for joint-space planning (retract) - longer timeout since it has fewer seeds
    retract_plan_config = MotionGenPlanConfig(
        timeout=40.0,           # Longer timeout for retract
        max_attempts=240,       # More attempts since seeds are limited
        enable_graph=False,     
        parallel_finetune=True, # Graph planner as fallback
        enable_graph_attempt=5,
        enable_finetune_trajopt=True,
        time_dilation_factor=config.time_dilation_factor,
    )
    accum_plans = []
    obj_to_current_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}
    
    # Initialize based on robot type
    if is_dual_arm:
        # Create left motion gen first, then share its collision world with right motion gen
        # This saves significant GPU memory by avoiding duplicate collision world allocation
        left_motion_gen = world.get_motion_gen(
            collision_activation_distance=config.world_activation_distance, arm="left"
        )
        _log.info("Sharing collision world between left and right arm motion generators")
        right_motion_gen = world.get_motion_gen(
            collision_activation_distance=config.world_activation_distance,
            arm="right",
            world_coll_checker=left_motion_gen.world_coll_checker,
        )
        state = DualArmState(
            left_motion_gen=left_motion_gen,
            right_motion_gen=right_motion_gen,
            left_kin_model=world.get_kin_model("left"),
            right_kin_model=world.get_kin_model("right"),
            left_tool_from_ee=world.get_tool_from_ee("left"),
            right_tool_from_ee=world.get_tool_from_ee("right"),
            left_js=JointState.from_position(world.left_q_init[None].clone()),
            right_js=JointState.from_position(world.right_q_init[None].clone()),
        )
        q_init = torch.cat([world.left_q_init, world.right_q_init])
        
        if config.warmup_motion_gen:
            with timer.time("curobo_motion_gen_warmup", log_callback=_log.debug):
                state.left_motion_gen.warmup()
                torch.cuda.empty_cache()
                state.right_motion_gen.warmup()
    else:
        motion_gen = world.get_motion_gen(collision_activation_distance=config.world_activation_distance)
        last_js = JointState.from_position(best_particle["q0"][None].clone())
        last_q_name = "q0"
        kin_model = world.kin_model
        tool_from_ee = world.tool_from_ee
        q_init = best_particle["q0"]
        state = None  # Not used for single-arm
        
        if config.warmup_motion_gen:
            with timer.time("curobo_motion_gen_warmup", log_callback=_log.debug):
                motion_gen.warmup()
    
    # Log initial state
    ts = 0.0
    visualizer.set_time_seconds(timeline, ts)
    visualizer.set_joint_positions(q_init)
    for obj, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj}", pose)

    # Main loop through plan skeleton
    for idx, ground_op in enumerate(plan_skeleton):
        metadata = ground_op.operator.metadata
        
        # Handle arm switching for dual-arm
        if is_dual_arm and metadata.arm is not None:
            op_arm = metadata.arm
            
            # Sync shared joints when switching arms
            if state.current_arm is not None and state.current_arm != op_arm:
                _log.debug(f"Switching from {state.current_arm} to {op_arm} arm")
                state.sync_shared_joints(state.current_arm, op_arm)
            
            state.current_arm = op_arm
            motion_gen, kin_model, tool_from_ee, last_js = state.get_state(op_arm)
            
            # Update locked arm position for accurate collision checking
            locked_arm = state.other_arm(op_arm)
            locked_arm_js = state.get_js(locked_arm)
            update_locked_arm_position(motion_gen, op_arm, locked_arm_js)
            
            last_q_name = state.get_q_name(op_arm)

        # Motion operators - track configuration names
        if metadata.is_motion and metadata.action_type is None:
            q_start = ground_op.values[-3] if len(ground_op.values) == 5 else ground_op.values[0]
            last_q_name = q_start
            if is_dual_arm:
                state.set_q_name(metadata.arm, q_start)

        # Retract operators - move arm towards home configuration
        elif metadata.action_type == "retract":
            q_retract_name = ground_op.values[-1]
            is_holding = len(ground_op.values) == 5
            
            if is_holding:
                obj, grasp_name, _, _, _ = ground_op.values
                obj_from_grasp = (action_4dof_to_mat4x4 if config.grasp_dof == 4 else action_6dof_to_mat4x4)(best_particle[grasp_name].clone())
            
            with timer.time("curobo_planning"):
                q_retract = best_particle[q_retract_name].clone()
                target_js = JointState.from_position(q_retract[None])
                retract_result = motion_gen.plan_single_js(last_js, target_js, retract_plan_config)
                if not retract_result.success:
                    raise RuntimeError(f"Failed to plan retract for {ground_op.name}: {retract_result.status}")
            
            dt = retract_result.interpolation_dt
            plan = retract_result.get_interpolated_plan()
            plan_entry = {"type": "trajectory", "plan": plan, "dt": dt}
            if is_dual_arm:
                plan_entry["arm"] = state.current_arm
            accum_plans.append(plan_entry)
            last_js = JointState.from_position(plan[-1:].position)
            
            if is_dual_arm:
                state.set_js(state.current_arm, last_js)
                state.set_q_name(state.current_arm, q_retract_name)
                
                if is_holding:
                    obj_poses = compute_held_obj_poses(state.current_arm, plan.position, state)
                    ts = log_dual_arm_traj(plan.position, state.current_arm, obj, obj_poses, ts, dt, state, visualizer, timeline, obj_to_current_pose)
                else:
                    ts = log_dual_arm_traj(plan.position, state.current_arm, None, None, ts, dt, state, visualizer, timeline, obj_to_current_pose)
            else:
                last_q_name = q_retract_name
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
            
            _log.debug(f"Retracted {state.current_arm if is_dual_arm else ''} arm to optimized configuration")

        # Pick operation
        elif metadata.action_type == "pick":
            obj, grasp, q = ground_op.values
            
            with timer.time("curobo_planning"):
                start_js = last_js
                
                # Retract from previous position if not at initial config
                if last_q_name not in {"q0", "left_q0", "right_q0"}:
                    world_from_ee = kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                    world_from_retract = world_from_ee.clone()
                    world_from_retract[2, 3] += APPROACH_HEIGHT
                    retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_retract), retract_plan_config)
                    if not retract_result.success:
                        raise RuntimeError(f"Failed to plan retract for {ground_op.name}: {retract_result.status}")
                    retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                else:
                    retract_result = None
                    retract_js = start_js
                
                # Compute grasp and approach poses
                world_from_obj = obj_to_current_pose[obj]
                obj_from_grasp = (action_4dof_to_mat4x4 if config.grasp_dof == 4 else action_6dof_to_mat4x4)(best_particle[grasp].clone())
                world_from_ee = world_from_obj @ obj_from_grasp @ tool_from_ee
                world_from_approach = world_from_ee.clone()
                world_from_approach[2, 3] += APPROACH_HEIGHT
                
                # Plan to approach
                approach_result = motion_gen.plan_single(retract_js, Pose.from_matrix(world_from_approach), plan_config)
                if not approach_result.success:
                    raise RuntimeError(f"Failed to plan approach for {ground_op.name}: {approach_result.status}")
                
                # Plan to grasp
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                end_result = motion_gen.plan_single(approach_js, Pose.from_matrix(world_from_ee), plan_config)
                if not end_result.success:
                    raise RuntimeError(f"Failed to plan grasp for {ground_op.name}: {end_result.status}")
            
            # Log trajectories
            for result in [retract_result, approach_result, end_result]:
                if result is None:
                    continue
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                plan_entry = {"type": "trajectory", "plan": plan, "dt": dt}
                if is_dual_arm:
                    plan_entry["arm"] = state.current_arm
                accum_plans.append(plan_entry)
                last_js = JointState.from_position(plan[-1:].position)
                
                if is_dual_arm:
                    ts = log_dual_arm_traj(plan.position, state.current_arm, None, None, ts, dt, state, visualizer, timeline, obj_to_current_pose)
                else:
                    ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
            
            # Update state after pick
            if is_dual_arm:
                state.set_js(state.current_arm, last_js)
                state.arm_holding[state.current_arm] = obj
                state.arm_grasp_transform[state.current_arm] = torch.inverse(obj_from_grasp)
            
            # Update object pose
            world_from_ee_final = kin_model.get_state(last_js.position).ee_pose.get_matrix()[0]
            obj_to_current_pose[obj] = world_from_ee_final @ torch.inverse(tool_from_ee) @ torch.inverse(obj_from_grasp)
            
            # Attach object to robot
            other_mg = state.right_motion_gen if is_dual_arm and state.current_arm == "left" else (state.left_motion_gen if is_dual_arm else None)
            _attach_object_to_robot(motion_gen, obj, last_js, world, timer, is_dual_arm, other_mg)
            
            # Gripper close animation
            gripper_interp = _get_gripper_interp(config.robot, closing=True)
            accum_plans.append({"type": "gripper", "action": "close"})
            if is_dual_arm:
                all_pos = make_gripper_anim_traj(last_js.position, gripper_interp, state.current_arm, state)
            else:
                all_pos = torch.cat([last_js.position.expand(GRIPPER_ANIM_STEPS, -1).cpu(), gripper_interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=GRIPPER_ANIM_DT,
                                                  arm=state.current_arm if is_dual_arm else None)

        # Place operation
        elif metadata.action_type == "place":
            obj, grasp, placement, surface, q = ground_op.values
            
            with timer.time("curobo_planning"):
                start_js = last_js
                world_from_ee_start = kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                
                # Retract
                world_from_retract = world_from_ee_start.clone()
                world_from_retract[2, 3] += APPROACH_HEIGHT
                retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_retract), retract_plan_config)
                if not retract_result.success:
                    raise RuntimeError(f"Failed to plan retract for {ground_op.name}: {retract_result.status}")
                
                # Compute place pose
                retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                world_from_obj_target = action_4dof_to_mat4x4(best_particle[placement].clone())
                obj_from_grasp = (action_4dof_to_mat4x4 if config.grasp_dof == 4 else action_6dof_to_mat4x4)(best_particle[grasp].clone())
                world_from_ee = world_from_obj_target @ obj_from_grasp @ tool_from_ee
                world_from_approach = world_from_ee.clone()
                world_from_approach[2, 3] += APPROACH_HEIGHT
                
                # Approach
                approach_result = motion_gen.plan_single(retract_js, Pose.from_matrix(world_from_approach), plan_config)
                if not approach_result.success:
                    raise RuntimeError(f"Failed to plan approach for {ground_op.name}: {approach_result.status}")
                
                # Place
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                end_result = motion_gen.plan_single(approach_js, Pose.from_matrix(world_from_ee), plan_config)
                if not end_result.success:
                    raise RuntimeError(f"Failed to plan place for {ground_op.name}: {end_result.status}")
            
            # Object tracking transform
            ee_from_obj = torch.inverse(torch.inverse(obj_to_current_pose[obj]) @ world_from_ee_start)
            
            # Log trajectories with object tracking
            for result in [retract_result, approach_result, end_result]:
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                plan_entry = {"type": "trajectory", "plan": plan, "dt": dt}
                if is_dual_arm:
                    plan_entry["arm"] = state.current_arm
                accum_plans.append(plan_entry)
                last_js = JointState.from_position(plan[-1:].position)
                
                # Compute object poses from FK
                robot_state = kin_model.get_state(plan.position)
                world_from_obj = robot_state.ee_pose.get_matrix() @ ee_from_obj
                
                if is_dual_arm:
                    ts = log_dual_arm_traj(plan.position, state.current_arm, obj, world_from_obj, ts, dt, state, visualizer, timeline, obj_to_current_pose)
                else:
                    ts = visualizer.log_joint_trajectory_with_mat4x4(
                        traj=plan.position, mat4x4_key=f"world/{obj}", mat4x4=world_from_obj,
                        timeline=timeline, start_time=ts, dt=dt
                    )
                
                obj_to_current_pose[obj] = world_from_obj[-1]
            
            if is_dual_arm:
                state.set_js(state.current_arm, last_js)
                state.arm_holding[state.current_arm] = None
                state.arm_grasp_transform[state.current_arm] = None
            
            # Detach object
            with timer.time("curobo_planning"):
                motion_gen.detach_object_from_robot("attached_object")
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
                motion_gen.world_collision.update_obstacle_pose(obj, Pose.from_matrix(obj_to_current_pose[obj]), update_cpu_reference=True)
                
                if is_dual_arm:
                    other_mg = state.right_motion_gen if state.current_arm == "left" else state.left_motion_gen
                    other_mg.world_coll_checker.enable_obstacle(enable=True, name=obj)
                    other_mg.world_collision.update_obstacle_pose(obj, Pose.from_matrix(obj_to_current_pose[obj]), update_cpu_reference=True)
            
            # Gripper open animation
            gripper_interp = _get_gripper_interp(config.robot, closing=False)
            accum_plans.append({"type": "gripper", "action": "open"})
            if is_dual_arm:
                all_pos = make_gripper_anim_traj(last_js.position, gripper_interp, state.current_arm, state)
            else:
                all_pos = torch.cat([last_js.position.expand(GRIPPER_ANIM_STEPS, -1).cpu(), gripper_interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=GRIPPER_ANIM_DT,
                                                  arm=state.current_arm if is_dual_arm else None)

        elif metadata.action_type in ("push", "push_stick"):
            raise NotImplementedError("Push operations not yet supported in motion planning.")
        
        else:
            raise NotImplementedError(f"Unsupported operator {ground_op.operator.name}")

        print(f"{idx + 1}. {ground_op.name}")

    _log.info(f"Motion planning metrics: {timer.get_summary('curobo_planning')}")
    return accum_plans
