# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List, Dict, Literal, Optional, TypedDict

import torch
from jaxtyping import Float

from cutamp.utils.common import Particles, action_6dof_to_mat4x4, action_4dof_to_mat4x4
from cutamp.config import TAMPConfiguration
from cutamp.tamp_domain import Conf
from cutamp.tamp_world import (
    TAMPWorld,
)
from cutamp.task_planning import PlanSkeleton


def get_conf_parameters(plan_skeleton: PlanSkeleton, is_dual_arm: bool = False) -> List[str]:
    """Get the parameters of the plan skeletons that are of type Conf. Returns a list of unique parameter names.
    
    Only includes configurations from actionable operators (Pick, Place, Push, PushStick).
    Motion operators (MoveFree, MoveHolding) configs are excluded as they're determined by actionable ops.
    
    Note: Initial configs (q0, left_q0, right_q0) are never included since they only appear in
    motion operators (MoveFree), so no explicit removal is needed.
    """
    conf_params = []
    for ground_op in plan_skeleton:
        metadata = ground_op.operator.metadata

        # Only include configs from actionable operators (not motion operators)
        if metadata.is_actionable:
            conf_idx = [idx for idx, param in enumerate(ground_op.operator.parameters) if param.type == Conf]
            op_conf_params = [ground_op.values[idx] for idx in conf_idx]
            conf_params.extend(op_conf_params)
    
    # remove duplicates while preserving order
    conf_params = list(dict.fromkeys(conf_params))  
    return conf_params


def get_retract_parameters(plan_skeleton: PlanSkeleton) -> List[str]:
    """Get the q_retract parameters from retract operators.
    
    Returns a list of unique q_retract parameter names in order.
    """
    retract_params = []
    for ground_op in plan_skeleton:
        metadata = ground_op.operator.metadata
        if metadata.action_type == "retract":
            # Last param is always q_retract
            q_retract = ground_op.values[-1]
            retract_params.append(q_retract)
    
    # remove duplicates while preserving order
    retract_params = list(dict.fromkeys(retract_params))
    return retract_params


def get_conf_to_arm(plan_skeleton: PlanSkeleton, is_dual_arm: bool) -> Dict[str, Optional[Literal["left", "right"]]]:
    """Build a mapping from configuration parameter names to their associated arm (for dual-arm robots)."""
    if not is_dual_arm:
        return {}
    
    conf_to_arm = {"left_q0": "left", "right_q0": "right"}
    for ground_op in plan_skeleton:
        arm = ground_op.operator.metadata.arm
        if arm:
            conf_idxs = [idx for idx, param in enumerate(ground_op.operator.parameters) if param.type == Conf]
            for idx in conf_idxs:
                conf_name = ground_op.values[idx]
                if conf_name not in conf_to_arm:
                    conf_to_arm[conf_name] = arm
    return conf_to_arm

def get_action_parameters(plan_skeleton: PlanSkeleton) -> List[str]:
    """
    Get action parameters (grasps, placements, push poses) in the order they appear in Pick/Place/Push operators.
    
    This matches the order used by RolloutFunction when building action_params.
    """
    action_params = []
    for ground_op in plan_skeleton:
        action_type = ground_op.operator.metadata.action_type
        
        if action_type == "pick":
            # Pick: [obj, grasp, q] -> grasp is action param
            _, grasp_name, _ = ground_op.values
            action_params.append(grasp_name)
        elif action_type == "place":
            # Place: [obj, grasp, placement, surface, q] -> placement is action param
            _, _, place_name, _, _ = ground_op.values
            action_params.append(place_name)
        elif action_type == "push":
            # Push: [button, pose, q] -> pose is action param
            _, pose_name, _ = ground_op.values
            action_params.append(pose_name)
        elif action_type == "push_stick":
            # PushStick: [button, obj, grasp, pose, q] -> pose is action param
            _, _, _, pose_name, _ = ground_op.values
            action_params.append(pose_name)
    
    return action_params


class Rollout(TypedDict):
    """Dict that stores results of a rollout."""

    num_particles: int
    confs: Float[torch.Tensor, "num_particles *h d"]
    conf_params: List[str]
    robot_spheres: Float[torch.Tensor, "num_particles *h 4"]
    world_from_ee: Float[torch.Tensor, "num_particles *h 4 4"]
    world_from_tool_desired: Float[torch.Tensor, "num_particles *h 4 4"]
    world_from_ee_desired: Float[torch.Tensor, "num_particles *h 4 4"]
    gripper_close: List[bool]
    action_params: List[str]
    obj_to_pose: Dict[str, Float[torch.Tensor, "num_particles *h 4 4"]]
    action_to_ts: Dict[str, int]
    action_to_pose_ts: Dict[str, int]
    ts_to_pose_ts: Dict[int, int]


class RolloutFunction:
    """Rollout function that rolls out the plan skeleton. Only supports robot and object kinematics right now."""

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectories are not supported in rollouts yet")
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self.is_dual_arm = world.is_dual_arm
        self.conf_params = get_conf_parameters(plan_skeleton, is_dual_arm=self.is_dual_arm)
        self.conf_to_arm = get_conf_to_arm(plan_skeleton, self.is_dual_arm)
        self.obj_to_initial_pose = {obj.name: self.world.get_object_pose(obj) for obj in self.world.movables}

        # Grasp to 4x4 matrix function
        if config.grasp_dof == 4:
            self.grasp_to_mat4x4_fn = action_4dof_to_mat4x4
        elif config.grasp_dof == 6:
            self.grasp_to_mat4x4_fn = action_6dof_to_mat4x4
        else:
            raise ValueError(f"Unsupported {config.grasp_dof=}")

        # Assume placements are 4-DOF
        if config.place_dof != 4:
            raise ValueError(f"Unsupported {config.place_dof=}")

        # Flag for first rollout, used to apply a runtime check
        self._is_first_rollout = True

    def __call__(self, particles: Particles) -> Rollout:
        """
        Rollout particles given the plan skeleton through the world. We keep the rollout information minimal to avoid
        unnecessary computations and backward passes.
        """
        # Determine num_particles from first available initial config
        if self.is_dual_arm:
            num_particles = particles["left_q0"].shape[0]
        else:
            num_particles = particles["q0"].shape[0]

        # Forward kinematics - handle dual-arm vs single-arm
        if self.is_dual_arm:
            # For dual-arm, we need to compute FK for each arm separately
            # Group configurations by arm
            left_conf_params = [c for c in self.conf_params if self.conf_to_arm.get(c) == "left"]
            right_conf_params = [c for c in self.conf_params if self.conf_to_arm.get(c) == "right"]
            
            all_world_from_ee = []
            all_robot_spheres = []
            
            # Process left arm configurations
            if left_conf_params:
                left_confs = torch.stack([particles[conf] for conf in left_conf_params], dim=1)
                left_confs_flat = left_confs.view(-1, left_confs.shape[-1])
                left_kin_model = self.world.get_kin_model("left")
                left_robot_state = left_kin_model.get_state(left_confs_flat)
                left_ee_flat = left_robot_state.ee_pose.get_matrix()
                left_ee = left_ee_flat.view(num_particles, left_confs.shape[1], 4, 4)
                left_spheres_flat = left_robot_state.get_link_spheres()
                left_spheres = left_spheres_flat.view(num_particles, left_confs.shape[1], -1, 4)
            else:
                left_ee = None
                left_spheres = None
            
            # Process right arm configurations  
            if right_conf_params:
                right_confs = torch.stack([particles[conf] for conf in right_conf_params], dim=1)
                right_confs_flat = right_confs.view(-1, right_confs.shape[-1])
                right_kin_model = self.world.get_kin_model("right")
                right_robot_state = right_kin_model.get_state(right_confs_flat)
                right_ee_flat = right_robot_state.ee_pose.get_matrix()
                right_ee = right_ee_flat.view(num_particles, right_confs.shape[1], 4, 4)
                right_spheres_flat = right_robot_state.get_link_spheres()
                right_spheres = right_spheres_flat.view(num_particles, right_confs.shape[1], -1, 4)
            else:
                right_ee = None
                right_spheres = None
            
            # Interleave results in original conf_params order
            world_from_ee_list = []
            robot_spheres_list = []
            left_idx, right_idx = 0, 0
            for conf in self.conf_params:
                arm = self.conf_to_arm.get(conf)
                if arm == "left":
                    world_from_ee_list.append(left_ee[:, left_idx])
                    robot_spheres_list.append(left_spheres[:, left_idx])
                    left_idx += 1
                elif arm == "right":
                    world_from_ee_list.append(right_ee[:, right_idx])
                    robot_spheres_list.append(right_spheres[:, right_idx])
                    right_idx += 1
            
            world_from_ee = torch.stack(world_from_ee_list, dim=1) if world_from_ee_list else torch.empty(num_particles, 0, 4, 4)
            robot_spheres = torch.stack(robot_spheres_list, dim=1) if robot_spheres_list else torch.empty(num_particles, 0, 0, 4)
            confs = torch.stack([particles[conf] for conf in self.conf_params], dim=1) if self.conf_params else torch.empty(num_particles, 0, 11)
        else:
            # Single-arm: original logic
            confs = torch.stack([particles[conf] for conf in self.conf_params], dim=1)
            confs_flat = confs.view(-1, confs.shape[-1])
            robot_state = self.world.kin_model.get_state(confs_flat)
            world_from_ee_flat = robot_state.ee_pose.get_matrix()
            world_from_ee = world_from_ee_flat.view(num_particles, confs.shape[1], 4, 4)

            # Robot link spheres for collision checking from cuRobo
            robot_spheres_flat = robot_state.get_link_spheres()
            robot_spheres = robot_spheres_flat.view(num_particles, confs.shape[1], -1, 4)

        # Stores the desired actions
        world_from_tool_desired = []
        gripper_close: List[bool] = []
        action_params: List[str] = []
        action_to_ts: Dict[str, int] = {}

        # For pose timestamp (pose_ts), we only accumulate the poses if the operator causes a change in the object pose.
        # These dicts are used to map actions and timestamps to their corresponding pose timestamps.
        action_to_pose_ts: Dict[str, int] = {}
        ts_to_pose_ts: Dict[int, int] = {}

        # 4x4 transformation matrices for grasp parameters (if any)
        grasp_to_mat4x4: Dict[str, Float[torch.Tensor, "num_particles 4 4"]] = {}

        def get_grasp_mat4x4(grasp_name_: str) -> Float[torch.Tensor, "num_particles 4 4"]:
            if grasp_name_ not in grasp_to_mat4x4:
                grasp_ = particles[grasp_name_]
                grasp_to_mat4x4[grasp_name_] = self.grasp_to_mat4x4_fn(grasp_)
            return grasp_to_mat4x4[grasp_name_]

        # Object poses in world frame for every timestep
        obj_to_pose = {
            obj.name: [self.obj_to_initial_pose[obj.name].expand(num_particles, -1, -1)] for obj in self.world.movables
        }

        def current_pose(obj: str) -> Float[torch.Tensor, "num_particles 4 4"]:
            return obj_to_pose[obj][-1]

        # Timestep of all actionable operators, and the corresponding timestep in the obj_to_pose list
        ts, pose_ts = 0, 0
        
        # Track which arm each action uses (for dual-arm tool_from_ee)
        action_to_arm: Dict[str, Optional[Literal["left", "right"]]] = {}

        # Rollout each ground operator in the plan skeleton
        for ground_op in self.plan_skeleton:
            metadata = ground_op.operator.metadata
            arm = metadata.arm

            # Skip motion operators (MoveFree and MoveHolding) as we don't support trajectories yet
            if metadata.is_motion:
                continue

            # Pick (single-arm and dual-arm)
            elif metadata.action_type == "pick":
                obj_name, grasp_name, _ = ground_op.values
                # Grasp is in object frame
                obj_from_grasp = get_grasp_mat4x4(grasp_name)

                # Compute tool pose in world frame given object and grasp pose
                world_from_obj = current_pose(obj_name)
                world_from_grasp = world_from_obj @ obj_from_grasp

                world_from_tool_desired.append(world_from_grasp)
                gripper_close.append(True)  # closing gripper at Pick
                action_params.append(grasp_name)
                action_to_ts[grasp_name] = ts
                action_to_pose_ts[grasp_name] = pose_ts
                action_to_arm[grasp_name] = arm

            # Place (single-arm and dual-arm)
            elif metadata.action_type == "place":
                obj_name, grasp_name, place_name, _, _ = ground_op.values

                # Place is desired object pose in world frame
                place_4dof = particles[place_name]
                world_from_obj = action_4dof_to_mat4x4(place_4dof)

                # Apply the grasp offset to get the tool frame pose
                obj_from_grasp = get_grasp_mat4x4(grasp_name)
                world_from_tool = world_from_obj @ obj_from_grasp

                # Accumulate poses of all the movable objects as we've moved the object
                for obj in self.world.movables:
                    if obj.name == obj_name:
                        obj_to_pose[obj.name].append(world_from_obj)  # use new desired pose
                    else:
                        obj_to_pose[obj.name].append(current_pose(obj.name))
                pose_ts += 1

                world_from_tool_desired.append(world_from_tool)
                gripper_close.append(False)  # opening gripper at Place
                action_params.append(place_name)
                action_to_ts[place_name] = ts
                action_to_pose_ts[place_name] = pose_ts
                ts_to_pose_ts[ts] = pose_ts
                action_to_arm[place_name] = arm

            # Push (single-arm and dual-arm)
            elif metadata.action_type == "push":
                button_name, pose_name, _ = ground_op.values

                # Push pose is desired tool pose
                push_4dof = particles[pose_name]
                world_from_push = action_4dof_to_mat4x4(push_4dof)

                world_from_tool_desired.append(world_from_push)
                gripper_close.append(True)  # close gripper at Push
                action_params.append(pose_name)
                action_to_ts[pose_name] = ts
                action_to_pose_ts[pose_name] = pose_ts
                action_to_arm[pose_name] = arm

            # PushStick (single-arm and dual-arm)
            elif metadata.action_type == "push_stick":
                button_name, stick_name, grasp_name, pose_name, _ = ground_op.values

                # Pose is desired stick pose
                stick_4dof = particles[pose_name]
                world_from_stick = action_4dof_to_mat4x4(stick_4dof)

                # Apply the grasp offset to get the tool frame pose
                obj_from_grasp = get_grasp_mat4x4(grasp_name)
                world_from_tool = world_from_stick @ obj_from_grasp

                # Accumulate poses of all the movable objects as we've moved and pushing with stick
                for obj in self.world.movables:
                    if obj.name == stick_name:
                        obj_to_pose[obj.name].append(world_from_stick)
                    else:
                        obj_to_pose[obj.name].append(current_pose(obj.name))
                pose_ts += 1

                world_from_tool_desired.append(world_from_tool)
                gripper_close.append(True)  # close gripper at PushStick (it's still closed)
                action_params.append(pose_name)
                action_to_ts[pose_name] = ts
                action_to_pose_ts[pose_name] = pose_ts
                action_to_arm[pose_name] = arm

            # Unknown
            else:
                raise ValueError(f"Unsupported operator {ground_op.operator.name}")

            # Increment time step
            ts_to_pose_ts[ts] = pose_ts
            ts += 1

        # Stack and store in rollout
        world_from_tool_desired = torch.stack(world_from_tool_desired, dim=1)
        
        # Compute world_from_ee_desired using arm-specific tool_from_ee for dual-arm
        if self.is_dual_arm:
            world_from_ee_desired_list = []
            for i, action in enumerate(action_params):
                arm = action_to_arm.get(action)
                tool_from_ee = self.world.get_tool_from_ee(arm) if arm else self.world.get_tool_from_ee("left")
                world_from_ee_desired_list.append(world_from_tool_desired[:, i] @ tool_from_ee)
            world_from_ee_desired = torch.stack(world_from_ee_desired_list, dim=1) if world_from_ee_desired_list else torch.empty(num_particles, 0, 4, 4)
        else:
            world_from_ee_desired = world_from_tool_desired @ self.world.tool_from_ee

        # Object poses for each timestep
        obj_to_pose = {k: torch.stack(v, dim=1) for k, v in obj_to_pose.items()}

        # Sanity check
        if self._is_first_rollout:
            assert (
                confs.shape[1]
                == world_from_ee.shape[1]
                == world_from_ee_desired.shape[1]
                == world_from_tool_desired.shape[1]
                == ts
            ), "Number of timesteps do not match"
            for obj, poses in obj_to_pose.items():
                assert poses.shape[1] == pose_ts + 1, f"Number of pose timesteps do not match for {obj}"
            self._is_first_rollout = False

        rollout = Rollout(
            num_particles=num_particles,
            confs=confs,
            conf_params=self.conf_params,
            robot_spheres=robot_spheres,
            world_from_ee=world_from_ee,
            world_from_tool_desired=world_from_tool_desired,
            world_from_ee_desired=world_from_ee_desired,
            gripper_close=gripper_close,
            action_params=action_params,
            obj_to_pose=obj_to_pose,
            action_to_ts=action_to_ts,
            action_to_pose_ts=action_to_pose_ts,
            ts_to_pose_ts=ts_to_pose_ts,
        )
        return rollout
