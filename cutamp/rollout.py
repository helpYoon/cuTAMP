# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Particle-rollout under cuRobo v0.8 single-MotionPlanner architecture.

Configurations are single 21-DOF vectors (T1's full cspace). One FK
call covers the whole chain; for multi-tool robots we extract per-timestep
ee/tool poses by looking up which arm each timestep's conf belongs to.
"""

from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import torch
from jaxtyping import Float

from curobo.types import JointState

from cutamp.config import TAMPConfiguration
from cutamp.t1_domain import Conf
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    action_6dof_to_mat4x4,
)


def get_conf_parameters(plan_skeleton: PlanSkeleton) -> List[str]:
    """Configuration parameter names appearing in actionable operators."""
    conf_params = []
    for ground_op in plan_skeleton:
        if ground_op.operator.metadata.is_actionable:
            conf_idx = [i for i, p in enumerate(ground_op.operator.parameters) if p.type == Conf]
            conf_params.extend([ground_op.values[i] for i in conf_idx])
    return list(dict.fromkeys(conf_params))


def get_retract_parameters(plan_skeleton: PlanSkeleton) -> List[Tuple[str, bool, Optional[str]]]:
    """``(q_retract_name, is_holding, held_obj_name)`` for each retract operator."""
    retract_info = []
    for ground_op in plan_skeleton:
        if ground_op.operator.metadata.action_type == "retract":
            q_retract = ground_op.values[-1]
            if "Holding" in ground_op.operator.name:
                held_obj = ground_op.values[0]
                retract_info.append((q_retract, True, held_obj))
            else:
                retract_info.append((q_retract, False, None))
    seen = set()
    unique = []
    for info in retract_info:
        if info[0] not in seen:
            seen.add(info[0])
            unique.append(info)
    return unique


def get_conf_to_arm(
    plan_skeleton: PlanSkeleton,
) -> Dict[str, Optional[Literal["left", "right"]]]:
    """Map each conf parameter name to its arm."""
    conf_to_arm = {"left_q0": "left", "right_q0": "right"}
    for ground_op in plan_skeleton:
        arm = ground_op.operator.metadata.arm
        if arm:
            conf_idxs = [i for i, p in enumerate(ground_op.operator.parameters) if p.type == Conf]
            for idx in conf_idxs:
                conf_name = ground_op.values[idx]
                conf_to_arm.setdefault(conf_name, arm)
    return conf_to_arm


def get_action_parameters(plan_skeleton: PlanSkeleton) -> List[str]:
    """Action parameter names (grasp/placement/push pose) per actionable op."""
    action_params = []
    for ground_op in plan_skeleton:
        atype = ground_op.operator.metadata.action_type
        vals = ground_op.values
        if atype == "pick":
            action_params.append(vals[1])
        elif atype == "place":
            action_params.append(vals[2])
        elif atype == "push":
            action_params.append(vals[1])
        elif atype == "push_stick":
            action_params.append(vals[3])
    return action_params


class Rollout(TypedDict):
    num_particles: int
    confs: Float[torch.Tensor, "num_particles *h d"]
    conf_params: List[str]
    robot_spheres: Float[torch.Tensor, "num_particles *h n 4"]
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
    """Roll out a plan skeleton under the v0.8 single-MotionPlanner kinematics.

    Multi-tool robots (T1) compute FK once over the full 21-DOF cspace; per
    timestep we select the active arm's tool-frame pose from the FK output.
    """

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectories are not supported in rollouts yet")
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self.conf_params = get_conf_parameters(plan_skeleton)
        self.conf_to_arm = get_conf_to_arm(plan_skeleton)
        self.obj_to_initial_pose = {
            obj.name: self.world.get_object_pose(obj) for obj in self.world.movables
        }

        if config.grasp_dof == 4:
            self.grasp_to_mat4x4_fn = action_4dof_to_mat4x4
        elif config.grasp_dof == 6:
            self.grasp_to_mat4x4_fn = action_6dof_to_mat4x4
        else:
            raise ValueError(f"Unsupported {config.grasp_dof=}")

        if config.place_dof != 4:
            raise ValueError(f"Unsupported {config.place_dof=}")

        from cutamp.robots.t1 import LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME
        self._tool_frame_for_arm = {
            "left": LEFT_TOOL_FRAME, "right": RIGHT_TOOL_FRAME,
        }

        self._is_first_rollout = True

    def _ee_pose_for_conf(self, ee_pose, conf_name: str):
        """Look up the active arm's tool-frame pose at the timestep of ``conf_name``."""
        arm = self.conf_to_arm.get(conf_name)
        tool_frame = self._tool_frame_for_arm.get(arm, self.world.tool_frames[0])
        return ee_pose.get_link_pose(tool_frame, make_contiguous=True)

    def __call__(self, particles: Particles) -> Rollout:
        num_particles = particles["left_q0"].shape[0]

        # Stack and FK in one shot. confs shape: (num_particles, num_timesteps, dof).
        if self.conf_params:
            confs = torch.stack([particles[c] for c in self.conf_params], dim=1)
        else:
            confs = torch.empty(num_particles, 0, 0, device=self.world.device)

        if confs.shape[1] > 0:
            confs_flat = confs.view(-1, confs.shape[-1])
            js = JointState.from_position(confs_flat)
            kin_state = self.world.kinematics.compute_kinematics(js)
            ee_pose = kin_state.tool_poses  # ToolPose [B*H, 1, num_links, 3/4]

            # Per-timestep ee selection: for each conf, pick that arm's tool frame.
            world_from_ee_per_ts = []
            for t, conf_name in enumerate(self.conf_params):
                pose = self._ee_pose_for_conf(ee_pose, conf_name)
                # pose: Pose [B*H, 3/4]; we want the row for batch=t (per-ts slicing).
                # ee_pose flattens [num_particles, num_timesteps] into B*H so row index = particle*T + t.
                # Easier: rebuild as 4x4 matrix and reshape.
                mat = pose.get_matrix().view(num_particles, confs.shape[1], 4, 4)
                world_from_ee_per_ts.append(mat[:, t])
            world_from_ee = torch.stack(world_from_ee_per_ts, dim=1)

            # Robot spheres: shared across tool frames; reshape directly.
            robot_spheres = kin_state.robot_spheres.view(num_particles, confs.shape[1], -1, 4)
        else:
            world_from_ee = torch.empty(num_particles, 0, 4, 4, device=self.world.device)
            robot_spheres = torch.empty(num_particles, 0, 0, 4, device=self.world.device)

        # Walk operators to produce action targets.
        world_from_tool_desired: List[torch.Tensor] = []
        gripper_close: List[bool] = []
        action_params: List[str] = []
        action_to_ts: Dict[str, int] = {}
        action_to_pose_ts: Dict[str, int] = {}
        ts_to_pose_ts: Dict[int, int] = {}
        action_to_arm: Dict[str, Optional[Literal["left", "right"]]] = {}

        grasp_to_mat4x4: Dict[str, torch.Tensor] = {}

        def get_grasp_mat4x4(grasp_name_: str) -> torch.Tensor:
            if grasp_name_ not in grasp_to_mat4x4:
                grasp_to_mat4x4[grasp_name_] = self.grasp_to_mat4x4_fn(particles[grasp_name_])
            return grasp_to_mat4x4[grasp_name_]

        obj_to_pose = {
            obj.name: [self.obj_to_initial_pose[obj.name].expand(num_particles, -1, -1)]
            for obj in self.world.movables
        }

        def current_pose(name: str) -> torch.Tensor:
            return obj_to_pose[name][-1]

        ts, pose_ts = 0, 0
        for ground_op in self.plan_skeleton:
            metadata = ground_op.operator.metadata
            arm = metadata.arm

            if metadata.is_motion:
                continue

            if metadata.action_type == "pick":
                obj_name, grasp_name, _ = ground_op.values
                world_from_grasp = current_pose(obj_name) @ get_grasp_mat4x4(grasp_name)
                world_from_tool_desired.append(world_from_grasp)
                gripper_close.append(True)
                action_params.append(grasp_name)
                action_to_ts[grasp_name] = ts
                action_to_pose_ts[grasp_name] = pose_ts
                action_to_arm[grasp_name] = arm

            elif metadata.action_type == "place":
                obj_name, grasp_name, place_name, _, _ = ground_op.values
                world_from_obj = action_4dof_to_mat4x4(particles[place_name])
                world_from_tool = world_from_obj @ get_grasp_mat4x4(grasp_name)
                for obj in self.world.movables:
                    if obj.name == obj_name:
                        obj_to_pose[obj.name].append(world_from_obj)
                    else:
                        obj_to_pose[obj.name].append(current_pose(obj.name))
                pose_ts += 1
                world_from_tool_desired.append(world_from_tool)
                gripper_close.append(False)
                action_params.append(place_name)
                action_to_ts[place_name] = ts
                action_to_pose_ts[place_name] = pose_ts
                ts_to_pose_ts[ts] = pose_ts
                action_to_arm[place_name] = arm

            elif metadata.action_type == "push":
                _, pose_name, _ = ground_op.values
                world_from_push = action_4dof_to_mat4x4(particles[pose_name])
                world_from_tool_desired.append(world_from_push)
                gripper_close.append(True)
                action_params.append(pose_name)
                action_to_ts[pose_name] = ts
                action_to_pose_ts[pose_name] = pose_ts
                action_to_arm[pose_name] = arm

            elif metadata.action_type == "push_stick":
                _, stick_name, grasp_name, pose_name, _ = ground_op.values
                world_from_stick = action_4dof_to_mat4x4(particles[pose_name])
                world_from_tool = world_from_stick @ get_grasp_mat4x4(grasp_name)
                for obj in self.world.movables:
                    if obj.name == stick_name:
                        obj_to_pose[obj.name].append(world_from_stick)
                    else:
                        obj_to_pose[obj.name].append(current_pose(obj.name))
                pose_ts += 1
                world_from_tool_desired.append(world_from_tool)
                gripper_close.append(True)
                action_params.append(pose_name)
                action_to_ts[pose_name] = ts
                action_to_pose_ts[pose_name] = pose_ts
                action_to_arm[pose_name] = arm

            else:
                raise ValueError(f"Unsupported operator {ground_op.operator.name}")

            ts_to_pose_ts[ts] = pose_ts
            ts += 1

        world_from_tool_desired = (
            torch.stack(world_from_tool_desired, dim=1)
            if world_from_tool_desired
            else torch.empty(num_particles, 0, 4, 4, device=self.world.device)
        )

        # world_from_ee_desired: per-action arm-specific tool_from_ee.
        if action_params:
            ee_desired_list = []
            for i, action in enumerate(action_params):
                arm = action_to_arm.get(action)
                tool_frame = self._tool_frame_for_arm.get(arm, self.world.tool_frames[0])
                tool_from_ee = self.world.get_tool_from_ee(tool_frame)
                ee_desired_list.append(world_from_tool_desired[:, i] @ tool_from_ee)
            world_from_ee_desired = torch.stack(ee_desired_list, dim=1)
        else:
            world_from_ee_desired = torch.empty(num_particles, 0, 4, 4, device=self.world.device)

        obj_to_pose = {k: torch.stack(v, dim=1) for k, v in obj_to_pose.items()}

        if self._is_first_rollout:
            assert (
                confs.shape[1]
                == world_from_ee.shape[1]
                == world_from_ee_desired.shape[1]
                == world_from_tool_desired.shape[1]
                == ts
            ), "Number of timesteps do not match"
            for obj, poses in obj_to_pose.items():
                assert poses.shape[1] == pose_ts + 1, f"pose_ts mismatch for {obj}"
            self._is_first_rollout = False

        return Rollout(
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
