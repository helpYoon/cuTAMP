# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Particle-rollout under cuRobo's single-MotionPlanner architecture.

Configurations are single vectors in the model's cspace (its 33 joints). One FK
call covers the whole chain; for multi-tool robots we extract per-timestep
ee/tool poses by looking up which arm each timestep's conf belongs to.
"""

from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import torch

from curobo.types import JointState

from cutamp.t1_domain import Conf
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    grasp_7dof_to_mat4x4,
    transform_spheres,
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


def movable_spheres_fn(world: TAMPWorld, plan_skeleton: PlanSkeleton, op_idx: int):
    """``fn(particles) -> [B, M, 4]``: every movable's collision spheres where the rollout puts the objects at
    the conf of ``plan_skeleton[op_idx]`` (a pick or a place), as ``robot_to_movables`` checks that conf: each
    object at its initial pose until a place moves it, then at that place's placement, the conf's own place
    included; objects in ``world.movables`` order. The initial poses are transformed here, so ``fn`` makes no
    host copy and captures (G_refresh)."""
    source = {obj.name: None for obj in world.movables}
    for op in list(plan_skeleton)[: op_idx + 1]:
        if op.operator.metadata.action_type == "place":
            obj_name, _grasp, placement, _surface, _q = op.values
            source[obj_name] = placement
    fixed = {name: transform_spheres(world.get_collision_spheres(name), world.get_object_pose(name))
             for name, placement in source.items() if placement is None}

    def fn(particles: Particles) -> torch.Tensor:
        batch = particles["left_q0"].shape[0]
        parts = [
            fixed[name].expand(batch, -1, -1) if placement is None
            else transform_spheres(world.get_collision_spheres(name), action_4dof_to_mat4x4(particles[placement]))
            for name, placement in source.items()
        ]
        return torch.cat(parts, dim=1) if parts else torch.empty(batch, 0, 4, device=world.device)

    return fn


class Rollout(TypedDict):
    num_particles: int
    confs: torch.Tensor  # [num_particles, *h, d]
    conf_params: List[str]
    robot_spheres: torch.Tensor  # [num_particles, *h, n, 4]
    world_from_ee: torch.Tensor  # [num_particles, *h, 4, 4]
    world_from_soles: torch.Tensor  # [num_particles, *h, 2, 4, 4]
    world_from_ee_desired: torch.Tensor  # [num_particles, *h, 4, 4]
    gripper_close: List[bool]
    action_params: List[str]
    obj_to_pose: Dict[str, torch.Tensor]  # each [num_particles, *h, 4, 4]
    action_to_pose_ts: Dict[str, int]
    ts_to_pose_ts: Dict[int, int]


class RolloutFunction:
    """Roll out a plan skeleton under the single-MotionPlanner kinematics.

    Multi-tool robots (T1) compute FK once over the model's full cspace; per
    timestep we select the active arm's tool-frame pose from the FK output.
    """

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld):
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.conf_params = get_conf_parameters(plan_skeleton)
        self.conf_to_arm = get_conf_to_arm(plan_skeleton)
        self.obj_to_initial_pose = {
            obj.name: self.world.get_object_pose(obj) for obj in self.world.movables
        }

    def __call__(self, particles: Particles) -> Rollout:
        num_particles = particles["left_q0"].shape[0]

        # Stack and FK in one shot. confs shape: (num_particles, num_timesteps, dof). Every plan picks and places,
        # so there is at least one actionable conf.
        confs = torch.stack([particles[c] for c in self.conf_params], dim=1)
        confs_flat = confs.view(-1, confs.shape[-1])
        js = JointState.from_position(confs_flat)
        kin_state = self.world.site_kinematics("rollout").compute_kinematics(js)
        ee_pose = kin_state.tool_poses  # ToolPose [B*H, 1, num_links, 3/4]

        # Per-timestep ee selection: for each conf, pick that arm's tool frame.
        # Build the 4x4 matrix once per unique tool frame (≤2 for T1), then
        # index per timestep. ee_pose flattens [num_particles, num_timesteps]
        # into B*H, so reshaping recovers the per-timestep axis.
        conf_frames = [
            self.world.tool_frame_for_arm(self.conf_to_arm.get(conf_name))
            for conf_name in self.conf_params
        ]
        frame_to_mat = {
            frame: ee_pose.get_link_pose(frame, make_contiguous=True).get_matrix()
            .view(num_particles, confs.shape[1], 4, 4)
            for frame in dict.fromkeys(conf_frames)
        }
        world_from_ee = torch.stack(
            [frame_to_mat[frame][:, t] for t, frame in enumerate(conf_frames)], dim=1
        )
        # Both soles at every conf, for the kinematic constraint's check against the stance (design 7.4). Adam
        # differentiates its sole_rot_err.
        world_from_soles = torch.stack([
            ee_pose.get_link_pose(frame, make_contiguous=True).get_matrix().view(*confs.shape[:2], 4, 4)
            for frame in self.world.model.sole_frames
        ], dim=2)

        # Robot spheres: shared across tool frames; reshape directly.
        robot_spheres = kin_state.robot_spheres.view(num_particles, confs.shape[1], -1, 4)

        # Walk operators to produce action targets.
        world_from_tool_desired: List[torch.Tensor] = []
        gripper_close: List[bool] = []
        action_params: List[str] = []
        action_to_pose_ts: Dict[str, int] = {}
        ts_to_pose_ts: Dict[int, int] = {}
        action_to_arm: Dict[str, Optional[Literal["left", "right"]]] = {}

        grasp_to_mat4x4: Dict[str, torch.Tensor] = {}

        def get_grasp_mat4x4(grasp_name_: str) -> torch.Tensor:
            if grasp_name_ not in grasp_to_mat4x4:
                grasp_to_mat4x4[grasp_name_] = grasp_7dof_to_mat4x4(particles[grasp_name_])
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
                action_to_pose_ts[place_name] = pose_ts
                ts_to_pose_ts[ts] = pose_ts
                action_to_arm[place_name] = arm

            else:
                raise ValueError(f"Unsupported operator {ground_op.operator.name}")

            ts_to_pose_ts[ts] = pose_ts
            ts += 1

        world_from_tool_desired = torch.stack(world_from_tool_desired, dim=1)

        # world_from_ee_desired: per-action arm-specific tool_from_ee.
        ee_desired_list = []
        for i, action in enumerate(action_params):
            arm = action_to_arm.get(action)
            tool_frame = self.world.tool_frame_for_arm(arm)
            tool_from_ee = self.world.tool_from_ee[tool_frame]
            ee_desired_list.append(world_from_tool_desired[:, i] @ tool_from_ee)
        world_from_ee_desired = torch.stack(ee_desired_list, dim=1)

        obj_to_pose = {k: torch.stack(v, dim=1) for k, v in obj_to_pose.items()}

        return Rollout(
            num_particles=num_particles,
            confs=confs,
            conf_params=self.conf_params,
            robot_spheres=robot_spheres,
            world_from_ee=world_from_ee,
            world_from_soles=world_from_soles,
            world_from_ee_desired=world_from_ee_desired,
            gripper_close=gripper_close,
            action_params=action_params,
            obj_to_pose=obj_to_pose,
            action_to_pose_ts=action_to_pose_ts,
            ts_to_pose_ts=ts_to_pose_ts,
        )
