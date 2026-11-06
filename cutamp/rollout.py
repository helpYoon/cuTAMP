# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Particle-rollout under cuRobo v0.8 single-MotionPlanner architecture.

Configurations are single vectors in the model's cspace (21 or 33 joints). One FK
call covers the whole chain; for multi-tool robots we extract per-timestep
ee/tool poses by looking up which arm each timestep's conf belongs to.
"""

from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import torch

from curobo.types import JointState

from cutamp.config import TAMPConfiguration
from cutamp.t1_domain import Conf
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
)


class _FKQuatGrad(torch.autograd.Function):
    """Identity on an FK quaternion ``[..., 4]`` (wxyz) whose backward re-encodes dL/dq for cuRobo v2's FK backward.

    That backward decodes the quaternion gradient with the body-frame map (``quaternion_gradient_to_angular_velocity``
    in quaternion_util.cuh) but dots the result with world-frame joint axes, which need
    Omega = 0.5 (-g_w v + w g_v + v x g_v). Passing g' = 2 q (x) (0, Omega) makes its decode return Omega exactly
    (for unit q, at any angle). cuRobo's own pose costs already encode this way; only PyTorch autograd needs it.
    If the fork's decode is ever changed to the world-frame formula, drop this:
    cutamp/tests/test_real_legs_fk_orientation_grad.py would then fail on the double correction.
    """

    @staticmethod
    def forward(ctx, quat: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(quat.detach().clone())  # cuRobo may reuse the FK output buffer before backward
        return quat.clone()

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> torch.Tensor:
        (q,) = ctx.saved_tensors
        w, v, g_v = q[..., :1], q[..., 1:], g[..., 1:]
        omega = 0.5 * (w * g_v - g[..., :1] * v + torch.cross(v, g_v, dim=-1))
        return 2.0 * torch.cat((-(v * omega).sum(-1, keepdim=True), w * omega + torch.cross(v, omega, dim=-1)), -1)


def _link_matrix(pose) -> torch.Tensor:
    """``pose.get_matrix()`` for a cuRobo FK link pose, with gradients correct w.r.t. the joint positions."""
    pose.quaternion = _FKQuatGrad.apply(pose.quaternion)
    return pose.get_matrix()


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


class Rollout(TypedDict):
    num_particles: int
    confs: torch.Tensor  # [num_particles, *h, d]
    conf_params: List[str]
    robot_spheres: torch.Tensor  # [num_particles, *h, n, 4]
    world_from_ee: torch.Tensor  # [num_particles, *h, 4, 4]
    world_from_soles: Optional[torch.Tensor]  # [num_particles, *h, 2, 4, 4] on the real legs; None without soles
    world_from_tool_desired: torch.Tensor  # [num_particles, *h, 4, 4]
    world_from_ee_desired: torch.Tensor  # [num_particles, *h, 4, 4]
    gripper_close: List[bool]
    action_params: List[str]
    obj_to_pose: Dict[str, torch.Tensor]  # each [num_particles, *h, 4, 4]
    action_to_pose_ts: Dict[str, int]
    ts_to_pose_ts: Dict[int, int]


class RolloutFunction:
    """Roll out a plan skeleton under the v0.8 single-MotionPlanner kinematics.

    Multi-tool robots (T1) compute FK once over the model's full cspace; per
    timestep we select the active arm's tool-frame pose from the FK output.
    """

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self.conf_params = get_conf_parameters(plan_skeleton)
        self.conf_to_arm = get_conf_to_arm(plan_skeleton)
        self.obj_to_initial_pose = {
            obj.name: self.world.get_object_pose(obj) for obj in self.world.movables
        }

        self.grasp_to_mat4x4_fn = action_4dof_to_mat4x4

        self._is_first_rollout = True

    def __call__(self, particles: Particles) -> Rollout:
        num_particles = particles["left_q0"].shape[0]
        sole_frames = self.world.model.sole_frames

        # Stack and FK in one shot. confs shape: (num_particles, num_timesteps, dof).
        if self.conf_params:
            confs = torch.stack([particles[c] for c in self.conf_params], dim=1)
        else:
            confs = torch.empty(num_particles, 0, 0, device=self.world.device)

        if confs.shape[1] > 0:
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
                frame: _link_matrix(ee_pose.get_link_pose(frame, make_contiguous=True))
                .view(num_particles, confs.shape[1], 4, 4)
                for frame in dict.fromkeys(conf_frames)
            }
            world_from_ee = torch.stack(
                [frame_to_mat[frame][:, t] for t, frame in enumerate(conf_frames)], dim=1
            )
            # Both soles at every conf, for the kinematic constraint's check against the stance (design 7.4). Adam
            # differentiates its sole_rot_err, so they go through _link_matrix like the hands.
            world_from_soles = torch.stack([
                _link_matrix(ee_pose.get_link_pose(frame, make_contiguous=True)).view(*confs.shape[:2], 4, 4)
                for frame in sole_frames
            ], dim=2) if sole_frames else None

            # Robot spheres: shared across tool frames; reshape directly.
            robot_spheres = kin_state.robot_spheres.view(num_particles, confs.shape[1], -1, 4)
        else:
            world_from_ee = torch.empty(num_particles, 0, 4, 4, device=self.world.device)
            world_from_soles = (
                torch.empty(num_particles, 0, len(sole_frames), 4, 4, device=self.world.device) if sole_frames else None
            )
            robot_spheres = torch.empty(num_particles, 0, 0, 4, device=self.world.device)

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
                tool_frame = self.world.tool_frame_for_arm(arm)
                tool_from_ee = self.world.tool_from_ee[tool_frame]
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
            world_from_soles=world_from_soles,
            world_from_tool_desired=world_from_tool_desired,
            world_from_ee_desired=world_from_ee_desired,
            gripper_close=gripper_close,
            action_params=action_params,
            obj_to_pose=obj_to_pose,
            action_to_pose_ts=action_to_pose_ts,
            ts_to_pose_ts=ts_to_pose_ts,
        )
