# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict, Optional, Union

import numpy as np
import rerun as rr
import torch
from curobo.scene import Mesh

from cutamp.config import TAMPConfiguration
from cutamp.robots import get_model, load_t1_rerun
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.rerun_utils import log_curobo_pose_to_rerun, curobo_to_rerun, log_curobo_mesh_to_rerun, AXIS_LENGTH


class Visualizer:
    """No-op visualizer base. Use directly when visualization is disabled."""

    def __init__(self, config: TAMPConfiguration, q_init: Optional[torch.Tensor]):
        self.config = config
        if q_init is not None:
            self.set_joint_positions(q_init)

    def set_time_sequence(self, timeline: str, val: int):
        pass

    def set_time_seconds(self, timeline: str, val: float):
        pass

    def set_joint_positions(self, q: Union[torch.Tensor, np.ndarray, list]):
        pass

    def log_joint_trajectory(self, traj: torch.Tensor, timeline: str, start_time: float, dt: float):
        pass

    def log_joint_trajectory_with_mat4x4s(
        self,
        traj: torch.Tensor,
        mat4x4s: Dict[str, torch.Tensor],
        timeline: str,
        start_time: float,
        dt: float,
    ):
        pass

    def log_tamp_world(self, world: TAMPWorld):
        pass

    def log_mat4x4(self, name: str, mat4x4: Union[torch.Tensor, np.ndarray]):
        pass

    def log_spheres(self, name: str, spheres: torch.Tensor):
        pass

    def log_scalar(self, name: str, value: float):
        pass

    def log_cost_dict(self, cost_dict: dict):
        pass


class RerunVisualizer(Visualizer):
    """Wrapper around rerun for easier visualization, and switching in different visualizers. Shows the robot on
    ``config.robot``'s planning model."""

    def __init__(
        self,
        config: TAMPConfiguration,
        q_init: torch.Tensor,
        application_id: str,
        recording_id: str,
        spawn: bool,
    ):
        rr.init(application_id, recording_id=recording_id, spawn=spawn)
        self.robot = load_t1_rerun(get_model(config.robot), load_mesh=config.viz_robot_mesh)
        super().__init__(config, q_init)

    def set_time_sequence(self, timeline: str, val: int):
        rr.set_time_sequence(timeline, val)

    def set_time_seconds(self, timeline: str, val: float):
        rr.set_time_seconds(timeline, val)

    def set_joint_positions(self, q: Union[torch.Tensor, np.ndarray, list]):
        if isinstance(q, torch.Tensor):
            q = q.tolist()
        self.robot.set_joint_positions(q)

    def log_joint_trajectory(self, traj: torch.Tensor, timeline: str, start_time: float, dt: float):
        return self.log_joint_trajectory_with_mat4x4s(traj, {}, timeline, start_time, dt)

    def log_joint_trajectory_with_mat4x4s(
        self,
        traj: torch.Tensor,
        mat4x4s: Dict[str, torch.Tensor],
        timeline: str,
        start_time: float,
        dt: float,
    ):
        for key, mat in mat4x4s.items():
            if traj.shape[0] != mat.shape[0]:
                raise ValueError(f"Trajectory and mat4x4 for {key} must have the same length.")
        end_time = start_time + len(traj) * dt
        times = [rr.TimeColumn(timeline, duration=np.linspace(start_time, end_time, len(traj)))]
        key_to_columns = self.robot.get_rr_columns(traj)

        for mat4x4_key, mat in mat4x4s.items():
            if mat4x4_key in key_to_columns:
                raise ValueError(f"Key {mat4x4_key} already exists in key_to_components.")
            mat = mat.detach().cpu()
            key_to_columns[mat4x4_key] = rr.Transform3D.columns(
                mat3x3=mat[:, :3, :3], translation=mat[:, :3, 3],
            )

        for key, columns in key_to_columns.items():
            rr.send_columns(key, indexes=times * len(columns), columns=columns)
        return end_time

    def log_tamp_world(self, world: TAMPWorld):
        rr_log_tamp_world(world)

    def log_mat4x4(self, name: str, mat4x4: Union[torch.Tensor, np.ndarray]):
        if isinstance(mat4x4, torch.Tensor):
            mat4x4 = mat4x4.detach().cpu()
        rr.log(name, rr.Transform3D(translation=mat4x4[:3, 3], mat3x3=mat4x4[:3, :3], axis_length=AXIS_LENGTH))

    def log_spheres(self, name: str, spheres: torch.Tensor):
        if isinstance(spheres, torch.Tensor):
            spheres = spheres.detach().cpu()
        rr.log(name, rr.Points3D(positions=spheres[:, :3], radii=spheres[:, 3]))

    def log_scalar(self, name: str, value: float):
        rr.log(name, rr.Scalars(value))

    def log_cost_dict(self, cost_dict: dict):
        for cost_type, cost_info in cost_dict.items():
            for name, vals in cost_info["values"].items():
                self.log_scalar(f"{cost_type}/{name}", vals.mean().item())


def rr_log_tamp_world(world: TAMPWorld, log_spheres: bool = True, log_arrows: bool = True):
    """Log TAMPWorld in rerun."""
    # Log movables
    for obj in world.movables:
        log_curobo_pose_to_rerun(f"world/{obj.name}", obj, static_transform=False, log_arrows=log_arrows)
        # Log mesh under subkey and as static since it's not changing
        rr.log(f"world/{obj.name}/mesh", curobo_to_rerun(obj.get_mesh(), compute_vertex_normals=True), static=True)

        # Get average color for the object
        if isinstance(obj, Mesh):
            rgb = np.array(obj.vertex_colors).mean(0) if obj.vertex_colors else [0.5, 0.5, 0.5]
        else:
            rgb = obj.color

        # Log collision spheres under subkey
        if log_spheres:
            spheres = world.get_collision_spheres(obj)
            rr.log(
                f"world/{obj.name}/spheres",
                rr.Points3D(positions=spheres[:, :3].cpu(), radii=spheres[:, 3].cpu(), colors=rgb),
                static=True,
            )

    # Static objects
    for obj in world.statics:
        log_curobo_mesh_to_rerun(f"world/{obj.name}", obj.get_mesh(), static_transform=True)
