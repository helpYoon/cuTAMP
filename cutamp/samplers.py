# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from curobo.scene import Cuboid, Mesh, Obstacle

from cutamp.utils.common import approximate_goal_aabb, pose_list_to_mat4x4, transform_points


def sample_yaw(num_samples: int, device: torch.device) -> torch.Tensor:
    """``[num_samples]`` yaws, uniform in [0, 2pi)."""
    return torch.rand(num_samples, device=device) * 2 * torch.pi


def place_4dof_sampler(
    num_samples: int, obj: Obstacle, obj_spheres: torch.Tensor, surface: Obstacle
) -> torch.Tensor:  # [n, 4]
    """Sample 4-DOF placement poses in the world frame. This does not yet fully support surfaces with yaw."""
    if not isinstance(surface, (Cuboid, Mesh)):
        raise NotImplementedError(f"Only Cuboid or Mesh surfaces supported for now, not {type(surface)}")

    # Determine the z-position of the min of the object spheres (in object frame)
    sph_bottom = obj_spheres[:, 2] - obj_spheres[:, 3]
    obj_bottom = sph_bottom.min()  # used as a delta
    obj_z_delta = -obj_bottom

    # Assume the surface is a cuboid, sample xy positions within AABB in local frame
    if isinstance(surface, Cuboid):
        aabb_xy = surface.device_cfg.to_device(
            [[-surface.dims[0] / 2, -surface.dims[1] / 2], [surface.dims[0] / 2, surface.dims[1] / 2]]
        )
        surface_z = surface.dims[2] / 2
    else:
        # Note: this was only used for the Rummy demos in the past, so not tested extensively.
        aabb = approximate_goal_aabb(surface).to(obj.device_cfg.device)
        aabb_xy = aabb[:, :2]
        surface_z = aabb[1, 2]

    xy = torch.rand(num_samples, 2, device=obj.device_cfg.device)
    xy = aabb_xy[0] + xy * (aabb_xy[1] - aabb_xy[0])

    # TODO: consider collision activation distance
    # Sample z-offset and combine with xy
    z_lower, z_upper = 1e-3, 1e-2
    z = torch.rand(num_samples, 1, device=obj.device_cfg.device)
    z = z_lower + (z_upper - z_lower) * z
    z += obj_z_delta + surface_z
    xyz = torch.cat([xy, z], dim=1)

    # Transform to surface coordinate frame
    if isinstance(surface, Cuboid):
        surface_mat4x4 = pose_list_to_mat4x4(surface.pose).to(obj.device_cfg.device)
        xyz_surface = transform_points(xyz, surface_mat4x4)
    else:
        xyz_surface = xyz

    # Sample yaw
    yaw = sample_yaw(num_samples, obj.device_cfg.device)
    place_4dof = torch.cat([xyz_surface, yaw.unsqueeze(-1)], dim=1)
    return place_4dof
