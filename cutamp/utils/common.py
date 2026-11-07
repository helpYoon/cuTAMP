# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict

import torch
from curobo.scene import Cuboid, Cylinder, Mesh, Obstacle, Scene
from roma import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, rotmat_to_unitquat, unitquat_to_rotmat

from cutamp.envs import TAMPEnvironment

Particles = Dict[str, torch.Tensor]  # each [num_particles, *h, d]


def pose_list_to_mat4x4(pose: list[float] | None) -> torch.Tensor:
    """cuRobo pose list to 4x4 transformation matrix."""
    mat4x4 = torch.eye(4)
    if pose is not None:
        mat4x4[:3, 3] = torch.tensor(pose[:3])
        # Curobo stores poses as [x y z qw qx qy qz]
        quat_wxyz = torch.tensor(pose[3:])
        quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
        mat4x4[:3, :3] = unitquat_to_rotmat(quat_xyzw)
    return mat4x4


def action_4dof_to_mat4x4(action_4dof: torch.Tensor) -> torch.Tensor:
    """Convert 4-DOF action to 4x4 transformation matrix."""
    if action_4dof.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got shape {action_4dof.shape}")

    # Create 4x4 matrix and set rotation and translation
    mat4x4 = torch.eye(4, device=action_4dof.device, dtype=action_4dof.dtype)
    mat4x4 = mat4x4.repeat(*action_4dof.shape[:-1], 1, 1)

    # Set the rotation component
    yaw = action_4dof[..., 3]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    mat4x4[..., 0, 0] = cos_yaw
    mat4x4[..., 0, 1] = -sin_yaw
    mat4x4[..., 1, 0] = sin_yaw
    mat4x4[..., 1, 1] = cos_yaw

    # Set the translation component
    mat4x4[..., :3, 3] = action_4dof[..., :3]
    return mat4x4


def grasp_7dof_to_mat4x4(grasp: torch.Tensor) -> torch.Tensor:
    """Convert a grasp ``[..., 7]`` = ``[x, y, z, qw, qx, qy, qz]`` (cuRobo's wxyz order) to ``[..., 4, 4]``.

    The pose is the gripper frame G in the object frame. The quaternion is not normalized: the loader stores unit ones.
    """
    if grasp.shape[-1] != 7:
        raise ValueError(f"Expected last dimension to be 7, got shape {grasp.shape}")
    mat4x4 = torch.eye(4, device=grasp.device, dtype=grasp.dtype).repeat(*grasp.shape[:-1], 1, 1)
    mat4x4[..., :3, :3] = unitquat_to_rotmat(quat_wxyz_to_xyzw(grasp[..., 3:]))
    mat4x4[..., :3, 3] = grasp[..., :3]
    return mat4x4


def mat4x4_to_grasp_7dof(mat: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., 4, 4]`` transforms to grasps ``[..., 7]`` = ``[x, y, z, qw, qx, qy, qz]``."""
    quat_wxyz = quat_xyzw_to_wxyz(rotmat_to_unitquat(mat[..., :3, :3]))
    return torch.cat([mat[..., :3, 3], quat_wxyz], dim=-1)


def transform_spheres(
    spheres: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    """Transform [num_spheres, 4] spheres by [*b, 4, 4] transformation matrices into [*b, num_spheres, 4]."""
    centers = spheres[:, :3]
    centers_hom = torch.cat([centers, torch.ones_like(centers[:, :1])], dim=1)
    radii = spheres[:, 3]

    # Add a dimension to broadcast over num_spheres
    if transform.ndim > 2:
        # (*b, 4, 4) -> (*b, 1, 4, 4)
        transform = transform.unsqueeze(-3)

    out_spheres = torch.einsum("...ij,...j->...i", transform, centers_hom)
    out_spheres[..., 3] = radii
    return out_spheres


def transform_points(
    points: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    """Transform [num_points, 3] points by [*b, 4, 4] transformation matrices into [*b, num_points, 3]."""
    points_hom = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)

    # Add a dimension to broadcast over num_points
    if transform.ndim > 2:
        # (*b, 4, 4) -> (*b, 1, 4, 4)
        transform = transform.unsqueeze(-3)

    out_points_hom = torch.einsum("...ij,...j->...i", transform, points_hom)
    out_points = out_points_hom[..., :3]
    return out_points


def approximate_goal_aabb(goal: Obstacle) -> torch.Tensor:
    """World-frame AABB [[lower], [upper]] (float32, [2, 3]) of a cuboid or mesh: the bounds of cuRobo's posed
    trimesh, so a rotated cuboid gets the box around all eight corners. trimesh normalizes the pose's quaternion;
    pose_list_to_mat4x4 does not. Other types raise (cuRobo's capsule trimesh ignores base/tip)."""
    if not isinstance(goal, (Cuboid, Mesh)):
        raise NotImplementedError(f"Goal type {type(goal)} not supported yet.")
    return torch.tensor(goal.get_trimesh_mesh(transform_with_pose=True).bounds, dtype=torch.float32)


def get_world_cfg(env: TAMPEnvironment, include_movables: bool = False) -> Scene:
    """Build a cuRobo Scene from the TAMP environment's obstacles: the statics, after the movables if
    ``include_movables``."""
    scene = Scene()
    for obj in [*env.movables, *env.statics] if include_movables else env.statics:
        if not isinstance(obj, (Cuboid, Cylinder, Mesh)):  # add_obstacle drops other types without raising
            raise ValueError(f"Unknown object type: {type(obj)}")
        scene.add_obstacle(obj)
    return scene
