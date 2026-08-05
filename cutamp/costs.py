# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch


def trajectory_length(confs: torch.Tensor) -> torch.Tensor:
    """
    Compute the length of a trajectory by summing the distances between consecutive configurations.
    """
    return (confs[..., 1:, :] - confs[..., :-1, :]).norm(dim=-1).sum(-1)


@torch.jit.script
def dist_from_bounds_jit(vals, lower, upper):
    """Euclidean distance of values from the given lower and upper bounds. If within the bounds, returns 0."""
    diff_lower = lower - vals
    diff_upper = vals - upper
    diff_max = torch.maximum(diff_lower, diff_upper)
    diff_max = diff_max.clamp(min=0.0)
    dists = diff_max.norm(p=2, dim=-1)
    return dists


def get_aabb_from_spheres(spheres: torch.Tensor) -> torch.Tensor:
    """Compute the axis-aligned bounding box (AABB) for a set of spheres that represent some object."""
    centers, radii = spheres[..., :3], spheres[..., 3]
    min_corners = centers - radii.unsqueeze(-1)
    min_corners = min_corners.min(-2).values  # take min over spheres in the batch

    max_corners = centers + radii.unsqueeze(-1)
    max_corners = max_corners.max(-2).values  # take max over spheres in the batch

    aabb = torch.stack([min_corners, max_corners], dim=-2)
    return aabb


def _sphere_to_sphere_overlap(
    spheres_1: torch.Tensor,
    spheres_2: torch.Tensor,
    activation_distance: float | None = None,
) -> torch.Tensor:
    """Compute the overlap volume between two sets of spheres. Can be used as a collision distance function."""
    centers_1, radii_1 = spheres_1[..., :3], spheres_1[..., 3]
    centers_2, radii_2 = spheres_2[..., :3], spheres_2[..., 3]

    dist_matrix = torch.cdist(centers_1, centers_2)
    radii_sum = radii_1.unsqueeze(-1) + radii_2.unsqueeze(-2)
    overlap = radii_sum - dist_matrix

    if activation_distance is not None:
        overlap += activation_distance
    return torch.relu(overlap).sum((-2, -1))


def sphere_to_sphere_overlap(
    spheres_1: torch.Tensor,
    spheres_2: torch.Tensor,
    activation_distance: float | None = None,
    use_aabb_check: bool = False,
) -> torch.Tensor:
    """
    Compute the overlap volume between two sets of spheres. Can be used as a collision distance function. If
    use_aabb_check=True, we compute the overlap only for batches of spheres that have intersecting AABBs.
    """
    if not use_aabb_check:
        return _sphere_to_sphere_overlap(spheres_1, spheres_2, activation_distance)

    # Compute AABB for each batch of spheres
    aabb_1 = get_aabb_from_spheres(spheres_1)  # [b, *h, 2, 3]
    aabb_2 = get_aabb_from_spheres(spheres_2)  # [b, *h, 2, 3]

    # Check intersection
    min_1, max_1 = aabb_1.unbind(-2)
    min_2, max_2 = aabb_2.unbind(-2)
    intersect = (min_1 <= max_2).all(-1) & (max_1 >= min_2).all(-1)  # [b, *h]

    output = torch.zeros_like(intersect, dtype=torch.float32)  # [b, *h]
    if intersect.any():
        # Only compute overlap for intersecting AABBs
        output[intersect] = _sphere_to_sphere_overlap(spheres_1[intersect], spheres_2[intersect], activation_distance)
    return output


def curobo_pose_error(
    pose_a_mat4x4: torch.Tensor, pose_b_mat4x4: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """Translational and rotational errors between two batched 4x4 pose tensors.

    Computes element-wise distances. Avoids ``curobo.types.Pose.distance``
    because v0.8's ``angular_distance_axis_angle`` has a broadcast bug that
    produces an ``[N, N]`` pairwise tensor for matching ``[N, 4]`` inputs.
    """
    # Translation: L2 distance in [b, *h]
    p_dist = torch.linalg.norm(pose_a_mat4x4[..., :3, 3] - pose_b_mat4x4[..., :3, 3], dim=-1)

    # Rotation: angle of relative rotation R_a^T @ R_b. Use the
    # ``angle = atan2(||skew_part||, (trace-1)/2)`` formulation, which is
    # numerically stable near identity (where arccos has infinite gradient).
    R_a = pose_a_mat4x4[..., :3, :3]
    R_b = pose_b_mat4x4[..., :3, :3]
    R_rel = torch.matmul(R_a.transpose(-1, -2), R_b)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_part = (trace - 1.0) * 0.5
    # off-diagonal vector that captures sin(angle) * axis
    skew_x = R_rel[..., 2, 1] - R_rel[..., 1, 2]
    skew_y = R_rel[..., 0, 2] - R_rel[..., 2, 0]
    skew_z = R_rel[..., 1, 0] - R_rel[..., 0, 1]
    sin_part = 0.5 * torch.sqrt(skew_x * skew_x + skew_y * skew_y + skew_z * skew_z + 1e-12)
    quat_dist = torch.atan2(sin_part, cos_part)

    assert p_dist.shape == quat_dist.shape
    return p_dist, quat_dist
