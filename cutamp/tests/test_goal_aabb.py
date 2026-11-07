"""``approximate_goal_aabb`` on a rotated cuboid: a true world AABB around all eight corners (CPU only)."""

import itertools
import math

import torch
from curobo.scene import Cuboid

from cutamp.utils.common import approximate_goal_aabb, pose_list_to_mat4x4, transform_points


def test_rotated_cuboid_aabb_contains_its_corners():
    # 0.7 rad about (1, 2, 3)/|.|: rotated about every axis, so no face stays axis-aligned.
    axis = torch.tensor([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    quat_wxyz = [math.cos(0.35), *(math.sin(0.35) * axis).tolist()]
    dims = [0.3, 0.1, 0.05]
    cuboid = Cuboid(name="rotated", pose=[0.4, -0.2, 0.8, *quat_wxyz], dims=dims)

    aabb = approximate_goal_aabb(cuboid)
    assert aabb.shape == (2, 3) and aabb.dtype == torch.float32
    assert (aabb[0] < aabb[1]).all(), aabb

    corners = torch.tensor(list(itertools.product(*[(-d / 2, d / 2) for d in dims])))
    corners = transform_points(corners, pose_list_to_mat4x4(cuboid.pose))
    assert (corners >= aabb[0] - 1e-6).all() and (corners <= aabb[1] + 1e-6).all(), (corners, aabb)
    # Tight: each face of the box touches a corner.
    torch.testing.assert_close(aabb, torch.stack([corners.min(0).values, corners.max(0).values]))
