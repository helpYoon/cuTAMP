# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Sphere-vs-scene collision cost for cuTAMP: spheres ``[batch, horizon, num_spheres, 4]`` in, a cost
``[batch, horizon]`` out, through cuRobo's ``SceneCollision``.

NOTE: ``SceneCollision`` is private (``curobo._src.geom.collision``). cuRobo v0.9.0's only public sphere query,
``curobo.collision_checking.RobotCollisionChecker.get_collision_constraint``, comes bundled with a robot Kinematics.
"""

import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo.scene import Scene
from curobo.types import DeviceCfg


class WorldCollisionCost:
    """Callable: ``cost(spheres) -> [batch, horizon]`` collision cost (non-negative, summed across spheres)."""

    def __init__(self, scene: Scene, device_cfg: DeviceCfg, activation_distance: float):
        self.scene_collision = SceneCollision.from_config(SceneCollisionCfg(device_cfg=device_cfg, scene_model=scene))
        self.device_cfg = device_cfg
        self.weight = device_cfg.to_device([1.0])
        self.activation_distance = device_cfg.to_device([activation_distance])

    def __call__(self, spheres: torch.Tensor) -> torch.Tensor:
        """The collision cost of query spheres ``[batch, horizon, num_spheres, 4]``."""
        # Fresh buffer per call: the query's autograd Function saves the buffer's `gradient` for backward and returns
        # its `distance` itself, and the Warp kernel writes both in place, past autograd's version counter. A reused
        # buffer would silently overwrite an earlier query's gradient and result.
        buffer = CollisionBuffer.from_shape(spheres.shape, self.device_cfg)
        per_sphere = self.scene_collision.get_sphere_distance_raw(
            spheres, buffer, self.weight, self.activation_distance, return_loss=True,
        )
        return per_sphere.sum(dim=-1)
