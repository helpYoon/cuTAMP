# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generic-sphere world-collision cost for cuTAMP (cuRobo v0.8).

The old API (``PrimitiveCollisionCost`` from ``curobo.rollout.cost.*``) is gone
in v0.8. We now wrap the lower-level ``SceneCollision`` + ``CollisionChecker``
to expose a callable ``cost(spheres) -> distance`` matching the prior cuTAMP
contract: input ``spheres`` is ``[batch, horizon, num_spheres, 4]``, output is
``[batch, horizon, num_spheres]`` collision distance.

NOTE: pulls from ``curobo._src.geom.collision`` (private) until cuRobo exposes
a public sphere-vs-scene query.
"""

import logging

import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo.scene import Scene
from curobo.types import DeviceCfg

_log = logging.getLogger(__name__)


class WorldCollisionCost:
    """Callable: ``cost(spheres) -> [batch, horizon]`` collision cost
    (non-negative, summed across spheres).
    """

    def __init__(
        self,
        scene: Scene,
        device_cfg: DeviceCfg,
        activation_distance: float,
        weight: float = 1.0,
    ):
        if activation_distance < 0.0:
            raise ValueError(f"Collision activation distance must be >= 0.0, got {activation_distance}")
        cfg = SceneCollisionCfg(device_cfg=device_cfg, scene_model=scene)
        self.scene_collision = SceneCollision.from_config(cfg)
        self.device_cfg = device_cfg
        self.weight = device_cfg.to_device([weight])
        self.activation_distance = device_cfg.to_device([activation_distance])

    def __call__(
        self, spheres: torch.Tensor, return_loss: bool = True,
    ) -> torch.Tensor:
        """Query collision cost for a batch of spheres.

        Args:
            spheres: Query spheres ``[batch, horizon, num_spheres, 4]``.
            return_loss: True if caller will scale the result before backward.
        """
        if spheres.ndim != 4 or spheres.shape[-1] != 4:
            raise ValueError(
                f"spheres must be [batch, horizon, num_spheres, 4], got {tuple(spheres.shape)}"
            )

        # Fresh buffer per call: the saved-for-backward `gradient` tensor inside
        # CollisionBuffer is mutated in-place by the underlying autograd
        # Function, which trips PyTorch's saved-tensor checks if a single
        # buffer is reused across forward passes.
        buffer = CollisionBuffer.from_shape(spheres.shape, self.device_cfg)

        per_sphere = self.scene_collision.checker.get_sphere_distance(
            scene=self.scene_collision.data,
            query_sphere=spheres,
            collision_buffer=buffer,
            weight=self.weight,
            activation_distance=self.activation_distance,
            return_loss=return_loss,
        )
        return per_sphere.clamp(min=0.0).sum(dim=-1)


def get_world_collision_cost(
    scene: Scene,
    device_cfg: DeviceCfg,
    activation_distance: float,
    weight: float = 1.0,
) -> WorldCollisionCost:
    """Build a callable world-collision cost for the given Scene."""
    return WorldCollisionCost(scene, device_cfg, activation_distance, weight)


def get_collision_checker(
    scene: Scene,
    device_cfg: DeviceCfg,
    max_distance: float = 0.1,
) -> SceneCollision:
    """Build a SceneCollision (low-level handle) for the given Scene.

    Use the higher-level ``WorldCollisionCost`` when possible; this is exposed
    for code paths that want direct access to the scene checker.
    """
    cfg = SceneCollisionCfg(
        device_cfg=device_cfg,
        scene_model=scene,
        max_distance=max_distance,
    )
    return SceneCollision.from_config(cfg)
