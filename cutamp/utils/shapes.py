# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import time
from collections import defaultdict

import torch
import trimesh
from _heapq import heappush, heappop

from curobo.scene import Obstacle

_log = logging.getLogger(__name__)


def sample_greedy_surface_spheres(
    obj: Obstacle,
    n_spheres: int,
    sphere_radius: float,
    n_samples: int = 1000,
) -> torch.Tensor:
    """
    Samples approximately non-overlapping surface spheres on a 3D object using a greedy algorithm. Thanks Caelan!

    Samples points on the object's surface mesh, constructs a neighborhood graph using a KD-tree,
    and greedily selects points that are far apart (non-overlapping) based on radius.
    """
    start_time = time.perf_counter()
    mesh = obj.get_trimesh_mesh()  # in the object's frame: cuRobo applies obj.pose only with transform_with_pose=True

    # Sample surface. The fixed seed makes the spheres part of the task's geometry: unseeded, trimesh draws from
    # numpy's global RNG, which nothing seeds, so every process planned against different spheres (known-bugs N4).
    n_samples = max(n_spheres, n_samples)
    n_pts = trimesh.sample.sample_surface(mesh, n_samples, seed=0)[0]
    cloud = trimesh.PointCloud(n_pts)

    # Build neighbor graph
    pairs = cloud.kdtree.query_pairs(sphere_radius, eps=1e-6)
    neighbors = defaultdict(set)
    for idx1, idx2 in pairs:
        neighbors[idx1].add(idx2)
        neighbors[idx2].add(idx1)
    # print(f"Neighbors) Samples: {n_samples} m | Elapsed: {time.perf_counter() - start_time:.3f}")

    # Greedy selection
    selected = []
    queue = []
    for idx in neighbors:
        heappush(queue, (-len(neighbors[idx]), idx))

    while queue and (len(selected) < n_spheres):
        num, idx = heappop(queue)
        if len(neighbors[idx]) != -num:
            heappush(queue, (-len(neighbors[idx]), idx))
            continue

        # Remove selected point and update neighbors
        for idx2 in list(neighbors[idx]):
            for idx3 in neighbors[idx2]:
                neighbors[idx3].discard(idx2)
            del neighbors[idx2]
        del neighbors[idx]
        selected.append(idx)

    # Convert selected points to tensor format
    selected_points = n_pts[selected]
    radii = torch.full((len(selected_points), 1), sphere_radius)

    # Combine points and radii into single tensor
    result = torch.cat([torch.tensor(selected_points, dtype=torch.float32), radii], dim=1)
    _log.debug(f"Sampled {len(result)} spheres for {obj.name} in {time.perf_counter() - start_time:.3f}s")
    return result
