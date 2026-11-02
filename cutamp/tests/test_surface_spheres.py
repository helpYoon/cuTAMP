"""The movables' collision spheres (utils/shapes.py), which TAMPWorld fits once per object."""
import os

import numpy as np
import torch

from cutamp.envs.utils import get_env_dir, load_env
from cutamp.utils.shapes import sample_greedy_surface_spheres


def test_block_spheres_ignore_numpy_global_state():
    # They came from numpy's unseeded global RNG, so two runs with the same --seed planned against different
    # blocks and diverged within a few steps (known-bugs N4).
    block = load_env(os.path.join(get_env_dir(), "blocks_t1.yml")).movables[0]
    state = np.random.get_state()
    try:
        np.random.seed(1)
        first = sample_greedy_surface_spheres(block, n_spheres=50, sphere_radius=0.005)
        np.random.seed(2)
        second = sample_greedy_surface_spheres(block, n_spheres=50, sphere_radius=0.005)
    finally:
        np.random.set_state(state)
    assert torch.equal(first, second)
