"""Shared pytest fixtures for cuTAMP tests (cuRobo v0.8 single-MotionPlanner)."""

import gc
import os

import pytest
import torch

#: Shared GPU-required marker (CUDA device visible or /dev/nvidia0 present).
needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def make_blocks_t1_world(**tamp_world_kwargs):
    """Build the standard blocks_t1 TAMPWorld used by GPU integration tests."""
    import torch
    from curobo.types import DeviceCfg
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.robots import load_robot_container
    from cutamp.robots.t1 import t1_home
    from cutamp.tamp_world import TAMPWorld
    dc = DeviceCfg()
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    robot = load_robot_container("t1", dc)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=dc.device)
    return TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=q_init, **tamp_world_kwargs)


@pytest.fixture(scope="session")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture
def device_cfg(cuda_available):
    from curobo.types import DeviceCfg
    return DeviceCfg()


@pytest.fixture
def t1_assets_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "robots", "assets", "t1_description",
    )


@pytest.fixture
def t1_planar_base_config_path(t1_assets_path):
    return os.path.join(t1_assets_path, "t1_planar_base.yml")


@pytest.fixture
def t1_planar_base_config(t1_planar_base_config_path):
    from curobo._src.util.config_io import load_yaml
    return load_yaml(t1_planar_base_config_path)


@pytest.fixture(autouse=True)
def _free_cuda_between_tests():
    """Release GPU memory after heavy tests. Integration tests build full
    TAMPWorld stacks (several GiB each); without reclaiming between tests,
    two heavy tests in one process OOM a 24 GiB card. Gated on reserved
    memory so the dozens of CPU/light tests skip the gc.collect() tax."""
    yield
    if torch.cuda.is_available() and torch.cuda.memory_reserved() > (1 << 30):
        gc.collect()
        torch.cuda.empty_cache()
