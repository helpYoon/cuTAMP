"""Shared pytest fixtures for cuTAMP tests (cuRobo v0.8 single-MotionPlanner)."""

import os

import pytest
import torch


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
    """Release GPU memory after each test. Integration tests build full
    TAMPWorld stacks (several GiB each); without reclaiming between tests,
    two heavy tests in one process OOM a 24 GiB card. Harmless for CPU tests."""
    yield
    import gc
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
