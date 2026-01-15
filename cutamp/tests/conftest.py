"""
Shared pytest fixtures for cuTAMP tests.
"""
import pytest
import torch
import os


@pytest.fixture(scope="session")
def cuda_available():
    """Check CUDA availability."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture
def tensor_args(cuda_available):
    """Shared tensor args fixture."""
    from curobo.types.base import TensorDeviceType
    return TensorDeviceType()


@pytest.fixture
def t1_assets_path():
    """Path to T1 robot assets directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "robots", "assets", "t1_description"
    )


@pytest.fixture
def t1_left_config_path(t1_assets_path):
    """Path to left arm 11 DOF config."""
    return os.path.join(t1_assets_path, "t1_left_11dof.yml")


@pytest.fixture
def t1_right_config_path(t1_assets_path):
    """Path to right arm 11 DOF config."""
    return os.path.join(t1_assets_path, "t1_right_11dof.yml")


@pytest.fixture
def t1_left_config(t1_left_config_path):
    """Load left arm config."""
    from curobo.util_file import load_yaml
    return load_yaml(t1_left_config_path)


@pytest.fixture
def t1_right_config(t1_right_config_path):
    """Load right arm config."""
    from curobo.util_file import load_yaml
    return load_yaml(t1_right_config_path)
