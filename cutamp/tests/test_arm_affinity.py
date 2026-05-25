"""Tests for arm-affinity priority in plan-skeleton search."""
import os
import pytest


needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _make_world():
    """Build a real TAMPWorld for blocks_t1. Used by tests that need real
    kinematics + scene state."""
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from curobo.types import DeviceCfg
    from cutamp.robots.t1 import t1_home
    import torch

    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    device_cfg = DeviceCfg()
    robot = load_robot_container("t1", device_cfg)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=device_cfg.device)
    return TAMPWorld(env=env, device_cfg=device_cfg, robot=robot, q_init=q_init)


@needs_cuda
def test_arm_home_ee_world_populated_after_init():
    """TAMPWorld.__init__ computes arm_home_ee_world for both arms.

    Each value is a 3-vector; left should be on +Y side, right on -Y side
    (T1 stands facing +X so left arm is to its left, which is +Y in world)."""
    import torch
    world = _make_world()
    assert hasattr(world, "arm_home_ee_world"), (
        "TAMPWorld must expose arm_home_ee_world after init"
    )
    assert set(world.arm_home_ee_world.keys()) == {"left", "right"}
    for arm in ("left", "right"):
        v = world.arm_home_ee_world[arm]
        assert isinstance(v, torch.Tensor), f"{arm} value must be a torch.Tensor"
        assert v.shape == (3,), f"{arm} value must be shape [3], got {tuple(v.shape)}"
    # Sanity: left arm is on +Y side, right arm on -Y side (T1 standing forward)
    assert world.arm_home_ee_world["left"][1].item() > 0, (
        "Left arm Y should be positive in world frame at home pose"
    )
    assert world.arm_home_ee_world["right"][1].item() < 0, (
        "Right arm Y should be negative in world frame at home pose"
    )
