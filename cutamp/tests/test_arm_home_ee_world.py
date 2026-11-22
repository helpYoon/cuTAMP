"""TAMPWorld.arm_home_ee_world, each palm's home position, which the A* (task_planning/search.py)
reads."""
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


@needs_cuda
def test_arm_home_ee_world_populated_after_init():
    """TAMPWorld.__init__ computes arm_home_ee_world for both arms.

    Each value is a 3-vector; left should be on +Y side, right on -Y side
    (T1 stands facing +X so left arm is to its left, which is +Y in world)."""
    import torch
    world = make_blocks_t1_world()
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
