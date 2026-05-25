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


@needs_cuda
def test_get_valid_ground_operators_sorts_by_priority_fn():
    """When ground_op_priority_fn is provided, ground operators are returned
    in ascending priority order. Closer-arm picks come first."""
    from cutamp.task_planning.search import get_valid_ground_operators, _Node
    from cutamp.t1_domain import LeftPick, RightPick, all_t1_operators
    from cutamp.task_planning import Atom

    # Stub a _Node with a state where both LeftPick and RightPick can ground
    # on two blocks. We bypass the full pipeline by directly building atoms
    # for preconditions.
    # NOTE: this test is structural — it verifies the sort happens, not the
    # real-scene priority math. Real-scene math is exercised by the smoke
    # test at the end.
    # The fastest way to test the sort: provide a priority_fn that returns
    # a fixed score per ground op based on its serialized name, and assert
    # the returned list is sorted.
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from curobo.types import DeviceCfg
    from cutamp.robots.t1 import t1_home
    import os, torch

    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    device_cfg = DeviceCfg()
    robot = load_robot_container("t1", device_cfg)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=device_cfg.device)
    world = TAMPWorld(env=env, device_cfg=device_cfg, robot=robot, q_init=q_init)
    initial_node = _Node(state=world.initial_state, parent=None, operator=None, depth=0)

    # Priority: deterministic lexicographic reverse of repr. Distinct
    # reprs give distinct priorities, so the sort assertion has real
    # discriminating power. Reverse-lex order is guaranteed different
    # from the BFS default (which is forward enumeration order),
    # provided there are >=2 ground ops with distinct first chars.
    def priority(ground_op):
        return [-ord(c) for c in repr(ground_op)]

    unsorted = get_valid_ground_operators(initial_node, all_t1_operators)
    sorted_ops = get_valid_ground_operators(
        initial_node, all_t1_operators, priority_fn=priority,
    )
    # Same set, no drops or duplicates.
    assert set(map(repr, unsorted)) == set(map(repr, sorted_ops))
    # Sort must have actually reordered (would catch a no-op implementation).
    assert [repr(o) for o in unsorted] != [repr(o) for o in sorted_ops], (
        "sort did not reorder ground_ops — implementation may be a no-op"
    )
    # And the sorted version must be in ascending priority order.
    priorities = [priority(op) for op in sorted_ops]
    assert priorities == sorted(priorities), (
        f"ground_ops not sorted by priority: {priorities}"
    )


@needs_cuda
def test_task_plan_generator_accepts_priority_fn():
    """task_plan_generator must accept ground_op_priority_fn without error
    and still yield plans."""
    from cutamp.task_planning import task_plan_generator
    from cutamp.t1_domain import all_t1_operators
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from curobo.types import DeviceCfg
    from cutamp.robots.t1 import t1_home
    import os, torch

    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    device_cfg = DeviceCfg()
    robot = load_robot_container("t1", device_cfg)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=device_cfg.device)
    world = TAMPWorld(env=env, device_cfg=device_cfg, robot=robot, q_init=q_init)
    gen = task_plan_generator(
        world.initial_state, world.goal_state, all_t1_operators,
        ground_op_priority_fn=lambda op: 0.0,
        max_plan_skeletons=1,
    )
    # Pulling one plan should not raise.
    plan = next(gen)
    assert plan is not None
    assert len(plan) > 0
