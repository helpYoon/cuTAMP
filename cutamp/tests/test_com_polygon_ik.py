"""Tests for the two-layer COM cost on the IK solver."""
import os
import pytest


needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _make_world(enable_com_polygon: bool = True):
    """Build a real TAMPWorld for blocks_t1 with optional COM toggle."""
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
    return TAMPWorld(
        env=env, device_cfg=device_cfg, robot=robot, q_init=q_init,
        enable_com_polygon=enable_com_polygon,
    )


def _ik_extra_costs(world):
    """Return the union of _extra_costs dicts across all IK rollout cost managers."""
    from cutamp._curobo_internals import iter_rollouts
    names = set()
    for rollout in iter_rollouts(world.ik_solver):
        for mgr in (
            getattr(rollout, "cost_manager", None),
            getattr(rollout, "metrics_cost_manager", None),
        ):
            if mgr is None:
                continue
            extras = getattr(mgr, "_extra_costs", {}) or {}
            names.update(extras.keys())
    return names


@needs_cuda
def test_ik_solver_has_com_polygon_extra_cost_when_enabled():
    """Default world (enable_com_polygon=True) registers com_polygon on
    the IK solver's rollouts via add_extra_cost."""
    world = _make_world(enable_com_polygon=True)
    assert "com_polygon" in _ik_extra_costs(world), (
        "IK solver should have com_polygon in its _extra_costs when "
        "enable_com_polygon=True"
    )


@needs_cuda
def test_ik_solver_no_com_polygon_when_disabled():
    """enable_com_polygon=False skips IK cost registration entirely."""
    world = _make_world(enable_com_polygon=False)
    assert "com_polygon" not in _ik_extra_costs(world), (
        "IK solver should NOT have com_polygon registered when "
        "enable_com_polygon=False"
    )
