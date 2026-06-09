"""Tests for CoM-aware seed-IK (LM residual fork). Spec:
docs/superpowers/specs/2026-06-09-com-aware-ik-design.md"""
import os
import pytest
import torch

needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _world(enable_com_aware_ik: bool):
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from cutamp.robots.t1 import t1_home
    from curobo.types import DeviceCfg
    dc = DeviceCfg()
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    robot = load_robot_container("t1", dc)
    qh = torch.as_tensor(t1_home, dtype=torch.float32, device=dc.device)
    return TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=qh,
                     enable_com_polygon=True, enable_com_aware_ik=enable_com_aware_ik)


@needs_cuda
def test_flag_off_seed_cfg_weight_zero_by_default():
    world = _world(enable_com_aware_ik=False)
    cfg = world.ik_solver.seed_ik_solver.config
    assert cfg.com_support_weight == 0.0


@needs_cuda
def test_flag_on_seed_cfg_carries_com_params():
    from cutamp.com_polygon_cost import COM_HALF_EXTENTS, COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT
    from cutamp.robots.t1 import T1_COM_IK_WEIGHT
    world = _world(enable_com_aware_ik=True)
    cfg = world.ik_solver.seed_ik_solver.config
    assert cfg.com_support_weight == T1_COM_IK_WEIGHT > 0.0
    assert list(cfg.com_half_extents) == list(COM_HALF_EXTENTS)
    assert cfg.com_inside_margin == COM_INSIDE_MARGIN
    assert cfg.com_inside_weight == COM_INSIDE_WEIGHT
    assert cfg.com_center_weight == 0.0
    assert cfg.com_base_link_name == "mobile_base_link"
