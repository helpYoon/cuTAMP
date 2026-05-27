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


@needs_cuda
def test_compute_com_polygon_mask_basic():
    """Verify batched COM-in-polygon check returns the expected shape +
    correctly classifies a home-pose batch as inside the polygon."""
    import torch
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    world = _make_world(enable_com_polygon=True)
    # At home pose all DOFs are 0; COM is directly above the wheelbase
    # center → inside polygon for sure. Build a [B=4, full_dof] batch all
    # at home and assert all four come back True.
    home = world.q_init.detach().clone()
    B = 4
    q_batch = home.unsqueeze(0).expand(B, -1).contiguous()
    mask = compute_com_polygon_mask(world, q_batch)
    assert mask.shape == (B,), f"expected shape ({B},), got {mask.shape}"
    assert bool(mask.all()), (
        f"home pose should be inside polygon; got mask={mask}"
    )


@needs_cuda
def test_compute_com_polygon_mask_excludes_extreme_lean():
    """A bent-far-forward configuration should be classified as OUTSIDE
    the polygon. Constructs a synthetic q_batch with deeply-bent
    Torso_Pitch + ankle_pitch + knee_pitch (mimicking the teetering pose
    we observed pre-fix)."""
    import torch
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    world = _make_world(enable_com_polygon=True)
    full_names = list(world.kinematics.joint_names)
    home = world.q_init.detach().clone()
    # Build a configuration with deep forward bend.
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    bent = home.clone()
    bent[name_to_idx["Torso_Pitch"]]  = -1.7
    bent[name_to_idx["ankle_pitch"]]  = -0.5
    bent[name_to_idx["knee_pitch"]]   = +0.8
    q_batch = torch.stack([home, bent], dim=0)  # [2, full_dof]
    mask = compute_com_polygon_mask(world, q_batch)
    assert mask.shape == (2,)
    assert bool(mask[0]),    f"home should be inside polygon; mask[0]={mask[0]}"
    assert not bool(mask[1]), f"deeply-bent pose should be OUTSIDE polygon; mask[1]={mask[1]}"
