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


@needs_cuda
def test_ik_for_pose_com_safe_returns_valid_result():
    """End-to-end smoke check: _ik_for_pose_com_safe returns an IK result
    with the expected shape on a real grasp target. We don't assert
    everything-in-polygon (Layer 1 should help, but hard targets may
    still fail) — just that the wrapper runs without error and returns
    a usable result."""
    import torch
    from cutamp.particle_initialization import _ik_for_pose_com_safe
    world = _make_world(enable_com_polygon=True)
    # Build a [B=4, 4, 4] batch of reachable left-hand targets.
    # Pose: hand at (0.4, +0.2, 0.5) world — well within left arm reach.
    B = 4
    target = torch.eye(4, device=world.kinematics.device_cfg.device).unsqueeze(0).expand(B, 4, 4).contiguous()
    target[..., 0, 3] = 0.4
    target[..., 1, 3] = 0.2
    target[..., 2, 3] = 0.5
    result = _ik_for_pose_com_safe(world, target, "left", max_retries=2)
    # Result must have success + solution fields with batch dim B.
    assert hasattr(result, "success") and result.success.shape[0] == B
    assert hasattr(result, "solution") and result.solution.shape[0] == B


def test_constraint_checker_filters_com_violators():
    """ComPolygon registered as hard constraint must filter violating
    particles from the satisfying mask. No CUDA needed — exercises
    ConstraintChecker on a synthetic cost dict."""
    import torch
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.scripts.utils import default_constraint_to_tol
    from cutamp.task_planning.constraints import ComPolygon

    checker = ConstraintChecker(default_constraint_to_tol.copy())
    # 4 particles: 0 and 2 inside polygon on every conf; 1 fails on
    # left_q1; 3 fails on right_q3.
    cost_dict = {
        ComPolygon.type: {
            "type": "constraint",
            "constraints": [],
            "values": {
                "left_q1":  torch.tensor([0.0, 1.0, 0.0, 0.0]),
                "right_q3": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "left_q0":  torch.tensor([0.0, 0.0, 0.0, 0.0]),
            },
        },
    }
    mask = checker.get_mask(cost_dict, verbose=False)
    assert mask.tolist() == [True, False, True, False], (
        f"Expected COM-violators (idx 1, 3) filtered; got mask={mask.tolist()}"
    )


@needs_cuda
def test_curobo_batched_com_kernel_returns_per_batch_distinct():
    """Regression test for bundled cuRobo's batched COM kernel bug.

    Pre-fix (kinematics_forward_kernel.cuh: passing local_batch_offset
    instead of 0 as matAddrBase): only batch index 0 of robot_com was
    populated correctly; subsequent slots returned zeros or aliased
    garbage. Affected any robot with num_spheres ≥ 100 (T1=164, G1=674).

    The fix is at
    curobo/_src/curobolib/kernels/kinematics/kinematics_forward_kernel.cuh
    lines 316 and 396 (passing 0 instead of local_batch_offset to
    process_center_of_mass). This test asserts that a B=4 batched COM
    call returns the SAME per-batch COMs as four separate B=1 calls."""
    import torch
    from curobo.types import JointState
    world = _make_world(enable_com_polygon=True)
    kin = world.kinematics_with_com
    full_names = list(world.kinematics.joint_names)
    name_to_idx = {n: i for i, n in enumerate(full_names)}

    # Build four distinct configurations spanning the COM excursion range.
    home = world.q_init.detach().clone()
    bent_forward = home.clone()
    bent_forward[name_to_idx["Torso_Pitch"]] = -1.5
    bent_forward[name_to_idx["knee_pitch"]] = +0.6
    twisted = home.clone()
    twisted[name_to_idx["Waist_Yaw"]] = +0.8
    reach = home.clone()
    reach[name_to_idx["Left_Shoulder_Pitch"]] = -1.2

    configs = [home, bent_forward, twisted, reach]

    # Reference: per-sample B=1 COMs (always correct).
    ref_coms = []
    for q in configs:
        js = JointState.from_position(q.unsqueeze(0), joint_names=full_names)
        ks = kin.compute_kinematics(kin.get_active_js(js))
        ref_coms.append(ks.robot_com[..., :3].reshape(3).clone())
    ref_coms = torch.stack(ref_coms, dim=0)  # [4, 3]

    # Batched: all 4 in one call.
    q_batch = torch.stack(configs, dim=0)
    js_batch = JointState.from_position(q_batch, joint_names=full_names)
    ks_batch = kin.compute_kinematics(kin.get_active_js(js_batch))
    batched_coms = ks_batch.robot_com[..., :3].reshape(-1, 3)  # [4, 3]

    # Each batch slot's COM must match its single-call reference.
    max_diff = (batched_coms - ref_coms).abs().max().item()
    assert max_diff < 1e-5, (
        f"Batched COM kernel returned non-matching COMs (max diff {max_diff:.6f}).\n"
        f"  ref (B=1 stacked):\n{ref_coms.cpu().numpy()}\n"
        f"  batched (B=4):\n{batched_coms.cpu().numpy()}\n"
        f"Bundled cuRobo's batched COM kernel may have regressed — see "
        f"curobo/_src/curobolib/kernels/kinematics/kinematics_forward_kernel.cuh "
        f"lines 316 and 396 (process_center_of_mass matAddrBase arg)."
    )


@needs_cuda
def test_compute_com_polygon_penalties_is_differentiable():
    """The shared helper must return tensors that carry an autograd
    connection back to the input joint positions. This is the property
    that makes the hard ComPolygon constraint actually usable as an
    Adam gradient source (the prior (~mask).float() lost this)."""
    import torch
    from cutamp.com_polygon_cost import compute_com_polygon_penalties

    world = _make_world(enable_com_polygon=True)
    full_names = list(world.kinematics.joint_names)
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    home = world.q_init.detach().clone()

    # Use a configuration with the COM outside the support polygon so the
    # penalty is non-zero and the gradient is non-trivially non-zero.
    # Torso_Pitch=-1.5 shifts com_base[x] to ~0.1146 > half_extents[x]=0.1115.
    leaning = home.clone()
    leaning[name_to_idx["Torso_Pitch"]] = -1.5

    B = 4
    q = leaning.unsqueeze(0).expand(B, -1).contiguous().clone().requires_grad_(True)
    particles = {"left_q1": q}

    pens = compute_com_polygon_penalties(world, particles)
    assert "left_q1" in pens, f"Expected 'left_q1' key in result; got {list(pens.keys())}"
    p = pens["left_q1"]
    assert p.shape == (B,), f"Expected shape ({B},), got {p.shape}"
    assert p.requires_grad, "Penalty tensor must carry an autograd connection."

    # Backward through the sum; q.grad must be populated and non-trivial.
    p.sum().backward()
    assert q.grad is not None, "Expected q.grad to be populated after backward."
    # Leaning configuration has COM outside the polygon, so the gradient
    # through the FK chain and penalty function is non-zero.
    assert q.grad.abs().sum().item() > 0, (
        f"Expected non-zero gradient on q (COM is outside polygon); got all-zero grad."
    )


@needs_cuda
def test_compute_com_polygon_penalties_matches_mask_at_tolerance():
    """Parity between the mask helper and the penalty helper: a
    particle's penalty <= inside_weight * inside_margin^2 iff its COM
    is inside the polygon. Calibrates the hard constraint's tolerance
    (4e-4) against the mask's definition of 'inside'."""
    import torch
    from cutamp.com_polygon_cost import (
        compute_com_polygon_mask,
        compute_com_polygon_penalties,
    )

    world = _make_world(enable_com_polygon=True)
    full_names = list(world.kinematics.joint_names)
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    home = world.q_init.detach().clone()
    bent = home.clone()
    bent[name_to_idx["Torso_Pitch"]] = -1.5  # COM far forward -> outside polygon
    bent[name_to_idx["knee_pitch"]] = +0.6

    q = torch.stack([home, bent], dim=0)  # [2, full_dof]

    mask = compute_com_polygon_mask(world, q)  # [B] Bool
    pens = compute_com_polygon_penalties(world, {"left_q1": q})
    p = pens["left_q1"]  # [B] float
    tol = 1.0 * (0.02) ** 2  # inside_weight * inside_margin^2 = 4e-4

    # mask True <-> COM inside polygon <-> penalty <= tol
    inside_mask = (p <= tol)
    assert torch.equal(mask, inside_mask), (
        f"Penalty-vs-mask disagree.\n  mask={mask.tolist()}\n"
        f"  penalties={p.tolist()}  tol={tol}\n  inside_mask={inside_mask.tolist()}"
    )
