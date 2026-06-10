"""Tests for the COM polygon penalty math, hard gate, and post-IK mask."""
import pytest

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def _make_world():
    """Build a real TAMPWorld for blocks_t1."""
    return make_blocks_t1_world()


@needs_cuda
def test_compute_com_polygon_mask_excludes_extreme_lean():
    """Batched COM-in-polygon check returns the expected shape, classifies
    home poses as inside (all DOFs 0 → COM directly above the wheelbase
    center) and a bent-far-forward configuration as OUTSIDE the polygon.
    The bent config has deeply-bent Torso_Pitch + ankle_pitch + knee_pitch
    (mimicking the teetering pose we observed pre-fix)."""
    import torch
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    world = _make_world()
    full_names = list(world.kinematics.joint_names)
    home = world.q_init.detach().clone()
    # Build a configuration with deep forward bend.
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    bent = home.clone()
    bent[name_to_idx["Torso_Pitch"]]  = -1.7
    bent[name_to_idx["ankle_pitch"]]  = -0.5
    bent[name_to_idx["knee_pitch"]]   = +0.8
    q_batch = torch.stack([home, home, home, bent], dim=0)  # [4, full_dof]
    mask = compute_com_polygon_mask(world, q_batch)
    assert mask.shape == (4,), f"expected shape (4,), got {mask.shape}"
    assert bool(mask[:3].all()), (
        f"home poses should be inside polygon; mask={mask}"
    )
    assert not bool(mask[3]), f"deeply-bent pose should be OUTSIDE polygon; mask[3]={mask[3]}"


@needs_cuda
def test_ik_for_pose_com_safe_returns_valid_result():
    """End-to-end smoke check: _ik_for_pose_com_safe returns an IK result
    with the expected shape on a real grasp target. We don't assert
    everything-in-polygon (Layer 1 should help, but hard targets may
    still fail) — just that the wrapper runs without error and returns
    a usable result."""
    import torch
    from cutamp.particle_initialization import _ik_for_pose_com_safe
    world = _make_world()
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
                # Penalty values: 0.0 = inside polygon; 1e-2 ≈ 1cm outside.
                # tol from default_constraint_to_tol[ComPolygon.type] = 4e-4.
                "left_q1":  torch.tensor([0.0, 1e-2, 0.0, 0.0]),
                "right_q3": torch.tensor([0.0, 0.0, 0.0, 1e-2]),
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
    world = _make_world()
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

    world = _make_world()
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


def test_com_polygon_penalty_corner_equals_single_edge():
    # A COM ~1mm inside BOTH edges (near a corner) must score the same as
    # being on a single edge — NOT double. With the buggy summed barrier it
    # scores ~7.2e-4 (> tol 4e-4) and gets wrongly rejected; with the correct
    # max-over-axes barrier it scores ~3.6e-4 (< tol).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = torch.eye(4).unsqueeze(0)
    e = 0.001  # 1mm inside each edge
    com_world = torch.tensor([[0.10 - e, 0.15 - e, 0.0]])
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight)
    # nearest-edge barrier: weight * (margin - e)**2 = 1 * 0.019**2 = 3.61e-4
    assert pen.item() == pytest.approx((margin - e) ** 2, rel=1e-3)
    assert pen.item() < 4e-4  # passes the COM tol; the summed bug gave 7.22e-4


def test_com_polygon_penalty_corner_on_boundary_equals_tol():
    # COM exactly on the corner -> penalty == weight*margin**2 == COM tol 4e-4.
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = torch.eye(4).unsqueeze(0)
    com_world = torch.tensor([[0.10, 0.15, 0.0]])
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight)
    assert pen.item() == pytest.approx(weight * margin ** 2, rel=1e-4)  # 4e-4


def test_com_polygon_penalty_single_edge_equals_tol():
    # COM exactly on ONE edge (X), deep inside the other (Y) -> penalty == tol.
    # This is the calibration case the tol was derived from; max-over-axes must
    # leave it unchanged from the old summed behavior (only the X axis is active).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = torch.eye(4).unsqueeze(0)
    com_world = torch.tensor([[0.10, 0.0, 0.0]])  # on X edge, centered in Y
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight)
    assert pen.item() == pytest.approx(weight * margin ** 2, rel=1e-4)  # 4e-4


def test_com_polygon_penalty_center_weight_defaults_off():
    # Default center_weight=0 -> barrier-only: deep-inside COM scores exactly 0
    # (the hard-gate path relies on this).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    base_T = torch.eye(4).unsqueeze(0)
    com_world = torch.tensor([[0.03, 0.04, 0.0]])  # well inside inset rect
    pen = com_polygon_penalty(com_world, base_T, half, 0.02, 1.0)
    assert pen.item() == 0.0


def test_com_polygon_penalty_center_pull_active_deep_inside():
    # With center_weight>0 a deep-inside COM gets a nonzero penalty that grows
    # with distance from center: center_weight * sum((com/half)^2).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    base_T = torch.eye(4).unsqueeze(0)
    cw = 0.01
    far = torch.tensor([[0.06, 0.0, 0.0]])      # inside inset, barrier=0
    center = torch.tensor([[0.0, 0.0, 0.0]])
    pen_far = com_polygon_penalty(far, base_T, half, 0.02, 1.0, center_weight=cw)
    pen_center = com_polygon_penalty(center, base_T, half, 0.02, 1.0, center_weight=cw)
    assert pen_center.item() == pytest.approx(0.0)
    assert pen_far.item() == pytest.approx(cw * (0.06 / 0.10) ** 2, rel=1e-4)


def test_com_polygon_penalty_just_outside_exceeds_tol():
    # COM 1mm past one edge -> penalty > tol (outside quadratic + saturated barrier).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = torch.eye(4).unsqueeze(0)
    com_world = torch.tensor([[0.101, 0.0, 0.0]])  # 1mm past X edge
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight)
    # outside=0.001 -> 0.001**2; inside barrier saturates at (0.001+0.02)**2.
    # Sum: 1e-6 + 4.41e-4 ≈ 4.42e-4 > tol 4e-4.
    assert pen.item() > 4e-4
    assert pen.item() == pytest.approx(0.001 ** 2 + (0.001 + margin) ** 2, rel=1e-3)


@needs_cuda
def test_mask_matches_penalty_gate():
    # The post-IK mask must equal (penalty <= COM_TOL) elementwise, so the IK
    # COM-safe retry loop and the ConstraintChecker can never disagree.
    # COM_TOL is inside_weight * inside_margin^2 = 4e-4, calibrating the hard
    # constraint's tolerance against the mask's definition of 'inside'.
    import torch
    from cutamp.com_polygon_cost import (
        COM_TOL, compute_com_polygon_mask, compute_com_polygon_penalties,
    )
    world = _make_world()
    full_names = list(world.kinematics.joint_names)
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    home = world.q_init.detach().clone()
    # Deep forward lean that is genuinely out-of-hull: forward torso AND
    # forward ankle. (Torso_Pitch=-1.5 with knee=+0.6 is NOT outside — knee
    # bend re-centers the COM, which is the whole point of the CoM-aware IK.)
    bent = home.clone()
    bent[name_to_idx["Torso_Pitch"]] = -1.7
    bent[name_to_idx["ankle_pitch"]] = -0.5
    bent[name_to_idx["knee_pitch"]] = +0.8
    # Spread a batch of configs around home so some land near the polygon
    # edge, then append the deterministic [home, bent] pair so both
    # classifications are guaranteed present in the batch.
    torch.manual_seed(0)
    q = home.unsqueeze(0).repeat(32, 1).clone()
    q[:, 3:7] += 0.3 * torch.randn(32, 4, device=q.device)   # perturb body DOFs
    q = torch.cat([q, torch.stack([home, bent], dim=0)], dim=0)  # [34, full_dof]
    mask = compute_com_polygon_mask(world, q)
    pens = compute_com_polygon_penalties(world, {"q": q})["q"]
    gate = pens <= COM_TOL
    assert torch.equal(mask, gate), (
        f"Penalty-vs-mask disagree.\n  mask={mask.tolist()}\n"
        f"  penalties={pens.tolist()}  tol={COM_TOL}\n  gate={gate.tolist()}"
    )
    # Both classifications must actually be present.
    assert bool(mask[-2]), "home must classify as inside the polygon"
    assert not bool(mask[-1]), "bent must classify as outside the polygon"
