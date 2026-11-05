"""Tests for the COM polygon penalty math, hard gate, and post-IK mask."""
import pytest

from cutamp.robots.t1 import SIMPLIFIED, get_model
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda

#: A lean from home that carries the CoM far past the support's front edge, per planning model ({joint: delta}): the
#: simplified model bows at its Torso_Pitch, ankle and knee (the teetering pose seen before the CoM fix); the real legs'
#: floating base carries the whole body 30 cm ahead of the stance its support region stays on.
EXTREME_LEAN = {"simplified": {"Torso_Pitch": -1.7, "ankle_pitch": -0.5, "knee_pitch": 0.8},
                "real_legs": {"base_j_x": 0.3}}

#: The spread test_mask_matches_penalty_gate adds to home, per planning model ({joint: sigma}, rad or m), so that its
#: rows straddle the keyframe rectangle's edge: the simplified model's CoM moves with its legs and trunk; the real
#: legs' trunk rides the floating base, whose x and y carry the CoM across the stance's edges.
EDGE_SPREAD = {"simplified": dict.fromkeys(SIMPLIFIED.joint_names[SIMPLIFIED.body], 0.3),
               "real_legs": {"base_j_x": 0.1, "base_j_y": 0.1}}


def _moved(world, moves: dict):
    """``world``'s home with each joint in ``moves`` ({name: delta}) moved."""
    names = list(world.kinematics.joint_names)
    q = world.q_init.detach().clone()
    for name, delta in moves.items():
        q[names.index(name)] += delta
    return q


@needs_cuda
def test_compute_com_polygon_mask_excludes_extreme_lean():
    """Batched COM-in-polygon check returns the expected shape, classifies
    home poses as inside (the CoM over the support's centre) and the
    model's EXTREME_LEAN as OUTSIDE the polygon."""
    import torch
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    world = make_blocks_t1_world()
    home = world.q_init.detach().clone()
    bent = _moved(world, EXTREME_LEAN[world.model.name])
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
    world = make_blocks_t1_world()
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
    """Regression test for cuRobo's batched COM kernel.

    History: at the NVlabs PR #626 base, the kernel double-offset the
    cumulative-transform address for batch slots > 0 (pre-offset pointer
    AND nonzero matAddrBase passed to process_center_of_mass) — only batch
    index 0 of robot_com was correct for robots with num_spheres >= 100
    (T1=164). Our fork patched it; PR #678 then rewrote the kernel
    (write_center_of_mass_tile) with the single-offset convention applied
    everywhere, fixing it upstream, and the fork patch was dropped. This
    test remains as the guard: a B=4 batched COM call must return the SAME
    per-batch COMs as four separate B=1 calls."""
    import torch
    from curobo.types import JointState
    world = make_blocks_t1_world()
    kin = world.kinematics
    full_names = list(world.kinematics.joint_names)

    # Build four distinct configurations spanning the COM excursion range: home, the lean, every leg and trunk
    # joint bent, and a left-arm reach.
    home = world.q_init.detach().clone()
    bent_forward = _moved(world, EXTREME_LEAN[world.model.name])
    bent_body = home.clone()
    bent_body[world.model.body] += 0.3
    reach = _moved(world, {world.model.joint_names[world.model.left_arm][0]: -1.2})

    configs = [home, bent_forward, bent_body, reach]

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
        f"Bundled cuRobo's batched COM kernel may have regressed — check the "
        f"per-batch matAddrBase handling in write_center_of_mass_tile "
        f"(curobo/_src/curobolib/kernels/kinematics/kinematics_forward_helper.cuh)."
    )


@needs_cuda
def test_compute_com_polygon_penalties_is_differentiable():
    """The shared helper must return tensors that carry an autograd
    connection back to the input joint positions. This is the property
    that makes the hard ComPolygon constraint actually usable as an
    Adam gradient source (the prior (~mask).float() lost this)."""
    import torch
    from cutamp.com_polygon_cost import COM_TOL, compute_com_polygon_penalties

    world = make_blocks_t1_world()

    # Use a configuration with the COM outside the support polygon so the
    # penalty is non-zero and the gradient is non-trivially non-zero.
    leaning = _moved(world, EXTREME_LEAN[world.model.name])

    B = 4
    q = leaning.unsqueeze(0).expand(B, -1).contiguous().clone().requires_grad_(True)
    particles = {"left_q1": q}

    pens = compute_com_polygon_penalties(world, particles)
    assert "left_q1" in pens, f"Expected 'left_q1' key in result; got {list(pens.keys())}"
    p = pens["left_q1"]
    assert p.shape == (B,), f"Expected shape ({B},), got {p.shape}"
    assert p.requires_grad, "Penalty tensor must carry an autograd connection."
    assert float(p[0]) > COM_TOL, f"premise: the lean must put the COM outside; penalty {float(p[0]):.6f}"

    # Backward through the sum; q.grad must be populated and non-trivial.
    p.sum().backward()
    assert q.grad is not None, "Expected q.grad to be populated after backward."
    # Leaning configuration has COM outside the polygon, so the gradient
    # through the FK chain and penalty function is non-zero.
    assert q.grad.abs().sum().item() > 0, (
        f"Expected non-zero gradient on q (COM is outside polygon); got all-zero grad."
    )


@needs_cuda
def test_compute_com_polygon_penalties_grads_match_per_conf():
    """With several confs, each conf's gradient must equal its gradient
    computed alone. world.kinematics keeps FK outputs and backward state in
    buffers shared by same-shape calls, so one FK call per conf left every
    backward but the last reading the last call's link transforms."""
    import torch
    from cutamp.com_polygon_cost import compute_com_polygon_penalties

    from curobo.types import JointState

    torch.manual_seed(0)
    world = make_blocks_t1_world()
    home = world.q_init.detach().clone()
    qs = {f"left_q{i}": (home + 0.4 * torch.randn(16, home.numel(), device=home.device)).requires_grad_(True)
          for i in range(3)}
    # An earlier FK of the batched call's shape (other q, so an overwrite shows).
    earlier = world.kinematics.compute_kinematics(JointState.from_position(home.expand(48, -1).contiguous())).robot_spheres
    snapshot = earlier.clone()
    pens = compute_com_polygon_penalties(world, qs)
    assert torch.equal(earlier, snapshot), "the batched FK overwrote an earlier same-shape FK's outputs"
    assert all((p > 0).any() for p in pens.values())  # non-trivial gradients
    sum(p.sum() for p in pens.values()).backward()
    for name, q in qs.items():
        together, q.grad = q.grad.clone(), None  # clone: FK may return grads in its buffers
        compute_com_polygon_penalties(world, {name: q})[name].sum().backward()
        torch.testing.assert_close(together, q.grad)


@needs_cuda
def test_retract_collision_grads_match_per_conf():
    """Same check for the retract-collision FK in CostFunction.collision_costs.
    In blocks_t1 the batched retract FK has the rollout FK's shape, so it must
    also leave the rollout's FK outputs alone."""
    import torch
    from cutamp.config import TAMPConfiguration
    from cutamp.cost_function import CostFunction
    from cutamp.particle_initialization import ParticleInitializer
    from cutamp.rollout import RolloutFunction
    from cutamp.t1_domain import all_t1_operators
    from cutamp.task_planning import task_plan_generator

    world = make_blocks_t1_world()
    config = TAMPConfiguration(num_particles=8)
    skeleton = next(task_plan_generator(world.initial_state, world.goal_state, all_t1_operators))
    particles = ParticleInitializer(world, config)(skeleton)
    cost_fn = CostFunction(skeleton, world, config)
    retract_info = list(cost_fn.retract_info)
    assert len(retract_info) > 1
    # Init leaves every retract conf identical, which would hide the aliasing.
    torch.manual_seed(0)
    qs = {name: (particles[name] + 0.3 * torch.randn_like(particles[name])).requires_grad_(True)
          for name, _, _ in retract_info}
    particles.update(qs)
    rollout = RolloutFunction(skeleton, world, config)(particles)
    rollout_spheres = rollout["robot_spheres"].detach().clone()
    # One big sphere per movable, around the robot, so every retract conf
    # overlaps the movables and gets a non-zero gradient.
    big = torch.tensor([0.3, 0.0, 0.8, 0.6], device=world.device)
    obj_to_spheres = {name: big.expand(8, pose.shape[1], 1, 4) for name, pose in rollout["obj_to_pose"].items()}

    def grads(info):
        cost_fn.retract_info, cost_fn._particles = info, particles
        vals = cost_fn.collision_costs(rollout, obj_to_spheres)["values"]
        loss = sum(vals[f"{name}_{k}"].sum() for name, _, _ in info for k in ("world", "movables"))
        # Clone: FK returns gradients in its own buffers, which later calls reuse.
        return [g.clone() for g in torch.autograd.grad(loss, [qs[name] for name, _, _ in info])]

    together = grads(retract_info)
    assert torch.equal(rollout["robot_spheres"], rollout_spheres), "retract FK overwrote the rollout's FK outputs"
    for info, g in zip(retract_info, together):
        assert g.abs().sum() > 0
        torch.testing.assert_close(g, grads([info])[0])


def _support_centre_at_origin():
    """A base pose that puts the support rectangle's centre (COM_CENTER) at the world origin, so the
    points below are offsets from it."""
    import torch
    from cutamp.com_polygon_cost import COM_CENTER
    base_T = torch.eye(4).unsqueeze(0)
    base_T[0, :2, 3] = -torch.tensor(COM_CENTER)
    return base_T


def test_com_polygon_penalty_corner_equals_single_edge():
    # A COM ~1mm inside BOTH edges (near a corner) must score the same as
    # being on a single edge — NOT double. With the buggy summed barrier it
    # scores ~7.2e-4 (> tol 4e-4) and gets wrongly rejected; with the correct
    # max-over-axes barrier it scores ~3.6e-4 (< tol).
    import torch
    from cutamp.com_polygon_cost import COM_CENTER, com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = _support_centre_at_origin()
    e = 0.001  # 1mm inside each edge
    com_world = torch.tensor([[0.10 - e, 0.15 - e, 0.0]])
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight, COM_CENTER)
    # nearest-edge barrier: weight * (margin - e)**2 = 1 * 0.019**2 = 3.61e-4
    assert pen.item() == pytest.approx((margin - e) ** 2, rel=1e-3)
    assert pen.item() < 4e-4  # passes the COM tol; the summed bug gave 7.22e-4


def test_com_polygon_penalty_corner_on_boundary_equals_tol():
    # COM exactly on the corner -> penalty == weight*margin**2 == COM tol 4e-4.
    import torch
    from cutamp.com_polygon_cost import COM_CENTER, com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = _support_centre_at_origin()
    com_world = torch.tensor([[0.10, 0.15, 0.0]])
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight, COM_CENTER)
    assert pen.item() == pytest.approx(weight * margin ** 2, rel=1e-4)  # 4e-4


def test_com_polygon_penalty_single_edge_equals_tol():
    # COM exactly on ONE edge (X), deep inside the other (Y) -> penalty == tol.
    # This is the calibration case the tol was derived from; max-over-axes must
    # leave it unchanged from the old summed behavior (only the X axis is active).
    import torch
    from cutamp.com_polygon_cost import COM_CENTER, com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = _support_centre_at_origin()
    com_world = torch.tensor([[0.10, 0.0, 0.0]])  # on X edge, centered in Y
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight, COM_CENTER)
    assert pen.item() == pytest.approx(weight * margin ** 2, rel=1e-4)  # 4e-4


def test_com_tol_is_the_on_boundary_penalty():
    # The gate's tolerance is the barrier's on-boundary value, derived from its constants so a new margin or
    # weight moves both (known-bugs N5). Pinned at 4e-4: the derivation must leave results bit-identical.
    from cutamp.com_polygon_cost import COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT, COM_TOL
    assert COM_TOL == COM_INSIDE_WEIGHT * COM_INSIDE_MARGIN ** 2 == 4e-4


def test_com_polygon_penalty_deep_inside_is_zero():
    # Barrier-only: deep-inside COM scores exactly 0 (the hard-gate path
    # relies on this).
    import torch
    from cutamp.com_polygon_cost import COM_CENTER, com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    base_T = _support_centre_at_origin()
    com_world = torch.tensor([[0.03, 0.04, 0.0]])  # well inside inset rect
    pen = com_polygon_penalty(com_world, base_T, half, 0.02, 1.0, COM_CENTER)
    assert pen.item() == 0.0


def test_com_polygon_penalty_just_outside_exceeds_tol():
    # COM 1mm past one edge -> penalty > tol (outside quadratic + saturated barrier).
    import torch
    from cutamp.com_polygon_cost import COM_CENTER, com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = _support_centre_at_origin()
    com_world = torch.tensor([[0.101, 0.0, 0.0]])  # 1mm past X edge
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight, COM_CENTER)
    # outside=0.001 -> 0.001**2; inside barrier saturates at (0.001+0.02)**2.
    # Sum: 1e-6 + 4.41e-4 ≈ 4.42e-4 > tol 4e-4.
    assert pen.item() > 4e-4
    assert pen.item() == pytest.approx(0.001 ** 2 + (0.001 + margin) ** 2, rel=1e-3)


@needs_cuda
@pytest.mark.parametrize("robot", ["simplified", "real_legs"])
def test_mask_matches_penalty_gate(robot):
    # The post-IK mask must equal (penalty <= COM_TOL) elementwise, so the IK
    # COM-safe retry loop and the ConstraintChecker can never disagree.
    # COM_TOL is inside_weight * inside_margin^2 = 4e-4, calibrating the hard
    # constraint's tolerance against the mask's definition of 'inside'.
    import torch
    from cutamp.com_polygon_cost import (
        COM_TOL, compute_com_margin, compute_com_polygon_mask, compute_com_polygon_penalties,
    )
    world = make_blocks_t1_world(get_model(robot))
    home = world.q_init.detach().clone()
    bent = _moved(world, EXTREME_LEAN[world.model.name])  # genuinely out-of-hull
    # Spread a batch of configs around home by the model's EDGE_SPREAD, so its
    # rows straddle the polygon edge, then append the deterministic [home, bent]
    # pair so both classifications are guaranteed present in the batch.
    torch.manual_seed(0)
    names = list(world.kinematics.joint_names)
    spread = EDGE_SPREAD[world.model.name]
    q = home.unsqueeze(0).repeat(32, 1).clone()
    sigma = torch.tensor(list(spread.values()), device=q.device)
    q[:, [names.index(n) for n in spread]] += sigma * torch.randn(32, len(spread), device=q.device)
    # The premise: the spread rows hold zero and positive penalties, and rows
    # within 1 cm of the edge on either side of it, where the two FK paths
    # (world.kinematics for the mask, the "com" site for the penalty) must agree.
    margin, spread_pens = compute_com_margin(world, q), compute_com_polygon_penalties(world, {"q": q})["q"]
    assert bool((spread_pens > 0).any() & (spread_pens == 0).any()), spread_pens.tolist()
    assert bool(((margin > 0) & (margin < 0.01)).any() & ((margin < 0) & (margin > -0.01)).any()), margin.tolist()
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
