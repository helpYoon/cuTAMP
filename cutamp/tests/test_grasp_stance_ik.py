"""T1 bounded local IK and its projected grasp-to-stance derivative."""

import json
from pathlib import Path

import pytest
import torch


def paired_inputs(device, dtype):
    path = Path(__file__).parent / "fixtures/t1_paired_grasps.json"
    f = json.loads(path.read_text())
    obj = torch.tensor(f["world_object"], device=device, dtype=dtype)
    wrist = obj @ torch.tensor([f["object_to_wrist"][k] for k in ("near", "far")], device=device, dtype=dtype)
    seed = torch.tensor(f["seed"], device=device, dtype=dtype)[None].repeat(2, 1)
    return wrist, seed


@pytest.fixture
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_solved_posture_and_total_derivative_follow_achieved_stance(one_cpu_thread):
    from cutamp.step_ik import solve_stance_ik

    wrist, seed = paired_inputs("cpu", torch.float64)
    wrist.requires_grad_()
    # Independent eager bfm_wm fixture and SciPy constrained solve at its measured terminal stance.
    soles = (
        torch.tensor(
            [[[0.153969496, 0.093833506, 0.022007490], [0.042344008, -0.117729239, 0.028862681]]], dtype=torch.float64
        )
        .repeat(2, 1, 1)
        .requires_grad_()
    )
    ik = solve_stance_ik(soles, wrist, seed, "left")
    assert ik.ok.tolist() == [True, True]
    assert ik.margin.tolist() == pytest.approx([0.058963015, 0.045475723], abs=0.0003)
    total, wrist_deriv = torch.autograd.grad(ik.margin.sum(), (soles, wrist), retain_graph=True)
    (hull,) = torch.autograd.grad(ik.hull_only_margin.sum(), soles)
    assert torch.all((total - hull).norm(dim=(1, 2)) > 0.05)
    delta = torch.zeros_like(soles)
    delta[:, 0, 0] = 0.00015
    dw = torch.zeros_like(wrist)
    dw[:, 0, 3] = 0.00015
    with torch.no_grad():
        up = solve_stance_ik(soles + delta, wrist, seed, "left").margin
        down = solve_stance_ik(soles - delta, wrist, seed, "left").margin
        farther = solve_stance_ik(soles, wrist + dw, seed, "left").margin
        closer = solve_stance_ik(soles, wrist - dw, seed, "left").margin
    torch.testing.assert_close(total[:, 0, 0], (up - down) / 0.0003, atol=0.012, rtol=0.02)
    torch.testing.assert_close(wrist_deriv[:, 0, 3], (farther - closer) / 0.0003, atol=0.002, rtol=0.02)


def test_sole_and_wrist_fk_match_every_tool_in_home_stance(one_cpu_thread):
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.step_ik import _kinematics, solve_stance_ik

    d = torch.device("cpu")
    k = _kinematics(d)
    q = torch.tensor(REAL_LEGS.home, dtype=torch.float64)[None]
    for arm in ("left", "right"):
        pose, _, com = k.fk(q, arm)
        frames = pose.reshape(1, 3, 12)
        soles = torch.cat((frames[:, :2, :2], torch.zeros(1, 2, 1, dtype=q.dtype)), -1)
        wrist = torch.eye(4, dtype=q.dtype)[None].clone()
        wrist[:, :3, 3] = frames[:, 2, :3]
        wrist[:, :3, :3] = frames[:, 2, 3:].reshape(-1, 3, 3).mT / 0.2
        r = solve_stance_ik(soles, wrist, q, arm)
        assert r.ok.all()
        torch.testing.assert_close(r.com, com, atol=1e-5, rtol=0)
        # Moving a target beyond leg/arm reach must give an uncertified keyframe.
        wrist[:, 0, 3] += 4.0
        assert not solve_stance_ik(soles, wrist, q, arm).ok.any()


def test_nonfinite_ik_target_rejects_only_its_particle(one_cpu_thread):
    from cutamp.step_ik import solve_stance_ik

    wrist, seed = paired_inputs("cpu", torch.float64)
    soles = seed.new_tensor([[[0.154, 0.094, 0.022], [0.042, -0.118, 0.029]]]).repeat(2, 1, 1).requires_grad_()
    wrist[1, 0, 3] = torch.nan
    ik = solve_stance_ik(soles, wrist, seed, "left")
    assert ik.ok.tolist() == [True, False]
    assert not torch.isfinite(ik.optimality[1])
    (grad,) = torch.autograd.grad(ik.margin[0], soles)
    assert torch.isfinite(grad).all() and grad[0].norm() > 0.1 and grad[1].norm() == 0


def test_posture_reference_gradient_on_free_and_joint_limited_branches(one_cpu_thread):
    from cutamp.step_ik import _kinematics, solve_stance_ik

    wrists, seeds = paired_inputs("cpu", torch.float64)
    wrists = wrists[:1].expand_as(wrists)
    # A lateral reference moves away from the ankle stop. A lower reference also
    # loads both knee limits. Both branches reach the exact same wrist and feet.
    seeds[0, 1] += 0.20
    seeds[1, 2] -= 0.20
    seeds.requires_grad_()
    soles = seeds.new_tensor([[[0.154, 0.094, 0.022], [0.042, -0.118, 0.029]]]).repeat(2, 1, 1)
    kin = _kinematics(soles.device)
    ik = solve_stance_ik(soles, wrists, seeds, "left")
    assert ik.ok.all()

    def active(q):
        return ((kin.upper - q) < 1e-7) | ((q - kin.lower) < 1e-7)

    counts = active(ik.q).sum(1)
    assert counts[0] == 0 and counts[1] >= 2
    (g,) = torch.autograd.grad(ik.margin.sum(), seeds)
    delta = torch.zeros_like(seeds)
    delta[:, 1] = 0.0001
    with torch.no_grad():
        low = solve_stance_ik(soles, wrists, seeds - delta, "left")
        high = solve_stance_ik(soles, wrists, seeds + delta, "left")
    assert torch.equal(active(ik.q), active(low.q)) and torch.equal(active(ik.q), active(high.q))
    torch.testing.assert_close(g[:, 1], (high.margin - low.margin) / 0.0002, rtol=0.02, atol=0.001)


@pytest.mark.parametrize("expired_on_entry", [False, True])
def test_expired_budget_interrupts_the_next_gap_evaluation(monkeypatch, one_cpu_thread, expired_on_entry):
    from cutamp import step_refinement

    now = [3.0 if expired_on_entry else 1.0]
    monkeypatch.setattr(step_refinement, "monotonic", lambda: now[0])
    wrist, seed = paired_inputs("cpu", torch.float32)
    soles = seed.new_tensor([[[0.154, 0.094, 0.022], [0.042, -0.118, 0.029]]]).repeat(2, 1, 1)
    calls = []

    def observed(prediction, ik):
        calls.append(1)
        now[0] = 3.0
        return {}

    with pytest.raises(TimeoutError, match="refinement"):
        step_refinement.refine_gap(
            None, soles, "", seed.new_empty(2, 0, 3), "left", wrist, seed, deadline=2.0, extra_rows=observed
        )
    assert len(calls) == (0 if expired_on_entry else 1)


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("center_posture", [False, True])
def test_other_arm_stays_home_or_moves_only_to_clear_collision(arm, center_posture, one_cpu_thread):
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.step_ik import _kinematics, solve_stance_ik

    home = torch.tensor(REAL_LEGS.home, dtype=torch.double)
    other = REAL_LEGS.right_arm if arm == "left" else REAL_LEGS.left_arm
    q = home.expand(3, -1).clone()
    kin = _kinematics(q.device)
    pose, com_home = kin.fk(q, arm, jacobian=False)
    tools = pose.reshape(3, 3, 12)
    soles = torch.cat([tools[:, :2, :2], q.new_zeros(3, 2, 1)], dim=-1)
    wrist = torch.eye(4, dtype=q.dtype).repeat(3, 1, 1)
    wrist[:, :3, :3] = tools[:, 2, 3:].reshape(3, 3, 3).mT / 0.2
    wrist[:, :3, 3] = tools[:, 2, :3]
    # An off-home witness can clear the obstacle in row 1. Nothing clears row 2.
    q[:, other.start] += 0.8
    q.requires_grad_()
    levels = q.new_tensor([-1.0, 0.25, 1.2])
    def blocked(posture):
        return posture[:, other.start] - home[other.start] < levels

    fitted = solve_stance_ik(soles, wrist, q, arm, collides=blocked, center_posture=center_posture)
    assert fitted.ok.tolist() == [True, True, False]
    assert torch.isfinite(fitted.optimality).all()
    torch.testing.assert_close(fitted.q[0, other], home[other], atol=1e-8, rtol=0)
    assert 0.25 <= float(fitted.q[1, other.start].detach() - home[other.start]) <= 0.8
    _, full_com = kin.fk(fitted.q, arm, jacobian=False)
    torch.testing.assert_close(fitted.com, full_com)
    assert float((fitted.com[1].detach() - com_home[1]).norm()) > 1e-4
    seed_grad, = torch.autograd.grad(fitted.com[0].sum(), q)
    assert torch.all(seed_grad[0, other] == 0)

    carrying = q.detach()[:, other].clamp(kin.lower[other], kin.upper[other])
    occupied = solve_stance_ik(soles, wrist, q.detach(), arm, other_arm_reference=carrying,
                               center_posture=center_posture)
    assert occupied.ok.all()
    torch.testing.assert_close(occupied.q[:, other], carrying)

def test_batched_com_jacobian_matches_perturbing_all_joints(one_cpu_thread):
    from cutamp.step_ik import _kinematics

    _, seed = paired_inputs("cpu", torch.float64)
    kin = _kinematics(seed.device)
    _, _, _, jac = kin.fk(seed, "left", com_jacobian=True)
    shifted = seed[:, None, :] + 1j * 1e-25 * torch.eye(seed.shape[-1], dtype=torch.double)
    _, com = kin.fk(shifted.reshape(-1, seed.shape[-1]), "left", False)
    torch.testing.assert_close(jac, (com.imag / 1e-25).reshape(len(seed), seed.shape[-1], 3).mT)


def test_centered_stance_ik_differentiates_through_wrist_soles_and_held_arm(one_cpu_thread):
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.step_ik import solve_stance_ik
    from cutamp.step_refinement import stance_center_residual

    wrist, seed = paired_inputs("cpu", torch.float64)
    feet = seed.new_tensor([[[0.154, 0.094, 0.022], [0.042, -0.118, 0.029]]]).repeat(2, 1, 1)
    idle = seed.new_tensor(REAL_LEGS.home[REAL_LEGS.right_arm]).repeat(2, 1)
    reference = solve_stance_ik(feet, wrist, seed, "left", other_arm_reference=idle)
    feet.requires_grad_()
    wrist.requires_grad_()
    idle.requires_grad_()

    def solve(s, w, a):
        return solve_stance_ik(s, w, seed, "left", other_arm_reference=a, center_posture=True)

    fitted = solve(feet, wrist, idle)
    assert fitted.ok.all()
    assert (stance_center_residual(fitted).norm(dim=1) < stance_center_residual(reference).norm(dim=1) - 0.005).all()
    torch.testing.assert_close(fitted.q[:, REAL_LEGS.right_arm], idle, rtol=0, atol=1e-9)
    cost = stance_center_residual(fitted).square().sum()
    gradients = torch.autograd.grad(cost, (feet, wrist, idle))
    for k, (x, index) in enumerate(zip((feet, wrist, idle), [(0, 0), (0, 3), (0,)])):
        delta = torch.zeros_like(x)
        delta[(slice(None), *index)] = 0.0001
        inputs = [feet, wrist, idle]
        with torch.no_grad():
            inputs[k] = x + delta
            plus = stance_center_residual(solve(*inputs)).square().sum(1)
            inputs[k] = x - delta
            minus = stance_center_residual(solve(*inputs)).square().sum(1)
        torch.testing.assert_close(gradients[k][(slice(None), *index)], (plus - minus) / 0.0002,
                                   rtol=0.025, atol=2e-5)


def test_centered_ik_sphere_residual_and_jacobian_match_direct_geometry(one_cpu_thread):
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.step_ik import _kinematics

    _, q = paired_inputs("cpu", torch.float64)
    q[0, REAL_LEGS.left_arm.start + 1] -= 0.4
    q[1] = q.new_tensor(REAL_LEGS.home)
    kin = _kinematics(q.device)
    residual, jac = kin.self_spheres.residual(kin.link_frames(q, True))
    spheres = kin.self_spheres
    pos, rot = kin.link_frames(q[:1])[:2]
    centers = pos[0, spheres.links] + (rot[0, spheres.links] @ spheres.offsets[..., None]).squeeze(-1)
    gap = (centers[spheres.i] - centers[spheres.j]).norm(dim=-1) - spheres.radii_sum
    assert gap.min() < 0 and (residual[1] == 0).all()
    torch.testing.assert_close(residual[0].norm(), (0.006 - gap).relu().norm())
    diff = 1j * 1e-25 * torch.eye(kin.n, dtype=q.dtype)
    z = (q[:, None] + diff).flatten(0, 1)
    value, _ = kin.self_spheres.residual(kin.link_frames(z), repeats=kin.n)
    torch.testing.assert_close((residual[:, :, None] * jac).sum(1),
                               (0.5 * value.square().sum(1).imag / 1e-25).reshape_as(q))


def test_particle_ik_follows_translated_feet_beyond_home_chart(one_cpu_thread):
    from cutamp.step_ik import solve_stance_ik

    wrist, seed = paired_inputs("cpu", torch.float64)
    soles = seed.new_tensor([[[0.154, 0.094, 0.022], [0.042, -0.118, 0.029]]]).repeat(2, 1, 1)
    regular = solve_stance_ik(soles, wrist, seed, "left")
    shift = seed.new_tensor([[1.4, -0.3], [-1.3, 0.5]])
    soles[..., :2] += shift[:, None]
    wrist[:, :2, 3] += shift
    seed[:, :2] += shift
    moved = solve_stance_ik(soles, wrist, seed, "left")
    assert moved.ok.all()
    torch.testing.assert_close(moved.q[:, :2] - regular.q[:, :2], shift, atol=1e-6, rtol=0)
    torch.testing.assert_close(moved.q[:, 2:], regular.q[:, 2:], atol=1e-6, rtol=0)
    torch.testing.assert_close(moved.margin, regular.margin)
