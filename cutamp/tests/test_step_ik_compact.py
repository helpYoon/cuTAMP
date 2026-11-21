"""Convergence of independent particles in a mixed bounded IK batch."""

import pytest
import torch

from cutamp.robots.t1 import REAL_LEGS
from cutamp.step_ik import _Kinematics


def test_solved_particles_leave_a_mixed_ik_batch(monkeypatch):
    kin = _Kinematics(torch.device("cpu"))
    seed = torch.tensor(REAL_LEGS.home, dtype=torch.float64).repeat(4, 1)
    lower = kin.lower.expand_as(seed).clone()
    upper = kin.upper.expand_as(seed).clone()
    # One wrist target is unreachable with its own frozen base coordinates.
    lower[2, :3] = upper[2, :3] = seed[2, :3]
    target = kin.fk(seed, "left", jacobian=False)[0]
    target[1:3, 24] += seed.new_tensor([0.04, 0.4])
    single = [
        kin.solve(target[i : i + 1], seed[i : i + 1], "left", bounds=(lower[i : i + 1], upper[i : i + 1]))
        for i in range(len(seed))
    ]
    sizes = []
    fitted = kin.fitted

    def record(q, *args, **kwargs):
        sizes.append(len(q))
        return fitted(q, *args, **kwargs)

    monkeypatch.setattr(kin, "fitted", record)
    batched = kin.solve(target, seed, "left", bounds=(lower, upper))
    assert min(sizes) < len(seed)
    for value, singles in zip(batched, zip(*single, strict=True), strict=True):
        torch.testing.assert_close(value, torch.cat(singles), atol=4e-6, rtol=1e-6)
    assert batched[2][2, :3].all()


def test_centered_ik_gradients_match_independent_particles():
    from cutamp.step_ik import solve_stance_ik
    from cutamp.tests.test_grasp_stance_ik import paired_inputs

    wrist, seed = paired_inputs("cpu", torch.float64)
    feet = seed.new_tensor([[[0.154, 0.094, 0.022], [0.042, -0.118, 0.029]]]).repeat(len(seed), 1, 1)

    def solve_and_differentiate(feet, wrist, seed):
        inputs = [value.clone().requires_grad_() for value in (feet, wrist, seed)]
        ik = solve_stance_ik(*inputs, "left", center_posture=True)
        assert ik.ok.all()
        loss = ik.com[:, :2].square().sum() + ik.margin.sum()
        return torch.autograd.grad(loss, inputs)

    batched = solve_and_differentiate(feet, wrist, seed)
    singles = [solve_and_differentiate(feet[i : i + 1], wrist[i : i + 1], seed[i : i + 1]) for i in range(len(seed))]
    for gradient, individual in zip(batched, zip(*singles, strict=True), strict=True):
        torch.testing.assert_close(gradient, torch.cat(individual), atol=1e-5, rtol=1e-5)


def test_warm_ik_changes_only_the_initial_guess():
    from cutamp.step_ik import solve_stance_ik
    from cutamp.tests.test_grasp_stance_ik import paired_inputs

    wrist, seed = paired_inputs("cpu", torch.float64)
    feet = seed.new_tensor([[[0.154, 0.094, 0.022], [0.042, -0.118, 0.029]]]).repeat(len(seed), 1, 1)
    previous = solve_stance_ik(feet, wrist, seed, "left", center_posture=True).q
    feet[:, 0, 0] += 0.0005
    feet.requires_grad_()
    seed.requires_grad_()

    def result(start):
        ik = solve_stance_ik(feet, wrist, seed, "left", center_posture=True, initial=start)
        assert ik.ok.all()
        return ik.q, *torch.autograd.grad(ik.margin.sum(), (feet, seed))

    for cold, warm in zip(result(None), result(previous), strict=True):
        torch.testing.assert_close(cold, warm, atol=2e-5, rtol=2e-5)


def test_numerical_ik_tolerance_is_scoped_to_the_solve():
    from cutamp.step_ik import solve_stance_ik, stance_ik_tolerance
    from cutamp.tests.test_grasp_stance_ik import paired_inputs

    wrist, seed = paired_inputs("cpu", torch.float64)
    feet = seed.new_tensor([[[0.154, 0.094, 0.022], [0.042, -0.118, 0.029]]]).repeat(len(seed), 1, 1)

    def solve():
        return solve_stance_ik(feet, wrist, seed, "left", center_posture=True)

    original = solve()
    with pytest.raises(RuntimeError, match="aborted objective"), stance_ik_tolerance(2e-7):
        relaxed = solve()
        assert relaxed.ok.all() and (relaxed.optimality < 2e-7).all()
        assert (relaxed.optimality > 2e-9).any()
        raise RuntimeError("aborted objective")
    torch.testing.assert_close(solve().q, original.q, rtol=0, atol=0)
    assert (original.optimality < 2e-9).all()
