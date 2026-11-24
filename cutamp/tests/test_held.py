import pytest
import torch

from cutamp import held
from cutamp.stance import stance_joint_limits
from cutamp.step_loop import _placements_and_grasps
from cutamp.tests.conftest import needs_cuda
from cutamp.tests.test_step_loop import make_plan, stepped_call  # noqa: F401


def pick(plan):
    """The pick's posture name, arm, stance [16, 2, 3] and hand target [16, 4, 4] at the plan's start."""
    p, op = plan.particles, plan.groups[0][1]
    name = op.arg("q")
    stance = plan.stance_table(p)[name].detach().clone()
    target = _placements_and_grasps(plan.groups, plan.world, plan.pools, 0, 16, bindings=p, grasp_indices=p,
                                    project=False)[2][0]
    return name, op.operator.metadata.arm, stance, target


@needs_cuda
def test_put_back_returns_the_hand_and_the_soles(stepped_call):
    plan = make_plan(stepped_call)
    name, arm, stance, target = pick(plan)
    q0 = plan.particles[name].detach().clone()
    near = held.held_rows(plan.world, q0, stance, target, arm)[0].abs().amax(1) < 5e-3    # in reach at the start
    assert near.sum() >= 8
    stance[:, 0, 0] += 0.01                                                               # the left sole 1 cm on
    q, jac, capped = held.put_back(plan.world, q0, stance, target, arm)
    assert jac.shape == (16, 18, 33) and jac.dtype == torch.float64 and q.dtype == q0.dtype
    assert capped.shape == (16,) and capped.dtype == torch.bool and not capped[near].any()
    assert held.held_rows(plan.world, q, stance, target, arm)[0][near].abs().amax() < 1e-4
    lower, upper = stance_joint_limits(stance, *plan.world.joint_limits)
    assert (q >= lower).all() and (q <= upper).all()
    again, _, _ = held.put_back(plan.world, q, stance, target, arm)
    torch.testing.assert_close(again[near], q[near], atol=1e-3, rtol=0)
    q_feet, jac_feet, _ = held.put_back(plan.world, q0, stance)                           # a retract: soles only
    assert jac_feet.shape == (16, 12, 33)
    assert held.held_rows(plan.world, q_feet, stance)[0][near].abs().amax() < 1e-4
    with pytest.raises(ValueError):
        held.held_rows(plan.world, q0, stance, target)


@needs_cuda
def test_the_correction_stays_inside_the_joint_limits(stepped_call):
    plan = make_plan(stepped_call)
    name, arm, stance, target = pick(plan)
    q0 = plan.particles[name].detach().clone()
    knee = list(plan.world.model.joint_names).index("Left_Knee_Pitch")
    stance[:, 0, 0] += 0.30                                   # a stance the posture cannot reach in one correction
    lower, upper = stance_joint_limits(stance, *plan.world.joint_limits)
    q0[:, knee] = upper[:, knee]
    q, jac, capped = held.put_back(plan.world, q0, stance, target, arm)
    assert q.isfinite().all() and (q >= lower).all() and (q <= upper).all() and capped.any()
    assert torch.equal(q[:, knee], upper[:, knee]) and (jac[:, :, knee] == 0).all()


@needs_cuda
def test_a_row_that_is_not_a_number_is_left_alone(stepped_call):
    plan = make_plan(stepped_call)
    name, arm, stance, target = pick(plan)
    q0 = plan.particles[name].detach().clone()
    clean, _, _ = held.put_back(plan.world, q0, stance, target, arm)
    stance[3] = float("nan")
    q, jac, capped = held.put_back(plan.world, q0, stance, target, arm)
    assert torch.equal(q[3], q0[3]) and (jac[3] == 0).all() and not capped[3]
    keep = torch.arange(16, device=q.device) != 3
    torch.testing.assert_close(q[keep], clean[keep])


@needs_cuda
def test_the_kept_gradient_moves_neither_hand_nor_feet(stepped_call):
    plan = make_plan(stepped_call)
    name, arm, stance, target = pick(plan)
    _, jac, _ = held.put_back(plan.world, plan.particles[name].detach().clone(), stance, target, arm)
    g = torch.randn(16, 33, generator=torch.Generator().manual_seed(0)).to(jac.device)
    kept, mu = held.split_gradient(g, jac)
    assert kept.dtype == g.dtype and mu.shape == (16, 18)
    assert (jac @ kept.double()[..., None]).abs().amax() < 1e-5
    g[2, 0] = float("nan")
    kept, mu = held.split_gradient(g, jac)
    assert kept[2].isnan().any() and (mu[2] == 0).all() and kept[3].isfinite().all()


def slopes(plan, w, move):
    """(measured, mu, q0) for the pick: the change of w . q per unit of `move(stance, target, d)`, by put_back."""
    name, arm, stance, target = pick(plan)
    q0, jac, _ = held.put_back(plan.world, plan.particles[name].detach().clone(), stance, target, arm)
    _, mu = held.split_gradient(w.expand(16, -1).contiguous(), jac)
    d = 1e-3
    moved_stance, moved_target = move(stance.clone(), target.clone(), d)
    q1, _, _ = held.put_back(plan.world, q0, moved_stance, moved_target, arm)
    return ((q1 - q0) @ w) / d, mu, (stance, target)


@needs_cuda
def test_the_soles_share_is_the_slope_along_the_held_surface(stepped_call):
    plan = make_plan(stepped_call)
    w = torch.randn(33, generator=torch.Generator().manual_seed(1)).cuda()
    for foot, k in ((0, 0), (1, 1), (0, 2)):
        def move(stance, target, d, foot=foot, k=k):
            stance[:, foot, k] += d
            return stance, target
        measured, mu, (stance, _) = slopes(plan, w, move)
        s = stance.clone().requires_grad_(True)
        share = torch.autograd.grad(held.stance_share(mu, s), s)[0][:, foot, k]
        error = (measured - share.to(measured)).abs() / (share.abs().to(measured) + 1e-3 * w.norm())
        assert error.median() < 0.05


@needs_cuda
def test_the_hands_share_is_the_slope_along_the_held_surface(stepped_call):
    plan = make_plan(stepped_call)
    w = torch.randn(33, generator=torch.Generator().manual_seed(2)).cuda()
    def move(stance, target, d):
        target[:, 0, 3] += d
        return stance, target
    measured, mu, (_, target) = slopes(plan, w, move)
    t = target.clone().requires_grad_(True)
    share = torch.autograd.grad(held.target_share(mu, t), t)[0][:, 0, 3]
    error = (measured - share.to(measured)).abs() / (share.abs().to(measured) + 1e-3 * w.norm())
    assert error.median() < 0.05


def test_a_share_ignores_a_row_that_is_not_a_number():
    mu = torch.ones(2, 12)
    stance = torch.zeros(2, 2, 3)
    stance[1] = float("nan")
    s = stance.clone().requires_grad_(True)
    grad = torch.autograd.grad(held.stance_share(mu, s), s)[0]
    assert (grad[0] != 0).any() and (grad[1] == 0).all()
    target = torch.eye(4).repeat(2, 1, 1)
    target[1] = float("nan")
    t = target.clone().requires_grad_(True)
    share = held.target_share(torch.ones(2, 18), t)
    grad = torch.autograd.grad(share, t)[0]
    assert share.isfinite() and (grad[0] != 0).any() and (grad[1] == 0).all()
    with pytest.raises(ValueError):
        held.target_share(torch.ones(2, 12), target)                    # a retract's soles are not hand rows
