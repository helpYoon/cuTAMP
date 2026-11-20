import os
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from cutamp.landing_predictor import LandingPredictor, prediction_precision
from cutamp.tests.conftest import needs_cuda


def assert_same_prediction(a, b):
    if is_dataclass(a):
        for f in fields(a):
            assert_same_prediction(getattr(a, f.name), getattr(b, f.name))
    elif isinstance(a, tuple):
        for x, y in zip(a, b, strict=True):
            assert_same_prediction(x, y)
    elif torch.is_tensor(a):
        assert torch.equal(a, b)
    else:
        assert a == b


@pytest.fixture(scope="module")
def predictor():
    default = Path(__file__).resolve().parents[2] / "data/t1-reachability/landing-accuracy/landing-seed3.pt"
    bundle = Path(os.environ.get("CUTAMP_LANDING_BUNDLE", default))
    return LandingPredictor(bundle)


@needs_cuda
@torch.no_grad()
def test_captured_rollout_matches_eager_and_retains_prior_results(predictor):
    default = Path(__file__).resolve().parents[2] / "data/t1-reachability/landing-accuracy/landing-seed3.pt"
    bundle = Path(os.environ.get("CUTAMP_LANDING_BUNDLE", default))
    fast = LandingPredictor(bundle, use_cuda_graph=True)
    ctx = torch.zeros(2, 38, device="cuda")
    ctx[:, -5:] = -1
    start = predictor.standing_start(ctx)
    commands = ctx.new_tensor([[[.12,.21,0], [.10,-.21,0]], [[.24,.21,0], [.06,-.21,0]]])
    expected = predictor.rollout(start, "LR", commands)
    result = fast.rollout(start, "LR", commands)
    assert expected.steps[0].clock.unique().numel() == 2
    torch.cuda.empty_cache()  # any scratch tensor referenced by the graph must still be owned
    changed = commands.clone()
    changed[..., 0] += .01
    ctx[:, 0] += .25
    other = predictor.standing_start(ctx)
    assert_same_prediction(fast.rollout(other, "LR", changed), predictor.rollout(other, "LR", changed))
    assert_same_prediction(result, expected)


@needs_cuda
def test_reference_single_step_and_nonsettled_precision_pass(predictor):
    context = torch.zeros(1, 38, device="cuda")
    context[:, 0] = (1.0 - 0.75) / 0.45
    context[:, -5:] = -1.0  # two actuator-lag substeps
    start = predictor.standing_start(context)
    commands = torch.tensor([[[0.12, 0.21, 0]]], device="cuda", requires_grad=True)
    out = predictor.rollout(start, "L", commands)
    step = out.steps[0]
    assert out.tracking_ok.tolist() == [True]
    assert step.near_rest.tolist() == [False]
    assert step.clock.tolist() == [46]
    # Independent eager bfm_wm/Python 3.13 evaluation of this fixture in paired-model.json.
    expected = commands.new_tensor([[[0.1539695, 0.0938335, 0.0220075], [0.0423440, -0.1177292, 0.0288627]]])
    torch.testing.assert_close(step.achieved, expected, rtol=0, atol=2e-5)
    assert abs(step.max_joint_rms.item() - 0.0127628) < 5e-5
    (grad,) = torch.autograd.grad(step.achieved[0, 0, 0], commands)
    torch.testing.assert_close(grad[0, 0], commands.new_tensor([0.899625, -0.016141, 0.011681]), rtol=0.006, atol=3e-4)
    assert all(not p.requires_grad for p in predictor.model.parameters())


@needs_cuda
@pytest.mark.parametrize("order", ["LR", "RL"])
def test_chain_uses_terminal_histories_and_correct_row_clocks(order, predictor):
    context = torch.zeros(2, 38, device="cuda")
    context[:, 0] = (1.0 - 0.75) / 0.45
    context[:, -5:] = -1.0
    start = predictor.standing_start(context)
    sign = 1 if order == "LR" else -1
    cmd = torch.tensor(
        [[[0.12, 0.21 * sign, 0], [0.10, -0.21 * sign, 0]], [[0.24, 0.21 * sign, 0], [0.06, -0.21 * sign, 0]]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    planner_precision = torch.backends.cuda.matmul.allow_tf32
    pred = predictor.rollout(start, order, cmd)
    assert torch.backends.cuda.matmul.allow_tf32 == planner_precision
    a, b = pred.steps
    assert a.clock[0] != a.clock[1]
    torch.testing.assert_close(a.valid_ticks.sum(1), a.clock)
    torch.testing.assert_close(b.valid_ticks.sum(1), b.clock + predictor.hold_ticks)
    rows = torch.arange(len(a.clock), device=a.clock.device)
    torch.testing.assert_close(a.achieved, a.sole_trace[rows, a.clock - 1])
    assert (a.terminal.tensors[-1][:, 4] == 1).all()  # The very next issue reads the deadline's foot anchors.
    assert a.max_joint_rms is None and a.max_root_speed is None and a.near_rest is None
    assert b.near_rest is not None
    support = "LR".index(order[0])
    torch.testing.assert_close(b.reference[:, support], a.achieved[:, support])
    with prediction_precision():
        (terminal_grad,) = torch.autograd.grad(b.achieved[:, 1 - support, 0].sum(), cmd)
    assert torch.isfinite(terminal_grad).all() and (terminal_grad[:, 0, :].abs().sum(1) > 0.01).all()
    with torch.no_grad():
        first = predictor.rollout(start, order[0], cmd[:, :1], final_hold_ticks=0)
        for actual, replayed in zip(a.terminal.tensors, first.terminal.tensors, strict=True):
            torch.testing.assert_close(actual, replayed)
        single = predictor.rollout(a.terminal, order[1], cmd[:, 1:])
        # Perturb only the first command. Recompute complete closed-loop predictions for both sides.
        delta = torch.zeros_like(cmd)
        delta[:, 0, 0] = 0.001
        minus = predictor.rollout(start, order, cmd - delta)
        plus = predictor.rollout(start, order, cmd + delta)
        for step, lower, upper in zip(pred.steps, minus.steps, plus.steps):
            assert torch.equal(step.clock, lower.clock) and torch.equal(step.clock, upper.clock)
        fd = (plus.steps[-1].achieved[:, 1 - support, 0] - minus.steps[-1].achieved[:, 1 - support, 0]) / 0.002
        torch.testing.assert_close(terminal_grad[:, 0, 0], fd, atol=0.02, rtol=0.03)
    torch.testing.assert_close(single.steps[0].achieved, b.achieved)
    empty = predictor.rollout(start, "", cmd[:, :0])
    assert empty.terminal is start and empty.tracking_ok.all()


@needs_cuda
def test_fresh_prediction_for_the_ranked_floor_pick(predictor):
    from cutamp.step_search import StepSearch, TargetRegion

    data = Path(__file__).resolve().parents[2] / "data/t1-reachability"
    search = StepSearch.load(
        data / "t1-nominal-hand-map.npz", accuracy_path=data / "landing-accuracy/home-model.json", lag=8
    )
    target = TargetRegion("left", (0.48, 0.18, 0.025), (0.05,) * 3, solid_pick=True)
    edge = next(search.schedules(search.home, [target])).edges[0]
    assert edge.order == "L" and edge.commands == ((0.24, 0.21, 0.0),)
    dynamics = torch.zeros(1, 38, device=predictor.device)
    dynamics[:, 0] = (1.0 - 0.75) / 0.45
    dynamics[:, -5:] = 1.0  # 8 physics ticks of actuator lag
    with torch.no_grad():
        start = predictor.standing_start(dynamics)
        commands = torch.tensor([edge.commands], device=predictor.device)
        prediction = predictor.rollout(start, edge.order, commands)
        assert prediction.tracking_ok.item()
        assert predictor.rollout(start, "none", commands[:, :0]).terminal is start


@needs_cuda
def test_importing_external_history_into_fresh_module_preserves_command_jacobian(predictor):
    """A public planner run may construct its own predictor after an observation packet was prepared."""
    default = Path(__file__).resolve().parents[2] / "data/t1-reachability/landing-accuracy/landing-seed3.pt"
    independent = LandingPredictor(Path(os.environ.get("CUTAMP_LANDING_BUNDLE", default)))
    context = torch.zeros(1, 38, device=predictor.device)
    context[:, 0] = 0.25 / 0.45
    context[:, -5:] = -1.0
    observed = predictor.standing_start(context)
    u = context.new_tensor([[[0.12, 0.21, 0.0]]]).requires_grad_()
    with prediction_precision():
        out = independent.rollout(observed, "L", u)
        (derivative,) = torch.autograd.grad(out.steps[0].achieved[0, 0, 0], u)
    delta = torch.zeros_like(u)
    delta[:, :, 0] = 0.001
    with torch.no_grad():
        left = independent.rollout(observed, "L", u - delta)
        right = independent.rollout(observed, "L", u + delta)
    fd = (right.steps[0].achieved[0, 0, 0] - left.steps[0].achieved[0, 0, 0]) / 0.002
    torch.testing.assert_close(derivative[0, 0, 0], fd, atol=0.004, rtol=0.008)
    torch.testing.assert_close(derivative[0, 0], u.new_tensor([0.899625, -0.016141, 0.011681]), rtol=0.006, atol=0.0003)


@needs_cuda
@pytest.mark.parametrize("side", ["L", "R"])
def test_command_grid_directional_derivative_on_smooth_clocks(side, predictor):
    c = torch.tensor(
        [[-0.12, 0.20, 0.03], [0.08, 0.23, -0.04], [0.15, 0.24, 0.02], [0.22, 0.26, 0.06]], device=predictor.device
    )
    sign = 1 if side == "L" else -1
    c = (c * c.new_tensor([1.0, sign, sign]))[:, None].requires_grad_()
    d = c.new_zeros(len(c), 38)
    d[:, 0] = 0.25 / 0.45
    d[:, -5:] = -1
    state = predictor.standing_start(d)
    v = c.new_tensor([0.8, -0.35 * sign, 0.2 * sign])

    def score(u):
        step = predictor.rollout(state, side, u).steps[0]
        return step.achieved[:, "LR".index(side), 0], step.clock

    with prediction_precision():
        value, clock = score(c)
        (jac,) = torch.autograd.grad(value.sum(), c)
        with torch.no_grad():
            lo, lc = score(c - 0.0007 * v)
            hi, hc = score(c + 0.0007 * v)
    smooth = (lc == clock) & (hc == clock)
    assert smooth.sum() >= 2
    torch.testing.assert_close((jac[:, 0] * v).sum(1)[smooth], ((hi - lo) / 0.0014)[smooth], rtol=0.035, atol=0.015)


@needs_cuda
@pytest.mark.parametrize("order", ["L", "RL", "LR"])
def test_cached_command_jacobian_matches_the_deployment_and_reverse_mode(order, predictor):
    batch = len(order)
    d = torch.zeros(batch, 38, device=predictor.device)
    d[:, 0] = 0.25 / 0.45
    d[:, -5:] = -1.0
    if batch > 1:
        d[1, -5:] = 1.0  # an independent actuator-lag condition
    initial = predictor.standing_start(d)
    c = d.new_tensor([[0.12, 0.21 if side == "L" else -0.21, 0.0] for side in order])[None].repeat(batch, 1, 1)
    c[-1, 0, 0] += 0.03
    c.requires_grad_()

    def diagnostic(p):
        s = p.steps[-1]
        # Terminal FK, trajectory, phase gates, pending commands and the second issue frame.
        return torch.stack(
            [
                s.achieved[:, 0, 0],
                s.terminal.tensors[-2][:, -1, :].mean(1),
                s.sole_trace[:, :, 1, 1].mean(1),
                s.actions[:, :, 2].mean(1),
                s.trajectory[4][:, :, 3].mean(1),
                s.reference[:, 0, 0] + s.phases.support_xy + s.phases.deadline_xy,
                p.final_xy_error.sum(1) + p.final_yaw_error.sum(1),
            ],
            -1,
        )

    with prediction_precision():
        a = predictor.rollout(initial, order, c)
        b = predictor.linearize_autodiff(initial, order, c)
        assert_same_prediction(a, b)
        assert all(type(s.actions.grad_fn).__name__ == "_CommandJacobianBackward" for s in b.steps)
        ea, eb = diagnostic(a), diagnostic(b)
        for k in range(ea.shape[1]):
            (ga,) = torch.autograd.grad(ea[:, k].sum(), c, retain_graph=True)
            (gb,) = torch.autograd.grad(eb[:, k].sum(), c, retain_graph=True)
            torch.testing.assert_close(gb, ga, rtol=7e-4, atol=1.5e-4)


@pytest.mark.parametrize("order", ["L", "R", "LR", "RL"])
@needs_cuda
def test_finite_difference_linearization_follows_each_particles_rollout(order, predictor, monkeypatch):
    # Distinct commands and actuator lags expose cross-particle mixing. The first
    # command of the alternating pair must also affect the other foot's landing.
    context = torch.zeros(2, 38, device=predictor.device)
    context[:, 0] = 0.25 / 0.45
    context[:, -5:] = context.new_tensor([[-1.0], [1.0]])
    initial = predictor.standing_start(context)
    c = context.new_tensor([[0.12, 0.21 if foot == "L" else -0.21, 0.0] for foot in order])[None].repeat(2, 1, 1)
    c[1, 0, 0] = 0.15
    c.requires_grad_()
    native = predictor.rollout(initial, order, c)
    calls = []
    rollout = predictor.rollout

    def observe(state, sequence, targets):
        result = rollout(state, sequence, targets)
        calls.append(len(targets))
        return result

    monkeypatch.setattr(predictor, "rollout", observe)
    approximate = predictor.linearize(initial, order, c)
    assert calls == [2, 2 * 2 * c.shape[1] * 3], "This fixture must exercise batched FD on stable clocks"
    assert_same_prediction(approximate, native)
    with prediction_precision():
        for get in (
            lambda p: p.steps[-1].achieved[:, "LR".index(order[-1]), :2].sum(-1),
            lambda p: p.steps[-1].achieved[:, "LR".index(order[-1]), 2],
            lambda p: p.steps[-1].actions[:, :12, 2].mean(-1),
            lambda p: p.steps[-1].terminal.tensors[7][:, -1, 5],
        ):
            (exact,) = torch.autograd.grad(get(native).sum(), c, retain_graph=True)
            (fd,) = torch.autograd.grad(get(approximate).sum(), c, retain_graph=True)
            torch.testing.assert_close(fd, exact, rtol=0.04, atol=0.015)
