import os
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from cutamp.landing_accuracy import LANDING_ACCURACY, LANDING_BUNDLE
from cutamp.landing_predictor import LandingPredictor, prediction_precision
from cutamp.reachability import REACH_MAP
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
    bundle = Path(os.environ.get("CUTAMP_LANDING_BUNDLE", LANDING_BUNDLE))
    return LandingPredictor(bundle)


@needs_cuda
@torch.no_grad()
@torch.no_grad()  # values only, as the planner runs the model
def test_captured_rollout_matches_eager_and_retains_prior_results(predictor):
    bundle = Path(os.environ.get("CUTAMP_LANDING_BUNDLE", LANDING_BUNDLE))
    fast = LandingPredictor(bundle, use_cuda_graph=True)
    ctx = torch.zeros(2, 38, device="cuda")
    ctx[:, -5:] = -1
    commands = ctx.new_tensor([[[.12,.21,0], [.10,-.21,0]], [[.24,.21,0], [.06,-.21,0]]])
    # TorchScript's fuser compiles during a model's first calls, which round differently from every later one
    for warm in (predictor, fast):
        warm.rollout(warm.standing_start(ctx), "LR", commands)
    start = predictor.standing_start(ctx)
    assert_same_prediction(fast.standing_start(ctx), start)
    expected = predictor.rollout(start, "LR", commands)
    result = fast.rollout(start, "LR", commands)
    assert expected.steps[0].clock.unique().numel() == 2
    torch.cuda.empty_cache()  # any scratch tensor referenced by the graph must still be owned
    changed = commands.clone()
    changed[..., 0] += .01
    ctx[:, 0] += .25
    other = predictor.standing_start(ctx)
    assert_same_prediction(fast.standing_start(ctx), other)
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
    from cutamp.task_planning.landing_options import LandingOptions, TargetRegion

    search = LandingOptions.load(
        REACH_MAP, accuracy_path=LANDING_ACCURACY, lag=8
    )
    target = TargetRegion("left", (0.48, 0.18, 0.025), (0.05,) * 3, solid_pick=True)
    edge = min(search.edges(search.home, target), key=lambda e: e.cost)
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
    independent = LandingPredictor(Path(os.environ.get("CUTAMP_LANDING_BUNDLE", LANDING_BUNDLE)))
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
@pytest.mark.parametrize("order", ["LR", "RL"])
def test_first_command_moves_second_issue_and_final_grasp_margin(order):
    from cutamp.step_ik import solve_stance_ik
    from cutamp.tests.test_grasp_stance_ik import paired_inputs

    bundle = Path(os.environ.get("CUTAMP_LANDING_BUNDLE", LANDING_BUNDLE))
    if not bundle.exists() and "CUTAMP_LANDING_BUNDLE" not in os.environ:
        pytest.skip("Requires the seed-3 deployment")
    p = LandingPredictor(bundle)
    wrists, seeds = paired_inputs(p.device, torch.float32)
    d = seeds.new_zeros(2, 38)
    d[:, 0] = 0.25 / 0.45
    d[:, -5:] = -1.0
    state = p.standing_start(d)
    c = d.new_tensor([[0.12, 0.21 if s == "L" else -0.21, 0.0] for s in order])[None].repeat(2, 1, 1).requires_grad_()

    def score(u):
        out = p.rollout(state, order, u)
        result = solve_stance_ik(out.steps[-1].achieved, wrists, seeds, "left")
        second_target_x = out.steps[-1].reference[:, "LR".index(order[-1]), 0]
        return result.margin, second_target_x, tuple(s.clock for s in out.steps)

    with prediction_precision():
        margin, second_x, clocks = score(c)
        (dm,) = torch.autograd.grad(margin.sum(), c, retain_graph=True)
        (dx,) = torch.autograd.grad(second_x.sum(), c)
        delta = torch.zeros_like(c)
        delta[:, 0, 0] = 0.0007
        with torch.no_grad():
            low, lx, lc = score(c - delta)
            high, hx, hc = score(c + delta)
    assert all(torch.equal(a, b) and torch.equal(b, e) for a, b, e in zip(lc, clocks, hc))
    for gradient, finite_difference in ((dm, (high - low) / 0.0014), (dx, (hx - lx) / 0.0014)):
        assert gradient[:, 0, 0].abs().min() > 0.05
        torch.testing.assert_close(gradient[:, 0, 0].double(), finite_difference.double(), atol=0.004, rtol=0.01)
