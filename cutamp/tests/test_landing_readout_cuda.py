"""The sole FK readout is equivalent to running the deployed FK separately at each control tick."""

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from cutamp import landing_predictor as lp
from cutamp.tests.conftest import needs_cuda


def _per_tick_readout(model, physical):
    outputs = [model.feet(tuple(x[:, t] for x in physical)) for t in range(physical[0].shape[1])]
    return tuple(torch.stack(values, dim=1) for values in zip(*outputs))


@needs_cuda
@pytest.mark.parametrize(("batch", "order"), [(1, "L"), (2, "LR"), (2, "RL")])
def test_batched_sole_readout_preserves_frames_clocks_and_command_gradients(batch, order, monkeypatch):
    bundle = Path(__file__).resolve().parents[2] / "data/t1-reachability/landing-accuracy/landing-seed3.pt"
    if not bundle.is_file():
        pytest.skip("Requires the exported seed-3 deployment")
    predictor = lp.LandingPredictor(bundle)
    context = torch.zeros(batch, 38, device=predictor.device)
    context[:, 0] = 0.25 / 0.45
    context[:, -5:] = -1.0
    initial = predictor.standing_start(context)
    sign = 1 if order[0] == "L" else -1
    command = context.new_tensor(
        [
            [[0.12, 0.21 * sign, 0.04 * sign], [0.10, -0.21 * sign, -0.02 * sign]],
            [[0.24, 0.21 * sign, 0.0], [0.06, -0.21 * sign, -0.06 * sign]],
        ]
    )[:batch, : len(order)]

    def evaluate():
        with lp.prediction_precision():
            u = command.clone().requires_grad_()
            result = predictor.rollout(initial, order, u)
            score = 0.0
            for step in result.steps:
                # A varying signed covector covers both soles, their heights, tilts,
                # the clock-dependent valid interval and the accumulated endpoint.
                t = torch.arange(step.sole_trace.shape[1], device=u.device, dtype=u.dtype)
                weights = ((t + 1) * 0.07).cos()[None, :, None] * step.valid_ticks[:, :, None]
                measure = step.sole_trace @ u.new_tensor([0.3, -0.7, 0.4]) + step.sole_height - 0.2 * step.sole_tilt
                score = score + (weights * measure * u.new_tensor([1.0, -0.5])).sum() / len(t)
            score = score + result.final_xy_error.sum() + result.final_yaw_error.sum()
            gradient = torch.autograd.grad(score, u)[0]
        return result, gradient

    actual, gradient = evaluate()
    monkeypatch.setattr(lp, "_sole_trajectory", _per_tick_readout)
    reference, reference_gradient = evaluate()
    assert gradient.abs().amax() > 0.01
    torch.testing.assert_close(gradient, reference_gradient, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(actual.tracking_ok, reference.tracking_ok)
    for step, ref in zip(actual.steps, reference.steps):
        assert torch.equal(step.clock, ref.clock)
        if batch > 1:
            assert actual.steps[0].clock[0] != actual.steps[0].clock[1]
        for key in (
            "valid_ticks",
            "sole_trace",
            "sole_height",
            "sole_tilt",
            "reference",
            "achieved",
            "xy_error",
            "yaw_error",
            "actions",
            "tracking_ok",
            "near_rest",
        ):
            torch.testing.assert_close(getattr(step, key), getattr(ref, key), atol=2e-6, rtol=2e-5)
        for field in fields(step.phases):
            torch.testing.assert_close(
                getattr(step.phases, field.name), getattr(ref.phases, field.name), atol=2e-6, rtol=2e-5
            )
        for v, original in zip(step.terminal.tensors, ref.terminal.tensors):
            torch.testing.assert_close(v, original, atol=0, rtol=0)
