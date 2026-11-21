"""Predict differentiable T1 step sequences for the particle optimization objective."""

from types import SimpleNamespace

import torch

from cutamp.landing_predictor import (
    _intended_pair,
    _sole_trajectory,
    landing_errors,
    phase_checks,
    prediction_precision,
)


@prediction_precision()
def predict_sequence(model, teacher, initial, order, commands):
    """Predict physical knots for the native IK and collision objectives, with gradients to both gaps."""
    if order not in ("L", "R", "LR", "RL") or teacher.hold_ticks != 75:
        raise ValueError("The trained interface supports L/R/LR/RL with 75 final HOLD ticks")
    batch, n, _ = commands.shape
    if n != len(order):
        raise ValueError("Commands and step order disagree")
    start_feet = teacher.model.feet(initial.tensors[:6])[0]
    ids = ["LR".index(side) for side in order]
    sides = torch.tensor(ids + [-1] * (2 - n), device=commands.device).expand(batch, -1)
    padded = torch.nn.functional.pad(commands, (0, 0, 0, 2 - n))
    body, soles = model(initial.tensors, initial.dynamics, start_feet, sides, padded)

    intended = achieved = start_feet
    tracking = torch.ones(batch, device=commands.device, dtype=torch.bool)
    steps = []
    for j, side in enumerate(ids):
        knots = 6 if j + 1 == n else 4
        block = body[:, j, :knots]
        p, R, v, w, q, qd = block.split([3, 9, 3, 3, 29, 29], dim=-1)
        trajectory = (p, R.reshape(batch, knots, 3, 3), v, w, q, qd)
        fk_soles, height, tilt = _sole_trajectory(teacher.model, trajectory)
        trace = soles[:, j, :knots]
        reference = _intended_pair(achieved, side, commands[:, j])
        intended = _intended_pair(intended, side, commands[:, j])
        achieved = trace[:, -1]
        xy, yaw, precise = landing_errors(achieved, reference)
        valid = torch.ones(batch, knots, dtype=torch.bool, device=commands.device)
        # phase_checks indexes samples, so the fourth knot is always this row's deadline.
        deadline = sides.new_full((batch,), 4)
        phases = phase_checks(trace, height, reference, side, deadline, valid, teacher.profile)
        steps.append(
            SimpleNamespace(
                side=order[j],
                achieved=achieved,
                reference=reference,
                xy_error=xy,
                yaw_error=yaw,
                tracking_ok=precise,
                trajectory=trajectory,
                sole_trace=trace,
                sole_height=height,
                sole_tilt=tilt,
                valid_ticks=valid,
                phases=phases,
                fk_soles=fk_soles,
            )
        )
        tracking = tracking & precise
    xy, yaw, final_ok = landing_errors(achieved, intended)
    # The next gap builds fresh controller history at the retracted posture via StepStateAdapter.
    terminal = SimpleNamespace(tensors=tuple(x[:, -1] for x in trajectory))
    return SimpleNamespace(
        steps=steps,
        terminal=terminal,
        intended_final=intended,
        final_xy_error=xy,
        final_yaw_error=yaw,
        tracking_ok=tracking & final_ok,
    )
