"""CPU regression for differentiable knot conversion and teacher-only verification."""

from unittest.mock import patch

import pytest
import torch

from cutamp import step_adam
from cutamp.landing_predictor import LandingPredictor, LandingProfile, LandingState, _intended_pair
from experiments.step_sequence_adapter import predict_sequence, student_objectives


class SimpleFeet:
    def feet(self, state):
        q = state[4]
        feet = q[:, :6].reshape(-1, 2, 3)
        return feet, q[:, 6:8], q[:, 8:10]


def simple_model(state, dynamics, initial_feet, sides, commands):
    batch = len(commands)
    feet = initial_feet
    initial_body = torch.cat([x.reshape(batch, -1) for x in state[:6]], -1)
    feet_steps, bodies = [], []
    for i in range(2):
        side = int(sides[0, i].clamp_min(0))
        feet = _intended_pair(feet, side, commands[:, i])
        motions = feet[:, None].expand(-1, 6, -1, -1)
        body = initial_body[:, None].expand(-1, 6, -1).clone()
        body[..., 18:24] = motions.reshape(batch, 6, 6)
        feet_steps.append(motions)
        bodies.append(body)
    return torch.stack(bodies, 1), torch.stack(feet_steps, 1)


def fixture(batch=3):
    p = torch.zeros(batch, 3)
    q = torch.zeros(batch, 29)
    state = LandingState((p, torch.eye(3).expand(batch, 3, 3), p, p, q, q, q, q, q), torch.zeros(batch, 38), "test")
    teacher = LandingPredictor.__new__(LandingPredictor)
    teacher.hold_ticks = 75
    teacher.profile = LandingProfile()
    teacher.model = SimpleFeet()
    return state, teacher


@pytest.mark.parametrize("order", ["L", "R", "LR", "RL"])
def test_knots_and_cross_gap_gradients(order):
    state, teacher = fixture()
    state.tensors[4].requires_grad_()
    commands = torch.full((3, len(order), 3), 0.1, requires_grad=True)
    result = predict_sequence(simple_model, teacher, state, order, commands)
    assert result.tracking_ok.all()
    assert [len(step.trajectory[0][0]) for step in result.steps] == ([6] if len(order) == 1 else [4, 6])
    # A downstream candidate starts from the preceding predicted physical posture.
    next_state = LandingState(result.terminal.tensors + state.tensors[6:], state.dynamics, "next")
    next_side = "R" if order[-1] == "L" else "L"
    later = predict_sequence(simple_model, teacher, next_state, next_side, torch.full((3, 1, 3), 0.05))
    loss = later.steps[-1].achieved.sum()
    cmd_grad, start_grad = torch.autograd.grad(loss, (commands, state.tensors[4]))
    assert (cmd_grad.abs().sum(-1) > 0).all() and (start_grad.abs().sum(-1) > 0).all()


@pytest.mark.parametrize("plateau,cap,expected", [(None, 2, 2), (step_adam.PlateauStop(1., 2), 32, 2)])
def test_original_rollout_used_for_all_selected_adam_rows(plateau, cap, expected):
    from cutamp.step_ik import _IK_TOLERANCE

    state, teacher = fixture()
    native_calls = []

    def native(teacher, state, order, commands, **kwargs):
        assert _IK_TOLERANCE.get() == 2e-9
        native_calls.append(order)
        return predict_sequence(simple_model, teacher, state, order, commands)

    with patch.object(LandingPredictor, "rollout", native):
        with student_objectives(simple_model, ik_objective_tol=2e-7) as diagnostics:
            initial = {"foot0": torch.full((3, 2, 3), 0.1)}

            def objective(values):
                assert _IK_TOLERANCE.get() == 2e-7
                result = teacher.rollout(state, "LR", values["foot0"])
                center = result.steps[-1].achieved.square().sum((1, 2))
                return torch.zeros_like(center), center

            best, _ = step_adam.optimize_particles(initial, objective, cap, 0.003,
                                                  plateau=plateau, trace_center=True)
            assert not native_calls
            teacher.rollout(state, "LR", best["foot0"])
        assert LandingPredictor.rollout is native and native_calls == ["LR"]
    assert diagnostics[0]["updates"] == expected and diagnostics[0]["max_updates"] == cap
    assert diagnostics[0]["stop_reason"] == ("plateau" if plateau is not None else "max_updates")
    assert len(diagnostics[0]["best_feasible_center_m"]) == diagnostics[0]["student_rollouts"] == expected + 1
    replay = diagnostics[0]["replayed_gaps"][0]
    assert replay["false_pass"] == replay["false_fail"] == 0
    assert replay["landing_p90_xy_cm"] == 0 and replay["exact_tracking_pass"] == 3
