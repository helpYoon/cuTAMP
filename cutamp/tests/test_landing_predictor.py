import math

import pytest
import torch

from cutamp.landing_predictor import LandingProfile, landing_errors, near_rest, phase_checks


def test_strict_radial_and_wrapped_tracking_is_independent_of_settling():
    reference = torch.zeros(4, 2, 3, dtype=torch.float64)
    achieved = reference.clone()
    achieved[0, 0, :2] = 0.015  # each component inside, diagonal outside
    achieved[1, 1, 0] = 0.020
    achieved[2, 1, 2] = math.pi / 180
    reference[3, 0, 2], achieved[3, 0, 2] = math.pi - 0.002, -math.pi + 0.003
    xy, yaw, good = landing_errors(achieved, reference)
    assert good.tolist() == [False, False, False, True]
    assert abs(yaw[3, 0].item() - 0.005) < 1.0e-12
    assert xy[0, 0] > 0.020
    # The last row tracks despite its elevated final-window velocities.
    assert not near_rest(torch.tensor(0.0128), torch.tensor(0.0082)).item()
    assert not near_rest(torch.tensor(0.01), torch.tensor(0.001)).item()
    assert near_rest(torch.tensor(0.009), torch.tensor(0.019)).item()


def test_phase_checks_use_each_rows_deadline_and_ignore_padded_ticks():
    reference = torch.zeros(3, 2, 3, dtype=torch.float64)
    trace = reference[:, None].repeat(1, 6, 1, 1)
    heights = torch.zeros_like(trace[..., 0])
    clock = torch.tensor([2, 3, 2])
    valid = torch.arange(6)[None] < torch.tensor([5, 6, 5])[:, None]
    trace[0, 0, 1, 0] = 0.011  # transient support slip before accurate arrival
    trace[1, 2, 0, 0] = 0.051  # swing is late, then drifts from its deadline pose
    heights[1, 2, 0] = 0.020  # strictly below 2 cm
    trace[2, 5, :, :] = 1e3  # padding belongs to neither phase
    checks = phase_checks(trace, heights, reference, 0, clock, valid, LandingProfile())
    assert checks.deadline_ok.tolist() == [True, False, True]
    assert checks.support_ok.tolist() == [False, True, True]
    assert checks.hold_ok.tolist() == [True, False, True]
    assert checks.hold_xy.shape == (3, 2)


def test_two_locally_accurate_issues_can_miss_the_intended_world_stance():
    from types import SimpleNamespace

    from cutamp.landing_predictor import LandingPredictor, LandingState, _intended_pair

    # Introduce 1.2 cm forward error on each swinging sole. R's second command
    # uses the achieved L foot, so the final world error accumulates to 2.4 cm.
    class Predictor:
        device = torch.device("cpu")
        model = SimpleNamespace(feet=lambda x: (x[0],))
        hold_ticks = 75

        def _step(self, state, side, command, hold_ticks):
            assert hold_ticks == (0 if side == 0 else self.hold_ticks)
            issued = _intended_pair(state.tensors[0], side, command)
            achieved = issued.clone()
            achieved[:, side, 0] += 0.012
            xy, yaw, good = landing_errors(achieved, issued)
            return SimpleNamespace(
                reference=issued,
                achieved=achieved,
                xy_error=xy,
                yaw_error=yaw,
                tracking_ok=good,
                terminal=LandingState((achieved,) * 9, state.dynamics, state.history_source),
            )

    standing = torch.tensor([[[0.0, 0.105, 0.0], [0.0, -0.105, 0.0]]])
    start = LandingState((standing,) * 9, torch.zeros(1, 38), "analytic regression")
    c = torch.tensor([[[0.12, 0.21, 0.0], [0.12, -0.21, 0.0]]])
    out = LandingPredictor.rollout(Predictor(), start, "LR", c)
    assert all(step.tracking_ok.item() for step in out.steps)
    assert not out.tracking_ok.item()
    torch.testing.assert_close(out.final_xy_error, torch.tensor([[0.012, 0.024]]))


def test_a_step_is_done_when_the_foot_is_down_at_its_last_tick():
    reference = torch.zeros(2, 2, 3, dtype=torch.float64)
    trace = reference[:, None].repeat(1, 4, 1, 1)
    heights = torch.zeros_like(trace[..., 0])
    clock, valid = torch.tensor([2, 2]), torch.ones(2, 4, dtype=torch.bool)
    trace[0, 1:, 0, 0] = 0.20          # lands 20 cm from where it was sent, and stays there
    heights[1, 1, 0] = 0.03            # still 3 cm up at its last tick
    checks = phase_checks(trace, heights, reference, 0, clock, valid, LandingProfile())
    assert checks.deadline_ok.tolist() == [True, False]
    assert checks.deadline_xy[0] == pytest.approx(0.20) and checks.hold_ok[0]
    assert not hasattr(LandingProfile(), "deadline_xy") and not hasattr(LandingProfile(), "deadline_yaw")
