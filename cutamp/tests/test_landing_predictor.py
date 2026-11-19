import math

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
