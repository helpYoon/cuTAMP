import os
from pathlib import Path

import pytest
import torch

from cutamp.landing_accuracy import LANDING_BUNDLE
from cutamp.landing_predictor import LandingPredictor
from cutamp.stance import rel_pose
from cutamp.step_refresh import NUDGE, refresh_gap
from cutamp.tests.conftest import needs_cuda

COMMANDS = [[[0.12, 0.21, 0.0], [0.10, -0.21, 0.0]], [[0.24, 0.21, 0.0], [0.06, -0.21, 0.0]]]


@pytest.fixture(scope="module")
def predictor():
    return LandingPredictor(Path(os.environ.get("CUTAMP_LANDING_BUNDLE", LANDING_BUNDLE)), use_cuda_graph=True)


def standing(predictor, n):
    context = torch.zeros(n, 38, device="cuda")
    context[:, 0] = 0.25 / 0.45
    context[:, -5:] = -1.0
    start = predictor.standing_start(context)
    return start, predictor.model.feet(start.tensors[:6])[0]


@needs_cuda
def test_first_rows_are_the_particles_own_rollout(predictor):
    start, feet = standing(predictor, 2)
    u = torch.tensor(COMMANDS, device="cuda")
    r = refresh_gap(predictor, start, "LR", u, feet[:, 1])  # "LR" swings the left foot first: the right sole is planted
    with torch.no_grad():
        plain = predictor.rollout(start, "LR", u)
    for k in range(2):
        torch.testing.assert_close(r.landed[:, k], plain.steps[k].achieved, rtol=0, atol=2e-5)
        assert torch.equal(r.prediction.steps[k].clock, plain.steps[k].clock)
    assert r.prediction.steps[-1].trajectory[0].shape[0] == 2
    assert r.slope.shape == (2, 2, 2, 3, 6) and r.ticks_changed.shape == (2, 6)
    torch.testing.assert_close(r.relative[:, 1, 0], rel_pose(feet[:, 1], plain.steps[1].achieved[:, 0]), rtol=0, atol=2e-5)
    assert torch.equal(r.commands, u) and r.commands is not u


@needs_cuda
def test_slope_matches_the_backward_pass(predictor):
    start, feet = standing(predictor, 2)
    u = torch.tensor(COMMANDS, device="cuda")
    r = refresh_gap(predictor, start, "LR", u, feet[:, 1])
    assert not r.ticks_changed.any()  # if this fails, move the offending command number by 5 mm
    w = u.clone().requires_grad_()
    out = predictor.rollout(start, "LR", w)
    for sole in range(2):
        for axis in range(3):
            value = rel_pose(feet[:, 1], out.steps[-1].achieved[:, sole])[:, axis].sum()
            (g,) = torch.autograd.grad(value, w, retain_graph=True)
            torch.testing.assert_close(r.slope[:, -1, sole, axis], g.reshape(2, 6), rtol=0.05, atol=0.02)


@needs_cuda
def test_a_nudge_that_changes_a_tick_count_is_not_used(predictor):
    start, feet = standing(predictor, 1)

    def ticks(x):
        with torch.no_grad():
            u = torch.tensor([[[x, 0.21, 0.0]]], device="cuda")
            return int(predictor.rollout(start, "L", u, final_hold_ticks=0).steps[0].clock)

    lo, hi = 0.10, 0.16
    low = ticks(lo)
    assert ticks(hi) != low
    for _ in range(12):  # close in on the x where the step gains a tick
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ticks(mid) == low else (lo, mid)
    u = torch.tensor([[[hi - NUDGE / 2, 0.21, 0.0]]], device="cuda")
    r = refresh_gap(predictor, start, "L", u, feet[:, 1])
    assert r.ticks_changed[0].tolist() == [True, False, False]
    as_commanded = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device="cuda")  # swung sole, planted sole
    torch.testing.assert_close(r.slope[0, 0, :, :, 0], as_commanded, rtol=0, atol=1e-4)
    again = refresh_gap(predictor, start, "L", u, feet[:, 1], previous_slope=torch.full_like(r.slope, 7.0))
    assert (again.slope[0, 0, :, :, 0] == 7.0).all() and not (again.slope[0, 0, :, :, 1] == 7.0).any()


@needs_cuda
def test_every_copy_of_the_rollout_is_kept(predictor):
    start, feet = standing(predictor, 2)
    u = torch.tensor(COMMANDS, device="cuda")                     # two particles, two steps: seven copies each
    r = refresh_gap(predictor, start, "LR", u, feet[:, 1])
    assert r.copy_commands.shape == (14, 2, 3) and len(r.copies.steps[0].achieved) == 14
    assert torch.equal(r.copy_commands[:2], u)
    torch.testing.assert_close(r.copies.steps[0].achieved[:2], r.prediction.steps[0].achieved)
    torch.testing.assert_close((r.copy_commands[2:4] - u)[:, 0],
                               torch.tensor([NUDGE, 0.0, 0.0], device="cuda").expand(2, 3), rtol=0, atol=1e-8)


@needs_cuda
def test_a_wild_row_does_not_change_its_neighbours(predictor):
    start, feet = standing(predictor, 2)
    u = torch.tensor(COMMANDS, device="cuda")
    wild = u.clone()
    wild[1, 0] = torch.tensor([0.9, 0.9, 1.5], device="cuda")  # far outside the allowed set
    a = refresh_gap(predictor, start, "LR", u, feet[:, 1])
    b = refresh_gap(predictor, start, "LR", wild, feet[:, 1])
    torch.testing.assert_close(b.landed[0], a.landed[0], rtol=0, atol=2e-5)
    torch.testing.assert_close(b.slope[0], a.slope[0], rtol=0, atol=2e-2)
