import math

import numpy as np
import torch

from cutamp.stance import abs_pose, rel_pose
from cutamp.step_refresh import StandIn

N = 3


def incoming():
    return torch.tensor([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]).repeat(N, 1, 1)  # left sole, right sole


def standin():
    torch.manual_seed(0)
    s = StandIn(("L", "", "RL"), [None] * 3, [None] * 3, [None] * 3)
    s.relative[0] = torch.tensor([[[0.2, 0.2, 0.1], [0.0, 0.0, 0.0]]]).repeat(N, 1, 1, 1)
    s.slope[0] = torch.zeros(N, 1, 2, 3, 3)
    s.slope[0][:, 0, 0] = torch.eye(3)  # the swung sole follows its command exactly
    s.at[0] = torch.tensor([[[0.2, 0.2, 0.1]]]).repeat(N, 1, 1)
    s.relative[2] = torch.randn(N, 2, 2, 3) * 0.05
    s.slope[2] = torch.randn(N, 2, 2, 3, 6) * 0.5
    s.at[2] = torch.randn(N, 2, 3) * 0.05
    return s


def test_equal_to_the_refresh_at_the_refresh_commands():
    s, inc = standin(), incoming()
    after, final = s.stances(inc, [s.at[0], None, s.at[2]])
    for sole in range(2):
        torch.testing.assert_close(after[0][:, 0, sole], abs_pose(inc[:, 1], s.relative[0][:, 0, sole]))
        for k in range(2):  # gap 2 swings the right foot first: its anchor is the left sole it starts on
            torch.testing.assert_close(after[2][:, k, sole], abs_pose(final[1][:, 0], s.relative[2][:, k, sole]))
    torch.testing.assert_close(final[0], after[0][:, -1])
    torch.testing.assert_close(final[2], after[2][:, -1])


def test_gap_without_steps_stands_on_the_stance_before_it():
    s, inc = standin(), incoming()
    after, final = s.stances(inc, [s.at[0], None, s.at[2]])
    assert after[1] is None
    torch.testing.assert_close(final[1], final[0])


def test_later_gap_rides_on_the_earlier_gap():
    s, inc = standin(), incoming()
    _, base = s.stances(inc, [s.at[0], None, s.at[2]])
    moved_command = s.at[0] + torch.tensor([0.03, -0.02, 0.05])
    after, final = s.stances(inc, [moved_command, None, s.at[2]])
    assert not torch.allclose(final[2], base[2])
    for sole in range(2):  # seen from the moved left sole, the later gap's landing is where the refresh had it
        torch.testing.assert_close(rel_pose(final[0][:, 0], final[2][:, sole]), s.relative[2][:, -1, sole], atol=1e-6, rtol=0)


def test_gradient_reaches_every_gaps_commands():
    s, inc = standin(), incoming()
    first, third = s.at[0].clone().requires_grad_(), s.at[2].clone().requires_grad_()
    _, final = s.stances(inc, [first, None, third])
    g_first, g_third = torch.autograd.grad(final[2].sum(), (first, third))
    assert g_first.abs().sum() > 0 and g_third.abs().sum() > 0


from cutamp.step_refresh import as_commanded, command_rows, draw_commands


def test_draws_lie_in_the_allowed_set_and_repeat():
    orders, inc = ("LR", "", "R"), incoming().repeat(22, 1, 1)[:64]
    draws = draw_commands(orders, inc, 64, torch.Generator().manual_seed(0))
    assert draws[1] is None and draws[0].shape == (64, 2, 3) and draws[2].shape == (64, 1, 3)
    after, _ = as_commanded(orders, inc, draws)
    rows = command_rows(orders, inc, draws, after)
    assert rows and all((v == 0).all() for v in rows.values())
    assert (draws[0][:, 0, 1] > 0).all() and (draws[0][:, 1, 1] < 0).all()  # a right swing is the mirror image
    again = draw_commands(orders, inc, 64, torch.Generator().manual_seed(0))
    assert all(torch.equal(a, b) for a, b in zip(draws, again) if a is not None)
    assert draws[0][:, 0, 0].std() > 0.05  # spread over the set, not one point


def test_rows_rise_outside_the_set_and_have_a_slope():
    inc = incoming()[:1]
    u = torch.tensor([[[0.50, 0.25, 0.0]]], requires_grad=True)  # x beyond the trained 0.33 m
    after, _ = as_commanded(("L",), inc, [u])
    rows = command_rows(("L",), inc, [u], after)
    assert rows["gap0_step1_command_bound"].item() > 0
    (g,) = torch.autograd.grad(rows["gap0_step1_command_bound"].sum(), u)
    assert g.abs().sum() > 0
    far = torch.tensor([[[0.50, 0.60, 0.0]]])  # 0.64 m from where the foot stands, against 0.60 m
    after, _ = as_commanded(("L",), inc, [far])
    assert command_rows(("L",), inc, [far], after)["gap0_step1_max_travel"].item() > 0


def test_accuracy_rows_read_the_landed_pair():
    inc = incoming()[:1]
    u = torch.tensor([[[0.15, 0.22, 0.0]]])
    after, _ = as_commanded(("L",), inc, [u])
    assert (command_rows(("L",), inc, [u], after)["gap0_step1_xy_precision"] == 0).all()
    missed = [after[0].clone()]
    missed[0][:, 0, 0, 0] += 0.03  # the swung sole lands 3 cm off, against a 2 cm tolerance
    row = command_rows(("L",), inc, [u], missed)["gap0_step1_xy_precision"]
    torch.testing.assert_close(row[0], torch.tensor([0.03**2 / 0.02**2 - 1, 0.0]), atol=1e-5, rtol=0)


from cutamp.reachability import _rectangle_separation
from cutamp.step_refresh import feet_clear, grid_commands


def test_the_grid_is_the_half_foot_lattice_inside_the_limit():
    half = np.array([0.224868, 0.10]) / 2  # the reach map's sole_box_m
    grid = grid_commands(half, 0.2125)
    assert grid.dtype == torch.float64 and grid.shape == (100, 3)
    xs, ys, turns = (sorted({round(v, 9) for v in grid[:, a].tolist()}) for a in range(3))
    assert xs == [round(k * 0.112434, 9) for k in (-2, -1, 0, 1, 2)]
    assert ys == [round(0.1625 + 0.05 * k, 9) for k in range(7)]
    assert turns == [0.0, round(math.radians(25), 9), round(math.radians(50), 9)]
    full = {(round(x, 9), round(y, 9), round(t, 9)) for x in xs for y in ys for t in turns}
    kept = {tuple(round(v, 9) for v in row) for row in grid.tolist()}
    refused = {(-0.112434, 50), (0.0, 25), (0.0, 50), (0.112434, 25), (0.112434, 50)}
    refused = {(round(x, 9), 0.1625, round(math.radians(t), 9)) for x, t in refused}
    assert full - kept == refused
    for x, y, t in grid.tolist():  # apply_command's static test (reachability.py:80-85) holds for every kept command
        assert -0.33 <= x <= 0.33 and 0.14 <= y <= 0.48 and math.radians(-20) <= t <= math.radians(50)
        assert _rectangle_separation([x, y], np.array([0.112434, 0.05]), np.array([0.112434, 0.05]), t) >= 0.02


def test_feet_clear_follows_the_refresh_rule():
    fh = torch.tensor([0.1124, 0.05, 0.0218], dtype=torch.float64)
    def box(x, y, z, hx, hy, hz):
        T = torch.eye(4, dtype=torch.float64); T[:3, 3] = torch.tensor([x, y, z]); return T, [hx, hy, hz]
    cases = [  # (foot (x, y, yaw), box, clear)
        ((0.30, 0.0, 0.0), box(0.30, 0.0, 0.005, 0.2, 0.2, 0.005), False),          # on a 1 cm pad
        ((-0.25, 0.0, 0.0), box(0.30, 0.0, 0.005, 0.2, 0.2, 0.005), True),          # beside it
        ((0.30, 0.0, 0.0), box(0.30, 0.0, -0.005, 0.2, 0.2, 0.005), True),          # flush pad, top at 0: ignored
        ((0.05, 0.0, 0.0), box(0.15, 0.0, 0.025, 0.025, 0.025, 0.025), False),      # toe in a movable cube
        ((0.0, 0.0, 0.0), box(0.1374 - 5e-7, 0.0, 0.025, 0.025, 0.025, 0.025), True),   # 5e-7 m overlap: tolerated
        ((0.0, 0.0, 0.0), box(0.1374 - 2e-6, 0.0, 0.025, 0.025, 0.025, 0.025), False),  # 2e-6 m: not
        ((0.0, 0.15, math.pi / 2), box(0.0, 0.25, 0.025, 0.025, 0.025, 0.025), False),  # a turned sole reaches it
        ((0.0, 0.15, 0.0), box(0.0, 0.25, 0.025, 0.025, 0.025, 0.025), True),           # a straight one does not
    ]
    feet = torch.tensor([[f] for f, _, _ in cases], dtype=torch.float64)          # [8, 1, 3]
    boxes = torch.stack([T for _, (T, _), _ in cases])[:, None]                   # [8, 1, 4, 4]
    for i, (_, (_, h), clear) in enumerate(cases):
        got = feet_clear(feet[i:i + 1], boxes[i:i + 1], torch.tensor([h], dtype=torch.float64), fh)
        assert got.tolist() == [clear], i
