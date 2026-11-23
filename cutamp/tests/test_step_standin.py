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


from cutamp.step_refresh import as_commanded, command_rows


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


import pytest

from cutamp import step_refresh
from cutamp.landing_predictor import _intended_pair
from cutamp.stance import STEP_TRAVEL
from cutamp.step_bound import planner_step_bound, travel
from cutamp.step_refresh import NoStartingPath, draw_grid_paths

FOOT_HALF = torch.tensor([0.1124, 0.05, 0.0218], dtype=torch.float64)


def pad_at(n, xy, half):
    """``(boxes [n, 1, 4, 4], halves [1, 3])`` float64: a 1 cm tall pad on the floor centred at ``xy``, half extents
    ``half`` in x and y."""
    T = torch.eye(4, dtype=torch.float64)
    T[:2, 3] = torch.tensor(xy, dtype=torch.float64)
    T[2, 3] = 0.005
    return T.expand(n, 1, 4, 4).clone(), torch.tensor([[*half, 0.005]], dtype=torch.float64)


PAD32 = pad_at(32, (0.25, 0.35), (0.06, 0.06))


def legal64(orders, inc, draws):
    """Every step inside the step limit (``g <= 0``) and its travel within ``STEP_TRAVEL``, in float64, each step from
    the intended pair of the steps before it."""
    bound, before, ok = planner_step_bound(torch.device("cpu"), torch.float64), inc.double(), True
    for order, u in zip(orders, draws):
        for k, foot in enumerate(order):
            side, c = "LR".index(foot), u[:, k].double()
            t = travel(rel_pose(before[:, 1 - side], before[:, side]), c)
            left = torch.full((len(c),), side == 0)
            ok &= bool(((bound.g(c, left) <= 0) & (t >= STEP_TRAVEL[0]) & (t <= STEP_TRAVEL[1])).all())
            before = _intended_pair(before, side, c)
    return ok


def end_pairs(orders, inc, draws):
    """Per gap the ``[n, 2, 3]`` float64 pair at its end, as numpy."""
    _, final = as_commanded(orders, inc.double(), [None if u is None else u.double() for u in draws])
    return [f.numpy() for f in final]


def landings(orders, inc, draws):
    """``[n, k, 3]`` float64: every landed sole of every gap, in step order."""
    after, _ = as_commanded(orders, inc.double(), [None if u is None else u.double() for u in draws])
    return torch.stack([after[j][:, k, "LR".index(f)] for j, order in enumerate(orders) for k, f in enumerate(order)], 1)


def near_grid(draws, grid):
    """Every command, as its left-swing equivalent (a right swing's y is negative), within half a cell of some grid row
    on every axis (+1e-6)."""
    half = torch.stack([grid[:, a].unique().diff().min() for a in range(3)]) / 2 + 1e-6
    u = torch.cat([d.reshape(-1, 3) for d in draws if d is not None]).double()
    u = torch.where(u[:, 1:2] < 0, u * torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64), u)
    return bool(((u[:, None] - grid[None]).abs() <= half).all(-1).any(-1).all())


def test_grid_paths_pass_their_tests_and_repeat():
    orders, inc = ("LR", "", "R"), incoming().repeat(22, 1, 1)[:64].double()
    grid = grid_commands(np.array([0.224868, 0.10]) / 2, 0.2)  # incoming()'s soles stand 0.2 m apart
    pad = pad_at(64, (0.25, 0.35), (0.06, 0.06))
    reach = [lambda p: p[:, :, 0].mean(1) > 0.05, None, lambda p: p[:, :, 0].mean(1) > 0.10]
    draw = lambda: draw_grid_paths(orders, inc, torch.Generator().manual_seed(0), grid, reach, [pad, None, pad], FOOT_HALF)
    draws, info = draw()
    assert draws[1] is None and draws[0].shape == (64, 2, 3) and draws[2].shape == (64, 1, 3)
    assert draws[0].dtype == torch.float32 and (info["rounds"], info["own"], info["reached"]) == (1, 64, True)
    assert info["noisy"] > 0
    assert (draws[0][:, 0, 1] > 0).all() and (draws[0][:, 1, 1] < 0).all()  # a right swing is the mirror image
    assert legal64(orders, inc, draws) and near_grid(draws, grid)
    ends = end_pairs(orders, inc, draws)
    assert reach[0](ends[0]).all() and reach[2](ends[2]).all()
    assert feet_clear(landings(orders, inc, draws), pad[0], pad[1], FOOT_HALF).all()
    again, _ = draw()
    assert all(torch.equal(a, b) for a, b in zip(draws, again) if a is not None)
    assert draws[0][:, 0, 0].std() > 0.05  # spread over the grid, not one path


def test_rounds_stop_at_the_first_dry_round_and_share_paths():
    inc, grid = incoming()[:1].repeat(16, 1, 1).double(), grid_commands(np.array([0.224868, 0.10]) / 2, 0.2)
    calls, target = [], inc[0, 1].numpy() + [0.224868, 0.35, 0.0]  # from the planted right sole, as float32 put it
    def one_path_once(pairs):  # passes only the landing of grid command (0.224868, 0.35, 0), and only in its first call
        calls.append(len(pairs))
        hit = (np.abs(pairs[:, 0] - target) < 1e-9).all(1)
        return hit & (len(calls) == 1)
    draws, info = draw_grid_paths(("L",), inc, torch.Generator().manual_seed(0), grid, [one_path_once], [None], FOOT_HALF)
    assert info["rounds"] == 2 and 0 < info["own"] < 16 and info["reached"] and info["noisy"] == 0
    assert torch.allclose(draws[0], torch.tensor([0.224868, 0.35, 0.0]).expand(16, 1, 3))


def test_a_shared_path_passes_the_receivers_own_tests():
    inc, grid = incoming()[:1].repeat(16, 1, 1).double(), grid_commands(np.array([0.224868, 0.10]) / 2, 0.2)
    far, near = inc[0, 1].numpy() + [0.224868, 0.35, 0.0], inc[0, 1].numpy() + [0.0, 0.35, 0.0]  # landings of 2 commands
    scene = pad_at(16, (5.0, 5.0), (0.03, 0.03))
    scene[0][8:, 0, :2, 3] = torch.from_numpy(far[:2])  # particles 8-15 have a pad where the far command lands
    calls = []
    def two_paths_but_round_two(pairs):  # passes the two landings, except in its second call: round 2 is dry
        calls.append(len(pairs))
        hit = (np.abs(pairs[:, 0] - far) < 1e-9).all(1) | (np.abs(pairs[:, 0] - near) < 1e-9).all(1)
        return hit & (len(calls) != 2)
    draws, info = draw_grid_paths(("L",), inc, torch.Generator().manual_seed(0), grid, [two_paths_but_round_two],
                                  [scene], FOOT_HALF)
    assert info["rounds"] == 2 and 0 < info["own"] < 16 and info["noisy"] == 0
    feet = landings(("L",), inc, draws)
    assert (feet[:8, 0, 0] > 0.2).any()  # some kept path is the far one, which fails particles 8-15
    assert feet_clear(feet, *scene, FOOT_HALF).all()  # so no particle of 8-15 was given it


def test_a_later_gaps_incoming_pair_is_feet_tested():
    orders, inc = ("L", "", "R"), incoming()[:1].repeat(32, 1, 1).double()
    grid = grid_commands(np.array([0.224868, 0.10]) / 2, 0.2)
    pad = pad_at(32, (0.22, 0.3), (0.08, 0.08))  # where gap 0's left sole often lands; in gap 2's scene only
    draws, _ = draw_grid_paths(orders, inc, torch.Generator().manual_seed(0), grid, [None] * 3, [None, None, pad],
                               FOOT_HALF)
    assert feet_clear(torch.from_numpy(end_pairs(orders, inc, draws)[0]), *pad, FOOT_HALF).all()


def test_a_plan_where_nothing_reaches_starts_on_legal_clear_paths():
    orders, inc = ("RL",), incoming()[:1].repeat(32, 1, 1).double()
    grid = grid_commands(np.array([0.224868, 0.10]) / 2, 0.2)
    draws, info = draw_grid_paths(orders, inc, torch.Generator().manual_seed(0), grid,
                                  [lambda p: np.zeros(len(p), bool)], [PAD32], FOOT_HALF)
    assert not info["reached"] and legal64(orders, inc, draws)
    assert feet_clear(landings(orders, inc, draws), *PAD32, FOOT_HALF).all()


def test_no_clear_path_raises(monkeypatch):
    monkeypatch.setattr(step_refresh, "REDRAW_ROUNDS", 2)
    grid = grid_commands(np.array([0.224868, 0.10]) / 2, 0.2)
    with pytest.raises(NoStartingPath):  # a 3 x 3 m pad, 1 cm tall, under every landing
        draw_grid_paths(("L",), incoming()[:1].repeat(4, 1, 1).double(), torch.Generator().manual_seed(0),
                        grid, [None], [pad_at(4, (0.0, 0.0), (1.5, 1.5))], FOOT_HALF)


def test_the_start_pair_is_not_feet_tested():
    grid = grid_commands(np.array([0.224868, 0.10]) / 2, 0.2)
    under_start = pad_at(8, (0.0, 0.0), (0.15, 0.16))  # under both starting soles, clear of the far landings
    draws, info = draw_grid_paths(("L",), incoming()[:1].repeat(8, 1, 1).double(), torch.Generator().manual_seed(0),
                                  grid, [None], [under_start], FOOT_HALF)  # would raise if the start pair were tested
    inc = incoming()[:1].repeat(8, 1, 1).double()
    assert info["reached"] and feet_clear(landings(("L",), inc, draws), *under_start, FOOT_HALF).all()
