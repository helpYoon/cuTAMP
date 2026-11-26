"""CPU tests for the balance-first GN core (cutamp/gn_refresh.py)."""
import math

import pytest
import roma
import torch

from cutamp.gn_refresh import GN_COM_MARGIN, _balanced_step, _rotmat_to_rotvec, _row_layout
from cutamp.robots.t1 import REAL_LEGS

#: The base's pitch and roll columns, which level 4 pulls toward upright.
UPRIGHT = (list(REAL_LEGS.joint_names).index("base_j_pitch"), list(REAL_LEGS.joint_names).index("base_j_roll"))


def _synthetic(seed, B=64, coll=False):
    """``(stack, blocks)``: one ``_balanced_step`` argument tuple on four dep-major ``B``-row
    blocks of the real legs' cspace and its four per-block slices, with 12 synthetic sole rows and, with
    ``coll``, a synthetic collision row, in contact on about half the rows, and level 3 on. Some
    joints sit on a limit and some CoM rows are outside, so every active set fires."""
    g = torch.Generator().manual_seed(seed)
    dof = len(REAL_LEGS.joint_names)
    lower = -(2 * torch.rand(dof, generator=g) + 0.2)
    upper = 2 * torch.rand(dof, generator=g) + 0.2
    half = REAL_LEGS.support.keyframe_half
    col_mask, _ = _row_layout(REAL_LEGS, ("left", "left", "right", "right"), 4 * B, dof, torch.device("cpu"),
                              torch.float32)

    deps = []
    for c in col_mask.split(B):
        q = lower + torch.rand(B, dof, generator=g) * (upper - lower)
        on_limit = torch.rand(B, dof, generator=g) < 0.05
        q = torch.where(on_limit, torch.where(torch.rand(B, dof, generator=g) < 0.5, lower, upper), q)
        jac = 0.4 * torch.randn(B, 6, dof, generator=g) * c.unsqueeze(1)
        jac_com = 0.05 * torch.randn(B, 2, dof, generator=g) * c.unsqueeze(1)
        com = (2 * torch.rand(B, 2, generator=g) - 1) * torch.tensor(half) * 1.08
        r = 0.01 * torch.randn(B, 6, generator=g)
        jac_sole = 0.3 * torch.randn(B, 12, dof, generator=g) * c.unsqueeze(1)
        r_sole = 1e-3 * torch.randn(B, 12, generator=g)
        pen = torch.relu(0.01 * torch.randn(B, 1, generator=g))
        jac_pen = 0.1 * torch.randn(B, 1, dof, generator=g) * c.unsqueeze(1)
        deps.append((q, r, jac, jac_com, com, jac_sole, r_sole, pen, jac_pen))

    def args(dep, c):
        q, r, jac, jac_com, com, jac_sole, r_sole, pen, jac_pen = dep
        return (q, r, jac, jac_com, com, lower, upper, c, half, jac_sole, r_sole, (pen, jac_pen) if coll else None,
                coll)

    stack = args([torch.cat(x) for x in zip(*deps)], col_mask)
    blocks = [args(d, c) for d, c in zip(deps, col_mask.split(B))]
    return stack, blocks


def test_rotmat_to_rotvec_matches_roma_bitwise():
    torch.manual_seed(0)
    axes = torch.nn.functional.normalize(torch.randn(4096, 3), dim=-1)
    sets = (roma.random_rotmat(4096),                              # generic
            roma.rotvec_to_rotmat(1e-4 * torch.randn(4096, 3)),    # small-angle series branch
            roma.rotvec_to_rotmat((math.pi - 1e-3) * axes))        # near pi: choice != 3 branches
    for dtype in (torch.float32, torch.float64):
        for R in sets:
            R = R.to(dtype)
            assert torch.equal(_rotmat_to_rotvec(R), roma.rotmat_to_rotvec(R))


@pytest.mark.parametrize("coll", [False, True])
def test_balanced_step_stack_matches_per_dep(coll):
    # Every op is row-wise and the active-set rounds always run in full, so
    # no row's step (levels 1-2's, or with the collision row levels 1-3's and 1-4's) depends on which other deps
    # share its call.
    for seed in range(10):
        stack, blocks = _synthetic(seed, coll=coll)      # one [4B] stack + its 4 [B] blocks
        steps = _balanced_step(*stack, upright=UPRIGHT)
        per_dep = [_balanced_step(*b, upright=UPRIGHT) for b in blocks]
        for k in range(3):
            assert torch.equal(steps[k], torch.cat([s[k] for s in per_dep]))


@pytest.mark.parametrize("coll", [False, True])
def test_balanced_step_moves_no_pinned_column(coll):
    stack, _ = _synthetic(0, coll=coll)
    (step, step3, step12), col_mask = _balanced_step(*stack, upright=UPRIGHT), stack[7]
    assert torch.all(step[col_mask == 0] == 0)
    assert torch.any(step[col_mask == 1] != 0)  # not vacuous: the free columns move
    # Levels 3 and 4 run beside the collision row only, and each changes none of the active sets above it: levels
    # 1-2's step is bitwise the one they take without level 3, levels 1-3's the one they take without level 4, and a
    # row in contact takes levels 1-2's alone.
    assert torch.equal(step3, step12) != coll and torch.equal(step, step3) != coll
    assert torch.equal(step3, _balanced_step(*stack)[0])
    if coll:
        assert torch.equal(step12, _balanced_step(*stack[:-1], False)[0])
        in_contact = stack[11][0][:, 0] > 0
        assert bool(in_contact.any()) and torch.equal(step[in_contact], step12[in_contact])


def test_level4_moves_no_row_above_it_and_lowers_pitch_and_roll():
    # Level 4's own step, the full step less levels 1-3's, moves none of the rows above it (the soles, the hand and
    # both CoM axes) past float32 rounding, and it lowers pitch^2 + roll^2 from where levels 1-3's step leaves them on
    # every row it moves: it aims from that step, its rows are coordinates moved by a gain below 1, and the box only
    # shortens it.
    stack, _ = _synthetic(0, coll=True)
    step, step3, _ = _balanced_step(*stack, upright=UPRIGHT)
    q, jac, jac_com, jac_sole = stack[0], stack[2], stack[3], stack[9]
    d = step - step3
    size = d.norm(dim=-1)
    big = size > 1e-3
    assert int(big.sum()) >= 50 and float(size[big].median()) > 0.1  # not vacuous: 68 rows, 0.13 median
    for rows in (jac_sole, jac, jac_com):
        assert float((rows @ d.unsqueeze(-1)).abs().max()) < 1e-6  # 2e-8 here
    before, after = ((q + s)[:, list(UPRIGHT)].norm(dim=-1) for s in (step3, step))
    assert bool((after <= before + 1e-6).all())
    assert bool((after[big] < before[big]).all())


def test_level3_never_carries_a_free_com_axis_out(monkeypatch):
    # A row takes levels 1-2's step where level 3's would carry a CoM axis that step leaves inside the rectangle
    # past its edge (linear, as levels 1-2's pushes are tested). Level 3 never pulls that far on these rows, so here
    # it pushes outward: without the veto, 65 of the 256 rows end outside.
    import cutamp.gn_refresh as gn
    real = gn._centring_step
    monkeypatch.setattr(gn, "_centring_step", lambda *a: -real(*a))
    stack, _ = _synthetic(0, coll=True)
    step, _, step12 = _balanced_step(*stack)
    jac_com, com, half = stack[3], stack[4], torch.tensor(stack[8])

    def lin(s):
        return com + (jac_com @ s.unsqueeze(-1)).squeeze(-1)

    assert bool((step != step12).any())  # not vacuous: level 3 still moves rows it keeps inside
    assert not bool(((lin(step).abs() > half) & (lin(step12).abs() <= half)).any())

    # The pins on an interleaved arm order, against a layout built block by block.
    # The base columns are free: the sole rows hold the stance.
    dof = len(REAL_LEGS.joint_names)
    layout, is_left = _row_layout(REAL_LEGS, ("left", "right", "left", "right"), 12, dof, torch.device("cpu"),
                                  torch.float32)
    expected = torch.ones(12, dof)
    expected[0:3, REAL_LEGS.right_arm] = 0
    expected[3:6, REAL_LEGS.left_arm] = 0
    expected[6:9, REAL_LEGS.right_arm] = 0
    expected[9:12, REAL_LEGS.left_arm] = 0
    assert torch.equal(layout, expected)
    assert torch.equal(is_left, torch.tensor([True, False, True, False]).repeat_interleave(3))
    with pytest.raises(ValueError):  # rows that don't split into equal dep blocks
        _row_layout(REAL_LEGS, ("left", "right"), 5, dof, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError):  # an arm that is neither left nor right
        _row_layout(REAL_LEGS, ("left", "up"), 4, dof, torch.device("cpu"), torch.float32)


def _rectangle_region(half, rows):
    """The keyframe rectangle as a region: four planes about a centre at 0."""
    normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]).expand(rows, 4, 2)
    bounds = torch.tensor([half[0], half[0], half[1], half[1]]).expand(rows, 4)
    return normals, bounds, torch.zeros(rows, 2)


def _on_region(args, region=None):
    """``_balanced_step`` on ``_synthetic``'s argument tuple with its rectangle handed over as a region."""
    region = _rectangle_region(args[8], len(args[0])) if region is None else region
    return _balanced_step(*args[:8], None, *args[9:], upright=UPRIGHT, region=region)


def _violation(region, com):
    normals, bounds, _ = region
    return ((normals * com.unsqueeze(-2)).sum(-1) - bounds).amax(-1)


@pytest.mark.parametrize("coll", [False, True])
def test_a_rectangles_region_matches_the_rectangle_on_levels_1_and_2(coll):
    for seed in range(10):
        stack, _ = _synthetic(seed, coll=coll)
        want = _balanced_step(*stack, upright=UPRIGHT)[2]
        torch.testing.assert_close(_on_region(stack)[2], want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("coll", [False, True])
def test_region_step_stack_matches_per_dep(coll):
    for seed in range(3):
        stack, blocks = _synthetic(seed, coll=coll)
        steps, per_dep = _on_region(stack), [_on_region(b) for b in blocks]
        for k in range(3):
            assert torch.equal(steps[k], torch.cat([s[k] for s in per_dep]))


def test_region_level3_pulls_toward_the_centre_and_crosses_no_plane():
    stack, _ = _synthetic(0, coll=True)
    jac_com, com = stack[3], stack[4]
    region = _rectangle_region(stack[8], len(com))
    _, step3, step12 = _on_region(stack, region)

    def after(s):
        return com + (jac_com @ s.unsqueeze(-1)).squeeze(-1)

    kept_inside = _violation(region, after(step12)) <= 0
    assert bool((_violation(region, after(step3))[kept_inside] <= 1e-6).all())
    moved = (step3 != step12).any(-1)
    assert bool(moved.any())
    assert float(after(step3)[moved].norm(dim=-1).median()) < float(after(step12)[moved].norm(dim=-1).median())


def test_a_corner_takes_two_balance_rows():
    g = torch.Generator().manual_seed(0)
    dof, B = len(REAL_LEGS.joint_names), 32
    half = REAL_LEGS.support.keyframe_half
    q, lower, upper = torch.zeros(B, dof), -3 * torch.ones(dof), 3 * torch.ones(dof)
    jac_com = 0.05 * torch.randn(B, 2, dof, generator=g)
    com = (torch.tensor(half) * 1.02).expand(B, 2)                # past +x and +y at once
    region = _rectangle_region(half, B)
    step12 = _balanced_step(q, torch.zeros(B, 6), 0.4 * torch.randn(B, 6, dof, generator=g), jac_com, com, lower,
                            upper, torch.ones(B, dof), None, 0.3 * torch.randn(B, 12, dof, generator=g),
                            torch.zeros(B, 12), region=region)[2]
    landed = com + (jac_com @ step12.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(landed, (torch.tensor(half) - GN_COM_MARGIN).expand(B, 2), rtol=0, atol=1e-4)


def _double(x):
    """``x``, or each item of a tuple ``x``, in float64 where it is a float tensor."""
    if isinstance(x, tuple):
        return tuple(_double(y) for y in x)
    return x.double() if torch.is_tensor(x) and x.is_floating_point() else x


def test_a_moved_region_gives_the_same_steps():
    # Level 1 reads the planes, and levels 3 and 4 the CoM, about the region's centre: a region and a CoM moved
    # together by a planar offset give the same three steps.
    # Float64: in float32 the offset rounds the bounds and the CoM by an ulp, which the solves amplify to about 1e-6.
    offset = torch.tensor([0.3, -0.2], dtype=torch.float64)
    for seed in range(3):
        stack = _double(_synthetic(seed, coll=True)[0])
        region = _double(_rectangle_region(stack[8], len(stack[0])))
        normals, bounds, _ = region
        moved = (normals, bounds + (normals * offset).sum(-1), offset.expand(len(bounds), 2))
        args = stack[:4] + (stack[4] + offset,) + stack[5:]
        for got, want in zip(_on_region(args, moved), _on_region(stack, region)):
            torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def test_planes_that_are_not_sides_change_nothing():
    stack, _ = _synthetic(1, coll=True)
    normals, bounds, centre = _rectangle_region(stack[8], len(stack[0]))
    spare = torch.nn.functional.normalize(torch.randn(len(bounds), 6, 2, generator=torch.Generator().manual_seed(2)), dim=-1)
    wide = (torch.cat([normals, spare], 1), torch.cat([bounds, torch.full((len(bounds), 6), float("inf"))], 1), centre)
    for got, want in zip(_on_region(stack, wide), _on_region(stack)):
        assert torch.equal(got, want)


def test_the_side_lean_row_reads_the_heading():
    # A stepped plan's side lean (step_costs.side_lean, the trunk's up axis along the left of the stance's heading) as
    # one row. Nothing above it here, so the row's own columns show: at heading 0 the roll alone moves and the forward
    # bow stays; a quarter turn on, the pitch does.
    from cutamp.gn_refresh import _null_space_step, _side_lean_row
    from cutamp.step_costs import side_lean

    dof, B = len(REAL_LEGS.joint_names), 3
    q = torch.zeros(B, dof)
    q[:, UPRIGHT[0]], q[:, UPRIGHT[1]] = 0.3, 0.2                                      # bowed forward and leaning
    heading = torch.tensor([0.0, math.pi / 2, 0.7])
    stance = torch.zeros(B, 2, 3)
    stance[..., 2] = heading[:, None]
    rows, target = _side_lean_row(q, UPRIGHT, torch.ones(B, dof), heading)
    d = _null_space_step(torch.zeros(B, 1, dof), torch.zeros(B, 6, dof), rows, target, own_complement=True).to(q.dtype)
    before, after = side_lean(q, stance, REAL_LEGS).abs(), side_lean(q + d, stance, REAL_LEGS).abs()
    assert bool((after < 0.2 * before).all())
    assert float(d[0, UPRIGHT[0]]) == 0 and float(d[0, UPRIGHT[1]]) < 0
    assert float(d[1, UPRIGHT[0]].abs()) > 5 * float(d[1, UPRIGHT[1]].abs())
    assert not d[:, [i for i in range(dof) if i not in UPRIGHT]].any()


def test_with_a_heading_the_side_lean_comes_before_the_centring():
    # A stepped plan's lower levels, in the order its plans are ranked by: level 3 the side lean, level 4 the CoM toward
    # the centre in what the side lean leaves, then level 5 the forward lean toward 0 in what both leave (``d4`` holds
    # levels 4 and 5). Levels 1-2 are as without the heading; no lower level moves the soles or the hand; levels 4-5
    # keep the side lean where level 3 put it; and the whole step crosses no plane that levels 1-2's leaves the CoM
    # inside.
    from unittest import mock

    from cutamp.gn_refresh import _null_space_step, _side_lean_row
    from cutamp.step_costs import forward_lean, side_lean

    stack, _ = _synthetic(0, coll=True)
    q, jac, jac_com, com, jac_sole = stack[0], stack[2], stack[3], stack[4], stack[9]
    region = _rectangle_region(stack[8], len(com))
    heading = (2 * torch.rand(len(q), generator=torch.Generator().manual_seed(1)) - 1) * 0.8
    stance = torch.zeros(len(q), 2, 3)
    stance[..., 2] = heading[:, None]
    step, step3, step12 = _balanced_step(*stack[:8], None, *stack[9:], upright=UPRIGHT, region=region, heading=heading)
    assert torch.equal(step12, _on_region(stack, region)[2])
    d3, d4 = step3 - step12, step - step3
    leaned = d3.norm(dim=-1) > 1e-3
    centred = leaned & (d4.norm(dim=-1) > 1e-3)
    assert int(leaned.sum()) >= 40 and int(centred.sum()) >= 40
    for rows in (jac_sole, jac):
        assert float((rows @ d3.unsqueeze(-1))[leaned].abs().max()) < 1e-6
        assert float((rows @ d4.unsqueeze(-1))[centred].abs().max()) < 1e-6
    lean12, lean3 = (side_lean(q + s, stance, REAL_LEGS).abs() for s in (step12, step3))
    assert bool((lean3[leaned] < lean12[leaned]).all())
    row = _side_lean_row(q + step12, UPRIGHT, torch.ones_like(q), heading)[0].to(q.dtype)
    assert float((row @ d4.unsqueeze(-1))[centred].abs().max()) < 1e-6

    def after(s):
        return com + (jac_com @ s.unsqueeze(-1)).squeeze(-1)

    assert float(after(step)[centred].norm(dim=-1).median()) < float(after(step3)[centred].norm(dim=-1).median())

    # Level 5 alone, the whole step against the same call with its step zeroed (its solve is the one with three rows
    # above it, the side lean's and the CoM's two): it lowers the forward lean wherever it moves a row, and moves
    # neither the soles, the hand, the CoM nor the side lean.
    def without_level_5(*args, extra=None, **kwargs):
        out = _null_space_step(*args, extra=extra, **kwargs)
        return out * 0 if extra is not None and extra.shape[1] == 3 else out

    with mock.patch("cutamp.gn_refresh._null_space_step", without_level_5):
        step4 = _balanced_step(*stack[:8], None, *stack[9:], upright=UPRIGHT, region=region, heading=heading)[0]
    d5 = step - step4
    moved = d5.norm(dim=-1) > 1e-4
    assert int(moved.sum()) >= 40
    bow4, bow5 = (forward_lean(q + s, stance, REAL_LEGS).abs() for s in (step4, step))
    assert bool((bow5[moved] < bow4[moved]).all())
    for rows in (jac_sole, jac, jac_com, row):
        assert float((rows @ d5.unsqueeze(-1))[moved].abs().max()) < 1e-6
    kept_inside = _violation(region, after(step12)) <= 0
    assert bool((_violation(region, after(step))[kept_inside] <= 1e-6).all())
    assert bool((_violation(region, after(step3))[kept_inside] <= 1e-6).all())
