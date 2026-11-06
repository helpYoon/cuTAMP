"""CPU tests for the balance-first GN core (cutamp/gn_refresh.py)."""
import math

import pytest
import roma
import torch

from cutamp.gn_refresh import _balanced_step, _rotmat_to_rotvec, _row_layout
from cutamp.robots.t1 import REAL_LEGS


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
        return (q, r, jac, jac_com, com, lower, upper, c, half, jac_sole, r_sole, None,
                (pen, jac_pen) if coll else None, coll)

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
    # no row's step (levels 1-2's, or with the collision row levels 1-3's) depends on which other deps share
    # its call.
    for seed in range(10):
        stack, blocks = _synthetic(seed, coll=coll)      # one [4B] stack + its 4 [B] blocks
        steps, per_dep = _balanced_step(*stack), [_balanced_step(*b) for b in blocks]
        for k in range(2):
            assert torch.equal(steps[k], torch.cat([s[k] for s in per_dep]))


@pytest.mark.parametrize("coll", [False, True])
def test_balanced_step_moves_no_pinned_column(coll):
    stack, _ = _synthetic(0, coll=coll)
    (step, step12), col_mask = _balanced_step(*stack), stack[7]
    assert torch.all(step[col_mask == 0] == 0)
    assert torch.any(step[col_mask == 1] != 0)  # not vacuous: the free columns move
    # Level 3 runs beside the collision row only, and changes none of levels 1-2's active sets: their step is
    # bitwise the one they take without it, and a row in contact takes it alone.
    assert torch.equal(step, step12) != coll
    if coll:
        assert torch.equal(step12, _balanced_step(*stack[:-1], False)[0])
        in_contact = stack[12][0][:, 0] > 0
        assert bool(in_contact.any()) and torch.equal(step[in_contact], step12[in_contact])
        # Under a tilt bound too, where a row whose full step ends past the bound takes levels 1-2's step.
        from cutamp.joint_limit_cost import trunk_tilt_cosine
        names = list(REAL_LEGS.joint_names)
        tilt = (names.index("base_j_pitch"), names.index("base_j_roll"), math.cos(1.0), math.cos(0.99))
        tilted = stack[:11] + (tilt,) + stack[12:]
        step, step12 = _balanced_step(*tilted)
        assert torch.equal(step12, _balanced_step(*tilted[:-1], False)[0])
        past = trunk_tilt_cosine(stack[0] + step, *tilt[:2]) < tilt[2]
        assert bool(past.any()) and torch.equal(step[past], step12[past])


def test_level3_never_carries_a_free_com_axis_out(monkeypatch):
    # A row takes levels 1-2's step where level 3's would carry a CoM axis that step leaves inside the rectangle
    # past its edge (linear, as levels 1-2's pushes are tested). Level 3 never pulls that far on these rows, so here
    # it pushes outward: without the veto, 65 of the 256 rows end outside.
    import cutamp.gn_refresh as gn
    real = gn._centring_step
    monkeypatch.setattr(gn, "_centring_step", lambda *a: -real(*a))
    stack, _ = _synthetic(0, coll=True)
    step, step12 = _balanced_step(*stack)
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


def test_tilt_row_is_the_tilt_cosine_and_its_gradient():
    # The tilt row: the Motion value's cosine and its analytic Jacobian, autograd's gradient on the base's pitch and
    # roll columns and zero elsewhere, masked by the column mask. Route A's envelope sets no tilt bound, so it has no
    # row; an envelope that bounds the tilt at 1.0 rad has one.
    import dataclasses
    from cutamp.gn_refresh import GN_TILT_MARGIN, _tilt_bound, _tilt_row
    from cutamp.joint_limit_cost import trunk_tilt_cosine

    assert _tilt_bound(REAL_LEGS) is None
    names = list(REAL_LEGS.joint_names)
    tilted = dataclasses.replace(REAL_LEGS, envelope=REAL_LEGS.envelope._replace(trunk_tilt_max=1.0))
    pitch, roll, cos_max, cos_target = _tilt_bound(tilted)
    assert (pitch, roll) == (names.index("base_j_pitch"), names.index("base_j_roll"))
    assert (cos_max, cos_target) == (math.cos(1.0), math.cos(1.0 - GN_TILT_MARGIN))
    g = torch.Generator().manual_seed(0)
    q = 2 * torch.rand(16, len(names), generator=g, dtype=torch.float64) - 1
    col_mask = torch.ones_like(q)
    col_mask[8:, roll] = 0.0
    c, jac = _tilt_row(q, pitch, roll, col_mask)
    qg = q.clone().requires_grad_(True)
    grad = torch.autograd.grad(trunk_tilt_cosine(qg, pitch, roll).sum(), qg)[0]
    assert c.shape == (16, 1) and jac.shape == (16, 1, len(names))
    torch.testing.assert_close(c[:, 0], trunk_tilt_cosine(q, pitch, roll), rtol=0, atol=0)
    torch.testing.assert_close(jac[:, 0], grad * col_mask, rtol=1e-12, atol=1e-12)
    assert bool(jac[:8, 0, roll].abs().gt(0).all()) and not bool(jac[8:, 0, roll].any())
