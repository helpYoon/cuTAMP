"""CPU tests for the balance-first GN core (cutamp/gn_refresh.py)."""
import math

import pytest
import roma
import torch

from cutamp.com_polygon_cost import COM_HALF_EXTENTS
from cutamp.gn_refresh import GN_MARGIN_DAMPING, _balanced_step, _rotmat_to_rotvec, _row_layout, _spd_solve
from cutamp.robots.t1 import BASE_INDICES, LEFT_ARM_INDICES, RIGHT_ARM_INDICES, SIMPLIFIED


def _synthetic(seed, dof=21, B=64):
    """``(stack, blocks)``: one ``_balanced_step`` argument tuple on four dep-major ``B``-row
    blocks and its four per-block slices. Some joints sit on a limit or in the margin band and some
    CoM rows are outside, so every active set fires."""
    g = torch.Generator().manual_seed(seed)
    lower = -(2 * torch.rand(dof, generator=g) + 0.2)
    upper = 2 * torch.rand(dof, generator=g) + 0.2
    margin = torch.full((dof,), 0.1)
    margin[BASE_INDICES] = 0.0
    shared = (lower, upper, lower + margin, upper - margin)
    half = torch.tensor(COM_HALF_EXTENTS)
    col_mask, _ = _row_layout(SIMPLIFIED, ("left", "left", "right", "right"), 4 * B, dof, torch.device("cpu"),
                              torch.float32)

    deps = []
    for c in col_mask.split(B):
        q = lower + torch.rand(B, dof, generator=g) * (upper - lower)
        on_limit = torch.rand(B, dof, generator=g) < 0.05
        q = torch.where(on_limit, torch.where(torch.rand(B, dof, generator=g) < 0.5, lower, upper), q)
        in_band = torch.rand(B, dof, generator=g) < 0.15
        q = torch.where(in_band, lower + 0.03 * torch.rand(B, dof, generator=g), q)
        jac = 0.4 * torch.randn(B, 6, dof, generator=g) * c.unsqueeze(1)
        jac_com = 0.05 * torch.randn(B, 2, dof, generator=g) * c.unsqueeze(1)
        com = (2 * torch.rand(B, 2, generator=g) - 1) * half * 1.08
        r = 0.01 * torch.randn(B, 6, generator=g)
        deps.append((q, r, jac, jac_com, com))

    stack = (*(torch.cat(x) for x in zip(*deps)), *shared, col_mask)
    blocks = [(*d, *shared, c) for d, c in zip(deps, col_mask.split(B))]
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


def test_balanced_step_stack_matches_per_dep():
    # Every op is row-wise and the active-set rounds always run in full, so
    # no row's step depends on which other deps share its call.
    for seed in range(10):
        stack, blocks = _synthetic(seed)                 # one [4B] stack + its 4 [B] blocks
        assert torch.equal(_balanced_step(*stack), torch.cat([_balanced_step(*b) for b in blocks]))

    # The stack-vs-blocks check can't see a mis-wired _spd_solve, the step's one hand-wired kernel: compare
    # it with LU on m m^T + damping I, m a row-masked null-space projector, as _margin_step builds it.
    g = torch.Generator().manual_seed(0)
    f64 = torch.float64
    basis = torch.linalg.qr(torch.randn(256, 21, 8, generator=g, dtype=f64))[0]
    sel = (torch.rand(256, 21, 1, generator=g) < 0.4).to(f64)
    m = sel * (torch.eye(21, dtype=f64) - basis @ basis.mT)
    M = m @ m.mT + GN_MARGIN_DAMPING * torch.eye(21, dtype=f64)
    b = torch.randn(256, 21, 1, generator=g, dtype=f64)
    assert torch.allclose(_spd_solve(M, b), torch.linalg.solve(M, b), rtol=1e-10, atol=1e-12)


def test_balanced_step_moves_no_pinned_column():
    stack, _ = _synthetic(0)
    step, col_mask = _balanced_step(*stack), stack[-1]
    assert torch.all(step[col_mask == 0] == 0)
    assert torch.any(step[col_mask == 1] != 0)  # not vacuous: the free columns move

    # The pins on an interleaved arm order, against a layout built block by block.
    layout, is_left = _row_layout(SIMPLIFIED, ("left", "right", "left", "right"), 12, 21, torch.device("cpu"),
                                  torch.float32)
    expected = torch.ones(12, 21)
    expected[:, BASE_INDICES] = 0
    expected[0:3, RIGHT_ARM_INDICES] = 0
    expected[3:6, LEFT_ARM_INDICES] = 0
    expected[6:9, RIGHT_ARM_INDICES] = 0
    expected[9:12, LEFT_ARM_INDICES] = 0
    assert torch.equal(layout, expected)
    assert torch.equal(is_left, torch.tensor([True, False, True, False]).repeat_interleave(3))
    with pytest.raises(ValueError):  # rows that don't split into equal dep blocks
        _row_layout(SIMPLIFIED, ("left", "right"), 5, 21, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError):  # an arm that is neither left nor right
        _row_layout(SIMPLIFIED, ("left", "up"), 4, 21, torch.device("cpu"), torch.float32)


def test_tilt_row_is_the_tilt_cosine_and_its_gradient():
    # The tilt row: the Motion value's cosine and its analytic Jacobian, autograd's gradient on the base's pitch and
    # roll columns and zero elsewhere, masked by the column mask. Route A's envelope sets no tilt bound, so it has no
    # row; an envelope that bounds the tilt at 1.0 rad has one.
    import dataclasses
    from cutamp.gn_refresh import GN_TILT_MARGIN, _tilt_bound, _tilt_row
    from cutamp.joint_limit_cost import trunk_tilt_cosine
    from cutamp.robots.t1 import REAL_LEGS

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
