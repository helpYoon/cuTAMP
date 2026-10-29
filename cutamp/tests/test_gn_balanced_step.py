"""CPU tests for the balance-first GN core (cutamp/gn_refresh.py).

The refresh stacks every IK dep into one sync-free call, which rests on
three claims pinned here: the branch-free rotation log map is roma's, bit
for bit; a dep-major stack steps every row exactly as its dep's own call
would; and each row's pins and frame follow its own dep's arm, checked on
the interleaved (left, right, left, right) order a real refresh uses.
"""
import math

import roma
import torch

from cutamp.com_polygon_cost import COM_HALF_EXTENTS
from cutamp.gn_refresh import GN_MARGIN_DAMPING, _balanced_step, _rotmat_to_rotvec, _row_layout, _spd_solve
from cutamp.robots.t1 import BASE_INDICES, LEFT_ARM_INDICES, RIGHT_ARM_INDICES


def _synthetic(seed, dof=21, B=64):
    """Inputs for one ``_balanced_step`` on four dep-major blocks of ``B`` rows
    (pick and place on each arm): ``(stack, blocks)``, the ``[4B]`` argument
    tuple and its four ``[B]`` slices. Some joints sit on a hard limit or in
    the margin band and some CoM rows are outside the support rectangle, so
    every active set fires."""
    g = torch.Generator().manual_seed(seed)
    lower = -(2 * torch.rand(dof, generator=g) + 0.2)
    upper = 2 * torch.rand(dof, generator=g) + 0.2
    margin = torch.full((dof,), 0.1)
    margin[BASE_INDICES] = 0.0
    shared = (lower, upper, lower + margin, upper - margin)
    half = torch.tensor(COM_HALF_EXTENTS)
    col_mask, _ = _row_layout(("left", "left", "right", "right"), B, dof, torch.device("cpu"), torch.float32)

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

    # That self-comparison can't see a mis-wired level-3 solve, the step's one
    # hand-wired kernel, so check it against the library's LU on systems built
    # like _margin_step's: m = sel * (projector onto the null space of 8 rows),
    # with about 60% of its rows zeroed, then m m^T + damping I.
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

    # The pins themselves, in a real blocks_t1 refresh's interleaved arm order
    # (left_q1, right_q1, left_q3, right_q3), against a layout built block by
    # block: the base and each block's inactive arm pinned, its own arm's frame.
    layout, is_left = _row_layout(("left", "right", "left", "right"), 3, 21, torch.device("cpu"), torch.float32)
    expected = torch.ones(12, 21)
    expected[:, BASE_INDICES] = 0
    expected[0:3, RIGHT_ARM_INDICES] = 0
    expected[3:6, LEFT_ARM_INDICES] = 0
    expected[6:9, RIGHT_ARM_INDICES] = 0
    expected[9:12, LEFT_ARM_INDICES] = 0
    assert torch.equal(layout, expected)
    assert torch.equal(is_left, torch.tensor([True, False, True, False]).repeat_interleave(3))
