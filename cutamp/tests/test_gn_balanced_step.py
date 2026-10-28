"""CPU tests for the balance-first GN core (cutamp/gn_refresh.py).

The refresh stacks every IK dep into one sync-free call, which rests on two
claims pinned here: the branch-free rotation log map is roma's, bit for bit,
and a dep-major stack steps every row exactly as its dep's own call would.
"""
import math

import roma
import torch

from cutamp.com_polygon_cost import COM_HALF_EXTENTS
from cutamp.gn_refresh import _balanced_step, _rotmat_to_rotvec, _row_layout
from cutamp.robots.t1 import BASE_INDICES


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


def test_balanced_step_moves_no_pinned_column():
    stack, _ = _synthetic(0)
    step, col_mask = _balanced_step(*stack), stack[-1]
    assert torch.all(step[col_mask == 0] == 0)
    assert torch.any(step[col_mask == 1] != 0)  # not vacuous: the free columns move
