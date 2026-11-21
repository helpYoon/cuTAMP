"""The hard bound of a step: every landing the planner optimizes stays reachable by the legs (design 5.5-5.7).

Torch only, fixed loop counts, no host syncs: every method can be recorded in a CUDA graph; its constants are built
once, in ``__init__``, on the device. Ported from the study's ``step_bound.py`` and ``step_bound2.py``.

A command ``c = (x, y, yaw)`` is the pose of the swung sole in the planted sole's frame. The grid is for a left swing; a
right swing is evaluated on the mirror image ``(x, -y, -yaw)`` (the legs are left-right symmetric up to 0.014 rad in the
vendor's hip-yaw limits).

* :meth:`StepBound.g` is the test, allowed where it is at most 0: at least ``STEP_KIN_MARGIN`` inside the kinematic set
  (:mod:`cutamp.step_set`) for a swing of either foot, so the landed stance is one the legs can stand on and lift
  either foot from (the derived part); and the command inside the range the controller's landing command was trained
  on (:func:`trained_box`, the controller's reach), mirrored for a right swing. The trained range is tested on the
  command alone: the other foot's next command from a legal landing may lie outside it.
"""

import math
from functools import lru_cache

import torch

from cutamp.sole_collision import rect_rect_sat, sole_half
from cutamp.stance import STEP_KIN_MARGIN, STEP_SOLE_GAP, STEP_X_MAX, STEP_Y_RANGE, STEP_YAW_RANGE
from cutamp.step_set import RHO, StepSetGrid


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def travel(c0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``[N]``: the distance of the sole centres plus ``RHO`` times the yaw change, for stances ``[N, 3]``."""
    return (c[:, :2] - c0[:, :2]).norm(dim=-1) + RHO * wrap(c[:, 2] - c0[:, 2]).abs()


def sole_gap(c: torch.Tensor) -> torch.Tensor:
    """``[N]`` m: the separating-axis distance of the two sole rectangles
    (:func:`~cutamp.sole_collision.rect_rect_sat`), the planted one at the origin and the swung one at ``c`` ``[N, 3]``;
    negative where they overlap."""
    return rect_rect_sat(torch.zeros_like(c), sole_half(), c, sole_half())


def trained_box(c_left: torch.Tensor) -> torch.Tensor:
    """``[N]``: at most 0 inside the range the controller's landing command was trained on, for left-swing commands
    ``c_left`` ``[N, 3]`` (yaw wrapped first): the largest of the x, y and yaw faces' distances (yaw counted at ``RHO``
    per radian) and the sole gap's shortfall from ``STEP_SOLE_GAP``."""
    x, y, yaw = c_left[:, 0], c_left[:, 1], wrap(c_left[:, 2])
    return torch.stack(
        [
            x.abs() - STEP_X_MAX,
            STEP_Y_RANGE[0] - y,
            y - STEP_Y_RANGE[1],
            RHO * (STEP_YAW_RANGE[0] - yaw),
            RHO * (yaw - STEP_YAW_RANGE[1]),
            STEP_SOLE_GAP - sole_gap(c_left),
        ],
        -1,
    ).amax(-1)


class StepBound:
    def __init__(self, grid: StepSetGrid, device, dtype=torch.float32):
        self.device, self.dtype = device, dtype
        xs, ys, ps = grid.xs, grid.ys, grid.ps
        self.vol = torch.tensor(grid.sdf, device=device, dtype=dtype)  # [nx, ny, nyaw]
        self.lo = torch.tensor([xs[0], ys[0], ps[0]], device=device, dtype=dtype)
        self.hi = torch.tensor([xs[-1], ys[-1], ps[-1]], device=device, dtype=dtype)
        self.metric = torch.tensor([1.0, 1.0, grid.rho], device=device, dtype=dtype)
        self.margin = STEP_KIN_MARGIN
        self.sign = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)  # the mirror

    # ---- the test -------------------------------------------------------------------------------------------------
    def _mirror(self, c):
        return c * self.sign

    @staticmethod
    def _swap(c):
        cs, sn = torch.cos(c[..., 2]), torch.sin(c[..., 2])
        return torch.stack([-(cs * c[..., 0] + sn * c[..., 1]), -(-sn * c[..., 0] + cs * c[..., 1]), -c[..., 2]], -1)

    def _interp(self, cc):
        """Trilinear interpolation of the signed distance in one call (grid_sample; its last coordinate indexes x)."""
        g = 2.0 * (cc - self.lo) / (self.hi - self.lo) - 1.0
        return torch.nn.functional.grid_sample(
            self.vol[None, None],
            g.flip(-1).view(1, -1, 1, 1, 3),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(-1)

    def _kin(self, c_left):
        """The derived part of stances ``[N, 3]`` given as left-swing commands: the kinematic set's signed distance plus
        ``STEP_KIN_MARGIN``."""
        c = torch.cat([c_left[:, :2], wrap(c_left[:, 2:3])], -1)
        cc = torch.minimum(torch.maximum(c, self.lo), self.hi)
        kin = self._interp(cc) + ((c - cc) * self.metric).norm(dim=-1)  # outside the grid it keeps growing
        return kin + self.margin

    def g(self, c: torch.Tensor, swing_left: torch.Tensor) -> torch.Tensor:
        """``[N]``: allowed where at most 0. ``c`` ``[N, 3]`` is the command of the foot ``swing_left`` ``[N]`` names.
        The derived part tests the stance as a left swing and as the other foot's swing; the trained range tests the
        command alone (:func:`trained_box`, on its left-swing equivalent)."""
        cl = torch.where(swing_left.unsqueeze(-1), c, self._mirror(c))
        kin = torch.maximum(self._kin(cl), self._kin(self._mirror(self._swap(cl))))
        return torch.maximum(kin, trained_box(cl))


@lru_cache(maxsize=8)
def planner_step_bound(device, dtype=torch.float32):
    from cutamp.step_set import load_step_set

    return StepBound(load_step_set(), device, dtype)
