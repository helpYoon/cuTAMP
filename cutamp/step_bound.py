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
  command alone: the other foot's next command from a legal landing may lie outside it, and its own bound moves that
  start to the nearest allowed stance.
* :meth:`StepBound.hard_bound` moves the start to the nearest allowed stance (a predicted or measured landing no bound
  ever acted on), cuts the request to ``STEP_TRAVEL[1]`` of travel from the true start, and keeps the longest allowed
  prefix of the straight line (256 samples in one call).
"""

import math
from functools import lru_cache

import numpy as np
import torch

from cutamp.sole_collision import rect_rect_sat, sole_half
from cutamp.stance import STEP_KIN_MARGIN, STEP_SOLE_GAP, STEP_TRAVEL, STEP_X_MAX, STEP_Y_RANGE, STEP_YAW_RANGE
from cutamp.step_set import RHO, StepSetGrid


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def travel(c0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``[N]``: the distance of the sole centres plus ``RHO`` times the yaw change, for stances ``[N, 3]``."""
    return (c[:, :2] - c0[:, :2]).norm(dim=-1) + RHO * wrap(c[:, 2] - c0[:, 2]).abs()


def limit_travel(c0: torch.Tensor, c: torch.Tensor, t_max) -> torch.Tensor:
    """``[N, 3]``: ``c`` scaled toward ``c0`` so that its travel is at most ``t_max``."""
    d = c - c0
    d = torch.cat([d[:, :2], wrap(d[:, 2:3])], -1)
    s = (t_max / travel(c0, c).clamp_min(1e-9)).clamp(max=1.0)
    return c0 + d * s.unsqueeze(-1)


def stance_coords(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(width, split)`` ``[N]`` of relative stances ``c`` ``[N, 3]`` in the stance frame (midway between the soles, facing
    their mean heading)."""
    h = c[:, 2] / 2
    cs, sn = torch.cos(h), torch.sin(h)
    return -sn * c[:, 0] + cs * c[:, 1], cs * c[:, 0] + sn * c[:, 1]


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
    def __init__(self, grid: StepSetGrid, device, dtype=torch.float32, line_samples: int = 256, near_samples: int = 64):
        from scipy import ndimage

        self.device, self.dtype = device, dtype
        xs, ys, ps = grid.xs, grid.ys, grid.ps
        self.vol = torch.tensor(grid.sdf, device=device, dtype=dtype)  # [nx, ny, nyaw]
        self.lo = torch.tensor([xs[0], ys[0], ps[0]], device=device, dtype=dtype)
        self.hi = torch.tensor([xs[-1], ys[-1], ps[-1]], device=device, dtype=dtype)
        self.step = torch.tensor([xs[1] - xs[0], ys[1] - ys[0], ps[1] - ps[0]], device=device, dtype=dtype)
        self.metric = torch.tensor([1.0, 1.0, grid.rho], device=device, dtype=dtype)
        self.margin = STEP_KIN_MARGIN
        self.t_max = STEP_TRAVEL[1]
        self.sign = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)  # the mirror
        self.t_line = torch.linspace(0.0, 1.0, line_samples + 1, device=device, dtype=dtype)[1:]
        self.t_near = torch.linspace(0.0, 1.0, near_samples + 1, device=device, dtype=dtype)[1:]
        # the nearest allowed grid point of every grid point, in the metric (a pull-back table for `nearest`)
        nx, ny, npsi = self.vol.shape
        ax = [
            torch.linspace(float(self.lo[i]), float(self.hi[i]), n, device=device, dtype=dtype)
            for i, n in enumerate((nx, ny, npsi))
        ]
        G = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        E = (
            (self.g(G, torch.ones(G.shape[0], dtype=torch.bool, device=device)) <= 0)
            .reshape(nx, ny, npsi)
            .cpu()
            .numpy()
        )
        _, ind = ndimage.distance_transform_edt(
            ~E, sampling=tuple(float(v) for v in (self.step * self.metric).cpu()), return_indices=True
        )
        self.near = torch.tensor(
            np.stack([ax[i].cpu().numpy()[ind[i]] for i in range(3)], -1), device=device, dtype=dtype
        )
        self.n_allowed_cells = int(E.sum())

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

    # ---- the bound --------------------------------------------------------------------------------------------------
    def _prefix(self, a, dlt, ts, swing_left):
        """``[N]``: the largest sampled t in [0, 1] such that g(a + t' dlt) <= 0 at every sample t' <= t (one batched call)."""
        N, S = a.shape[0], ts.shape[0]
        pts = a.unsqueeze(1) + ts.view(1, S, 1) * dlt.unsqueeze(1)
        sl = swing_left.unsqueeze(1).expand(N, S).reshape(-1)
        good = self.g(pts.reshape(-1, 3), sl).reshape(N, S) <= 0
        return good.to(a.dtype).cumprod(dim=1).sum(dim=1) / S

    @torch.no_grad()
    def nearest(self, c: torch.Tensor, swing_left: torch.Tensor) -> torch.Tensor:
        """``[N, 3]``: requests already allowed come back (yaw wrapped); the others move to the nearest allowed grid point
        (a table) and then back toward the request along that line as far as every sample stays allowed."""
        cl = torch.where(swing_left.unsqueeze(-1), c, self._mirror(c))
        cw = torch.cat([cl[:, :2], wrap(cl[:, 2:3])], -1)
        cc = torch.minimum(torch.maximum(cw, self.lo), self.hi)
        i = ((cc - self.lo) / self.step).round().long()
        p = self.near[i[:, 0], i[:, 1], i[:, 2]]
        dlt = cw - p
        t = self._prefix(p, dlt, self.t_near, torch.ones_like(swing_left))
        out = torch.where((t >= 1.0).unsqueeze(-1), cw, p + t.unsqueeze(-1) * dlt)
        return torch.where(swing_left.unsqueeze(-1), out, self._mirror(out))

    @torch.no_grad()
    def hard_bound(self, c0: torch.Tensor, c: torch.Tensor, swing_left: torch.Tensor) -> torch.Tensor:
        """``[N, 3]``: the bounded command. ``c0`` is the true start (predicted or measured), ``c`` the request, both
        for the foot ``swing_left`` names."""
        line0 = self.nearest(c0, swing_left)
        c = limit_travel(c0, c, self.t_max)  # travel from the TRUE start
        d = c - line0
        d = torch.cat([d[:, :2], wrap(d[:, 2:3])], -1)
        return line0 + self._prefix(line0, d, self.t_line, swing_left).unsqueeze(-1) * d


@lru_cache(maxsize=8)
def planner_step_bound(device, dtype=torch.float32):
    from cutamp.step_set import load_step_set

    return StepBound(load_step_set(), device, dtype)
