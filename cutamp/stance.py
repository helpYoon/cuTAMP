"""T1 sole command conventions and planar frame transforms, in meters and radians."""

import json
import math
from functools import lru_cache

import torch

STEP_X_MAX = 0.33  # m, either way
STEP_Y_RANGE = (0.14, 0.48)  # m
STEP_YAW_RANGE = (math.radians(-20.0), math.radians(50.0))  # rad, toes out positive
STEP_SOLE_GAP = 0.02  # m between the two sole rectangles
STEP_TRAVEL = (0.05, 0.60)  # m
STEP_KIN_MARGIN = 0.03  # m; measured: the kinematic grid's error (design 5.5)
STEP_LINE_POSES = 8  # the tested value
STEP_LIFT = 0.025  # m; placeholder: the study's tested height of the swung sole in the lifted set


def rel_pose(planted: torch.Tensor, sole: torch.Tensor) -> torch.Tensor:
    """``[N, 3]``: ``sole``'s floor pose (x, y, yaw) in ``planted``'s frame, yaw wrapped to [-pi, pi). Torch, differentiable."""
    d = sole[:, :2] - planted[:, :2]
    c, s = torch.cos(planted[:, 2]), torch.sin(planted[:, 2])
    yaw = sole[:, 2] - planted[:, 2]
    return torch.stack(
        [c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], torch.atan2(torch.sin(yaw), torch.cos(yaw))], -1
    )


def abs_pose(planted: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``[N, 3]``: the pose ``c`` (in ``planted``'s frame) in the world; the inverse of :func:`rel_pose` up to the yaw's
    wrap. Torch, differentiable."""
    cs, sn = torch.cos(planted[:, 2]), torch.sin(planted[:, 2])
    return torch.stack(
        [
            planted[:, 0] + cs * c[:, 0] - sn * c[:, 1],
            planted[:, 1] + sn * c[:, 0] + cs * c[:, 1],
            planted[:, 2] + c[:, 2],
        ],
        -1,
    )

@lru_cache(maxsize=8)
def _base_reach(device, dtype):
    from cutamp.robots.t1 import REAL_LEGS

    data = json.loads((REAL_LEGS.asset_root / "sole_base_reach.json").read_text())
    reach = torch.tensor(data["sole_base_xy"], device=device, dtype=dtype)
    return reach.mean(-1), (reach[..., 1] - reach[..., 0]) * 0.5, data["base_margin"]


def stance_joint_limits(soles, lower, upper):
    """Transport both leg reach boxes into world XY and intersect them for each stance."""
    center, radius, padding = _base_reach(soles.device, soles.dtype)
    yaw = soles[..., 2]
    c, s = yaw.cos(), yaw.sin()
    rot = torch.stack((c, -s, s, c), -1).reshape(*yaw.shape, 2, 2)
    xy = soles[..., :2] + (rot @ center.unsqueeze(-1)).squeeze(-1)
    extents = (rot.abs() @ radius.unsqueeze(-1)).squeeze(-1)
    lo = (1000 * ((xy - extents).amax(-2) - padding)).floor() / 1000
    hi = (1000 * ((xy + extents).amin(-2) + padding)).ceil() / 1000
    leading = soles.shape[:-2]
    return torch.cat((lo, lower[2:].expand(*leading, -1)), -1), torch.cat((hi, upper[2:].expand(*leading, -1)), -1)
