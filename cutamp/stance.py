"""T1 sole command conventions and planar frame transforms, in meters and radians."""

import math

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
