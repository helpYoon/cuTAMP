"""Grasp geometry for the motion solver: each pick and place approaches along the grasp's axis and leaves the same way,
or lifts and comes in from above (design 7.8).

``world_from_ee`` is the hand's tool frame (``*_base_link``, ``robots/t1.py``), whose +x is GraspGenX's approach axis
(G's +z). The knobs are search settings, not robot limits (design section 12).
"""

import math
from typing import Literal, Tuple

import torch

APPROACH_DISTANCE = 0.05
"""m: how far back along the hand's +x a pick's or a place's straight insert starts, and a back-out ends: upstream
cuTAMP's Franka pre-grasp distance."""

LIFT_HEIGHT = 0.05
"""m: the straight lift after a pick whose approach is not vertical, and the height a non-vertical place comes in
from."""

VERTICAL_CONE = math.pi / 4
"""An approach within this angle of straight down backs out along its own axis, which rises at least as fast as it
moves sideways. Any other approach lifts or comes in from above."""

LINE_TOL = 0.005
"""m: the most a straight segment's palm may stray from its line: the IK's own position tolerance
(``scripts/utils.py``'s ``pos_err``)."""

LINE_BACKTRACK = 0.001
"""m: the most a straight segment's progress along its line may drop."""

LINE_END_TOL = 0.005
"""m: the most a straight segment's last frame may end from the line's end."""


def is_vertical(world_from_ee: torch.Tensor) -> bool:
    """Whether the hand's +x (the approach) of ``world_from_ee`` ``[4, 4]`` is within ``VERTICAL_CONE`` of world -z."""
    return bool(-world_from_ee[2, 0] >= math.cos(VERTICAL_CONE))


def approach_targets(world_from_ee: torch.Tensor, kind: Literal["pick", "place"]) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(pre, post)``: the hand poses ``[4, 4]`` where a pick's or a place's straight insert to ``world_from_ee``
    starts, and where its leave ends.

    ``back`` is ``world_from_ee`` moved ``APPROACH_DISTANCE`` back along its own +x, and ``up`` is it moved
    ``LIFT_HEIGHT`` up world z, each with its rotation. A pick comes in from ``back`` and leaves to ``back`` when its
    approach is vertical (``is_vertical``), else lifts to ``up``. A place comes in from ``back`` when vertical, else
    from ``up``, and always backs out to ``back``."""
    back = world_from_ee.clone()
    back[:3, 3] -= APPROACH_DISTANCE * world_from_ee[:3, 0]
    up = world_from_ee.clone()
    up[2, 3] += LIFT_HEIGHT
    vertical = is_vertical(world_from_ee)
    if kind == "pick":
        pre, post = back, back if vertical else up
    elif kind == "place":
        pre, post = back if vertical else up, back
    else:
        raise ValueError(f"kind must be 'pick' or 'place', got {kind!r}")
    return pre.clone(), post.clone()
