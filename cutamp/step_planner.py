"""The inputs and outputs of step-aware particle refinement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cutamp.landing_predictor import LandingState


@dataclass
class StepInputs:
    """An optional observed landing state, with its controller history and dynamics context, to plan the steps from."""

    initial_state: LandingState | None = None

    def matches_accuracy_context(self, lag):
        """The deployed home table assumes nominal friction, offsets and a shared actuator lag."""
        if self.initial_state is None:
            return True
        context = torch.as_tensor(self.initial_state.dynamics)
        if context.ndim not in (1, 2) or context.shape[-1] != 38:
            raise ValueError("Expected 38 normalized dynamics channels")
        expected = torch.zeros_like(context)
        expected[..., 0] = 0.25 / 0.45
        expected[..., -5:] = (lag - 5) / 3.0
        return torch.allclose(context, expected, atol=1e-6, rtol=0)


@dataclass
class StepRow:
    index: int
    particle: dict
    soft_cost: float
    metrics: dict
    motion_plan: object | None = None
    evidence: Callable[[], dict] | None = None

    @property
    def max_center_distance_m(self) -> float:
        """Worst planned pick/place CoM distance to the physical support centroid, in meters."""
        if "com_center_distances_m" in self.metrics:
            return max(self.metrics["com_center_distances_m"])
        return max(gap["center_distance_m"] for gap in self.metrics["gaps"])

    @property
    def effort(self) -> float:
        return self.metrics["secondary_effort"]


@dataclass
class StepResult:
    """Rows that passed final same-particle checks, even if no requested motion connected."""

    rows: list[StepRow]
    endpoint_ok: torch.Tensor | None
    status: str
    reason: str | None = None
    diagnostics: dict = field(default_factory=dict)


def furthest_endpoint(endpoint_ok):
    """Largest number of consecutive manipulation endpoints reached on a single row."""
    if endpoint_ok is None:
        return None
    flags = torch.as_tensor(endpoint_ok)
    if flags.dtype != torch.bool or flags.ndim != 2:
        raise ValueError("Endpoint feedback must be a [particles, manipulations] Boolean mask")
    if flags.shape[0] == 0:
        return None
    return int(flags.long().cumprod(dim=1).sum(dim=1).amax())


def standing_endpoint_flags(masks, rollout, plan, costs):
    """Associate each fixed-stance hard check with the manipulation that owns it."""
    flags = masks["ComComfort"]["margin_shortfall_squared"].clone()
    for mask in masks["KinematicConstraint"].values():
        flags &= mask.reshape(*flags.shape, -1).all(-1)

    gap = -1
    conf_gaps, placement_gaps = {}, {}
    for op in plan:
        kind = op.operator.metadata.action_type
        if kind in ("pick", "place"):
            gap += 1
            conf_gaps[op.arg("q")] = gap
            if kind == "place":
                placement_gaps[op.arg("placement")] = gap
        elif kind == "retract":
            conf_gaps[op.arg("q_retract")] = gap

    for conf, mask in masks.get("ComPolygon", {}).items():
        flags[:, conf_gaps.get(conf, 0)] &= mask  # q0 belongs to the initial gap
    motion = masks["Motion"]
    flags &= motion["joint_limit"] & motion["self_collision"]
    collision = masks["Collision"]
    object_pose_times = list(rollout["ts_to_pose_ts"].values())
    retract_checks = set()
    for t, (conf, _, _) in enumerate(costs.retract_info):
        column = conf_gaps[conf]
        flags[:, column] &= motion["retract_joint_limit"][:, t] & motion["retract_self_collision"][:, t]
        for key in (f"{conf}_world", f"{conf}_movables"):
            flags[:, column] &= collision[key].flatten(1).all(1)
            retract_checks.add(key)
    for key, value in collision.items():
        if key in ("robot_to_world", "robot_to_movables"):
            flags &= value
        elif key not in retract_checks:
            flags &= value[:, object_pose_times]
    placement = masks.get("StablePlacement", {})
    for surface, group in costs.surface_to_placements.items():
        supported = placement[f"{surface}_support"] & placement[f"{surface}_in_xy"]
        for t, (_, name) in enumerate(group):
            flags[:, placement_gaps[name]] &= supported[:, t]
    return flags
