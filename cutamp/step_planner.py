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

    @property
    def order(self) -> tuple[float, float, float]:
        """A stepped plan's place among valid plans, smaller first: its score (:func:`cutamp.step_costs.plan_score`),
        then its rank, then its worst joint effort (2026-11-25)."""
        return self.metrics["score"], self.metrics["rank"], self.metrics["joint_effort"]


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


