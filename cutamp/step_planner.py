"""Per-skeleton search and the inputs and outputs of step-aware particle refinement."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING

import torch

from cutamp.reachability import ORDERS
from cutamp.step_search import TargetRegion
from cutamp.task_planning.step_schedules import ranked_step_skeletons, unseeded_step_skeletons

if TYPE_CHECKING:
    from cutamp.grasps import GraspSet
    from cutamp.landing_predictor import LandingState


@dataclass
class StepInputs:
    """Optional supplied task bindings and continuous seeds for a controlled planning run.

    Commands have one list per manipulation, with signed x/y/yaw in the incoming support
    frame. A supplied grasp pool binds through normal particle selection from pick to place.
    """

    grasps: dict[str, GraspSet] = field(default_factory=dict)
    orders: tuple[str, ...] | None = None
    arms: tuple[str, ...] | None = None
    commands: tuple | None = None
    initial_state: LandingState | None = None
    dynamics: torch.Tensor | None = None
    posture_seeds: tuple | None = None
    placements: dict | None = None
    transition_validator: Callable | None = None

    def __post_init__(self):
        if self.orders is not None:
            self.orders = tuple("none" if o == "" else o for o in self.orders)
            if any(order not in ORDERS for order in self.orders):
                raise ValueError("Use none, L, R, LR or RL at each manipulation")
        if self.arms is not None and any(arm not in ("left", "right") for arm in self.arms):
            raise ValueError("Acting arms must be left or right")
        if self.commands is not None:
            if self.orders is None or len(self.commands) != len(self.orders):
                raise ValueError("Supply one fixed order for each gap of initial commands")
            for order, cmds in zip(self.orders, self.commands):
                if len(cmds) != (0 if order == "none" else len(order)):
                    raise ValueError("Each fixed order needs one seed per swing")
                if any(len(c) != 3 or not all(math.isfinite(v) for v in c) for c in cmds):
                    raise ValueError("Seed commands require finite x, y, yaw")

    def matches_accuracy_context(self, lag):
        """The deployed home table assumes nominal friction, offsets and a shared actuator lag."""
        context = self.dynamics
        if context is None and self.initial_state is not None:
            context = self.initial_state.dynamics
        if context is None:
            return True
        context = torch.as_tensor(context)
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


@dataclass
class StepResult:
    """Rows that passed final same-particle checks, even if a requested motion remains unavailable."""

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


def manipulation_regions(skeleton, world):
    regions = []
    for op in skeleton:
        meta = op.operator.metadata
        if meta.action_type == "pick":
            regions.append(TargetRegion.pick(meta.arm, world.get_object(op.arg("obj"))))
        elif meta.action_type == "place":
            regions.append(
                TargetRegion.place(meta.arm, world.get_object(op.arg("obj")), world.get_object(op.arg("surface")))
            )
    return regions


class ScheduleSearch:
    """Rank and retry foot orders for one fixed arm/object skeleton."""

    def __init__(self, search, skeleton, regions, world, incoming, *, fixed=None, strategy="astar", deadline=None):
        self.args = (search, skeleton, regions, world.initial_state, world.goal_state, incoming)
        self.seen = set()
        self.targeted = deque()
        self.fixed = fixed
        self.deadline = deadline
        self.gaps = len(regions)
        self.count_first = (
            strategy == "count"
            or not search.hands
            or any(target is None or target.center is None for target in regions)
        )
        self.ranked = ranked_step_skeletons(*self.args, deadline=deadline)
        self.fallback = None
        self.cap = None

    def _exact(self, orders, fallback=None):
        proposal = next(ranked_step_skeletons(*self.args, orders=orders, deadline=self.deadline), None)
        if proposal is None and fallback is None:
            fallback = next(unseeded_step_skeletons(*self.args, fixed=orders), None)
        return proposal if proposal is not None else fallback

    def _counted(self, max_steps):
        # Actual count first; within that count A* prioritizes the best sampled evidence.
        # After the sampled paths, a legal order with no seed can still enter refinement.
        limit = 2 * self.gaps if max_steps is None else min(2 * self.gaps, max_steps)
        for count in range(limit + 1):
            yield from ranked_step_skeletons(*self.args, step_count=count, deadline=self.deadline)
            yield from unseeded_step_skeletons(
                *self.args, tried_orders=self.seen, min_steps=count, max_steps=count, deadline=self.deadline
            )

    def next(self, max_steps=None):
        if self.cap != max_steps:
            self.cap, self.fallback = max_steps, None
        while self.deadline is None or monotonic() < self.deadline:
            if self.fixed is not None:
                if self.seen:
                    return None
                proposal = self._exact(self.fixed)
            elif self.targeted:
                order = self.targeted.popleft()
                if order in self.seen or (max_steps is not None and _count(order) > max_steps):
                    continue
                proposal = self._exact(order)
                if proposal is None:
                    continue
            elif self.count_first or max_steps is not None:
                if self.fallback is None:
                    self.fallback = self._counted(max_steps)
                proposal = next(self.fallback, None)
            else:
                proposal = next(self.ranked, None)
                if proposal is None:
                    self.count_first = True
                    continue
            if proposal is None:
                return None
            if proposal.orders not in self.seen and (max_steps is None or proposal.step_count <= max_steps):
                self.seen.add(proposal.orders)
                return proposal
            if self.fixed is not None:
                return None
        return None

    def feedback(self, orders, endpoint_ok):
        if self.fixed is not None:
            return
        self.count_first = True
        prefix = furthest_endpoint(endpoint_ok)
        if prefix is None or prefix >= len(orders):
            return
        current = orders[prefix]
        extensions = ("L", "R") if current == "none" else ("LR", "RL") if len(current) == 1 else ()
        for new in extensions:
            self.targeted.append(orders[:prefix] + (new,) + orders[prefix + 1 :])


def _count(orders):
    return sum(len(o) for o in orders if o != "none")
