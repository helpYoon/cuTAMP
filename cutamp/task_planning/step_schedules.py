"""Attach A*-ranked footstep choices and nominal endpoint seeds to a standing T1 plan."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from math import comb
from time import monotonic

from cutamp.reachability import ORDERS
from cutamp.step_search import Schedule, StepSearch, TargetRegion
from cutamp.task_planning.base_structs import State
from cutamp.task_planning.insert_steps import with_steps
from cutamp.task_planning.tamp_structs import PlanSkeleton


@dataclass
class GroundedStepSeed:
    """Signed, support-relative commands by sole_to binding. Edges retain world soles and reach witnesses."""

    schedule: Schedule
    commands: dict[str, tuple[float, float, float]]


@dataclass
class RankedStepSkeleton:
    skeleton: PlanSkeleton
    orders: tuple[str, ...]
    seed: GroundedStepSeed | None
    alternative_seeds: Iterator[GroundedStepSeed]

    @property
    def cost(self) -> tuple[int, int] | None:
        return self.seed.schedule.cost if self.seed is not None else None

    @property
    def step_count(self) -> int:
        return sum(len(order) for order in self.orders if order != "none")


def _bound_seed(schedule, soles):
    commands = (command for edge in schedule.edges for command in edge.commands)
    return GroundedStepSeed(schedule, dict(zip(soles, commands, strict=True)))


def _alternatives(search, incoming, targets, orders, primary_commands, soles, options):
    seen = {primary_commands}
    for path in search.schedules(incoming, targets, orders=orders, **options):
        seed = _bound_seed(path, soles)
        commands = tuple(seed.commands.values())
        if commands not in seen:
            seen.add(commands)
            yield seed


def _targets(skeleton, regions, initial):
    if len(regions) != sum(op.operator.metadata.action_type in ("pick", "place") for op in skeleton):
        raise ValueError("Expected one target region per pick or place")
    state = initial
    targets = []
    for op in skeleton:
        meta = op.operator.metadata
        if meta.action_type == "step":
            raise ValueError("Expected a standing skeleton before step scheduling")
        if meta.action_type in ("pick", "place"):
            target = regions[len(targets)]
            if target is None:
                target = TargetRegion(meta.arm, None, None)
            if target.arm != meta.arm:
                raise ValueError(f"Expected a {meta.arm}-arm region for {op.name}")
            if not target.loaded and any(fact.name in ("LeftHolding", "RightHolding") for fact in state):
                target = replace(target, loaded=True)
            targets.append(target)
        state = op.apply(state)
    return targets


def _ground(skeleton, orders, initial, goal):
    gaps = {i: order for i, order in enumerate(orders) if order != "none"}
    return with_steps(skeleton, gaps, initial, goal)


def ranked_step_skeletons(
    search: StepSearch,
    skeleton: PlanSkeleton,
    regions: Sequence[TargetRegion | None],
    initial: State,
    goal: State,
    incoming_soles,
    *,
    orders=None,
    deadline=None,
    step_count=None,
) -> Iterator[RankedStepSkeleton]:
    """Yield each sampled order schedule once, ranked by its best compatible endpoint path.

    Supply one world target region per pick/place, or None for unknown geometry.
    A command is relative to that step's incoming support foot. The edge
    retains the nominal outgoing world soles and the hand map's posture witness.
    """
    targets = _targets(skeleton, regions, initial)
    incoming = tuple(tuple(row) for row in incoming_soles)
    seen_orders = set()
    max_orders = len(ORDERS) ** len(targets) if orders is None else 1
    options = {} if deadline is None else {"deadline": deadline}
    if step_count is not None:
        options["step_count"] = step_count
        n = len(targets)
        if orders is None:
            # Choose the gaps containing two and one steps, then their first feet.
            max_orders = sum(
                comb(n, two) * comb(n - two, one) * 2 ** (two + one)
                for two in range(step_count // 2 + 1)
                if 0 <= (one := step_count - 2 * two) <= n - two
            )
    fixed = {} if orders is None else {"orders": orders}
    for path in search.schedules(incoming, targets, **fixed, **options):
        orders = tuple(edge.order for edge in path.edges)
        if orders in seen_orders:
            continue
        seen_orders.add(orders)
        grounded = _ground(skeleton, orders, initial, goal)
        soles = [op.arg("sole_to") for op in grounded if op.operator.metadata.action_type == "step"]
        primary = _bound_seed(path, soles)
        alternatives = _alternatives(
            search, incoming, targets, orders, tuple(primary.commands.values()), soles, options
        )
        yield RankedStepSkeleton(grounded, orders, primary, alternatives)
        if len(seen_orders) == max_orders:
            return


def unseeded_step_skeletons(
    search: StepSearch,
    skeleton: PlanSkeleton,
    regions: Sequence[TargetRegion | None],
    initial: State,
    goal: State,
    incoming_soles,
    *,
    tried_orders=(),
    fixed=None,
    max_steps=None,
    min_steps=0,
    deadline=None,
) -> Iterator[RankedStepSkeleton]:
    """Yield remaining symbolic schedules in increasing step count for refinement.

    A finite command menu can miss a viable target. These alternatives carry
    no sampled command or reach/accuracy score. Omit a prefix if an unmoved
    starting sole is known to overlap that manipulation's solid floor object.
    """
    targets = _targets(skeleton, regions, initial)
    blocked = [set(search.blocked_feet(incoming_soles, target)) for target in targets]
    tried = {tuple(orders) for orders in tried_orders}

    def paths(prefix, moved, remaining):
        if deadline is not None and monotonic() >= deadline:
            return
        if len(prefix) == len(targets):
            if remaining == 0 and prefix not in tried:
                yield prefix
            return
        capacity_after = 2 * (len(targets) - len(prefix) - 1)
        for order in ORDERS if fixed is None else (fixed[len(prefix)],):
            sides = set() if order == "none" else set(order)
            steps_left = remaining - len(sides)
            after = moved | sides
            if 0 <= steps_left <= capacity_after and blocked[len(prefix)] <= after:
                yield from paths((*prefix, order), after, steps_left)

    limit = 2 * len(targets)
    if fixed is not None:
        if len(fixed) != len(targets) or any(order not in ORDERS for order in fixed):
            raise ValueError("Supply one legal foot order per manipulation")
        counts = [sum(len(o) for o in fixed if o != "none")]
    else:
        counts = range(min_steps, min(limit, max_steps if max_steps is not None else limit) + 1)
    for count in counts:
        for orders in paths((), set(), count):
            yield RankedStepSkeleton(_ground(skeleton, orders, initial, goal), orders, None, iter(()))
