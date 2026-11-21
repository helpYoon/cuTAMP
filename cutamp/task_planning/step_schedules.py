"""Attach A*-ranked footstep choices and nominal endpoint seeds to a standing T1 plan."""

from collections.abc import Iterator
from dataclasses import dataclass

from cutamp.step_search import Schedule
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
    seed: GroundedStepSeed
    alternative_seeds: Iterator[GroundedStepSeed]
    arm_approach_m: float | None = None

    @property
    def cost(self) -> tuple[int, int]:
        return self.seed.schedule.cost

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


def _ground(skeleton, orders, initial, goal):
    gaps = {i: order for i, order in enumerate(orders) if order != "none"}
    return with_steps(skeleton, gaps, initial, goal)


def bind_schedule(search, skeleton, targets, initial, goal, incoming, path, options):
    orders = tuple(edge.order for edge in path.edges)
    grounded = _ground(skeleton, orders, initial, goal)
    soles = [op.arg("sole_to") for op in grounded if op.operator.metadata.action_type == "step"]
    primary = _bound_seed(path, soles)
    alternatives = _alternatives(search, incoming, targets, orders, tuple(primary.commands.values()), soles, options)
    return RankedStepSkeleton(grounded, orders, primary, alternatives)
