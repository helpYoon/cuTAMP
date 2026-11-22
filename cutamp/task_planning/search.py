"""A* over object-bound arm choices and nominal landing sequences. ``ManipulationPrefix`` grounds the moves from one
pick or place to the next; ``cutamp.step_search`` offers the foot landings before each."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from heapq import heappop, heappush
from itertools import count
from time import monotonic

import numpy as np

from cutamp.reachability import _rot, canonical_stance
from cutamp.step_search import Schedule, TargetRegion
from cutamp.task_planning.base_structs import GroundOperator, Operator, State
from cutamp.task_planning.grounding import _Node, get_valid_ground_operators
from cutamp.task_planning.insert_steps import with_steps
from cutamp.task_planning.tamp_structs import PlanSkeleton


def _search_key(state: State) -> State:
    """Ignore facts which only record a unique binding (for example the latest T1 rest posture)."""
    return frozenset(atom for atom in state if atom.fluent.tells_states_apart)


class ManipulationPrefix:
    """A symbolic prefix shared by the different foot landings for its last action."""

    def __init__(self, initial: State | _Node, goal: State, operators: Sequence[Operator], *, deadline=None):
        self._node = initial if isinstance(initial, _Node) else _Node(initial, None, None)
        self._goal = goal
        self._operators = operators
        self._deadline = deadline
        self._children = None
        op = self._node.operator
        self.action = op if op and op.operator.metadata.is_actionable else None
        self.complete = goal.issubset(self._node.state)
        incoming = self._node.parent.state if self.action else self._node.state
        self.loaded = any(fact.name in ("LeftHolding", "RightHolding") for fact in incoming)

    def skeleton(self) -> list[GroundOperator]:
        return self._node.extract_solution()

    def choices(self) -> tuple["ManipulationPrefix", ...]:
        """Ground intermediate motions up to the next task action or satisfied goal."""
        if self.complete or (self._deadline is not None and monotonic() >= self._deadline):
            return ()
        if self._children is None:
            children = []
            frontier = deque([self._node])
            seen = {_search_key(self._node.state)}
            while frontier:
                if self._deadline is not None and monotonic() >= self._deadline:
                    return ()
                node = frontier.popleft()
                ops = get_valid_ground_operators(node, self._operators, deadline=self._deadline)
                for op in sorted(ops, key=lambda op: op.name):
                    if self._deadline is not None and monotonic() >= self._deadline:
                        return ()
                    state = op.apply(node.state)
                    child = _Node(state, node, op)
                    if op.operator.metadata.is_actionable or self._goal.issubset(state):
                        children.append(ManipulationPrefix(child, self._goal, self._operators, deadline=self._deadline))
                    else:
                        key = _search_key(state)
                        if key not in seen:
                            seen.add(key)
                            frontier.append(child)
            self._children = tuple(children)
        return self._children


@dataclass
class RankedStepSkeleton:
    """A task plan with its steps inserted, the foot order (``none``, ``L``, ``R``, ``LR`` or ``RL``) before each pick
    and place, and the landing path the A* ranked it by. Every particle starts its landings at the path's commands
    (:func:`cutamp.step_particles._landing_starts`)."""

    skeleton: PlanSkeleton
    orders: tuple[str, ...]
    schedule: Schedule
    arm_approach_m: float | None = None

    @property
    def cost(self) -> tuple[int, int]:
        return self.schedule.cost

    @property
    def step_count(self) -> int:
        return sum(len(order) for order in self.orders if order != "none")


@dataclass(frozen=True, slots=True)
class _Branch:
    prefix: ManipulationPrefix
    feet: tuple
    edges: tuple = ()
    targets: tuple = ()
    pending: bool = False
    approach: float = 0.
    gap: float = 0.
    missing_reach: bool = False
    unresolved: int = 0
    steps: int = 0
    remaining: float = 0.

    @property
    def priority(self):
        return self.missing_reach, self.approach + self.gap + self.remaining, self.unresolved, self.steps


def arm_step_skeletons(search, world, operators, incoming, *, max_base_plans, admitted, deadline=None):
    """Score a bound pick before querying its step edges."""
    root = ManipulationPrefix(world.initial_state, world.goal_state, operators, deadline=deadline)
    feet = tuple(tuple(row) for row in incoming)
    _, initial_frame = canonical_stance(feet)
    palms = {}
    for arm, position in world.arm_home_ee_world.items():
        xyz = position.detach().cpu().numpy().astype(float)
        palms[arm] = ((xyz[:2] - initial_frame[:2]) @ _rot(-initial_frame[2]), xyz[2])

    # A known object origin can still guide the approach when bounds for a reach query are missing.
    @cache
    def pick_point(name):
        obj = world.get_object(name)
        point = TargetRegion.pick("left", obj).center
        if point is None and getattr(obj, "pose", None) is not None:
            point = tuple(obj.pose[:3])
        return point

    # Planar footsteps cannot reduce the vertical component of a future pick.
    unpicked = {a.values[0] for a in world.initial_state if a.name == "HasNotPickedUp"}
    required = {a.values[0] for a in world.goal_state - world.initial_state
                if a.name in ("On", "LeftHolding", "RightHolding")}
    vertical = {}
    for obj in sorted(unpicked & required):
        center = pick_point(obj)
        if center is not None:
            vertical[obj] = min(abs(z - center[2]) for _, z in palms.values())

    # Many different landing paths lead to the same symbolic choice.
    @cache
    def target_for(prefix):
        op = prefix.action
        meta = op.operator.metadata
        obj = world.get_object(op.arg("obj"))
        if meta.action_type == "pick":
            return TargetRegion.pick(meta.arm, obj, loaded=prefix.loaded)
        return TargetRegion.place(meta.arm, obj, world.get_object(op.arg("surface")))

    serial = count()
    queue = []

    def push(branch):
        heappush(queue, (branch.priority, next(serial), branch))

    push(_Branch(root, feet, remaining=sum(vertical.values())))
    while queue and (deadline is None or monotonic() < deadline):
        _, _, node = heappop(queue)
        if node.pending:
            for edge in search.edges(node.feet, node.targets[-1]):
                unresolved, steps = edge.cost
                push(replace(
                    node, feet=edge.soles, edges=(*node.edges, edge), pending=False,
                    gap=max(node.gap, edge.distance_m or 0.),
                    missing_reach=node.missing_reach or edge.distance_m is None,
                    unresolved=node.unresolved + unresolved, steps=node.steps + steps,
                ))
            continue

        if node.prefix.complete:
            skeleton = node.prefix.skeleton()
            identity = tuple(op.name for op in skeleton)
            if identity not in admitted and len(admitted) >= max_base_plans:
                continue
            orders = tuple(e.order for e in node.edges)
            if identity not in admitted:
                admitted[identity] = skeleton, set()
            _, seen = admitted[identity]
            if orders not in seen:
                seen.add(orders)
                gaps = {i: order for i, order in enumerate(orders) if order != "none"}
                stepped = with_steps(skeleton, gaps, world.initial_state, world.goal_state)
                yield RankedStepSkeleton(stepped, orders, Schedule(node.edges, (node.unresolved, node.steps)), node.approach)
            continue

        for child in node.prefix.choices():
            if child.action is None:
                push(replace(node, prefix=child))
                continue
            target = target_for(child)
            approach = node.approach
            remaining = node.remaining
            point = pick_point(child.action.arg("obj")) if child.action.operator.metadata.action_type == "pick" else None
            if point is not None:
                _, frame = canonical_stance(node.feet)
                local_palm, z = palms[target.arm]
                xy = frame[:2] + local_palm @ _rot(frame[2])
                approach += float(np.linalg.norm(np.r_[xy, z] - point))
                remaining = max(0., remaining - vertical.get(child.action.arg("obj"), 0.))
            push(replace(node, prefix=child, targets=(*node.targets, target), pending=True,
                         approach=approach, remaining=remaining))
