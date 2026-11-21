"""A* over object-bound arm choices and nominal landing sequences."""

from dataclasses import dataclass, replace
from functools import cache
from heapq import heappop, heappush
from itertools import count
from time import monotonic

import numpy as np

from cutamp.reachability import _rot, canonical_stance
from cutamp.step_search import Schedule, TargetRegion
from cutamp.task_planning.manipulation_prefix import ManipulationPrefix
from cutamp.task_planning.step_schedules import bind_schedule


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
                path = Schedule(node.edges, (node.unresolved, node.steps))
                proposal = bind_schedule(
                    search, skeleton, node.targets, world.initial_state, world.goal_state,
                    feet, path, {"prefer_reach": True, "deadline": deadline},
                )
                proposal.arm_approach_m = node.approach
                yield proposal
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
