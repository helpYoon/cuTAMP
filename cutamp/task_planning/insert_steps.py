"""Ground legal T1 step prefixes ahead of the approaches to picks and places."""

from cutamp.t1_domain import all_t1_operators, all_t1_step_operators
from cutamp.task_planning.base_structs import State
from cutamp.task_planning.search import _Node, get_valid_ground_operators
from cutamp.task_planning.tamp_structs import PlanSkeleton

GAP_CHOICES = {
    "L": ("LeftStep",),
    "R": ("RightStep",),
    "LR": ("LeftStep", "RightStep"),
    "RL": ("RightStep", "LeftStep"),
}


def with_steps(skeleton: PlanSkeleton, gaps: dict[int, str], initial: State, goal: State) -> PlanSkeleton:
    """Re-ground the chosen skeleton. Missing pick/place indices specify an empty gap."""
    actions = [i for i, op in enumerate(skeleton) if op.operator.metadata.action_type in ("pick", "place")]
    prefixes = {}
    for gap, order in gaps.items():
        if not 0 <= gap < len(actions) or order not in GAP_CHOICES:
            raise ValueError(f"Invalid step gap {gap}: {order!r}; expected a manipulation index and L, R, LR or RL")
        approach = actions[gap] - 1
        meta = skeleton[approach].operator.metadata if approach >= 0 else None
        if meta is None or not meta.is_motion or meta.action_type is not None:
            raise ValueError(f"Manipulation {gap} must follow an approach")
        prefixes[approach] = GAP_CHOICES[order]

    operators = {op.name: op for op in (*all_t1_operators, *all_t1_step_operators)}
    wanted = []
    for i, ground in enumerate(skeleton):
        wanted.extend((name, {}) for name in prefixes.get(i, ()))
        bindings = {p.name: ground.arg(p.name) for p in ground.operator.parameters if p.type in ("movable", "surface")}
        wanted.append((ground.operator.name, bindings))

    node = _Node(initial, None, None)
    for name, bindings in wanted:
        candidates = [
            op
            for op in get_valid_ground_operators(node, [operators[name]])
            if all(op.arg(param) == value for param, value in bindings.items())
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one applicable grounding for {name} {bindings}, found {len(candidates)}")
        op = candidates[0]
        node = _Node(op.apply(node.state), node, op)
    if not goal.issubset(node.state):
        raise ValueError(f"Re-grounded skeleton misses goal facts {goal - node.state}")
    return node.extract_solution()
