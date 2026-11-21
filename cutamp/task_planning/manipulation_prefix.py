"""Incremental symbolic search between successive picks and places."""

from collections import deque
from collections.abc import Sequence
from time import monotonic

from .base_structs import GroundOperator, Operator, State
from .search import _Node, _search_key, get_valid_ground_operators


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
