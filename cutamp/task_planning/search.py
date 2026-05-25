# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re
import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Generator, List, Optional, Sequence

from cutamp.task_planning import Atom, GroundOperator, Operator, State

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Node:
    state: frozenset[Atom]
    parent: Optional["_Node"]
    operator: Optional[GroundOperator]
    depth: int

    def parameters(self) -> dict[str, set[str]]:
        """Returns all the parameters ever encountered"""
        param_type_to_literals = self.parent.parameters() if self.parent else defaultdict(set)

        # All the literals in the state
        for atom in self.state:
            param_types = [param.type for param in atom.fluent.parameters]
            literals = atom.values
            for param_type, literal in zip(param_types, literals):
                param_type_to_literals[param_type].add(literal)

        # All the literals in the operator
        if self.operator is not None:
            for param, values in zip(self.operator.operator.parameters, self.operator.values):
                param_type_to_literals[param.type].add(values)

        return param_type_to_literals

    def extract_solution(self) -> list[GroundOperator]:
        """Extract the solution path from this node."""
        solution = []
        node = self
        while node.parent is not None:
            assert node.operator is not None
            solution.append(node.operator)
            node = node.parent
        return list(reversed(solution))

def _extract_number(s: str) -> int:
    """Extract the trailing number from a string like 'q0', 'left_q5', 'grasp12'."""
    match = re.search(r'(\d+)$', s)
    return int(match.group(1)) if match else 0

def _extract_prefix(s: str, param_type: str) -> str:
    """Extract the prefix before the base name. E.g., 'left_q0' -> 'left_', 'q0' -> ''"""
    # Map param_type to base name
    base_names = {"conf": "q", "traj": "traj", "pose": "pose", "grasp": "grasp"}
    base = base_names.get(param_type, param_type)
    
    # Find where base name starts
    idx = s.find(base)
    return s[:idx] if idx > 0 else ""

def get_valid_ground_operators(
    node: _Node,
    operators: Sequence[Operator],
    verbose: bool = False,
    priority_fn: Optional["Callable[[GroundOperator], float]"] = None,
) -> list[GroundOperator]:
    """
    Get all valid ground operators by testing the operators, binding samples for the unspecified variables, and
    checking preconditions are satisfied.

    When ``priority_fn`` is provided, the returned ground operators are
    sorted ascending by priority (smaller = explored first by BFS). This
    biases BFS exploration without changing search correctness — all valid
    ground ops are still returned, just in a different order.
    """
    ground_ops = []
    state = node.state

    # Fluent names to their corresponding atoms
    fluent_to_atoms: dict[str, set[Atom]] = defaultdict(set)
    for atom in state:
        fluent_to_atoms[atom.name].add(atom)

    # For each operator, we try to ground it given the current state
    for operator in operators:
        # FIXME: this sampling is slow once we have lots of plan skeletons, so improve it in the future
        # Got to do it inside here so we reset for each operator
        param_type_to_literals = node.parameters()

        # Sample new parameter names, handling both prefixed (left_q0) and unprefixed (q0) formats.
        # Base names map parameter types to their string representations.
        base_names = {"conf": "q", "traj": "traj", "pose": "pose", "grasp": "grasp"}
        
        # Determine arm prefix
        op_name = operator.name
        arm_prefix = ""
        if op_name.startswith("Left"):
            arm_prefix = "left_"
        elif op_name.startswith("Right"):
            arm_prefix = "right_"
        
        def _sample_param_type(param_type: str) -> str:
            if param_type not in param_type_to_literals:
                # For dual-arm operators, use arm prefix for conf parameters.
                if arm_prefix and param_type == "conf":
                    return f"{arm_prefix}{base_names.get(param_type, param_type)}1"
                return f"{base_names.get(param_type, param_type)}1"
                
            literals = param_type_to_literals[param_type]
            base = base_names.get(param_type)
            if base is None:
                raise NotImplementedError(f"Unknown param type: {param_type}")
            
            # Group literals by prefix and find max number for each prefix
            prefix_to_max = defaultdict(lambda: -1)
            for lit in literals:
                prefix = _extract_prefix(lit, param_type)
                num = _extract_number(lit)
                prefix_to_max[prefix] = max(prefix_to_max[prefix], num)
            
            # For dual-arm operators with conf parameters, use the arm-specific prefix
            if arm_prefix and param_type == "conf":
                max_num = prefix_to_max.get(arm_prefix, -1)
                return f"{arm_prefix}{base}{max_num + 1}"
            
            # Find the prefix with the highest max number and increment it
            # This ensures we generate unique names across all prefixes
            best_prefix = ""
            best_max = -1
            for prefix, max_num in prefix_to_max.items():
                if max_num > best_max:
                    best_max = max_num
                    best_prefix = prefix
            
            return f"{best_prefix}{base}{best_max + 1}"

        def sample_param_type(param_type: str) -> str:
            new_sample = _sample_param_type(param_type)
            param_type_to_literals[param_type].add(new_sample)
            return new_sample

        # 1. Check all preconditions are satisfied, and map each parameter to it's associated values in the ground atom
        param_to_literals: dict[str, set[str]] = defaultdict(set)
        pre_check = True
        for pre_fluent in operator.preconditions:
            if pre_fluent.name not in fluent_to_atoms:
                pre_check = False
                break

            # Bind the operator parameters to their values in the ground atom
            atoms = fluent_to_atoms[pre_fluent.name]
            for atom in atoms:
                if len(atom.values) != len(pre_fluent.parameters):
                    raise RuntimeError(
                        f"Number of values in atom {atoms} does not match number of parameters in {pre_fluent}"
                    )
                for param, value in zip(pre_fluent.parameters, atom.values):
                    param_to_literals[param.name].add(value)

        if not pre_check:
            if verbose:
                _log.debug(f"Skipping {operator} because not all preconditions are satisfied.")
            continue

        if verbose:
            _log.debug(
                f"Preconditions all satisfied for {operator}, parameter to literal bindings: {param_to_literals}"
            )

        # 2. Since we're dealing with funny planning, some of the parameters might not be bound. We need to check and
        # sample placeholder values for them.
        for param in operator.parameters:
            if param.name not in param_to_literals:
                sampled_value = sample_param_type(param.type)
                # sampled_value = param.sample_name()
                param_to_literals[param.name].add(sampled_value)
                if verbose:
                    _log.debug(f"Parameter '{param}' not bound, sampled placeholder value {sampled_value}")

        if len(param_to_literals) != len(operator.parameters):
            raise RuntimeError(
                f"Number of parameter bindings {len(param_to_literals)} "
                f"does not match number of parameters in {operator}"
            )

        # 3. Generate all combinations of the parameter bindings, so we ground the operator
        param_names = list(param_to_literals.keys())
        combinations = list(itertools.product(*param_to_literals.values()))
        for combo in combinations:
            substitutions = dict(zip(param_names, combo))
            ground_op = operator.ground(substitutions)

            # Check preconditions are satisfied
            if not ground_op.preconditions.issubset(state):
                if verbose:
                    _log.debug(f"Skipping {ground_op} because not all preconditions are satisfied.")
            else:
                ground_ops.append(ground_op)
                if verbose:
                    _log.debug(f"Grounded operator {operator} with substitutions {substitutions} to get {ground_op}")

    if priority_fn is not None:
        # Stable sort: equal-priority groundings preserve declaration order.
        ground_ops.sort(key=priority_fn)
    return ground_ops


def breadth_first_search(
    initial_state: State,
    goal_state: State,
    operators: Sequence[Operator],
    continue_branch_after_goal: bool = False,
    explored_state_check: bool = True,
    verbose: bool = False,
    ground_op_priority_fn: Optional[Callable[[GroundOperator], float]] = None,
) -> Generator[List[GroundOperator], None, None]:
    """
    Performs a breadth-first search to find a solution to the given planning problem.

    Parameters
    ----------
    initial_state: State
    goal_state: State
    operators: List[Operator]
        The possible operators for the given search problem.
    continue_branch_after_goal: bool
        Whether to continue searching on a given branch after finding a solution.
    explored_state_check: bool
        Whether to check if a state has been explored before adding it to the frontier.
    verbose: bool
        Whether to log debugging information. We have a flag so the log doesn't become polluted.
    ground_op_priority_fn: Optional[Callable[[GroundOperator], float]]
        Optional priority function for biased BFS sibling ordering.
        When provided, ground operators returned at each expansion are
        sorted ascending by priority. Used by the arm-affinity feature
        to prefer same-side picks. Does not affect search correctness —
        all valid ground ops are still enumerated, just in priority order.

    Returns
    -------
    Generator[List[GroundOperator], None, None]
        A generator that yields plans which are lists of ground operators.
    """
    # Sanity check, make sure the initial and goal state contain atoms only
    for elem in initial_state:
        if not isinstance(elem, Atom):
            raise ValueError(f"Initial state must contain only atoms, got {elem.__class__.__name__} {elem}")
    for elem in goal_state:
        if not isinstance(elem, Atom):
            raise ValueError(f"Goal state must contain only atoms, got {elem.__class__.__name__} {elem}")

    initial_node = _Node(state=initial_state, parent=None, operator=None, depth=0)
    frontier: list[_Node] = [initial_node]
    explored: set[State] = {initial_node.state}

    while frontier:
        node = frontier.pop(0)
        if verbose:
            _log.debug(f"Exploring node: {node}")

        # Check if goal is satisfied
        if goal_state.issubset(node.state):
            plan = node.extract_solution()
            yield plan
            if not continue_branch_after_goal:
                continue  # stop exploring this branch as we already satisfied the goal

        # Get successor and add to frontier
        ground_ops = get_valid_ground_operators(
            node, operators, verbose=verbose, priority_fn=ground_op_priority_fn,
        )
        for ground_op in ground_ops:
            successor_state = ground_op.apply(node.state)

            # If the successor state hasn't been explored, add it to the frontier.
            # Also if successor is a goal state (we want different ways of getting to the goal)
            if not explored_state_check or successor_state not in explored or goal_state.issubset(successor_state):
                child = _Node(state=successor_state, parent=node, operator=ground_op, depth=node.depth + 1)
                frontier.append(child)
                explored.add(successor_state)
