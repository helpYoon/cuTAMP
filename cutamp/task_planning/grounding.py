# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Operator grounding: the actions whose preconditions hold in a state, with fresh names for new variables."""

import re
import itertools
from collections import defaultdict
from dataclasses import dataclass
from time import monotonic
from typing import Optional, Sequence

from .base_structs import Atom, GroundOperator, Operator

# Maps parameter types to the base string used when sampling literal names (e.g. "q0", "left_q1").
_BASE_NAMES = {"conf": "q", "traj": "traj", "pose": "pose", "grasp": "grasp", "sole": "sole"}


@dataclass(frozen=True)
class _Node:
    state: frozenset[Atom]
    parent: Optional["_Node"]
    operator: Optional[GroundOperator]

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
    base = _BASE_NAMES.get(param_type, param_type)

    # Find where base name starts
    idx = s.find(base)
    return s[:idx] if idx > 0 else ""

def get_valid_ground_operators(
    node: _Node,
    operators: Sequence[Operator],
    *,
    deadline: float | None = None,
) -> list[GroundOperator]:
    """
    Get all valid ground operators by testing the operators, binding samples for the unspecified variables, and
    checking preconditions are satisfied.
    """
    ground_ops = []
    state = node.state

    # Fluent names to their corresponding atoms
    fluent_to_atoms: dict[str, set[Atom]] = defaultdict(set)
    for atom in state:
        fluent_to_atoms[atom.name].add(atom)

    # Walk the node chain once; each operator below gets a fresh copy so the placeholder
    # names sampled for one operator don't leak into the next.
    base_param_type_to_literals = node.parameters()

    # For each operator, we try to ground it given the current state
    for operator in operators:
        param_type_to_literals: dict[str, set[str]] = defaultdict(set)
        for param_type, literals in base_param_type_to_literals.items():
            param_type_to_literals[param_type] = set(literals)

        # Sample new parameter names, handling both prefixed (left_q0) and unprefixed (q0) formats.
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
                    return f"{arm_prefix}{_BASE_NAMES.get(param_type, param_type)}1"
                return f"{_BASE_NAMES.get(param_type, param_type)}1"

            literals = param_type_to_literals[param_type]
            base = _BASE_NAMES.get(param_type)
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
            continue

        # 2. Since we're dealing with funny planning, some of the parameters might not be bound. We need to check and
        # sample placeholder values for them.
        for param in operator.parameters:
            if param.name not in param_to_literals:
                sampled_value = sample_param_type(param.type)
                param_to_literals[param.name].add(sampled_value)

        if len(param_to_literals) != len(operator.parameters) + len(operator.context_defaults):
            raise RuntimeError(
                f"Number of parameter bindings {len(param_to_literals)} "
                f"does not match number of parameters in {operator}"
            )

        # 3. Generate all combinations of the parameter bindings, so we ground the operator
        param_names = list(param_to_literals.keys())
        for combo in itertools.product(*param_to_literals.values()):
            if deadline is not None and monotonic() >= deadline:
                return []
            substitutions = dict(zip(param_names, combo))
            ground_op = operator.ground(substitutions)

            # Check preconditions are satisfied
            if ground_op.preconditions.issubset(state):
                ground_ops.append(ground_op)

    return ground_ops

