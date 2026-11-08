# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict, Set, Optional

import torch


class CostReducer:
    """Reduces the cost dictionary to a single cost per particle by applying a weighted sum of costs."""

    def __init__(self, cost_config: Dict[str, Dict[str, float]]):
        self.cost_config = cost_config
        # Flatten the nested config for fast lookup
        self.cost_to_multiplier = {
            (cost_type, name): multiplier
            for cost_type, costs in cost_config.items()
            for name, multiplier in costs.items()
        }

    def _get_multiplier(self, cost_type: str, name: str) -> Optional[float]:
        direct = self.cost_to_multiplier.get((cost_type, name))
        if direct is not None:
            return direct
        # The longest "*suffix" key the name ends with, as in ConstraintChecker._get_tol
        suffixes = [
            n for (t, n) in self.cost_to_multiplier if t == cost_type and n.startswith("*") and name.endswith(n[1:])
        ]
        if suffixes:
            return self.cost_to_multiplier[(cost_type, max(suffixes, key=len))]
        # Mirror ConstraintChecker._get_tol's "default" fallback so
        # constraints with plan-skeleton-dependent inner names (e.g.
        # ComPolygon's per-conf entries left_q0, right_q3, ...) can
        # share a single weight without enumerating every possible name.
        # Returning None when neither is found is intentional — it means
        # "no multiplier, pass value through" (no implicit default-for-all).
        return self.cost_to_multiplier.get((cost_type, "default"))

    def get_cost(self, cost_dict: Dict[str, dict], consider_types: Set[str]) -> torch.Tensor:
        """Returns total cost per particle by taking weighted sum of considered cost types."""
        cost = None
        for cost_type, entry in cost_dict.items():
            if entry["type"] not in consider_types:
                continue

            for name, values in entry["values"].items():
                if values.ndim == 2:
                    values = values.sum(dim=1)  # Sum over time
                multiplier = self._get_multiplier(cost_type, name)
                if multiplier is not None:
                    values = values * multiplier
                cost = values if cost is None else cost + values
        return cost

    def soft_costs(self, cost_dict: Dict[str, dict]) -> torch.Tensor:
        """Reduce only the soft costs."""
        return self.get_cost(cost_dict, consider_types={"cost"})

    def __call__(
        self, cost_dict: Dict[str, dict], consider_types: Set[str] = frozenset(("constraint", "cost"))
    ) -> torch.Tensor:
        """Sum both soft and hard costs."""
        return self.get_cost(cost_dict, consider_types=consider_types)
