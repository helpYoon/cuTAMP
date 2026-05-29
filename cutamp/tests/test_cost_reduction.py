"""Tests for CostReducer multiplier lookup semantics."""
import torch

from cutamp.cost_reduction import CostReducer


def test_default_multiplier_fallback_used_when_name_missing():
    """When (cost_type, name) is missing but (cost_type, 'default') exists,
    the default multiplier applies. Mirrors ConstraintChecker._get_tol's
    existing fallback so plan-skeleton-dependent constraint names
    (left_q0, right_q3, ...) can share a single default weight."""
    cost_config = {"MyCostType": {"default": 7.0}}
    reducer = CostReducer(cost_config)
    cost_dict = {
        "MyCostType": {
            "type": "constraint",
            "constraints": [],
            "values": {
                "some_specific_name": torch.tensor([1.0, 2.0, 3.0]),
            },
        },
    }
    cost = reducer.get_cost(cost_dict, consider_types={"constraint"})
    # value × 7.0 = [7, 14, 21]
    assert torch.allclose(cost, torch.tensor([7.0, 14.0, 21.0])), (
        f"Expected default multiplier 7.0 to apply; got cost={cost}"
    )


def test_exact_name_multiplier_takes_precedence_over_default():
    """If both (cost_type, name) and (cost_type, 'default') exist, the
    exact match wins. Default is only a fallback."""
    cost_config = {"MyCostType": {"default": 7.0, "exact_name": 100.0}}
    reducer = CostReducer(cost_config)
    cost_dict = {
        "MyCostType": {
            "type": "constraint",
            "constraints": [],
            "values": {
                "exact_name": torch.tensor([1.0, 2.0]),
            },
        },
    }
    cost = reducer.get_cost(cost_dict, consider_types={"constraint"})
    # value × 100.0 = [100, 200]
    assert torch.allclose(cost, torch.tensor([100.0, 200.0])), (
        f"Exact-name multiplier 100.0 should win over default; got cost={cost}"
    )


def test_no_multiplier_when_neither_exact_nor_default_exists():
    """When neither (cost_type, name) nor (cost_type, 'default') is in
    the config, values pass through unmultiplied (multiplier == None
    path). This is the pre-existing behavior for constraints with no
    explicit mult (e.g. Collision)."""
    cost_config = {"OtherCostType": {"default": 99.0}}  # different type
    reducer = CostReducer(cost_config)
    cost_dict = {
        "MyCostType": {
            "type": "constraint",
            "constraints": [],
            "values": {
                "name": torch.tensor([1.0, 2.0, 3.0]),
            },
        },
    }
    cost = reducer.get_cost(cost_dict, consider_types={"constraint"})
    assert torch.allclose(cost, torch.tensor([1.0, 2.0, 3.0])), (
        f"Unmultiplied passthrough expected; got cost={cost}"
    )


def test_get_cost_zero_tensor_when_no_matching_type():
    # cost_dict has only a 'cost' entry; asking for 'constraint' must yield a
    # zero tensor shaped to the particle batch, not None (so the optimizer's
    # hard_costs(...) + soft_costs(...) never hits None + tensor).
    reducer = CostReducer({})
    cost_dict = {"traj": {"type": "cost", "values": {"traj_length": torch.ones(5)}}}
    out = reducer.get_cost(cost_dict, consider_types={"constraint"})
    assert out is not None
    assert out.shape == (5,)
    assert torch.equal(out, torch.zeros(5))


def test_get_cost_normal_path_unchanged():
    reducer = CostReducer({"soft": {"traj_length": 2.0}})
    cost_dict = {"soft": {"type": "cost", "values": {"traj_length": torch.ones(5)}}}
    out = reducer.get_cost(cost_dict, consider_types={"cost"})
    assert torch.equal(out, torch.full((5,), 2.0))
