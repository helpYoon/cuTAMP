import pytest
import torch

from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.stance import UNIT_M
from cutamp.step_costs import GROWTH, WEIGHT_CAP, RequirementWeights, start_weight


def make():
    reducer = CostReducer({"ComPolygon": {"default": 10.0}, "ComComfort": {"margin_shortfall": 10.0}})
    checker = ConstraintChecker({"ComComfort": {"margin_shortfall": 0.0}, "Collision": {"default": 1e-3},
                                 "ComPolygon": {"default": 4e-4}, "KinematicConstraint": {"pos_err": 0.005},
                                 "Motion": {"self_collision": 0.0}})
    return reducer, RequirementWeights(reducer, checker)


def test_starting_weights():
    reducer, _ = make()
    assert start_weight("KinematicConstraint", "pos_err", reducer) == 0.0
    assert start_weight("ComComfort", "margin_shortfall", reducer) == 1 / UNIT_M
    assert start_weight("Motion", "self_collision", reducer) == 1 / UNIT_M
    assert start_weight("ComPolygon", "left_q1", reducer) == 10.0
    assert start_weight("Collision", "robot_to_world", reducer) == 1.0


def test_a_weight_doubles_only_while_its_row_is_violated():
    _, weights = make()
    short = torch.tensor([[0.0, 0.01], [0.0, 0.0]])
    cost = {"ComComfort": {"type": "constraint", "values": {"margin_shortfall": short}},
            "soft": {"type": "cost", "values": {"x": torch.zeros(2)}}}
    weights.apply(cost)
    start = 1 / UNIT_M
    grown = torch.tensor([[start, GROWTH * start], [start, start]])
    torch.testing.assert_close(cost["ComComfort"]["weights"]["margin_shortfall"], torch.full((2, 2), start))
    assert "weights" not in cost["soft"]
    weights.grow()
    short[0, 1] = 0.0                                   # met now: the weight stays where it is
    weights.apply(cost)
    torch.testing.assert_close(cost["ComComfort"]["weights"]["margin_shortfall"], grown)
    weights.grow()
    weights.apply(cost)
    torch.testing.assert_close(cost["ComComfort"]["weights"]["margin_shortfall"], grown)


def test_a_weight_stops_at_its_cap_and_a_held_row_weighs_nothing():
    _, weights = make()
    cost = {"Motion": {"type": "constraint", "values": {"self_collision": torch.tensor([[1e-3]])}},
            "KinematicConstraint": {"type": "constraint", "values": {"pos_err": torch.tensor([[0.02]])}}}
    for _ in range(40):
        weights.apply(cost)
        weights.grow()
    weights.apply(cost)
    assert cost["Motion"]["weights"]["self_collision"].item() == pytest.approx(WEIGHT_CAP / UNIT_M)
    assert cost["KinematicConstraint"]["weights"]["pos_err"].item() == 0.0


def test_inside_half_its_tolerance_a_row_is_not_violated():
    _, weights = make()
    cost = {"Collision": {"type": "constraint", "values": {"robot_to_world": torch.tensor([[4e-4, 6e-4]])}}}
    weights.apply(cost)
    weights.grow()
    weights.apply(cost)
    assert cost["Collision"]["weights"]["robot_to_world"].tolist() == [[1.0, GROWTH]]
