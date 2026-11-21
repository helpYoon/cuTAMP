"""Ground the symbolic step schedules that the nominal search's command menu missed."""

import pytest

from cutamp import t1_domain as domain
from cutamp.reachability import ReachMap
from cutamp.step_search import StepSearch, TargetRegion
from cutamp.task_planning import task_plan_generator
from cutamp.task_planning.step_schedules import unseeded_step_skeletons

HOME = ((0.0, 0.10, 0.0), (0.0, -0.10, 0.0))


def scene():
    initial = domain.get_initial_state(["block"], ["table"])
    goal = frozenset({domain.On.ground("block", "table")})
    skeleton = next(task_plan_generator(initial, goal, domain.all_t1_operators))
    return skeleton, initial, goal


def test_target_count_and_acting_arm_must_match_manipulations():
    skeleton, initial, goal = scene()
    arm = next(a.operator.metadata.arm for a in skeleton if a.operator.metadata.action_type == "pick")
    other = "right" if arm == "left" else "left"
    search = StepSearch(ReachMap(0.01, {"home_soles": HOME}), commands=())
    for regions in ([None], [None, None, None], [TargetRegion(other, None, None), None]):
        with pytest.raises(ValueError):
            next(unseeded_step_skeletons(search, skeleton, regions, initial, goal, HOME))


def test_fallback_orders_unseeded_counts_and_excludes_proven_fixed_sole_collision():
    skeleton, initial, goal = scene()
    arm = next(a.operator.metadata.arm for a in skeleton if a.operator.metadata.action_type == "pick")
    search = StepSearch(ReachMap(0.01, {"home_soles": HOME}), commands=())  # no sampled landing commands
    empty_region = (None, None)
    retried = list(
        unseeded_step_skeletons(search, skeleton, empty_region, initial, goal, HOME, tried_orders=[("none", "none")])
    )
    assert len(retried) == len({p.orders for p in retried}) == 24
    assert [p.step_count for p in retried] == sorted(p.step_count for p in retried)
    assert all(p.seed is None and p.cost is None and not list(p.alternative_seeds) for p in retried)
    cube_under_sole = TargetRegion(arm, (0.0, 0.1, 0.025), (0.05, 0.05, 0.05), solid_pick=True)
    blocked = (cube_under_sole, None)
    possible = list(unseeded_step_skeletons(search, skeleton, blocked, initial, goal, HOME))
    assert len(possible) == 15
    assert all("L" in p.orders[0] and p.seed is None for p in possible)
