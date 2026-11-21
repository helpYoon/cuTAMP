"""Rank grounded symbolic step schedules from nominal grasp-free search."""

from itertools import product

import pytest

from cutamp import t1_domain as domain
from cutamp.reachability import ORDERS, ReachMap, apply_command
from cutamp.step_search import StepSearch, TargetRegion
from cutamp.task_planning import task_plan_generator
from cutamp.task_planning.step_schedules import ranked_step_skeletons
from cutamp.tests.test_step_search import passing

HOME = ((0.0, 0.10, 0.0), (0.0, -0.10, 0.0))
MENU = ((0.12, 0.20, 0.0), (0.18, 0.20, 0.0))


def scene():
    initial = domain.get_initial_state(["block"], ["table"])
    goal = frozenset({domain.On.ground("block", "table")})
    skeleton = next(task_plan_generator(initial, goal, domain.all_t1_operators))
    return skeleton, initial, goal


def test_unknown_regions_yield_ranked_25_unique_orders_and_keep_seed_paths(monkeypatch):
    import numpy as np

    skeleton, initial, goal = scene()
    search = StepSearch(None, commands=MENU)
    seen_targets = []
    original = search.schedules

    def record(soles, targets, **kwargs):
        seen_targets.append((targets, kwargs))
        return original(soles, targets, **kwargs)

    monkeypatch.setattr(search, "schedules", record)
    ranked = list(ranked_step_skeletons(search, skeleton, [None, None], initial, goal, HOME))
    assert len(ranked) == len({p.orders for p in ranked}) == 25
    assert {p.orders for p in ranked} == set(product(ORDERS, repeat=2))
    assert [p.cost for p in ranked] == sorted(p.cost for p in ranked)
    assert [a.name for a in ranked[0].skeleton] == [a.name for a in skeleton]
    assert len(seen_targets) == 1  # no replay until another seed is requested
    targets, _ = seen_targets[0]
    assert [t.arm for t in targets] == [a.operator.metadata.arm for a in skeleton if a.operator.metadata.is_actionable]
    assert [t.loaded for t in targets] == [False, True]
    assert all(t.center is None for t in targets)
    variant = next(p for p in ranked if p.orders == ("LR", "RL"))
    arms = [a for a in variant.skeleton if a.operator.metadata.is_actionable]
    assert arms[0].arg("obj") == arms[1].arg("obj") == "block"
    assert arms[0].arg("grasp") == arms[1].arg("grasp")
    assert arms[1].arg("surface") == "table"
    step_ops = [a for a in variant.skeleton if a.operator.metadata.action_type == "step"]
    expected_names = [a.arg("sole_to") for a in step_ops]
    observed = set()
    for seed in [variant.seed, *variant.alternative_seeds]:
        assert list(seed.commands) == expected_names
        signature = tuple(seed.commands.values())
        assert signature not in observed
        observed.add(signature)
        feet = HOME
        flattened = []
        for edge in seed.schedule.edges:
            for foot, command, after in zip(edge.order, edge.commands, edge.steps):
                feet = apply_command(feet, foot, command)
                np.testing.assert_allclose(feet, after)
                flattened.append(command)
            np.testing.assert_allclose(feet, edge.soles)
        assert tuple(flattened) == signature
    assert len(observed) > 1
    assert len(seen_targets) == 2 and seen_targets[1][1] == {"orders": ("LR", "RL")}


def test_reach_evidence_changes_first_grounded_skeleton_before_any_grasp_or_ik(monkeypatch):
    def no_sampling_or_ik(*args, **kwargs):
        pytest.fail("ranking performed online grasp generation, particle initialization or IK")

    monkeypatch.setattr("cutamp.grasps.grasp_sets", no_sampling_or_ik)
    monkeypatch.setattr("cutamp.particle_initialization.ParticleInitializer.__call__", no_sampling_or_ik)
    monkeypatch.setattr("cutamp.gn_refresh.gn_project_balanced", no_sampling_or_ik)
    skeleton, initial, goal = scene()
    arm = next(a.operator.metadata.arm for a in skeleton if a.operator.metadata.action_type == "pick")
    atlas = ReachMap(0.01, {"home_soles": HOME, "sole_box_m": [0.20, 0.08]})
    atlas.add_stance(
        HOME, [{"arm": arm, "posture": "crouch", "points": [(0.005, 0.005, 0.505)], "q": [[42.0]], "margin": [0.06]}]
    )
    region = TargetRegion(arm, (0.065, 0.005, 0.505), (0.004, 0.004, 0.004))
    ranked = ranked_step_skeletons(
        StepSearch(atlas, accuracy=passing("L"), lag=8, commands=MENU[:1]), skeleton, [region, region], initial, goal, HOME
    )
    best = next(ranked)
    assert best.cost == (0, 1) and best.orders == ("L", "none")
    assert tuple(best.seed.commands.values()) == (MENU[0],)
    assert all(e.posture == "crouch" and e.witness == (42.0,) for e in best.seed.schedule.edges)


def test_target_count_and_acting_arm_must_match_manipulations():
    skeleton, initial, goal = scene()
    arm = next(a.operator.metadata.arm for a in skeleton if a.operator.metadata.action_type == "pick")
    other = "right" if arm == "left" else "left"
    search = StepSearch(None, commands=MENU)
    for regions in ([None], [None, None, None], [TargetRegion(other, None, None), None]):
        with pytest.raises(ValueError):
            next(ranked_step_skeletons(search, skeleton, regions, initial, goal, HOME))


def test_fallback_orders_unseeded_counts_and_excludes_proven_fixed_sole_collision():
    from cutamp.task_planning.step_schedules import unseeded_step_skeletons

    skeleton, initial, goal = scene()
    arm = next(a.operator.metadata.arm for a in skeleton if a.operator.metadata.action_type == "pick")
    search = StepSearch(None, commands=())  # no sampled landing commands
    empty_region = (None, None)
    ranked = list(ranked_step_skeletons(search, skeleton, empty_region, initial, goal, HOME))
    assert [p.orders for p in ranked] == [("none", "none")]
    retried = list(
        unseeded_step_skeletons(
            search, skeleton, empty_region, initial, goal, HOME, tried_orders=[p.orders for p in ranked]
        )
    )
    assert len(retried) == len({p.orders for p in retried}) == 24
    assert [p.step_count for p in retried] == sorted(p.step_count for p in retried)
    assert all(p.seed is None and p.cost is None and not list(p.alternative_seeds) for p in retried)
    cube_under_sole = TargetRegion(arm, (0.0, 0.1, 0.025), (0.05, 0.05, 0.05), solid_pick=True)
    blocked = (cube_under_sole, None)
    assert not list(ranked_step_skeletons(search, skeleton, blocked, initial, goal, HOME))
    possible = list(unseeded_step_skeletons(search, skeleton, blocked, initial, goal, HOME))
    assert len(possible) == 15
    assert all("L" in p.orders[0] and p.seed is None for p in possible)
