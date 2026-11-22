"""Grounded T1 manipulation gaps and backward-compatible arm bindings."""

from itertools import product

import pytest

from cutamp import t1_domain as domain
from cutamp.tests.conftest import skeleton_through


def scene(objects=("block",)):
    """Each object picked by its own hand (left, then right), then placed on the table in the same order."""
    initial = domain.get_initial_state(objects, ["table"])
    goal = frozenset(domain.On.ground(obj, "table") for obj in objects)
    hands = ("Left", "Right")[:len(objects)]
    skeleton = skeleton_through(initial, goal, *[(h + "Pick", o) for h, o in zip(hands, objects)],
                                *[(h + "Place", o) for h, o in zip(hands, objects)])
    return skeleton, initial, goal


def expand(skeleton, gaps, initial, goal):
    from cutamp.task_planning.insert_steps import with_steps

    return with_steps(skeleton, gaps, initial, goal)


@pytest.mark.parametrize("pick,place", product(("", "L", "R", "LR", "RL"), repeat=2))
def test_every_pick_and_place_order(pick, place):
    skeleton, initial, goal = scene()
    plan = expand(skeleton, {i: s for i, s in enumerate((pick, place)) if s}, initial, goal)
    state = initial
    soles = ["sole0", "sole1"]
    rest = "left_q0"
    seen = []
    pending = ""
    held = None
    for op in plan:
        meta = op.operator.metadata
        if meta.action_type == "step":
            side = 0 if meta.foot == "left" else 1
            pending += "LR"[side]
            assert op.arg("sole_from") == soles[side]
            assert op.arg("sole_planted") == soles[1 - side]
            assert op.arg("q_stand") == rest
            soles[side], rest = op.arg("sole_to"), op.arg("q_land")
            state = op.apply(state)
            for arm in ("Left", "Right"):
                assert [(a.values[0]) for a in state if a.name == arm + "At"] == [rest]
            if held:
                assert held in state
            continue
        assert [op.arg("sole_l"), op.arg("sole_r"), op.arg("q_stand")] == [*soles, rest]
        if meta.action_type is None:
            seen.append(pending)
            pending = ""
        if meta.action_type == "retract":
            rest = op.arg("q_retract")
        state = op.apply(state)
        if meta.action_type == "pick":
            held = next(a for a in state if a.name.endswith("HoldingWithGrasp"))
        if meta.action_type == "place":
            assert op.arg("grasp") == held.values[1]
    assert seen == [pick, place]
    assert goal.issubset(state)


@pytest.mark.parametrize("gaps", [{0: "LL"}, {0: "RR"}, {0: "LRL"}, {-1: "L"}, {2: "R"}])
def test_illegal_order_or_gap(gaps):
    plan, initial, goal = scene()
    with pytest.raises(ValueError):
        expand(plan, gaps, initial, goal)


def test_steps_obey_domain_preconditions_even_when_requested_directly():
    from cutamp.task_planning.grounding import _Node, get_valid_ground_operators

    _, initial, _ = scene()
    root = _Node(initial, None, None)
    first = next(
        op for op in get_valid_ground_operators(root, domain.all_t1_step_operators) if op.operator == domain.LeftStep
    )
    child = _Node(first.apply(initial), root, first)
    assert [op.operator.name for op in get_valid_ground_operators(child, domain.all_t1_step_operators)] == ["RightStep"]
    move = next(
        op for op in get_valid_ground_operators(child, domain.all_t1_operators) if op.operator == domain.LeftMoveFree
    )
    busy = _Node(move.apply(child.state), child, move)
    assert not get_valid_ground_operators(busy, domain.all_t1_step_operators)


def test_empty_schedule_preserves_multi_object_arms_bindings_and_names():
    skeleton, initial, goal = scene(("block_a", "block_b"))
    assert [op.name for op in expand(skeleton, {}, initial, goal)] == [op.name for op in skeleton]
    for op in skeleton:
        assert len(op.values) == len(op.operator.parameters)
        if op.operator.metadata.action_type == "pick":
            obj, grasp, conf = op.values
            assert (obj, grasp, conf) == tuple(op.arg(n) for n in ("obj", "grasp", "q"))
        assert op.arg("sole_l") == "sole0" and op.arg("sole_r") == "sole1"
    assert not any(op.metadata.action_type == "step" for op in domain.all_t1_operators)


def test_expansion_rejects_inapplicable_or_nongoal_sequence():
    skeleton, initial, goal = scene()
    without_pick = [op for op in skeleton if op.operator.metadata.action_type != "pick"]
    with pytest.raises(ValueError, match="ground"):
        expand(without_pick, {}, initial, goal)
    with pytest.raises(ValueError, match="goal"):
        expand(skeleton, {}, initial, frozenset({domain.On.ground("missing", "table")}))
