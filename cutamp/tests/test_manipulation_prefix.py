from cutamp.t1_domain import On, all_t1_fluents, all_t1_operators, get_initial_state
from cutamp.task_planning import Fluent, Operator, manipulation_prefix


def test_immediate_pick_choices_are_shared_and_bound_before_expanding_place(monkeypatch):
    initial = get_initial_state(("a", "b"), ("pad",))
    goal = frozenset((On.ground("a", "pad"), On.ground("b", "pad")))
    root = manipulation_prefix.ManipulationPrefix(initial, goal, all_t1_operators)
    grounded_from = []
    ground = manipulation_prefix.get_valid_ground_operators

    def trace(node, *args, **kwargs):
        grounded_from.append(node.operator)
        return ground(node, *args, **kwargs)

    monkeypatch.setattr(manipulation_prefix, "get_valid_ground_operators", trace)
    picks = root.choices()
    assert root.action is None and root.skeleton() == []
    assert {(p.action.operator.metadata.arm, p.action.arg("obj")) for p in picks} == {
        ("left", "a"), ("right", "a"), ("left", "b"), ("right", "b"),
    }
    assert all(p.action.operator.metadata.action_type == "pick" and not p.loaded for p in picks)
    assert all(op is None or op.operator.metadata.is_motion for op in grounded_from)
    count = len(grounded_from)
    assert root.choices() is picks and len(grounded_from) == count

    left_a = next(p for p in picks if p.action.operator.metadata.arm == "left" and p.action.arg("obj") == "a")
    options = left_a.choices()
    place_a = next(p for p in options if p.action.operator.metadata.action_type == "place")
    right_b = next(p for p in options if p.action.operator.metadata.action_type == "pick")
    assert right_b.action.arg("obj") == "b" and right_b.loaded
    assert place_a.loaded and not place_a.complete
    assert [op.operator.name for op in place_a.skeleton()] == [
        "LeftMoveFree", "LeftPick", "LeftRetractHolding", "LeftMoveHolding", "LeftPlace",
    ]
    pick_b = next(p for p in place_a.choices() if p.action.operator.metadata.arm == "right")
    (place_b,) = pick_b.choices()
    assert place_b.complete and place_b.choices() == ()
    state = initial
    for op in place_b.skeleton():
        state = op.apply(state)
    assert goal <= state


def test_holding_rest_and_already_satisfied_goals_stop_at_first_satisfying_state():
    facts = {f.name: f for f in all_t1_fluents}
    initial = get_initial_state(("a",), ("pad",))
    holding = facts["RightHolding"].ground("a")
    pick_goal = manipulation_prefix.ManipulationPrefix(initial, {holding}, all_t1_operators)
    right_pick = next(p for p in pick_goal.choices() if p.action.operator.metadata.arm == "right")
    assert right_pick.complete and right_pick.choices() == ()
    rest_goal = {holding, facts["RightCanMove"].ground()}
    root = manipulation_prefix.ManipulationPrefix(initial, rest_goal, all_t1_operators)
    pick = next(p for p in root.choices() if p.action.operator.metadata.arm == "right")
    (rest,) = pick.choices()
    assert rest.complete and rest.action is None and rest.loaded
    assert rest.skeleton()[-1].operator.name == "RightRetractHolding"
    done = manipulation_prefix.ManipulationPrefix(initial, initial, all_t1_operators)
    assert done.complete and done.choices() == ()


def test_deadline_and_cycle_before_first_manipulation():
    a, b, goal = Fluent("A"), Fluent("B"), Fluent("Goal")
    ops = [Operator("toB", [], [a], [b], [a]), Operator("toA", [], [b], [a], [b])]
    root = manipulation_prefix.ManipulationPrefix(frozenset((a.ground(),)), {goal.ground()}, ops)
    assert root.choices() == ()
    expired = manipulation_prefix.ManipulationPrefix(
        get_initial_state(("a",), ("pad",)), {On.ground("a", "pad")}, all_t1_operators, deadline=0.0
    )
    assert expired.choices() == ()
