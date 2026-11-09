import math
from types import SimpleNamespace

import pytest
import torch

import cutamp.algorithm as alg
import cutamp.com_polygon_cost as cpc
from cutamp.algorithm import grasp_group_order
from cutamp.particle_initialization import (
    _BASE_JOINTS, ParticleInitializer, PickChoice, choice_stats, choose_grasps, next_place, pick_place_confs,
    worst_posture,
)


def _choose(rows, k):  # rows: (ok, score, tilt, travel, margin), one per grasp
    ok, *rest = (torch.tensor(c) for c in zip(*rows))
    out = choose_grasps(ok, *rest, k=k)
    return None if out is None else out.tolist()


def _op(kind, *values):
    return SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type=kind)), values=values)


def _fake_world(home=(0.0, 0.0, 0.7, 0.0, 0.0, 0.0)):  # the floating base alone
    return SimpleNamespace(model=SimpleNamespace(joint_names=_BASE_JOINTS, home=home))


def test_cutoff_beats_posture():
    # 0.7 itself clears; an upright 0.69 waits behind both tilted grasps that clear.
    rows = [(True, 0.69, 0.0, 0.0, 0.08), (True, 0.7, 1.0, 0.2, 0.01), (True, 0.9, 0.5, 0.2, 0.01)]
    assert _choose(rows, 3) == [2, 1, 0]


def test_below_cutoff_fills_by_score_then_posture():
    # One grasp clears; the two best scores below fill (not the more upright 0.4), ordered by tilt among themselves.
    rows = [(True, 0.8, 0.9, 0.0, 0.0), (True, 0.5, 0.3, 0.0, 0.0), (True, 0.6, 0.1, 0.0, 0.0),
            (True, 0.4, 0.0, 0.0, 0.0)]
    assert _choose(rows, 3) == [0, 2, 1]


def test_unbalanced_never_chosen():
    rows = [(False, 0.99, 0.0, 0.0, 0.087), (True, 0.71, 0.6, 0.2, 0.01)]
    assert _choose(rows, 10) == [1]  # fewer balanced than k: every balanced grasp


def test_nothing_balanced_is_none():
    assert _choose([(False, 0.9, 0.0, 0.0, 0.0)] * 3, 10) is None


def test_ties_break_by_travel_then_margin_then_index():
    rows = [(True, 0.8, 0.0, 0.05, 0.02), (True, 0.8, 0.0, 0.05, 0.08), (True, 0.8, 0.0, 0.02, 0.01),
            (True, 0.8, 0.0, 0.05, 0.08)]
    assert _choose(rows, 4) == [2, 1, 3, 0]
    assert _choose(rows, 2) == [2, 1]


def test_next_place_pairs_each_pick_with_its_own_place():
    skeleton = [_op("pick", "a", "g1", "q1"), _op("pick", "b", "g2", "q2"), _op("place", "b", "g2", "p2", "s", "q3"),
                _op("place", "a", "g1", "p1", "s", "q4"), _op("retract", "q4", "t", "r1"),
                _op("pick", "a", "g3", "q5"), _op("place", "a", "g3", "p3", "s", "q6"), _op("pick", "b", "g4", "q7")]
    assert [next_place(skeleton, i) for i in (0, 1, 5, 7)] == ["q4", "q3", "q6", None]
    assert pick_place_confs(skeleton) == ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]


def test_worst_posture_takes_the_worse_conf(monkeypatch):
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: q[:, 5])  # the yaw stands in for the margin
    pick = torch.tensor([[0.0, 0.0, 0.5, 0.3, 0.0, 0.08], [0.0, 0.0, 0.7, 0.0, 0.0, 0.02]])
    place = torch.tensor([[0.0, 0.0, 0.7, 0.0, 0.4, 0.05], [0.3, 0.4, 0.7, 0.0, 0.0, 0.06]])
    tilt, travel, margin = worst_posture(_fake_world(), [pick, place])
    torch.testing.assert_close(tilt, torch.tensor([0.4, 0.0]))
    torch.testing.assert_close(travel, torch.tensor([0.2, 0.5]))
    torch.testing.assert_close(margin, torch.tensor([0.05, 0.02]))


def test_choice_stats_describe_the_chosen():
    c = PickChoice("block3", "grasp1", torch.zeros(4, 7), torch.tensor([0.9, 0.8, 0.5, 0.95]),
                   torch.tensor([True, True, True, False]), torch.tensor([0.1, 0.2, 0.0, 0.0]),
                   torch.tensor([0.20, 0.25, 0.1, 0.0]), torch.tensor([0.08, 0.05, 0.08, 0.0]), torch.tensor([0, 1]))
    assert choice_stats(c) == {
        "object": "block3", "pool": 4, "balanced": 3, "balanced_at_cutoff": 2, "chosen": 2,
        "tilt_deg": pytest.approx([math.degrees(0.1), math.degrees(0.2)]),
        "travel": pytest.approx([0.20, 0.25]), "com_margin": pytest.approx([0.05, 0.08])}
    none = choice_stats(c._replace(chosen=None))
    assert none["chosen"] == 0 and none["tilt_deg"] is none["travel"] is none["com_margin"] is None


def test_init_judges_each_grasp_at_its_pick_and_its_place(monkeypatch):
    # __call__'s pairing on the CPU, the walks stubbed (D4): the pick accepts grasp 1, the most upright there, but its
    # place rejects it, so it is not balanced and not chosen. Posture is the worse of pick and place, so grasp 2 (tilt
    # 0.3) goes before grasp 0, whose place tilts 0.5. Judged at the pick alone, the choice would be [1, 0, 2].
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: q[:, 5])  # the yaw stands in for the margin
    conf = lambda x, pitch, margin: [x, 0.0, 0.7, pitch, 0.0, margin]
    pick = torch.tensor([conf(0.0, 0.1, 0.08), conf(0.0, 0.0, 0.08), conf(0.0, 0.3, 0.08)])
    place = torch.tensor([conf(0.2, 0.5, 0.08), conf(0.0, 0.0, 0.08), conf(0.0, 0.2, 0.03)])
    confs = {"q1": (pick, torch.tensor([True, True, True])), "q2": (place, torch.tensor([True, False, True]))}
    walks = []

    def walk(plan_skeleton, n, pools, chosen=None, pool_q=None, log_debug=None):
        walks.append((n, chosen))
        return {}, confs

    checker = SimpleNamespace(_get_tol=lambda constraint, name: 0.0)
    init = ParticleInitializer(_fake_world(), SimpleNamespace(num_particles=4, motion_plan_max_retries=10), checker)
    init._pick_pools = lambda plan_skeleton: {"g1": (torch.eye(7)[:3], torch.tensor([0.9, 0.8, 0.75]))}
    init._walk = walk
    init([_op("pick", "a", "g1", "q1"), _op("place", "a", "g1", "p1", "s", "q2")])
    (c,) = init.grasp_choice
    assert c.ok.tolist() == [True, False, True]
    assert c.chosen.tolist() == [2, 0]
    # The pool walk, one row per grasp, then the particle walk on the chosen grasps.
    assert [n for n, _ in walks] == [3, 4] and walks[1][1]["g1"].tolist() == [2, 0]


def test_groups_go_cutoff_first_then_upright():
    # Per row: its group's pool indices, soft cost, all at 0.7, tilt, travel, margin.
    rows = [((0, 0), 1.0, True, 0.3, 0.0, 0.0),     # group a, last by soft cost
            ((0, 0), 0.5, True, 0.1, 0.2, 0.08),    # group a, first
            ((1, 0), 2.0, True, 0.0, 0.2, 0.08),    # group b, first: the most upright
            ((2, 1), 0.1, False, 0.0, 0.0, 0.08),   # below the cutoff: after every group that clears it
            ((1, 0), 3.0, True, 0.9, 0.0, 0.0),     # group b, second (its own tilt doesn't count)
            ((3, 0), 0.2, True, 0.2, 0.1, 0.02),
            ((4, 0), 0.3, True, 0.2, 0.1, 0.08),    # a larger margin first
            ((5, 0), 0.4, True, 0.2, 0.05, 0.02),   # less travel first
            ((0, 2), 0.6, True, 0.2, 0.1, 0.02),    # all equal: pool order, (0, 1) < (0, 2) < (3, 0)
            ((0, 1), 0.7, True, 0.2, 0.1, 0.02),
            ((0, 0), 0.5, True, 0.1, 0.2, 0.08)]    # group a, second: row 1's soft cost, so after it by index
    idx, soft, hi, tilt, travel, margin = (torch.tensor(c) for c in zip(*rows))
    assert grasp_group_order(idx, soft, hi, tilt, travel, margin, 12) == [2, 1, 7, 6, 9, 8, 5, 3, 4, 10, 0]
    assert grasp_group_order(idx, soft, hi, tilt, travel, margin, 3) == [2, 1, 7]


def _ranking_inputs(pool_rows, pitch, mask, soft):
    """get_top_satisfying_particles' inputs on the CPU: one pick of a 3-grasp pool (scores 0.9, 0.8, 0.5), its
    place, and a retract tilted 1 rad that the ranking must ignore. Particle i's left_q1 carries i in its yaw."""
    n = len(pool_rows)
    q1 = torch.zeros(n, 6)
    q1[:, 3], q1[:, 5] = torch.tensor(pitch), torch.arange(n, dtype=torch.float)
    retract = torch.zeros(n, 6)
    retract[:, 3] = 1.0
    zeros = torch.zeros(3)
    choice = PickChoice("block", "grasp1", torch.eye(7)[:3], torch.tensor([0.9, 0.8, 0.5]),
                        torch.ones(3, dtype=torch.bool), zeros, zeros, zeros, torch.arange(3))
    rollout_fn = lambda particles: None
    rollout_fn.world = _fake_world(home=(0.0,) * 6)
    skeleton = [_op("pick", "block", "grasp1", "left_q1"), _op("place", "block", "grasp1", "pose1", "table", "left_q2"),
                _op("retract", "block", "grasp1", "left_q2", "traj", "left_q3")]
    particles = {"grasp1": torch.eye(7)[pool_rows], "left_q1": q1, "left_q2": torch.zeros(n, 6), "left_q3": retract}
    plan_info = {"plan_skeleton": skeleton, "particles": particles, "rollout_fn": rollout_fn,
                 "cost_fn": lambda rollout, particles: {}, "grasp_choice": [choice]}
    checker = SimpleNamespace(get_mask=lambda cost_dict, verbose=True: torch.tensor(mask))
    reducer = SimpleNamespace(soft_costs=lambda cost_dict: torch.tensor(soft))
    return plan_info, checker, reducer


def test_top_particles_go_round_the_grasps(monkeypatch):
    # Grasp 1's group is upright (0 rad) and grasp 0's tilts 0.1 at its best particle; grasp 2 scores 0.5, so it
    # comes after both. Particle 5 doesn't satisfy.
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: torch.full((len(q),), 0.05))
    inputs = _ranking_inputs([0, 0, 1, 2, 1, 0], [0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
                             [True] * 5 + [False], [1.0, 0.5, 2.0, 0.1, 3.0, 0.0])
    tries = alg.get_top_satisfying_particles(*inputs, n=10)
    assert [int(t["left_q1"][5]) for t in tries] == [2, 1, 3, 4, 0]
    assert [int(t["left_q1"][5]) for t in alg.get_top_satisfying_particles(*inputs, n=4)] == [2, 1, 3, 4]


def test_ranking_rejects_a_grasp_outside_the_pool(monkeypatch):
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: torch.full((len(q),), 0.05))
    plan_info, checker, reducer = _ranking_inputs([0, 1], [0.0, 0.0], [True, True], [0.0, 0.0])
    plan_info["particles"]["grasp1"][1, 0] += 1e-3
    with pytest.raises(RuntimeError, match="grasp1"):
        alg.get_top_satisfying_particles(plan_info, checker, reducer, n=10)
