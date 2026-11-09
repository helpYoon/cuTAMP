import math
from types import SimpleNamespace

import pytest
import torch

import cutamp.com_polygon_cost as cpc
from cutamp.particle_initialization import (
    _BASE_JOINTS, PickChoice, choice_stats, choose_grasps, next_place, pick_place_confs, worst_posture,
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
    rows = [(True, 0.8, 0.9, 0.0, 0.0), (True, 0.5, 0.3, 0.0, 0.0), (True, 0.6, 0.1, 0.0, 0.0), (True, 0.4, 0.0, 0.0, 0.0)]
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
