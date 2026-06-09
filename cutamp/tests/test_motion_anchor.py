"""Tests for COM-anchored pick/place terminals (plan_cspace to the conf)."""
import pytest
import torch

import cutamp.motion_solver as ms


def test_plan_arm_to_conf_targets_the_conf(monkeypatch):
    # _plan_arm_to_conf must plan with plan_cspace to a target built FROM the
    # conf (not a Cartesian pose), and return the cuRobo result on success.
    sentinel_target = object()
    monkeypatch.setattr(ms, "_planner_js_from_full",
                        lambda planner, conf, names: sentinel_target)
    monkeypatch.setattr(ms, "_cspace_plan_succeeded", lambda result, target: True)
    calls = {}

    class FakePlanner:
        def plan_cspace(self, target_js, last_js, max_attempts, enable_graph_attempt):
            calls["target_js"] = target_js
            calls["last_js"] = last_js
            calls["max_attempts"] = max_attempts
            return "FAKE_RESULT"

    res = ms._plan_arm_to_conf(
        FakePlanner(), torch.zeros(21), last_js="LAST", full_joint_names=["a"],
        disable_obstacle=None, retries=1, ground_op="OP",
    )
    assert res == "FAKE_RESULT"
    assert calls["target_js"] is sentinel_target  # planned to the conf, not a pose
    assert calls["last_js"] == "LAST"
    assert calls["max_attempts"] == ms.CSPACE_MAX_ATTEMPTS


def test_plan_arm_to_conf_raises_on_failure(monkeypatch):
    monkeypatch.setattr(ms, "_planner_js_from_full", lambda *a: object())
    monkeypatch.setattr(ms, "_cspace_plan_succeeded", lambda *a: False)

    class FakePlanner:
        def __init__(self):
            self.n = 0
        def plan_cspace(self, *a, **k):
            self.n += 1
            return "R"

    fp = FakePlanner()
    with pytest.raises(RuntimeError, match="cspace-anchored"):
        ms._plan_arm_to_conf(fp, torch.zeros(21), "LAST", ["a"],
                             disable_obstacle=None, retries=3, ground_op="OP")
    assert fp.n == 3  # retried `retries` times before raising
