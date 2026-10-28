"""Unit tests for _ReikDriftGate (re-IK refresh instrumentation).

Exercises the drift bookkeeping and the stats records, not the IK solver.
CPU except the deferred-stats test, which needs CUDA events.
"""
import math

import pytest
import torch

from cutamp.optimize_plan import _ReikDriftGate
from cutamp.tests.conftest import needs_cuda


def _targets(n=4):
    return torch.eye(4).expand(n, 4, 4).clone()


def _rot_z(t, angle):
    out = t.clone()
    c, s = math.cos(angle), math.sin(angle)
    out[:, 0, 0], out[:, 0, 1] = c, -s
    out[:, 1, 0], out[:, 1, 1] = s, c
    return out


def test_first_refresh_has_no_reference():
    gate = _ReikDriftGate(1, pos_tol=0.005, rot_tol=0.05)
    pos_drift, rot_drift = gate.measure_drift(0, _targets())
    assert pos_drift is None and rot_drift is None


def test_drift_measured_per_interval():
    gate = _ReikDriftGate(1, pos_tol=0.005, rot_tol=0.05)
    t0 = _targets()
    gate.measure_drift(0, t0)
    t1 = t0.clone()
    t1[:, 0, 3] += 0.003
    pos_drift, _ = gate.measure_drift(0, t1)
    assert torch.allclose(pos_drift, torch.tensor(0.003), atol=1e-6)
    # Reference advanced: no further motion -> zero drift, not cumulative.
    pos_drift, _ = gate.measure_drift(0, t1)
    assert torch.all(pos_drift == 0)


def test_tiny_rotation_drift_resolves_in_float32():
    # curobo_pose_error's atan2 formulation must resolve sub-mrad drift (an
    # acos-based metric collapses to exactly 0.0 below ~5e-4 rad in float32).
    gate = _ReikDriftGate(1, pos_tol=0.005, rot_tol=0.05)
    t0 = _targets()
    gate.measure_drift(0, t0)
    _, rot_drift = gate.measure_drift(0, _rot_z(t0, 4e-4))
    assert rot_drift.max().item() > 1e-4


def test_log_stat_records_drift_histogram():
    gate = _ReikDriftGate(1, pos_tol=0.005, rot_tol=0.05)
    pos_drift = torch.tensor([0.0004, 0.0015, 0.003, 0.02])
    rot_drift = torch.zeros(4)
    gate.log_stat(5, "left_q3", 4, 1, 0.1, pos_drift, rot_drift, gn_fail=2,
                  gn_fail_pose=1, gn_fail_com=1)
    rec = gate.stats[-1]
    assert rec["conf"] == "left_q3" and rec["gn_fail"] == 2
    assert rec["frac_drift_below"]["2mm"] == 0.5


def test_flush_without_refreshes_is_noop():
    # A loop that ends before its first refresh (early stop, max_loop_dur)
    # queues nothing, so reik_refresh stays absent.
    gate = _ReikDriftGate(2, pos_tol=0.005, rot_tol=0.05)
    gate.flush()
    assert gate.stats == []


@needs_cuda
def test_log_batch_flush_matches_log_stat():
    # Balance-first stats stay on the device until flush(), which turns them
    # into the records log_stat writes: same keys, types and drift stats.
    D, B = 2, 4
    names = ["left_q1", "right_q2"]
    pose_ok = torch.tensor([1, 1, 0, 1, 0, 0, 1, 1], device="cuda").bool()
    com_ok = torch.tensor([1, 0, 1, 1, 1, 0, 1, 1], device="cuda").bool()
    pos_drift = torch.tensor([0.0004, 0.0015, 0.003, 0.02], device="cuda").repeat(D)
    rot_drift = torch.zeros(D * B, device="cuda")
    gate = _ReikDriftGate(D, pos_tol=0.005, rot_tol=0.05)
    events = []
    # GPU waits of different lengths, so solve_s is nonzero and tells the
    # two refreshes apart.
    for step, drift, cycles in ((7, (pos_drift, rot_drift), 2_000_000), (12, (None, None), 6_000_000)):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(cycles)
        end.record()
        events += [(start, end)] * D
        gate.log_batch(step, names, B, pose_ok & com_ok, pose_ok, com_ok, *drift, start, end)
    gate.flush()

    assert [(r["step"], r["conf"]) for r in gate.stats] == [
        (7, names[0]), (7, names[1]), (12, names[0]), (12, names[1]),
    ]
    for rec, counts, (start, end) in zip(gate.stats, [(2, 1, 1), (2, 2, 1)] * 2, events):
        assert (rec["gn_fail"], rec["gn_fail_pose"], rec["gn_fail_com"]) == counts
        assert rec["n_ik_fail"] == rec["gn_fail"] and rec["fallback"] is False and rec["solve_s"] >= 0
        # The refresh's event time, split evenly over its deps.
        assert rec["solve_s"] == round(start.elapsed_time(end) / 1e3 / D, 4)
    assert 0 < gate.stats[0]["solve_s"] < gate.stats[2]["solve_s"]

    ref_gate = _ReikDriftGate(1, pos_tol=0.005, rot_tol=0.05)
    ref_gate.log_stat(7, names[0], B, 2, 0.0, pos_drift[:B], rot_drift[:B],
                      gn_fail=2, gn_fail_pose=1, gn_fail_com=1, fallback=False)
    ref, rec = ref_gate.stats[0], gate.stats[0]
    assert rec.keys() == ref.keys()
    assert all(type(rec[k]) is type(ref[k]) for k in ref), {k: (type(rec[k]), type(ref[k])) for k in ref}
    assert rec["frac_drift_below"] == ref["frac_drift_below"] and rec["frac_drift_below"]["2mm"] == 0.5
    for key in ("pos_drift_mean", "pos_drift_max", "rot_drift_max"):
        assert rec[key] == pytest.approx(ref[key], rel=1e-6)
    drift_keys = {"pos_drift_mean", "pos_drift_max", "rot_drift_max", "frac_drift_below"}
    for rec in gate.stats[2:]:
        assert rec.keys() == ref.keys() - drift_keys
