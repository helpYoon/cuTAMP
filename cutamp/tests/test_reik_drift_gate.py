"""Unit tests for _ReikDriftGate (re-IK refresh instrumentation).

CPU-only: exercises the drift bookkeeping, not the IK solver.
"""
import math

import torch

from cutamp.optimize_plan import _ReikDriftGate


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
