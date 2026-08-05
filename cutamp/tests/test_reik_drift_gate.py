"""Unit tests for _ReikDriftGate (drift-gated selective re-IK).

CPU-only: exercises the drift math and skip logic, not the IK solver.
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


def _gate(drift_tol=0.002):
    return _ReikDriftGate(1, drift_tol, pos_tol=0.005, rot_tol=0.05)


def _prime(gate, target, ok=True, pos_err=0.001, rot_err=0.001):
    n = target.shape[0]
    gate.update(
        0, target, torch.arange(n),
        torch.full((n,), ok, dtype=torch.bool),
        torch.full((n,), pos_err), torch.full((n,), rot_err),
    )


def test_first_refresh_solves_everything():
    gate = _gate()
    need, pos_drift, rot_drift = gate.need_mask(0, _targets())
    assert need.all() and pos_drift is None and rot_drift is None


def test_small_drift_skips_large_drift_solves():
    gate = _gate(drift_tol=0.002)
    t0 = _targets()
    _prime(gate, t0)
    t1 = t0.clone()
    t1[:2, 0, 3] += 0.001  # 1 mm: under budget
    t1[2:, 0, 3] += 0.004  # 4 mm: over budget
    need, pos_drift, _ = gate.need_mask(0, t1)
    assert torch.allclose(pos_drift[:2], torch.tensor(0.001), atol=1e-6)
    assert (~need[:2]).all() and need[2:].all()


def test_rotation_drift_measured_and_gated():
    gate = _gate(drift_tol=0.002)
    t0 = _targets()
    _prime(gate, t0)
    t1 = _rot_z(t0, 0.1)  # 0.1 rad >> rot budget (0.002 * 10 = 0.02)
    need, _, rot_drift = gate.need_mask(0, t1)
    assert torch.allclose(rot_drift, torch.tensor(0.1), atol=1e-4)
    assert need.all()


def test_low_ik_headroom_forces_resolve():
    # 4.5 mm achieved error + 1 mm drift exceeds the 5 mm checker tolerance,
    # so drift-under-budget alone must not skip.
    gate = _gate(drift_tol=0.002)
    t0 = _targets()
    _prime(gate, t0, pos_err=0.0045)
    t1 = t0.clone()
    t1[:, 0, 3] += 0.001
    need, _, _ = gate.need_mask(0, t1)
    assert need.all()


def test_failed_solve_forces_resolve_at_zero_drift():
    gate = _gate(drift_tol=0.002)
    t0 = _targets()
    _prime(gate, t0, ok=False)
    need, pos_drift, _ = gate.need_mask(0, t0)
    assert need.all() and torch.all(pos_drift == 0)


def test_tiny_rotation_drift_resolves_in_float32():
    # Review-confirmed scenario: a solve at the success/checker rot boundary
    # (0.0497 rad, tolerance 0.05) followed by a 4e-4 rad target drift. An
    # acos-based drift metric collapses 4e-4 to exactly 0.0 in float32,
    # wrongly passing the headroom test; the atan2 metric (curobo_pose_error)
    # resolves it and must deny the skip.
    gate = _gate(drift_tol=0.002)
    t0 = _targets()
    _prime(gate, t0, rot_err=0.0497)
    t1 = _rot_z(t0, 4e-4)
    need, _, rot_drift = gate.need_mask(0, t1)
    assert rot_drift.max().item() > 1e-4
    assert need.all()


def test_tol_zero_disables_gating_but_records_drift():
    gate = _gate(drift_tol=0.0)
    t0 = _targets()
    _prime(gate, t0)
    need, pos_drift, _ = gate.need_mask(0, t0)
    assert need.all() and pos_drift is not None
