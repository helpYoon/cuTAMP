"""Unit tests for plan_processor — covers quat sign canonicalization,
joint-name alignment guard, and the world-frame schema invariants."""
import numpy as np
import pytest


def test_quat_angular_velocity_robust_to_sign_flip():
    """A wxyz quat sequence that flips sign (q -> -q) on some samples
    represents the SAME physical rotation, but element-wise FD treats the
    flip as a huge motion. The naive bug only shows in the IMAGINARY part
    of dq ⊗ conj(q) when the rotation axis is non-trivial AND successive
    quats differ by a real rotation increment — pure ``q0 → -q0 → q0`` has
    dq ∥ q, which puts all spurious magnitude in the scalar component and
    leaves omega zero (vacuous). This test uses a continuous rotation about
    Y with sign flips on alternating samples so the bug fires in omega's
    imaginary part if canonicalization is absent."""
    from cutamp.utils.plan_processor import (
        _angular_velocity_from_quat,
        _quat_conjugate_wxyz,
        _quat_mul_wxyz,
    )

    # Continuous rotation about +Y at omega_true = 10 rad/s, dt = 0.1
    # (large step on purpose so the bug magnitude is unambiguous).
    omega_true = 10.0
    dt = 0.1
    theta_step = omega_true * dt  # 1.0 rad per step
    qstep = np.array(
        [np.cos(theta_step / 2), 0.0, np.sin(theta_step / 2), 0.0],
        dtype=np.float64,
    )

    # Build baseline (no flips): q_{k+1} = qstep ⊗ q_k. 5 samples → 4 steps.
    q_baseline = [np.array([1.0, 0.0, 0.0, 0.0])]
    for _ in range(4):
        q_baseline.append(_quat_mul_wxyz(qstep, q_baseline[-1]))
    q_baseline = np.stack(q_baseline, axis=0)

    # Flip sign on every other sample — physically identical rotation.
    q_flipped = q_baseline.copy()
    q_flipped[1::2] *= -1.0

    omega_baseline = _angular_velocity_from_quat(q_baseline, dt=dt)
    omega_with_flips = _angular_velocity_from_quat(q_flipped, dt=dt)

    # Baseline should report omega ≈ (0, omega_true, 0) on every step.
    expected = np.zeros_like(omega_baseline)
    expected[:, 1] = omega_true
    # FD on a finite rotation step underestimates magnitude slightly; allow
    # ~5% tol against the analytic truth.
    assert np.allclose(omega_baseline, expected, atol=0.5), (
        f"Baseline omega disagrees with analytic truth:\n"
        f"  got={omega_baseline}\n  expected≈{expected}"
    )

    # CORE CHECK: canonicalization must make the flipped output match the
    # baseline exactly (same physical rotation → same omega).
    assert np.allclose(omega_with_flips, omega_baseline, atol=1e-9), (
        f"Sign flips produced spurious omega:\n"
        f"  with_flips={omega_with_flips}\n  baseline={omega_baseline}\n"
        f"  diff={omega_with_flips - omega_baseline}"
    )

    # Sanity: compute what NAIVE FD (no canonicalization) would produce on
    # the flipped sequence — sign of the Y component flips entirely. This
    # confirms the test would fail if canonicalization were absent.
    dq_naive = (q_flipped[1:] - q_flipped[:-1]) / dt
    conj_q_naive = _quat_conjugate_wxyz(q_flipped[:-1])
    omega_naive_quat = 2.0 * _quat_mul_wxyz(dq_naive, conj_q_naive)
    omega_naive = omega_naive_quat[..., 1:]
    # Spurious omega differs from the canonicalized result by ≈ 2·omega_true
    # in the Y component (sign flip in the imag part of dq ⊗ conj(q)).
    diff_magnitude = np.max(np.abs(omega_naive - omega_baseline[:-1]))
    assert diff_magnitude > 0.5 * omega_true, (
        f"Sanity check: naive FD should differ from canonicalized by "
        f">~omega_true. Got diff_magnitude={diff_magnitude}, "
        f"omega_naive={omega_naive}, omega_baseline={omega_baseline}."
    )


def test_to_active_cspace_falls_through_on_name_mismatch():
    """Same DOF count but DIFFERENT joint name order must not short-circuit.
    Otherwise the same-shape positional gather mis-indexes every column."""
    import torch
    from cutamp.utils.plan_processor import _to_active_cspace, _build_processing_kinematics

    kin = _build_processing_kinematics()
    active = list(kin.joint_names)
    n = len(active)
    src_names = list(reversed(active))
    tensor = torch.zeros(5, n, device=kin.device_cfg.device)
    tensor[0, 0] = 1.0
    out = _to_active_cspace(tensor, src_names, kin)
    assert not torch.equal(out, tensor), (
        "_to_active_cspace short-circuited despite joint-name mismatch — "
        "downstream gathers would mis-index."
    )
