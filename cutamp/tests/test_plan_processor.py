"""Unit tests for plan_processor — covers quat sign canonicalization,
joint-name alignment guard, and the world-frame schema invariants."""
import numpy as np
import pytest


def test_quat_angular_velocity_robust_to_sign_flip():
    """A wxyz quat sequence that flips sign (q -> -q) at each step represents
    the SAME rotation. Without canonicalization, element-wise FD treats the
    flip as a huge motion. This test asserts ω stays near zero across flips."""
    from cutamp.utils.plan_processor import _angular_velocity_from_quat

    q = np.array(
        [[1.0, 0, 0, 0],
         [-1.0, 0, 0, 0],
         [1.0, 0, 0, 0],
         [-1.0, 0, 0, 0]],
        dtype=np.float64,
    )
    omega = _angular_velocity_from_quat(q, dt=0.1)
    assert omega.shape == (4, 3)
    assert np.allclose(omega, 0.0, atol=1e-9), (
        f"Sign flips produced spurious omega: {omega}"
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
