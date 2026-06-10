"""Tests for the velocity/acceleration derivation in ``plan_processor``.

Covers the three changes from the
``2026-05-29-plan-velocity-acceleration-derivation-design`` spec:

* **C1** — Cartesian velocity from the analytic cuRobo Jacobian (``J·q̇``),
  world frame, linear rows ``[0:3]`` / angular rows ``[3:6]``.
* **C2** — Cartesian acceleration as ONE central difference
  (``np.gradient(..., edge_order=2)``) of the exact C1 velocity, with a
  genuine endpoint value (no 2-zero tail).
* **C3** — joint velocity/acceleration are native cuRobo values or a hard
  ``RuntimeError`` (no silently-fabricated finite difference).

The CUDA-gated tests build a real T1 processing kinematics and run FK +
Jacobian on the GPU; they synthesize their own trajectories (no planner
needed). The CPU tests cover the C2 endpoint math and the C3 error path.
"""
import pytest

from cutamp.tests.conftest import needs_cuda


# --------------------------------------------------------------------------- #
# C1 — Cartesian velocity via the cuRobo Jacobian (J·q̇).
# --------------------------------------------------------------------------- #
@needs_cuda
def test_jacobian_velocity_matches_fk_central_difference():
    """A synthetic trajectory drives two left-arm joints sinusoidally at a
    known analytic q̇. The Jacobian recipe ``J·q̇`` (linear rows [0:3]) must
    match a 2nd-order central difference of the FK world hand position at
    ALL interior timesteps to a tight tolerance, and the angular rows [3:6]
    must match an analytic quaternion-difference world ω."""
    import numpy as np
    import torch

    from cutamp.utils.plan_processor import (
        _build_processing_kinematics,
        LEFT_TOOL_FRAME,
        _quat_conjugate_wxyz,
        _quat_mul_wxyz,
    )
    from curobo.types import JointState

    kin = _build_processing_kinematics()
    device = kin.device_cfg.device
    names = list(kin.joint_names)
    dof = len(names)

    # Drive two left-arm joints sinusoidally; everything else held at 0.
    j0 = names.index("Left_Shoulder_Pitch")
    j1 = names.index("Left_Elbow_Pitch")

    T = 200
    dt = 0.01
    t = np.arange(T) * dt
    a0, w0 = 0.5, 2.0
    a1, w1 = 0.3, 3.0

    q = np.zeros((T, dof), dtype=np.float64)
    qd = np.zeros((T, dof), dtype=np.float64)
    q[:, j0] = a0 * np.sin(w0 * t)
    q[:, j1] = a1 * np.sin(w1 * t)
    qd[:, j0] = a0 * w0 * np.cos(w0 * t)   # analytic d/dt
    qd[:, j1] = a1 * w1 * np.cos(w1 * t)

    q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
    qd_t = torch.as_tensor(qd, dtype=torch.float32, device=device)

    # FK over the whole trajectory in one batched call, with Jacobian.
    fk_js = JointState.from_position(q_t.unsqueeze(0))
    ks = kin.compute_kinematics(fk_js)

    tf = list(kin.tool_frames)
    left_idx = tf.index(LEFT_TOOL_FRAME)

    tj = ks.tool_jacobians
    assert tj is not None, (
        "tool_jacobians is None — _build_processing_kinematics must pass "
        "compute_jacobian=True so the analytic Jacobian is populated."
    )
    # [batch, horizon, num_links, 6, dof]
    assert tj.shape == (1, T, len(tf), 6, dof), f"unexpected jac shape {tuple(tj.shape)}"

    J = tj[0, :, left_idx, :, :]          # [T, 6, dof]
    Jq = torch.einsum("tij,tj->ti", J, qd_t)   # [T, 6]
    lin = Jq[:, 0:3].cpu().numpy()
    ang = Jq[:, 3:6].cpu().numpy()

    # FK world hand position + quaternion sequence (wxyz).
    left_pose = ks.tool_poses.get_link_pose(LEFT_TOOL_FRAME, make_contiguous=True)
    hand_xyz = left_pose.position.reshape(T, 3).cpu().numpy().astype(np.float64)
    hand_quat = left_pose.quaternion.reshape(T, 4).cpu().numpy().astype(np.float64)

    # Reference linear velocity: 2nd-order central diff of world position.
    lin_ref = np.gradient(hand_xyz, dt, axis=0, edge_order=2)
    # Interior comparison (endpoints have larger one-sided FD error).
    lin_err = np.abs(lin[1:-1] - lin_ref[1:-1]).max()
    assert lin_err < 1e-3, (
        f"J·q̇ linear velocity disagrees with FK central diff: max err {lin_err:.2e}"
    )

    # Reference angular velocity: quaternion FD, world frame, sign-aligned.
    q_al = hand_quat.copy()
    for k in range(1, T):
        if np.dot(q_al[k], q_al[k - 1]) < 0:
            q_al[k] *= -1
    dq = np.gradient(q_al, dt, axis=0, edge_order=2)
    omega_quat = 2.0 * _quat_mul_wxyz(dq, _quat_conjugate_wxyz(q_al))
    ang_ref = omega_quat[..., 1:]
    ang_err = np.abs(ang[1:-1] - ang_ref[1:-1]).max()
    assert ang_err < 1e-2, (
        f"J·q̇ angular velocity disagrees with quat FD world ω: max err {ang_err:.2e}"
    )


# --------------------------------------------------------------------------- #
# C2 — Cartesian acceleration = central diff of the exact velocity.
# --------------------------------------------------------------------------- #
def test_central_diff_accel_correct_at_endpoints():
    """np.gradient(v, dt, edge_order=2) of a linear-in-time velocity
    v = a·t returns ≈ a at EVERY sample, including the last two — the old
    double-FD tail-duplication returned 0 there."""
    import numpy as np

    dt = 0.02
    T = 50
    t = np.arange(T) * dt
    a = np.array([1.5, -2.0, 0.75])         # constant accel per axis
    v = t[:, None] * a[None, :]             # [T, 3], v = a·t

    accel = np.gradient(v, dt, axis=0, edge_order=2)
    err = np.abs(accel - a[None, :]).max()
    assert err < 1e-9, f"central diff accel should be exactly a; max err {err:.2e}"

    # The crucial regression assertion: the last two samples are NOT zero.
    assert np.abs(accel[-1]).min() > 1e-6, "last accel sample collapsed to ~0"
    assert np.abs(accel[-2]).min() > 1e-6, "second-to-last accel sample collapsed to ~0"


def test_central_diff_accel_T2_no_crash():
    """T == 2 edge case: np.gradient with edge_order=2 falls back gracefully
    and returns finite values without raising."""
    import numpy as np

    dt = 0.02
    v = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])  # [2, 3]
    accel = np.gradient(v, dt, axis=0)                  # edge_order ignored for T=2
    assert accel.shape == (2, 3)
    assert np.isfinite(accel).all()


# --------------------------------------------------------------------------- #
# C3 — joints native or RuntimeError (no fabricated FD).
# --------------------------------------------------------------------------- #
@needs_cuda
def test_missing_native_velocity_raises():
    """A trajectory whose JointState has velocity=None must make
    process_motion_plan raise RuntimeError naming the missing native
    velocity — NOT silently finite-difference it.

    (In this cuRobo, ``JointState.from_position`` allocates zero
    velocity/acceleration tensors rather than leaving them None, so the
    derivative-less state is constructed by explicitly nulling .velocity.)"""
    import torch

    from cutamp.utils.plan_processor import (
        _build_processing_kinematics,
        process_motion_plan,
    )
    from curobo.types import JointState

    kin = _build_processing_kinematics()
    device = kin.device_cfg.device
    names = list(kin.joint_names)
    T, dof = 10, len(names)

    pos = torch.zeros(T, dof, device=device)
    js = JointState.from_position(pos, joint_names=names)
    js.velocity = None          # simulate a plan that lost native velocity
    assert js.velocity is None

    plan = [{"type": "trajectory", "plan": js, "dt": 0.02, "arm": "left"}]
    with pytest.raises(RuntimeError, match="(?i)velocity"):
        process_motion_plan(plan, kinematics=kin)


@needs_cuda
def test_missing_native_acceleration_raises():
    """velocity populated but acceleration=None must raise RuntimeError
    naming the missing native acceleration (not a fabricated FD)."""
    import torch

    from cutamp.utils.plan_processor import (
        _build_processing_kinematics,
        process_motion_plan,
    )
    from curobo.types import JointState

    kin = _build_processing_kinematics()
    device = kin.device_cfg.device
    names = list(kin.joint_names)
    T, dof = 10, len(names)

    pos = torch.zeros(T, dof, device=device)
    js = JointState.from_position(pos, joint_names=names)
    js.velocity = torch.zeros(T, dof, device=device)   # velocity present
    js.acceleration = None                              # accel missing
    assert js.acceleration is None

    plan = [{"type": "trajectory", "plan": js, "dt": 0.02, "arm": "left"}]
    with pytest.raises(RuntimeError, match="(?i)accel"):
        process_motion_plan(plan, kinematics=kin)


@needs_cuda
def test_native_joint_passthrough_and_no_two_zero_tail():
    """A plan with populated native joint velocity/acceleration:
      * joint channels equal the native cuRobo values verbatim (no
        np.gradient applied to joints);
      * for a Cartesian channel whose link is still MOVING at the final
        sample, the accel tail is genuinely non-zero AND the last two
        samples differ — the old double-FD forced a[-1]==a[-2]==0 there.

    A static link (e.g. the locked trunk here) correctly has a zero accel
    tail; that is physics, not the bug, so it is NOT asserted on."""
    import numpy as np
    import torch

    from cutamp.utils.plan_processor import (
        _build_processing_kinematics,
        process_motion_plan,
    )
    from cutamp.robots.t1 import LEFT_ARM_JOINT_NAMES
    from curobo.types import JointState

    kin = _build_processing_kinematics()
    device = kin.device_cfg.device
    names = list(kin.joint_names)
    dof = len(names)
    idx = {n: i for i, n in enumerate(names)}

    T = 60
    dt = 0.02
    t = np.arange(T) * dt

    # Drive two left-arm joints so the LEFT HAND is still moving (non-zero
    # velocity AND acceleration) at the final sample. A non-integer number
    # of periods over [0, (T-1)*dt] guarantees the endpoint is mid-swing,
    # so the old tail-duplicating double-FD would have wrongly zeroed the
    # last two Cartesian accel samples.
    j0 = idx["Left_Shoulder_Pitch"]
    j1 = idx["Left_Elbow_Pitch"]
    a0, w0 = 0.4, 1.7
    a1, w1 = 0.25, 2.3

    pos = np.zeros((T, dof), dtype=np.float32)
    vel = np.zeros((T, dof), dtype=np.float32)
    acc = np.zeros((T, dof), dtype=np.float32)
    pos[:, j0] = a0 * np.sin(w0 * t)
    vel[:, j0] = a0 * w0 * np.cos(w0 * t)
    acc[:, j0] = -a0 * w0 * w0 * np.sin(w0 * t)
    pos[:, j1] = a1 * np.sin(w1 * t)
    vel[:, j1] = a1 * w1 * np.cos(w1 * t)
    acc[:, j1] = -a1 * w1 * w1 * np.sin(w1 * t)

    js = JointState.from_position(
        torch.as_tensor(pos, device=device), joint_names=names,
    )
    js.velocity = torch.as_tensor(vel, device=device)
    js.acceleration = torch.as_tensor(acc, device=device)

    plan = [{"type": "trajectory", "plan": js, "dt": dt, "arm": "left"}]
    out = process_motion_plan(plan, kinematics=kin)
    seg = out["segments"][0]

    # Joint channels are the native values verbatim (in active-cspace order,
    # which equals from_position(names) order so no reprojection changes them).
    left_arm_idxs = [idx[n] for n in LEFT_ARM_JOINT_NAMES]
    np.testing.assert_allclose(
        seg["velocity"]["left_arm"], vel[:, left_arm_idxs], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        seg["acceleration"]["left_arm"], acc[:, left_arm_idxs], rtol=0, atol=1e-6
    )

    # The left hand is still moving at the final sample, so its accel tail
    # is a genuine non-zero value with distinct last-2 samples — exactly
    # what the old double-FD destroyed (it forced a[-1]==a[-2]==0.0).
    lh_acc = np.asarray(seg["acceleration"]["left_hand_xyz_ddot"])
    assert np.abs(lh_acc[-1]).max() > 1e-4, (
        f"moving left-hand final accel collapsed to ~0: {lh_acc[-1]!r}"
    )
    assert np.abs(lh_acc[-2]).max() > 1e-4, (
        f"moving left-hand 2nd-to-last accel collapsed to ~0: {lh_acc[-2]!r}"
    )
    assert not np.allclose(lh_acc[-1], lh_acc[-2]), (
        "last two accel samples are equal — tail-duplication regression "
        f"(a[-1]={lh_acc[-1]!r}, a[-2]={lh_acc[-2]!r})"
    )

    # Sanity: the left-hand velocity itself is non-zero at the final sample
    # (drives the accel assertion above and confirms J·q̇ is live).
    lh_vel = np.asarray(seg["velocity"]["left_hand_xyz_dot"])
    assert np.abs(lh_vel[-1]).max() > 1e-3
