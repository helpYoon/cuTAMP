"""Reshape a raw ``solve_curobo`` motion plan into a compact, downstream-friendly
structured form for MPC tracking on the real T1.

**Schema version: 3** (BREAKING change from v2). Stored as
``plan["schema_version"]``. Older pickles will be rejected by the consumer
example with a clear regenerate-with-current-code message.

v3 stores ``trunk_xyz`` as the raw sim FK output (sim ``t1_simplified.urdf``
Trunk world pose). The on-robot model the MPC uses, ``t1.urdf``, has a Trunk
frame IDENTICAL to the sim's (verified: FK of every Trunk child matches to
0.0 m), so the saved value IS the real Trunk world pose directly — no
compensation needed on the consumer side. (Do NOT confuse with the obsolete
``actual_robot.urdf``, whose Trunk frame sits 6.25 cm behind this one; that
file is not what the MPC uses.)

Output schema per segment::

    {
        "arm": "left" | "right" | None,
        "dt": float,
        "T": int,
        "position": {
            "trunk_xyz":             [T, 3],     # WORLD, sim t1_simplified.urdf Trunk pose
            "trunk_quat_wxyz":       [T, 4],     # WORLD
            "trunk_quat_xyzw":       [T, 4],
            "trunk_height":          [T],        # alias for trunk_xyz[:, 2]
            "trunk_pitch":           [T],     # sim Torso_Pitch == both Hip_Pitches on real (broadcast)
            "trunk_yaw":             [T],
            "ankle_pitch":           [T],     # single sim joint; broadcast to both ankles on real
            "knee_pitch":            [T],     # single sim joint; broadcast to both knees on real
            "right_arm":             [T, 7],
            "left_arm":              [T, 7],
            # Hand poses in WORLD frame (real-URDF-native)
            "right_hand_xyz":        [T, 3],     # WORLD
            "right_hand_quat_wxyz":  [T, 4],
            "right_hand_quat_xyzw":  [T, 4],
            "left_hand_xyz":         [T, 3],
            "left_hand_quat_wxyz":   [T, 4],
            "left_hand_quat_xyzw":   [T, 4],
        },
        "velocity": {
            # Joint velocities: native cuRobo JointState.velocity (required)
            "trunk_pitch":                     [T],
            "trunk_yaw":                       [T],
            "ankle_pitch":                     [T],
            "knee_pitch":                      [T],
            "right_arm":                       [T, 7],
            "left_arm":                        [T, 7],
            # Cartesian velocity from the analytic Jacobian J·q̇ (WORLD frame)
            "trunk_xyz_dot":                     [T, 3],   # m/s, world
            "trunk_height":                      [T],
            "right_hand_xyz_dot":                [T, 3],   # m/s, WORLD
            "left_hand_xyz_dot":                 [T, 3],
            "trunk_angular_velocity_world":      [T, 3],
            "right_hand_angular_velocity_world": [T, 3],
            "left_hand_angular_velocity_world":  [T, 3],
        },
        "acceleration": {
            # Joint accelerations: native cuRobo JointState.acceleration
            # (required; cuRobo's bspline trajopt populates it). A missing
            # native derivative raises rather than fabricating an FD.
            "trunk_pitch":                     [T],
            "trunk_yaw":                       [T],
            "ankle_pitch":                     [T],
            "knee_pitch":                      [T],
            "right_arm":                       [T, 7],
            "left_arm":                        [T, 7],
            # Cartesian linear accel: central diff (np.gradient) of J·q̇ vel
            "trunk_xyz_ddot":                       [T, 3],   # m/s^2, world
            "trunk_height":                         [T],
            "right_hand_xyz_ddot":                  [T, 3],   # m/s^2, WORLD
            "left_hand_xyz_ddot":                   [T, 3],
            # Angular accel: central diff (np.gradient) of the J·q̇ ω
            "trunk_angular_acceleration_world":     [T, 3],   # rad/s^2
            "right_hand_angular_acceleration_world": [T, 3],
            "left_hand_angular_acceleration_world":  [T, 3],
        },
        "held_objs": {arm: (obj_name, [T, 4, 4] world_from_obj)},
    }

**Quaternion convention**: BOTH wxyz (cuRobo native) AND xyzw (ROS / scipy /
Eigen / MuJoCo) variants are emitted side-by-side. Pick whichever matches
your stack — they encode the same rotation.

**Angular velocity convention**: emitted ω is the analytic Jacobian
(``J·q̇`` rows ``[3:6]``), already in the **world** frame. It equals the
quaternion-derivative identity ``ω = 2 · (dq/dt) ⊗ conj(q)`` (imag part) for
``q = world_q_link`` — the cross-check the derivatives test uses. All angular
velocity fields are in WORLD frame. Units: rad/s.

**Why this schema for MPC tracking on ``t1.urdf``**:

* Hand poses are emitted in WORLD frame so the consumer (manipulation MPC)
  can use them directly without any frame transform. ``*_hand_link`` is
  used as the FK target (present in both the sim and t1.urdf).
* ``trunk_xyz`` is the Trunk world pose, directly usable on the real robot:
  the sim and the MPC model t1.urdf share the Trunk frame (verified by FK to
  0.0 m), so no X compensation is needed.
* ``trunk_pitch`` and ``trunk_yaw`` are JOINT VALUES (Torso_Pitch and
  Waist_Yaw in our simplified URDF), not Trunk-link Euler angles. Joint
  values are frame-independent.
* We emit ``ankle_pitch`` and ``knee_pitch`` (each a single sim joint,
  broadcast to both legs on the real robot). The remaining 3 leg DOFs per
  side (Hip_Roll, Hip_Yaw, Ankle_Roll) are NOT emitted — the MPC chooses
  them for balance, matching the saved Trunk world pose.

Velocity / acceleration sources:

* Joint velocities AND accelerations (trunk_pitch / trunk_yaw / arms / legs)
  come from the native ``JointState.velocity`` / ``.acceleration`` that
  trajopt populates alongside positions. They are required — a ``None``
  native derivative raises ``RuntimeError`` rather than being fabricated.
* Cartesian velocities are the exact analytic Jacobian product ``J·q̇``
  (WORLD frame: linear rows ``[0:3]``, angular rows ``[3:6]``), using the
  ``tool_jacobians`` populated by ``compute_jacobian=True``.
* Cartesian accelerations are one 2nd-order central difference
  (``np.gradient(..., edge_order=2)``) of that exact velocity — genuine
  endpoint values, no fabricated 2-zero tail at segment boundaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import copy
import numpy as np
import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import DeviceCfg, JointState


# FK target link names (both hands + Trunk). Must match URDF link names.
LEFT_TOOL_FRAME = "left_hand_link"
RIGHT_TOOL_FRAME = "right_hand_link"
TRUNK_LINK = "Trunk"


def _build_processing_kinematics(device_cfg: Optional[DeviceCfg] = None) -> Kinematics:
    """Build a Kinematics for FK-only use, with ``Trunk`` added to
    ``tool_frames`` alongside both hands so the trunk's world pose is
    accessible via ``state.tool_poses``. This kinematics is NOT used for
    planning — only as a one-time FK over the saved trajectory points.

    Mobile base is locked (same as the planner) so the active cspace
    matches ``planner.kinematics.joint_names``.
    """
    from cutamp.robots.t1 import t1_curobo_cfg, _lock_mobile_base

    if device_cfg is None:
        device_cfg = DeviceCfg()
    cfg_dict = _lock_mobile_base(copy.deepcopy(t1_curobo_cfg()))
    cfg_dict["robot_cfg"]["kinematics"]["tool_frames"] = [
        LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME, TRUNK_LINK,
    ]
    kin_cfg = KinematicsCfg.from_robot_yaml_file(cfg_dict, device_cfg=device_cfg)
    # compute_jacobian=True populates KinematicsState.tool_jacobians
    # ([batch, horizon, num_links, 6, dof]) so Cartesian velocity can be
    # derived exactly as J·q̇ instead of finite-differencing the FK pose.
    return Kinematics(kin_cfg, compute_jacobian=True)


def _to_active_cspace(
    tensor: Optional[torch.Tensor],
    src_joint_names: Optional[List[str]],
    kin: Kinematics,
) -> Optional[torch.Tensor]:
    """Reproject a per-DOF [..., T, dof] tensor into the kinematics' active
    cspace order. cuRobo's interpolated trajectory typically arrives in the
    fully-articulated DOF order (head + grippers included); this trims +
    reorders. Returns ``None`` if ``tensor`` is ``None`` or the source names
    are unavailable for a shape mismatch.

    Used for BOTH positions and velocities — cuRobo's ``get_active_js``
    reorders via ``JointState.from_position(...).position`` so the same call
    pattern works for any per-DOF tensor (this is a documented duck-type
    use of ``from_position``).
    """
    if tensor is None:
        return None
    while tensor.dim() > 2:
        tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor[0]
    tensor = tensor.to(kin.device_cfg.device)

    active_dof = len(kin.joint_names)
    # Only short-circuit if shape AND joint-name order match the kinematics'
    # active cspace. Same DOF count in a different order would silently
    # mis-index every downstream per-joint gather.
    if tensor.shape[-1] == active_dof and (
        src_joint_names is None or list(src_joint_names) == list(kin.joint_names)
    ):
        return tensor
    if src_joint_names is None:
        return None
    src_js = JointState.from_position(tensor, joint_names=list(src_joint_names))
    return kin.get_active_js(src_js).position


def _central_diff(values: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference along axis 0 with genuine endpoint values.

    Uses 2nd-order one-sided stencils at the endpoints (``edge_order=2``)
    so the last two samples are NOT fabricated/duplicated — this is what
    fixes the old double-FD 2-zero acceleration tail. ``edge_order=2``
    needs >= 3 samples; degrade to 1 for shorter segments (T==2) and
    return zeros for a single sample.
    """
    T = values.shape[0]
    if T < 2:
        return np.zeros_like(values)
    edge_order = 2 if T >= 3 else 1
    return np.gradient(values, dt, axis=0, edge_order=edge_order)


def _xyzw_from_wxyz(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] → [x, y, z, w]."""
    return q[..., [1, 2, 3, 0]]


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    """wxyz conjugate (= inverse for unit quaternions)."""
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions (broadcasts)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


def process_motion_plan(
    curobo_plan: List[Dict[str, Any]],
    kinematics: Optional[Kinematics] = None,
) -> Dict[str, Any]:
    """Transform raw ``solve_curobo`` output into the structured form
    documented at the top of this module."""
    if kinematics is None:
        kinematics = _build_processing_kinematics()

    from cutamp.robots.t1 import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
    active_names: List[str] = list(kinematics.joint_names)
    idx = {n: i for i, n in enumerate(active_names)}
    torso_pitch_idx = idx["Torso_Pitch"]
    waist_yaw_idx = idx["Waist_Yaw"]
    ankle_pitch_idx = idx["ankle_pitch"]
    knee_pitch_idx = idx["knee_pitch"]
    left_arm_idxs = [idx[n] for n in LEFT_ARM_JOINT_NAMES]
    right_arm_idxs = [idx[n] for n in RIGHT_ARM_JOINT_NAMES]

    processed_segments: List[Dict[str, Any]] = []

    for entry in curobo_plan:
        if entry.get("type") != "trajectory":
            continue
        plan_js = entry["plan"]
        dt = float(entry["dt"])

        src_names = list(plan_js.joint_names) if plan_js.joint_names is not None else None
        pos_active = _to_active_cspace(plan_js.position, src_names, kinematics)
        if pos_active is None:
            raise ValueError(
                f"Cannot reproject trajectory: position dof "
                f"{tuple(plan_js.position.shape)[-1]} != active dof "
                f"{len(kinematics.joint_names)} and joint_names is None."
            )
        vel_active = _to_active_cspace(plan_js.velocity, src_names, kinematics)
        acc_active = _to_active_cspace(
            getattr(plan_js, "acceleration", None), src_names, kinematics,
        )
        # cuRobo's trajopt always populates native joint velocity AND
        # acceleration alongside positions (B-spline parameterization).
        # A None here means the plan lost its native derivatives upstream —
        # we refuse to fabricate a finite difference and surface it loudly.
        # vel_active is also required by C1 (J·q̇) for the Cartesian channels,
        # so this gate covers both joint and Cartesian velocity.
        if vel_active is None:
            raise RuntimeError(
                "Motion plan segment lacks native joint velocity "
                "(JointState.velocity is None); cuRobo trajopt should always "
                "provide it. Refusing to fabricate a finite-difference "
                "velocity — regenerate the plan with derivatives populated."
            )
        if acc_active is None:
            raise RuntimeError(
                "Motion plan segment lacks native joint acceleration "
                "(JointState.acceleration is None); cuRobo trajopt should "
                "always provide it. Refusing to fabricate a finite-difference "
                "acceleration — regenerate the plan with derivatives populated."
            )
        T = pos_active.shape[0]

        # FK over the segment to get tool-frame WORLD poses for both hands + Trunk.
        fk_js = JointState.from_position(pos_active.unsqueeze(0))
        ks = kinematics.compute_kinematics(fk_js)

        # All link poses are in WORLD frame here. Flatten to [T, ...].
        left_pose = ks.tool_poses.get_link_pose(LEFT_TOOL_FRAME, make_contiguous=True)
        right_pose = ks.tool_poses.get_link_pose(RIGHT_TOOL_FRAME, make_contiguous=True)
        trunk_pose = ks.tool_poses.get_link_pose(TRUNK_LINK, make_contiguous=True)

        left_xyz_w = left_pose.position.reshape(T, 3).cpu().numpy()
        left_quat_w = left_pose.quaternion.reshape(T, 4).cpu().numpy()
        right_xyz_w = right_pose.position.reshape(T, 3).cpu().numpy()
        right_quat_w = right_pose.quaternion.reshape(T, 4).cpu().numpy()
        trunk_xyz_w = trunk_pose.position.reshape(T, 3).cpu().numpy()
        trunk_quat_w = trunk_pose.quaternion.reshape(T, 4).cpu().numpy()

        pos_np = pos_active.cpu().numpy()
        position = {
            # Trunk world pose (raw sim FK; URDFs share Trunk origin → real-native)
            "trunk_xyz": trunk_xyz_w,
            "trunk_quat_wxyz": trunk_quat_w,
            "trunk_quat_xyzw": _xyzw_from_wxyz(trunk_quat_w),
            "trunk_height": trunk_xyz_w[:, 2],
            # Joint values (frame-independent; broadcast both sides on real)
            "trunk_pitch": pos_np[:, torso_pitch_idx],   # sim Torso_Pitch == both Hip_Pitches on real
            "trunk_yaw": pos_np[:, waist_yaw_idx],
            "ankle_pitch": pos_np[:, ankle_pitch_idx],   # single sim joint; broadcast to both ankles on real
            "knee_pitch": pos_np[:, knee_pitch_idx],     # single sim joint; broadcast to both knees on real
            # Arms (per-side joint values, names match real URDF)
            "right_arm": pos_np[:, right_arm_idxs],
            "left_arm": pos_np[:, left_arm_idxs],
            # Hand poses in WORLD frame
            "right_hand_xyz": right_xyz_w,
            "right_hand_quat_wxyz": right_quat_w,
            "right_hand_quat_xyzw": _xyzw_from_wxyz(right_quat_w),
            "left_hand_xyz": left_xyz_w,
            "left_hand_quat_wxyz": left_quat_w,
            "left_hand_quat_xyzw": _xyzw_from_wxyz(left_quat_w),
        }

        # --- C1: Cartesian velocity via the analytic Jacobian (J·q̇). ---
        # ks.tool_jacobians is [batch, horizon, num_links, 6, dof]; rows
        # [0:3] are WORLD-frame LINEAR velocity, rows [3:6] are WORLD-frame
        # ANGULAR velocity (probe-verified — no rotation needed). vel_active
        # is q̇ in kinematics.joint_names order, which equals the Jacobian's
        # dof order. Link indices are resolved by name (not hard-coded) so a
        # future tool_frames reorder cannot silently mis-index.
        tool_jac = ks.tool_jacobians
        if tool_jac is None:
            raise RuntimeError(
                "Processing kinematics did not produce tool_jacobians; "
                "_build_processing_kinematics must use compute_jacobian=True "
                "for analytic Cartesian velocity (J·q̇)."
            )
        tf = list(kinematics.tool_frames)
        left_idx = tf.index(LEFT_TOOL_FRAME)
        right_idx = tf.index(RIGHT_TOOL_FRAME)
        trunk_idx = tf.index(TRUNK_LINK)

        def _link_twist(link_idx: int):
            """Return (world_linear [T,3], world_angular [T,3]) for a link."""
            J = tool_jac[0, :, link_idx, :, :]            # [T, 6, dof]
            Jq = torch.einsum("tij,tj->ti", J, vel_active)  # [T, 6]
            twist = Jq.cpu().numpy()
            return twist[:, 0:3], twist[:, 3:6]

        trunk_lin_w, trunk_ang_w = _link_twist(trunk_idx)
        right_lin_w, right_ang_w = _link_twist(right_idx)
        left_lin_w, left_ang_w = _link_twist(left_idx)

        velocity: Dict[str, Any] = {
            # World-frame linear/angular velocity from J·q̇ (exact, no FD).
            "trunk_xyz_dot": trunk_lin_w,
            "trunk_height": trunk_lin_w[:, 2],
            "right_hand_xyz_dot": right_lin_w,
            "left_hand_xyz_dot": left_lin_w,
            "trunk_angular_velocity_world": trunk_ang_w,
            "right_hand_angular_velocity_world": right_ang_w,
            "left_hand_angular_velocity_world": left_ang_w,
        }
        # Native joint velocity passthrough (no fabrication; gated above).
        vel_np = vel_active.cpu().numpy()
        velocity["trunk_pitch"] = vel_np[:, torso_pitch_idx]
        velocity["trunk_yaw"] = vel_np[:, waist_yaw_idx]
        velocity["ankle_pitch"] = vel_np[:, ankle_pitch_idx]
        velocity["knee_pitch"] = vel_np[:, knee_pitch_idx]
        velocity["right_arm"] = vel_np[:, right_arm_idxs]
        velocity["left_arm"] = vel_np[:, left_arm_idxs]

        # --- C2: Cartesian acceleration = ONE central diff of the exact
        # C1 velocity (np.gradient, 2nd-order interior + one-sided endpoints).
        # Never duplicates the tail, so there is no 2-zero accel boundary.
        acceleration: Dict[str, Any] = {
            "trunk_xyz_ddot": _central_diff(velocity["trunk_xyz_dot"], dt),
            "trunk_height": _central_diff(velocity["trunk_height"], dt),
            "right_hand_xyz_ddot": _central_diff(velocity["right_hand_xyz_dot"], dt),
            "left_hand_xyz_ddot": _central_diff(velocity["left_hand_xyz_dot"], dt),
            "trunk_angular_acceleration_world":
                _central_diff(velocity["trunk_angular_velocity_world"], dt),
            "right_hand_angular_acceleration_world":
                _central_diff(velocity["right_hand_angular_velocity_world"], dt),
            "left_hand_angular_acceleration_world":
                _central_diff(velocity["left_hand_angular_velocity_world"], dt),
        }
        # Native joint acceleration passthrough (no fabrication; gated above).
        acc_np = acc_active.cpu().numpy()
        acceleration["trunk_pitch"] = acc_np[:, torso_pitch_idx]
        acceleration["trunk_yaw"] = acc_np[:, waist_yaw_idx]
        acceleration["ankle_pitch"] = acc_np[:, ankle_pitch_idx]
        acceleration["knee_pitch"] = acc_np[:, knee_pitch_idx]
        acceleration["right_arm"] = acc_np[:, right_arm_idxs]
        acceleration["left_arm"] = acc_np[:, left_arm_idxs]

        held_objs_np = {}
        for arm, (obj_name, poses) in (entry.get("held_objs") or {}).items():
            if isinstance(poses, torch.Tensor):
                poses = poses.detach().cpu().numpy()
            held_objs_np[arm] = (obj_name, poses)

        processed_segments.append({
            "arm": entry.get("arm"),
            "dt": dt,
            "T": T,
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "held_objs": held_objs_np,
        })

    # Runtime metadata is kept compact — the authoritative schema reference
    # is the module docstring at the top of this file.
    return {
        "schema_version": 3,
        "segments": processed_segments,
        "joint_name_groups": {
            "left_arm": list(LEFT_ARM_JOINT_NAMES),
            "right_arm": list(RIGHT_ARM_JOINT_NAMES),
            "trunk_pitch": "Torso_Pitch (broadcast to BOTH Left_Hip_Pitch + Right_Hip_Pitch on real)",
            "trunk_yaw": "Waist_Yaw (= real Waist joint)",
            "left_hand": LEFT_TOOL_FRAME,
            "right_hand": RIGHT_TOOL_FRAME,
        },
        "quaternion_convention": "wxyz (cuRobo) + xyzw (ROS/scipy/Eigen) both emitted; pick to match your stack",
        "schema_doc": "see cutamp/utils/plan_processor.py module docstring for full schema",
    }
