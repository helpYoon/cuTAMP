"""Reshape a raw ``solve_curobo`` motion plan into a compact, downstream-friendly
structured form for MPC tracking on the real T1.

**Schema version: 2** (BREAKING change from v1). Stored as
``plan["schema_version"]``. v1 pickles will be rejected by the consumer
example with a clear regenerate-with-current-code message.

Output schema per segment::

    {
        "arm": "left" | "right" | None,
        "dt": float,
        "T": int,
        "position": {
            "trunk_xyz":             [T, 3],     # WORLD, REAL-URDF Trunk (-0.0625 X applied)
            "trunk_quat_wxyz":       [T, 4],     # WORLD
            "trunk_quat_xyzw":       [T, 4],
            "trunk_height":          [T],        # alias for trunk_xyz[:, 2]
            "trunk_pitch":           [T],
            "trunk_yaw":             [T],
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
            # Joint velocities (from JointState.velocity, frame-independent)
            "trunk_pitch":                     [T],
            "trunk_yaw":                       [T],
            "right_arm":                       [T, 7],
            "left_arm":                        [T, 7],
            "trunk_xyz_dot":                     [T, 3],   # m/s, world
            "trunk_height":                      [T],
            "right_hand_xyz_dot":                [T, 3],   # m/s, WORLD
            "left_hand_xyz_dot":                 [T, 3],
            "trunk_angular_velocity_world":      [T, 3],
            "right_hand_angular_velocity_world": [T, 3],   # renamed from _trunk
            "left_hand_angular_velocity_world":  [T, 3],
        },
        "held_objs": {arm: (obj_name, [T, 4, 4] world_from_obj)},
    }

**Quaternion convention**: BOTH wxyz (cuRobo native) AND xyzw (ROS / scipy /
Eigen / MuJoCo) variants are emitted side-by-side. Pick whichever matches
your stack — they encode the same rotation.

**Angular velocity convention**: ``ω = 2 · (dq/dt) ⊗ conj(q)`` (take imag
part). For ``q = world_q_link`` this yields ω in **world** frame. All
angular velocity fields in v2 are in WORLD frame. Units: rad/s.

**Why this schema for MPC tracking on ``actual_robot.urdf``**:

* Hand poses are emitted in WORLD frame so the consumer (manipulation MPC)
  can use them directly without any frame transform. ``*_hand_link`` is
  used as the FK target since it exists in both URDFs (``*_base_link`` is
  sim-only).
* ``trunk_xyz`` has the +0.0625 m X offset (sim vs real URDF, see
  docs/sim_to_real_mapping.md #1) **already subtracted** so the saved
  value represents real-URDF's Trunk world pose. No compensation needed
  downstream.
* ``trunk_pitch`` and ``trunk_yaw`` are JOINT VALUES (Torso_Pitch and
  Waist_Yaw in our simplified URDF), not Trunk-link Euler angles. Joint
  values are frame-independent.
* We don't emit ankle_pitch, knee_pitch, or any other leg joint —
  expectation is that the MPC solves leg IK to match the saved Trunk
  world pose, choosing the missing 3 DOFs (Hip_Roll, Hip_Yaw, Ankle_Roll)
  for balance per its own logic.

Velocity sources:

* Joint velocities (trunk_pitch / trunk_yaw / arms) come from
  ``JointState.velocity`` which trajopt populates alongside positions.
* Cartesian velocities are first-order forward finite differences with
  the segment's ``dt``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import copy
import numpy as np
import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import DeviceCfg, JointState


# Frame names for the two end-effectors. Must match URDF link names.
LEFT_TOOL_FRAME = "left_hand_link"
RIGHT_TOOL_FRAME = "right_hand_link"
TRUNK_LINK = "Trunk"

# sim's Trunk frame sits +0.0625m in X of where actual_robot.urdf places it
# (see docs/sim_to_real_mapping.md #1). We subtract this from saved
# trunk_xyz before pickling so the saved value is real-URDF-native — the
# consumer needs no compensation.
SIM_TO_REAL_TRUNK_X_OFFSET_M = 0.0625


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
    return Kinematics(kin_cfg)


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


def _forward_finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    """Forward FD along axis 0; last sample duplicated to preserve shape."""
    if values.shape[0] < 2:
        return np.zeros_like(values)
    diff = np.diff(values, axis=0) / dt
    return np.concatenate([diff, diff[-1:]], axis=0)


def _angular_velocity_from_quat(q_seq: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity (rad/s) from a wxyz quaternion sequence via FD.

    Returns ω in the REFERENCE frame in which q_seq is defined
    (q_seq = world_q_link → ω in world).
    Math: ω = 2·(dq/dt) ⊗ conj(q), imaginary part. Last sample duplicated.
    """
    T = q_seq.shape[0]
    if T < 2:
        return np.zeros((T, 3), dtype=q_seq.dtype)
    # Canonicalize sign so consecutive samples are aligned. q and -q encode
    # the same rotation, but element-wise FD treats them as opposite — one
    # uncanonicalized flip yields ~4/dt rad/s spurious omega spike. Sequential
    # propagation: once a flip happens, every subsequent sample needs the flip
    # too. WHY sequential not element-wise: a single forward sweep handles
    # consecutive flips correctly.
    q_aligned = q_seq.copy()
    for t in range(1, T):
        if np.dot(q_aligned[t], q_aligned[t - 1]) < 0:
            q_aligned[t] *= -1
    dq = (q_aligned[1:] - q_aligned[:-1]) / dt
    conj_q = _quat_conjugate_wxyz(q_aligned[:-1])
    omega_quat = 2.0 * _quat_mul_wxyz(dq, conj_q)
    omega = omega_quat[..., 1:]
    return np.concatenate([omega, omega[-1:]], axis=0)


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


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (q wxyz, v 3-vec; broadcasts)."""
    v_quat = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return _quat_mul_wxyz(_quat_mul_wxyz(q, v_quat), _quat_conjugate_wxyz(q))[..., 1:]


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

        # Apply -0.0625 X compensation to Trunk world pose: saved value
        # represents real-URDF's Trunk world pose (not sim's), so the MPC
        # consumer needs no compensation.
        trunk_xyz_w_real = trunk_xyz_w.copy()
        trunk_xyz_w_real[:, 0] -= SIM_TO_REAL_TRUNK_X_OFFSET_M

        pos_np = pos_active.cpu().numpy()
        position = {
            # Trunk world pose (real-URDF-native)
            "trunk_xyz": trunk_xyz_w_real,
            "trunk_quat_wxyz": trunk_quat_w,
            "trunk_quat_xyzw": _xyzw_from_wxyz(trunk_quat_w),
            "trunk_height": trunk_xyz_w_real[:, 2],
            # Joint values (frame-independent; broadcast both sides on real)
            "trunk_pitch": pos_np[:, torso_pitch_idx],
            "trunk_yaw": pos_np[:, waist_yaw_idx],
            # Arms (per-side joint values, names match real URDF)
            "right_arm": pos_np[:, right_arm_idxs],
            "left_arm": pos_np[:, left_arm_idxs],
            # Hand poses in WORLD frame (was Trunk in v1)
            "right_hand_xyz": right_xyz_w,
            "right_hand_quat_wxyz": right_quat_w,
            "right_hand_quat_xyzw": _xyzw_from_wxyz(right_quat_w),
            "left_hand_xyz": left_xyz_w,
            "left_hand_quat_wxyz": left_quat_w,
            "left_hand_quat_xyzw": _xyzw_from_wxyz(left_quat_w),
        }

        velocity: Dict[str, Any] = {
            # Trunk linear velocity is unchanged by the X offset (constant
            # subtraction has zero derivative).
            "trunk_xyz_dot": _forward_finite_diff(trunk_xyz_w_real, dt),
            "trunk_height": _forward_finite_diff(trunk_xyz_w_real[:, 2], dt),
            # Hand linear velocities now in WORLD frame
            "right_hand_xyz_dot": _forward_finite_diff(right_xyz_w, dt),
            "left_hand_xyz_dot": _forward_finite_diff(left_xyz_w, dt),
            # Angular velocities — all WORLD frame
            "trunk_angular_velocity_world": _angular_velocity_from_quat(trunk_quat_w, dt),
            "right_hand_angular_velocity_world": _angular_velocity_from_quat(right_quat_w, dt),
            "left_hand_angular_velocity_world": _angular_velocity_from_quat(left_quat_w, dt),
        }
        if vel_active is not None:
            vel_np = vel_active.cpu().numpy()
            velocity["trunk_pitch"] = vel_np[:, torso_pitch_idx]
            velocity["trunk_yaw"] = vel_np[:, waist_yaw_idx]
            velocity["right_arm"] = vel_np[:, right_arm_idxs]
            velocity["left_arm"] = vel_np[:, left_arm_idxs]
        else:
            velocity["trunk_pitch"] = _forward_finite_diff(position["trunk_pitch"], dt)
            velocity["trunk_yaw"] = _forward_finite_diff(position["trunk_yaw"], dt)
            velocity["right_arm"] = _forward_finite_diff(position["right_arm"], dt)
            velocity["left_arm"] = _forward_finite_diff(position["left_arm"], dt)

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
            "held_objs": held_objs_np,
        })

    # Runtime metadata is kept compact — the authoritative schema reference
    # is the module docstring at the top of this file.
    return {
        "schema_version": 2,
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
