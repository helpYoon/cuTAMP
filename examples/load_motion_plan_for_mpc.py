"""Consumer-side example: load a cuTAMP-generated motion plan and convert it
into the form an MPC tracking on the real T1 (using ``actual_robot.urdf``)
expects.

This is a reference implementation of the recipe in
``docs/sim_to_real_mapping.md``. Copy / paste / adapt for your stack.

Usage::

    python examples/load_motion_plan_for_mpc.py [path/to/motion_plan.pkl]

If the path is omitted, defaults to ``data/motion_plan.pkl`` in this repo.

This example expects ``schema_version=3`` pickles. Older plans (v1 with
Trunk-frame hand poses, v2 with a -0.0625 X compensation on trunk_xyz)
will be rejected with a regenerate-with-current-code message.

v3 saves ``trunk_xyz`` as the raw sim FK output (sim
``t1_simplified.urdf`` Trunk world pose). The on-robot ``actual_robot.urdf``
shares the same Trunk origin, so this value is the real Trunk world pose
directly — no compensation needed.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# ============================================================================
# Constants from docs/sim_to_real_mapping.md
# ============================================================================

# Joint-name mapping from sim's abstracted joints to real T1's joints. Arm
# names come from the canonical lists in cutamp.robots.t1 so there's only
# one definition. Broadcasts ("trunk_pitch" → both Hip_Pitches) are local
# to the MPC consumer convention and stay here.
try:
    from cutamp.robots.t1 import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
except ImportError:  # support running this file outside the cuTAMP env
    LEFT_ARM_JOINT_NAMES = (
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
        "Left_Elbow_Yaw", "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    )
    RIGHT_ARM_JOINT_NAMES = (
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
        "Right_Elbow_Yaw", "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    )

JOINT_MAP = {
    "trunk_pitch": ["Left_Hip_Pitch",   "Right_Hip_Pitch"],    # broadcast
    "trunk_yaw":   ["Waist"],
    "ankle_pitch": ["Left_Ankle_Pitch", "Right_Ankle_Pitch"],  # broadcast
    "knee_pitch":  ["Left_Knee_Pitch",  "Right_Knee_Pitch"],   # broadcast
    "left_arm":    list(LEFT_ARM_JOINT_NAMES),
    "right_arm":   list(RIGHT_ARM_JOINT_NAMES),
}

# Real-robot leg joints NOT commanded by this plan. The MPC's balance
# controller is expected to choose values for these (typically near 0
# for nominal upright stance; non-zero for balance correction).
MPC_FREE_LEG_JOINTS = [
    "Left_Hip_Roll",  "Left_Hip_Yaw",  "Left_Ankle_Roll",
    "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Ankle_Roll",
]


def segment_to_mpc_commands(seg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a per-timestep command stream for the MPC from one segment.

    Returns a dict with::

        {
            "dt": float,
            "T": int,
            "trunk_world_pose": {        # for MPC's world estimator
                "xyz":                  [T, 3],   # real-URDF-native (no compensation needed)
                "quat_xyzw":            [T, 4],   # ROS / scipy convention
                "xyz_dot":              [T, 3],   # linear velocity, world (m/s)
                "xyz_ddot":             [T, 3],   # linear acceleration, world (m/s^2)
                "angular_velocity":     [T, 3],   # angular velocity, world (rad/s)
                "angular_acceleration": [T, 3],   # angular acceleration, world (rad/s^2)
            },
            "joint_commands": {          # real-T1 joint values to track
                "<real_joint_name>": [T],  # ... 21 joints total
            },                            # 14 arm + Waist + 2 Hip_Pitch + 2 Knee_Pitch + 2 Ankle_Pitch
            "joint_velocities":     {"<real_joint_name>": [T]},   # dq/dt
            "joint_accelerations":  {"<real_joint_name>": [T]},   # d^2q/dt^2
            "hand_targets_in_world": {   # Cartesian targets in WORLD frame
                "right": {
                    "xyz": [T, 3], "quat_xyzw": [T, 4],
                    "xyz_dot": [T, 3], "xyz_ddot": [T, 3],
                    "angular_velocity": [T, 3], "angular_acceleration": [T, 3],
                },
                "left":  {...},
            },
            "held_objs": dict,
        }
    """
    P = seg["position"]
    V = seg["velocity"]
    A = seg["acceleration"]

    trunk_world_pose = {
        "xyz": P["trunk_xyz"].copy(),
        "quat_xyzw": P["trunk_quat_xyzw"],
        "xyz_dot": V["trunk_xyz_dot"],
        "xyz_ddot": A["trunk_xyz_ddot"],
        "angular_velocity": V["trunk_angular_velocity_world"],
        "angular_acceleration": A["trunk_angular_acceleration_world"],
    }

    joint_commands: Dict[str, np.ndarray] = {}
    joint_velocities: Dict[str, np.ndarray] = {}
    joint_accelerations: Dict[str, np.ndarray] = {}
    # .copy() per assignment so mutation of one broadcast target doesn't
    # silently mutate the other (Left_Hip_Pitch and Right_Hip_Pitch share
    # the same source array; MPC asymmetric trim would otherwise apply
    # both biases to both joints).
    for sim_field in ("trunk_pitch", "trunk_yaw", "ankle_pitch", "knee_pitch"):
        for rn in JOINT_MAP[sim_field]:
            joint_commands[rn] = P[sim_field].copy()
            joint_velocities[rn] = V[sim_field].copy()
            joint_accelerations[rn] = A[sim_field].copy()
    for sim_field in ("left_arm", "right_arm"):
        for i, rn in enumerate(JOINT_MAP[sim_field]):
            joint_commands[rn] = P[sim_field][:, i].copy()
            joint_velocities[rn] = V[sim_field][:, i].copy()
            joint_accelerations[rn] = A[sim_field][:, i].copy()

    hand_targets_in_world = {
        "right": {
            "xyz": P["right_hand_xyz"],
            "quat_xyzw": P["right_hand_quat_xyzw"],
            "xyz_dot": V["right_hand_xyz_dot"],
            "xyz_ddot": A["right_hand_xyz_ddot"],
            "angular_velocity": V["right_hand_angular_velocity_world"],
            "angular_acceleration": A["right_hand_angular_acceleration_world"],
        },
        "left": {
            "xyz": P["left_hand_xyz"],
            "quat_xyzw": P["left_hand_quat_xyzw"],
            "xyz_dot": V["left_hand_xyz_dot"],
            "xyz_ddot": A["left_hand_xyz_ddot"],
            "angular_velocity": V["left_hand_angular_velocity_world"],
            "angular_acceleration": A["left_hand_angular_acceleration_world"],
        },
    }

    return {
        "dt": seg["dt"],
        "T": seg["T"],
        "trunk_world_pose": trunk_world_pose,
        "joint_commands": joint_commands,
        "joint_velocities": joint_velocities,
        "joint_accelerations": joint_accelerations,
        "hand_targets_in_world": hand_targets_in_world,
        "held_objs": seg.get("held_objs", {}),
    }


def load_for_mpc(path: Path) -> List[Dict[str, Any]]:
    """One-call: load pickle, build per-segment MPC commands."""
    # WARNING: pickle.load executes arbitrary Python from the source file.
    # Only load motion_plan.pkl from trusted sources (own filesystem,
    # generated by cutamp on a machine you control). An attacker-supplied
    # pickle = arbitrary code execution.
    try:
        with open(path, "rb") as f:
            plan = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"motion_plan.pkl not found at {path}. Generate one with:\n"
            f"  python -m cutamp.scripts.run_cutamp --motion_plan --save_plan {path}"
        )
    except (EOFError, pickle.UnpicklingError) as e:
        raise RuntimeError(
            f"motion_plan.pkl at {path} appears corrupt or incomplete "
            f"({type(e).__name__}: {e}). Regenerate with --save_plan."
        )
    except (AttributeError, ModuleNotFoundError) as e:
        # AttributeError fires when the pickled class's module still exists
        # but the class itself was renamed/removed; ModuleNotFoundError
        # (a subclass of ImportError) fires when the module itself was
        # deleted/renamed, which is the more common case during refactors.
        raise RuntimeError(
            f"motion_plan.pkl at {path} references a class that's been "
            f"renamed/removed ({e}). Schema likely evolved; regenerate the plan."
        )

    schema_version = plan.get("schema_version")
    if schema_version is None:
        raise RuntimeError(
            f"motion_plan.pkl at {path} has no 'schema_version' key — it is a "
            f"legacy/unversioned or corrupt plan, not a versioned one. "
            f"Regenerate with the current code:\n"
            f"  python -m cutamp.scripts.run_cutamp --motion_plan --save_plan {path}"
        )
    if schema_version != 3:
        raise RuntimeError(
            f"This example expects schema_version=3, got {schema_version}. "
            f"Regenerate the plan with the current code:\n"
            f"  python -m cutamp.scripts.run_cutamp --motion_plan --save_plan {path}"
        )
    return [segment_to_mpc_commands(seg) for seg in plan["segments"]]


# ============================================================================
# Entry point — print a sanity summary
# ============================================================================

def main(path: Path) -> None:
    print(f"Loading: {path}")
    segments = load_for_mpc(path)
    print(f"Segments: {len(segments)}")
    print(f"MPC must additionally solve for these {len(MPC_FREE_LEG_JOINTS)} leg joints:")
    for n in MPC_FREE_LEG_JOINTS:
        print(f"  · {n}")
    print()

    total_duration = sum(s["T"] * s["dt"] for s in segments)
    print(f"Total trajectory duration: {total_duration:.2f} s")
    print()

    # Show segment 0 detail
    s0 = segments[0]
    print(f"--- Segment 0 (T={s0['T']}, dt={s0['dt']:.4f}s) ---")
    print(f"  Trunk world xyz at t=0:  {s0['trunk_world_pose']['xyz'][0]}")
    print(f"  Trunk world xyz at t=-1: {s0['trunk_world_pose']['xyz'][-1]}")
    print(f"  Trunk quat xyzw at t=0:  {s0['trunk_world_pose']['quat_xyzw'][0]}")
    print()
    print("  Joint commands at t=0 (real-T1 joint names):")
    for name, traj in sorted(s0["joint_commands"].items()):
        print(f"    {name:30s} = {traj[0]:+.4f} rad")
    print()
    rh = s0["hand_targets_in_world"]["right"]
    print(f"  Right hand target in WORLD @ t=0: xyz={rh['xyz'][0]}")
    print(f"                                    quat_xyzw={rh['quat_xyzw'][0]}")
    print(f"                                    xyz_dot={rh['xyz_dot'][0]} (m/s)")
    print(f"                                    angular_velocity={rh['angular_velocity'][0]} (rad/s)")
    print(f"  Trunk angular velocity @ t=0:    {s0['trunk_world_pose']['angular_velocity'][0]} (rad/s, world)")
    print(f"  Held objects this segment: {list(s0['held_objs'].keys())}")


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parent.parent / "data" / "motion_plan.pkl"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    try:
        main(path)
    except FileNotFoundError as e:
        sys.exit(str(e))
