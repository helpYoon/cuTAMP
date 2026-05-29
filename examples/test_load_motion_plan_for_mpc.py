"""Aliasing-safety test for the MPC consumer example."""
import importlib.util
from pathlib import Path

import numpy as np

_MOD_PATH = Path(__file__).resolve().parent / "load_motion_plan_for_mpc.py"
_spec = importlib.util.spec_from_file_location("load_motion_plan_for_mpc", _MOD_PATH)
mpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpc)


def _vel_acc_block(T, xyz_keys):
    # Joint channels: scalar [T] (broadcast joints) or [T,7] (arms).
    d = {}
    for k in ("trunk_pitch", "trunk_yaw", "ankle_pitch", "knee_pitch"):
        d[k] = np.zeros(T)
    for k in ("left_arm", "right_arm"):
        d[k] = np.zeros((T, 7))
    for k in xyz_keys:
        d[k] = np.zeros((T, 3))
    return d


def _fake_segment(T=4):
    pos = {
        "trunk_xyz": np.zeros((T, 3)),
        "trunk_quat_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)),
        "trunk_pitch": np.zeros(T),
        "trunk_yaw": np.zeros(T),
        "ankle_pitch": np.zeros(T),
        "knee_pitch": np.zeros(T),
        "left_arm": np.zeros((T, 7)),
        "right_arm": np.zeros((T, 7)),
        "right_hand_xyz": np.zeros((T, 3)),
        "right_hand_quat_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)),
        "left_hand_xyz": np.zeros((T, 3)),
        "left_hand_quat_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)),
    }
    vel = _vel_acc_block(T, (
        "trunk_xyz_dot", "right_hand_xyz_dot", "left_hand_xyz_dot",
        "trunk_angular_velocity_world", "right_hand_angular_velocity_world",
        "left_hand_angular_velocity_world",
    ))
    acc = _vel_acc_block(T, (
        "trunk_xyz_ddot", "right_hand_xyz_ddot", "left_hand_xyz_ddot",
        "trunk_angular_acceleration_world", "right_hand_angular_acceleration_world",
        "left_hand_angular_acceleration_world",
    ))
    return {"dt": 0.02, "T": T, "position": pos, "velocity": vel,
            "acceleration": acc, "held_objs": {}}


def test_outputs_do_not_alias_source_segment():
    seg = _fake_segment()
    cmd = mpc.segment_to_mpc_commands(seg)
    # Mutate every emitted command stream in place; the source seg must not change.
    cmd["trunk_world_pose"]["xyz"][:] += 1.0
    for d in (cmd["joint_commands"], cmd["joint_velocities"], cmd["joint_accelerations"]):
        for arr in d.values():
            arr[:] += 1.0
    assert np.all(seg["position"]["trunk_xyz"] == 0.0)
    assert np.all(seg["position"]["left_arm"] == 0.0)
    assert np.all(seg["position"]["right_arm"] == 0.0)


import pickle
import pytest


def test_missing_schema_version_distinct_message(tmp_path):
    p = tmp_path / "plan.pkl"
    with open(p, "wb") as f:
        pickle.dump({"segments": []}, f)  # no schema_version key
    # Assert on the exception message directly (not via `match`, whose regex
    # would otherwise spuriously match the tmp_path that contains this test's
    # own function name). The absent-key branch must say the key is missing and
    # must NOT report a bogus "got 1".
    with pytest.raises(RuntimeError) as excinfo:
        mpc.load_for_mpc(p)
    msg = str(excinfo.value)
    assert "no 'schema_version' key" in msg
    assert "got 1" not in msg


def test_wrong_schema_version_says_got(tmp_path):
    p = tmp_path / "plan.pkl"
    with open(p, "wb") as f:
        pickle.dump({"schema_version": 2, "segments": []}, f)
    with pytest.raises(RuntimeError, match="got 2"):
        mpc.load_for_mpc(p)
