"""Precision criteria for the offline seed-3 landing prior, in meters and degrees."""

import math

CHECKPOINT_SHA256 = "766d602d8cabf1ccacfad7b59d7d09baac53ad17b59b9f6ead57e09739435f8b"
SOURCE_SHA256 = {
    "assets/t1/t1_description/urdf/T1_29dof.urdf": "e2bb70467a5b7685a47fe79d406a396cb06a4551623e5e24f97e031206e938ea",
    "assets/t1/mujoco/t1_actuators.npz": "d03a1cfd4314b55df8282b0731c58434cde87d273fd24a1d0fc4afe5f010518f",
    "robot/t1.py": "32edd74e0700d93587e42685aebc966cb9d3d316e93df411c028072b29e04f12",
    "src/dr.py": "8f50b42ca432e4c2fe5d7b3c2099090a8abbdf504a2577efb9d59f4315780205",
    "src/landing.py": "dddec07c9796b4cf450bb3d06c5f7d627e21ea827d14a40cbc84a5a138c10285",
    "src/mdp.py": "e1e6a47c24cd2648fecffd73cba5d085d1aa7271b1216cac5c24cfba9ed7c4ce",
    "src/motion.py": "7de50101728ff5359706817b8a28fe7ebe381b4d9d97a1a65a4b5acac6312669",
    "src/networks.py": "9101edfea0fe98fab3f46e4028aa73d885f1c9c2aed2266749205219abc942db",
    "utils/export_landing.py": "2cd45d16ad435cb7475677aea7000b46469501075b72c2a00ea1a6abc968c1a4",
    "utils/rotations.py": "0b009b02dd1adef8ca3f1cacb18ad201f722422f44e81792d0676f1ee4ec5223",
}


def check_profile(provenance):
    expected = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_sha256": SOURCE_SHA256,
        "model_only": True,
        "control_hz": 50,
        "warmup_ticks": 50,
        "hold_ticks": 75,
        "xy_threshold_m": 0.02,
        "yaw_threshold_deg": 1.0,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"incompatible landing accuracy profile: {key}")


def _within_pair(handoff, reference):
    within = True
    for measure, threshold in (("xy_error_m", 0.020), ("yaw_error_deg", 1.0)):
        errors = handoff.get(f"{reference}_{measure}")
        if not isinstance(errors, list) or len(errors) != 2:
            return None
        for error in errors:
            if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
                return None
            within &= error < threshold
    return within


def handoff_precision(stages, order):
    """Both soles at each issued handoff and the original final pair. Incomplete evidence yields None."""
    if not isinstance(stages, list) or len(stages) != len(order) or order not in ("L", "R", "LR", "RL"):
        return None
    passed = True
    for foot, stage in zip(order, stages):
        if not isinstance(stage, dict) or stage.get("foot") != foot or type(stage.get("issue_legal")) is not bool:
            return None
        handoff = stage.get("handoff")
        if not isinstance(handoff, dict):
            return None
        issued = _within_pair(handoff, "issued")
        if issued is None:
            return None
        passed &= stage["issue_legal"] and issued
    final = _within_pair(stages[-1]["handoff"], "nominal")
    return None if final is None else passed and final
