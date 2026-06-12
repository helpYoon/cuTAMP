"""Report per-joint distances to position limits in a saved motion plan.

Regression gate for the joint-limit margin work
(spec: docs/superpowers/specs/2026-06-11-joint-limit-margin-design.md):
exits 1 if any PROTECTED-side approach (sides with a non-zero margin in
cutamp.robots.t1.T1_LIMIT_MARGIN_LOWER/UPPER) comes closer to its limit than
``--threshold`` (default 0.05 rad = half the 0.1 band), or if any waypoint
outright violates a limit. Home-side bounds (zero margin) are reported but
never gated — the standing posture legitimately sits ON them.

Reads schema-v3 plan pickles (see cutamp/utils/plan_processor.py). The 3
virtual base DOFs are not stored in the plan and are not checked.

Usage:
    python -m cutamp.scripts.check_plan_margins --plan data/motion_plan.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cutamp.robots.t1 import (
    JOINT_NAMES_FULL,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    T1_ASSETS_DIR,
    T1_LIMIT_MARGIN_LOWER,
    T1_LIMIT_MARGIN_UPPER,
)

# Plan schema v3 stores the 18 non-base joints (base DOFs are emitted as the
# world-frame trunk pose instead). Order matches the full cspace order.
STORED_JOINTS = JOINT_NAMES_FULL[3:]

# position-dict key -> joint name; arm entries are [T, 7] blocks appended after.
_SCALAR_KEYS = {
    "ankle_pitch": "ankle_pitch",
    "knee_pitch": "knee_pitch",
    "trunk_pitch": "Torso_Pitch",
    "trunk_yaw": "Waist_Yaw",
}

# segment_joint_matrix's column order must equal STORED_JOINTS — this gate
# silently lies about every distance if they ever diverge.
assert tuple(_SCALAR_KEYS.values()) + LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES == STORED_JOINTS


def parse_urdf_limits(urdf_path: Path) -> dict:
    """``{joint_name: (lower, upper)}`` for every revolute joint in the URDF."""
    limits = {}
    for joint in ET.parse(urdf_path).getroot().iter("joint"):
        limit = joint.find("limit")
        if joint.get("type") == "revolute" and limit is not None:
            limits[joint.get("name")] = (
                float(limit.get("lower")), float(limit.get("upper")),
            )
    return limits


def segment_joint_matrix(segment: dict) -> np.ndarray:
    """``[T, 18]`` joint positions in STORED_JOINTS order."""
    pos = segment["position"]
    cols = [np.asarray(pos[k], dtype=np.float64) for k in _SCALAR_KEYS]
    left = np.asarray(pos["left_arm"], dtype=np.float64)    # [T, 7]
    right = np.asarray(pos["right_arm"], dtype=np.float64)  # [T, 7]
    return np.column_stack(cols + [left, right])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default="data/motion_plan.pkl", help="Schema-v3 plan pickle.")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Min allowed distance (rad) to a PROTECTED-side limit.")
    args = parser.parse_args(argv)

    with open(args.plan, "rb") as f:
        plan = pickle.load(f)
    if plan.get("schema_version") != 3:
        sys.exit(f"Unsupported plan schema_version={plan.get('schema_version')!r} (need 3)")

    urdf_limits = parse_urdf_limits(T1_ASSETS_DIR / "t1_simplified.urdf")
    name_to_full_idx = {n: i for i, n in enumerate(JOINT_NAMES_FULL)}

    failures, violations = [], []
    header = f"{'joint':22s} {'side':5s} {'limit':>8s} {'min_dist':>9s} {'seg':>4s} {'where':>9s} {'gated':>6s}"
    print(header)
    print("-" * len(header))

    seg_mats = [segment_joint_matrix(seg) for seg in plan["segments"]]

    for j, name in enumerate(STORED_JOINTS):
        lo, hi = urdf_limits[name]
        fi = name_to_full_idx[name]
        for side, limit_val, margin in (
            ("lower", lo, T1_LIMIT_MARGIN_LOWER[fi]),
            ("upper", hi, T1_LIMIT_MARGIN_UPPER[fi]),
        ):
            best = None  # (dist, seg_idx, where)
            for s, mat in enumerate(seg_mats):
                col = mat[:, j]
                dist = (col - limit_val) if side == "lower" else (limit_val - col)
                ends = float(min(dist[0], dist[-1]))
                interior = float(dist[1:-1].min()) if len(dist) > 2 else ends
                d = min(ends, interior)
                where = "endpoint" if ends <= interior else "interior"
                if best is None or d < best[0]:
                    best = (d, s, where)
            d, s, where = best
            gated = margin > 0.0
            print(f"{name:22s} {side:5s} {limit_val:8.3f} {d:9.4f} {s:4d} {where:>9s} {str(gated):>6s}")
            # Strict: ANY limit overshoot fails, even on ungated home-side
            # bounds — a real violation there is still a hardware problem.
            # (Current plans dwell at exactly 0.0 on those bounds, never below.)
            if d < 0.0:
                violations.append((name, side, d, s))
            elif gated and d < args.threshold:
                failures.append((name, side, d, s))

    print()
    if violations:
        print(f"VIOLATIONS ({len(violations)}): " + ", ".join(
            f"{n}/{sd} {d:.4f} rad (seg {s})" for n, sd, d, s in violations))
    if failures:
        print(f"PROTECTED-SIDE FAILURES (< {args.threshold} rad, {len(failures)}): " + ", ".join(
            f"{n}/{sd} {d:.4f} rad (seg {s})" for n, sd, d, s in failures))
    if failures or violations:
        return 1
    print(f"OK: all protected-side approaches >= {args.threshold} rad, no violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
