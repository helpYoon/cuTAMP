"""Phase-0 probe: can the squat basin reach floor targets with Torso_Pitch margin?

Measures the kinematic ceiling for squat-biased IK seeding
(spec: docs/superpowers/specs/2026-06-11-squat-bias-ik-design.md §0):

1. Extracts the deep-lean endpoint configurations from a schema-v3 plan
   pickle (segments where trunk_pitch reaches <= --lean_threshold), rebuilds
   the full 21-DOF q (base DOFs are 0 — base locked in those plans), and
   FKs them through world.kinematics to recover the EXACT tool-frame world
   pose IK was solving (no hand-frame guesswork).
2. For each target: ONE batched IK call whose batch rows are a grid of leg
   warm-starts (ankle x knee x Torso_Pitch; arms home, base 0). Each row is
   `current_state_q`, which becomes LM seed row 0 -> basin selector.
3. Filters solutions by: IK success AND CoM-in-polygon
   (compute_com_polygon_mask) AND every protected-side joint-limit margin
   >= --margin (default 0.05 rad, per cutamp.robots.t1 margin vectors).
4. Prints per-target results and, on success, a copy-pasteable
   `T1_SQUAT_ANCHORS = (...)` literal (the best SEED posture — the anchor's
   job is seeding, not being the solution).

Exit 0 = squat basin can deliver the margin on every probed target (GO).
Exit 1 = ceiling: prints best achievable margin per target (NO-GO — stop
the feature and report, per the spec's success bar).

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True \
      python -m cutamp.scripts.probe_squat_ceiling --plan /tmp/jlm_motion_plan.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
import torch

from curobo.types import JointState

from cutamp.robots.t1 import (
    JOINT_NAMES_FULL,
    T1_LIMIT_MARGIN_LOWER,
    T1_LIMIT_MARGIN_UPPER,
    t1_home,
)

# Leg-posture grid: 4 ankle x 8 knee x 2 torso = 64 warm-start rows.
ANKLE_GRID = (0.0, -0.29, -0.58, -0.87)
KNEE_GRID = (0.0, 0.33, 0.67, 1.0, 1.34, 1.67, 2.0, 2.34)
TORSO_GRID = (0.0, -0.85)


def deep_lean_targets(plan: dict, lean_threshold: float):
    """[(seg_idx, arm, q_full np[21])] for the deepest-lean endpoint of each
    segment whose trunk_pitch dips below lean_threshold."""
    out = []
    for s, seg in enumerate(plan["segments"]):
        arm = seg.get("arm")
        if arm is None:
            continue
        pos = seg["position"]
        tp = np.asarray(pos["trunk_pitch"], dtype=np.float64)
        if tp.min() > lean_threshold:
            continue
        t = 0 if tp[0] <= tp[-1] else -1
        q = np.zeros(21, dtype=np.float32)  # base DOFs 0 (locked in saved plans)
        q[3] = pos["ankle_pitch"][t]
        q[4] = pos["knee_pitch"][t]
        q[5] = pos["trunk_pitch"][t]
        q[6] = pos["trunk_yaw"][t]
        q[7:14] = np.asarray(pos["left_arm"])[t]
        q[14:21] = np.asarray(pos["right_arm"])[t]
        out.append((s, arm, q))
    return out


def seed_grid(device, dtype) -> torch.Tensor:
    """[64, 21] full-cspace warm-starts: leg grid, arms home, base 0."""
    home = torch.tensor(t1_home, dtype=dtype, device=device)
    rows = []
    for a in ANKLE_GRID:
        for k in KNEE_GRID:
            for tq in TORSO_GRID:
                q = home.clone()
                q[3], q[4], q[5] = a, k, tq
                rows.append(q)
    return torch.stack(rows, dim=0)


def protected_margin_ok(q_full: torch.Tensor, limits: torch.Tensor, margin: float) -> torch.Tensor:
    """[B] bool: every protected-side joint distance >= margin (rad)."""
    m_lo = torch.tensor(T1_LIMIT_MARGIN_LOWER, device=q_full.device, dtype=q_full.dtype)
    m_hi = torch.tensor(T1_LIMIT_MARGIN_UPPER, device=q_full.device, dtype=q_full.dtype)
    lo_ok = torch.where(m_lo > 0, q_full - limits[0] >= margin, torch.ones_like(q_full, dtype=torch.bool))
    hi_ok = torch.where(m_hi > 0, limits[1] - q_full >= margin, torch.ones_like(q_full, dtype=torch.bool))
    return (lo_ok & hi_ok).all(dim=-1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--plan", default="data/motion_plan.pkl", help="Schema-v3 plan pickle with deep-lean endpoints.")
    parser.add_argument("--margin", type=float, default=0.05, help="Required protected-side margin (rad).")
    parser.add_argument("--lean_threshold", type=float, default=-1.5, help="trunk_pitch below this marks a deep-lean segment.")
    args = parser.parse_args(argv)

    with open(args.plan, "rb") as f:
        plan = pickle.load(f)
    if plan.get("schema_version") != 3:
        sys.exit(f"Unsupported plan schema_version={plan.get('schema_version')!r} (need 3)")

    targets = deep_lean_targets(plan, args.lean_threshold)
    if not targets:
        sys.exit("No deep-lean endpoints found in the plan — nothing to probe.")
    print(f"Probing {len(targets)} deep-lean endpoint target(s) from {args.plan}")

    # Heavy imports after arg parsing.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.particle_initialization import _ik_for_pose, _ik_solution_to_full_q, _ik_success_mask
    from cutamp.tests.conftest import make_blocks_t1_world

    world = make_blocks_t1_world()
    device = world.q_init.device
    dtype = world.q_init.dtype
    full_names = list(world.kinematics.joint_names)
    limits = world.kinematics.get_joint_limits().position.to(device=device, dtype=dtype)
    grid = seed_grid(device, dtype)  # [64, 21]
    B = grid.shape[0]
    torso_idx = list(JOINT_NAMES_FULL).index("Torso_Pitch")

    # FK the recorded endpoint configs -> exact tool-frame world poses.
    q_targets = torch.tensor(np.stack([q for _, _, q in targets]), dtype=dtype, device=device)
    js = JointState.from_position(q_targets, joint_names=full_names)
    kin_state = world.kinematics.compute_kinematics(js)

    go = True
    anchor_rows = []  # (target_z, best seed posture) per target
    for i, (s, arm, _q) in enumerate(targets):
        frame = world.tool_frame_for_arm(arm)
        target_mat = kin_state.tool_poses.get_link_pose(frame, make_contiguous=True).get_matrix()[i]  # [4, 4]
        target_z = float(target_mat[2, 3])
        goals = target_mat.unsqueeze(0).expand(B, 4, 4).contiguous()
        result = _ik_for_pose(world, goals, arm, current_state_q=grid)
        q_sol = _ik_solution_to_full_q(result, world)
        ok = (
            _ik_success_mask(result)
            & compute_com_polygon_mask(world, q_sol)
            & protected_margin_ok(q_sol, limits, args.margin)
        )
        torso_margin = q_sol[:, torso_idx] - limits[0, torso_idx]
        print(f"\nseg {s} ({arm}), tool-frame target z={target_z:.3f} m: "
              f"{int(ok.sum())}/{B} grid seeds reach margin >= {args.margin}")
        if ok.any():
            best = int(torch.argmax(torch.where(ok, torso_margin, torch.full_like(torso_margin, -1e9))))
            sp = grid[best]
            print(f"  best: Torso margin {float(torso_margin[best]):.3f} rad | "
                  f"seed ankle={float(sp[3]):.2f} knee={float(sp[4]):.2f} torso={float(sp[5]):.2f} | "
                  f"solution ankle={float(q_sol[best, 3]):.2f} knee={float(q_sol[best, 4]):.2f} "
                  f"torso={float(q_sol[best, torso_idx]):.2f}")
            anchor_rows.append((target_z, sp))
        else:
            best_any = int(torch.argmax(torso_margin))
            print(f"  CEILING: best achievable Torso margin {float(torso_margin[best_any]):.4f} rad "
                  f"(success={bool(_ik_success_mask(result)[best_any])}) — below {args.margin}")
            go = False

    print()
    if not go:
        print("NO-GO: the squat basin cannot deliver the required margin on at "
              "least one probed target. Record the ceiling above in the spec and stop.")
        return 1

    # Anchor emission: floor anchor = seed posture of the lowest-z target's winner.
    z_floor, sp = min(anchor_rows, key=lambda r: r[0])
    print("GO. Paste into cutamp/robots/t1.py (Task 2):")
    print("T1_SQUAT_ANCHORS = (")
    print("    (T1_SQUAT_Z_STAND, 0.0, 0.0, 0.0),")
    print(f"    ({z_floor:.3f}, {float(sp[3]):.3f}, {float(sp[4]):.3f}, {float(sp[5]):.3f}),")
    print(")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
