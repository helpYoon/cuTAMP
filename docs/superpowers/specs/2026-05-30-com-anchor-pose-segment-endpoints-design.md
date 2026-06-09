# cspace-anchor pick/place trajectory endpoints to the COM-safe particle conf — design

**Date:** 2026-05-30
**Status:** investigation complete — design proposed, pending review
**Branch:** `curobo_v2`

## Goal

Stop the motion plan's center of mass (COM) from leaving the foot-support hull at
pick/place trajectory **arrivals**, by anchoring those trajectory endpoints to the
COM-safe optimizer particle conf (which reaches the identical hand pose) instead of
letting cuRobo freely re-resolve arm/torso redundancy about a bare Cartesian pose.

## Problem (root-caused this session)

A saved `data/motion_plan.pkl` had ~18 frames at the **LeftPlace arrival** out of the
COM support hull (penalty `1.36e-3` > tol `4e-4`, ≈ **3.7 cm past the +X / toe edge**;
trunk leaned `Torso_Pitch ≈ −0.99` + arm extended). Multi-stage instrumented
investigation established:

- **The hard COM gate works.** ComPolygon is enforced by `ConstraintChecker.get_mask`
  on the optimizer **particle confs** (`left_q*`/`right_q*`, name-prefix matched in
  `compute_com_polygon_penalties`). All confs pass 64/64; the place conf **`left_q3`
  itself is 2.0 cm INSIDE the +X edge, penalty exactly 0.0.** Coverage and gate
  effectiveness both confirmed.

- **The planner's soft COM cost is not "broken."** It is a soft rollout cost
  (`add_extra_cost`, `weight=5e5`) — a penalty, never a per-frame filter — and it
  cannot override a hard planner goal.

- **The real cause is free terminal redundancy resolution.** `solve_curobo`'s place
  branch (`motion_solver.py:454-500`) plans via `plan_single_arm_pose` → cuRobo
  `plan_pose` to a **Cartesian pose goal** (`target_pose`), NOT a cspace goal pinned to
  `left_q3`. So cuRobo re-resolves the arm/torso redundancy at the endpoint. A decisive
  probe compared the two for the same selected particle:

  | | `left_q3` (hard-gated conf) | LeftPlace trajectory **arrival** |
  |---|---|---|
  | left-hand world pose | — | **identical to 1 µm / 0°** |
  | COM penalty | **0.0** | 8.46e-5 *(this fresh run; 1.36e-3 in the saved pkl)* |
  | com_in_base X | 0.0915 (2.0 cm inside) | 0.1007 (drifts +0.92 cm toward toe) |
  | Torso_Pitch | −1.40 | −1.80 |
  | per-DOF Δ | — | up to ~1.0 rad (Elbow_Yaw −1.01, Shoulder_Pitch +0.86, knee +0.65, Torso −0.40…) |

  Same task pose, wholesale redundancy re-resolution, COM systematically shifted toward
  the toe edge — **the same direction and mechanism that put the saved pkl's arrival out
  of hull.** The drift is **run-dependent in magnitude, consistent in direction**: a
  fresh run happened to land in-hull (8.46e-5), the saved run did not (1.36e-3).

- **Note on terminology:** an earlier mapping mislabeled this boundary as the "retract"
  conf. It is the **LeftPlace arrival** (seg index 4 end). Retract confs return to
  `t1_home` (penalty 0, well inside) and are not the issue.

**Why anchoring is well-founded:** place's `target_pose` is built from the *same
particles* as the conf — `world_from_ee = action_4dof_to_mat4x4(best_particle[place_name])
@ obj_from_grasp @ tool_from_ee` — and `left_q3` (= `op.values[-1]`) is an IK solution for
exactly that pose. So the COM-safe particle conf is, by construction, a valid terminal
that reaches the identical hand pose; we are choosing the COM-safe redundancy branch
cuRobo already had available, not changing the task.

## Scope

- **In scope:** `place` and `pick` trajectory **terminal** redundancy in `solve_curobo`.
  The particle conf for each (place → `op.values[-1]`; pick grasp conf → `op.values[-1]`)
  is COM-hard-checked and reaches the target pose, so it is the anchor.
- **Out of scope:** retract/navigate (already cspace-planned, COM-safe); the soft
  planner COM cost weights (left as-is); mid-trajectory interpolation between waypoints
  (a complementary per-frame hard check is noted as future work, not done here); push /
  push_stick (no COM excursion observed — leave unless a probe shows otherwise).

## Approach (cspace-anchor the endpoint)

The chosen fix: make the pick/place trajectory **terminate at the COM-safe particle
conf**, not at a free Cartesian-pose redundancy solution, while preserving the exact hand
pose and the existing approach/collision handling.

The decision to resolve **during implementation** (it determines the exact cuRobo call):
cuRobo's `plan_pose` resolves redundancy itself, so the cleanest anchor is to give the
pose-plan a **cspace seed/terminal** at the particle conf. Two concrete mechanisms,
pick whichever the cuRobo v0.8 `plan_pose`/`plan_grasp` API actually supports (verify in
`curobo/_src/...`, do NOT edit curobo):

- **(A) Seed + retract-cfg toward the conf.** Pass the particle conf as the pose-plan's
  seed and/or `retract_config`/`cspace` bias so trajopt resolves redundancy at the conf's
  branch. Lightest touch; keeps `plan_pose`. Risk: a seed is a bias, not a hard pin —
  may reduce but not eliminate the drift; must be verified by re-probe.
- **(B) Two-phase: pose-plan the approach, then cspace-pin the terminal.** Plan the
  reach as today, but make the **final** waypoint a `plan_cspace`-style hard terminal at
  the particle conf (like retract already does at `motion_solver.py:377-402`). Strongest
  guarantee that the executed arrival == the COM-safe conf. Risk: more restructuring of
  the place/pick branch; must keep approach collision handling
  (`_disabled_world_obstacle`, attach/detach) intact.

Recommended: try **(A)** first (minimal, reuses `plan_single_arm_pose`); if a re-probe
shows residual drift past tol, escalate to **(B)**. Both keep the identical hand pose
because the anchor conf is an IK solution for `target_pose`.

For **pick** (`motion_solver.py:403-452`): the trajectory is approach→grasp→lift via
`plan_single_arm_grasp`/`plan_grasp`. The COM-relevant terminal is the **grasp** conf
(`op.values[-1]`); anchor the grasp waypoint's redundancy to it. The lift offset is a
fixed tool-frame translation and is separate; verify lift doesn't itself leave the hull
(probe showed pick arrivals in-hull, so this is a guard, not a known bug).

## Files

- `cutamp/motion_solver.py` — place branch (`454-500`) and pick branch (`403-452`):
  thread the particle conf (`best_particle[op.values[-1]]`) into the pose/grasp plan as a
  cspace seed/terminal anchor.
- `cutamp/grasp_planning.py` — `plan_single_arm_pose` (and possibly `plan_single_arm_grasp`):
  add an optional `seed_conf` / `terminal_conf` parameter threaded to the underlying
  cuRobo call. Keep the multi-frame-goal + inactive-arm-pin behavior unchanged.
- (No `curobo/` edits — use existing `plan_pose`/`plan_cspace`/seed parameters.)

## Tests / verification

- **Endpoint-COM regression (the key test):** generate a plan and assert **every
  pick/place segment arrival** has COM penalty ≤ `COM_TOL` (reuse the audit:
  reconstruct the 21-DOF arrival, `compute_com_polygon_penalties`). Must pass across
  several seeds (the bug was run-dependent — a single in-hull run is not sufficient
  evidence; run N≥5 and require all arrivals in-hull).
- **Hand-pose preserved:** the anchored arrival's left/right-hand world pose must still
  match the target (the particle conf's pose) to tight tol (µm / sub-deg) — i.e. the
  place/pick still reaches the object.
- **Endpoint == anchor conf:** assert the trajectory terminal config ≈ the particle conf
  (the whole point), within trajopt convergence tol.
- **No regression:** full suite green (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 … pytest
  cutamp/tests/ examples/`); end-to-end smoke (`run_cutamp blocks_t1 -n 64 --motion_plan`)
  still finds ≥1 satisfying plan and **motion-plan success rate does not drop** (the main
  risk of anchoring is reduced planner freedom → harder trajopt). Compare success rate
  before/after over several runs.
- **Per-frame audit:** rerun the full-trajectory COM audit (`/tmp/com_audit.py` pattern);
  the seg4/5 boundary excursion must be gone. Whole-trajectory 100%-in-hull is the goal
  for endpoints; note any residual mid-interpolation excursion for the future per-frame
  guard.

## Risks / notes

- **Reduced redundancy freedom may lower trajopt success** (anchoring constrains the
  terminal). This is the primary risk — verify success rate, and prefer the lighter
  seed-based (A) before the hard-pin (B). If (B) drops success materially, that argues
  for (A) + a soft weight bump, or accepting the per-frame-check approach instead.
- **Run-dependence of the original bug** means the fix must be validated over multiple
  seeds, not one lucky run.
- **Mid-trajectory frames** between waypoints are still only soft-constrained; this spec
  fixes endpoints. A hard per-frame COM verification + replan is the complementary
  guarantee on executed frames and is noted as future work (it was the runner-up fix
  option).
- **No `curobo/` edits / no fork** — the anchor uses cuRobo's existing
  seed/`plan_cspace` mechanisms; all changes are cuTAMP-side.
- **Regenerate `data/motion_plan.pkl`** after landing so the committed sample is endpoint-
  COM-clean (the current on-disk one has the excursion).
