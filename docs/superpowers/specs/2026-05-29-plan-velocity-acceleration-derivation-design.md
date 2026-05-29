# Plan velocity/acceleration derivation — design

**Date:** 2026-05-29
**Status:** approved-pending-review
**Branch:** `curobo_v2`
**Origin:** carved out of the COM-polygon & MPC-consumer review-fix spec
(`2026-05-29-com-and-mpc-review-fixes-design.md`, root cause "C") because the
root-cause investigation (C4) is open-ended enough to warrant its own
spec → plan → implementation cycle.

## Goal

Make the velocities and accelerations that `cutamp/utils/plan_processor.py`
writes into the MPC plan physically correct — in particular, eliminate the
fabricated-zero acceleration at every segment boundary, and use cuRobo's native
quantities (or analytic Jacobian) instead of lossy finite differences.

## Constraints

- **No edits to `curobo/`** (vendored, read-only; would need a fork). All
  changes are cuTAMP-side. The Jacobian and native derivatives this design uses
  are already produced by cuRobo — we only opt into them.
- Tests run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Stage only specific files; no `git add -A`.

## Scope

Review findings **8** and **13** (the double-finite-difference acceleration
tail). Out of scope: the COM-polygon, 0.0625-offset-purge, numpy-`.copy()`, and
misc-hardening fixes — those are the A/B/D/E causes in the sibling spec.

## Problem

`_forward_finite_diff` (`cutamp/utils/plan_processor.py:187-192`) duplicates its
last sample to preserve shape:

```python
diff = np.diff(values, axis=0) / dt
return np.concatenate([diff, diff[-1:]], axis=0)
```

Acceleration is this function applied twice (velocity = FD(position),
acceleration = FD(velocity)). Each FD duplicates its tail, so the **last two
samples of every acceleration channel collapse toward 0** — the duplicate makes
`v[T-1] == v[T-2]`, so `a[T-2] = (v[T-1]-v[T-2])/dt = 0`, and `a[T-1] = a[T-2]`
is the second duplicate. An MPC feedforwarding torque from accel gets zero
feedforward at every segment **boundary** — the worst place for it.

**What cuRobo provides (verified against source):**

- **Joint space — native.** `JointState` carries `velocity`/`acceleration`/`jerk`
  (`curobo/.../state/state_joint.py:73-76`); trajopt fills them from its B-spline
  parameterization and the interpolation kernel resamples all four with correct
  `dt`-scaling (`curobo/.../util/warp_interpolation.py:95-116`). So the joint FD
  branches (`plan_processor.py:361-367, 391-397`) are a *fallback*.
- **Cartesian space — NOT native.** FK (`KinematicsState`) returns only poses,
  `tool_jacobians`, spheres, and COM — **no link velocity/acceleration**
  (`curobo/.../robot/kinematics/kinematics_state.py:8-18`). The hand/trunk world
  velocity must be derived; today that is FD-of-pose then FD-again.
- **Jacobian IS available.** `KinematicsState.tool_jacobians`
  (`[batch, horizon, num_links, 6, dof]`) is populated when the kinematics is
  built with `compute_jacobian=True` (`kinematics.py:52,136,167`). This is a
  cuTAMP-side constructor flag on our **own** `_build_processing_kinematics`
  (`plan_processor.py:130`) — **no cuRobo edit, no fork.**

**Empirical confirmation (loaded `data/motion_plan.pkl`, schema v3, 8 segs):**

- Cartesian accel tails are the exact double-FD signature —
  `right_hand_xyz_ddot` last-5 per-row norms `[2.4e-4, 2.4e-4, 2.4e-4, 0, 0]`;
  same for `trunk_xyz_ddot`, `trunk_angular_acceleration_world` (three real
  values then **two zeros**).
- Joint channels also show the alternating-zero FD-duplicate pattern
  (`0.0` at `[-3]` and `[-1]`), which means **trajopt's native joint derivatives
  are NOT reaching `plan_processor`** for this plan — the joint FD fallback is
  firing when it shouldn't. (See C4.)

## Fix

Four sub-fixes — Cartesian via Jacobian, joints prefer-native, the underlying
finite-difference made endpoint-correct, and a root-cause investigation of the
missing native joint derivatives.

**C1 — Cartesian velocity via the Jacobian (`J·q̇`).** Build
`_build_processing_kinematics` with `compute_jacobian=True`. For each link,
`tool_jacobians[..., link, :, :]` is the `[6, dof]` spatial Jacobian; with the
joint velocity `q̇` (native from the plan, else C3's central diff), the link
spatial velocity is the analytic `v = J · q̇` (rows 0-2 linear, 3-5 angular).
This replaces FD-of-pose for `*_hand_xyz_dot`, `trunk_xyz_dot`, and the angular
velocities — no finite difference, no endpoint fabrication, physically exact.
(Angular velocity from `J·q̇` also retires `_angular_velocity_from_quat` and its
quaternion-FD entirely.)

**C2 — Cartesian acceleration.** No second analytic option (cuRobo exposes no
`J̇` and no native link accel), so accel stays one differentiation of the *exact*
C1 velocity — but via C3's endpoint-correct central difference, not the tail-
duplicating forward diff. Result: `v̇` from a clean `v`, last samples genuine.

**C3 — Joints: prefer native, central-diff fallback, and fix the helper.**
- Replace `_forward_finite_diff` with `np.gradient(x, dt, axis=0, edge_order=2)`
  (central diff interior, 2nd-order one-sided at both ends, never duplicates).
  Used for any remaining FD (joint fallback + Cartesian accel in C2).
- Keep the "native if populated" branches (`vel_active`/`acc_active`) — native
  B-spline derivatives are higher quality than any FD.

**C4 — Investigate why native joint derivatives are missing (root-cause).**
The pkl shows the joint FD fallback firing, meaning `plan_js.velocity` /
`.acceleration` are `None` (or get dropped) by the time `plan_processor` runs.
Trace the path from trajopt result → `_interp_plan`
(`motion_solver.py:141-171`, `get_interpolated_plan`) → `entry["plan"]` →
`_to_active_cspace`. Candidates: (a) the planner returns `js_solution` (no
derivatives) rather than the interpolated trajectory on some branches;
(b) `get_full_dof_from_solution` / `augment_joint_state` re-augments locked DOFs
in a way that returns position-only; (c) `_to_active_cspace` only reprojects
`.position`. Fix at the source so native joint vel/accel survive to the
processor; if a branch genuinely lacks them, C3's central-diff fallback is the
correct backstop. **This sub-task is investigation-first** (systematic-debugging:
find the layer that drops the derivatives before changing code).

## Files

- `cutamp/utils/plan_processor.py` — `compute_jacobian=True` in
  `_build_processing_kinematics`; `J·q̇` Cartesian velocity (C1); accel from
  central diff of it (C2); `_forward_finite_diff` → `np.gradient` (C3); remove
  `_angular_velocity_from_quat` if fully superseded by C1.
- `cutamp/motion_solver.py` and/or the trajopt-result path — only if C4 finds
  the derivative drop there (scope confirmed during investigation).

## Tests (new test module or extend an existing one)

- **C1 Jacobian velocity:** a trajectory moving one joint at known `q̇` → assert
  `J·q̇` link linear velocity matches an analytic FK-difference to tight tol, at
  **all** timesteps including endpoints.
- **C2/C3 endpoint correctness:** constant-acceleration joint trajectory
  `q(t)=0.5·a·t²` → `np.gradient` twice yields `≈ a` at **every** sample
  including the last two (the old code returns 0 there).
- **C3 native passthrough:** a plan JointState *with* populated velocity/accel →
  processor emits those values verbatim (no FD applied).
- **C4 regression:** after the source fix, a freshly generated plan's joint
  vel/accel tails are non-degenerate (no alternating-zero FD signature).
- Linear trajectory: second derivative `≈ 0` (sanity). T==2: finite, no crash.

## Sequencing

Investigation-first, then implementation:

1. **C4 investigation** — find where native joint derivatives are dropped before
   `plan_processor`. May land as a source fix in `motion_solver.py` if the drop
   is upstream of the processor. Do this first: its outcome decides whether the
   joint path needs the FD fallback at all.
2. **C3** — `_forward_finite_diff` → `np.gradient` helper + native passthrough.
3. **C1/C2** — Jacobian `J·q̇` Cartesian velocity + accel via central diff of it;
   retire `_angular_velocity_from_quat`.

## Verification

End-to-end: regenerate a plan and confirm Cartesian and joint accel tails are
non-degenerate (no alternating-zero / two-zero FD signature), and `J·q̇` link
velocity matches an analytic FK-difference to tight tol.

```
PYTORCH_ALLOC_CONF=expandable_segments:True python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 64 --num_opt_steps 50 --motion_plan --save_plan /tmp/c_check.pkl
```
Then probe `/tmp/c_check.pkl`'s accel-channel tails.

## Risks / notes

- **C convention change:** Cartesian velocity moves from FD-of-pose to the
  analytic `J·q̇`, and all remaining finite differences move from forward to
  central — so values shift vs today across the whole trajectory, not just the
  tail. Strict accuracy improvement (exact velocity, no fabricated endpoints);
  fine for an offline reference plan. Only revisit if an MPC were numerically
  tuned against the *current* (buggy) arrays.
- **C1 frame/units:** confirm during implementation whether `tool_jacobians` is
  world- or link-frame and its row order (linear-first vs angular-first) so
  `J·q̇` lands in the WORLD frame the schema documents. The C1 analytic-FK-diff
  test catches a frame/order mismatch.
- **C4 may be upstream:** if native joint derivatives are dropped inside the
  trajopt-result path rather than `plan_processor`, the fix moves to
  `motion_solver.py` (still cuTAMP-side). If the drop is inside cuRobo itself,
  do **not** edit `curobo/` — fall back to C3's central diff and note it.
- **Data-changing:** unlike the sibling spec's B fix, this DOES change the saved
  pkl's velocity/acceleration values. No schema bump is required (field shapes
  and names are unchanged), but regenerate `data/motion_plan.pkl` after landing.
