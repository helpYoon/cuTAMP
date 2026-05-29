# COM-polygon & MPC-consumer review fixes — design

**Date:** 2026-05-29
**Status:** approved-pending-review
**Branch:** `curobo_v2`

## Goal

Fix the five distinct root causes surfaced by the `/code-review max` pass over
the committed COM-over-base-polygon work plus the uncommitted MPC plan
consumer. No new features — only correctness fixes, a small refactor to remove
duplicated COM constants, and doc/comment corrections.

## Constraints

- **No edits to `curobo/`** (vendored, read-only; would need a fork). Every
  change here lives in `cutamp/`, `examples/`, or `docs/`. None require a fork.
- Tests run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Stage only specific files; no `git add -A`.

## Scope

Five root causes, each independently landable:

| ID | Root cause | Review findings | Severity |
|----|-----------|-----------------|----------|
| A  | COM penalty/mask/tol disagree near support-rectangle corners | 3, 4, 7, 14 | high — silently drops COM-feasible particles |
| B  | 0.0625 m trunk-X offset is now dead but still referenced | 1, 5, 6 | high — misleads MPC integrator (now: fully purge) |
| C  | Acceleration is a double forward-difference; last 2 samples collapse to ~0 | 8, 13 | medium — wrong feedforward at segment boundaries |
| D  | numpy aliasing — missing `.copy()` on shared views | 9, 10 | low — defensive |
| E  | misc hardening (priority sentinel, IK-seed clamp, None-cost, schema diag) | 11, 12, 15, 2 | low |

Out of scope: behavior changes beyond these fixes, performance work, the
COM-cost-on-IK / coupled-reik / unified-IK plans (separate specs).

---

## A — Unify the COM penalty, mask, and tolerance (findings 3, 4, 7, 14)

### Problem

`com_polygon_penalty` (`cutamp/com_polygon_cost.py:90`) sums the inside-margin
barrier over **both** axes:

```python
return (outside ** 2).sum(dim=-1) + inside_weight * (inside ** 2).sum(dim=-1)
```

The hard-constraint tolerance `4e-4` (`cutamp/scripts/utils.py:75`) was
calibrated as a **single-axis** edge penalty: `inside_weight · inside_margin²
= 1 · 0.02² = 4e-4`. Summing double-counts near a corner, so a COM that is
geometrically inside both edges by less than ~5.86 mm scores `> 4e-4` and is
marked **violating** by `ConstraintChecker`, even though it is on/inside the
support rectangle.

Compounding it, `compute_com_polygon_mask`
(`cutamp/com_polygon_cost.py:117-172`) uses a *different*, margin-free test
`abs(com_in_base) <= half`, so the IK COM-safe retry loop
(`cutamp/particle_initialization.py:183`) sees the near-corner config as safe,
stops retrying, and hands downstream a particle the hard constraint then drops.

Finding 14: the half-extents are documented three different ways —
`ComOverBasePolygonCostCfg` default/docstring says `[0.10, 0.15]`
(`com_polygon_cost.py:39, 55`), the mask docstring says `[0.05, 0.10]`
(`:137`), while the live defaults say `[0.1115, 0.156]` (`:151`, `:218`). The
authoritative value is **`[0.1115, 0.156]`** (two-foot support hull per
`actual_robot.urdf`).

### Fix

**A1 — Max-over-axes inside barrier (geometry fix).** Change the inside term
from `.sum(dim=-1)` to `.max(dim=-1).values`; leave the outside term summed
(it is squared Euclidean distance out of support — correct).

```python
return (outside ** 2).sum(dim=-1) + inside_weight * (inside ** 2).max(dim=-1).values
```

The inside barrier now measures distance to the *nearest* edge. A corner scores
`w · margin² = 4e-4 = tol`, identical to a single edge → passes. This one change
fixes the geometry in the cost, the mask, and the penalties-helper, since all
three call `com_polygon_penalty`.

Continuity / correctness checks (must hold; encode as tests):
- COM strictly inside the inset rectangle → penalty `0`.
- COM exactly on a single edge → penalty `4e-4` (= tol, satisfies).
- COM exactly on a corner → penalty `4e-4` (was `8e-4`; now satisfies).
- COM 1 mm outside one edge → penalty `≈ 4.42e-4` (> tol, fails).
- With max-over-axes, `penalty ≤ tol` is mathematically equivalent to
  `abs(com_in_base) ≤ half` (inside-or-on the rectangle). This equivalence is
  what makes A2 safe.

**A2 — Route the mask through the penalty (unify the gate).** Replace the
inline `abs(com_in_base) <= half` in `compute_com_polygon_mask` with a call to
`com_polygon_penalty(...) <= COM_TOL` (reusing the com_world / base_T it already
computes at `:162-167`). The IK COM-safe retry loop and `ConstraintChecker`
then evaluate the *identical* function — they cannot diverge.

**A3 — Single source of truth for the constants (resolves finding 14).** Add
module-level constants to `cutamp/com_polygon_cost.py`:

```python
COM_HALF_EXTENTS = (0.1115, 0.156)   # two-foot support hull per actual_robot.urdf
COM_INSIDE_MARGIN = 0.02
COM_INSIDE_WEIGHT = 1.0
COM_TOL = 4e-4                        # = COM_INSIDE_WEIGHT * COM_INSIDE_MARGIN**2
```

Point every site at them:
- `ComOverBasePolygonCostCfg.__post_init__` default (`:55`) → `list(COM_HALF_EXTENTS)`.
- `compute_com_polygon_mask` default (`:151`) → `COM_HALF_EXTENTS`.
- `compute_com_polygon_penalties` defaults (`:218`, and the `inside_margin` /
  `inside_weight` kwarg defaults) → the constants.
- `cutamp/scripts/utils.py:75` ComPolygon tol → import and use `COM_TOL`.
- `cutamp/robots/t1.py` ComOverBasePolygonCostCfg instantiations (planner-side
  and IK-side) → reference the constants for `half_extents`, `inside_margin`,
  `inside_weight`.

**A4 — Doc/comment fixes.** Update the tol comment
(`cutamp/scripts/utils.py:69-75`): with max-over-axes, "at the edge or inside
satisfies" is now true at corners too. Fix the three stale half-extents
docstrings (`com_polygon_cost.py:39`, `:137`, `:55`) to `[0.1115, 0.156]`.

### Files

- `cutamp/com_polygon_cost.py` — add constants; `com_polygon_penalty` inside
  term → max; `compute_com_polygon_mask` → penalty ≤ tol; defaults reference
  constants; docstrings.
- `cutamp/scripts/utils.py` — tol references `COM_TOL`; comment.
- `cutamp/robots/t1.py` — cost cfgs reference constants.
- `cutamp/tests/test_com_polygon_ik.py` — see Tests.

### Tests

- Two-axis-corner parity (new): COM ~1 mm inside both edges → assert
  `compute_com_polygon_mask == True`, `penalty ≤ COM_TOL`, and a
  `ConstraintChecker` built with `COM_TOL` marks it satisfied. This is the
  case the old code got wrong.
- Single-edge boundary: COM on one edge → penalty `== COM_TOL` (within fp tol).
- Just-outside: COM 1 mm past one edge → penalty `> COM_TOL`, mask `False`.
- Mask/constraint equivalence: random COMs → `mask` agrees with
  `(penalty ≤ COM_TOL)` elementwise.
- Update any existing test asserting the old summed-corner value.

---

## B — Purge the 0.0625 m trunk-X offset entirely (findings 1, 5, 6)

### Problem

The 0.0625 m Trunk-origin difference between sim `t1_simplified.urdf` and
`actual_robot.urdf` is **no longer real**: the MPC-side `actual_robot.urdf` was
modified so its Trunk origin matches sim. So sim Trunk FK = real Trunk pose,
and nobody needs to compensate. But offset references survive and contradict
each other:

- `cutamp/utils/plan_processor.py` docstring lines 8-12 (v3 note: "consumers
  using actual_robot.urdf must subtract 0.0625") and lines 92-95 ("offset
  already subtracted … no compensation needed downstream") — contradictory and
  both now wrong.
- `cutamp/utils/plan_processor.py:312-314` — dead comment "Apply -0.0625 X
  compensation" with no code beneath it (v3 removed the code).
- `examples/load_motion_plan_for_mpc.py:14-21` — header tells consumers to
  subtract 0.0625.
- `docs/sim_to_real_mapping.md` — TL;DR line 18 ("no offset needed") vs section
  1 lines 28-55 ("must subtract 0.0625"); line 36 literally asks "which is it?".

v3 already stores raw sim Trunk FK (`plan_processor.py:319`), which equals the
real Trunk pose under the modified URDF. **The stored data does not change** —
only the offset language is removed. Schema stays v3 (no bump).

### Fix

Remove all offset references; state plainly that sim Trunk pose = real Trunk
pose, directly usable.

- `cutamp/utils/plan_processor.py`:
  - Docstring lines 8-12: drop the offset note → "v3 stores the raw sim Trunk
    world pose; `actual_robot.urdf` now shares the sim Trunk origin, so it is
    directly usable with no compensation."
  - Docstring lines 92-95: remove "offset already subtracted / no compensation
    needed"; replace with "Trunk world pose is directly usable on the real
    robot (URDFs share the Trunk origin)."
  - **Keep** the separate fix to lines 99-102: the module *does* emit
    `ankle_pitch` / `knee_pitch` (lines 326-327), so the "we don't emit leg
    joints" claim must become "emits ankle_pitch and knee_pitch (broadcast to
    both sides)". This is unrelated to the offset but is part of finding 5.
  - Delete the dead comment at lines 312-314; clean the "real-URDF-native"
    label at 318 (now genuinely true, no offset caveat).
- `examples/load_motion_plan_for_mpc.py`: delete the header paragraph
  (lines 18-21) instructing the 0.0625 subtraction. The recipe already copies
  `trunk_xyz` raw → correct as-is. Do **not** add a `_TRUNK_X_OFFSET` constant.
- `docs/sim_to_real_mapping.md`: rewrite section 1 (lines 28-39) to state the
  URDFs now share the Trunk origin (actual_robot.urdf modified MPC-side);
  remove the "which is it?" contradiction; clean TL;DR line 18; update the
  section-1 body lines 48-55.

### Files

- `cutamp/utils/plan_processor.py`
- `examples/load_motion_plan_for_mpc.py`
- `docs/sim_to_real_mapping.md`

### Tests

Doc/comment-only; no behavior change. Verify by `grep -rn 0.0625 cutamp
examples docs` returning **only** the lines that *quote the URDF joint origins*
for explanation (`docs/sim_to_real_mapping.md` lines like
`<origin xyz="0.0625 0 -0.1155"/>` — those remain true and load-bearing). No
match may state the offset as a *required consumer action* or *applied
compensation*. (Do not blindly drive grep to zero — that would delete real
explanation of why the sim joint origin is shaped the way it is.)

---

## C — Derive plan velocities/accelerations correctly (findings 8, 13)

### Problem

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
  firing when it shouldn't. (See C3.)

### Fix

Three sub-fixes — Cartesian via Jacobian, joints prefer-native, and the
underlying finite-difference made endpoint-correct.

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

### Files

- `cutamp/utils/plan_processor.py` — `compute_jacobian=True` in
  `_build_processing_kinematics`; `J·q̇` Cartesian velocity (C1); accel from
  central diff of it (C2); `_forward_finite_diff` → `np.gradient` (C3); remove
  `_angular_velocity_from_quat` if fully superseded by C1.
- `cutamp/motion_solver.py` and/or the trajopt-result path — only if C4 finds
  the derivative drop there (scope confirmed during investigation).

### Tests (new test module or extend an existing one)

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

---

## D — Add defensive `.copy()` to shared numpy views (findings 9, 10)

### Problem

In `examples/load_motion_plan_for_mpc.py`, `segment_to_mpc_commands` hands out
**views** into the pickled segment arrays for `trunk_world_pose["xyz"]` (copies
the raw `P["trunk_xyz"]`) and the 14 per-arm columns (`P[field][:, i]`), while
deliberately `.copy()`-ing the broadcast joints just above. Any in-place edit
downstream would silently mutate the shared segment. (Finding 9's specific
"-= 0.0625" trigger is moot after B, but the general footgun remains.)

### Fix

Add `.copy()` to the trunk-xyz assignment and each per-arm column assignment so
every emitted array is independent, matching the broadcast-joint handling.

### Files

- `examples/load_motion_plan_for_mpc.py`

### Tests

- Mutate a returned command array in place; assert the source segment dict is
  unchanged.

---

## E — Misc hardening (findings 11, 12, 15, 2)

### E1 — Arm-affinity priority sentinel (finding 11)

`make_arm_affinity_priority_fn` (`cutamp/algorithm.py:98-113`) returns `0.0` for
non-pick / unresolved-block / no-pose branches, but resolved picks return a
strictly-positive distance. With ascending BFS sibling sort, `0.0` sorts these
sentinels **first**, contradicting the docstring's "cross-body groundings
enumerated just later". **Fix:** return `float('inf')` for the sentinel
branches so they sort after real-metric picks. **Test:** a mixed sibling list
(resolved pick + non-pick) sorts the resolved pick first.

### E2 — Clamp IK seed to joint limits (finding 12)

The noise-perturbed IK seed (`cutamp/particle_initialization.py:197-206`,
`noise_std = 0.1·(attempt+1)`) is fed to cuRobo without re-clamping, so
large-σ retries can hand an out-of-limits seed. **Fix:** `torch.clamp` the
perturbed seed per-DOF to the robot's joint limits before `_ik_for_pose`
(limits sourced from `world.ik_solver.kinematics`; exact accessor pinned in the
plan). **Test:** a seed pushed past a known limit comes back clamped within
limits.

### E3 — None-cost guard in CostReducer (finding 15)

`CostReducer.get_cost` (`cutamp/cost_reduction.py:42`) initializes `cost = None`
and returns `None` when no entry matches `consider_types`; the optimizer does
`hard_costs(...) + soft_costs(...)`, which would raise `TypeError` if a side is
empty. **Fix:** initialize to a zero tensor (shaped per num_particles) so an
empty side yields a zero objective. **Note:** the review's claim that
`soft_costs` defaults to `("constraint",)` is **wrong** — the code correctly
uses `{"cost"}` / `{"constraint"}` (`:56-62`); no change there. **Test:** a
cost_dict with only one type returns a finite zero-or-value tensor, not None.

### E4 — Schema-version diagnostic (finding 2)

`examples/load_motion_plan_for_mpc.py` defaults a missing `schema_version` to
`1`, so a keyless pickle reports "got 1" (misleading). **Fix:** default to a
sentinel and emit a distinct "schema_version key absent" vs "wrong version"
message. Lowest priority. **Test:** a dict with no `schema_version` raises the
"absent" message, not "got 1".

### Files

- `cutamp/algorithm.py` (E1)
- `cutamp/particle_initialization.py` (E2)
- `cutamp/cost_reduction.py` (E3)
- `examples/load_motion_plan_for_mpc.py` (E4)

---

## Testing strategy

- Unit tests per cause as listed; all run under
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- End-to-end smoke after A (the behavioral one):
  ```
  PYTORCH_ALLOC_CONF=expandable_segments:True python -m cutamp.scripts.run_cutamp \
    --env blocks_t1 --disable_visualizer -n 64 --num_opt_steps 50 --motion_plan
  ```
  Pass: ≥1 satisfying, and near-corner particles that were dropped before are
  now retained (satisfying count ≥ pre-fix; plan regenerates clean).
- `grep -rn 0.0625 cutamp examples docs` → zero matches (B done).

## Sequencing

Independent commits, suggested order:

1. **A** — constants + max-over-axes + mask unification + tests (core behavior).
2. **E3** — cheap None-cost safety.
3. **B + D** — same file region (`examples/`, `plan_processor.py`, docs).
4. **C** — finite-diff tail.
5. **E1, E2, E4** — remaining hardening.

Land as one PR with logical commits, or split A out — decide at finishing.

## Risks / notes

- **A behavior change:** max-over-axes changes the near-corner *soft*-cost
  gradient — the inside barrier now pulls toward the nearest edge rather than
  both axes. This is the intended Option-1 tradeoff (geometrically correct
  distance-to-nearest-edge) and only differs within the 2 cm corner band.
  `torch.max` introduces a subgradient kink along the corner diagonal; autograd
  routes the gradient to the argmax axis, which is fine.
- **A equivalence:** after A1, `penalty ≤ COM_TOL` ⇔ `abs(com) ≤ half`
  (inside-or-on), so A2 cannot change which configs the mask accepts vs the
  pre-fix geometric test — it only guarantees the mask and the hard constraint
  use one function.
- **C convention change:** velocities/accelerations differ from today only in
  the final 1-2 samples (interior forward-diff unchanged). Acceptable for an
  offline plan.
- **B is data-neutral:** stored pickle bytes are unchanged; only docs/comments
  change. No schema bump.
