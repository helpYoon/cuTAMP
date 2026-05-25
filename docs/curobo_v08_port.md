# Porting cuTAMP from cuRobo v0.7 → v0.8 (cuRoboV2)

This document tracks **what changed** when porting `cutamp/` from the v0.7-era
cuRobo bundled at `cuTAMP/curobo/` to v0.8.0+ (jumped on 2026-05-01). The
`main` branch is the v0.7 reference and "worked as intended"; the goal of the
port (`curobo_v2` branch) is to preserve that behavior with **no unnecessary
divergence**. Where we diverged, this doc explains why.

## TL;DR

- v0.8 broke most of cuRobo's public API: `MotionGen` → `MotionPlanner`,
  `plan_single` → `plan_pose`, `plan_single_js` → `plan_cspace`,
  `MotionGenConfig` → `MotionPlannerCfg`, `Pose` is now constructed via
  `GoalToolPose.from_poses({tool_frame: Pose(...)})`. Per-call knobs like
  `parallel_finetune`, `enable_finetune_trajopt`, and `time_dilation_factor`
  are gone.
- For T1, v0.8's `GoalToolPose` enabled folding both arms into a **single
  21-DOF MotionPlanner** (vs v0.7's two MotionGens, one per 11-DOF arm). This
  is the only architectural change forced by v0.8. Everything else is API
  shuffling or compensation for lost tuning knobs.
- The port traded several explicit per-call knobs for a new **outer retry**
  mechanism at the particle level (try the next-best satisfying particle if
  motion planning fails). This compensates for the per-call retry budget being
  effectively much smaller in v0.8.
- One real **regression** was caught and fixed: `MoveFree` / `MoveHolding` was
  left as a no-op during the port, which hid a discontinuity between arm
  switches that v0.7 handled implicitly via per-arm state managers. See
  § "Move ops are no longer no-ops" below.

---

## API rename map

| v0.7 (`main`) | v0.8 (`curobo_v2`) | Notes |
|---|---|---|
| `MotionGen` | `MotionPlanner` | Single-problem planner. `BatchMotionPlanner` for batched. |
| `MotionGenConfig.load_from_robot_config(...)` | `MotionPlannerCfg.create(robot=..., scene_model=...)` | Accepts dicts, paths, or already-constructed configs. |
| `motion_gen.plan_single(start_js, Pose, plan_config)` | `planner.plan_pose(GoalToolPose.from_poses({frame: Pose}), start_js, max_attempts=..., enable_graph_attempt=...)` | Per-call knobs reduced from ~10 to 4. |
| `motion_gen.plan_single_js(start_js, target_js, retract_plan_config)` | `planner.plan_cspace(target_js, start_js, max_attempts=..., enable_graph_attempt=...)` | Same shrink. Goal goes first now. |
| `motion_gen.plan_grasp(...)` | `planner.plan_grasp(GoalToolPose, start_js, ...)` | New "goalset → approach → grasp → lift" pipeline; no exposed `max_attempts`. |
| `motion_gen.attach_objects_to_robot(...)` | `planner.attachment_manager.attach_from_scene(...)` | New `AttachmentManager` class; auto-disables world obstacle on attach. |
| `motion_gen.detach_object_from_robot(...)` | `planner.attachment_manager.detach(link_name=...)` | Tracks only the **most recent** attach; multi-arm requires per-link bookkeeping. |
| `result.get_interpolated_plan()` | `result.interpolated_trajectory` | Direct field, not method. |
| `result.interpolation_dt` | `result.js_solution.dt` | Now lives on the JointState. |

A couple v0.8 quirks worth knowing:

- `MotionPlanner.attachment_manager` is a property that delegates to
  `trajopt_solver.attachment_manager` — but that delegation is **broken in
  v0.8**: the attribute lives on `trajopt_solver.core`. Our `_get_attachment_manager`
  helper walks the path manually.
- `planner.plan_cspace` always passes `finetune_attempts=1` to the underlying
  solver — there's no caller knob. (`plan_pose` bumps to 3 when graph-seeded;
  `plan_cspace` never does. We tested bumping it to 3; it didn't move our
  failure rate, so we left it alone.)

## Per-call knobs that no longer exist

These v0.7 `MotionGenPlanConfig` fields have no v0.8 equivalent on the
public API:

| v0.7 knob | What it did | What we lost |
|---|---|---|
| `parallel_finetune=True` | Ran final TrajOpt finetune passes in parallel | Some convergence robustness for trajopt |
| `enable_finetune_trajopt=True` | Whether to run finetune at all | v0.8 always runs ≥1 finetune pass; no caller control |
| `time_dilation_factor` | Scaled the trajectory time | Gone entirely; trajectories run at solver-chosen dt |
| `timeout=40.0` | Wall-clock cap per planning call | Gone; only `max_attempts` controls budget |

Net effect: failed retracts in v0.8 burn through their `max_attempts` budget
without the v0.7 tuning that helped marginal cases converge. We compensated
with the outer retry described below.

---

## T1 architectural change: dual MotionGen → single MotionPlanner

This is the **only architectural change forced by v0.8**.

### v0.7 (`main`)

T1 used **two MotionGens**, one per arm (11 DOF each), connected via a
`DualArmStateManager`:

- `state.left_motion_gen` and `state.right_motion_gen` — separate planners.
- `state.left_js`, `state.right_js`, `state.left_q_name`, `state.right_q_name` — separate state.
- `state.sync_shared_joints("left", "right")` — copied body+base joints from one arm's last_js to the other when switching arms.
- `state.get_state("right")` returned the **right arm's stored last_js** — meaning the right arm's pick planned from `right_q0` (or whatever was last set), not from left's last position.

This worked because v0.7's `plan_single` planned **one ee_link at a time**.

### v0.8 (`curobo_v2`)

v0.8's `GoalToolPose.from_poses({left_frame: ..., right_frame: ...})` lets a
single planner reason about multiple tool frames at once, so we collapsed the
two MotionGens into **one 21-DOF MotionPlanner**. New supporting code:

- `cutamp/t1_state.py` — `T1State` dataclass holding the single planner +
  current_js + per-arm holding dict.
- `T1State.pin_for_arm_action(active_arm)` — uses cspace target weights
  (`cspace_target_dof_weight`) to pin base + inactive arm during left/right
  arm operations. Inactive tool's pose criterion is **kept enabled** (its
  current FK pose is supplied as the IK target) — disabling caused 2+ rad
  of inactive-arm drift in our IK tests.
- `T1State.pin_for_movebase()` — pins both arms + body for `MoveBaseTo`.

### The cost: lost per-arm state tracking

The v0.8 single-planner has only **one** `last_js`. There's no
`state.get_state(other_arm)` returning the other arm's stored config —
because there's only one chain. Cross-arm transitions now flow through the
single `last_js`, which carries the *prior arm's* final body+base+arm config
into the *next arm's* first plan call.

This is what caused the "Move ops" regression below.

---

## ⚠️ `propagate_shared_joints` was load-bearing — and we removed it (REGRESSION)

This is the **biggest unintentional regression** in the port. Found 2026-05-04
after Fix A exposed it.

### What main did

`cutamp/particle_initialization.py:49 propagate_shared_joints(...)` — after
each IK solve for one arm, **copy the 4 shared body DOFs** (lift1, lift2,
torso_pitch, waist_yaw) from the active arm's solution into all of the
inactive arm's currently-unsolved configurations. Called from every
pick/place/retract IK branch (8 call sites).

This **coupled** the optimizer's q-variables across arms: left and right
configurations couldn't independently choose extreme body poses, because the
shared DOFs propagated from whichever arm solved IK first.

### What v0.8 port did

Removed it. The current docstring (`particle_initialization.py:14`) says:

> "The old `propagate_shared_joints` is gone — there are no shared 11-DOF
> representations to keep in sync. Downstream cspace pinning (in
> `cutamp/t1_state.py`) ensures inactive-arm joints don't drift during planning."

**That rationale is wrong.** The cspace pinning in `T1State.pin_for_arm_action`
is applied during **motion planning** (trajopt), not during particle
initialization. At particle init, each `q` still gets an independent IK
solution and is free to land at any extreme body+base pose. The pinning
controls drift *within a single trajectory*, not consistency *between* the
optimizer's chosen configs.

### Symptoms

When `MoveFree` ignores the optimizer's `q_end` (the no-op behavior we
restored), the next pick/place plans from `last_js` and finds its own IK
solution. The optimizer's extreme `q_end` is hidden, the pipeline runs.

When `MoveFree` plans to `q_end` (Fix A), the planner has to reach that
extreme `q_end` — typically 80cm of base translation, 90° base yaw, and
DOFs at the boundary of joint limits. With `CSPACE_MAX_ATTEMPTS=5` and
v0.8 defaults, those plans fail consistently across all candidate particles
in the satisfying set.

### Fix path

1. Restore a v0.8 analog of `propagate_shared_joints` in
   `cutamp/particle_initialization.py`. The "shared" DOFs in the new 21-DOF
   layout are the **7 body DOFs**: `BASE_INDICES (0-2) + LEG_INDICES (3-4) +
   TORSO_INDICES (5) + WAIST_INDICES (6)`. After each IK solve, copy this
   slice into all inactive-arm unsolved configs.
2. Once that lands, re-evaluate Fix A (plan MoveFree to `q_end`). With
   coupled body DOFs, the cspace deltas should be small enough for v0.8
   default budgets to converge.
3. Document the propagate-shared-joints behavior so it doesn't get removed
   again.

Until step 1 is done, **Fix A stays reverted**. The pipeline runs with the
original v0.7-equivalent behavior (MoveFree no-op, pick uses plan_grasp's
own IK), at the cost of motion-plan trajectories that don't match the
optimizer's particle q-values.

---

## Move ops are no longer no-ops (Fix A, 2026-05-04) — REVERTED 2026-05-04

### What `MoveFree` / `MoveHolding` did in v0.7

In v0.7, `MoveFree` / `MoveHolding` were no-ops in the motion-plan loop —
they just bumped `last_q_name`. The actual cross-action movement was handled
by:

1. The next `Pick`'s "**retract from previous position**" step — explicitly
   planned a `plan_single` from `start_js` up to `APPROACH_HEIGHT` above the
   previous EE pose, then planned approach to the new grasp.
2. The dual-arm `state.get_state(arm)` returning the **per-arm stored
   last_js** — so the right pick planned from `right_q0` (the right arm's
   stored start), not from `left_q2`.

Together these gave the appearance of MoveFree being "just bookkeeping," but
the actual continuity was load-bearing on (1) and (2).

### What broke in v0.8

The port carried `MoveFree` over as a no-op (`last_q_name = q_start`). But:

- The single-planner architecture removed (2) — there's no per-arm last_js
  to switch to.
- The `Pick` branch in v0.8 uses `plan_grasp` instead of v0.7's manual
  approach+grasp sequence, dropping (1) — `plan_grasp` plans `current_state →
  approach` directly without first lifting away from the previous grasp.

Result: at every cross-arm transition, the next pick planned from the prior
arm's last_js (e.g., `left_q2`), **ignoring** the optimizer's chosen
`right_q0` for that Move. The visualization showed the robot leaning into
weird body poses that the optimizer never had to reason about.

### The fix

`MoveFree` / `MoveHolding` now plans an **explicit cspace transition** to
the optimizer's intended `q_end`:

```python
elif metadata.is_motion and metadata.action_type is None:
    q_end_name = ground_op.values[-1]
    target_q = best_particle[q_end_name].clone()
    target_js = JointState.from_position(target_q[None])
    result = planner.plan_cspace(target_js, last_js, ...)
    ...
    last_js = _last_timestep_js(plan, planner)
    last_q_name = q_end_name
```

This is conceptually equivalent to v0.7's per-arm `state.get_state(arm)`
behavior — we explicitly transition to the next operator's expected start
configuration, so subsequent pick/place planning sees what the optimizer
expects.

### Why Fix A was reverted

When deployed, **all 5 outer-retry particles failed** because the optimizer's
`q_end` configurations were in extreme body poses (see "propagate_shared_joints
was load-bearing" above). The cspace deltas to reach those extremes were
80cm+ of base translation, 90°+ base yaw, and DOFs at joint-limit boundaries —
none of which converge in 5 attempts.

The bigger insight: the no-op behavior of `MoveFree` was hiding a real
optimizer issue. We can't safely re-introduce Fix A until shared-joint
propagation is restored.

---

## Retract budget: 240 → 5 + outer retry (2026-05-04)

### v0.7

```python
retract_plan_config = MotionGenPlanConfig(
    timeout=40.0,
    max_attempts=240,
    enable_graph_attempt=5,
    parallel_finetune=True,
    enable_finetune_trajopt=True,
    time_dilation_factor=config.time_dilation_factor,
)
```

Per-particle retract was hammered with up to 240 attempts (5 unseeded + 235
graph-seeded), parallel finetune passes, and explicit time dilation. Per-particle
success rate was high (~80% by analogous measurement on v0.8 with 240
attempts).

### Initial v0.8 port (which we then changed)

We initially carried over `CSPACE_MAX_ATTEMPTS=240, CSPACE_ENABLE_GRAPH_ATTEMPT=5`
to mirror v0.7. Per-particle success matched v0.7 (~80%). But:

- v0.8 lost `parallel_finetune`, `enable_finetune_trajopt`, `time_dilation_factor`.
- A failed retract burned ~3-4 minutes (240 × ~1s with finetune).
- We have no v0.7 baseline run on the same scene to A/B against — the user's
  recollection that v0.7 worked well is the only baseline.

### Current (intentional divergence from v0.7)

- `CSPACE_MAX_ATTEMPTS = 5` and `CSPACE_ENABLE_GRAPH_ATTEMPT = 1` —
  **mirror v0.8 defaults exactly** (1 unseeded attempt + 4 graph-seeded).
- New **outer retry**: `motion_plan_max_retries = 5` — when `solve_curobo`
  raises, retry with the next-best satisfying particle (see
  `cutamp/algorithm.py: get_top_satisfying_particles`).
- New **planner-state cleanup** at top of `solve_curobo`
  (`_reset_attachment_state`) — detaches both T1 attachment links and
  re-enables all movables, so a prior aborted run can't leave stale state.

Measured on `blocks_t1` at n=20:

| Config | Per-particle success | Effective success | Worst-case wall time |
|---|---|---|---|
| 240 attempts, 3 retries (legacy) | 80% | 95% | ~12 min |
| 5 attempts, 5 retries (current) | 45% | 95% | ~2 min |

Same end-to-end success rate, much better tail latency. **Per-particle
success dropped, which means v0.7's 240-attempt budget was doing real work** —
we just compensate at a different layer now.

### Symmetry consideration (not yet applied)

The same logic could extend to `plan_pose` (place) and `plan_grasp` (pick):
trim their attempt budgets to v0.8 defaults and rely on the outer retry. We
have **not** done this because we have no observed pick/place failures and
their failure cost is already much lower (~40s for pick, ~3min for place).
Worth considering for consistency if we ever observe issues there.

---

## Pick: `plan_grasp` replaces manual approach+grasp+lift

### v0.7

The Pick branch explicitly chained three `plan_single` calls:

1. **Retract**: `start_js → APPROACH_HEIGHT above start ee_pose`
2. **Approach**: `retract_js → APPROACH_HEIGHT above target grasp pose`
3. **End**: `approach_js → grasp pose`

Plus manual `attach_objects_to_robot` after the lift.

### v0.8

We use cuRobo's new `plan_grasp` API which internally does:

1. **Goalset**: IK to one of N grasp candidates → trajectory from `current_state` to selected approach pose
2. **Linear approach → grasp** (skipped if `plan_approach_to_grasp=False`)
3. **Linear grasp → lift** (skipped if `plan_grasp_to_lift=False`)

**Currently we skip steps 2 and 3** (`plan_approach_to_grasp=False, plan_grasp_to_lift=False`),
so only the goalset trajectory is used. Then `attach_from_scene` mounts the
block to the gripper via `AttachmentManager`.

### Divergence from v0.7

- v0.7 explicitly planned the linear approach→grasp; v0.8 currently does not.
  This is a TODO — see "Open question" in the Move ops section above.
- v0.7's "retract from previous position" before approach is gone (replaced
  by Fix A's explicit Move planning, which serves the same function but at a
  different layer).
- v0.7 wrapped its planning calls in plan_config with `max_attempts=120` and
  full finetune; v0.8 `plan_grasp` doesn't expose `max_attempts`. We added
  `GRASP_RETRY=4` outer-loop attempts in `motion_solver.py` to compensate.

---

## AttachmentManager (new in v0.8)

`attach_from_scene` does the principled thing in one call:

- Fits N spheres to the held object (we use `SphereFitType.VOXEL` with
  `num_spheres=6` for cube blocks).
- Mounts them on the configured `attached_object_{left,right}` link
  (50 reserved sphere slots in `t1_planar_base.yml`).
- **Auto-disables** the world obstacle so it isn't double-counted in collision.

`detach(link_name=...)` reverses both. We call it explicitly per-arm-link
because v0.8's manager only tracks **the most recent attach** in its
`_attached_link_name` field — multi-arm scenarios overflow that bookkeeping.

### Cleanup gotcha (handled in `_reset_attachment_state`)

If `solve_curobo` aborts mid-plan with both arms holding, the manager's
single `_attached_link_name` only points at the *most recent* attach. The
other arm's spheres stay on its link, and the other movable obstacle stays
disabled. The outer-retry cleanup walks both attachment links and re-enables
all movables to clear this.

---

## URDF refactor (separate from v0.8 port)

Not strictly v0.8-related, but landed in the same branch:

- `t1_simplified.urdf` lifting columns (`waist_lift_1`, `waist_lift_2`,
  `column_stage_*`) → serial revolute leg (`ankle_pitch`, `knee_pitch`).
- `t1_planar_base.yml` is the new canonical T1 config; the legacy
  `t1_left_11dof.yml` and `t1_right_11dof.yml` (one-arm-at-a-time configs)
  were deleted.
- Planar base remains via `extra_links.base_link_yaw.child_link_name = mobile_base_link`.

See `/home/yoonwoo/.claude/plans/shiny-riding-hummingbird.md` for the
detailed plan that drove this.

---

## Things we kept from v0.7 (intentional)

- Algorithm structure: `sample_initial_plans → optimize → curobo motion plan`.
- Cost functions, constraint checker, particle initialization architecture
  (single-MotionPlanner refactor preserved the public shape).
- `APPROACH_HEIGHT = 0.05` (used for v0.7's explicit retract; relevant again
  if we resurrect the linear approach→grasp).
- Visualizer (rerun) flow: `accum_plans` collected during planning, replayed
  frame-by-frame at end with `log_joint_trajectory`.

## Things to revisit (in priority order)

1. **Restore `propagate_shared_joints`** in `cutamp/particle_initialization.py`.
   This is the highest-priority regression. Until this lands, the optimizer's
   q-variables can have wildly inconsistent body+base configs, which forces
   us to keep MoveFree as a no-op (and hides the optimizer's intent from
   motion planning).
2. **Re-land Fix A** after shared-joint propagation works. Plan MoveFree/
   MoveHolding to `q_end` so cross-arm transitions follow the optimizer's
   chosen body+base+arm config.
3. **Pick should plan the linear approach→grasp** (v0.7 did; v0.8 skips it
   with `plan_approach_to_grasp=False, plan_grasp_to_lift=False`). Once
   Fix A is back, the linear segment becomes important so the full grasp
   is animated.
4. **Coordinate-frame issue** ("robot not reaching target") — separate from
   the discontinuity. Need to verify whether (a) `tool_from_ee` is correct,
   (b) the rerun viewer applies the planar-base SE(3) transform, and
   (c) the optimizer's IK target matches what `plan_grasp` sees.
5. **Symmetric attempt-budget cuts** for `plan_pose` (place) and `plan_grasp`
   (pick). Not blocking; would just make tail latency more uniform.
6. **MotionPlanner warmup** is called in `solve_curobo` (~6.5s one-off); when
   the outer retry triggers a fresh `solve_curobo`, do we re-warmup
   unnecessarily? Quick check warranted.
7. **`finetune_attempts` for cspace**: we left it at v0.8 default (1).
   Tested bumping to 3, no measurable improvement. Could reconsider if
   we ever revisit per-particle success.

---

## Per-file change inventory (vs `main`)

This table lists every file the `curobo_v2` working tree touches, the
**reason** for the change, and a **necessity rating**:

- **A** = forced by v0.8 (mechanical rename or API surface change)
- **B** = required by single-MotionPlanner T1 architecture
- **C** = required by URDF refactor (separate change, landed in same branch)
- **D** = our own behavior change (worth scrutinizing — may be unnecessary)

When auditing, focus on **D**-rated changes — these are where we may have
diverged from main without good reason.

### Modified files

| File | Reason | Rating |
|---|---|---|
| `cutamp/algorithm.py` | (1) `tensor_args` → `DeviceCfg` rename; (2) **D** added `get_top_satisfying_particles` + outer-retry loop wrapping `solve_curobo`; (3) tiny defensive `sat_list` list-coerce fix. | A + D |
| `cutamp/config.py` | **D** Added `motion_plan_max_retries: int = 5` field for the outer-retry loop. | D |
| `cutamp/cost_function.py` | (1) Self-collision cost imports/config moved (`SelfCollisionCostConfig` → `SelfCollisionCostCfg`); (2) collapsed two SelfCollisionCost instances (per-arm) into one for the unified planner; (3) renamed `is_dual_arm` → `is_multi_tool`; (4) renamed soft cost `minimize_lift_movement` → `minimize_body_movement` (lift columns gone, body slice now leg+torso). | A + B |
| `cutamp/costs.py` | (1) `curobo.types.math.Pose` → `curobo.types.Pose` import; (2) **D** rewrote `curobo_pose_error` to bypass `Pose.distance` (v0.8's `angular_distance_axis_angle` has a broadcast bug producing `[N,N]` instead of `[N]`). | A |
| `cutamp/envs/book_shelf.py` | `curobo.geom.types.Cuboid` → `curobo.scene.Cuboid`. | A |
| `cutamp/envs/stick_button.py` | Same import rename. | A |
| `cutamp/envs/tetris.py` | Same import rename + `TensorDeviceType` → `DeviceCfg`. | A |
| `cutamp/envs/utils.py` | Same import renames. | A |
| `cutamp/motion_solver.py` | **Largest diff (1232 lines).** Many concerns mixed in: (a) all v0.7 `motion_gen.plan_*` calls rewritten to v0.8 `planner.plan_pose/plan_cspace/plan_grasp` — A; (b) attachment via `AttachmentManager.attach_from_scene` — A; (c) `_get_attachment_manager` workaround for v0.8's broken `MotionPlanner.attachment_manager` property — A; (d) Pick now uses `plan_grasp` (drops v0.7's manual approach+grasp+lift) — A but with **D** caveat that we set `plan_approach_to_grasp=False, plan_grasp_to_lift=False` (skipping linear segments); (e) `_walk_trajectory_collisions` diagnostic walker — D; (f) `_log_cspace_failure_diagnostic` — D; (g) `_reset_attachment_state` for outer-retry cleanup — D; (h) `CSPACE_MAX_ATTEMPTS=5, ENABLE_GRAPH=1` to mirror v0.8 defaults — D (we were at 240 in main). | A + B + D |
| `cutamp/optimize_plan.py` | (1) **D** New `apply_grad_masks` + `build_grad_masks` integration via `cutamp/conf_locking.py`; (2) **D** "IK-init early-exit" — skip Adam when init particles already satisfy constraints (rationale in code: Adam's first step has uniform `lr` magnitude per active DOF, which destroys near-optimal inits); (3) `t1_home_left, t1_home_right` → unified `t1_home`. | B + D |
| `cutamp/particle_initialization.py` | (1) Single 21-DOF particle layout (was 11-DOF per arm) — B; (2) **⚠️ Removed `propagate_shared_joints`** — see top-priority regression section above. **This is the most consequential D-rated change in the port.** | B + **REGRESSION** |
| `cutamp/robots/__init__.py` | Export updates for new T1 single-planner API. | A + B |
| `cutamp/robots/franka.py` | `MotionGenConfig.load_from_robot_config(...)` → `MotionPlannerCfg.create(...)` etc. Pure API rename. | A |
| `cutamp/robots/ur5.py` | Same. | A |
| `cutamp/robots/t1.py` | (1) Single 21-DOF MotionPlanner construction — B; (2) URDF-DOF and CUROBO-DOF constants for new leg/torso layout — C; (3) `t1_home`, `BASE_INDICES`, `LEG_INDICES`, `TORSO_INDICES` etc. — B/C; (4) `T1RerunRobot` viz helper updates — B. | B + C |
| `cutamp/robots/assets/t1_description/t1_simplified.urdf` | URDF refactor: lifting columns → serial revolute leg (ankle_pitch + knee_pitch). | C |
| `cutamp/robots/assets/t1_description/t1_spheres.yml` | Sphere updates for new leg + tweaked radii. | C |
| `cutamp/rollout.py` | Single-MotionPlanner kinematics. | B |
| `cutamp/samplers.py` | `obj.tensor_args` → `obj.device_cfg` everywhere; `Cuboid` import path. | A |
| `cutamp/scripts/gripper_sphere_editor.py` | Default link name list updated for new leg layout. | C |
| `cutamp/scripts/robot_sphere_editor.py` | Same. | C |
| `cutamp/scripts/run_cutamp.py` | Soft-cost choices renamed (`minimize_lift_movement` → `minimize_body_movement`); minor flag wiring. | A + C |
| `cutamp/scripts/utils.py` | Trivial import. | A |
| `cutamp/t1_domain.py` | T1 domain operators with generic `q`/`q_start`/`q_end` parameter names instead of arm-prefixed. | B |
| `cutamp/tamp_world.py` | `MotionPlannerCfg.create(...)` instead of v0.7's `get_motion_gen` per-arm pattern. Single planner factory. | A + B |
| `cutamp/task_planning/base_structs.py` | Trivial. | A |
| `cutamp/tests/conftest.py` | Updated fixtures for new APIs. | A + B |
| `cutamp/tests/test_t1_config.py` | Updated assertions for new 21-DOF layout. | B + C |
| `cutamp/tests/test_t1_robot_module.py` | Same. | B + C |
| `cutamp/utils/collision.py` | Collision API updates for v0.8. | A |
| `cutamp/utils/common.py` | Minor v0.8 import. | A |
| `cutamp/utils/rerun_utils.py` | Minor. | A |
| `cutamp/utils/shapes.py` | Minor v0.8 imports. | A |
| `cutamp/utils/visualizer.py` | Minor. | A |

### Deleted files

| File | Reason | Rating |
|---|---|---|
| `cutamp/robots/assets/t1_description/t1_left_11dof.yml` | Superseded by `t1_planar_base.yml` (single 21-DOF unified config). | B |
| `cutamp/robots/assets/t1_description/t1_right_11dof.yml` | Same. | B |
| `cutamp/tests/test_shared_joint_consistency.py` | Test for `propagate_shared_joints` — deleted along with the function. **This deletion is suspect**: the function may need to be restored (see top-priority regression), in which case this test should also be revived. | **D / suspect** |
| `cutamp/tests/test_tamp_world_dual_arm.py` | Tested the dual-MotionGen `DualArmStateManager` which doesn't exist in v0.8 architecture. Replacing with single-planner equivalent tests is a TODO. | B |

### New files

| File | Purpose | Rating |
|---|---|---|
| `cutamp/conf_locking.py` | **D** New per-conf gradient masking: hard-zeros gradients on DOFs that should not move during a given operator (e.g., base + inactive arm during arm ops). Rationale: Adam's first step has uniform `lr` magnitude per active DOF; without locking, the optimizer drifts these DOFs even though they shouldn't move. | D |
| `cutamp/grasp_planning.py` | Thin wrappers `plan_single_arm_grasp` / `plan_single_arm_pose` around v0.8 `MotionPlanner.plan_grasp` / `plan_pose`. The grasp wrapper sets up cspace pinning before calling. | A + B |
| `cutamp/t1_state.py` | T1's single-MotionPlanner state container: holds the planner, current_js, per-arm holding dict; provides `pin_for_arm_action` / `pin_for_movebase` cspace-weight helpers. | B |
| `cutamp/robots/assets/t1_description/t1_planar_base.yml` | The new canonical T1 robot config: single 21-DOF MotionPlanner, planar base via `extra_links`, 50 reserved sphere slots per attached_object_{left,right} link. | B + C |
| `cutamp/robots/assets/t1_description/actual_robot.urdf` | Reference: real Booster T1 URDF (used during the URDF refactor for joint-limit values; not loaded at runtime). | C |
| `cutamp/robots/assets/t1_description/actual_meshes/` | Reference real-robot STL meshes; not loaded at runtime. | C |
| `cutamp/dataset/` | Unrelated to the v0.8 port — dataset-collection/storage scaffolding. Likely user's separate work; should not block port review. | unrelated |

### Audit checklist (D-rated changes worth reviewing)

1. **`cutamp/particle_initialization.py`** — confirm whether removing
   `propagate_shared_joints` was justified, or restore it (see top-priority
   regression section).
2. **`cutamp/algorithm.py`** — outer retry loop. Doesn't exist in main but
   compensates for our smaller per-call attempt budget. Is the trade-off
   the right one?
3. **`cutamp/config.py: motion_plan_max_retries`** — pairs with above.
4. **`cutamp/motion_solver.py`** — multiple D-rated additions; the
   diagnostic walker and cspace failure logger could be removed if/when
   the underlying issues are resolved.
5. **`cutamp/optimize_plan.py`** — IK-init early-exit and gradient masking.
   Both are real changes from main's behavior; need the user's eye on
   whether they preserve main's solution quality.
6. **`cutamp/conf_locking.py`** — entirely new module. Its policy
   (lock base + inactive arm during arm ops; lock body + arms during
   navigate) approximates main's per-arm IK behavior at optimizer time.
   Worth confirming it actually does so.
7. **`cutamp/costs.py: curobo_pose_error`** — rewritten to work around a
   v0.8 `Pose.distance` bug. Verify the rewrite is correct (single-axis
   rotation tests).
8. **`cutamp/tests/test_shared_joint_consistency.py`** — deletion may
   need to be reverted depending on (1).

---

## Pointers in code

- `cutamp/motion_solver.py`: the per-operator dispatch (navigate, retract,
  pick, place, MoveFree/MoveHolding) is in `solve_curobo`.
- `cutamp/t1_state.py`: T1's single-MotionPlanner state container, cspace
  pinning helpers.
- `cutamp/algorithm.py`: `get_top_satisfying_particles` and the outer-retry
  loop wrapping `solve_curobo`.
- `cutamp/grasp_planning.py`: `plan_single_arm_grasp` / `plan_single_arm_pose`
  thin wrappers around v0.8 `MotionPlanner.plan_grasp` / `plan_pose`.
- `cutamp/robots/assets/t1_description/t1_planar_base.yml`: the canonical
  v0.8 T1 robot config (single 21-DOF MotionPlanner, planar base via
  `extra_links`).
- `cuTAMP/curobo/CHANGELOG.md`: v0.8 changelog from upstream.
