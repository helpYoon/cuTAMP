# Code review fixes — design

**Date**: 2026-05-23
**Scope**: Address all 15 findings from the `/code-review` max-effort pass on the `curobo_v2` branch as one bundled change, organized as 5 thematic commits.

## Context

A max-effort code review (5 finder angles + verification + sweep) on the in-progress cuRobo v0.7 → v0.8 port produced 15 distinct defects spanning hard crashes, silent correctness bugs, latent structural fragility, a frame-mismatch in the MPC consumer schema, and a security/UX gap in the consumer-facing pickle loader.

The user elected to fix all 15 in one bundle, split into 5 thematic commits so each is reviewable in isolation and bisection narrows a regression to one commit. Each commit ends with a green smoke test run before moving on.

## The 15 findings (severity-ordered, from the review)

`Label` column matches the original finder-agent codes used in commit headings below (`A1` = Angle A finding 1, `S2` = Sweep finding 2, etc.).

| # | Label | Commit | File | Summary |
|---|---|---|---|---|
| 1 | A1 | 1 | `optimize_plan.py:240` | `_refresh_ik_deps` passes `current_state_q` to `_ik_for_pose` which doesn't accept it → `--coupled_reik` always TypeErrors |
| 2 | S2 + S1 | 1 | `algorithm.py:220` + `cost_function.py:387` | `resample_plan_info` calls `cost_fn(rollout)` with no particles; sibling at line 160 passes them. Combined with `cost_function.py:387`'s `if self.retract_info and self._particles is not None` guard, this silently drops retract collision AND hard-raises if soft cost needs particles. |
| 3 | S3 | 2 | `t1_simplified.urdf:219` | Trunk inertial origin missing the `-0.0625` X compensation that visual/collision/spheres all carry |
| 4 | S4 | 2 | `_curobo_internals.py:264` | `cc.add(value, name)` omits the `weight` kwarg → `CostCollection.weights` falls one short of `.values` once any extra cost is registered |
| 5 | F1 | 3 | `plan_processor.py:104` | Saves `*_base_link` (palm) poses under `*_hand_xyz` keys; `actual_robot.urdf` has only `*_hand_link` |
| 6 | E4 | 5 | `load_motion_plan_for_mpc.py:130` | Broadcast joints alias the same numpy array to `Left_Hip_Pitch` AND `Right_Hip_Pitch` |
| 7 | F5 | 2 | `tamp_world.py:286` | `get_motion_planner` doesn't forward `enable_com_polygon` to `get_t1_motion_planner` |
| 8 | E3 | 5 | `tests/debug_t1_tool_frame.py:87` | Imports stale symbols (`get_t1_kinematics_model`, `t1_home_left`, `load_t1_rerun(arm=…)`) |
| 9 | D3 | 3 | `plan_processor.py:179` | `_angular_velocity_from_quat` produces spurious huge omega on quaternion sign flips |
| 10 | B2 | 5 | `_curobo_internals.py:74` | `cspace_plan_succeeded` squeeze loop's `> 2` should be `>= 2` |
| 11 | D2 | 3 | `plan_processor.py:153` | `_to_active_cspace` skips reorder when DOF count matches, ignoring joint-name order |
| 12 | C3 + S5 | 4 | `t1_state.py:110` | `_apply_pin` is non-atomic; tool-pose criteria leak on raise. `_disabled_tool_pose_frames` bookkeeping set by caller AFTER the planner mutation makes the recovery path worse. |
| 13 | F2 | 3 | `load_motion_plan_for_mpc.py:78` | `compensate_trunk_x_offset` mutates in place with no idempotency guard |
| 14 | F4 | 5 | `load_motion_plan_for_mpc.py:170` | `pickle.load` only catches `FileNotFoundError`; RCE risk + bad UX on corrupt pickle |
| 15 | C2 | 4 | `t1_state.py:157` | `pin_for_movebase` lacks the "already-pinned" guard `pin_for_arm_action` has |

Bonus (in scope, not from the top-15): the URDF Y-asymmetry between `left_base_link` (Y = 0.080) and `right_base_link` (Y = -0.084) lands in Commit 2 alongside the inertial fix.

## Design decisions (locked in)

- **F1 (hand frame)**: Switch the FK target in `plan_processor.py` from `*_base_link` to `*_hand_link`. Both URDFs have this link → consumer needs no extra transforms. Existing `motion_plan.pkl` files become invalid (intentional breaking change).
- **A1 (coupled_reik)**: Fix the API properly + add a tiny smoke test so the feature can't silently re-break.
- **F4 (pickle.load)**: Broaden the except clause to catch corrupt/truncated/version-mismatch errors with friendly messages; add a `# WARNING: pickle is unsafe with untrusted files` block at the load site and a paragraph in `docs/sim_to_real_mapping.md`. Keep the format (numpy/tensor round-trip stays simple).
- **S4 (cc.add weight)**: Investigate cuRobo's built-in cost dispatch (`RobotCostManager.compute_costs`) for the pattern used by `tool_pose` / `cspace`, then mirror it in our wrapper. Investigation runs as part of Commit 2; the design doesn't commit to a specific value yet.
- **URDF Y asymmetry** (related to F1): While in the URDF, also fix the `*_base_link` joint Y offset asymmetry — `left_base_link` is `0 0.08 0`, `right_base_link` is `0 -0.084 0`. Make both symmetric at `0.084`. After F1 lands, the saved trajectory no longer references `*_base_link`, but the asymmetry still affects internal grasp planning + palm collision spheres.
- **C3/S5 (pin atomicity)**: Move `_disabled_tool_pose_frames` bookkeeping inside `_apply_pin` BEFORE the `update_tool_pose_criteria` call so `unpin` can always recover any partial mutation. Don't add full atomic rollback yet — the trigger is currently latent and over-engineering risks.
- **F5 (enable_com_polygon)**: Expose via both `TAMPConfiguration.enable_com_polygon: bool = True` and CLI flag `--no_enable_com_polygon` (store_false).
- **Commit granularity**: 5 thematic commits (one per finding category). Each commit reviewable in isolation; smoke test runs between commits.

---

## Commit 1 — `coupled_reik` + resample correctness (A1, S1+S2)

**Files**:
- `cutamp/particle_initialization.py`: extend `_ik_for_pose` signature.
- `cutamp/optimize_plan.py`: verify `_refresh_ik_deps` call site matches new signature; remove the kwarg-passing bug.
- `cutamp/algorithm.py:220`: pass `plan_particles` to `cost_fn(rollout, plan_particles)`.
- `cutamp/tests/test_coupled_reik_smoke.py` (NEW, ~30 LOC).

**Changes**:

1. `particle_initialization._ik_for_pose` — extend signature:
   ```python
   def _ik_for_pose(
       world: TAMPWorld,
       world_from_ee: torch.Tensor,
       arm: Optional[str],
       *,
       current_state_q: Optional[torch.Tensor] = None,
       num_seeds: Optional[int] = None,
   ):
   ```
   When `current_state_q` is provided: reorder it to IK joint order and use it as the `current_state` instead of broadcasting `world.q_init`. When `num_seeds` is provided: override the default seed count for warm-start IK (so re-IK refresh runs single-seed and is ~10–20× faster). Existing callers (passing nothing) are unchanged.

2. `algorithm.py:220` — `resample_plan_info`:
   ```python
   # Before:
   cost_dict = plan_info["cost_fn"](rollout)
   # After:
   cost_dict = plan_info["cost_fn"](rollout, plan_particles)
   ```
   Fixes both the silent retract-collision drop (finding S1: when `_particles is None`, the retract collision branch is skipped via `if self.retract_info and self._particles is not None`) AND the hard `RuntimeError("Particles must be provided for ... soft cost")` that fires whenever any soft cost in `{retract_close_to_home, minimize_body_movement, com_polygon}` is active.

3. New smoke test `tests/test_coupled_reik_smoke.py`:
   ```python
   def test_coupled_reik_smoke():
       """Asserts --coupled_reik can run a few Adam steps without crash."""
       config = TAMPConfiguration(robot="t1", env="blocks_t1",
                                  num_particles=8, num_opt_steps=5,
                                  coupled_reik=True, reik_interval=2,
                                  disable_visualizer=True, ...)
       run_cutamp(config)  # raises if --coupled_reik path TypeErrors
   ```
   Marked `@pytest.mark.slow` if it's >30s; otherwise unmarked.

**Success criteria**: smoke test passes; `python -m cutamp.scripts.run_cutamp --coupled_reik ...` runs to completion without TypeError.

---

## Commit 2 — URDF + cost dispatch correctness (S3, S4, F5, URDF Y-symm)

**Files**:
- `cutamp/robots/assets/t1_description/t1_simplified.urdf`: Trunk inertial X + `left_base_link` joint Y.
- `cutamp/_curobo_internals.py:264`: pass correct weight to `cc.add`.
- `cutamp/config.py`: add `enable_com_polygon` field.
- `cutamp/scripts/run_cutamp.py`: add `--no_enable_com_polygon` CLI flag.
- `cutamp/tamp_world.py`: accept + forward `enable_com_polygon` in `get_motion_planner`.

**Changes**:

1. **S3 — Trunk inertial origin** (`t1_simplified.urdf:219`):
   ```xml
   <!-- Before: -->
   <origin xyz="-0.0073634598906924 -1.42058017623659E-06 0.105062332707657" rpy="0 0 0" />
   <!-- After (X minus 0.0625): -->
   <origin xyz="-0.0698634598906924 -1.42058017623659E-06 0.105062332707657" rpy="0 0 0" />
   ```
   Now visual (line 225), collision (line 234), AND inertial all share the `-0.0625` X compensation that the doc comment at lines 212-216 already promises.

2. **URDF Y-symm** (`t1_simplified.urdf:563`):
   ```xml
   <!-- Before: -->
   <joint name="left_base_link" type="fixed">
     <origin xyz="0 0.08 0" rpy="0 0 1.5708" />
   <!-- After: -->
   <joint name="left_base_link" type="fixed">
     <origin xyz="0 0.084 0" rpy="0 0 1.5708" />
   ```
   `right_base_link` already has `-0.084` Y; this brings left into symmetry.

3. **S4 — `cc.add` weight** (`_curobo_internals.py:264`):
   - **Investigation step** (read-only, no code change): grep `cc.add(` in `curobo/_src/rollout/cost_manager/cost_manager_robot.py` and adjacent files to see what value/kwargs cuRobo's built-in costs pass.
   - Apply the matching pattern. If built-ins pass `weight=cost._weight`, use that; if `1.0`, use that; if they pass nothing and `CostCollection` has matching default-population logic we missed, fix the underlying assumption instead.
   - **Anticipated fix** (subject to investigation): change `cc.add(cost.forward(state), name)` → `cc.add(cost.forward(state), name, weight=1.0)`. Justification: `BaseCost.forward()` already applies `self._weight` internally; the collection-level weight is for downstream re-weighting and defaulting to 1.0 keeps total contribution unchanged.

4. **F5 — `enable_com_polygon` plumbed through** (3 files):
   - `cutamp/config.py`: add `enable_com_polygon: bool = True` to `TAMPConfiguration`.
   - `cutamp/scripts/run_cutamp.py`: add `parser.add_argument("--no_enable_com_polygon", dest="enable_com_polygon", action="store_false")`. Pass through to `TAMPConfiguration(..., enable_com_polygon=args.enable_com_polygon)`.
   - `cutamp/tamp_world.py:286`: `def get_motion_planner(self, ..., enable_com_polygon: bool = True)`, forward to `get_t1_motion_planner(..., enable_com_polygon=enable_com_polygon)`. Wire `TAMPWorld` itself to pass `config.enable_com_polygon` at the call site in `algorithm.py` / wherever `get_motion_planner` is invoked.

**Success criteria**: smoke test passes with default `--enable_com_polygon=True`; `--no_enable_com_polygon` runs and disables the cost (verified by inspecting `mgr._extra_costs` after planner construction).

---

## Commit 3 — `plan_processor` schema change (F1, D2, D3, F2)

**BREAKING CHANGE**: Old `motion_plan.pkl` files become invalid. Existing consumer code (anything that reads `*_hand_xyz` expecting palm frame) needs to update.

**Files**:
- `cutamp/utils/plan_processor.py`: change FK target, add joint-name alignment, fix quaternion sign flip, add idempotency sentinel for `compensate_trunk_x_offset` (note: this sentinel lives in the example, not the processor — moved to commit list below).
- `examples/load_motion_plan_for_mpc.py`: update to consume hand_link poses, add idempotency sentinel to `compensate_trunk_x_offset`.
- `docs/sim_to_real_mapping.md`: rewrite discrepancy #4 (poses now in real-URDF-native frame).
- `cutamp/tests/test_plan_processor.py` (NEW or amend): add small unit test for quaternion sign canonicalization.

**Changes**:

1. **F1 — Switch to `*_hand_link`** (`plan_processor.py:104-105`):
   ```python
   LEFT_TOOL_FRAME = "left_hand_link"
   RIGHT_TOOL_FRAME = "right_hand_link"
   ```
   Update `_build_processing_kinematics` to add `left_hand_link, right_hand_link, Trunk` to `tool_frames` (drop the `*_base_link` entries). Update the module docstring (lines 26-32): hand poses are now FK of `*_hand_link` — the link that exists in both `t1_simplified.urdf` and `actual_robot.urdf`. Rewrite `docs/sim_to_real_mapping.md` discrepancy #4 to reflect this; remove the now-stale note about palm offsets.

2. **D2 — joint-name alignment check** (`plan_processor.py:153`):
   ```python
   # Before:
   if tensor.shape[-1] == active_dof:
       return tensor
   # After:
   if tensor.shape[-1] == active_dof and (
       src_joint_names is None or list(src_joint_names) == list(kin.joint_names)
   ):
       return tensor
   ```
   Falls through to the reorder branch when names mismatch. `src_joint_names is None` short-circuits to the existing assumption (callers without names trust positional alignment).

3. **D3 — Quaternion sign canonicalization** (`plan_processor.py:169-183`):
   ```python
   def _angular_velocity_from_quat(q_seq: np.ndarray, dt: float) -> np.ndarray:
       T = q_seq.shape[0]
       if T < 2:
           return np.zeros((T, 3), dtype=q_seq.dtype)
       # Canonicalize sign so consecutive samples are aligned. q and -q are the
       # same rotation but element-wise FD treats them as opposite — without
       # this, a single sign flip yields ~4/dt rad/s spurious omega spike.
       q_aligned = q_seq.copy()
       dots = np.sum(q_aligned[1:] * q_aligned[:-1], axis=-1)
       flip_mask = dots < 0
       q_aligned[1:][flip_mask] *= -1
       dq = (q_aligned[1:] - q_aligned[:-1]) / dt
       conj_q = _quat_conjugate_wxyz(q_aligned[:-1])
       omega_quat = 2.0 * _quat_mul_wxyz(dq, conj_q)
       omega = omega_quat[..., 1:]
       return np.concatenate([omega, omega[-1:]], axis=0)
   ```
   Note: sign-flip detection is sequential (one flip at index `t` propagates through subsequent samples), so a single forward sweep with running propagation is technically more correct than an element-wise mask. The element-wise version above suffices when flips are isolated (the typical FK case). Spec acknowledges this — if implementer finds a case with consecutive flips, upgrade to the running-propagation version (~5 extra lines).

   Add a unit test:
   ```python
   def test_quat_angular_velocity_robust_to_sign_flip():
       q = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float64)
       omega = _angular_velocity_from_quat(q, dt=0.1)
       assert np.allclose(omega, 0.0, atol=1e-9), f"Sign-flip yielded {omega}"
   ```

4. **F2 — Idempotency sentinel for `compensate_trunk_x_offset`** (`examples/load_motion_plan_for_mpc.py:70-82`):
   ```python
   def compensate_trunk_x_offset(plan: Dict[str, Any]) -> Dict[str, Any]:
       """..."""
       if plan.get("_trunk_offset_applied", False):
           warnings.warn(
               "compensate_trunk_x_offset called on a plan that's already been "
               "compensated; skipping to avoid double-subtraction.",
               stacklevel=2,
           )
           return plan
       for seg in plan["segments"]:
           seg["position"]["trunk_xyz"][:, 0] -= TRUNK_X_OFFSET_M
       plan["_trunk_offset_applied"] = True
       return plan
   ```
   Document the sentinel in the module docstring + add a note in `docs/sim_to_real_mapping.md`.

**Success criteria**: smoke test passes (fresh pickle generated from current code is loadable by the updated example); unit test for sign-flip passes; manual check: `load_for_mpc(path)` then `compensate_trunk_x_offset(plan)` warns instead of double-subtracting.

---

## Commit 4 — Pin atomicity (C2, C3, S5)

**Files**:
- `cutamp/t1_state.py`: refactor `_apply_pin` signature + body, add guard to `pin_for_movebase`.

**Changes**:

1. **C3 + S5 — Move bookkeeping into `_apply_pin`** (`t1_state.py:104-178`):
   ```python
   def _apply_pin(
       self, *,
       pin_joint_names: List[str],
       disabled_tool_frames: List[str],
       pin_weight: float,
       default_weight: float,
       hosts: List[Any],
   ) -> None:
       if self._saved_target_dof_weight is None:
           self._saved_target_dof_weight = snapshot_cspace_target_dof_weight(hosts)
           self._saved_pin_hosts = hosts

       # Set bookkeeping BEFORE the planner mutation so unpin can always
       # recover any partial state if update_tool_pose_criteria raises.
       self._disabled_tool_pose_frames = list(disabled_tool_frames)

       weights = self._build_weights_tensor(pin_joint_names, pin_weight, default_weight)
       write_cspace_target_dof_weight(hosts, weights)

       if disabled_tool_frames:
           self.planner.update_tool_pose_criteria({
               f: ToolPoseCriteria.disabled() for f in disabled_tool_frames
           })
   ```
   Delete the redundant `self._disabled_tool_pose_frames = [...]` assignment from `pin_for_movebase` (now lives inside `_apply_pin`).

2. **C2 — `pin_for_movebase` already-pinned guard** (`t1_state.py:157`):
   ```python
   def pin_for_movebase(self, *, pin_weight=1000.0, default_weight=1.0):
       if self._saved_target_dof_weight is not None:
           raise RuntimeError(
               "pin_for_movebase called while a pin is already active; "
               "call unpin() first to avoid silently overwriting state."
           )
       # ... existing body ...
   ```
   Matches the guard at `pin_for_arm_action:135-138`.

**Success criteria**: smoke test passes; unit test (NEW, ~15 LOC) that asserts double-`pin_for_movebase` raises and that `unpin` after a simulated mid-call exception restores both cspace AND tool-pose criteria.

---

## Commit 5 — Misc fixes (E3, E4, B2, F4)

**Files**:
- `cutamp/tests/debug_t1_tool_frame.py`: fix stale imports.
- `examples/load_motion_plan_for_mpc.py`: `.copy()` on broadcast joints + broader pickle exception.
- `cutamp/_curobo_internals.py:74`: fix squeeze loop dim check.
- `docs/sim_to_real_mapping.md`: add the pickle-safety paragraph.

**Changes**:

1. **E3 — Debug script imports** (`tests/debug_t1_tool_frame.py:87-92`):
   ```python
   # Before:
   from cutamp.robots.t1 import get_t1_kinematics_model, t1_home_left, load_t1_rerun
   ...
   left_kin = get_t1_kinematics_model("left")
   rerun_robot = load_t1_rerun(load_mesh=True, arm="left")
   # After:
   from cutamp.robots.t1 import get_t1_kinematics, t1_home, load_t1_rerun
   ...
   kin = get_t1_kinematics(device_cfg=DeviceCfg())  # single kinematics; arm picked via tool_frame
   rerun_robot = load_t1_rerun(load_mesh=True)
   ```
   Verify by `python -m cutamp.tests.debug_t1_tool_frame` runs without ImportError/TypeError (or at minimum imports succeed).

2. **E4 — Broadcast joint aliasing** (`examples/load_motion_plan_for_mpc.py:130-131`):
   ```python
   for sim_field, real_names in [...]:
       for rn in real_names:
           joint_commands[rn] = P[sim_field].copy()
           joint_velocities[rn] = V[sim_field].copy()
   ```
   Plus a one-line comment explaining why (mutation-isolation for asymmetric MPC trims).

3. **B2 — Squeeze loop dim check** (`_curobo_internals.py:74`):
   ```python
   # Before:
   while end.dim() > 1:
       end = end[..., -1, :] if end.dim() > 2 else end[-1]
   # After: always slice the last-but-one axis until 1-D.
   while end.dim() > 1:
       end = end[..., -1, :]
   ```
   The `[..., -1, :]` form correctly handles all `dim() >= 2` cases (1D → unreachable). Add a 6-line test:
   ```python
   def test_cspace_plan_succeeded_inspects_all_batches():
       # [B=2, T=4, dof=3], batch 0 reached the goal, batch 1 didn't
       ... assert cspace_plan_succeeded(...) is True
   ```

4. **F4 — Broaden pickle except + safety doc** (`examples/load_motion_plan_for_mpc.py:167-174`):
   ```python
   # WARNING: pickle.load executes arbitrary Python from the source file.
   # Only load motion_plan.pkl files from trusted sources (own filesystem,
   # generated by cutamp on a machine you control). Loading an attacker-
   # supplied pickle is equivalent to running arbitrary code.
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
           f"motion_plan.pkl at {path} appears corrupt or incomplete ({type(e).__name__}: {e}). "
           f"Regenerate with --save_plan."
       )
   except AttributeError as e:
       raise RuntimeError(
           f"motion_plan.pkl at {path} references a class that's been renamed/removed "
           f"({e}). The schema has likely evolved; regenerate the plan."
       )
   ```
   Plus a paragraph in `docs/sim_to_real_mapping.md` flagging the pickle-safety expectation for consumers.

**Success criteria**: smoke test passes; debug script imports succeed; cspace_plan_succeeded test passes; manual check that loading a truncated pickle yields the friendly RuntimeError.

---

## Verification

After EACH commit (not just at the end):

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan
```

Pass:
- ≥1 satisfying solution
- All 12 motion plans succeed (or fail-then-retry to success — warning OK)
- No new tracebacks
- No new `UserWarning` lines from our edits

Additionally, after the FINAL commit:
- `pytest cutamp/tests/ -v` — full test suite passes
- New tests pass: `test_coupled_reik_smoke.py`, `test_plan_processor.py` (sign-flip), the cspace_plan_succeeded test, the pin-atomicity test.
- Regenerate `data/motion_plan.pkl` fresh, run `python examples/load_motion_plan_for_mpc.py` — sanity output matches expectations, includes `_trunk_offset_applied` sentinel in the loaded plan.

## Open items requiring investigation during implementation

1. **S4 (cc.add weight)**: Read `curobo/_src/rollout/cost_manager/cost_manager_robot.py` for the built-in cost dispatch pattern; mirror it. If the answer is `weight=1.0`, fix is one line; if `weight=cost._weight`, also one line with a different value; if the manager has automatic-default-weight population we missed, fix the deeper assumption.

2. **D3 (quat sign canonicalization)**: The element-wise mask handles isolated flips. If implementer finds the FK emits consecutive flips (rare but possible during fast rotations), upgrade to a running-propagation loop. Both versions are ~5-10 lines.

3. **F1 schema break — consumer notification**: After the schema breaks, the `examples/load_motion_plan_for_mpc.py` comment block must explicitly say "schema v2 (2026-05-23): hand poses are now *_hand_link, not *_base_link". Consider adding `plan["_schema_version"] = 2` to the saved dict for runtime detection.

## Risk

- **Low** for commits 1, 4, 5 — narrow surgical changes with clear success criteria.
- **Medium** for commit 2 — URDF inertial change shifts the COM cost target; visual verification recommended (the cost should now steer toward poses where the visible mesh stays above the wheelbase).
- **Medium-high** for commit 3 — schema break invalidates existing pickles. Any consumer code anywhere needs updating. Mitigated by the `_schema_version` field in open items.
- **Low risk** the smoke test regresses after any single commit — but each commit is reviewable + revertible independently.

## Out of scope

Items that surfaced during review but did NOT make the top-15, explicitly NOT addressed in this bundle:

- **C4** — `pin_for_arm_action` / `pin_for_movebase` called outside the `try:` block in `motion_solver.py:384`. Verified CONFIRMED but currently latent; needs a structural rework of the pin/unpin lifecycle (move the pin into the try, or factor a `with state.pinned(arm):` context manager). Track for a follow-up PR.
- **A3** — `val.requires_grad = True` on possibly non-leaf tensor at `optimize_plan.py:309`. PLAUSIBLE only; depends on cuRobo's internal `requires_grad` discipline. Ship a defensive `.detach()` only if a real reproducer surfaces.
- **D1** — `_to_active_cspace` silently drops batches when input has batch dim > 1. PLAUSIBLE/latent — cuRobo's current `get_interpolated_plan` raises before returning a true B>1 trajectory. Revisit if/when we use `plan_batch_*` APIs.
- **D4** — `trunk_height` aliases `trunk_xyz_w[:, 2]` as a numpy view. PLAUSIBLE/latent — today's only in-tree consumer mutates column 0. Add `.copy()` only if a Z-axis mutation enters the consumer path.
- **E2** — `conf_locking.py` first-match-wins for confs shared between MoveBaseTo and arm ops. PLAUSIBLE/latent — current `blocks_t1` env has no navigate. Revisit when a multi-arm env exercises MoveBaseTo + LeftPick on a shared conf.
- **General refactoring** of `motion_solver.py` or `optimize_plan.py`: out of scope; bug fixes only.
