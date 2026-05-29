# Plan velocity/acceleration derivation — design

**Date:** 2026-05-29
**Status:** C4 investigation complete — design finalized, ready to implement
**Branch:** `curobo_v2`
**Origin:** carved out of the COM-polygon & MPC-consumer review-fix spec
(`2026-05-29-com-and-mpc-review-fixes-design.md`, root cause "C") because the
root-cause investigation (C4) was open-ended enough to warrant its own
spec → plan → implementation cycle.

## Goal

Make the velocities and accelerations that `cutamp/utils/plan_processor.py`
writes into the MPC plan physically correct: use cuRobo's native joint
derivatives, derive exact world-frame Cartesian velocity from the analytic
Jacobian, and eliminate the fabricated-zero acceleration at every segment
boundary. **Never fabricate a derivative silently** — if a derivative cuRobo
should provide is missing, raise rather than substitute a made-up finite
difference.

## Constraints

- **No edits to `curobo/`** (vendored, read-only; would need a fork). All
  changes are cuTAMP-side. The Jacobian and native derivatives this design uses
  are already produced by cuRobo — we only opt into them.
- Tests run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Project python:
  `/home/yoonwoo/miniconda3/envs/tamp/bin/python`. CUDA GPU present.
- Stage only specific files; no `git add -A`.

## Scope

Review findings **8** and **13** (the finite-difference acceleration tail). Out
of scope: the COM-polygon, 0.0625-offset-purge, numpy-`.copy()`, and
misc-hardening fixes — those are the A/B/D/E causes in the sibling spec
(already implemented and committed).

---

## C4 investigation outcome (RESOLVED — original hypothesis REFUTED)

The carved-out C4 task hypothesized that cuRobo's native joint velocity/
acceleration get dropped to `None` before `plan_processor`, forcing the
finite-difference fallback. **A multi-layer trace (4 static layer reads of the
cuRobo→cutamp pipeline + 1 dynamic GPU probe that instrumented
`process_motion_plan` during a real `--save_plan` smoke) refuted this.**

**Findings:**

- **Joints carry native derivatives end-to-end.** At `process_motion_plan`, for
  every one of the 8 trajectory segments, `entry["plan"].velocity`,
  `.acceleration`, `.jerk`, and `.joint_names` are all **non-None**. The native
  values flow: `interpolated_trajectory [1,1,5000,31]` →
  `get_full_dof_from_solution` → `trim_joint_state_trajectory` → `[1,1,81,31]`
  → `_interp_plan` → `entry["plan"]`. No layer nulls them. The buffer is backed
  by `JointState.zeros(...)` (allocates real tensors), not
  `JointState.from_position(...)` (which would null derivatives).
- **The joint FD fallback never fires** on the current code path — it is dead
  code, not an active bug.
- **The apparent "joint FD signature" in the saved pkl was a misread.** Joint
  accel tails like `[…, 6e-6, 0.0, 3e-6, 0.0]` are *native* B-spline data
  genuinely tapering to rest at the segment end (the robot decelerates to a
  stop), not an FD artifact.
- **The real bug is Cartesian-only.** cuRobo FK exposes no link velocity, so the
  Cartesian channels are finite-differenced from the FK pose sequence; accel =
  `FD(FD(pose))` with a tail-duplicating `_forward_finite_diff`, so the **last
  two acceleration samples of every Cartesian channel** (`trunk_xyz_ddot`, hand
  accels, angular accels) collapse to exactly `0.0` at each segment boundary.
  Confirmed in a freshly generated plan: Cartesian `*_ddot` last-5 norms
  `[…, 9.9e-5, 0.0, 0.0]` (two trailing zeros) vs joint `*_acc` which taper
  smoothly.
- **Original code-review finding 8 (joint double-FD) was wrong**; **finding 13
  (Cartesian double-FD) is correct and confirmed.**

**Jacobian discovery probe (for C1):** building the processing kinematics with
`Kinematics(cfg, compute_jacobian=True)` populates
`tool_jacobians [batch, horizon, num_links, 6, dof]`. Verified empirically on a
T=200 synthetic trajectory: `J·q̇` matches the world-frame FK-difference hand
velocity to **corr = 1.000000, max error 8.6e-5**. **Rows `[0:3]` = linear,
`[3:6]` = angular, both already in WORLD frame (no rotation needed).** The `dof`
dim equals the active joint count in `kinematics.joint_names` order, so a `q̇`
assembled in that order contracts correctly. Link index ↔ frame order matches
`tool_frames`: `left_hand_link=0, right_hand_link=1, Trunk=2`. (FK is also plain
torch-autograd differentiable, but the native `compute_jacobian` path is exact
and simpler, so autodiff is not used.)

---

## Problem (the confirmed Cartesian-only bug)

`_forward_finite_diff` (`cutamp/utils/plan_processor.py`) duplicates its last
sample to preserve shape:

```python
diff = np.diff(values, axis=0) / dt
return np.concatenate([diff, diff[-1:]], axis=0)
```

For the **Cartesian** channels, acceleration is this applied twice (velocity =
FD(FK pose), acceleration = FD(velocity)). Each FD duplicates its tail, so
`v[T-1] == v[T-2]` ⇒ `a[T-2] = 0` and `a[T-1] = a[T-2] = 0`: the last two
acceleration samples collapse to zero at every segment boundary. An MPC
feedforwarding Cartesian acceleration gets zero feedforward exactly at the
hand-off between segments — the worst place for it. `_angular_velocity_from_quat`
has the same tail-duplication, so angular acceleration double-FDs too.

The **joint** channels do not have this problem — they use native cuRobo
derivatives (see C4 outcome). The joint FD fallback is dead code.

## Fix

Three changes, all in `cutamp/utils/plan_processor.py` (cuTAMP-side only):

**C1 — Cartesian velocity via the Jacobian (`J·q̇`).** Build
`_build_processing_kinematics` with `compute_jacobian=True`
(`Kinematics(kin_cfg, compute_jacobian=True)`). In `process_motion_plan`, after
the FK call, for each tracked link take `J = ks.tool_jacobians[0, :, link_idx,
:, :]` (`[T, 6, dof]`) and the native joint velocity `q̇ = vel_active` (`[T,
dof]`, already in `kinematics.joint_names` order). Then
`Jq = einsum("tij,tj->ti", J, q̇)` (`[T, 6]`); **world-frame linear velocity =
`Jq[:, 0:3]`, world-frame angular velocity = `Jq[:, 3:6]`** — no rotation. Use
this for `trunk_xyz_dot`, `left/right_hand_xyz_dot`, and the three
`*_angular_velocity_world` channels (`trunk`, `left_hand`, `right_hand`), and
`trunk_height` velocity = the z-component of the trunk linear velocity. This
**retires both** `_forward_finite_diff`-on-pose for Cartesian velocity **and**
`_angular_velocity_from_quat` entirely.

`link_idx` is resolved from `kinematics.tool_frames` (or the `ToolPose` frame
order), not hard-coded, so a future tool_frames reordering can't silently
mis-index.

**C2 — Cartesian acceleration.** cuRobo exposes no `J̇` and no native link
accel, so Cartesian acceleration is one numerical differentiation of the *exact*
C1 velocity — via `np.gradient(v, dt, axis=0, edge_order=2)` (2nd-order central
interior, 2nd-order one-sided at both endpoints; never duplicates). Applies to
`trunk_xyz_ddot`, `left/right_hand_xyz_ddot`, the three
`*_angular_acceleration_world`, and `trunk_height` accel. Result: a genuine
endpoint value, no 2-zero tail.

**C3 — Joints: native or error (NO fabricated FD).** Use the native joint
velocity/acceleration (`vel_active` / `acc_active`), which C4 confirmed are
always present. **Delete the finite-difference fallback** for the joint channels
(`trunk_pitch`, `trunk_yaw`, `ankle_pitch`, `knee_pitch`, `left_arm`,
`right_arm`). If `vel_active` or `acc_active` is `None`, **raise a clear
`RuntimeError`** naming which is missing and that the plan lacks native
derivatives — do **not** substitute a made-up finite difference. Rationale
(user directive): a silently-fabricated derivative is misleading; a missing
native derivative means something upstream is wrong and must surface loudly.
(Note: `vel_active` is also required by C1 for `J·q̇`, so this check naturally
gates the Cartesian path too — no `q̇`, no velocity, hard error.)

**Helper cleanup.** After C1–C3, `_forward_finite_diff` and
`_angular_velocity_from_quat` have no remaining callers — delete them. The
quaternion helpers `_quat_*` may still be used elsewhere (e.g. xyzw conversion);
delete only what becomes unused (verify by grep before removing each).

## Files

- `cutamp/utils/plan_processor.py` — the only production file:
  - `_build_processing_kinematics`: `Kinematics(kin_cfg, compute_jacobian=True)`.
  - `process_motion_plan`: Cartesian velocity via `J·q̇` (C1); Cartesian accel
    via `np.gradient(edge_order=2)` of that velocity (C2); joint vel/accel
    native-or-`RuntimeError`, FD fallback deleted (C3).
  - Remove now-unused `_forward_finite_diff` and `_angular_velocity_from_quat`
    (and any `_quat_*` helper left with zero callers).
- `cutamp/utils/test_plan_processor_derivatives.py` (or extend an existing test
  module) — see Tests.

`cutamp/motion_solver.py` is **not** touched — C4 proved the derivatives are not
dropped there.

## Tests

CUDA is available, so Jacobian/FK tests run for real (gate with the repo's
`needs_cuda` pattern so they skip cleanly on a CPU box).

- **C1 Jacobian velocity (needs_cuda):** a synthetic trajectory moving a couple
  of arm joints at a known analytic `q̇` → assert the `J·q̇` linear velocity
  matches a central-difference of the FK world hand position to tight tol
  (e.g. max abs err < 1e-3) at **all** timesteps including endpoints; assert the
  angular block similarly matches a quaternion-difference world ω.
- **C2 endpoint correctness (CPU):** feed a known-curvature velocity (e.g.
  linear-in-time `v = a·t`) to the accel computation → `np.gradient` returns
  `≈ a` at **every** sample including the last two (the old double-FD returned 0
  there). T==2 edge case returns finite values, no crash.
- **C3 native passthrough (needs_cuda or fixture):** a plan whose JointState has
  populated velocity/accel → processor emits those native values verbatim for
  the joint channels (no `np.gradient` applied to joints).
- **C3 error path (CPU):** a JointState with `velocity=None` (or
  `acceleration=None`) → `process_motion_plan` raises `RuntimeError` mentioning
  the missing native derivative — NOT a silently finite-differenced result.
- **End-to-end regression (needs_cuda):** a freshly generated plan has Cartesian
  `*_ddot` tails that are NOT `[…, 0.0, 0.0]` (no 2-zero tail) and joint accel
  tails that match the native values.

## Sequencing

C4 is done (investigation complete). Remaining implementation, TDD per step:

1. **C3 first** (smallest, decoupled): joints native-or-error; delete the joint
   FD fallback. Add the native-passthrough + error-path tests.
2. **C1**: `compute_jacobian=True` + `J·q̇` Cartesian velocity (linear + angular),
   retire `_angular_velocity_from_quat`. Add the Jacobian-velocity test.
3. **C2**: Cartesian accel via `np.gradient` of the C1 velocity; delete
   `_forward_finite_diff`. Add the endpoint-correctness test.
4. Helper cleanup + end-to-end regression test + regenerate `data/motion_plan.pkl`.

May land as one focused commit or a few; single file, so a small logical split
is fine.

## Verification

End-to-end: regenerate a plan and confirm (a) Cartesian `*_ddot` tails are
non-degenerate (no two trailing zeros), (b) `J·q̇` linear velocity matches an
FK-difference to tight tol, (c) joint channels equal the native cuRobo values,
(d) a derivative-less JointState raises instead of fabricating.

```
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 64 --num_opt_steps 50 --motion_plan \
  --save_plan /tmp/c_check.pkl
```
Then probe `/tmp/c_check.pkl`'s Cartesian `*_ddot` tails.

## Risks / notes

- **Value change:** Cartesian velocity moves from FD-of-pose to the exact
  analytic `J·q̇`, and Cartesian accel moves to central differences — so saved
  velocity/accel values shift vs today across the whole trajectory, not just the
  tail. This is a strict accuracy improvement (exact velocity, world-frame, no
  fabricated endpoints), fine for an offline reference plan. Only revisit if an
  MPC were numerically tuned against the *current* (buggy) arrays.
- **C1 frame/units — RESOLVED by the discovery probe:** `tool_jacobians` linear
  rows `[0:3]` / angular rows `[3:6]`, both WORLD frame, no rotation. The C1 test
  re-checks this so a future cuRobo change that flips the convention fails loudly.
- **C3 is a behavior change to a hard error:** if some untested plan path *does*
  arrive with `None` joint derivatives, this turns a previously-silent (wrong)
  result into a loud failure. That is the intended, user-directed behavior — the
  fix surfaces the problem instead of hiding it.
- **Schema:** field shapes and names are unchanged — **no schema bump** (stays
  v3). Regenerate `data/motion_plan.pkl` after landing so the committed sample
  reflects the corrected derivatives.
- **No `curobo/` edits, no fork** — `compute_jacobian` is a constructor flag on
  cuTAMP's own processing kinematics; the native derivatives already exist.
