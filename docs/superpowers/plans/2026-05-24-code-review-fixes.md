# Code Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land all 15 `/code-review` findings as 5 thematic commits on `curobo_v2`.

**Architecture:** TDD where possible (new code gets a failing test first). One commit per finding category. Smoke test runs between commits so a regression points at one commit. Schema-breaking change (Commit 3) is intentionally isolated so the diff is easy to bisect.

**Tech Stack:** Python 3.10+, PyTorch, cuRobo v0.8, pytest. Conda env at `/home/yoonwoo/miniconda3/envs/tamp/`.

**Spec:** `docs/superpowers/specs/2026-05-23-code-review-fixes-design.md`

**Smoke test command** (used across tasks):
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan
```
Pass criteria: ≥1 satisfying solution, motion plans succeed (or retry-to-success), no new tracebacks.

**Pre-flight investigation result** (done during plan-writing): Finding **S4 (cc.add weight)** is REFUTED. cuRobo's own built-in cost dispatch calls `cost_collection.add(value, name)` without the `weight` kwarg in all four built-in sites (`cost_manager_robot.py:238, 258, 266, 283`), and `CostCollection.clone()` / `merge()` / `copy_at_batch_seed_indices()` all tolerate `len(weights) < len(values)`. Our wrapper at `_curobo_internals.py:264` matches cuRobo's own idiom. Drop S4 from the plan; document in Commit 2's commit message.

---

## File Structure

**New files**:
- `cutamp/tests/test_coupled_reik_smoke.py` — Commit 1
- `cutamp/tests/test_plan_processor.py` — Commit 3
- `cutamp/tests/test_pin_lifecycle.py` — Commit 4
- `cutamp/tests/test_curobo_internals.py` — Commit 5

**Files modified**:
- `cutamp/particle_initialization.py` — Commit 1 (`_ik_for_pose` signature)
- `cutamp/algorithm.py` — Commit 1 (`resample_plan_info` passes particles)
- `cutamp/robots/assets/t1_description/t1_simplified.urdf` — Commit 2 (Trunk inertial, left_base_link Y)
- `cutamp/config.py` — Commit 2 (`enable_com_polygon` field)
- `cutamp/scripts/run_cutamp.py` — Commit 2 (`--no_enable_com_polygon` CLI flag)
- `cutamp/tamp_world.py` — Commit 2 (forward `enable_com_polygon`)
- `cutamp/utils/plan_processor.py` — Commit 3 (world-frame + hand_link + schema_version + D2 + D3)
- `examples/load_motion_plan_for_mpc.py` — Commit 3 (delete compensate helper) + Commit 5 (.copy(), broader except)
- `docs/sim_to_real_mapping.md` — Commit 3 (rewrite discrepancies #1, #2, #4) + Commit 5 (pickle-safety note)
- `cutamp/t1_state.py` — Commit 4 (`_apply_pin` move bookkeeping inside, `pin_for_movebase` guard)
- `cutamp/tests/debug_t1_tool_frame.py` — Commit 5 (fix imports)
- `cutamp/_curobo_internals.py` — Commit 5 (squeeze loop fix)

---

## Task 1: Commit 1 — coupled_reik + resample correctness

**Findings**: A1 (`--coupled_reik` TypeError), S1 (retract collision silently dropped), S2 (`resample_plan_info` missing particles)

**Files:**
- Modify: `cutamp/particle_initialization.py:60-99`
- Modify: `cutamp/algorithm.py:220`
- Create: `cutamp/tests/test_coupled_reik_smoke.py`

### Step 1.1: Write the failing smoke test for `--coupled_reik`

- [ ] Create `cutamp/tests/test_coupled_reik_smoke.py`:

```python
"""Smoke test for --coupled_reik path. Catches the A1-class regression where
the coupled-reIK feature TypeErrors on first refresh."""
import os
import pytest


@pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)
def test_coupled_reik_runs_without_error(tmp_path):
    """Run cutamp_demo with --coupled_reik for a few Adam steps. Assert no exception."""
    # Lazy import inside the test so pytest collection doesn't pull in torch/cuda.
    from cutamp.config import TAMPConfiguration
    from cutamp.scripts.run_cutamp import cutamp_demo

    config = TAMPConfiguration(
        robot="t1",
        env="blocks_t1",
        num_particles=8,
        num_opt_steps=5,
        coupled_reik=True,
        reik_interval=2,
        optimize_soft_costs=True,
        soft_cost=("place_close_to_base",),
        disable_visualizer=True,
        motion_plan=False,  # skip motion plan to keep test fast
        experiment_root=str(tmp_path),
    )
    # cutamp_demo returns without raising if --coupled_reik path is healthy.
    cutamp_demo(config)
```

### Step 1.2: Run the test to verify it fails

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_coupled_reik_smoke.py -v -s 2>&1 | tail -30
```

Expected: FAIL with `TypeError: _ik_for_pose() got an unexpected keyword argument 'current_state_q'`.

If the test instead fails because TAMPConfiguration has no `coupled_reik` field, jump to step 1.3 (the kwarg must already exist per the design); if it lacks `reik_interval`, ditto. Check `cutamp/config.py` and confirm both fields exist on `TAMPConfiguration`. If not, this means coupled_reik was never wired up; in that case extend `cutamp/config.py` to add both fields with safe defaults (`coupled_reik: bool = False`, `reik_interval: int = 5`) before continuing.

### Step 1.3: Extend `_ik_for_pose` signature

- [ ] Edit `cutamp/particle_initialization.py:60-99`. Current signature is `def _ik_for_pose(world, world_from_ee, arm)`. Replace it with:

```python
def _ik_for_pose(
    world: TAMPWorld,
    world_from_ee: torch.Tensor,
    arm: Optional[str],
    *,
    current_state_q: Optional[torch.Tensor] = None,
    num_seeds: Optional[int] = None,
):
    """Solve IK for a batch of EE poses, targeting the active arm's tool frame.

    Uses the same multi-frame-goal + cspace-pin pattern the motion planner
    applies during arm operators: the inactive frame is targeted at its
    current FK pose, and its joints are pinned via ``cspace_target_dof_weight``.
    Without this, IK leaves the inactive arm unconstrained and it drifts to
    arbitrary joint values, polluting downstream particle optimization.

    ``current_state_q`` (full-cspace ``[B, len(world.kinematics.joint_names)]``)
    seeds IK from a non-home pose — used by the coupled-reIK refresh during
    Adam to warm-start from the current best particle's q. When ``None``,
    seeds from ``world.q_init`` as before.

    ``num_seeds`` overrides the IK solver's seed count. Set to 1 for warm-start
    refresh (~10-20x faster than the default multi-seed); leave ``None`` for
    initial IK from home (which needs the default for reliability).
    """
    ik = world.ik_solver
    active_frame = world.tool_frame_for_arm(arm)
    if active_frame not in ik.kinematics.tool_frames:
        raise RuntimeError(
            f"IK solver does not expose tool frame {active_frame!r}; "
            f"available: {ik.kinematics.tool_frames}"
        )
    target_pose = Pose.from_matrix(world_from_ee)
    batch_size = world_from_ee.shape[0]
    full_names = list(world.kinematics.joint_names)
    if current_state_q is None:
        seed_full = world.q_init.unsqueeze(0).expand(batch_size, -1)
    else:
        # Caller passes full-cspace q; trust their batch shape.
        seed_full = current_state_q
    full_js = JointState.from_position(seed_full, joint_names=full_names)
    current_state = ik.kinematics.get_active_js(full_js)
    goal = _build_multi_frame_goal(ik, active_frame, target_pose, current_state)

    snapshot = snapshot_cspace_target_dof_weight([ik])
    try:
        if arm is not None:
            write_cspace_target_dof_weight(
                [ik], inactive_arm_cspace_weights(ik, arm),
            )
        solve_kwargs = {"goal_tool_poses": goal, "current_state": current_state}
        if num_seeds is not None:
            solve_kwargs["num_seeds"] = num_seeds
        return ik.solve_pose(**solve_kwargs)
    finally:
        restore_cspace_target_dof_weight([ik], snapshot)
```

### Step 1.4: Fix `resample_plan_info` to pass particles

- [ ] Edit `cutamp/algorithm.py:220`. Replace:

```python
        cost_dict = plan_info["cost_fn"](rollout)
```

with:

```python
        cost_dict = plan_info["cost_fn"](rollout, plan_particles)
```

This matches the sibling call at `algorithm.py:160` and fixes BOTH the silent retract-collision drop (via the `if self._particles is not None` guard at `cost_function.py:387`) AND the hard `RuntimeError("Particles must be provided for ... soft cost")` that fires when any particle-dependent soft cost is active.

### Step 1.5: Run the smoke test — should now pass

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_coupled_reik_smoke.py -v -s 2>&1 | tail -20
```

Expected: PASS.

### Step 1.6: Run end-to-end smoke test (no regression)

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan 2>&1 | tail -40
```

Expected: ≥1 satisfying solution, all 12 motion plans succeed. No new tracebacks.

### Step 1.7: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/particle_initialization.py \
  cutamp/algorithm.py \
  cutamp/tests/test_coupled_reik_smoke.py && \
git commit -m "$(cat <<'EOF'
fix: coupled_reik TypeError + resample dropping particles

- Add current_state_q + num_seeds kwargs to _ik_for_pose, threading them
  to ik.solve_pose. Fixes the TypeError that fired immediately when
  --coupled_reik was passed (the refresh path was calling _ik_for_pose
  with current_state_q= but the function didn't accept it).
- resample_plan_info now passes plan_particles to cost_fn (matching
  sample_plan_skeleton at algorithm.py:160). Fixes two bugs together:
  (a) silent retract-collision drop via cost_function.py:387's
  `if self._particles is not None` guard, and
  (b) hard RuntimeError for particle-dependent soft costs (com_polygon,
  retract_close_to_home, minimize_body_movement).
- Add smoke test cutamp/tests/test_coupled_reik_smoke.py so the
  --coupled_reik path can't silently re-break.

Findings: A1, S1, S2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Commit 2 — URDF + cost dispatch correctness

**Findings**: S3 (Trunk inertial offset), F5 (`enable_com_polygon` not plumbed), URDF Y-asymmetry. **S4 (cc.add weight) REFUTED during plan-writing investigation — no change needed.**

**Files:**
- Modify: `cutamp/robots/assets/t1_description/t1_simplified.urdf:219` (Trunk inertial X), `:563` (left_base_link Y)
- Modify: `cutamp/config.py` (add `enable_com_polygon: bool = True`)
- Modify: `cutamp/scripts/run_cutamp.py` (add `--no_enable_com_polygon` CLI flag)
- Modify: `cutamp/tamp_world.py:273-296` (forward `enable_com_polygon` to `get_t1_motion_planner`)

### Step 2.1: Fix Trunk inertial origin X compensation (S3)

- [ ] Edit `cutamp/robots/assets/t1_description/t1_simplified.urdf:219`. Replace:

```xml
      <origin xyz="-0.0073634598906924 -1.42058017623659E-06 0.105062332707657" rpy="0 0 0" />
```

with:

```xml
      <origin xyz="-0.0698634598906924 -1.42058017623659E-06 0.105062332707657" rpy="0 0 0" />
```

This subtracts `0.0625` from X, bringing the Trunk inertial origin in line with the `-0.0625` compensation that the visual (line 225) and collision (line 234) origins already carry. The doc comment at lines 212-216 explicitly promises this — we're making the code match the comment.

### Step 2.2: Fix asymmetric `*_base_link` Y offset

- [ ] Edit `cutamp/robots/assets/t1_description/t1_simplified.urdf:563`. Replace:

```xml
    <origin xyz="0 0.08 0" rpy="0 0 1.5708" />
```

with:

```xml
    <origin xyz="0 0.084 0" rpy="0 0 1.5708" />
```

Brings left into symmetry with right (which already has Y = -0.084 at line 937).

### Step 2.3: Add `enable_com_polygon` field to `TAMPConfiguration`

- [ ] Edit `cutamp/config.py`. Locate the `TAMPConfiguration` class (the only one with `enable_traj: bool = False` at line 81 per grep). Add a new field next to other `enable_*` fields:

```python
    enable_com_polygon: bool = True  # Pass through to MotionPlanner's COM-over-base-polygon soft cost.
```

Place it near line 81-111 (anywhere among the other `enable_*` fields is fine — match the surrounding style).

### Step 2.4: Forward `enable_com_polygon` in `TAMPWorld.get_motion_planner`

- [ ] Edit `cutamp/tamp_world.py:273-296`. Locate the `get_motion_planner` method:

```python
    def get_motion_planner(
        self,
        ...kwargs...
    ):
        ...
        return get_t1_motion_planner(
            scene_with_movables,
            ...kwargs...
        )
```

Add `enable_com_polygon: bool = True` to the method signature and forward it:

```python
    def get_motion_planner(
        self,
        ...existing kwargs...
        enable_com_polygon: bool = True,
    ):
        ...
        return get_t1_motion_planner(
            scene_with_movables,
            ...existing kwargs...
            enable_com_polygon=enable_com_polygon,
        )
```

Then find every caller of `world.get_motion_planner(...)` in the repo:

```bash
grep -rn "get_motion_planner(" /home/yoonwoo/cuTAMP/cutamp/ 2>/dev/null
```

For each caller that has access to `config`, pass `enable_com_polygon=config.enable_com_polygon`. If a caller doesn't have access to config, it should accept the new default (True, matching today's hardcoded behavior).

### Step 2.5: Add `--no_enable_com_polygon` CLI flag

- [ ] Edit `cutamp/scripts/run_cutamp.py`. Find the argparse block (`parser.add_argument` calls around lines 96-172) and add:

```python
    parser.add_argument(
        "--no_enable_com_polygon",
        dest="enable_com_polygon",
        action="store_false",
        help="Disable the COM-over-base-polygon soft cost on the motion planner. "
             "By default the cost is enabled (1e5 weight) and keeps the planner's "
             "configurations from tipping over the wheelbase.",
    )
    parser.set_defaults(enable_com_polygon=True)
```

Then locate the `TAMPConfiguration(...)` constructor call (where args are gathered into a config dataclass) and add:

```python
        enable_com_polygon=args.enable_com_polygon,
```

### Step 2.6: Verify CLI surface

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp --help 2>&1 | \
  grep -A1 -E "no_enable_com_polygon|enable_com_polygon"
```

Expected: `--no_enable_com_polygon` appears with the help text.

### Step 2.7: Smoke-test default behavior (com_polygon enabled)

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan 2>&1 | tail -20
```

Expected: ≥1 satisfying solution. The Trunk URDF change shifts COM by 6.25 cm in X; the COM-polygon cost is sensitive to this so check that satisfying counts haven't crashed.

### Step 2.8: Smoke-test `--no_enable_com_polygon`

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan \
  --no_enable_com_polygon 2>&1 | tail -20
```

Expected: ≥1 satisfying solution. Should run without the COM-polygon cost active.

### Step 2.9: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/robots/assets/t1_description/t1_simplified.urdf \
  cutamp/config.py \
  cutamp/scripts/run_cutamp.py \
  cutamp/tamp_world.py && \
git commit -m "$(cat <<'EOF'
fix: Trunk inertial offset + symmetric *_base_link Y + --no_enable_com_polygon

- t1_simplified.urdf: Trunk inertial origin X gets the -0.0625 compensation
  (was already on visual/collision/spheres per the link comment at L212-216).
  The COM-over-base-polygon cost reads mass-weighted COM via cuRobo's
  state.robot_com; the missing offset shifted Trunk's 11.7 kg contribution
  6.25 cm in +X. Now matches the visible mesh.
- Symmetrize left_base_link Y from 0.08 to 0.084 (right already -0.084).
  After Commit 3 the saved trajectory no longer references *_base_link,
  but internal grasp planning + palm collision spheres are still off
  by 4 mm without this.
- Add TAMPConfiguration.enable_com_polygon and --no_enable_com_polygon
  CLI flag, plumbed through TAMPWorld.get_motion_planner. Previously
  the kwarg existed on get_t1_motion_planner but the call site never
  forwarded it.

S4 (cc.add weight off-by-one) was investigated and REFUTED: cuRobo's
own built-in cost dispatch (cost_manager_robot.py:238, 258, 266, 283)
calls cost_collection.add(value, name) without a weight kwarg in all
four sites, and CostCollection.{clone,merge,copy_at_batch_seed_indices}
all tolerate len(weights) < len(values). Our wrapper at
_curobo_internals.py:264 matches cuRobo's idiom.

Findings: S3, F5, URDF Y-asymm. (S4 refuted.)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Commit 3 — `plan_processor` schema rewrite (BREAKING)

**Findings**: F1 (revised: world-frame), D2 (joint-name alignment), D3 (quat sign flip), F2 (obsoleted).

**Files:**
- Modify: `cutamp/utils/plan_processor.py` (FK target, world frame, in-processor Trunk compensation, schema_version, D2, D3)
- Modify: `examples/load_motion_plan_for_mpc.py` (delete `compensate_trunk_x_offset`, update consumer to read world-frame fields, drop sanity check)
- Modify: `docs/sim_to_real_mapping.md` (rewrite resolved discrepancies)
- Create: `cutamp/tests/test_plan_processor.py`

### Step 3.1: Write the failing unit tests for plan_processor

- [ ] Create `cutamp/tests/test_plan_processor.py`:

```python
"""Unit tests for plan_processor — covers quat sign canonicalization,
joint-name alignment guard, and the world-frame schema invariants."""
import numpy as np
import pytest


def test_quat_angular_velocity_robust_to_sign_flip():
    """A wxyz quat sequence that flips sign (q -> -q) at each step represents
    the SAME rotation. Without canonicalization, element-wise FD treats the
    flip as a huge motion. This test asserts ω stays near zero across flips."""
    from cutamp.utils.plan_processor import _angular_velocity_from_quat

    q = np.array(
        [[1.0, 0, 0, 0],
         [-1.0, 0, 0, 0],
         [1.0, 0, 0, 0],
         [-1.0, 0, 0, 0]],
        dtype=np.float64,
    )
    omega = _angular_velocity_from_quat(q, dt=0.1)
    assert omega.shape == (4, 3)
    assert np.allclose(omega, 0.0, atol=1e-9), (
        f"Sign flips produced spurious omega: {omega}"
    )


def test_to_active_cspace_falls_through_on_name_mismatch():
    """Same DOF count but DIFFERENT joint name order must not short-circuit.
    Otherwise the same-shape positional gather mis-indexes every column."""
    import torch
    from cutamp.utils.plan_processor import _to_active_cspace, _build_processing_kinematics

    kin = _build_processing_kinematics()
    active = list(kin.joint_names)
    n = len(active)
    # Tensor with the active DOF count but joint names REVERSED.
    src_names = list(reversed(active))
    tensor = torch.zeros(5, n, device=kin.device_cfg.device)
    # Mark position-zero so we can detect mis-routing: a permutation must
    # produce a non-passthrough tensor.
    tensor[0, 0] = 1.0
    out = _to_active_cspace(tensor, src_names, kin)
    # If short-circuit fires the value at column 0 would survive; the reorder
    # branch moves it elsewhere.
    assert not torch.equal(out, tensor), (
        "_to_active_cspace short-circuited despite joint-name mismatch — "
        "downstream gathers would mis-index."
    )
```

### Step 3.2: Run the new tests — both should FAIL

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_plan_processor.py -v 2>&1 | tail -20
```

Expected:
- `test_quat_angular_velocity_robust_to_sign_flip` FAILS — current `_angular_velocity_from_quat` has no sign canonicalization.
- `test_to_active_cspace_falls_through_on_name_mismatch` FAILS — current code short-circuits on DOF match.

### Step 3.3: Implement quat sign canonicalization (D3)

- [ ] Edit `cutamp/utils/plan_processor.py:169-183`. Replace the body of `_angular_velocity_from_quat` with:

```python
def _angular_velocity_from_quat(q_seq: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity (rad/s) from a wxyz quaternion sequence via FD.

    Returns ω in the REFERENCE frame in which q_seq is defined
    (q_seq = world_q_trunk → ω in world; q_seq = trunk_q_hand → ω in Trunk).
    Math: ω = 2·(dq/dt) ⊗ conj(q), imaginary part. Last sample duplicated.
    """
    T = q_seq.shape[0]
    if T < 2:
        return np.zeros((T, 3), dtype=q_seq.dtype)
    # Canonicalize sign so consecutive samples are aligned. q and -q encode
    # the same rotation, but element-wise FD treats them as opposite — one
    # uncanonicalized flip yields ~4/dt rad/s spurious omega spike. Sequential
    # propagation: once a flip happens, every subsequent sample needs the flip
    # too. WHY sequential not element-wise: a single forward sweep handles
    # consecutive flips correctly.
    q_aligned = q_seq.copy()
    for t in range(1, T):
        if np.dot(q_aligned[t], q_aligned[t - 1]) < 0:
            q_aligned[t] *= -1
    dq = (q_aligned[1:] - q_aligned[:-1]) / dt
    conj_q = _quat_conjugate_wxyz(q_aligned[:-1])
    omega_quat = 2.0 * _quat_mul_wxyz(dq, conj_q)
    omega = omega_quat[..., 1:]
    return np.concatenate([omega, omega[-1:]], axis=0)
```

### Step 3.4: Implement joint-name alignment guard (D2)

- [ ] Edit `cutamp/utils/plan_processor.py:152-158`. Replace:

```python
    active_dof = len(kin.joint_names)
    if tensor.shape[-1] == active_dof:
        return tensor
    if src_joint_names is None:
        return None
    src_js = JointState.from_position(tensor, joint_names=list(src_joint_names))
    return kin.get_active_js(src_js).position
```

with:

```python
    active_dof = len(kin.joint_names)
    # Only short-circuit if shape AND joint-name order match the kinematics'
    # active cspace. Same DOF count in a different order would silently
    # mis-index every downstream per-joint gather.
    if tensor.shape[-1] == active_dof and (
        src_joint_names is None or list(src_joint_names) == list(kin.joint_names)
    ):
        return tensor
    if src_joint_names is None:
        return None
    src_js = JointState.from_position(tensor, joint_names=list(src_joint_names))
    return kin.get_active_js(src_js).position
```

### Step 3.5: Run the two unit tests — should now PASS

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_plan_processor.py::test_quat_angular_velocity_robust_to_sign_flip \
  cutamp/tests/test_plan_processor.py::test_to_active_cspace_falls_through_on_name_mismatch \
  -v 2>&1 | tail -10
```

Expected: both PASS.

### Step 3.6: Switch FK target to `*_hand_link` + emit world frame (F1 revised)

- [ ] Edit `cutamp/utils/plan_processor.py:104-105`. Replace:

```python
LEFT_TOOL_FRAME = "left_base_link"
RIGHT_TOOL_FRAME = "right_base_link"
TRUNK_LINK = "Trunk"
```

with:

```python
LEFT_TOOL_FRAME = "left_hand_link"
RIGHT_TOOL_FRAME = "right_hand_link"
TRUNK_LINK = "Trunk"

# sim's Trunk frame sits +0.0625m in X of where actual_robot.urdf places it
# (see docs/sim_to_real_mapping.md #1). We subtract this from saved
# trunk_xyz before pickling so the saved value is real-URDF-native — the
# consumer needs no compensation.
SIM_TO_REAL_TRUNK_X_OFFSET_M = 0.0625
```

- [ ] In the same file, locate `_build_processing_kinematics` and verify `tool_frames` already lists `[LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME, TRUNK_LINK]`. After the constant change above, this auto-updates to use `left_hand_link`/`right_hand_link`. No edit needed inside the function — just confirm by reading it.

- [ ] Now find the `process_motion_plan` body where it computes `right_xyz_t, right_quat_t = _world_to_trunk(...)` and `left_xyz_t, left_quat_t = _world_to_trunk(...)`. Replace those two calls with the world-frame retention:

```python
        # Hand poses stay in WORLD frame (was Trunk in v1). Consumer using
        # actual_robot.urdf computes the relative pose itself if needed.
        # The variables keep the _t suffix only by historical accident in
        # nearby lines — rename them.
```

Actually delete the `_world_to_trunk` call lines and the `_t` variables entirely. Replace the assignment block in `position` to use the world-frame values directly. Concretely, the existing block:

```python
        right_xyz_t, right_quat_t = _world_to_trunk(
            right_xyz_w, right_quat_w, trunk_xyz_w, trunk_quat_w,
        )
        left_xyz_t, left_quat_t = _world_to_trunk(
            left_xyz_w, left_quat_w, trunk_xyz_w, trunk_quat_w,
        )

        pos_np = pos_active.cpu().numpy()
        position = {
            # Trunk world pose (MPC's anchor)
            "trunk_xyz": trunk_xyz_w,
            "trunk_quat_wxyz": trunk_quat_w,
            "trunk_quat_xyzw": _xyzw_from_wxyz(trunk_quat_w),
            "trunk_height": trunk_xyz_w[:, 2],
            ...
            "right_hand_xyz": right_xyz_t,
            "right_hand_quat_wxyz": right_quat_t,
            "right_hand_quat_xyzw": _xyzw_from_wxyz(right_quat_t),
            "left_hand_xyz": left_xyz_t,
            "left_hand_quat_wxyz": left_quat_t,
            "left_hand_quat_xyzw": _xyzw_from_wxyz(left_quat_t),
        }
```

becomes:

```python
        # Apply -0.0625 X compensation to Trunk world pose: saved value
        # represents real-URDF's Trunk world pose (not sim's), so the MPC
        # consumer needs no compensation.
        trunk_xyz_w_real = trunk_xyz_w.copy()
        trunk_xyz_w_real[:, 0] -= SIM_TO_REAL_TRUNK_X_OFFSET_M

        pos_np = pos_active.cpu().numpy()
        position = {
            # Trunk world pose (real-URDF-native)
            "trunk_xyz": trunk_xyz_w_real,
            "trunk_quat_wxyz": trunk_quat_w,
            "trunk_quat_xyzw": _xyzw_from_wxyz(trunk_quat_w),
            "trunk_height": trunk_xyz_w_real[:, 2],
            # Joint values (frame-independent; broadcast both sides on real)
            "trunk_pitch": pos_np[:, torso_pitch_idx],
            "trunk_yaw": pos_np[:, waist_yaw_idx],
            # Arms (per-side joint values, names match real URDF)
            "right_arm": pos_np[:, right_arm_idxs],
            "left_arm": pos_np[:, left_arm_idxs],
            # Hand poses in WORLD frame (was Trunk in v1)
            "right_hand_xyz": right_xyz_w,
            "right_hand_quat_wxyz": right_quat_w,
            "right_hand_quat_xyzw": _xyzw_from_wxyz(right_quat_w),
            "left_hand_xyz": left_xyz_w,
            "left_hand_quat_wxyz": left_quat_w,
            "left_hand_quat_xyzw": _xyzw_from_wxyz(left_quat_w),
        }
```

- [ ] Update the `velocity` dict similarly. Currently:

```python
        velocity: Dict[str, Any] = {
            "trunk_xyz_dot": _forward_finite_diff(trunk_xyz_w, dt),
            "trunk_height": _forward_finite_diff(trunk_xyz_w[:, 2], dt),
            "right_hand_xyz_dot": _forward_finite_diff(right_xyz_t, dt),
            "left_hand_xyz_dot": _forward_finite_diff(left_xyz_t, dt),
            "trunk_angular_velocity_world": _angular_velocity_from_quat(trunk_quat_w, dt),
            "right_hand_angular_velocity_trunk": _angular_velocity_from_quat(right_quat_t, dt),
            "left_hand_angular_velocity_trunk": _angular_velocity_from_quat(left_quat_t, dt),
        }
```

Replace with:

```python
        velocity: Dict[str, Any] = {
            # Trunk linear velocity is unchanged by the X offset (constant
            # subtraction has zero derivative).
            "trunk_xyz_dot": _forward_finite_diff(trunk_xyz_w_real, dt),
            "trunk_height": _forward_finite_diff(trunk_xyz_w_real[:, 2], dt),
            # Hand linear velocities now in WORLD frame
            "right_hand_xyz_dot": _forward_finite_diff(right_xyz_w, dt),
            "left_hand_xyz_dot": _forward_finite_diff(left_xyz_w, dt),
            # Angular velocities — all WORLD frame
            "trunk_angular_velocity_world": _angular_velocity_from_quat(trunk_quat_w, dt),
            "right_hand_angular_velocity_world": _angular_velocity_from_quat(right_quat_w, dt),
            "left_hand_angular_velocity_world": _angular_velocity_from_quat(left_quat_w, dt),
        }
```

- [ ] Now find the unused `_world_to_trunk` helper definition (lines 216-226) and delete it. The quat helpers (`_quat_conjugate_wxyz`, `_quat_mul_wxyz`, `_quat_rotate_wxyz`) are still used by `_angular_velocity_from_quat` so keep them.

### Step 3.7: Update the module docstring + add schema_version

- [ ] In `cutamp/utils/plan_processor.py`, replace the module docstring (lines 1-89) so the schema reflects world-frame hand poses. Specifically update the per-segment schema block:

```
        "position": {
            "trunk_xyz":             [T, 3],     # WORLD, REAL-URDF Trunk (-0.0625 X applied)
            "trunk_quat_wxyz":       [T, 4],     # WORLD
            "trunk_quat_xyzw":       [T, 4],
            "trunk_height":          [T],        # alias for trunk_xyz[:, 2]
            "trunk_pitch":           [T],
            "trunk_yaw":             [T],
            "right_arm":             [T, 7],
            "left_arm":              [T, 7],
            # Hand poses in WORLD frame (real-URDF-native)
            "right_hand_xyz":        [T, 3],     # WORLD
            "right_hand_quat_wxyz":  [T, 4],
            "right_hand_quat_xyzw":  [T, 4],
            "left_hand_xyz":         [T, 3],
            "left_hand_quat_wxyz":   [T, 4],
            "left_hand_quat_xyzw":   [T, 4],
        },
        "velocity": {
            ...joint velocities unchanged...
            "trunk_xyz_dot":                     [T, 3],   # m/s, world
            "trunk_height":                      [T],
            "right_hand_xyz_dot":                [T, 3],   # m/s, WORLD
            "left_hand_xyz_dot":                 [T, 3],
            "trunk_angular_velocity_world":      [T, 3],
            "right_hand_angular_velocity_world": [T, 3],   # renamed from _trunk
            "left_hand_angular_velocity_world":  [T, 3],
        },
```

Drop any "in TRUNK frame" mentions and the "Hand poses in TRUNK frame" comment block.

- [ ] In the `process_motion_plan` return statement, add the schema version:

```python
    return {
        "schema_version": 2,
        "segments": processed_segments,
        ...
    }
```

### Step 3.8: Update the consumer example (delete `compensate_trunk_x_offset`, read world-frame fields)

- [ ] Edit `examples/load_motion_plan_for_mpc.py`. Delete the `TRUNK_X_OFFSET_M` constant and the `compensate_trunk_x_offset` function (lines ~35 and ~70-82 per spec).

- [ ] In `load_for_mpc`, remove the line `plan = compensate_trunk_x_offset(plan)`. Add a schema-version guard:

```python
def load_for_mpc(path: Path) -> List[Dict[str, Any]]:
    """One-call: load pickle, build per-segment MPC commands."""
    try:
        with open(path, "rb") as f:
            plan = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"motion_plan.pkl not found at {path}. Generate one with:\n"
            f"  python -m cutamp.scripts.run_cutamp --motion_plan --save_plan {path}"
        )

    schema_version = plan.get("schema_version", 1)
    if schema_version != 2:
        raise RuntimeError(
            f"This example expects schema_version=2, got {schema_version}. "
            f"Regenerate the plan with the current code:\n"
            f"  python -m cutamp.scripts.run_cutamp --motion_plan --save_plan {path}"
        )
    return [segment_to_mpc_commands(seg) for seg in plan["segments"]]
```

- [ ] In `segment_to_mpc_commands`, rename the hand-targets block to reflect world frame:

```python
    hand_targets_in_world = {
        "right": {
            "xyz": P["right_hand_xyz"],
            "quat_xyzw": P["right_hand_quat_xyzw"],
            "xyz_dot": V["right_hand_xyz_dot"],
            "angular_velocity": V["right_hand_angular_velocity_world"],
        },
        "left": {
            "xyz": P["left_hand_xyz"],
            "quat_xyzw": P["left_hand_quat_xyzw"],
            "xyz_dot": V["left_hand_xyz_dot"],
            "angular_velocity": V["left_hand_angular_velocity_world"],
        },
    }
    ...
    return {
        "dt": seg["dt"],
        "T": seg["T"],
        "trunk_world_pose": trunk_world_pose,
        "joint_commands": joint_commands,
        "trunk_joint_velocities": joint_velocities,
        "hand_targets_in_world": hand_targets_in_world,  # was _in_trunk
        "held_objs": seg.get("held_objs", {}),
    }
```

- [ ] Update the module docstring and the `main()` printout to reference `hand_targets_in_world` and to delete the "Trunk X at t=0 ≈ -0.0625" sanity assertion (it no longer applies — `trunk_xyz` is now real-URDF-native and at home pose should be near 0, not -0.0625).

### Step 3.9: Rewrite `docs/sim_to_real_mapping.md`

- [ ] Edit `docs/sim_to_real_mapping.md`. Discrepancies #1 (Trunk 6.25 cm offset), #2 (no world frame on real), and #4 (hand poses in Trunk frame) are now RESOLVED at the source. Update each to ✅ resolved with a short note: "Resolved by Commit 3 (schema_version=2): plan_processor.py applies -0.0625 X to saved trunk_xyz and emits hand poses in WORLD frame. Consumer needs no compensation." Update the TL;DR at the bottom to a 3-step recipe:

```
## TL;DR for the MPC consumer (schema_version=2)

1. Load `motion_plan.pkl` (schema_version=2). Hand poses are world-frame,
   real-URDF-native — no compensation needed.
2. Broadcast `trunk_pitch` to both `Left_Hip_Pitch` and `Right_Hip_Pitch`.
3. Solve your own leg IK (Hip_Roll, Hip_Yaw, Ankle_Pitch, Ankle_Roll,
   Knee_Pitch) from the saved `trunk_xyz` + `trunk_quat_wxyz`.
```

### Step 3.10: Run smoke test — fresh pickle, fresh consumer

- [ ] Delete any stale pickle:

```bash
rm -f /home/yoonwoo/cuTAMP/data/motion_plan.pkl
```

- [ ] Generate a fresh plan:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan \
  --save_plan /home/yoonwoo/cuTAMP/data/motion_plan.pkl 2>&1 | tail -20
```

Expected: ≥1 satisfying, plan saved to `data/motion_plan.pkl`.

- [ ] Run the consumer example:

```bash
/home/yoonwoo/miniconda3/envs/tamp/bin/python /home/yoonwoo/cuTAMP/examples/load_motion_plan_for_mpc.py 2>&1 | tail -30
```

Expected: Loads, prints "Segments: N", prints "Trunk world xyz at t=0" near (0, 0, 0.67) (standing home pose, real-URDF Trunk). No exception.

### Step 3.11: Run plan_processor unit tests

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_plan_processor.py -v 2>&1 | tail -15
```

Expected: both tests PASS.

### Step 3.12: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/utils/plan_processor.py \
  examples/load_motion_plan_for_mpc.py \
  docs/sim_to_real_mapping.md \
  cutamp/tests/test_plan_processor.py && \
git commit -m "$(cat <<'EOF'
fix!: plan_processor schema_version=2 — world-frame hand poses

BREAKING: motion_plan.pkl schema changes. v1 pickles will be rejected
by the consumer with a clear message; regenerate with --save_plan.

- F1 revised: FK target switched from *_base_link (palm, not in
  actual_robot.urdf) to *_hand_link (in both URDFs). Hand poses now
  emitted in WORLD frame, not Trunk frame. Manipulation MPCs work in
  world; consumer no longer needs frame transforms.
- F1 corollary: -0.0625 X compensation applied to trunk_xyz inside
  plan_processor.py. Saved trunk_xyz now represents real-URDF Trunk
  world pose, not sim's. Consumer needs no compensation.
- F1 corollary: compensate_trunk_x_offset helper deleted from the
  example (its work is done upstream). F2 (idempotency) obsoleted.
- F1 corollary: plan["schema_version"] = 2 lets the consumer reject
  stale v1 pickles with a clear regen instruction.
- D2: _to_active_cspace now compares joint NAMES (not just DOF count)
  before short-circuiting. Same-shape-different-order tensors fall
  through to the reorder branch.
- D3: _angular_velocity_from_quat canonicalizes quaternion sign with
  a sequential forward sweep. Eliminates the ~4/dt rad/s spurious ω
  spike that fired on any FK sign flip.
- Docs: sim_to_real_mapping.md discrepancies #1, #2, #4 now ✅
  resolved at the source. TL;DR shrinks from 6 steps to 3.

Findings: F1 (revised), D2, D3. (F2 obsoleted.)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Commit 4 — pin atomicity

**Findings**: C2 (pin_for_movebase no guard), C3 (`_apply_pin` non-atomic), S5 (`_disabled_tool_pose_frames` set by caller).

**Files:**
- Modify: `cutamp/t1_state.py:76-178`
- Create: `cutamp/tests/test_pin_lifecycle.py`

### Step 4.1: Write the failing test for double-pin guard (C2)

- [ ] Create `cutamp/tests/test_pin_lifecycle.py`:

```python
"""Pin lifecycle invariants for T1State."""
import pytest


def _make_state():
    """Build a real T1State with a real planner. We use the smallest possible
    setup so the test runs in a few seconds."""
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.config import TAMPConfiguration
    import os

    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    config = TAMPConfiguration(
        robot="t1", env="blocks_t1", num_particles=4, disable_visualizer=True,
    )
    world = TAMPWorld(env, config)
    from cutamp.t1_state import T1State
    from curobo.types import JointState
    planner = world.get_motion_planner()
    js = JointState.from_position(
        world.q_init[:1].to(planner.kinematics.device_cfg.device),
    )
    return T1State(
        planner=planner,
        kinematics=world.kinematics,
        tool_from_ee=world.tool_from_ee,
        current_js=planner.kinematics.get_active_js(js),
    )


def test_double_pin_for_movebase_raises():
    """Calling pin_for_movebase while a pin is active must raise, matching
    pin_for_arm_action's guard."""
    state = _make_state()
    state.pin_for_movebase()
    try:
        with pytest.raises(RuntimeError, match="pin"):
            state.pin_for_movebase()
    finally:
        state.unpin()


def test_unpin_re_enables_tool_pose_after_partial_apply():
    """Even if pin_for_movebase succeeds in disabling tool_pose criteria,
    unpin must always re-enable them. Specifically: after a successful pin
    + unpin cycle, calling pin_for_movebase a second time must work
    (snapshot is None again), proving unpin fully cleared state."""
    state = _make_state()
    state.pin_for_movebase()
    state.unpin()
    # Second cycle — must not raise.
    state.pin_for_movebase()
    state.unpin()
```

### Step 4.2: Run the tests — should fail

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_pin_lifecycle.py -v 2>&1 | tail -20
```

Expected: `test_double_pin_for_movebase_raises` FAILS (currently no guard). `test_unpin_re_enables_tool_pose_after_partial_apply` likely PASSES today (the second cycle works because unpin clears `_saved_target_dof_weight`); keep it as a regression guard.

### Step 4.3: Move `_disabled_tool_pose_frames` bookkeeping into `_apply_pin` (C3 + S5)

- [ ] Edit `cutamp/t1_state.py:76-112`. Replace the body of `_apply_pin` with:

```python
    def _apply_pin(
        self,
        pin_joint_names: list,
        disabled_tool_frames: list,
        pin_weight: float,
        default_weight: float,
        hosts: Iterable[Any],
    ) -> None:
        """Set per-DOF cspace weights (joints addressed by NAME) on every host.

        Joint names that aren't in the planner's active cspace are silently
        skipped — e.g., a base DOF requested for pinning when the base is
        already locked has nothing to pin in the active cspace.

        Bookkeeping for ``_disabled_tool_pose_frames`` is set BEFORE the
        planner-mutating call so ``unpin`` can recover any partial state if
        ``update_tool_pose_criteria`` raises.
        """
        hosts = list(hosts)
        active_names = self._planner_joint_names
        weights = torch.full(
            (len(active_names),),
            default_weight,
            device=self.current_js.position.device,
            dtype=self.current_js.position.dtype,
        )
        name_to_idx = {n: i for i, n in enumerate(active_names)}
        for n in pin_joint_names:
            i = name_to_idx.get(n)
            if i is not None:
                weights[i] = pin_weight

        if self._saved_target_dof_weight is None:
            self._saved_target_dof_weight = snapshot_cspace_target_dof_weight(hosts)
            self._saved_pin_hosts = hosts

        # Record the disabled frames BEFORE mutating the planner so unpin
        # can re-enable them even if update_tool_pose_criteria raises
        # mid-mutation. Use list() to capture by value.
        self._disabled_tool_pose_frames = list(disabled_tool_frames)

        write_cspace_target_dof_weight(hosts, weights)

        if disabled_tool_frames:
            self.planner.update_tool_pose_criteria(
                {f: ToolPoseCriteria.disabled() for f in disabled_tool_frames}
            )
```

- [ ] In the same file, `pin_for_movebase` (lines 157-178) currently sets `self._disabled_tool_pose_frames = [...]` AFTER `_apply_pin` returns. Now `_apply_pin` does this work; delete the redundant assignment:

```python
    def pin_for_movebase(
        self,
        pin_weight: float = 1000.0,
        default_weight: float = 1.0,
        *,
        hosts: Optional[Iterable[Any]] = None,
    ) -> None:
        """Lock body + both arms; only the base DOFs remain free.

        WARNING: the planner config locks the mobile base, so MoveBaseTo
        cannot actually move the base in this planner. Envs needing
        navigation must build a separate planner without the base lock.

        Idempotency: caller must ``unpin()`` before a second pin.
        """
        if self._saved_target_dof_weight is not None:
            raise RuntimeError(
                "pin_for_movebase called while a pin is already active; "
                "call unpin() first."
            )
        body_names = list(JOINT_NAMES_FULL[BODY_INDICES])
        self._apply_pin(
            pin_joint_names=body_names + list(LEFT_ARM_JOINT_NAMES) + list(RIGHT_ARM_JOINT_NAMES),
            disabled_tool_frames=[LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME],
            pin_weight=pin_weight,
            default_weight=default_weight,
            hosts=hosts if hosts is not None else [self.planner],
        )
        # `_disabled_tool_pose_frames` is now set inside `_apply_pin`.
```

Note both changes happen together: the C2 guard (`if self._saved_target_dof_weight is not None: raise`) is added at the top of `pin_for_movebase`, AND the redundant trailing assignment is deleted.

### Step 4.4: Run tests — both should PASS

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_pin_lifecycle.py -v 2>&1 | tail -20
```

Expected: both PASS.

### Step 4.5: Smoke test — pin/unpin still works end-to-end

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan 2>&1 | tail -20
```

Expected: ≥1 satisfying. Pin happens for every arm operator so this exercises the new bookkeeping path.

### Step 4.6: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/t1_state.py \
  cutamp/tests/test_pin_lifecycle.py && \
git commit -m "$(cat <<'EOF'
fix: T1State pin atomicity + double-pin guard

- _apply_pin now sets self._disabled_tool_pose_frames BEFORE calling
  update_tool_pose_criteria. If the planner mutation raises partway
  through, unpin() can recover instead of leaving tool-pose criteria
  silently disabled forever.
- pin_for_movebase gains the same already-pinned guard pin_for_arm_action
  has. A misordered double-pin now raises with a clear message instead
  of silently losing the original snapshot.

Findings: C2, C3, S5.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Commit 5 — misc fixes

**Findings**: E3 (stale imports), E4 (numpy aliasing), B2 (squeeze loop), F4 (pickle except + safety doc).

**Files:**
- Modify: `cutamp/tests/debug_t1_tool_frame.py:87-92`
- Modify: `examples/load_motion_plan_for_mpc.py` (.copy() on broadcast joints + broader except)
- Modify: `cutamp/_curobo_internals.py:74-75`
- Modify: `docs/sim_to_real_mapping.md` (pickle safety note)
- Create: `cutamp/tests/test_curobo_internals.py`

### Step 5.1: Write the failing test for `cspace_plan_succeeded` squeeze loop (B2)

- [ ] Create `cutamp/tests/test_curobo_internals.py`:

```python
"""Unit tests for cuRobo workarounds in _curobo_internals.py."""
from types import SimpleNamespace
import pytest
import torch


def _make_plan_result(positions, success=False, cspace_error=None,
                     joint_names=None):
    """Build a minimal duck-typed result object that cspace_plan_succeeded
    will accept."""
    plan = SimpleNamespace(
        position=torch.tensor(positions),
        joint_names=joint_names,
    )
    return SimpleNamespace(
        success=torch.tensor([success]),
        cspace_error=cspace_error,
        interpolated_trajectory=plan,
        js_solution=None,
    )


def test_cspace_plan_succeeded_inspects_all_batches():
    """For a [B, T, dof] plan (sim returns these), the salvage path must
    check every batch element's last timestep, not just the last batch.
    Regression for the > 2 vs >= 2 squeeze-loop bug."""
    from cutamp._curobo_internals import cspace_plan_succeeded

    # Two batches, T=2 timesteps, dof=3. Batch 0 lands EXACTLY on goal;
    # batch 1 is far off. The salvage path should report success because
    # at least one batch met tolerance — UNLESS the squeeze loop drops
    # batch 0 by picking only the last batch.
    positions = [
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],   # batch 0 ends at goal
        [[0.0, 0.0, 0.0], [9.9, 9.9, 9.9]],   # batch 1 ends far away
    ]
    target = SimpleNamespace(
        position=torch.tensor([0.5, 0.5, 0.5]),
        joint_names=None,
    )
    result = _make_plan_result(positions, success=False, cspace_error=None)
    assert cspace_plan_succeeded(result, target, tol=1e-3) is True, (
        "cspace_plan_succeeded missed batch 0's success because the "
        "squeeze loop dropped earlier batches."
    )
```

### Step 5.2: Run the test — should FAIL

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_curobo_internals.py -v 2>&1 | tail -15
```

Expected: FAIL — the current squeeze loop picks the last batch only.

### Step 5.3: Fix the squeeze loop (B2)

The salvage path needs to check whether ANY batch's last timestep is within tolerance of the goal — not collapse the batch axis. Replace lines 67-89 of `cutamp/_curobo_internals.py` wholesale:

- [ ] Edit `cutamp/_curobo_internals.py:67-89`. Find:

```python
    plan = (
        getattr(result, "interpolated_trajectory", None)
        or getattr(result, "js_solution", None)
    )
    if plan is None:
        return False
    end = plan.position
    while end.dim() > 1:
        end = end[..., -1, :] if end.dim() > 2 else end[-1]
    # Align dims: ``end`` may be in the planner's all-articulated order
    # (e.g., 31 with locked joints) while ``target_js`` is in active cspace
    # (e.g., 18). Pick out the active-cspace entries from ``end`` by name.
    plan_names = getattr(plan, "joint_names", None)
    target_names = getattr(target_js, "joint_names", None)
    if end.shape[0] != target_js.position.shape[-1]:
        if plan_names is None or target_names is None:
            return False
        idx = [plan_names.index(n) for n in target_names if n in plan_names]
        if len(idx) != target_js.position.shape[-1]:
            return False
        end = end[idx]
    goal = target_js.position.view(-1).to(end)
    return torch.max(torch.abs(end - goal)).item() < tol
```

Replace with:

```python
    plan = (
        getattr(result, "interpolated_trajectory", None)
        or getattr(result, "js_solution", None)
    )
    if plan is None:
        return False
    pos = plan.position
    # cuRobo plans are [..., T, dof]. Take the last timestep along the time
    # axis (second-to-last). After this, `end` is [B?, dof]. We then check
    # whether ANY batch element lands within tolerance — previously the
    # `> 2` vs `else end[-1]` squeeze loop dropped all but the last batch.
    if pos.dim() >= 2:
        end = pos[..., -1, :]
    else:
        end = pos
    if end.dim() == 1:
        end = end.unsqueeze(0)   # treat single-batch as [1, dof]
    # Align dims: ``end`` may be in the planner's all-articulated order
    # (e.g., 31 with locked joints) while ``target_js`` is in active cspace
    # (e.g., 18). Pick out the active-cspace entries from ``end`` by name.
    plan_names = getattr(plan, "joint_names", None)
    target_names = getattr(target_js, "joint_names", None)
    if end.shape[-1] != target_js.position.shape[-1]:
        if plan_names is None or target_names is None:
            return False
        idx = [plan_names.index(n) for n in target_names if n in plan_names]
        if len(idx) != target_js.position.shape[-1]:
            return False
        end = end[..., idx]
    goal = target_js.position.view(-1).to(end)
    # Per-batch max-abs-error against goal; True if any batch is within tol.
    per_batch_err = torch.max(torch.abs(end - goal), dim=-1).values
    return bool((per_batch_err < tol).any().item())
```

### Step 5.4: Run the test — should PASS

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_curobo_internals.py -v 2>&1 | tail -10
```

Expected: PASS. (The test feeds a `[B=2, T=2, dof=3]` tensor where batch 0 reaches the goal and batch 1 doesn't; the fix should report success because batch 0 is within tolerance.)

### Step 5.5: Fix stale imports in `debug_t1_tool_frame.py` (E3)

- [ ] Edit `cutamp/tests/debug_t1_tool_frame.py`. The current file imports `get_t1_kinematics_model`, `t1_home_left`, `load_t1_rerun` and calls `get_t1_kinematics_model("left")` and `load_t1_rerun(load_mesh=True, arm="left")`.

First check what's actually in the file:

```bash
grep -n "from cutamp.robots.t1\|get_t1_kinematics_model\|t1_home_left\|t1_home_right\|load_t1_rerun" \
  /home/yoonwoo/cuTAMP/cutamp/tests/debug_t1_tool_frame.py
```

Then replace the imports with the current API:

```python
from cutamp.robots.t1 import get_t1_kinematics, t1_home, load_t1_rerun
```

And update each call site:
- `get_t1_kinematics_model("left")` / `get_t1_kinematics_model("right")` → `get_t1_kinematics()` (single shared kinematics; arm-specific tool frame selected via `tool_frame_for_arm` if needed)
- `t1_home_left` / `t1_home_right` → `t1_home` (single shared home pose)
- `load_t1_rerun(load_mesh=True, arm="left")` → `load_t1_rerun(load_mesh=True)`

If the script depends on per-arm specifics that don't exist in the new API (e.g. arm-specific kinematics that was a real per-arm model in the old v0.7 codebase), and the script is purely diagnostic, the minimum-viable fix is to update it so it imports without error and runs to first checkpoint. If it relies on functionality that no longer exists, the script may need substantive rewrite — note in the commit message and minimize scope (delete dead diagnostic paths, keep the bits that still apply).

### Step 5.6: Verify the debug script imports cleanly

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -c \
  "import cutamp.tests.debug_t1_tool_frame" 2>&1 | tail -5
```

Expected: no `ImportError`. (TypeError from running the script body is fine — we only need imports to succeed.)

### Step 5.7: Fix broadcast-joint aliasing in the example (E4)

- [ ] Edit `examples/load_motion_plan_for_mpc.py`. Find the broadcast loop:

```python
    for sim_field, real_names in [("trunk_pitch", JOINT_MAP["trunk_pitch"]),
                                  ("trunk_yaw",   JOINT_MAP["trunk_yaw"])]:
        for rn in real_names:
            joint_commands[rn] = P[sim_field]
            joint_velocities[rn] = V[sim_field]
```

Replace with:

```python
    # .copy() per assignment so mutation of one broadcast target doesn't
    # silently mutate the other (Left_Hip_Pitch and Right_Hip_Pitch share
    # the same source array; MPC asymmetric trim would otherwise apply
    # both biases to both joints).
    for sim_field, real_names in [("trunk_pitch", JOINT_MAP["trunk_pitch"]),
                                  ("trunk_yaw",   JOINT_MAP["trunk_yaw"])]:
        for rn in real_names:
            joint_commands[rn] = P[sim_field].copy()
            joint_velocities[rn] = V[sim_field].copy()
```

### Step 5.8: Broaden pickle except + add safety comment (F4)

- [ ] Edit `examples/load_motion_plan_for_mpc.py`. Find the pickle load block:

```python
    try:
        with open(path, "rb") as f:
            plan = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"motion_plan.pkl not found at {path}. Generate one with:\n"
            f"  python -m cutamp.scripts.run_cutamp --motion_plan --save_plan {path}"
        )
```

Replace with:

```python
    # WARNING: pickle.load executes arbitrary Python from the source file.
    # Only load motion_plan.pkl from trusted sources (own filesystem,
    # generated by cutamp on a machine you control). An attacker-supplied
    # pickle = arbitrary code execution.
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
            f"motion_plan.pkl at {path} appears corrupt or incomplete "
            f"({type(e).__name__}: {e}). Regenerate with --save_plan."
        )
    except AttributeError as e:
        raise RuntimeError(
            f"motion_plan.pkl at {path} references a class that's been "
            f"renamed/removed ({e}). Schema likely evolved; regenerate the plan."
        )
```

### Step 5.9: Add pickle-safety note to docs

- [ ] Edit `docs/sim_to_real_mapping.md`. Append a section near the end:

```markdown
## Pickle safety

`motion_plan.pkl` uses Python pickle. Loading a pickle from any source
executes arbitrary code inside that file — if you `scp` a teammate's
pickle and load it, you are running their code with your permissions.
Mitigations:

- Only load from filesystems you control.
- If you need to share plans across hosts, share the source script + CLI
  invocation and regenerate, not the pickle itself.
- Long-term migration to a safe format (msgpack / safetensors) is on the
  backlog but not currently scheduled.
```

### Step 5.10: Verify all tests pass

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/ -v 2>&1 | tail -40
```

Expected: all tests PASS (including the new tests from commits 1, 3, 4, 5).

### Step 5.11: Final end-to-end smoke

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan \
  --save_plan /home/yoonwoo/cuTAMP/data/motion_plan.pkl 2>&1 | tail -20
```

Then:

```bash
/home/yoonwoo/miniconda3/envs/tamp/bin/python \
  /home/yoonwoo/cuTAMP/examples/load_motion_plan_for_mpc.py 2>&1 | tail -20
```

Expected: both pass; consumer loads schema_version=2 plan without error.

### Step 5.12: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/tests/debug_t1_tool_frame.py \
  examples/load_motion_plan_for_mpc.py \
  cutamp/_curobo_internals.py \
  docs/sim_to_real_mapping.md \
  cutamp/tests/test_curobo_internals.py && \
git commit -m "$(cat <<'EOF'
fix: misc — debug script imports + numpy aliasing + squeeze loop + pickle safety

- cspace_plan_succeeded salvage path now checks EVERY batch's last-
  timestep against the cspace goal, not just the last batch. Previous
  squeeze-loop logic (`> 2` vs `else end[-1]`) silently dropped earlier
  batches on the [B, T, dof] shape that cuRobo's plan_cspace returns,
  causing the salvage path to report False for plans where an earlier
  seed succeeded.
- tests/debug_t1_tool_frame.py imports updated to current API
  (get_t1_kinematics / t1_home / load_t1_rerun without arm= kwarg).
  The old per-arm API (get_t1_kinematics_model, t1_home_left,
  load_t1_rerun(arm=...)) was removed during the cuRobo v0.7→v0.8 port.
- examples/load_motion_plan_for_mpc.py: .copy() per assignment in the
  broadcast-joint loop so Left/Right_Hip_Pitch don't alias the same
  numpy array. MPC asymmetric trim was otherwise applying both biases
  to both joints.
- examples/load_motion_plan_for_mpc.py: pickle.load now catches
  EOFError / UnpicklingError / AttributeError with friendlier messages.
  Added a # WARNING block about pickle being unsafe with untrusted files.
- docs/sim_to_real_mapping.md: pickle-safety section.

Findings: B2, E3, E4, F4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

### Step F.1: Full test suite

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/ -v 2>&1 | tail -50
```

Expected: all PASS.

### Step F.2: Smoke test (default config)

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan 2>&1 | tail -20
```

Expected: ≥1 satisfying, motion plans succeed.

### Step F.3: Smoke test (coupled_reik + place_close_to_base)

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan \
  --coupled_reik --optimize_soft_costs --soft_cost place_close_to_base 2>&1 | tail -20
```

Expected: ≥1 satisfying. This exercises Commit 1's fix end-to-end.

### Step F.4: Smoke test (--no_enable_com_polygon)

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan \
  --no_enable_com_polygon 2>&1 | tail -20
```

Expected: ≥1 satisfying. This exercises Commit 2's F5 fix.

### Step F.5: Consumer round-trip

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan \
  --save_plan /home/yoonwoo/cuTAMP/data/motion_plan.pkl && \
/home/yoonwoo/miniconda3/envs/tamp/bin/python \
  /home/yoonwoo/cuTAMP/examples/load_motion_plan_for_mpc.py 2>&1 | tail -30
```

Expected: pickle generated, consumer loads it, prints world-frame Trunk pose near (0, 0, 0.67) at t=0.

### Step F.6: Review the commit log

- [ ] Run:

```bash
git log --oneline curobo_v2 -10
```

Expected: 5 new commits in order (Commit 1 → Commit 5), all with `Co-Authored-By: Claude Opus 4.7` lines.

---

## Self-review checklist (executed by plan-writer, post-write)

**1. Spec coverage**: All 15 findings (minus S4 refuted) are addressed across the 5 commits. F2 obsoleted by Commit 3's deletion of `compensate_trunk_x_offset`. ✅

**2. Placeholder scan**: No "TBD", "TODO", "implement later". Concrete code snippets in every implementation step. ✅

**3. Type consistency**: `_ik_for_pose` signature change is used by `_refresh_ik_deps` (already calls with `current_state_q=`) — call site already correct after Step 1.3. ✅

**4. Investigation findings folded in**: S4 verified REFUTED during plan-writing; dropped from Commit 2 and noted in commit message. ✅
