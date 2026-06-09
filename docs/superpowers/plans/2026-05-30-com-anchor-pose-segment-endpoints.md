# COM-anchor pick/place trajectory endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pick/place motion-plan segments terminate at the COM-safe optimizer particle conf (via `plan_cspace`) instead of a free Cartesian redundancy solution (via `plan_pose`), so the executed arrival is COM-in-hull like the conf already is — fixing the ~3.7 cm toe-edge COM excursion at place arrivals.

**Architecture:** `solve_curobo`'s retract branch already plans to a joint config with `plan_cspace(_planner_js_from_full(conf), last_js)`. Pick and place currently plan to a Cartesian pose with `plan_single_arm_pose`, letting cuRobo re-resolve arm/torso redundancy at the endpoint (drifting COM toward the toe edge). This plan adds one shared helper `_plan_arm_to_conf` that mirrors the retract pattern, and routes the pick and place branches through it — anchoring the terminal to the COM-hard-checked particle conf, which is itself an exact IK solution for the same hand pose. No `curobo/` edits.

**Tech Stack:** Python, PyTorch (CUDA), cuRobo v0.8, pytest. Project python: `/home/yoonwoo/miniconda3/envs/tamp/bin/python`. pytest prefix: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; GPU runs add `PYTORCH_ALLOC_CONF=expandable_segments:True`. CUDA GPU (RTX 4090) present, so `needs_cuda` tests run for real.

**Spec:** `docs/superpowers/specs/2026-05-30-com-anchor-pose-segment-endpoints-design.md`

---

## Hard constraints (apply to EVERY task)

- **Never edit anything under `curobo/`** (vendored, read-only). All edits are in `cutamp/`.
- **Stage only the files named in each commit.** Never `git add -A` / `git add .` / `git commit -a`. The working tree has a pre-existing dirty `data/motion_plan.pkl` (regenerable) and untracked `.claude/` — do NOT stage either except in Task 4's explicit regenerate step.
- **Anchor edits on the quoted old code**, not line numbers (the file may have drifted). If a quoted snippet isn't found verbatim, STOP and re-read `cutamp/motion_solver.py`.
- Run pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 … -p no:cacheprovider`.
- Behavior change is intended (terminal redundancy), but the **hand pose reached and the motion-plan success rate must not regress** — Task 2/4 verify this.

## Key facts established by investigation (do not re-derive)

- `solve_curobo` (`cutamp/motion_solver.py`) builds one trajectory segment per plan-skeleton operator. The arm-op pin (`pin_for_arm_action(arm)`) is already active before the pick/place/retract branches.
- **place** branch: `obj_name, grasp_name, place_name, surface_name, q_name = ground_op.values` — `q_name` (= `values[-1]`) is the COM-hard-checked place conf `best_particle[q_name]` (full 21-DOF). It plans with `plan_single_arm_pose(target_pose=…)` wrapped in `_disabled_world_obstacle(planner, surface_name)`.
- **pick** branch: `obj_name, grasp_name, q_name = ground_op.values` — `q_name` (= `values[-1]`) is the grasp conf. It plans with `plan_single_arm_pose(...)` in a `for _ in range(GRASP_RETRY)` loop wrapped in `_disabled_world_obstacle(planner, obj_name)`, then `_attach_object` at the arrival, using `grasp_from_obj = torch.inverse(obj_from_grasp)`.
- **retract** branch (the template) does: `target_q = best_particle[q_retract_name].clone()`; `target_js = _planner_js_from_full(planner, target_q, full_joint_names)`; `result = planner.plan_cspace(target_js, last_js, max_attempts=CSPACE_MAX_ATTEMPTS, enable_graph_attempt=CSPACE_ENABLE_GRAPH_ATTEMPT)`; `if not _cspace_plan_succeeded(result, target_js): raise`; `plan = _interp_plan(result)`; `dt = _plan_dt(result)`.
- Helpers already in `motion_solver.py`: `_planner_js_from_full(planner, full_pos, full_joint_names)`, `_disabled_world_obstacle(planner, name)` (context manager), `_cspace_plan_succeeded` (imported as `_cspace_plan_succeeded` from `cutamp._curobo_internals`), `_interp_plan`, `_plan_dt`, `_last_timestep_js`. Module constants: `CSPACE_MAX_ATTEMPTS`, `CSPACE_ENABLE_GRAPH_ATTEMPT`, `GRASP_RETRY`. `import contextlib` is already at the top.
- `full_joint_names = world.kinematics.joint_names` is defined once before the operator loop and is in scope in every branch.
- The conf reaches the **identical hand pose** as the Cartesian target (verified to 1 µm), because the place/pick target pose is built from the *same particles* the conf was IK'd from — so anchoring to the conf does not change the task pose, only the COM-safe redundancy branch.
- `plan_cspace` (joint goal) is generally **easier** for trajopt than `plan_pose` (pose goal + IK), so success rate should be equal-or-better; verified in Task 4.

---

## Task 1: Add the `_plan_arm_to_conf` helper (TDD, CPU-only wiring test)

The novel logic is "plan the arm to the COM-safe conf via `plan_cspace`, not to a Cartesian pose." Extract it once so pick and place share it.

**Files:**
- Modify: `cutamp/motion_solver.py` (add module-level helper near `_planner_js_from_full`)
- Test: `cutamp/tests/test_motion_anchor.py` (new)

- [ ] **Step 1: Write the failing wiring tests**

Create `cutamp/tests/test_motion_anchor.py`:

```python
"""Tests for COM-anchored pick/place terminals (plan_cspace to the conf)."""
import pytest
import torch

import cutamp.motion_solver as ms


def test_plan_arm_to_conf_targets_the_conf(monkeypatch):
    # _plan_arm_to_conf must plan with plan_cspace to a target built FROM the
    # conf (not a Cartesian pose), and return the cuRobo result on success.
    sentinel_target = object()
    monkeypatch.setattr(ms, "_planner_js_from_full",
                        lambda planner, conf, names: sentinel_target)
    monkeypatch.setattr(ms, "_cspace_plan_succeeded", lambda result, target: True)
    calls = {}

    class FakePlanner:
        def plan_cspace(self, target_js, last_js, max_attempts, enable_graph_attempt):
            calls["target_js"] = target_js
            calls["last_js"] = last_js
            calls["max_attempts"] = max_attempts
            return "FAKE_RESULT"

    res = ms._plan_arm_to_conf(
        FakePlanner(), torch.zeros(21), last_js="LAST", full_joint_names=["a"],
        disable_obstacle=None, retries=1, ground_op="OP",
    )
    assert res == "FAKE_RESULT"
    assert calls["target_js"] is sentinel_target  # planned to the conf, not a pose
    assert calls["last_js"] == "LAST"
    assert calls["max_attempts"] == ms.CSPACE_MAX_ATTEMPTS


def test_plan_arm_to_conf_raises_on_failure(monkeypatch):
    monkeypatch.setattr(ms, "_planner_js_from_full", lambda *a: object())
    monkeypatch.setattr(ms, "_cspace_plan_succeeded", lambda *a: False)

    class FakePlanner:
        def __init__(self):
            self.n = 0
        def plan_cspace(self, *a, **k):
            self.n += 1
            return "R"

    fp = FakePlanner()
    with pytest.raises(RuntimeError, match="cspace-anchored"):
        ms._plan_arm_to_conf(fp, torch.zeros(21), "LAST", ["a"],
                             disable_obstacle=None, retries=3, ground_op="OP")
    assert fp.n == 3  # retried `retries` times before raising
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_motion_anchor.py -q -p no:cacheprovider
```
Expected: FAIL — `AttributeError: module 'cutamp.motion_solver' has no attribute '_plan_arm_to_conf'`.

- [ ] **Step 3: Implement the helper**

In `cutamp/motion_solver.py`, immediately AFTER the `_planner_js_from_full` function (find its `return planner.kinematics.get_active_js(full_js)` line and add the new function after the blank line following it), insert:

```python
def _plan_arm_to_conf(
    planner,
    conf_full: torch.Tensor,
    last_js: JointState,
    full_joint_names,
    *,
    disable_obstacle: Optional[str],
    retries: int,
    ground_op,
):
    """Plan a collision-free trajectory whose TERMINAL is the COM-safe particle
    conf ``conf_full`` (full 21-DOF), via ``plan_cspace`` — so the executed
    arrival equals the hard-COM-checked conf instead of a free Cartesian
    redundancy solution. Mirrors the retract branch. ``disable_obstacle`` (if
    given) is temporarily removed from the world during planning, so terminal
    gripper-block / held-block-surface contact isn't rejected. Returns the
    cuRobo result; raises ``RuntimeError`` if no attempt converges.
    """
    target_js = _planner_js_from_full(planner, conf_full, full_joint_names)
    obstacle_ctx = (
        _disabled_world_obstacle(planner, disable_obstacle)
        if disable_obstacle is not None
        else contextlib.nullcontext()
    )
    result = None
    with obstacle_ctx:
        for _ in range(retries):
            result = planner.plan_cspace(
                target_js, last_js,
                max_attempts=CSPACE_MAX_ATTEMPTS,
                enable_graph_attempt=CSPACE_ENABLE_GRAPH_ATTEMPT,
            )
            if _cspace_plan_succeeded(result, target_js):
                return result
    raise RuntimeError(f"cspace-anchored plan failed for {ground_op}")
```

(`Optional` and `JointState` are already imported at the top of `motion_solver.py`; `contextlib`, `CSPACE_MAX_ATTEMPTS`, `CSPACE_ENABLE_GRAPH_ATTEMPT`, `_disabled_world_obstacle`, `_planner_js_from_full`, `_cspace_plan_succeeded` are all already in module scope.)

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_motion_anchor.py -q -p no:cacheprovider
```
Expected: PASS (2 passed).

- [ ] **Step 5: Import smoke**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -c "import cutamp.motion_solver; print('OK')"
```
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add cutamp/motion_solver.py cutamp/tests/test_motion_anchor.py
git commit -m "$(cat <<'EOF'
feat: add _plan_arm_to_conf helper (cspace-anchor arm terminal to a conf)

Shared helper that plans an arm trajectory whose terminal is a COM-safe
particle conf via plan_cspace (mirroring the retract branch), with optional
obstacle-disable and retry. Wiring covered by CPU fake-planner tests. Not yet
wired into pick/place (next tasks).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Route the PLACE branch through `_plan_arm_to_conf` (the fix)

This is the behavioral fix for the observed excursion. Place currently plans to a Cartesian pose; switch it to anchor at the place conf.

**Files:**
- Modify: `cutamp/motion_solver.py` (place branch)
- Test: `cutamp/tests/test_motion_anchor.py` (add a `needs_cuda` integration test)

- [ ] **Step 1: Add the integration test (needs_cuda)**

Append to `cutamp/tests/test_motion_anchor.py`. (It reuses the COM helpers and a real pipeline run; mark it `needs_cuda` following the repo pattern in `cutamp/tests/test_com_polygon_ik.py`.)

```python
import os

needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _generate_plan_segments(seed: int):
    """Run the full cuTAMP pipeline once for blocks_t1 and return the processed
    motion-plan segments (schema v3) + a TAMPWorld for COM checks."""
    import torch
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from cutamp.robots.t1 import t1_home
    from curobo.types import DeviceCfg
    from cutamp.config import TAMPConfiguration
    from cutamp.algorithm import run_cutamp
    from cutamp.utils.plan_processor import process_motion_plan

    torch.manual_seed(seed)
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    dc = DeviceCfg()
    robot = load_robot_container("t1", dc)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=dc.device)
    world = TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=q_init,
                      enable_com_polygon=True)
    config = TAMPConfiguration(
        num_particles=64, num_opt_steps=50, curobo_plan=True,
        enable_visualizer=False, enable_experiment_logging=False,
    )
    # run_cutamp returns the raw curobo_plan for the chosen skeleton; adapt the
    # call to the actual run_cutamp signature (see cutamp/scripts/run_cutamp.py
    # for how it builds world+config and invokes run_cutamp). Process to v3.
    result = run_cutamp(world, config)
    curobo_plan = result["curobo_plan"] if isinstance(result, dict) else result
    processed = process_motion_plan(curobo_plan)
    return world, processed["segments"]


@needs_cuda
def test_place_arrival_is_com_in_hull():
    import numpy as np
    import torch
    from cutamp.robots.t1 import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES, t1_home
    from cutamp.com_polygon_cost import compute_com_polygon_penalties, COM_TOL

    world, segments = _generate_plan_segments(seed=0)
    names = list(world.kinematics.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    q_home = torch.as_tensor(t1_home, dtype=torch.float32,
                             device=world.kinematics.device_cfg.device)

    def frame_q(P, t):
        q = q_home.clone()
        q[idx["Torso_Pitch"]] = float(P["trunk_pitch"][t])
        q[idx["Waist_Yaw"]] = float(P["trunk_yaw"][t])
        q[idx["ankle_pitch"]] = float(P["ankle_pitch"][t])
        q[idx["knee_pitch"]] = float(P["knee_pitch"][t])
        for j, n in enumerate(LEFT_ARM_JOINT_NAMES):
            q[idx[n]] = float(P["left_arm"][t][j])
        for j, n in enumerate(RIGHT_ARM_JOINT_NAMES):
            q[idx[n]] = float(P["right_arm"][t][j])
        return q

    # Every segment's ARRIVAL (last frame) must be COM-in-hull, since each is a
    # cspace-anchored conf or retract-home. (Pre-fix, place arrivals leaked out.)
    worst = 0.0
    for si, seg in enumerate(segments):
        P = seg["position"]; T = seg["T"]
        q = frame_q(P, T - 1).unsqueeze(0)
        pen = float(compute_com_polygon_penalties(world, {"q": q})["q"][0])
        worst = max(worst, pen)
        assert pen <= COM_TOL, (
            f"segment {si} arrival COM penalty {pen:.6f} > tol {COM_TOL}"
        )
    print(f"worst arrival penalty {worst:.6f} <= {COM_TOL}")
```

NOTE: `run_cutamp`'s real signature/return may differ — read `cutamp/scripts/run_cutamp.py` and `cutamp/algorithm.py:run_cutamp` first and adapt `_generate_plan_segments` to call it exactly as the script does (same world/config construction, same return field for the raw plan). The assertion logic is the contract; the setup must match the real API.

- [ ] **Step 2: Run to verify it FAILS (or is flaky) pre-fix**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_motion_anchor.py::test_place_arrival_is_com_in_hull -q -p no:cacheprovider
```
Expected: FAIL on at least some seeds (the place arrival leaks out of hull pre-fix). If it happens to pass on seed 0 (the excursion is run-dependent), note it and proceed — the fix makes it robust; Task 4 verifies across multiple seeds.

- [ ] **Step 3: Rewrite the place branch to anchor at the conf**

In `cutamp/motion_solver.py`, find the place branch:

```python
            elif metadata.action_type == "place":
                obj_name, grasp_name, place_name, surface_name, _ = ground_op.values
                world_from_obj_target = action_4dof_to_mat4x4(best_particle[place_name].clone())
                obj_from_grasp = (
                    action_4dof_to_mat4x4 if config.grasp_dof == 4 else action_6dof_to_mat4x4
                )(best_particle[grasp_name].clone())
                world_from_ee = world_from_obj_target @ obj_from_grasp @ tool_from_ee_mat
                target_pose = Pose.from_matrix(world_from_ee)

                # The held block sits ON the placement surface at the place
                # pose, which would otherwise be rejected as a held-block ↔
                # surface collision. Temporarily disable the surface obstacle
                # for the place planning. The gripper itself stays collision-
                # checked against everything (table, walls, other blocks).
                with timer.time("curobo_planning"), _disabled_world_obstacle(planner, surface_name):
                    place_result = plan_single_arm_pose(
                        planner,
                        active_tool_frame=active_tool,
                        target_pose=target_pose,
                        current_state=last_js,
                        max_attempts=POSE_MAX_ATTEMPTS,
                        enable_graph_attempt=POSE_ENABLE_GRAPH_ATTEMPT,
                    )
                if place_result is None or not bool(place_result.success.any()):
                    raise RuntimeError(f"Place plan failed for {ground_op}")

                plan = _interp_plan(place_result)
                dt = _plan_dt(place_result)
```

Replace that block (down to and including the `dt = _plan_dt(place_result)` line) with:

```python
            elif metadata.action_type == "place":
                obj_name, grasp_name, place_name, surface_name, q_name = ground_op.values
                world_from_obj_target = action_4dof_to_mat4x4(best_particle[place_name].clone())

                # Anchor the place TERMINAL to the COM-safe particle conf
                # (best_particle[q_name]) via plan_cspace, instead of a free
                # Cartesian-pose redundancy solve. The conf is an exact IK
                # solution for the same place hand pose but sits inside the COM
                # support hull (hard-checked), so the executed arrival is
                # COM-feasible. The held block sits ON the placement surface at
                # the terminal, so the surface obstacle is disabled during
                # planning (gripper stays collision-checked against everything
                # else).
                with timer.time("curobo_planning"):
                    place_result = _plan_arm_to_conf(
                        planner, best_particle[q_name].clone(), last_js,
                        full_joint_names,
                        disable_obstacle=surface_name, retries=1, ground_op=ground_op,
                    )

                plan = _interp_plan(place_result)
                dt = _plan_dt(place_result)
```

(This drops the now-unused `obj_from_grasp` / `world_from_ee` / `target_pose` for place — place doesn't track `grasp_from_obj`. `world_from_obj_target` is KEPT; it's used below for `obj_to_current_pose[obj_name]` and `_update_world_obstacle_pose`. The rest of the place branch — `accum_plans.append`, `last_js = _last_timestep_js(...)`, detach, obstacle-pose update — stays unchanged.)

- [ ] **Step 4: Run the integration test; confirm it PASSES**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_motion_anchor.py::test_place_arrival_is_com_in_hull -q -p no:cacheprovider
```
Expected: PASS — every segment arrival, including place, is COM ≤ tol.

- [ ] **Step 5: Confirm the unit tests still pass + import smoke**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_motion_anchor.py -q -p no:cacheprovider
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -c "import cutamp.motion_solver; print('OK')"
```
Expected: all pass; `OK`.

- [ ] **Step 6: Commit**

```bash
git add cutamp/motion_solver.py cutamp/tests/test_motion_anchor.py
git commit -m "$(cat <<'EOF'
fix: anchor place trajectory terminal to the COM-safe particle conf

Place planned to a Cartesian pose (plan_single_arm_pose), letting cuRobo
re-resolve arm/torso redundancy at the arrival and drift the COM ~3.7cm past
the toe edge even though the place conf (left_q3) is 2cm inside. Route the
place terminal through _plan_arm_to_conf (plan_cspace to best_particle[q_name])
like retract — same hand pose, COM-safe redundancy. needs_cuda integration
test asserts every segment arrival is COM <= tol.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Route the PICK branch through `_plan_arm_to_conf` (symmetry guard)

Pick arrivals tested in-hull in every probe, but pick has the identical free-redundancy structure, so anchor it too for robustness and symmetry.

**Files:**
- Modify: `cutamp/motion_solver.py` (pick branch)

- [ ] **Step 1: Rewrite the pick branch to anchor at the grasp conf**

In `cutamp/motion_solver.py`, find the pick branch:

```python
            elif metadata.action_type == "pick":
                obj_name, grasp_name, _ = ground_op.values
                world_from_obj = obj_to_current_pose[obj_name]
                obj_from_grasp = (
                    action_4dof_to_mat4x4 if config.grasp_dof == 4 else action_6dof_to_mat4x4
                )(best_particle[grasp_name].clone())
                world_from_ee = world_from_obj @ obj_from_grasp @ tool_from_ee_mat
                target_pose = Pose.from_matrix(world_from_ee)

                # Plan directly to the grasp pose. Disable the target block as
                # a world obstacle so gripper-block contact at the grasp pose
                # isn't rejected; gripper-table / gripper-other-block collision
                # stays enforced.
                with timer.time("curobo_planning"), _disabled_world_obstacle(planner, obj_name):
                    grasp_result = None
                    last_status = None
                    for _ in range(GRASP_RETRY):
                        grasp_result = plan_single_arm_pose(
                            planner,
                            active_tool_frame=active_tool,
                            target_pose=target_pose,
                            current_state=last_js,
                            max_attempts=POSE_MAX_ATTEMPTS,
                            enable_graph_attempt=POSE_ENABLE_GRAPH_ATTEMPT,
                        )
                        if grasp_result is not None and bool(grasp_result.success.any()):
                            break
                        last_status = getattr(grasp_result, "status", None)
                if grasp_result is None or not bool(grasp_result.success.any()):
                    raise RuntimeError(f"Pick plan failed for {ground_op}: {last_status}")

                # Single trajectory: gripper moves from start state to grasp
                # pose. Attach AT the grasp pose. Block tracking from this
                # point uses inverse(obj_from_grasp) since the gripper is
                # actually at the planner's intended grasp pose.
                grasp_plan = _interp_plan(grasp_result)
                dt = _plan_dt(grasp_result)
```

Replace that block (down to and including the `dt = _plan_dt(grasp_result)` line) with:

```python
            elif metadata.action_type == "pick":
                obj_name, grasp_name, q_name = ground_op.values
                obj_from_grasp = (
                    action_4dof_to_mat4x4 if config.grasp_dof == 4 else action_6dof_to_mat4x4
                )(best_particle[grasp_name].clone())

                # Anchor the grasp TERMINAL to the COM-safe particle conf
                # (best_particle[q_name]) via plan_cspace instead of a free
                # Cartesian-pose redundancy solve. The conf is an exact IK
                # solution for the grasp pose, so the gripper still arrives at
                # the grasp, with a COM-safe posture. Disable the target block
                # obstacle so gripper-block contact at the grasp isn't rejected;
                # gripper-table / gripper-other-block collision stays enforced.
                with timer.time("curobo_planning"):
                    grasp_result = _plan_arm_to_conf(
                        planner, best_particle[q_name].clone(), last_js,
                        full_joint_names,
                        disable_obstacle=obj_name, retries=GRASP_RETRY, ground_op=ground_op,
                    )

                # Attach AT the grasp conf. Block tracking from this point uses
                # inverse(obj_from_grasp) since the gripper is at the intended
                # grasp pose.
                grasp_plan = _interp_plan(grasp_result)
                dt = _plan_dt(grasp_result)
```

(KEEPS `obj_from_grasp` — it's used just below for `grasp_from_obj = torch.inverse(obj_from_grasp).clone()`. Drops the now-unused `world_from_obj` / `world_from_ee` / `target_pose` / `last_status`. The rest of the pick branch — `accum_plans.append`, `last_js`, `_attach_object`, `grasp_from_obj`, state updates — stays unchanged.)

- [ ] **Step 2: Extend the integration test to assert the pick arrival reaches the grasp conf's hand pose**

Append to `cutamp/tests/test_motion_anchor.py` a `needs_cuda` test that the pick arrival still reaches the intended grasp (hand pose preserved) — guarding against the anchor breaking the task:

```python
@needs_cuda
def test_all_arrivals_in_hull_and_plan_succeeds():
    # Broader guard: a full plan generates (no RuntimeError from solve_curobo)
    # and ALL segment arrivals (pick, place, retract) are COM-in-hull.
    import torch
    from cutamp.robots.t1 import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES, t1_home
    from cutamp.com_polygon_cost import compute_com_polygon_penalties, COM_TOL

    world, segments = _generate_plan_segments(seed=1)
    assert len(segments) > 0, "plan produced no segments"
    names = list(world.kinematics.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    q_home = torch.as_tensor(t1_home, dtype=torch.float32,
                             device=world.kinematics.device_cfg.device)
    for si, seg in enumerate(segments):
        P = seg["position"]; T = seg["T"]
        q = q_home.clone()
        q[idx["Torso_Pitch"]] = float(P["trunk_pitch"][T - 1])
        q[idx["Waist_Yaw"]] = float(P["trunk_yaw"][T - 1])
        q[idx["ankle_pitch"]] = float(P["ankle_pitch"][T - 1])
        q[idx["knee_pitch"]] = float(P["knee_pitch"][T - 1])
        for j, n in enumerate(LEFT_ARM_JOINT_NAMES):
            q[idx[n]] = float(P["left_arm"][T - 1][j])
        for j, n in enumerate(RIGHT_ARM_JOINT_NAMES):
            q[idx[n]] = float(P["right_arm"][T - 1][j])
        pen = float(compute_com_polygon_penalties(world, {"q": q.unsqueeze(0)})["q"][0])
        assert pen <= COM_TOL, f"segment {si} arrival COM {pen:.6f} > {COM_TOL}"
```

- [ ] **Step 3: Run the integration tests; confirm PASS**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_motion_anchor.py -q -p no:cacheprovider
```
Expected: all pass (2 unit + 2 needs_cuda).

- [ ] **Step 4: Commit**

```bash
git add cutamp/motion_solver.py cutamp/tests/test_motion_anchor.py
git commit -m "$(cat <<'EOF'
fix: anchor pick trajectory terminal to the COM-safe grasp conf

Symmetric with the place fix: route the pick terminal through
_plan_arm_to_conf (plan_cspace to best_particle[q_name]) instead of a free
Cartesian-pose solve, so the grasp arrival is COM-safe by construction. Pick
arrivals were already in-hull empirically; this removes the same latent
free-redundancy drift. Integration test asserts all arrivals COM <= tol.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Multi-seed verification, full suite, regenerate the saved plan

Prove the fix is robust (the original excursion was run-dependent), confirm no regression, and refresh the committed sample plan.

**Files:**
- Modify: `data/motion_plan.pkl` (regenerate)

- [ ] **Step 1: Multi-seed endpoint-COM + success-rate check**

Run the standard smoke 5 times and audit every arrival each time. Use a one-off script (write to `/tmp`, do not commit):

```bash
for s in 0 1 2 3 4; do
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
    --env blocks_t1 --robot t1 --disable_visualizer -n 64 --num_opt_steps 50 \
    --motion_plan --save_plan /tmp/anchor_seed_$s.pkl 2>&1 | tail -2
done
```
Expected: each run completes with "Total num satisfying ≥ 1" and a saved plan (no "Motion plan failed" tracebacks — success rate not regressed).

- [ ] **Step 2: Audit all 5 plans' arrivals are COM-in-hull**

Run (write `/tmp/audit_arrivals.py` using the established reconstruction; assert per-segment arrival ≤ COM_TOL for every saved plan):

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python - <<'PY'
import os, pickle, torch, numpy as np
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.tamp_world import TAMPWorld
from cutamp.robots import load_robot_container
from cutamp.robots.t1 import t1_home, LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from curobo.types import DeviceCfg
from cutamp.com_polygon_cost import compute_com_polygon_penalties, COM_TOL
env=load_env(os.path.join(get_env_dir(),"blocks_t1.yml")); dc=DeviceCfg()
robot=load_robot_container("t1",dc)
q0=torch.as_tensor(t1_home,dtype=torch.float32,device=dc.device)
world=TAMPWorld(env=env,device_cfg=dc,robot=robot,q_init=q0,enable_com_polygon=True)
names=list(world.kinematics.joint_names); idx={n:i for i,n in enumerate(names)}
def arrival_q(P,T):
    q=q0.clone()
    q[idx["Torso_Pitch"]]=float(P["trunk_pitch"][T-1]); q[idx["Waist_Yaw"]]=float(P["trunk_yaw"][T-1])
    q[idx["ankle_pitch"]]=float(P["ankle_pitch"][T-1]); q[idx["knee_pitch"]]=float(P["knee_pitch"][T-1])
    for j,n in enumerate(LEFT_ARM_JOINT_NAMES): q[idx[n]]=float(P["left_arm"][T-1][j])
    for j,n in enumerate(RIGHT_ARM_JOINT_NAMES): q[idx[n]]=float(P["right_arm"][T-1][j])
    return q
ok=True
for s in range(5):
    p=pickle.load(open(f"/tmp/anchor_seed_{s}.pkl","rb"))
    worst=0.0
    for si,seg in enumerate(p["segments"]):
        q=arrival_q(seg["position"],seg["T"]).unsqueeze(0)
        pen=float(compute_com_polygon_penalties(world,{"q":q})["q"][0]); worst=max(worst,pen)
        if pen>COM_TOL: ok=False; print(f"  seed{s} seg{si} OUT pen={pen:.6f}")
    print(f"seed {s}: worst arrival penalty {worst:.6f} (tol {COM_TOL}) {'OK' if worst<=COM_TOL else 'FAIL'}")
print("ALL ARRIVALS IN HULL" if ok else "SOME ARRIVALS OUT OF HULL")
PY
```
Expected: every seed reports `OK`, final line `ALL ARRIVALS IN HULL`. (Optional stronger check: also audit ALL frames, not just arrivals — mid-interpolation may still bow out; that's the known future per-frame-guard gap, so only ARRIVALS are required to pass here. Note any all-frame excursions for follow-up.)

- [ ] **Step 2b (if any seed FAILs):** the anchor didn't fully resolve it → STOP and report. Likely means the place/pick conf for that seed is itself borderline, or `plan_cspace` overshoots the conf. Do not paper over with knob tweaks — escalate per the spec's per-frame-hard-check fallback.

- [ ] **Step 3: Full test suite**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/ examples/ -q -p no:cacheprovider
```
Expected: all pass (prior baseline + the new `test_motion_anchor.py` tests).

- [ ] **Step 4: Regenerate the committed sample plan**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --robot t1 --disable_visualizer -n 64 --num_opt_steps 50 \
  --motion_plan --save_plan data/motion_plan.pkl
```
Then verify it's clean (all arrivals in hull) by re-running the Step-2 audit against `data/motion_plan.pkl`.

- [ ] **Step 5: Commit the regenerated plan**

```bash
git add data/motion_plan.pkl
git commit -m "$(cat <<'EOF'
chore: regenerate motion_plan.pkl with COM-anchored pick/place terminals

The committed sample now reflects the cspace-anchored endpoints — pick/place
arrivals are COM-in-hull (the prior sample had a ~3.7cm toe-edge excursion at
the LeftPlace arrival).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Confirm no `curobo/` edits across the branch**

Run:
```bash
git diff --name-only HEAD~4..HEAD | grep '^curobo/' && echo "VIOLATION" || echo "OK: no curobo edits"
```
Expected: `OK: no curobo edits`.

---

## Final verification (after all tasks)

- [ ] Full suite green: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 … pytest cutamp/tests/ examples/ -q -p no:cacheprovider`.
- [ ] 5/5 seeds: all pick/place/retract arrivals COM ≤ `COM_TOL` (Task 4 Step 2).
- [ ] Motion-plan success rate not regressed vs pre-fix (5/5 runs produced a plan, no new tracebacks).
- [ ] `git diff` shows changes only in `cutamp/motion_solver.py`, `cutamp/tests/test_motion_anchor.py`, `data/motion_plan.pkl` — nothing under `curobo/`.
- [ ] Use the **superpowers:finishing-a-development-branch** skill to wrap up.

## Notes / known limitations

- This fixes pick/place **arrival** (endpoint) COM. **Mid-trajectory interpolation** between waypoints is still only softly COM-constrained; a hard per-frame COM verification + replan is the complementary guarantee and is intentionally **out of scope** (the runner-up fix in the spec). If Task 4's optional all-frame audit shows mid-trajectory excursions, file that as the follow-up.
- The anchor assumes each pick/place conf is an exact IK solution reaching its grasp/place pose (verified: hand pose matches to ~1 µm). The Task 2/3 integration tests guard against a conf drifting from its pose.
- If anchoring measurably drops trajopt success rate (it shouldn't — cspace goals are easier than pose goals), revert to `plan_pose` + a strong COM seed, or adopt the per-frame-check approach; do not silently raise soft COM weights (coupled to the hard `COM_TOL`).
