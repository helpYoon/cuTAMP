# cuTAMP with Booster T1

## Quick Links

The full README with installation instructions, examples, and detailed
documentation can be found in [README_DETAILED.md](README_DETAILED.md). Notes
on the cuRobo v0.8 port that defines the current architecture are in
[docs/curobo_v08_port.md](docs/curobo_v08_port.md).

> **cuRobo fork dependency:** cuTAMP relies on local edits to the vendored
> `curobo/` tree (the `compute_com` plumbing and the CoM-aware seed-IK
> residual — 7 Python files; the earlier batched-COM kernel fix was dropped
> when the base moved to NVlabs PR #678, which fixes it upstream). The edits
> live on the `main` branch of the cuRobo fork (helpYoon/curobo) — a plain
> NVlabs checkout without them breaks cuTAMP. Grep `cuTAMP fork` under
> `curobo/curobo/_src/` for the full inventory.

## T1 Single-Planner Architecture (cuRobo v0.8)

cuTAMP plans for the T1 humanoid through a **single 21-DOF
MotionPlanner** built from `t1_planar_base.yml`. The cspace is:

```
3 base (planar: base_j_x, base_j_y, base_j_yaw)   ← locked at IK time, free during navigate
2 leg  (ankle_pitch, knee_pitch)
2 torso (Torso_Pitch, Waist_Yaw)
7 left arm  (Left_Shoulder_Pitch ... Left_Hand_Roll)
7 right arm (Right_Shoulder_Pitch ... Right_Hand_Roll)
= 21 DOF
```

The mobile base is added as a virtual chain via `extra_links`
(`world → base_j_x → base_j_y → base_j_yaw → mobile_base_link`) and locked at
IK / motion-planner construction time, leaving an 18-DOF active cspace for arm
operators. Two tool frames (`left_base_link`, `right_base_link`) are exposed
on the same kinematics; per-arm IK uses `GoalToolPose` to target one frame at
a time.

```mermaid
graph TD
    subgraph Planner [Single 21-DOF MotionPlanner]
        Base["Planar base (3, locked at IK)"]
        Body["Leg + Torso (4, always free)"]
        L["Left arm (7)"]
        R["Right arm (7)"]
    end

    subgraph Pinning [Cspace pinning per operator]
        ArmOp["pin_for_arm_action(active_arm)<br/>→ inactive arm pinned to start;<br/>active arm + body free"]
        Nav["pin_for_movebase()<br/>→ both arms + body pinned;<br/>only base DOFs free"]
    end

    subgraph Tools [Tool frames]
        LT["left_base_link"]
        RT["right_base_link"]
    end

    Planner --> ArmOp
    Planner --> Nav
    Planner --> LT
    Planner --> RT
```

There is no per-arm "shared joint" propagation — the body joints are
genuinely shared in one cspace, so trajopt updates them coherently. The
inactive arm's tool moves through world during single-arm planning because
the body translates; the visualizer tracks both arms' held objects through
every trajectory (see `T1State.compute_all_held_obj_poses`).

## Layout

### 📁 [`cutamp/robots/`](cutamp/robots/)

- **[`assets/t1_description/`](cutamp/robots/assets/t1_description/)**:
  - `t1_simplified.urdf` — URDF the planner loads (adds the `world` link the
    planar-base chain attaches to). `actual_robot.urdf` is the unmodified
    upstream URDF kept for reference.
  - `t1_planar_base.yml` — single cuRobo robot config: kinematics, cspace,
    `lock_joints` (head + grippers), `collision_link_names`, planar-base
    `extra_links`, and the two tool frames.
  - `t1_spheres.yml` — collision spheres in each link's local frame; finger
    spheres are consolidated into `left_base_link` / `right_base_link` since
    the gripper joints are locked open.
  - `left_gripper_spheres.pt`, `right_gripper_spheres.pt` — pre-fit gripper
    spheres used by grasp sampling.
  - `meshes/` — STL meshes referenced by the URDF.

- [`__init__.py`](cutamp/robots/__init__.py) — `RobotContainer` factory and
  `load_robot_container("t1")`. T1 exposes two tool frames; tool↔EE transforms
  account for T1's +X-toward-fingertips EE convention.

- [`t1.py`](cutamp/robots/t1.py) — T1 module:
  - cuRobo loaders: `get_t1_kinematics`, `get_t1_ik_solver(scene)` (a single
    multi-tool-frame IK over both arms; with `enable_com_aware_ik=True`, the
    default, its seed-IK LM stage carries a CoM-over-support-rectangle
    residual — a cuRobo fork edit — weighted by `T1_COM_IK_WEIGHT` and the
    pull-to-center `T1_COM_IK_CENTER_WEIGHT`), and `get_t1_motion_planner`
    (the single planner with the base lock and the COM-over-polygon rollout
    soft cost applied).
  - Joint-name + index constants: `JOINT_NAMES_FULL`, `BASE_INDICES`,
    `BODY_INDICES`, `LEFT_ARM_JOINT_NAMES`, `RIGHT_ARM_JOINT_NAMES`,
    `t1_home`.
  - `T1RerunRobot` for visualization: maps the planner's cspace (18 active or
    21 full) into the URDF's joint count, splices in head + per-arm gripper
    state.

### 📁 [`cutamp/`](cutamp/) — core

- [`tamp_world.py`](cutamp/tamp_world.py) — `TAMPWorld` wraps a TAMPEnvironment
  with the collision Scene, the single `Kinematics`, a single multi-tool-frame
  `InverseKinematics` solver, and a `MotionPlanner` factory. `q_init` is one
  21-DOF tensor. A lazy `kinematics_with_com` (built with `compute_com=True`)
  serves the COM checks and audits.

- [`t1_state.py`](cutamp/t1_state.py) — `T1State` carries the planner,
  kinematics, current joint state, and per-arm `arm_holding` /
  `arm_grasp_transform`. Cspace pinning lives here:
  - `pin_for_arm_action(arm)` — pin inactive arm; body + active arm free.
  - `pin_for_movebase()` — pin both arms + body; only base free.
  - `unpin()` — restore original weights.
  - `compute_held_obj_poses(arm, plan)` — per-timestep `world_from_obj` for a
    held object along a planner trajectory.
  - `compute_all_held_obj_poses(plan)` — `{arm: (obj_name, poses)}` for every
    arm currently holding (visualization needs both because the inactive
    arm's tool still translates with the body).

- [`_curobo_internals.py`](cutamp/_curobo_internals.py) — every cuRobo
  private-API workaround in one file (with `# TODO: file upstream issue`
  markers): `get_attachment_manager`, `cspace_plan_succeeded`,
  `iter_rollouts`, the cspace target-weight `write…` / `snapshot…` /
  `restore…` trio (hosts-iterable), `inactive_arm_cspace_weights`, and
  `add_extra_cost` (custom-cost dispatch; effective for the planner's
  rollouts — for IK solve-time costs it is inert, which is why the CoM
  residual lives inside the seed-IK fork instead).

- [`com_polygon_cost.py`](cutamp/com_polygon_cost.py) — single source of
  truth for the COM-over-support-rectangle geometry (`COM_HALF_EXTENTS`,
  `COM_INSIDE_MARGIN` / `COM_INSIDE_WEIGHT`, `COM_TOL`) and its consumers:
  `com_polygon_penalty` (outside-quadratic + inside-margin barrier +
  optional pull-to-center), `ComOverBasePolygonCost` (the planner rollout
  cost), `compute_com_polygon_mask` / `compute_com_polygon_penalties` (the
  hard `ComPolygon` gate + post-IK COM-safe retry), and
  `compute_com_in_base` (audits / tests).

- [`motion_solver.py`](cutamp/motion_solver.py) — drives the plan skeleton
  through the single planner. Per-operator: pin the cspace, plan, append a
  trajectory entry to `accum_plans` (with `held_objs` for both arms), then
  attach/detach via the AttachmentManager. Playback walks `accum_plans` and
  hands each entry to the visualizer.

- [`grasp_planning.py`](cutamp/grasp_planning.py) — `_build_multi_frame_goal`
  builds the `GoalToolPose` covering every tool frame for single-arm actions:
  the active frame gets the target pose, while inactive frames keep their
  ENABLED pose criteria targeting their FK poses at the segment's start
  state. Pick/place terminal confs are anchored in cspace.

- [`particle_initialization.py`](cutamp/particle_initialization.py) — IK-seeded
  particle init for `LeftPick` / `RightPick` / `LeftPlace` / etc. Cache
  hits route through `_apply_cached`.

- [`t1_domain.py`](cutamp/t1_domain.py) — TAMP domain (fluents + operators)
  for T1: `LeftAt` / `RightAt`, `LeftHolding` / `RightHolding`,
  `LeftMoveFree` / `RightMoveFree`, `LeftPick` / `RightPick`, `LeftPlace` /
  `RightPlace`, `LeftPush` / `RightPush`, `LeftPushStick` / `RightPushStick`,
  and the `RetractHolding` / `RetractFree` family.

- [`config.py`](cutamp/config.py) — `TAMPConfiguration` (only T1 supported)
  and `SUPPORTED_SOFT_COSTS`. Notable defaults (all validated together):
  `optimize_soft_costs=True` + `coupled_reik=True` +
  `soft_cost=["place_close_to_base"]` (the trio), `enable_com_polygon=True`
  (planner COM cost + hard gate), `enable_com_aware_ik=True` (seed-IK CoM
  residual), `ik_com_retry_max` (post-IK COM-mask backstop),
  `debug_motion_failures`.

- [`algorithm.py`](cutamp/algorithm.py) — `setup_cutamp` + `run_cutamp`
  outer loop. The motion-plan retry sweep lives in `_motion_plan_with_retries`.

- [`optimize_plan.py`](cutamp/optimize_plan.py) — Adam-based optimization over
  particles; respects `left_q0` / `right_q0` as fixed initial-state.

- [`cost_function.py`](cutamp/cost_function.py) — hard constraints
  (kinematic, collision, stable placement, and the `ComPolygon` COM-in-hull
  gate over every `q_*` conf) plus the soft costs listed in
  `SUPPORTED_SOFT_COSTS` (`place_close_to_base`, `retract_close_to_home`,
  `minimize_body_movement`, `com_polygon`, the multi-object placement
  costs, …).

- [`rollout.py`](cutamp/rollout.py) — FK rollout helpers used by the cost
  graph (forward kinematics on the full 21-DOF cspace).

- [`conf_locking.py`](cutamp/conf_locking.py) — utility for freezing a
  configuration during optimization (e.g., already-satisfied IKs).

- [`utils/motion_diagnostics.py`](cutamp/utils/motion_diagnostics.py) —
  cspace-failure diagnostic helpers (gated behind
  `config.debug_motion_failures`; not on the hot path by default).

- [`utils/visualizer.py`](cutamp/utils/visualizer.py) — Rerun visualizer.
  `log_joint_trajectory_with_mat4x4s(traj, mat4x4s, …)` accepts any number
  of held-object transforms per trajectory.

### 📁 [`cutamp/tests/`](cutamp/tests/) — 63 tests

Shared fixtures live in [`conftest.py`](cutamp/tests/conftest.py)
(`needs_cuda` marker, `make_blocks_t1_world` factory, GPU-memory cleanup).
Highlights:

- [`test_t1_config.py`](cutamp/tests/test_t1_config.py) /
  [`test_t1_robot_module.py`](cutamp/tests/test_t1_robot_module.py) — YAML
  schema, cspace, lock-joints, tool frames, kinematics smoke.
- [`test_com_polygon_ik.py`](cutamp/tests/test_com_polygon_ik.py) — COM
  penalty math, mask/gate parity, batched-COM kernel regression.
- [`test_com_aware_seed_ik.py`](cutamp/tests/test_com_aware_seed_ik.py) —
  CoM-aware seed-IK: fork/cuTAMP penalty parity, FD-checked gradients,
  centering A/B with joint-limit and pose-accuracy guards.
- [`test_motion_anchor.py`](cutamp/tests/test_motion_anchor.py) — full
  pipeline: cspace-anchored pick/place terminals, all arrivals COM-in-hull.
- Plus pin lifecycle, arm-affinity priority, plan-processor derivatives,
  and a coupled-reIK smoke test.

Run with:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTORCH_ALLOC_CONF=expandable_segments:True pytest cutamp/tests/ -q`
(GPU required for the integration tests; ~80 s on an RTX 4090).

### 📁 [`cutamp/scripts/`](cutamp/scripts/)

- [`run_cutamp.py`](cutamp/scripts/run_cutamp.py) — main entry point.
  T1 / `blocks_t1` only (no `--robot` flag, no other envs in this branch).
- [`gripper_sphere_editor.py`](cutamp/scripts/gripper_sphere_editor.py) —
  interactive gripper-sphere editor.
- [`robot_sphere_editor.py`](cutamp/scripts/robot_sphere_editor.py) —
  interactive robot-sphere editor.

## Usage

By default a run uses the validated trio — soft-cost optimization ON with
`place_close_to_base` under re-IK-coupled Adam — plus CoM-aware IK, so
placements stay near the base and every arrival lands with the COM centered
in the support polygon.

```bash
# Smoke test (no motion plan)
python -m cutamp.scripts.run_cutamp --env blocks_t1 -n 16 --num_opt_steps 50

# Full run with motion planning, no visualizer
PYTORCH_ALLOC_CONF=expandable_segments:True python -m cutamp.scripts.run_cutamp \
    --env blocks_t1 --motion_plan -n 64 --disable_visualizer

# With Rerun visualizer
PYTORCH_ALLOC_CONF=expandable_segments:True python -m cutamp.scripts.run_cutamp \
    --env blocks_t1 --motion_plan -n 64

# Save the processed plan (schema v3) for the MPC consumer
PYTORCH_ALLOC_CONF=expandable_segments:True python -m cutamp.scripts.run_cutamp \
    --env blocks_t1 --motion_plan -n 64 --save_plan data/motion_plan.pkl

# Different soft costs (optimization is already on by default)
python -m cutamp.scripts.run_cutamp --env blocks_t1 --motion_plan \
    --soft_cost retract_close_to_home minimize_body_movement

# Classic single-level optimization, no soft costs
python -m cutamp.scripts.run_cutamp --env blocks_t1 --motion_plan \
    --no_optimize_soft_costs --no_coupled_reik
```

**Available soft costs** (see `SUPPORTED_SOFT_COSTS`; T1-only ones marked):
- `place_close_to_base` *(default)* — pull placed objects toward the base;
  keeps placements inside the reach range where the CoM-aware IK can center
  the COM.
- `retract_close_to_home` *(T1)* — penalize retract configs far from `t1_home`.
- `minimize_body_movement` *(T1)* — penalize body (leg/torso/waist) drift.
- `com_polygon` *(T1)* — differentiable COM-in-polygon penalty on confs.
- `dist_from_origin`, `max_obj_dist` / `min_obj_dist`, `min_y` / `max_y`,
  `align_yaw` — multi-object placement costs.

Pose-class soft costs (e.g. `place_close_to_base`) require the re-IK-coupled
optimizer (`coupled_reik`, on by default): outer Adam moves pose/grasp
particles while their dependent confs are refreshed by exact IK every
`reik_interval` steps.

VRAM at `-n 64`: roughly 4–5 GiB above baseline (peaks at the end of motion
planning). Comfortably fits a 24 GB GPU.

## Key Design Decisions

1. **Single 21-DOF MotionPlanner** — cuRobo v0.8's `GoalToolPose` lets us
   plan for one tool frame at a time on a kinematics that exposes both arms,
   so we no longer need two MotionGens or per-arm "shared joint" propagation.

2. **Planar base via `extra_links`** — `world → base_j_x → base_j_y →
   base_j_yaw → mobile_base_link` adds 3 virtual DOFs that get hard-locked at
   IK / motion-planner construction time for arm operators (so the base
   doesn't drift while reaching). `MoveBaseTo` is currently unfireable —
   wiring it up needs a second, base-unlocked planner.

3. **Cspace pinning replaces per-arm planners** — `pin_for_arm_action` /
   `pin_for_movebase` write into each rollout's
   `cspace_target_dof_weight` to keep the inactive joints at the start
   state. Restored by `unpin()`.

4. **Both arms tracked through every trajectory** — body joints are unlocked
   during single-arm planning, so an inactive arm holding a block will still
   translate that block through world. `compute_all_held_obj_poses` returns
   poses for every holding arm so the visualizer renders both correctly.

5. **Retract after pick/place** — `RetractHolding` (collision checks exclude
   held object) and `RetractFree` (empty gripper) move toward `t1_home` after
   each manipulation. Hard collision constraints + the
   `retract_close_to_home` soft cost shape the result.

6. **cuRobo internals isolated** — every reach into cuRobo's private API
   lives in [`cutamp/_curobo_internals.py`](cutamp/_curobo_internals.py) so a
   future cuRobo upgrade has one place to audit. The exceptions are the
   fork edits inside the vendored `curobo/` tree itself (grep `cuTAMP fork`);
   those must travel with any cuRobo upgrade.

7. **Layered COM safety** — the COM is kept inside the two-foot support
   polygon by independent layers, each guarding a different frontier:
   - **CoM-aware seed-IK** (cuRobo fork residual, on by default): the IK
     itself trades hand pose vs COM, recruiting the knees/ankles within
     joint limits, with a pull-to-center term — endpoint confs come out
     centered, not edge-parked.
   - **Hard `ComPolygon` gate**: every optimizer conf must have COM penalty
     ≤ `COM_TOL`; a post-IK mask + retry backstops the IK.
   - **COM-anchored terminals**: pick/place trajectories end exactly at the
     gate-approved conf (`plan_cspace`), so cuRobo can't re-resolve the
     redundancy at the endpoint.
   - **Planner rollout COM cost**: a soft barrier keeps mid-trajectory
     frames in-hull between anchored endpoints.
   - **`place_close_to_base`** (default soft cost): biases chosen placements
     away from the far-reach regime where no in-limit posture can center
     the COM.
