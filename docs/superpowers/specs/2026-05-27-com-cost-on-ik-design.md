# COM cost on IK solver — design

**Date**: 2026-05-27
**Scope**: Add two-layer COM-feasibility defense for particle-initialization IK so the IK-init pose doesn't produce teetering, out-of-base-polygon configurations.

## Context

The COM-over-base-polygon cost is currently registered ONLY on the motion planner (`cutamp/robots/t1.py:get_t1_motion_planner` lines 213-240, via `add_extra_cost(planner, "com_polygon", ...)` with `compute_com=True` plumbed through `trajopt_transition_model`).

The IK solver (`get_t1_ik_solver`) is built with cuRobo defaults: no `compute_com=True`, no `add_extra_cost` registration. Result: particle-init IK happily finds extreme leaning configurations whose COM lies way outside the wheelbase polygon. The Adam-side `com_polygon` soft cost (in `cost_function.py`) runs AFTER IK init, but it operates on Adam particles under the `KinematicConstraint` — pulling the body back to feasibility would violate the IK pose target, so Adam often can't recover.

Observed failure: with a Pick(block_on_ground) target, IK returns a configuration with the trunk bent deeply forward, one leg extended way back, COM clearly over the front edge of the wheelbase. The Adam loop accepts this as the IK-init state and the motion planner inherits it.

The bug is **structural**: IK has no signal about COM. The fix has to give IK access to the cost.

## Decisions (locked in)

- **Two-layer defense**: Option A (register cost on IK) as primary fix; Option C (post-IK batched retry) as safety net for the LBFGS-fails-to-recover edge case.
- **Cost config on IK**: same as motion planner — `weight=[5.0e5]`, `half_extents=[0.05, 0.10]`, `inside_margin=0`, `inside_weight=0`. Outside-quadratic only, boundary-only safety barrier. (IK doesn't have the `body_home_posture` cost that previously fought the inside-margin barrier — but we still start simple and match the planner-side shape. Inside-margin can be added later if outside-only is insufficient.)
- **Single config flag**: existing `enable_com_polygon: bool = True` gates BOTH the planner-side and the IK-side cost. One toggle for the whole COM machinery.
- **Layer 2 retry budget**: `ik_com_retry_max: int = 3` (new `TAMPConfiguration` field).
- **Cost evaluator**: cost fires during the LBFGS refinement stage of IK (not during the LM seeder, which doesn't use `cost_manager`). Acceptable — LBFGS has plenty of iterations to recover.

## Architecture

```
particle init for Pick/Place
  ↓
sample grasp/pose (with random offsets)
  ↓
_ik_for_pose_com_safe(world, target_pose, arm)        ← NEW wrapper (Layer 2)
  │
  ├─ _ik_for_pose(world, target, arm)                  ← unchanged; now Layer-1-equipped
  │     ↓
  │     world.ik_solver.solve_pose(...)                ← LBFGS uses cost_manager
  │     cost_manager evaluates "com_polygon" each iter ← Layer 1 (Option A)
  │     ↓
  │     [B, 18] joints + [B] success
  │
  ├─ batch-compute COM mask via world.kinematics_with_com
  │
  └─ if any out of polygon: retry only failed particles up to ik_com_retry_max times
        ↓ (each retry uses cuRobo's seed randomization)
     return final batch (best effort)
  ↓
_store_ik_q stores result in particles
```

## Layer 1 — COM cost on IK (Option A)

### Files
- `cutamp/robots/t1.py` — new `_ik_transition_dict_with_compute_com()` helper; modify `get_t1_ik_solver`.
- `cutamp/tamp_world.py` — forward `enable_com_polygon` to `get_t1_ik_solver`.

### New helper `_ik_transition_dict_with_compute_com()`

Exact mirror of the existing `_trajopt_transition_dict_with_compute_com()` (currently at `t1.py:159-172`):

```python
def _ik_transition_dict_with_compute_com() -> dict:
    """Default IK transition YAML with compute_com=True injected.

    cuRobo's IK kernel template is fixed at IK-build time; we inject
    compute_com=True so state.cuda_robot_model_state.robot_com is
    populated during LBFGS refinement. Required for the COM-over-base
    soft cost registered on the IK rollouts.
    """
    from curobo._src.util.file_path import join_path, load_yaml, get_robot_configs_path
    yaml_path = join_path(get_robot_configs_path(), "ik/transition_ik.yml")
    d = load_yaml(yaml_path)
    d["transition_model_cfg"]["compute_com"] = True
    return d
```

Identical pattern, different yaml file path. If `ik/transition_ik.yml` doesn't exist or has a different structure than `trajopt/transition_trajopt.yml`, the implementer must inspect cuRobo's IK config layout and adapt.

### Modify `get_t1_ik_solver`

```python
def get_t1_ik_solver(
    scene: Scene,
    *,
    num_seeds: int = 32,
    self_collision_check: bool = True,
    max_batch_size: int = 64,
    enable_com_polygon: bool = True,            # NEW
    device_cfg: DeviceCfg = None,
) -> InverseKinematics:
    """...existing docstring...

    When ``enable_com_polygon`` is True (default), the IK solver gets
    the same COM-over-base-polygon soft cost as the motion planner.
    Without this, IK produces extreme leaning configurations whose
    COM projects outside the wheelbase rectangle — the Adam-side soft
    cost can't recover from these because pulling the body back would
    violate the KinematicConstraint.
    """
    if device_cfg is None:
        device_cfg = DeviceCfg()
    cfg_dict = _lock_mobile_base(copy.deepcopy(t1_curobo_cfg()))
    cfg = InverseKinematicsCfg.create(
        robot=cfg_dict,
        scene_model=scene,
        num_seeds=num_seeds,
        self_collision_check=self_collision_check,
        max_batch_size=max_batch_size,
        device_cfg=device_cfg,
        use_cuda_graph=False,
        transition_model=_ik_transition_dict_with_compute_com(),   # NEW
    )
    ik_solver = InverseKinematics(cfg)
    if enable_com_polygon:
        from cutamp._curobo_internals import add_extra_cost
        from cutamp.com_polygon_cost import (
            ComOverBasePolygonCost, ComOverBasePolygonCostCfg,
        )
        rollout_device_cfg = ik_solver.optimizer_rollouts[0].device_cfg
        cost_cfg = ComOverBasePolygonCostCfg(
            weight=[5.0e5],
            device_cfg=rollout_device_cfg,
            half_extents=[0.05, 0.10],
            inside_margin=0.0,
            inside_weight=0.0,
        )
        add_extra_cost(ik_solver, "com_polygon", ComOverBasePolygonCost(cost_cfg))
    return ik_solver
```

### TAMPWorld plumbing

```python
# cutamp/tamp_world.py, in __init__ (existing get_t1_ik_solver call):
self.ik_solver: InverseKinematics = get_t1_ik_solver(
    self.world_cfg,
    device_cfg=device_cfg,
    max_batch_size=512,
    enable_com_polygon=enable_com_polygon,   # NEW (new __init__ kwarg)
)
```

Add `enable_com_polygon: bool = True` to `TAMPWorld.__init__` signature and accept from caller. The wire-up from `TAMPConfiguration.enable_com_polygon` already exists for the motion planner side; extend to also pass into `TAMPWorld(__init__, ..., enable_com_polygon=config.enable_com_polygon)`.

## Layer 2 — Post-IK batched verification (Option C)

### Files
- `cutamp/com_polygon_cost.py` — new `compute_com_polygon_mask` helper.
- `cutamp/particle_initialization.py` — new `_ik_for_pose_com_safe` wrapper + `_splice_ik_result` helper; replace `_ik_for_pose` call sites.
- `cutamp/config.py` — `ik_com_retry_max: int = 3` field.

### `compute_com_polygon_mask` helper

```python
def compute_com_polygon_mask(
    world,
    q_batch: torch.Tensor,            # [B, full_dof] full-cspace joint positions
    *,
    half_extents: List[float] = [0.05, 0.10],
) -> torch.Tensor:                    # [B] bool, True = COM inside polygon
    """Batched COM-in-polygon check.

    Uses world.kinematics_with_com (built with compute_com=True) to FK +
    extract COM in one CUDA kernel call. Then projects COM into
    mobile_base_link frame via the same SE(3)-inverse math used by
    com_polygon_penalty, and applies per-axis bounds check.

    Returns True for batch elements where the projected COM falls inside
    the rectangle (per-axis abs <= half_extents).
    """
    from curobo.types import JointState
    kin = world.kinematics_with_com
    js = JointState.from_position(q_batch)
    ks = kin.compute_kinematics(js)
    com = ks.cuda_robot_model_state.robot_com[..., :3].view(-1, 3)  # [B, 3]
    base_T = (
        ks.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True)
        .get_matrix()
        .view(-1, 4, 4)                                              # [B, 4, 4]
    )
    R = base_T[:, :3, :3]
    t = base_T[:, :3, 3]
    com_in_base = (R.transpose(-1, -2) @ (com - t).unsqueeze(-1)).squeeze(-1)[:, :2]
    half = torch.tensor(
        half_extents, device=com.device, dtype=com.dtype,
    )
    return (torch.abs(com_in_base) <= half).all(dim=-1)
```

### `_ik_for_pose_com_safe` wrapper

```python
def _ik_for_pose_com_safe(
    world: TAMPWorld,
    world_from_ee: torch.Tensor,        # [B, 4, 4]
    arm: Optional[str],
    *,
    max_retries: int = 3,
):
    """Call _ik_for_pose; verify COM-in-polygon; retry failed particles.

    cuRobo's seed sampler has internal randomization, so re-calling IK
    with the same target may converge to a different (possibly feasible)
    solution. We retry only the failed particles to keep batch size small.

    Returns the last IK result (best effort). Logs final failure count.
    """
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    result = _ik_for_pose(world, world_from_ee, arm)
    for attempt in range(max_retries):
        q_batch = _ik_solution_to_full_q(result, world)
        in_polygon = compute_com_polygon_mask(world, q_batch)
        if bool(in_polygon.all()):
            return result
        fail_idx = (~in_polygon).nonzero(as_tuple=True)[0]
        retry_targets = world_from_ee[fail_idx]
        retry_result = _ik_for_pose(world, retry_targets, arm)
        result = _splice_ik_result(result, retry_result, fail_idx)
    # Final check for logging.
    q_batch_final = _ik_solution_to_full_q(result, world)
    in_polygon_final = compute_com_polygon_mask(world, q_batch_final)
    n_fail = int((~in_polygon_final).sum().item())
    if n_fail > 0:
        _log.warning(
            f"_ik_for_pose_com_safe: {n_fail}/{len(q_batch_final)} particles "
            f"still out of COM polygon after {max_retries} retries"
        )
    return result
```

### `_splice_ik_result` helper

Merges a retry batch into the original at the specified indices. Implementation depends on cuRobo's `IKResult` structure. Sketch:

```python
def _splice_ik_result(orig, retry, idx: torch.Tensor):
    """Replace orig fields at `idx` with retry values. Mutates orig in place
    where possible; otherwise returns a new IKResult-shaped object."""
    # orig.solution: [B, 1, dof]; retry.solution: [len(idx), 1, dof]
    orig.solution[idx] = retry.solution
    orig.success[idx] = retry.success
    # IKResult may have other fields (e.g., position_error, rotation_error)
    # — copy whichever are present and indexed by particle.
    for field_name in ("position_error", "rotation_error"):
        if hasattr(orig, field_name) and hasattr(retry, field_name):
            getattr(orig, field_name)[idx] = getattr(retry, field_name)
    return orig
```

The exact set of fields to splice is determined by reading `IKResult` (cuRobo internal) at implementation time.

### Replace call sites

`particle_initialization.py` has 4 call sites of `_ik_for_pose` (around lines 265, 341, 395, 470 per earlier grep). Replace each with `_ik_for_pose_com_safe(world, world_from_ee, arm, max_retries=config.ik_com_retry_max)`. Threading `config` requires adding it to the call signatures of `_ik_for_pose_com_safe` or making `max_retries` a global config read.

For simplicity, thread `max_retries` as a constructor-time arg into `ParticleInitializer` (which already has `self.config`), then pass through to each `_ik_for_pose_com_safe` call.

### Optional: pre-warm `world.kinematics_with_com`

`world.kinematics_with_com` is a `@cached_property` — first access lazy-builds the heavier compute_com kernel. First call inside `_ik_for_pose_com_safe` will pay this cost. Could pre-warm in `TAMPWorld.__init__` (after Layer 1 already enables it). Optional — pay-as-you-go is fine.

## Configuration

```python
# cutamp/config.py — add to TAMPConfiguration:
ik_com_retry_max: int = 3   # Layer 2 retry budget for post-IK COM verification.
```

Existing `enable_com_polygon: bool = True` gates BOTH the motion-planner cost AND (new) the IK-solver cost.

## Files / LOC estimate

| File | Modification | LOC |
|---|---|---|
| `cutamp/robots/t1.py` | `_ik_transition_dict_with_compute_com` + modify `get_t1_ik_solver` | ~30 |
| `cutamp/tamp_world.py` | `enable_com_polygon` to `__init__` + forward to IK builder | ~5 |
| `cutamp/com_polygon_cost.py` | `compute_com_polygon_mask` helper | ~25 |
| `cutamp/particle_initialization.py` | `_ik_for_pose_com_safe` + `_splice_ik_result` + replace 4 call sites | ~60 |
| `cutamp/config.py` | `ik_com_retry_max` field | ~3 |
| `cutamp/algorithm.py` | Pass `config.enable_com_polygon` to `TAMPWorld(...)` constructor | ~3 |
| `cutamp/tests/test_com_polygon_ik.py` (NEW) | Unit + integration tests | ~60 |

Total: ~185 LOC.

## Verification

### Unit tests
1. `test_compute_com_polygon_mask_on_synthetic_input`: build a mock `world.kinematics_with_com` (or use the real one), pass joint configs that put COM at known positions (inside polygon, outside in X, outside in Y, on boundary). Assert mask values match expectations.

2. `test_ik_solver_has_com_polygon_extra_cost`: after `TAMPWorld` init with `enable_com_polygon=True`, assert `world.ik_solver` has `"com_polygon"` in `mgr._extra_costs` for the LBFGS rollout's cost_manager.

3. `test_ik_solver_no_com_polygon_when_disabled`: `enable_com_polygon=False` → no `com_polygon` in `_extra_costs`.

### Integration test
4. `test_ik_pose_com_safe_retries_until_feasible`: construct a TAMPWorld, call `_ik_for_pose_com_safe(world, hard_target, "left")` where `hard_target` is a deliberately edge-of-reach pose. Assert that AT LEAST the retry loop runs (count via logging or mock injection) AND the final result has higher COM-feasibility than the first IK attempt would have alone.

### Visual regression
5. Re-run the exact pickle-generation command that produced the teetering image (`--env blocks_t1 -n 64 --num_opt_steps 200 --motion_plan --optimize_soft_costs --soft_cost com_polygon place_close_to_base --save_plan ...`). Open in rerun viewer. Verify the IK-init poses no longer show the teetering-on-tip configuration.

### Smoke
6. Existing smoke test: `python -m cutamp.scripts.run_cutamp --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan` passes (≥1 satisfying, motion plans succeed). Confirms no regression from Layer 1 changes.

## Risks / open questions

1. **LM seeder doesn't see COM cost**: cuRobo internal. If LM consistently seeds in bad configs, Layer 1's LBFGS may not have enough budget to recover; falls through to Layer 2 retries. If Layer 2 consistently fails, may need to bump `override_iters_for_multi_link_ik`.

2. **IK satisfaction rate may drop**: if `weight=5e5` is too strong relative to pose target, IK may fail to converge on hard grasps. Monitor satisfaction rate; tune `weight` down if observed (e.g., 1e5 or 5e4). The same cost applies to the planner where 5e5 has been shown to work — but planner has a horizon to amortize over, IK is single-point.

3. **Retry latency**: each retry is a full IK call on the failed-particle subset. Worst case (all 64 particles fail every retry): 4× IK time. Mitigation: bound at `max_retries=3` and log the rate. If high, fix Layer 1 weight rather than burning more retries.

4. **`_splice_ik_result` field coverage**: cuRobo's `IKResult` may have fields we don't enumerate. Missing fields would leave stale values for retried particles. Implementer must read the IKResult class definition and ensure all per-batch fields are spliced.

5. **`world.kinematics_with_com` build cost on first call**: one-time hit per process. Pre-warming in `TAMPWorld.__init__` adds ~100ms at startup but eliminates first-call latency.

## Out of scope

- Tuning the LM seeder to be COM-aware (cuRobo internal restructure).
- Grasp/pose resampling on Layer 2 failure (currently retries use the SAME target — only IK seed randomization can save it). If consistent failures observed, a future change might add grasp re-sampling in the retry loop.
- Touching the existing planner-side `com_polygon` cost or its weight.
- Adding new soft costs to the particle Adam loop (the existing `com_polygon` Adam soft cost stays as-is — it now operates on COM-feasible IK init thanks to Layers 1+2).
