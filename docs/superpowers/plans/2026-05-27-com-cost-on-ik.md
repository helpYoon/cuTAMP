# COM cost on IK solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the particle-init IK from producing teetering, out-of-base-polygon configurations by (a) registering the COM cost natively on the IK solver's LBFGS rollouts and (b) adding a batched post-IK COM verification + retry safety net.

**Architecture:** Two-layer defense. Layer 1: mirror the existing motion-planner-side pattern — inject `compute_com=True` through cuRobo's `ik/transition_ik.yml` dict, then `add_extra_cost(ik_solver, "com_polygon", ...)`. Layer 2: wrap `_ik_for_pose` in a helper that batch-computes COM-in-polygon mask via `world.kinematics_with_com` and retries failed particles up to N times.

**Tech Stack:** Python 3.10+, PyTorch, cuRobo v0.8, pytest. Conda env at `/home/yoonwoo/miniconda3/envs/tamp/bin/python`.

**Spec:** `docs/superpowers/specs/2026-05-27-com-cost-on-ik-design.md`

**Smoke test command** (used across tasks):
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan \
  --optimize_soft_costs --soft_cost com_polygon place_close_to_base
```
Pass: ≥1 satisfying solution, motion plans succeed.

**Notes for the implementer:**
- Branch is `curobo_v2`. Working tree has ~100 files of uncommitted prior port work. **Do NOT use `git add -A` / `git add .`.** Only stage the EXACT files listed.
- For pytest, prepend `PYTHONPATH=""` to avoid ROS Humble's pytest plugin collision (`launch_testing` requires `lark`).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `cutamp/robots/t1.py` | Modify | `_ik_transition_dict_with_compute_com()` helper; extend `get_t1_ik_solver` with `enable_com_polygon` param |
| `cutamp/tamp_world.py` | Modify | Forward `enable_com_polygon` from `__init__` to `get_t1_ik_solver` |
| `cutamp/algorithm.py` | Modify | Pass `config.enable_com_polygon` to `TAMPWorld(...)` constructor |
| `cutamp/com_polygon_cost.py` | Modify | New `compute_com_polygon_mask(world, q_batch)` helper |
| `cutamp/particle_initialization.py` | Modify | New `_ik_for_pose_com_safe` wrapper + `_splice_ik_result` helper; replace 4 `_ik_for_pose` call sites |
| `cutamp/config.py` | Modify | New `ik_com_retry_max: int = 3` field |
| `cutamp/tests/test_com_polygon_ik.py` | Create | Unit tests for Layer 1 wiring + Layer 2 mask helper |

Total: ~185 LOC.

---

## Task 1: Layer 1 — Register COM cost on the IK solver

**Findings A1, A2**: IK has no signal about COM. Register the same cost we use on the motion planner, via the same `add_extra_cost` + `transition_model` plumbing.

**Files:**
- Modify: `cutamp/robots/t1.py`
- Modify: `cutamp/tamp_world.py`
- Modify: `cutamp/algorithm.py`
- Test: `cutamp/tests/test_com_polygon_ik.py` (NEW)

### Step 1.1: Write the failing test for IK cost registration

- [ ] Create `cutamp/tests/test_com_polygon_ik.py`:

```python
"""Tests for the two-layer COM cost on the IK solver."""
import os
import pytest


needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _make_world(enable_com_polygon: bool = True):
    """Build a real TAMPWorld for blocks_t1 with optional COM toggle."""
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from curobo.types import DeviceCfg
    from cutamp.robots.t1 import t1_home
    import torch

    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    device_cfg = DeviceCfg()
    robot = load_robot_container("t1", device_cfg)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=device_cfg.device)
    return TAMPWorld(
        env=env, device_cfg=device_cfg, robot=robot, q_init=q_init,
        enable_com_polygon=enable_com_polygon,
    )


def _ik_extra_costs(world):
    """Return the union of _extra_costs dicts across all IK rollout cost managers."""
    from cutamp._curobo_internals import iter_rollouts
    names = set()
    for rollout in iter_rollouts(world.ik_solver):
        for mgr in (
            getattr(rollout, "cost_manager", None),
            getattr(rollout, "metrics_cost_manager", None),
        ):
            if mgr is None:
                continue
            extras = getattr(mgr, "_extra_costs", {}) or {}
            names.update(extras.keys())
    return names


@needs_cuda
def test_ik_solver_has_com_polygon_extra_cost_when_enabled():
    """Default world (enable_com_polygon=True) registers com_polygon on
    the IK solver's rollouts via add_extra_cost."""
    world = _make_world(enable_com_polygon=True)
    assert "com_polygon" in _ik_extra_costs(world), (
        "IK solver should have com_polygon in its _extra_costs when "
        "enable_com_polygon=True"
    )


@needs_cuda
def test_ik_solver_no_com_polygon_when_disabled():
    """enable_com_polygon=False skips IK cost registration entirely."""
    world = _make_world(enable_com_polygon=False)
    assert "com_polygon" not in _ik_extra_costs(world), (
        "IK solver should NOT have com_polygon registered when "
        "enable_com_polygon=False"
    )
```

The `_make_world` helper passes `enable_com_polygon` as a kwarg — that kwarg must exist on `TAMPWorld.__init__` (we'll add it in step 1.4). The test references will surface that gap first via TypeError, then via the cost-registration logic.

### Step 1.2: Run the tests to verify they fail

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP
PYTHONPATH="" PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py::test_ik_solver_has_com_polygon_extra_cost_when_enabled \
  cutamp/tests/test_com_polygon_ik.py::test_ik_solver_no_com_polygon_when_disabled \
  -v 2>&1 | tail -20
```

Expected: both FAIL. The "enabled" test fails because the COM cost isn't yet registered on IK; the "disabled" test likely fails first with `TypeError: TAMPWorld.__init__ got an unexpected keyword argument 'enable_com_polygon'` until step 1.5 adds it.

### Step 1.3: Add `_ik_transition_dict_with_compute_com()` helper

- [ ] Edit `cutamp/robots/t1.py`. Locate `_trajopt_transition_dict_with_compute_com()` (currently around line 159-172). Add the IK mirror immediately after it:

```python
def _ik_transition_dict_with_compute_com() -> dict:
    """Default IK transition YAML with ``compute_com=True`` injected.

    cuRobo's IK kernel template is fixed at IK-build time (analogous to
    the motion planner). We inject ``compute_com=True`` so
    ``state.cuda_robot_model_state.robot_com`` is populated during the
    LBFGS refinement stage, where the COM-over-base soft cost reads it.
    """
    from curobo._src.util.config_io import resolve_config, join_path
    from curobo.content import get_task_configs_path

    d = resolve_config(join_path(get_task_configs_path(), "ik/transition_ik.yml"))
    d["transition_model_cfg"]["compute_com"] = True
    return d
```

This is an exact mirror of the trajopt version one block above; only the YAML filename changes from `trajopt/transition_bspline_trajopt.yml` to `ik/transition_ik.yml`. Verified via `find /home/yoonwoo/cuTAMP/curobo -name "transition_ik*"` that the file exists at the expected path.

### Step 1.4: Modify `get_t1_ik_solver` to accept `enable_com_polygon` + wire the cost

- [ ] Edit `cutamp/robots/t1.py`. Find `get_t1_ik_solver` (current signature at line 244-251). Replace the function with:

```python
def get_t1_ik_solver(
    scene: Scene,
    *,
    num_seeds: int = 32,
    self_collision_check: bool = True,
    max_batch_size: int = 64,
    enable_com_polygon: bool = True,
    device_cfg: DeviceCfg = None,
) -> InverseKinematics:
    """InverseKinematics solver for T1 with the mobile base locked.

    Always built with both tool frames from ``t1_planar_base.yml``. The
    inactive arm during particle-init IK calls is held at home via the
    same cspace-pin + multi-frame-goal mechanism the motion planner uses
    (``T1State.pin_for_arm_action`` + ``_build_multi_frame_goal``).

    The 3 base DOFs (``base_j_x``, ``base_j_y``, ``base_j_yaw``) are pinned
    at 0 — IK cannot drift the base during particle init. Legs / torso /
    waist / arms remain free.

    When ``enable_com_polygon`` is True (default), the IK solver gets the
    same COM-over-base-polygon soft cost as the motion planner. Without
    this, IK produces extreme leaning configurations whose COM projects
    outside the wheelbase rectangle — the Adam-side soft cost can't
    recover from these because pulling the body back would violate the
    KinematicConstraint.
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
        # CUDA graphs specialize for the input shape captured at first call.
        # Particle initialization first warms IK without ``current_state``
        # (no seed) and later wants to pass ``current_state`` (per-particle
        # seed) for body chaining; that shape change can't be re-captured
        # ("CUDA graph reset is not available."). Disable so each call
        # re-prepares its execution plan rather than crashing.
        use_cuda_graph=False,
        transition_model=_ik_transition_dict_with_compute_com(),
    )
    ik_solver = InverseKinematics(cfg)
    if enable_com_polygon:
        from cutamp._curobo_internals import add_extra_cost
        from cutamp.com_polygon_cost import (
            ComOverBasePolygonCost,
            ComOverBasePolygonCostCfg,
        )
        rollout_device_cfg = ik_solver.optimizer_rollouts[0].device_cfg
        cost_cfg = ComOverBasePolygonCostCfg(
            weight=[5.0e5],
            device_cfg=rollout_device_cfg,
            half_extents=[0.05, 0.10],   # 10×20 cm support
            inside_margin=0.0,            # no inside barrier
            inside_weight=0.0,            # disable barrier term entirely
        )
        add_extra_cost(ik_solver, "com_polygon", ComOverBasePolygonCost(cost_cfg))
    return ik_solver
```

Two notable additions relative to current code:
- `enable_com_polygon: bool = True` keyword-only kwarg.
- `transition_model=_ik_transition_dict_with_compute_com()` passed to `InverseKinematicsCfg.create`.
- `if enable_com_polygon: ... add_extra_cost(...)` block at the end.

Preserve the existing `use_cuda_graph=False` line and comment — don't remove it during the edit.

**If `ik_solver.optimizer_rollouts` is not a valid attribute name** (the planner uses `trajopt_solver.optimizer_rollouts`; we already use `iter_rollouts(host)` to abstract this), surface as NEEDS_CONTEXT — but our existing `_curobo_internals.iter_rollouts` covers both cases, so this should work.

### Step 1.5: Add `enable_com_polygon` to `TAMPWorld.__init__`

- [ ] Edit `cutamp/tamp_world.py`. Find the `__init__` signature (currently `def __init__(self, env, device_cfg, robot, q_init, collision_activation_distance=0.0, coll_n_spheres=50, coll_sphere_radius=0.005)` around line 56-65). Add `enable_com_polygon: bool = True` to the end of the signature:

```python
    def __init__(
        self,
        env: TAMPEnvironment,
        device_cfg: DeviceCfg,
        robot: Union[str, RobotContainer],
        q_init: Float[torch.Tensor, "dof"],
        collision_activation_distance: float = 0.0,
        coll_n_spheres: int = 50,
        coll_sphere_radius: float = 0.005,
        enable_com_polygon: bool = True,
    ):
```

- [ ] In the same `__init__`, find the existing `self.ik_solver: InverseKinematics = get_t1_ik_solver(...)` call (around line 98). Add `enable_com_polygon=enable_com_polygon`:

```python
        self.ik_solver: InverseKinematics = get_t1_ik_solver(
            self.world_cfg, device_cfg=device_cfg, max_batch_size=512,
            enable_com_polygon=enable_com_polygon,
        )
```

### Step 1.6: Forward `config.enable_com_polygon` to `TAMPWorld` from the call site

- [ ] Grep `TAMPWorld(` in `cutamp/algorithm.py` to find the constructor call:

```bash
grep -n "TAMPWorld(" /home/yoonwoo/cuTAMP/cutamp/algorithm.py
```

- [ ] Edit the call site to pass `enable_com_polygon=config.enable_com_polygon`. The exact line context depends on what you find. Sketch:

```python
world = TAMPWorld(
    env=env,
    device_cfg=device_cfg,
    robot=robot_container,
    q_init=q_init,
    enable_com_polygon=config.enable_com_polygon,   # NEW
    # ...other existing kwargs...
)
```

If the existing call uses positional args, switch to kwargs for clarity.

### Step 1.7: Run the unit tests — should now PASS

- [ ] Run:

```bash
PYTHONPATH="" PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py::test_ik_solver_has_com_polygon_extra_cost_when_enabled \
  cutamp/tests/test_com_polygon_ik.py::test_ik_solver_no_com_polygon_when_disabled \
  -v 2>&1 | tail -15
```

Expected: both PASS.

### Step 1.8: Run smoke test (no regression)

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan \
  --optimize_soft_costs --soft_cost com_polygon place_close_to_base 2>&1 | tail -25
```

Expected: ≥1 satisfying. IK times may be slightly higher (~5-10%) due to the extra cost evaluations per LBFGS iteration; tolerable.

**If satisfying count drops to 0** (or significantly): the IK COM cost weight may be too strong for IK to find any solution. Mitigation: try `weight=[1.0e5]` instead of `5e5` and re-run. If still bad, surface as DONE_WITH_CONCERNS — the weight needs tuning.

### Step 1.9: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/robots/t1.py \
  cutamp/tamp_world.py \
  cutamp/algorithm.py \
  cutamp/tests/test_com_polygon_ik.py && \
git commit -m "$(cat <<'EOF'
feat: register COM-over-base cost on IK solver (Layer 1)

Mirrors the motion-planner-side pattern: inject compute_com=True via the
ik/transition_ik.yml transition model dict, then call
add_extra_cost(ik_solver, "com_polygon", ...) so the cost fires during
LBFGS refinement. Gated on enable_com_polygon (same flag as planner).

Fixes the failure mode where particle-init IK produced teetering,
out-of-base-polygon configurations because IK had no signal about COM.
The Adam-side soft cost couldn't recover because pulling the body back
violated the KinematicConstraint.

Layer 2 (post-IK retry safety net) lands in the next commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Layer 2 — `compute_com_polygon_mask` helper

**Finding C1**: Need a batched COM-in-polygon check to drive the Layer 2 retry decision.

**Files:**
- Modify: `cutamp/com_polygon_cost.py` (add helper)
- Test: extend `cutamp/tests/test_com_polygon_ik.py`

### Step 2.1: Write the failing test for `compute_com_polygon_mask`

- [ ] Append to `cutamp/tests/test_com_polygon_ik.py`:

```python
@needs_cuda
def test_compute_com_polygon_mask_basic():
    """Verify batched COM-in-polygon check returns the expected shape +
    correctly classifies a home-pose batch as inside the polygon."""
    import torch
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    world = _make_world(enable_com_polygon=True)
    # At home pose all DOFs are 0; COM is directly above the wheelbase
    # center → inside polygon for sure. Build a [B=4, full_dof] batch all
    # at home and assert all four come back True.
    home = world.q_init.detach().clone()
    B = 4
    q_batch = home.unsqueeze(0).expand(B, -1).contiguous()
    mask = compute_com_polygon_mask(world, q_batch)
    assert mask.shape == (B,), f"expected shape ({B},), got {mask.shape}"
    assert bool(mask.all()), (
        f"home pose should be inside polygon; got mask={mask}"
    )


@needs_cuda
def test_compute_com_polygon_mask_excludes_extreme_lean():
    """A bent-far-forward configuration should be classified as OUTSIDE
    the polygon. Constructs a synthetic q_batch with deeply-bent
    Torso_Pitch + ankle_pitch + knee_pitch (mimicking the teetering pose
    we observed pre-fix)."""
    import torch
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    world = _make_world(enable_com_polygon=True)
    full_names = list(world.kinematics.joint_names)
    home = world.q_init.detach().clone()
    # Build a configuration with deep forward bend.
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    bent = home.clone()
    bent[name_to_idx["Torso_Pitch"]]  = -1.7
    bent[name_to_idx["ankle_pitch"]]  = -0.5
    bent[name_to_idx["knee_pitch"]]   = +0.8
    q_batch = torch.stack([home, bent], dim=0)  # [2, full_dof]
    mask = compute_com_polygon_mask(world, q_batch)
    assert mask.shape == (2,)
    assert bool(mask[0]),    f"home should be inside polygon; mask[0]={mask[0]}"
    assert not bool(mask[1]), f"deeply-bent pose should be OUTSIDE polygon; mask[1]={mask[1]}"
```

### Step 2.2: Run the tests to verify they fail

- [ ] Run:

```bash
PYTHONPATH="" PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py::test_compute_com_polygon_mask_basic \
  cutamp/tests/test_com_polygon_ik.py::test_compute_com_polygon_mask_excludes_extreme_lean \
  -v 2>&1 | tail -15
```

Expected: both FAIL with `ImportError: cannot import name 'compute_com_polygon_mask' from 'cutamp.com_polygon_cost'`.

### Step 2.3: Implement `compute_com_polygon_mask`

- [ ] Edit `cutamp/com_polygon_cost.py`. Add at the end of the file (after the `ComOverBasePolygonCost` class definition):

```python
def compute_com_polygon_mask(
    world,
    q_batch: torch.Tensor,
    *,
    half_extents: Optional[List[float]] = None,
) -> torch.Tensor:
    """Batched COM-in-polygon check.

    Computes per-link mass-weighted world COM via ``world.kinematics_with_com``
    (which is built with ``compute_com=True``), projects it into the
    ``mobile_base_link`` frame, and returns a boolean mask of which batch
    elements have their COM inside an axis-aligned rectangle around the
    base origin.

    Args:
        world: TAMPWorld instance. Must have ``kinematics_with_com``
            available (cached property — first call builds it lazily).
        q_batch: ``[B, full_dof]`` joint positions in
            ``world.kinematics.joint_names`` order.
        half_extents: ``[2]``-list of (X, Y) half-sizes in meters. Default
            ``[0.05, 0.10]`` matches the safety rectangle used by the
            planner-side COM cost.

    Returns:
        ``[B]`` bool tensor; True where the projected COM is inside the
        rectangle (per-axis ``abs(com_in_base[i]) <= half_extents[i]``).
    """
    from curobo.types import JointState
    if half_extents is None:
        half_extents = [0.05, 0.10]
    kin = world.kinematics_with_com
    js = JointState.from_position(
        q_batch.to(kin.device_cfg.device),
        joint_names=list(world.kinematics.joint_names),
    )
    active_js = kin.get_active_js(js)
    ks = kin.compute_kinematics(active_js)
    com = ks.cuda_robot_model_state.robot_com[..., :3].view(-1, 3)
    base_T = (
        ks.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True)
        .get_matrix()
        .view(-1, 4, 4)
    )
    R = base_T[:, :3, :3]
    t = base_T[:, :3, 3]
    com_in_base = (R.transpose(-1, -2) @ (com - t).unsqueeze(-1)).squeeze(-1)[:, :2]
    half = torch.tensor(
        half_extents, device=com.device, dtype=com.dtype,
    )
    return (torch.abs(com_in_base) <= half).all(dim=-1)
```

Note: the import-side has `from typing import List, Optional` and `import torch` already present at the top of `com_polygon_cost.py` (per the existing file). If `Optional` isn't imported, add it.

**Adapt if needed**: if `kin.get_active_js(...)` is not the right reprojection (the planner's full→active conversion), use whichever mechanism the existing `_to_active_cspace` pattern uses. The end requirement is: `q_batch` is in full cspace order; cuRobo's `compute_kinematics` needs the active-cspace JointState. The intent is shown in `cutamp/utils/plan_processor.py:_to_active_cspace` if you need a reference.

### Step 2.4: Run the tests to verify they pass

- [ ] Run:

```bash
PYTHONPATH="" PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py::test_compute_com_polygon_mask_basic \
  cutamp/tests/test_com_polygon_ik.py::test_compute_com_polygon_mask_excludes_extreme_lean \
  -v 2>&1 | tail -15
```

Expected: both PASS.

### Step 2.5: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/com_polygon_cost.py \
  cutamp/tests/test_com_polygon_ik.py && \
git commit -m "$(cat <<'EOF'
feat: compute_com_polygon_mask batched helper

Adds a batched COM-in-polygon check used by Layer 2's post-IK retry
wrapper (next commit). Uses world.kinematics_with_com (compute_com=True)
to FK + extract COM in one CUDA kernel call, projects into the
mobile_base_link frame, returns a boolean mask of batch elements inside
the axis-aligned safety rectangle.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Layer 2 — `_ik_for_pose_com_safe` wrapper + replace call sites

**Findings C2-C5**: Wire the helper into the IK retry loop and replace existing `_ik_for_pose` call sites.

**Files:**
- Modify: `cutamp/particle_initialization.py` (new helpers + replace 4 call sites)
- Modify: `cutamp/config.py` (new field)

### Step 3.1: Add `ik_com_retry_max` to TAMPConfiguration

- [ ] Edit `cutamp/config.py`. Find the existing `enable_com_polygon: bool = True` field (added in an earlier commit). Add immediately after it:

```python
    ik_com_retry_max: int = 3   # Layer 2: max retries when post-IK COM check fails.
```

If `enable_com_polygon` doesn't exist (it should from earlier work — verify via grep), this means the prior code-review-fixes bundle wasn't applied. Surface as NEEDS_CONTEXT.

### Step 3.2: Add `_splice_ik_result` helper to `particle_initialization.py`

- [ ] Edit `cutamp/particle_initialization.py`. Add the helper right after `_ik_for_pose` (after line ~115):

```python
def _splice_ik_result(orig, retry, idx: torch.Tensor):
    """Replace ``orig``'s per-batch fields at ``idx`` with values from ``retry``.

    cuRobo's IKSolverResult (subclass of BaseSolverResult) has per-batch
    fields ``success``, ``solution``, ``position_error``, ``rotation_error``,
    ``js_solution``, ``goalset_index``. Scalar fields (``solve_time``,
    ``debug_info``) are not per-batch and are not spliced.

    Mutates ``orig`` in place; returns it for chaining.
    """
    for field_name in (
        "success", "solution", "position_error", "rotation_error",
        "js_solution", "goalset_index",
    ):
        orig_val = getattr(orig, field_name, None)
        retry_val = getattr(retry, field_name, None)
        if orig_val is None or retry_val is None:
            continue
        # All per-batch fields are tensors indexable on dim 0.
        orig_val[idx] = retry_val
    return orig
```

The field list is taken from `BaseSolverResult` (`curobo/_src/solver/solver_base_result.py:55-59` and surrounding fields). If cuRobo introduces a new per-batch field, this helper will silently skip it. Acceptable for now; if observed, add to the list.

### Step 3.3: Write the failing integration test for `_ik_for_pose_com_safe`

- [ ] Append to `cutamp/tests/test_com_polygon_ik.py`:

```python
@needs_cuda
def test_ik_for_pose_com_safe_returns_valid_result():
    """End-to-end smoke check: _ik_for_pose_com_safe returns an IK result
    with the expected shape on a real grasp target. We don't assert
    everything-in-polygon (Layer 1 should help, but hard targets may
    still fail) — just that the wrapper runs without error and returns
    a usable result."""
    import torch
    from cutamp.particle_initialization import _ik_for_pose_com_safe
    world = _make_world(enable_com_polygon=True)
    # Build a [B=4, 4, 4] batch of reachable left-hand targets.
    # Pose: hand at (0.4, +0.2, 0.5) world — well within left arm reach.
    B = 4
    target = torch.eye(4, device=world.kinematics.device_cfg.device).unsqueeze(0).expand(B, 4, 4).contiguous()
    target[..., 0, 3] = 0.4
    target[..., 1, 3] = 0.2
    target[..., 2, 3] = 0.5
    result = _ik_for_pose_com_safe(world, target, "left", max_retries=2)
    # Result must have success + solution fields with batch dim B.
    assert hasattr(result, "success") and result.success.shape[0] == B
    assert hasattr(result, "solution") and result.solution.shape[0] == B
```

### Step 3.4: Run the test to verify it fails

- [ ] Run:

```bash
PYTHONPATH="" PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py::test_ik_for_pose_com_safe_returns_valid_result \
  -v 2>&1 | tail -10
```

Expected: FAIL with `ImportError: cannot import name '_ik_for_pose_com_safe' from 'cutamp.particle_initialization'`.

### Step 3.5: Implement `_ik_for_pose_com_safe` wrapper

- [ ] Edit `cutamp/particle_initialization.py`. Add right after `_splice_ik_result` (from step 3.2):

```python
def _ik_for_pose_com_safe(
    world: TAMPWorld,
    world_from_ee: torch.Tensor,
    arm: Optional[str],
    *,
    max_retries: int = 3,
):
    """Call ``_ik_for_pose``; verify COM-in-polygon; retry failed particles.

    Layer 2 of the two-layer COM defense. Layer 1 (cost registered on
    IK rollouts) makes IK natively COM-aware; this wrapper catches the
    LBFGS-didn't-converge-to-COM-feasible edge case by re-calling IK
    on the failed-particle subset. cuRobo's seed sampler has internal
    randomization so retries may converge to different (feasible)
    solutions.

    Returns the last batch result (best effort). Logs final failure count
    at WARNING level so persistent failures are visible.
    """
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    result = _ik_for_pose(world, world_from_ee, arm)
    for _attempt in range(max_retries):
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

### Step 3.6: Replace `_ik_for_pose` call sites with `_ik_for_pose_com_safe`

- [ ] Grep to find all call sites:

```bash
grep -n "_ik_for_pose(world" /home/yoonwoo/cuTAMP/cutamp/particle_initialization.py
```

You should find 4 call sites (per earlier inspection: around lines 265, 341, 395, 470). For each, replace `_ik_for_pose(world, world_from_ee, arm)` with `_ik_for_pose_com_safe(world, world_from_ee, arm, max_retries=self.config.ik_com_retry_max)`.

The 4 call sites are all inside `ParticleInitializer` methods, so `self.config` is in scope. Verify the exact line context before each edit so you don't accidentally replace a different call (e.g., the test_coupled_reik path also calls `_ik_for_pose` but with `current_state_q=`).

For the `_refresh_ik_deps` call in `cutamp/optimize_plan.py` (which uses `current_state_q=`), DO NOT replace — that path serves a different purpose (warm-start refresh during Adam) and shouldn't trigger COM retries. Leave that call as-is.

### Step 3.7: Run the integration test

- [ ] Run:

```bash
PYTHONPATH="" PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py::test_ik_for_pose_com_safe_returns_valid_result \
  -v 2>&1 | tail -10
```

Expected: PASS.

### Step 3.8: Run the full file's tests

- [ ] Run:

```bash
PYTHONPATH="" PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py -v 2>&1 | tail -15
```

Expected: all 5 tests pass (2 from Task 1 + 2 from Task 2 + 1 from Task 3).

### Step 3.9: Run smoke test

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan \
  --optimize_soft_costs --soft_cost com_polygon place_close_to_base 2>&1 | tail -25
```

Expected: ≥1 satisfying solution, motion plans succeed. Look for any `_ik_for_pose_com_safe: N/M particles still out of COM polygon` WARNING messages — if they appear, note the rate (small is fine; many is a sign Layer 1 weight may need tuning).

### Step 3.10: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/particle_initialization.py \
  cutamp/config.py \
  cutamp/tests/test_com_polygon_ik.py && \
git commit -m "$(cat <<'EOF'
feat: _ik_for_pose_com_safe — post-IK batched COM retry (Layer 2)

Wraps _ik_for_pose to verify COM-in-polygon via the new
compute_com_polygon_mask helper. Failed particles get retried up to
TAMPConfiguration.ik_com_retry_max times (default 3), with cuRobo's
seed-sampler randomization driving different IK convergence on retry.

Replaces 4 _ik_for_pose call sites in ParticleInitializer with the
safe wrapper. The _refresh_ik_deps call in optimize_plan stays
unchanged — it serves coupled-reIK refresh, not initial particle IK.

Together with Layer 1 (COM cost on IK rollouts), this eliminates the
teetering-pose failure mode where particle-init produced configurations
with COM outside the wheelbase polygon.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Visual regression verification

**Pure verification task — no code changes.**

The original failure was visible in rerun viewer (the teetering pose). Confirm visually that Layers 1+2 eliminate it.

### Step 4.1: Regenerate the pickle with visualization

- [ ] Run:

```bash
rm -f /home/yoonwoo/cuTAMP/data/motion_plan.pkl && \
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 -n 64 --num_opt_steps 200 --motion_plan \
  --optimize_soft_costs --soft_cost com_polygon place_close_to_base \
  --save_plan /home/yoonwoo/cuTAMP/data/motion_plan.pkl 2>&1 | tail -25
```

Pass criteria:
- ≥1 satisfying solution.
- All 12 motion plan segments succeed.
- No `_ik_for_pose_com_safe: ... still out of COM polygon` WARNING (or a small count — say ≤5 per run; ratio sub-10%).
- In the rerun viewer (auto-spawns when not `--disable_visualizer`), confirm IK-init poses look stable — body upright, no extreme leg-back lean, no foot lifted into the air.

### Step 4.2: Capture the result for the record

- [ ] Take a screenshot or note the satisfying counts + final plan skeleton to confirm the fix held end-to-end.

Note: this step is informational. If poses STILL show teetering, the most likely cause is that the 5e5 cost weight is overpowered by the pose target on hard grasps. Mitigation paths in that case:
- Bump `weight` to `1e6` or higher.
- Add the inside-margin barrier (`inside_margin=0.02, inside_weight=1.0`) — safe on IK since there's no `body_home_posture` to fight.
- Increase `ik_com_retry_max` to 5 or 10 — diminishing returns above ~5.

### Step 4.3: Optional pre-warm (if first-call latency observed)

- [ ] `world.kinematics_with_com` is a `@cached_property` — first access builds the heavier compute_com kernel. If you notice ~100ms latency on the first `_ik_for_pose_com_safe` call (visible as a stutter on the first particle batch), add a pre-warm to `TAMPWorld.__init__`:

```python
        # End of TAMPWorld.__init__ — after arm_home_ee_world block
        # Pre-warm the compute_com kernel so the first _ik_for_pose_com_safe
        # call doesn't pay the lazy-build cost.
        _ = self.kinematics_with_com
```

If first-call latency isn't noticeable, skip this step.

---

## Self-review checklist (executed by plan-writer, post-write)

**1. Spec coverage**:
- Layer 1 cost registration → Task 1 ✅
- `_ik_transition_dict_with_compute_com` helper → Step 1.3 ✅
- `enable_com_polygon` flag on `get_t1_ik_solver` + `TAMPWorld` + caller → Steps 1.4-1.6 ✅
- `compute_com_polygon_mask` helper → Task 2 ✅
- `_ik_for_pose_com_safe` wrapper + retry → Task 3 ✅
- `_splice_ik_result` helper → Step 3.2 ✅
- `ik_com_retry_max` config field → Step 3.1 ✅
- Replace 4 `_ik_for_pose` call sites → Step 3.6 ✅
- Tests: unit for mask + unit for cost registration + integration for retry → all covered ✅
- Visual regression → Task 4 ✅
- Pre-warm optional → Step 4.3 ✅

**2. Placeholder scan**: no TBD/TODO; all code blocks complete. Adapt-if-needed branches in steps 1.4, 1.6, 2.3, 3.1 specify what to do in each branch. ✅

**3. Type consistency**:
- `enable_com_polygon` is `bool = True` everywhere (cfg, `__init__`, `get_t1_ik_solver`).
- `ik_com_retry_max` is `int = 3` (cfg) and `max_retries: int = 3` (wrapper signature) — different name but same semantics; intentional (wrapper accepts the value, config holds the source).
- `compute_com_polygon_mask` returns `[B]` bool; `_ik_for_pose_com_safe` calls `bool(in_polygon.all())` and uses `(~in_polygon).nonzero(...)` — both work on the bool tensor. ✅
- Cost config matches motion planner exactly (`weight=[5e5]`, `half_extents=[0.05, 0.10]`, `inside_margin=0`, `inside_weight=0`) — per spec decision. ✅
