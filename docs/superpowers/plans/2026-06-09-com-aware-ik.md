# CoM-aware IK (seed-IK LM residual) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cuRobo's seed-IK LM solver natively trade hand-pose vs center-of-mass so T1 pick/place endpoint confs come out centered in the support rectangle (legs recruited within joint limits) instead of at the tipping edge.

**Architecture:** Add an rhs-only CoM residual to cuRobo's `SeedIKErrorCalculator` pose block (single shared autograd backward → jTerror; scalar added to `error_norm`; **no** jacobian rows, so the LM kernel/templates are untouched), plumbed through the existing `seed_*` cfg pass-through chain and gated by a new `enable_com_aware_ik` flag in cuTAMP (default off → byte-identical). Spec: `docs/superpowers/specs/2026-06-09-com-aware-ik-design.md`.

**Tech Stack:** Python/torch only (no CUDA). Vendored cuRobo at `curobo/` is a **nested git repo**: fork edits there stay **uncommitted in the nested repo** (same convention as the existing 3 modified files — check with `git -C curobo diff --stat`; NEVER `git add` inside `curobo/`, never push upstream). All commits below are in the cuTAMP repo. Stage only named files (never `git add -A`).

**Test command prefix (always):**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest
```
GPU is a single 24 GiB RTX 4090 — never run GPU tests while a pipeline run is active.

**File map:**
- Fork (vendored cuRobo, uncommitted): `curobo/curobo/_src/solver/solver_ik_cfg.py` (6 `seed_com_*` fields + create params), `curobo/curobo/_src/solver/solver_ik.py` (pass-through), `curobo/curobo/_src/solver/seed_ik/seed_ik_solver_cfg.py` (6 `com_*` fields), `curobo/curobo/_src/solver/seed_ik/seed_ik_solver.py` (`compute_com` + guard), `curobo/curobo/_src/solver/seed_ik/seed_ik_error_calculator.py` (residual).
- cuTAMP: `cutamp/com_polygon_cost.py` (center_weight param), `cutamp/robots/t1.py` (flag + weight constant), `cutamp/config.py`, `cutamp/tamp_world.py`, `cutamp/algorithm.py`, `cutamp/scripts/run_cutamp.py` (threading/CLI), `cutamp/particle_initialization.py` (success-mask + num_seeds fixes).
- Tests: `cutamp/tests/test_com_aware_seed_ik.py` (new), `cutamp/tests/conftest.py` (fixture hoist), `cutamp/tests/test_com_polygon_ik.py` (Task 10 replacements), `cutamp/tests/test_motion_anchor.py` (fixture removal).

---

### Task 1: `com_polygon_penalty` regains optional `center_weight`

The fork's residual includes a pull-to-center term; cuTAMP's reference penalty needs the same optional term for the parity test (Task 4). Default 0 keeps every existing caller (hard gate, mask, planner cost) byte-identical.

**Files:**
- Modify: `cutamp/com_polygon_cost.py:74-105` (`com_polygon_penalty`)
- Test: `cutamp/tests/test_com_polygon_ik.py`

- [ ] **Step 1: Write the failing tests** — append to `cutamp/tests/test_com_polygon_ik.py` (before `test_com_polygon_penalty_just_outside_exceeds_tol`):

```python
def test_com_polygon_penalty_center_weight_defaults_off():
    # Default center_weight=0 -> barrier-only: deep-inside COM scores exactly 0
    # (the hard-gate path relies on this).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    base_T = torch.eye(4).unsqueeze(0)
    com_world = torch.tensor([[0.03, 0.04, 0.0]])  # well inside inset rect
    pen = com_polygon_penalty(com_world, base_T, half, 0.02, 1.0)
    assert pen.item() == 0.0


def test_com_polygon_penalty_center_pull_active_deep_inside():
    # With center_weight>0 a deep-inside COM gets a nonzero penalty that grows
    # with distance from center: center_weight * sum((com/half)^2).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    base_T = torch.eye(4).unsqueeze(0)
    cw = 0.01
    far = torch.tensor([[0.06, 0.0, 0.0]])      # inside inset, barrier=0
    center = torch.tensor([[0.0, 0.0, 0.0]])
    pen_far = com_polygon_penalty(far, base_T, half, 0.02, 1.0, center_weight=cw)
    pen_center = com_polygon_penalty(center, base_T, half, 0.02, 1.0, center_weight=cw)
    assert pen_center.item() == pytest.approx(0.0)
    assert pen_far.item() == pytest.approx(cw * (0.06 / 0.10) ** 2, rel=1e-4)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ... -m pytest cutamp/tests/test_com_polygon_ik.py -k center -q`
Expected: 1 FAIL (`unexpected keyword argument 'center_weight'`), 1 PASS (defaults test).

- [ ] **Step 3: Implement** — in `cutamp/com_polygon_cost.py`, change `com_polygon_penalty`'s signature and tail:

```python
def com_polygon_penalty(
    com_world: torch.Tensor,
    base_T: torch.Tensor,
    half_extents: torch.Tensor,
    inside_margin: float,
    inside_weight: float,
    center_weight: float = 0.0,
) -> torch.Tensor:
```
Add to the docstring Args: `center_weight: optional pull-to-center term
``center_weight * Σ(com_in_base/half_extents)²`` active everywhere inside;
0 (default) preserves the pure barrier the hard gate's COM_TOL is calibrated to.`
Replace the final `return` line with:

```python
    penalty = (outside ** 2).sum(dim=-1) + inside_weight * (inside ** 2).max(dim=-1).values
    if center_weight:
        penalty = penalty + center_weight * ((com_in_base / half_extents) ** 2).sum(dim=-1)
    return penalty
```

- [ ] **Step 4: Run to verify pass** — same command; expected: all `center` tests PASS. Then run the file's full CPU set: `-k "not cuda" cutamp/tests/test_com_polygon_ik.py -q` → all pass.

- [ ] **Step 5: Commit**
```bash
git add cutamp/com_polygon_cost.py cutamp/tests/test_com_polygon_ik.py
git commit -m "feat: optional center_weight term in com_polygon_penalty (default off)"
```

---

### Task 2: cfg plumbing chain + cuTAMP flag threading

Six `com_*` knobs flow: `TAMPConfiguration.enable_com_aware_ik` → `TAMPWorld` → `get_t1_ik_solver` → `InverseKinematicsCfg.create(seed_com_*=...)` → `solver_ik.py` pass-through → `SeedIKSolverCfg`. Defaults keep everything off.

**Files:**
- Modify (fork): `curobo/curobo/_src/solver/seed_ik/seed_ik_solver_cfg.py` (fields after `acceleration_weight`, ~line 94), `curobo/curobo/_src/solver/solver_ik_cfg.py` (fields ~line 93, create params ~line 168, pass-through ~line 284), `curobo/curobo/_src/solver/solver_ik.py:160-164` (forward to `SeedIKSolverCfg.create` — it accepts `**kwargs`, no signature change there)
- Modify (cuTAMP): `cutamp/robots/t1.py` (`get_t1_ik_solver`), `cutamp/config.py:84`, `cutamp/tamp_world.py:65,101`, `cutamp/algorithm.py:374`, `cutamp/scripts/run_cutamp.py:186` area
- Test: `cutamp/tests/test_com_aware_seed_ik.py` (new)

- [ ] **Step 1: Write the failing test** — create `cutamp/tests/test_com_aware_seed_ik.py`:

```python
"""Tests for CoM-aware seed-IK (LM residual fork). Spec:
docs/superpowers/specs/2026-06-09-com-aware-ik-design.md"""
import os
import pytest
import torch

needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _world(enable_com_aware_ik: bool):
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from cutamp.robots.t1 import t1_home
    from curobo.types import DeviceCfg
    dc = DeviceCfg()
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    robot = load_robot_container("t1", dc)
    qh = torch.as_tensor(t1_home, dtype=torch.float32, device=dc.device)
    return TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=qh,
                     enable_com_polygon=True, enable_com_aware_ik=enable_com_aware_ik)


@needs_cuda
def test_flag_off_seed_cfg_weight_zero_by_default():
    world = _world(enable_com_aware_ik=False)
    cfg = world.ik_solver.seed_ik_solver.config
    assert cfg.com_support_weight == 0.0


@needs_cuda
def test_flag_on_seed_cfg_carries_com_params():
    from cutamp.com_polygon_cost import COM_HALF_EXTENTS, COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT
    from cutamp.robots.t1 import T1_COM_IK_WEIGHT
    world = _world(enable_com_aware_ik=True)
    cfg = world.ik_solver.seed_ik_solver.config
    assert cfg.com_support_weight == T1_COM_IK_WEIGHT > 0.0
    assert list(cfg.com_half_extents) == list(COM_HALF_EXTENTS)
    assert cfg.com_inside_margin == COM_INSIDE_MARGIN
    assert cfg.com_inside_weight == COM_INSIDE_WEIGHT
    assert cfg.com_center_weight == 0.0
    assert cfg.com_base_link_name == "mobile_base_link"
```

- [ ] **Step 2: Run to verify failure** — `... -m pytest cutamp/tests/test_com_aware_seed_ik.py -q` → FAIL (`TAMPWorld.__init__() got an unexpected keyword argument 'enable_com_aware_ik'`).

- [ ] **Step 3: Fork — `seed_ik_solver_cfg.py`**: change the typing import to `from typing import Dict, List, Optional, Union` and add after the `acceleration_weight` field+docstring (~line 94, before `@staticmethod def create`):

```python
    # --- cuTAMP fork: CoM-over-support-rectangle residual (rhs-only) -------
    # See cuTAMP docs/superpowers/specs/2026-06-09-com-aware-ik-design.md.
    # Default 0.0 keeps the solver byte-identical to upstream.
    com_support_weight: float = 0.0
    """Weight of the CoM-over-support-rectangle residual; 0 disables (default)."""
    com_half_extents: Optional[List[float]] = None
    """Support rectangle half-extents (x, y), meters, in com_base_link_name frame."""
    com_inside_margin: float = 0.02
    """Inside-edge soft-barrier band width, meters."""
    com_inside_weight: float = 1.0
    """Inside-barrier scale vs the outside quadratic."""
    com_center_weight: float = 0.0
    """Optional pull-to-center term scale; 0 disables."""
    com_base_link_name: str = "mobile_base_link"
    """Tool frame whose pose defines the rectangle's local frame (must be in tool_frames)."""
```

- [ ] **Step 4: Fork — `solver_ik_cfg.py`** (mirror the `seed_position_weight` pattern exactly): add fields after `seed_solver_num_seeds` (~line 93):

```python
    # --- cuTAMP fork: pass-through for the seed-IK CoM residual ------------
    seed_com_support_weight: float = 0.0
    seed_com_half_extents: Optional[List[float]] = None
    seed_com_inside_margin: float = 0.02
    seed_com_inside_weight: float = 1.0
    seed_com_center_weight: float = 0.0
    seed_com_base_link_name: str = "mobile_base_link"
```
(ensure `Optional, List` are imported in this file's typing import). Add the same six as keyword params of `create()` after `seed_solver_num_seeds: int = 32` (~line 168), and forward them in the `cls(...)` construction after `seed_solver_num_seeds=seed_solver_num_seeds` (~line 284):

```python
        seed_com_support_weight: float = 0.0,
        seed_com_half_extents: Optional[List[float]] = None,
        seed_com_inside_margin: float = 0.02,
        seed_com_inside_weight: float = 1.0,
        seed_com_center_weight: float = 0.0,
        seed_com_base_link_name: str = "mobile_base_link",
```
```python
            seed_com_support_weight=seed_com_support_weight,
            seed_com_half_extents=seed_com_half_extents,
            seed_com_inside_margin=seed_com_inside_margin,
            seed_com_inside_weight=seed_com_inside_weight,
            seed_com_center_weight=seed_com_center_weight,
            seed_com_base_link_name=seed_com_base_link_name,
```

- [ ] **Step 5: Fork — `solver_ik.py:160-164`**: inside the `SeedIKSolverCfg.create(...)` call, after `acceleration_weight=self.config.seed_acceleration_weight,` add:

```python
                    com_support_weight=self.config.seed_com_support_weight,
                    com_half_extents=self.config.seed_com_half_extents,
                    com_inside_margin=self.config.seed_com_inside_margin,
                    com_inside_weight=self.config.seed_com_inside_weight,
                    com_center_weight=self.config.seed_com_center_weight,
                    com_base_link_name=self.config.seed_com_base_link_name,
```
(`SeedIKSolverCfg.create` takes `**kwargs` and `config_dict.update(kwargs)` — no change needed there.)

- [ ] **Step 6: cuTAMP — `cutamp/robots/t1.py`**: add a module constant near the other COM imports/constants (top of the factory section, after `T1_CONFIG_PATH`):

```python
# Starting weight for the seed-IK CoM residual (tuned by the Task-8 sweep;
# pose/joint-limit weights in that solver are O(1) — the planner's 5e5 rollout
# weight is NOT transferable here).
T1_COM_IK_WEIGHT = 1.0
```
In `get_t1_ik_solver` (def at line 272): add kwarg `enable_com_aware_ik: bool = False,` after `enable_com_polygon: bool = True,` (line 278) and extend the `InverseKinematicsCfg.create(...)` call (after `transition_model=...`):

```python
        # CoM-aware seed-IK residual (fork): pulls the CoM toward the support
        # rectangle while the LM solver reaches the hand pose. Off by default.
        seed_com_support_weight=(T1_COM_IK_WEIGHT if enable_com_aware_ik else 0.0),
        seed_com_half_extents=(list(COM_HALF_EXTENTS) if enable_com_aware_ik else None),
        seed_com_inside_margin=COM_INSIDE_MARGIN,
        seed_com_inside_weight=COM_INSIDE_WEIGHT,
        seed_com_center_weight=0.0,
        seed_com_base_link_name="mobile_base_link",
```
The `COM_HALF_EXTENTS, COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT` imports already exist inside the `enable_com_polygon` block — move that import line to the top of `get_t1_ik_solver` (unconditional) so both blocks can use it.

- [ ] **Step 7: cuTAMP threading**:
  - `cutamp/config.py` after line 84 (`ik_com_retry_max`):
    ```python
    # CoM-aware seed-IK residual (cuRobo fork): IK natively trades hand-pose vs
    # COM, recruiting legs within limits. Default off until validated (spec
    # 2026-06-09-com-aware-ik-design.md acceptance gates).
    enable_com_aware_ik: bool = False
    ```
  - `cutamp/tamp_world.py`: `__init__` gains `enable_com_aware_ik: bool = False,` after `enable_com_polygon` (line 65); pass `enable_com_aware_ik=enable_com_aware_ik,` in the `get_t1_ik_solver(...)` call (line ~101).
  - `cutamp/algorithm.py:374` area: alongside `enable_com_polygon=config.enable_com_polygon,` add `enable_com_aware_ik=config.enable_com_aware_ik,` to the `TAMPWorld(...)` construction.
  - `cutamp/scripts/run_cutamp.py`: after the `--no_enable_com_polygon` block (~line 186):
    ```python
    parser.add_argument(
        "--enable_com_aware_ik",
        action="store_true",
        help="Enable the CoM-aware seed-IK residual (cuRobo fork): IK trades "
             "hand-pose vs COM and recruits the legs to center the COM. "
             "Default off until validated.",
    )
    ```
    and `enable_com_aware_ik=args.enable_com_aware_ik,` in the `TAMPConfiguration(...)` construction (next to `enable_com_polygon=...`, ~line 247).

- [ ] **Step 8: Run to verify pass** — `... -m pytest cutamp/tests/test_com_aware_seed_ik.py -q` → 2 passed. Also import smoke: `python -c "import cutamp.robots.t1, cutamp.tamp_world, cutamp.config"`.

- [ ] **Step 9: Commit (cuTAMP files only — curobo/ stays uncommitted)**
```bash
git add cutamp/robots/t1.py cutamp/config.py cutamp/tamp_world.py cutamp/algorithm.py \
        cutamp/scripts/run_cutamp.py cutamp/tests/test_com_aware_seed_ik.py
git commit -m "feat: enable_com_aware_ik flag threaded to seed-IK cfg (fork plumbing; off by default)"
```

---

### Task 3: fork — `compute_com` in seed-IK kinematics + construction guards

Seed-IK builds its own `Kinematics` with `compute_com` off — its `robot_com` is allocated but all zeros today (verified by probe). Enable it when the residual is on; fail fast on bad configs.

**Files:**
- Modify (fork): `curobo/curobo/_src/solver/seed_ik/seed_ik_solver.py:81-87`
- Test: `cutamp/tests/test_com_aware_seed_ik.py`

- [ ] **Step 1: Write the failing test** — append:

```python
@needs_cuda
def test_flag_on_seed_ik_robot_com_is_real():
    from curobo.types import JointState
    world = _world(enable_com_aware_ik=True)
    model = world.ik_solver.seed_ik_solver._robot_model
    assert model.compute_com is True
    js = JointState.from_position(
        torch.zeros(1, model.get_dof(), device=model.device_cfg.device),
        joint_names=list(model.joint_names),
    )
    ks = model.compute_kinematics(js)
    assert float(ks.robot_com.abs().max()) > 0.01, "robot_com must be populated, not zeros"


@needs_cuda
def test_flag_off_seed_ik_robot_com_stays_off():
    world = _world(enable_com_aware_ik=False)
    assert world.ik_solver.seed_ik_solver._robot_model.compute_com is False
```

- [ ] **Step 2: Run to verify failure** — `-k robot_com` → `test_flag_on_seed_ik_robot_com_is_real` FAILS (`compute_com is False`).

- [ ] **Step 3: Implement** — in `seed_ik_solver.py` replace the `_robot_model` construction (~line 82):

```python
        # cuTAMP fork: the CoM residual reads robot_com from this model's FK;
        # compute_com is a construction-time kernel-template flag (default off
        # upstream — robot_com would be allocated but zeros).
        self._robot_model = Kinematics(
            robot_model_config,
            compute_spheres=False,
            compute_jacobian=True,
            compute_com=config.com_support_weight > 0.0,
        )
```
(`_aux_robot_model` unchanged.) Immediately after the `self._robot_model`/`_aux_robot_model` block, add the guard:

```python
        # cuTAMP fork: the CoM residual derives jTerror from the autograd
        # backward path; the analytical path does not include it.
        if config.com_support_weight > 0.0 and not config.use_backward:
            log_and_raise("com_support_weight > 0 requires use_backward=True")
```
(check `log_and_raise` is imported in this file; if not, add `from curobo._src.util.logging import log_and_raise`).

- [ ] **Step 4: Run to verify pass** — `-k robot_com` → 2 passed.

- [ ] **Step 5: Commit (test only)**
```bash
git add cutamp/tests/test_com_aware_seed_ik.py
git commit -m "test: seed-IK robot_com populated iff CoM-aware IK enabled (fork: compute_com flag)"
```

---

### Task 4: fork — the CoM residual block

Module-level `com_support_penalty` (unit-testable, parity with cuTAMP's penalty) + integration into `_compute_pose_errors` via a single combined backward; scalar added to `error_norm`. No jacobian rows, no `_combine_errors` change, no `_calculate_n_residuals` change.

**Files:**
- Modify (fork): `curobo/curobo/_src/solver/seed_ik/seed_ik_error_calculator.py` (`__init__` ~line 53, `setup_batch_tensors` ~line 89, `_compute_pose_errors` ~lines 265-288, new module function + method)
- Test: `cutamp/tests/test_com_aware_seed_ik.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
def test_fork_penalty_parity_with_cutamp_reference():
    # The fork's com_support_penalty must agree with cuTAMP's com_polygon_penalty
    # (single source of truth enforced by test, since curobo cannot import cutamp).
    # Random rotated/translated base frames catch quaternion-projection sign errors.
    import roma
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import com_support_penalty
    from cutamp.com_polygon_cost import com_polygon_penalty
    torch.manual_seed(0)
    B = 256
    com_world = torch.randn(B, 3) * 0.3
    base_pos = torch.randn(B, 3) * 0.2
    quat_xyzw = roma.random_unitquat(B)
    quat_wxyz = torch.cat([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], dim=-1)
    R = roma.unitquat_to_rotmat(quat_xyzw)
    base_T = torch.eye(4).repeat(B, 1, 1)
    base_T[:, :3, :3] = R
    base_T[:, :3, 3] = base_pos
    half = torch.tensor([0.1115, 0.156])
    for cw in (0.0, 0.01):
        ref = com_polygon_penalty(com_world, base_T, half, 0.02, 1.0, center_weight=cw)
        got = com_support_penalty(com_world, base_pos, quat_wxyz, half, 0.02, 1.0, cw)
        assert torch.allclose(got, ref, atol=1e-5), f"max diff {(got-ref).abs().max()}"


def test_fork_penalty_jterror_matches_autograd_reference():
    # jTerror = d(weight*penalty)/d(inputs) via autograd must be finite and
    # nonzero for an outside-rectangle CoM (the LM rhs the residual contributes).
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import com_support_penalty
    com = torch.tensor([[0.15, 0.0, 0.3]], requires_grad=True)  # outside +x edge
    half = torch.tensor([0.1115, 0.156])
    pos = torch.zeros(1, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pen = com_support_penalty(com, pos, quat, half, 0.02, 1.0, 0.0)
    pen.sum().backward()
    assert com.grad is not None and torch.isfinite(com.grad).all()
    assert float(com.grad.abs().max()) > 0.0


@needs_cuda
def test_residual_fires_iff_enabled():
    from curobo._src.solver.seed_ik.seed_ik_error_calculator import SeedIKErrorCalculator
    from curobo.types import JointState
    from cutamp.particle_initialization import _ik_for_pose
    calls = {"n": 0}
    orig = SeedIKErrorCalculator._compute_com_penalty
    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)
    SeedIKErrorCalculator._compute_com_penalty = counting
    try:
        for flag, expect_calls in ((False, False), (True, True)):
            calls["n"] = 0
            world = _world(enable_com_aware_ik=flag)
            names = list(world.kinematics.joint_names)
            qh = world.q_init
            af = world.tool_frame_for_arm("left")
            hm = world.kinematics.compute_kinematics(
                world.kinematics.get_active_js(
                    JointState.from_position(qh.unsqueeze(0), joint_names=names))
            ).tool_poses.get_link_pose(af, make_contiguous=True).get_matrix().reshape(4, 4)
            T = hm.unsqueeze(0).repeat(4, 1, 1).clone()
            T[:, 0, 3] += 0.2
            _ik_for_pose(world, T, "left")
            if expect_calls:
                assert calls["n"] > 0, "residual must fire when enabled"
            else:
                assert calls["n"] == 0, "residual must not fire when disabled"
            del world
            import gc; gc.collect(); torch.cuda.empty_cache()
    finally:
        SeedIKErrorCalculator._compute_com_penalty = orig
```

- [ ] **Step 2: Run to verify failure** — `-k "parity or jterror or fires"` → ImportError (`com_support_penalty` not defined).

- [ ] **Step 3: Implement the module function** — in `seed_ik_error_calculator.py`, above `class ErrorJacobianResult`:

```python
def com_support_penalty(
    com_world: torch.Tensor,
    base_position: torch.Tensor,
    base_quaternion: torch.Tensor,
    half_extents: torch.Tensor,
    inside_margin: float,
    inside_weight: float,
    center_weight: float,
) -> torch.Tensor:
    """CoM-over-support-rectangle penalty, [B].

    cuTAMP fork (spec 2026-06-09-com-aware-ik-design.md). Same shape as
    cuTAMP's com_polygon_penalty — parity enforced by a cuTAMP-side test,
    since curobo cannot import cutamp: outside-quadratic + inside-margin
    max-over-axes barrier + optional pull-to-center. Branch-free tensor
    math except host-constant scalars (CUDA-graph capture safe).

    Args:
        com_world: [B, 3] world-frame robot CoM.
        base_position: [B, 3] world position of the rectangle's base link.
        base_quaternion: [B, 4] wxyz world orientation of the base link.
        half_extents: [2] rectangle half-sizes (x, y) in the base frame.
    """
    v = com_world - base_position
    w = base_quaternion[:, 0:1]
    u = base_quaternion[:, 1:4]
    # Rotate v by the conjugate quaternion (world -> base frame):
    # R(q)^T v = v + 2*(u x (u x v) - w*(u x v)).
    t = torch.cross(u, v, dim=-1)
    com_in_base = (v + 2.0 * (torch.cross(u, t, dim=-1) - w * t))[:, :2]
    offset = com_in_base.abs() - half_extents
    outside = torch.clamp(offset, min=0.0)
    inside = torch.clamp(offset + inside_margin, min=0.0)
    penalty = (outside ** 2).sum(dim=-1) + inside_weight * (inside ** 2).max(dim=-1).values
    if center_weight > 0.0:
        penalty = penalty + center_weight * ((com_in_base / half_extents) ** 2).sum(dim=-1)
    return penalty
```

- [ ] **Step 4: Wire into the calculator** —
  (a) In `__init__` (after `self.pose_cost = self._setup_cost_function()`):

```python
        # cuTAMP fork: CoM-over-support-rectangle residual (rhs-only).
        self._com_enabled = getattr(config, "com_support_weight", 0.0) > 0.0
        self._com_ones = None
        if self._com_enabled:
            if config.com_half_extents is None:
                log_and_raise("com_support_weight > 0 requires com_half_extents")
            if config.com_base_link_name not in robot_model.tool_frames:
                log_and_raise(
                    f"com_base_link_name {config.com_base_link_name!r} not in "
                    f"tool_frames {list(robot_model.tool_frames)}"
                )
            self._com_base_idx = list(robot_model.tool_frames).index(
                config.com_base_link_name
            )
            self._com_half = torch.tensor(
                config.com_half_extents, dtype=torch.float32,
                device=device_cfg.device,
            )
```
  (b) In `setup_batch_tensors`, inside the `if batch_size != ...` block after `self._cost_shape = ...`:

```python
            if self._com_enabled:
                self._com_ones = torch.ones(
                    (self._num_problems,), dtype=torch.float32,
                    device=self.device_cfg.device,
                )
```
  (c) New method (after `_compute_pose_errors`):

```python
    def _compute_com_penalty(self, kin_state, batch_size: int) -> torch.Tensor:
        """Weighted CoM residual scalar per problem, [batch_size] (cuTAMP fork)."""
        cfg = self.config
        com_world = kin_state.robot_com.view(batch_size, -1)[:, :3]
        positions = kin_state.tool_poses.position.view(batch_size, self.num_links, 3)
        quaternions = kin_state.tool_poses.quaternion.view(batch_size, self.num_links, 4)
        penalty = com_support_penalty(
            com_world,
            positions[:, self._com_base_idx, :],
            quaternions[:, self._com_base_idx, :],
            self._com_half,
            cfg.com_inside_margin,
            cfg.com_inside_weight,
            cfg.com_center_weight,
        )
        return cfg.com_support_weight * penalty
```
  (d) In `_compute_pose_errors`: replace the single line `cost.backward(cost_shape)` (~line 272) with:

```python
        # Trigger backward pass. cuTAMP fork: when the CoM residual is enabled,
        # one COMBINED backward over [pose, CoM] populates joint_position.grad
        # with both contributions (a second separate backward would hit a freed
        # graph / alias the fused-kinematics grad buffers).
        if self._com_enabled:
            com_cost = self._compute_com_penalty(kin_state, num_problems)
            torch.autograd.backward([cost, com_cost], [cost_shape, self._com_ones])
        else:
            cost.backward(cost_shape)
```
  and after the `position_errors, orientation_errors, error_norm = self._reduce_pose_errors(...)` call add:

```python
        if self._com_enabled:
            # error_norm feeds LM trust-ratio/acceptance; success stays pose-only.
            error_norm = error_norm + com_cost.detach()
```

- [ ] **Step 5: Run to verify pass** — `... -m pytest cutamp/tests/test_com_aware_seed_ik.py -q` → all pass (parity, jterror, fires-iff-enabled, plus Tasks 2-3 tests).

- [ ] **Step 6: Run the existing suite (regression, GPU clear)** — `... -m pytest cutamp/tests/ -q` → 55+ passed (flag defaults off everywhere → byte-identical).

- [ ] **Step 7: Commit (tests only; fork stays uncommitted)**
```bash
git add cutamp/tests/test_com_aware_seed_ik.py
git commit -m "test: seed-IK CoM residual parity + gating (fork: rhs-only residual in pose block)"
```

---

### Task 5: integration — centering A/B, leg limits, pose tolerance

**Files:**
- Test: `cutamp/tests/test_com_aware_seed_ik.py`

- [ ] **Step 1: Write the tests** — append:

```python
def _forward_reach_solutions(world, n_fwd=4, n_lat=3):
    """IK a forward-reach grid for the left arm; return (q_full[B,21], success[B], result)."""
    from curobo.types import JointState
    from cutamp.particle_initialization import _ik_for_pose, _ik_solution_to_full_q
    names = list(world.kinematics.joint_names)
    qh = world.q_init
    af = world.tool_frame_for_arm("left")
    hm = world.kinematics.compute_kinematics(
        world.kinematics.get_active_js(
            JointState.from_position(qh.unsqueeze(0), joint_names=names))
    ).tool_poses.get_link_pose(af, make_contiguous=True).get_matrix().reshape(4, 4)
    Ts = []
    for f in torch.linspace(0.15, 0.33, n_fwd, device=qh.device):
        for l in torch.linspace(-0.05, 0.15, n_lat, device=qh.device):
            m = hm.clone(); m[0, 3] += float(f); m[1, 3] = float(l); Ts.append(m)
    res = _ik_for_pose(world, torch.stack(Ts, 0), "left")
    s = res.success
    succ = (s.reshape(s.shape[0], -1)[:, 0] if s.ndim > 1 else s).bool().reshape(-1)
    return _ik_solution_to_full_q(res, world), succ, res


def _com_abs_x(world, q_full):
    from curobo.types import JointState
    kin = world.kinematics_with_com
    names = list(world.kinematics.joint_names)
    ks = kin.compute_kinematics(kin.get_active_js(
        JointState.from_position(q_full.to(kin.device_cfg.device), joint_names=names)))
    cw = ks.robot_com[..., :3].reshape(-1, 3)
    bT = ks.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True) \
        .get_matrix().reshape(-1, 4, 4)
    cib = (bT[:, :3, :3].transpose(-1, -2) @ (cw - bT[:, :3, 3]).unsqueeze(-1)).squeeze(-1)
    return cib[:, 0].abs()


@needs_cuda
def test_centering_ab_legs_within_limits_pose_preserved():
    import gc
    from cutamp.com_polygon_cost import COM_HALF_EXTENTS, COM_INSIDE_MARGIN
    # OFF baseline
    world_off = _world(enable_com_aware_ik=False)
    q_off, s_off, _ = _forward_reach_solutions(world_off)
    absx_off = _com_abs_x(world_off, q_off)[s_off]
    del world_off; gc.collect(); torch.cuda.empty_cache()
    # ON
    world_on = _world(enable_com_aware_ik=True)
    q_on, s_on, res_on = _forward_reach_solutions(world_on)
    assert int(s_on.sum()) > 0, "CoM-aware IK produced no successful solutions"
    absx_on = _com_abs_x(world_on, q_on)[s_on]

    # (1) centering: mean |com_x| reduced >=10% OR already inside the inset rect
    inset = COM_HALF_EXTENTS[0] - COM_INSIDE_MARGIN  # 0.0915
    assert (float(absx_on.mean()) < 0.9 * float(absx_off.mean())
            or float(absx_on.mean()) <= inset), \
        f"on={float(absx_on.mean()):.4f} off={float(absx_off.mean()):.4f}"

    # (2) joint limits: EVERY returned solution within URDF bounds (+1e-3 slack;
    #     limits are soft residual + filter — out-of-limit output is a fork bug)
    names = list(world_on.kinematics.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    bounds = {"ankle_pitch": (-0.87, 0.0), "knee_pitch": (0.0, 2.34),
              "Torso_Pitch": (-1.8, 0.0)}
    for jn, (lo, hi) in bounds.items():
        col = q_on[:, idx[jn]]
        assert float(col.min()) >= lo - 1e-3 and float(col.max()) <= hi + 1e-3, \
            f"{jn} out of [{lo},{hi}]: [{float(col.min()):.3f},{float(col.max()):.3f}]"

    # (3) pose accuracy on successes: solver tolerances (5mm / 0.05rad)
    pe = res_on.position_error
    pe = (pe.reshape(pe.shape[0], -1)[:, 0] if pe.ndim > 1 else pe)[s_on]
    assert float(pe.max()) <= 0.005 + 1e-4, f"pose err {float(pe.max()):.4f}"

    # (4) soft recruitment signal (informational; generous threshold)
    knee_delta = float((q_on[s_on][:, idx["knee_pitch"]]).abs().max())
    print(f"recruitment: max knee_pitch={knee_delta:.3f} rad; "
          f"absx on/off mean {float(absx_on.mean()):.4f}/{float(absx_off.mean()):.4f}")
    del world_on; gc.collect(); torch.cuda.empty_cache()
```

- [ ] **Step 2: Run** — `-k centering -q -s`; expected: PASS with the printed recruitment line. If assertion (1) fails: the weight is too low/high — proceed to Task 8's sweep FIRST, then re-run (the test must pass with the final `T1_COM_IK_WEIGHT`); if (2) fails: fork bug — debug before proceeding (do NOT loosen the bound).

- [ ] **Step 3: Run the full new file + suite** — `cutamp/tests/test_com_aware_seed_ik.py -q` then `cutamp/tests/ -q` → all green.

- [ ] **Step 4: Commit**
```bash
git add cutamp/tests/test_com_aware_seed_ik.py
git commit -m "test: CoM-aware IK centering A/B with joint-limit and pose-accuracy guards"
```

---

### Task 6: hoist the GPU-memory fixture to conftest

**Files:**
- Modify: `cutamp/tests/conftest.py`, `cutamp/tests/test_motion_anchor.py:61-75`

- [ ] **Step 1: Add to `cutamp/tests/conftest.py`** (end of file):

```python
@pytest.fixture(autouse=True)
def _free_cuda_between_tests():
    """Release GPU memory after each test. Integration tests build full
    TAMPWorld stacks (several GiB each); without reclaiming between tests,
    two heavy tests in one process OOM a 24 GiB card. Harmless for CPU tests."""
    yield
    import gc
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
```

- [ ] **Step 2: Delete the now-duplicate fixture** from `cutamp/tests/test_motion_anchor.py` (lines 61-75, the `_free_cuda_between_tests` autouse fixture; keep its `import gc`/`os` only if still used elsewhere in the file).

- [ ] **Step 3: Run** — `cutamp/tests/test_motion_anchor.py cutamp/tests/test_com_aware_seed_ik.py -q` → all pass (no OOM).

- [ ] **Step 4: Commit**
```bash
git add cutamp/tests/conftest.py cutamp/tests/test_motion_anchor.py
git commit -m "test: hoist _free_cuda_between_tests fixture to conftest"
```

---

### Task 7: close the success-mask hole + remove the dormant `num_seeds` kwarg

Review finding: `_store_ik_q` stores IK solutions without success-masking → `success=False` confs (possibly out-of-limit; nothing clamps) silently enter particles. Separately, `_ik_for_pose`'s `num_seeds` kwarg forwards to `solve_pose`, which has no such parameter (`solver_ik.py:631-638`) → latent TypeError.

**Files:**
- Modify: `cutamp/particle_initialization.py` (`_ik_for_pose` ~lines 60-115, `_store_ik_q` ~line 242)
- Test: `cutamp/tests/test_com_aware_seed_ik.py`

- [ ] **Step 1: Write the failing test** — append:

```python
@needs_cuda
def test_store_ik_q_masks_failed_solutions_to_home():
    from cutamp.particle_initialization import _store_ik_q
    world = _world(enable_com_aware_ik=False)
    q_full, succ, res = _forward_reach_solutions(world, n_fwd=2, n_lat=2)
    # Forge a failure on problem 0 and store.
    if res.success.ndim > 1:
        res.success[0, :] = False
    else:
        res.success[0] = False
    particles = {}
    _store_ik_q(particles, "left_q1", res, world, "left")
    stored = particles["left_q1"]
    assert torch.allclose(stored[0], world.q_init), \
        "failed-IK row must fall back to q_init, not keep an unvetted conf"


def test_ik_for_pose_has_no_dormant_num_seeds_kwarg():
    import inspect
    from cutamp.particle_initialization import _ik_for_pose
    assert "num_seeds" not in inspect.signature(_ik_for_pose).parameters, \
        "num_seeds forwarded to solve_pose would TypeError (no such param)"
```

- [ ] **Step 2: Run to verify failure** — `-k "store_ik_q or dormant"` → both FAIL.

- [ ] **Step 3: Implement** —
  (a) `_store_ik_q` becomes:

```python
def _store_ik_q(particles: dict, q_name: str, ik_result, world: TAMPWorld, arm: Optional[str]) -> None:
    """Store the IK result's q under ``q_name``, reordered to the full kin's joint order.

    Rows whose IK reported failure fall back to ``world.q_init``: cuRobo's joint
    limits are a soft residual + success filter (nothing clamps), so an
    unsuccessful solution may be out-of-limit and must not enter particles.
    """
    q_full = _ik_solution_to_full_q(ik_result, world)
    s = ik_result.success
    success = (s.reshape(s.shape[0], -1)[:, 0] if s.ndim > 1 else s).bool().reshape(-1)
    home = world.q_init.to(q_full.device).unsqueeze(0).expand_as(q_full)
    particles[q_name] = torch.where(success.to(q_full.device).unsqueeze(-1), q_full, home)
```
  (b) In `_ik_for_pose`: delete the `num_seeds: Optional[int] = None,` parameter, the docstring paragraph about it, and the `solve_kwargs` indirection — the call becomes:

```python
        return ik.solve_pose(goal_tool_poses=goal, current_state=current_state)
```
  (c) Guard: `grep -rn "num_seeds" cutamp/ --include=*.py | grep _ik_for_pose` → must return nothing (no caller passes it; if one appears, fix that call site).

- [ ] **Step 4: Run to verify pass** — `-k "store_ik_q or dormant"` → 2 passed; then full suite `cutamp/tests/ -q` → green (this changes init behavior only for failed-IK rows, which the downstream gate rejected anyway).

- [ ] **Step 5: Commit**
```bash
git add cutamp/particle_initialization.py cutamp/tests/test_com_aware_seed_ik.py
git commit -m "fix: success-mask IK confs before storing; drop dormant num_seeds kwarg"
```

---

### Task 8: weight sweep → set `T1_COM_IK_WEIGHT`

Manual validation step (nondeterministic pipeline → compare means, not single runs). The sweep script instruments fallback engagement (the CoM-blind MPPI→LBFGS stage runs iff the seed stage misses 100% batch success).

**Files:**
- Create: `/tmp/com_ik_weight_sweep.py` (throwaway — do not commit)
- Modify: `cutamp/robots/t1.py` (`T1_COM_IK_WEIGHT` final value)

- [ ] **Step 1: Write the sweep script** `/tmp/com_ik_weight_sweep.py`:

```python
"""Sweep seed-IK com_support_weight; report centering / success / fallback metrics.
Run: PYTORCH_ALLOC_CONF=expandable_segments:True python /tmp/com_ik_weight_sweep.py"""
import gc, json, os, torch
import cutamp.robots.t1 as t1mod

RESULTS = []
for w in [0.0, 0.1, 0.3, 1.0, 3.0]:
    t1mod.T1_COM_IK_WEIGHT = w  # module constant read at IK construction
    torch.manual_seed(0)
    import importlib
    from cutamp.tests.test_com_aware_seed_ik import _world, _forward_reach_solutions, _com_abs_x
    world = _world(enable_com_aware_ik=(w > 0))
    # count fallback engagements: each engagement calls core.optimizer.optimize once
    opt = world.ik_solver.optimizer
    n_opt = {"n": 0}
    orig = opt.optimize
    opt.optimize = lambda *a, **k: (n_opt.__setitem__("n", n_opt["n"] + 1), orig(*a, **k))[1]
    q, s, res = _forward_reach_solutions(world, n_fwd=4, n_lat=3)
    absx = _com_abs_x(world, q)[s]
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    in_hull = compute_com_polygon_mask(world, q)[s]
    pe = res.position_error
    pe = (pe.reshape(pe.shape[0], -1)[:, 0] if pe.ndim > 1 else pe)[s]
    row = dict(weight=w, n_success=int(s.sum()), n_targets=int(s.numel()),
               absx_mean=round(float(absx.mean()), 4) if s.any() else None,
               absx_max=round(float(absx.max()), 4) if s.any() else None,
               in_hull=int(in_hull.sum()), fallback_engagements=n_opt["n"],
               pose_err_max=round(float(pe.max()), 5) if s.any() else None)
    print("ROW", json.dumps(row)); RESULTS.append(row)
    del world; gc.collect(); torch.cuda.empty_cache()
print("\nSWEEP", json.dumps(RESULTS, indent=1))
```

- [ ] **Step 2: Run it** and record the table in the task notes. **Selection rule (spec):** the largest weight with (a) `n_success` not below the `weight=0.0` row, (b) `fallback_engagements` not above it, (c) `pose_err_max ≤ 0.005`, (d) best `absx_mean`. If 1.0 satisfies all, keep 1.0.

- [ ] **Step 3: Set the chosen value** in `cutamp/robots/t1.py` (`T1_COM_IK_WEIGHT = <chosen>`), update its comment with the sweep date/result one-liner.

- [ ] **Step 4: Re-run Task 5's centering test** with the final weight → PASS required.

- [ ] **Step 5: Commit**
```bash
git add cutamp/robots/t1.py
git commit -m "feat: T1_COM_IK_WEIGHT=<chosen> from sweep (centering vs success/fallback gates)"
```

---

### Task 9: acceptance gates → flip the default ON (conditional)

**Files:**
- Modify: `cutamp/config.py` (`enable_com_aware_ik` default), `cutamp/scripts/run_cutamp.py` (flag inverted to `--no_enable_com_aware_ik` if flipped)

- [ ] **Step 1: Gates (all must pass; record numbers):**
  1. Full suite green: `... -m pytest cutamp/tests/ -q`.
  2. Pipeline A/B, 3 runs each (nondeterministic — compare means):
     ```bash
     for i in 1 2 3; do PYTORCH_ALLOC_CONF=expandable_segments:True \
       /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
       --env blocks_t1 -n 64 --num_opt_steps 50 [--enable_com_aware_ik] \
       2>&1 | grep -oE "Total num satisfying [0-9]+"; done
     ```
     Gate: mean(on) ≥ mean(off) − 3.
  3. Endpoint audit: one `--enable_com_aware_ik --motion_plan --save_plan /tmp/mp_comik.pkl` run; reconstruct every segment arrival (pattern in `cutamp/tests/test_motion_anchor.py::test_place_arrival_is_com_in_hull`) — gate: all arrivals in-hull AND worst arrival penalty < the recorded baseline range (~3.9e-4 ≈ 99% tol; expect a clear drop for recruitable confs).

- [ ] **Step 2 (only if ALL gates pass): flip the default** — `config.py`: `enable_com_aware_ik: bool = True` (+ comment "validated <date>, see plan Task 9 numbers"); `run_cutamp.py`: replace `--enable_com_aware_ik` (store_true) with `--no_enable_com_aware_ik` (`dest="enable_com_aware_ik", action="store_false"` + `parser.set_defaults(enable_com_aware_ik=True)`), mirroring `--no_enable_com_polygon`. Re-run the full suite.
  **If any gate fails:** leave the default False, document the failing numbers in the plan file, and STOP — escalation options live in the spec (lower weight / `run_optimizer=False` / LBFGS-stage cost).

- [ ] **Step 3: Commit**
```bash
git add cutamp/config.py cutamp/scripts/run_cutamp.py
git commit -m "feat: enable CoM-aware IK by default (acceptance gates: <numbers>)"
```

---

### Task 10: cleanup (per the reviewed redundancy map)

Only after Task 9's gates have been evaluated.

**Files:**
- Modify: `cutamp/robots/t1.py`, `cutamp/tamp_world.py`, `cutamp/config.py`, `cutamp/particle_initialization.py`, `cutamp/_curobo_internals.py`, `cutamp/scripts/run_cutamp.py`, `cutamp/tests/test_com_polygon_ik.py`

- [ ] **Step 1: Delete the inert IK soft-cost registration** — `cutamp/robots/t1.py`: remove the whole `if enable_com_polygon:` block in `get_t1_ik_solver` (lines ~319-343: imports, comment, `cost_cfg`, `add_extra_cost(ik_solver, ...)`), the now-unused `enable_com_polygon` parameter (line 278), and the docstring paragraph claiming the IK "gets the same COM-over-base-polygon soft cost" (~lines 294-300) — replace with one line: `When ``enable_com_aware_ik`` is True, the seed-IK LM solver carries a CoM-over-support-rectangle residual (cuRobo fork) that recruits the legs to keep the COM centered.`

- [ ] **Step 2: Update callers** — `cutamp/tamp_world.py:101`: drop `enable_com_polygon=enable_com_polygon,` from the `get_t1_ik_solver` call (keep the `TAMPWorld` param itself — the planner cost + hard gate still use it).

- [ ] **Step 3: Replace the two dead registration tests** — in `cutamp/tests/test_com_polygon_ik.py` delete `test_ik_solver_has_com_polygon_extra_cost_when_enabled`, `test_ik_solver_no_com_polygon_when_disabled`, and the `_ik_extra_costs` helper (their coverage moved to `test_com_aware_seed_ik.py::test_flag_*`); update the module docstring (line 1) from "two-layer COM cost on the IK solver" to "COM polygon penalty math, hard gate, and post-IK mask".

- [ ] **Step 4: Shrink the retry budget** — `cutamp/config.py`: `ik_com_retry_max: int = 3` with comment `# Backstop: post-IK COM mask retries (CoM-aware IK makes violations rare).` Only if Task 9 flipped the default ON; otherwise leave 15.

- [ ] **Step 5: Doc sweep** (comment-only; verify each with grep before editing):
  - `cutamp/particle_initialization.py` `_ik_for_pose_com_safe` docstring: "Layer 1 (cost registered on IK rollouts)" → "Layer 1 (the seed-IK CoM residual, when enable_com_aware_ik is on)".
  - `cutamp/robots/t1.py` `_ik_transition_dict_with_compute_com` docstring: correct the claim that it feeds the seed-IK FK — it does NOT (seed-IK builds its own Kinematics; the fork's `compute_com` flag covers that); this dict serves the rollout/metrics kinematics.
  - `cutamp/_curobo_internals.py` `add_extra_cost` docstring: note the IKSolver path is inert for solve-time costs (seed-IK error calculator bypasses cost managers); planner remains the supported target.
  - `cutamp/scripts/run_cutamp.py` `--no_enable_com_polygon` help: "(1e5 weight)" → "(5e5 weight)".

- [ ] **Step 6: Full suite + skeleton sanity** — `... -m pytest cutamp/tests/ -q` → green; `python -c "import cutamp.robots.t1, cutamp.particle_initialization, cutamp._curobo_internals"`.

- [ ] **Step 7: Commit**
```bash
git add cutamp/robots/t1.py cutamp/tamp_world.py cutamp/config.py \
        cutamp/particle_initialization.py cutamp/_curobo_internals.py \
        cutamp/scripts/run_cutamp.py cutamp/tests/test_com_polygon_ik.py
git commit -m "chore: remove inert IK soft-cost registration; doc sweep for CoM-aware IK"
```

---

## Verification (end of plan)

1. Full suite: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTORCH_ALLOC_CONF=expandable_segments:True /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/ -q` → green.
2. Fork inventory: `git -C curobo diff --stat` → exactly the 5 fork files (+ the 3 pre-existing modified files); nothing staged/committed in `curobo/`.
3. End-to-end: `run_cutamp --env blocks_t1 -n 64 --num_opt_steps 50 --motion_plan --save_plan /tmp/mp_final.pkl` (+ flag per Task 9 outcome) → ≥1 satisfying plan; endpoint audit all-in-hull with improved worst-arrival margin.
