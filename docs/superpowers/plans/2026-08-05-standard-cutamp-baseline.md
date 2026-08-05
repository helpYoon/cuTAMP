# Standard-cuTAMP 2×2 Ablation Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the T1 blocks domain under a 2×2 factorial (CoM machinery × re-IK) with shared harness and one post-hoc yardstick, producing paper-grade tables and figures.

**Architecture:** All four conditions run the *same* code; a condition is a flag set. Four new config flags gate the hardwired CoM layers (L2 init gate, L3 hard constraint) and two fairness fixes (`force_adam`, `soft_in_adam`). Measure-only metrics (CoM-feasible satisfying, CoM margin, trajectory CoM audit) are recorded in **every** arm so all arms are scored identically. A runner script sweeps cells × heights × seeds into standard experiment dirs + a manifest; an analysis script turns manifests into CSV/markdown/figures.

**Tech Stack:** Python 3.10, PyTorch (CUDA), cuRobo v0.8 fork (vendored `curobo/`), pytest, matplotlib. Test env: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest` (the empty `PYTHONPATH=` is required — ROS Humble otherwise breaks pytest).

## Global Constraints

- Branch: `ablation-baseline` (already exists; spec committed as `22ece0d`).
- Every new flag defaults to **today's behavior** — the full test suite (86 tests) must stay green with defaults after every task.
- `docs/` is gitignored — use `git add -f` for anything under `docs/`.
- CUDA tests use `make_blocks_t1_world` + `needs_cuda` from `cutamp/tests/conftest.py`.
- Cell flag sets (from the spec): **full** = defaults · **−CoM** = `--no_com_all` · **−reIK** = `--no_coupled_reik --force_adam --soft_in_adam always` · **standard** = union of both.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: CoM margin helper

**Files:**
- Modify: `cutamp/com_polygon_cost.py` (append after `compute_com_polygon_mask`, ~line 190)
- Test: `cutamp/tests/test_com_margin.py` (create)

**Interfaces:**
- Produces: `com_margin_from_base(com_in_base: torch.Tensor) -> torch.Tensor` — `[N, ≥2]` base-frame CoM points → `[N]` margin in metres (min over axes of `h − |c|`; positive inside, negative outside).
- Produces: `compute_com_margin(world, q_batch: torch.Tensor) -> torch.Tensor` — `[B, full_dof]` configs → `[B]` margins via FK (CUDA).
- Consumed by: Task 6 (end-of-run metrics), Task 9 (analysis figure labels only — no import).

- [ ] **Step 1: Write the failing tests**

```python
# cutamp/tests/test_com_margin.py
"""Static stability margin: min-axis distance to the support-rectangle edge."""
import torch

from cutamp.com_polygon_cost import COM_HALF_EXTENTS, com_margin_from_base


def test_margin_center_edge_outside():
    hx, hy = COM_HALF_EXTENTS
    pts = torch.tensor([
        [0.0, 0.0],          # center: margin = min(hx, hy) = hx
        [hx, 0.0],           # on the fore edge: 0
        [0.0, hy + 0.03],    # 3 cm outside laterally: -0.03
        [hx - 0.01, hy - 0.05],  # 1 cm from fore edge, 5 cm from side: 0.01
    ])
    m = com_margin_from_base(pts)
    assert torch.allclose(m, torch.tensor([hx, 0.0, -0.03, 0.01]), atol=1e-6)


def test_margin_sign_matches_polygon_mask_semantics():
    hx, hy = COM_HALF_EXTENTS
    inside = torch.tensor([[hx - 1e-3, 0.0]])
    outside = torch.tensor([[hx + 1e-3, 0.0]])
    assert com_margin_from_base(inside).item() > 0
    assert com_margin_from_base(outside).item() < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_com_margin.py -q`
Expected: FAIL with `ImportError: cannot import name 'com_margin_from_base'`

- [ ] **Step 3: Implement**

Append to `cutamp/com_polygon_cost.py`:

```python
def com_margin_from_base(com_in_base: torch.Tensor) -> torch.Tensor:
    """Static stability margin (McGhee & Frank): min over the two axes of
    ``h - |c|`` for base-frame CoM points ``[N, >=2]``. Positive inside the
    support rectangle, zero on an edge, negative outside. Metres."""
    half = torch.tensor(
        COM_HALF_EXTENTS, device=com_in_base.device, dtype=com_in_base.dtype
    )
    return (half - com_in_base[..., :2].abs()).min(dim=-1).values


def compute_com_margin(world, q_batch: torch.Tensor) -> torch.Tensor:
    """``[B]`` stability margins for full-cspace configs via FK.

    Measure-only companion to :func:`compute_com_polygon_mask`; used by the
    ablation study's shared yardstick (recorded in every arm)."""
    com, base_T = _fk_com_and_base(world, q_batch)
    return com_margin_from_base(_project_com_to_base(com, base_T))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_com_margin.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add cutamp/com_polygon_cost.py cutamp/tests/test_com_margin.py
git commit -m "feat(ablation): CoM stability-margin helper

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Config flags, validation, CLI plumbing, and `--seed`

**Files:**
- Modify: `cutamp/config.py` (fields near the other `enable_*` flags ~line 60-90; validation in `validate_tamp_config`)
- Modify: `cutamp/scripts/run_cutamp.py` (argparse block + `TAMPConfiguration(...)` construction + seed at entrypoint)
- Test: `cutamp/tests/test_ablation_flags.py` (create)

**Interfaces:**
- Produces config fields (exact names, consumed by Tasks 3-5):
  `enable_com_hard_constraint: bool = True`, `enable_com_init_gate: bool = True`,
  `force_adam: bool = False`, `soft_in_adam: str = "auto"` (allowed: `"auto"`, `"always"`).
- Produces CLI: `--no_com_all`, `--force_adam`, `--soft_in_adam {auto,always}`, `--seed INT` (default `None` = no seeding).

- [ ] **Step 1: Write the failing tests**

```python
# cutamp/tests/test_ablation_flags.py
"""Flag plumbing for the standard-cuTAMP 2x2 ablation baseline."""
import pytest

from cutamp.config import TAMPConfiguration, validate_tamp_config


def test_new_flags_default_to_current_behavior():
    c = TAMPConfiguration()
    assert c.enable_com_hard_constraint is True
    assert c.enable_com_init_gate is True
    assert c.force_adam is False
    assert c.soft_in_adam == "auto"
    validate_tamp_config(c)  # defaults must validate


def test_soft_in_adam_rejects_unknown_value():
    c = TAMPConfiguration(soft_in_adam="sometimes")
    with pytest.raises(ValueError, match="soft_in_adam"):
        validate_tamp_config(c)


def test_baseline_cell_config_validates():
    # The "standard cuTAMP" cell from the spec.
    c = TAMPConfiguration(
        enable_com_aware_ik=False,
        enable_com_init_gate=False,
        enable_com_hard_constraint=False,
        enable_com_polygon=False,
        coupled_reik=False,
        force_adam=True,
        soft_in_adam="always",
    )
    validate_tamp_config(c)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_ablation_flags.py -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'enable_com_hard_constraint'`

- [ ] **Step 3: Add config fields + validation**

In `cutamp/config.py`, after the `enable_com_aware_ik` field, add:

```python
    # --- Ablation-study flags (2x2 baseline; docs/superpowers/specs/
    # 2026-08-05-standard-cutamp-baseline-design.md). All default to the
    # full system; the baseline cells turn them off/on explicitly. ---
    # L3: register the ComPolygon hard constraint (satisfying-mask member).
    enable_com_hard_constraint: bool = True
    # L2: particle init uses the CoM-safe retry wrapper; False = plain IK.
    enable_com_init_gate: bool = True
    # Fairness: run Phase 1 even when init already satisfies and coupled_reik
    # is off (today's skip_adam shortcut would let the baseline skip
    # optimization entirely on easy tasks — upstream never skipped).
    force_adam: bool = False
    # "auto" = soft costs join the Adam loss only when coupled_reik is on
    # (today's rule). "always" = join regardless — the upstream-faithful
    # attempt for the -reIK arms (spec decision (a)).
    soft_in_adam: str = "auto"
```

In `validate_tamp_config`, after the `reik_lbfgs_iters` check, add:

```python
    if config.soft_in_adam not in ("auto", "always"):
        raise ValueError(
            f"soft_in_adam must be 'auto' or 'always', not {config.soft_in_adam!r}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_ablation_flags.py -q`
Expected: 3 passed

- [ ] **Step 5: CLI plumbing**

In `cutamp/scripts/run_cutamp.py`, after the `--no_enable_com_aware_ik` argument block, add:

```python
    parser.add_argument(
        "--no_com_all", action="store_true",
        help="Ablation umbrella: disable ALL CoM machinery — seed-IK residual "
             "(L1), init gate+retries (L2), ComPolygon hard constraint (L3), "
             "trajopt cost (L5). Equivalent to the four individual flags.",
    )
    parser.add_argument(
        "--force_adam", action="store_true",
        help="Run Phase 1 even when init already satisfies with coupled_reik "
             "off (fairness fix for the ablation baseline).",
    )
    parser.add_argument(
        "--soft_in_adam", choices=["auto", "always"], default=None,
        help="'always' puts soft costs in the Adam loss even without "
             "coupled_reik (upstream-faithful attempt for -reIK arms).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="torch.manual_seed for reproducible sampling (ablation study "
             "uses seeds 0..4).",
    )
```

In the `TAMPConfiguration(...)` construction, replace the line
`enable_com_aware_ik=args.enable_com_aware_ik,` with:

```python
        enable_com_aware_ik=args.enable_com_aware_ik and not args.no_com_all,
        enable_com_init_gate=not args.no_com_all,
        enable_com_hard_constraint=not args.no_com_all,
        force_adam=args.force_adam,
        soft_in_adam=args.soft_in_adam or TAMPConfiguration.soft_in_adam,
```

and replace `enable_com_polygon=args.enable_com_polygon,` with:

```python
        enable_com_polygon=args.enable_com_polygon and not args.no_com_all,
```

At the top of `entrypoint()` after `args = parser.parse_args()`, add:

```python
    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)
```

- [ ] **Step 6: Verify CLI parses and defaults are unchanged**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp --help | grep -A2 no_com_all`
Expected: help text prints; exit 0.

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_ablation_flags.py cutamp/tests/test_best_snapshot.py -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add cutamp/config.py cutamp/scripts/run_cutamp.py cutamp/tests/test_ablation_flags.py
git commit -m "feat(ablation): config flags, CLI umbrella, and --seed

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Gate the L3 hard constraint

**Files:**
- Modify: `cutamp/cost_function.py` (~line 517: `add_cost(ComPolygon.type, com_polygon_cost)`)
- Test: append to `cutamp/tests/test_ablation_flags.py`

**Interfaces:**
- Consumes: `config.enable_com_hard_constraint` (Task 2).
- Produces: when `False`, no `ComPolygon.type` entry exists in the cost dict, so the constraint checker's satisfying mask cannot include it (the checker iterates cost-dict entries).

- [ ] **Step 1: Locate the registration and inspect context**

Read `cutamp/cost_function.py` around line 517. The call is inside `CostFunction.__call__` (or a builder it calls) and has `self.config` in scope. Confirm the exact indentation before editing.

- [ ] **Step 2: Write the failing test (CUDA)**

Append to `cutamp/tests/test_ablation_flags.py`:

```python
import torch

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def _tiny_cost_dict(enable_com_hard_constraint: bool):
    from cutamp.config import TAMPConfiguration
    from cutamp.cost_function import CostFunction
    from cutamp.rollout import RolloutFunction
    from cutamp.particle_initialization import ParticleInitializer
    from cutamp.t1_domain import all_t1_operators
    from cutamp.task_planning import task_plan_generator

    world = make_blocks_t1_world()
    config = TAMPConfiguration(
        num_particles=8,
        enable_com_hard_constraint=enable_com_hard_constraint,
    )
    plan_skeleton = next(task_plan_generator(
        world.initial_state, world.goal_state, all_t1_operators,
    ))
    particles = ParticleInitializer(world, config)(plan_skeleton)
    rollout = RolloutFunction(plan_skeleton, world, config)(particles)
    return CostFunction(plan_skeleton, world, config)(rollout, particles)


@needs_cuda
def test_com_hard_constraint_flag_controls_cost_dict():
    from cutamp.task_planning.constraints import ComPolygon
    with torch.no_grad():
        on = _tiny_cost_dict(True)
        off = _tiny_cost_dict(False)
    assert ComPolygon.type in on
    assert ComPolygon.type not in off
```

Note: if `task_plan_generator` requires a priority function argument in its
current signature, mirror the call used in `cutamp/algorithm.py`
(`task_plan_generator(initial_state, goal_state, all_t1_operators, ...)`) —
copy the minimal working invocation from `sample_plan_skeleton`'s call chain.

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_ablation_flags.py::test_com_hard_constraint_flag_controls_cost_dict -q`
Expected: FAIL on `assert ComPolygon.type not in off`

- [ ] **Step 4: Implement the gate**

In `cutamp/cost_function.py`, wrap the registration:

```python
        if self.config.enable_com_hard_constraint:
            add_cost(ComPolygon.type, com_polygon_cost)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_ablation_flags.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add cutamp/cost_function.py cutamp/tests/test_ablation_flags.py
git commit -m "feat(ablation): gate ComPolygon hard constraint (L3) behind flag

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Gate the L2 init retries

**Files:**
- Modify: `cutamp/particle_initialization.py` (two `_ik_for_pose_com_safe(...)` call sites, ~lines 361 and 406, plus a small selector)
- Test: append to `cutamp/tests/test_ablation_flags.py`

**Interfaces:**
- Consumes: `config.enable_com_init_gate` (Task 2).
- Produces: `_select_init_ik(config) -> callable` with signature
  `(world, world_from_ee, arm, *, context: str = "") -> ik_result` — returns the
  CoM-safe wrapper when the gate is on, a plain-IK adapter when off.

- [ ] **Step 1: Write the failing test (CPU)**

```python
def test_init_ik_selector_respects_gate():
    from cutamp.config import TAMPConfiguration
    from cutamp.particle_initialization import (
        _ik_for_pose_com_safe, _ik_for_pose_no_gate, _select_init_ik,
    )
    assert _select_init_ik(TAMPConfiguration()) is _ik_for_pose_com_safe
    assert _select_init_ik(
        TAMPConfiguration(enable_com_init_gate=False)
    ) is _ik_for_pose_no_gate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_ablation_flags.py::test_init_ik_selector_respects_gate -q`
Expected: FAIL with ImportError on `_ik_for_pose_no_gate`

- [ ] **Step 3: Implement selector + adapter, rewire call sites**

In `cutamp/particle_initialization.py`, directly under `_ik_for_pose_com_safe`, add:

```python
def _ik_for_pose_no_gate(world, world_from_ee, arm, *, context: str = ""):
    """Ablation (enable_com_init_gate=False): plain IK, no CoM gate or
    retries — layer L2 off. Same call shape as _ik_for_pose_com_safe."""
    return _ik_for_pose(world, world_from_ee, arm)


def _select_init_ik(config):
    """The init-IK entry point for this run's config (L2 on/off)."""
    return _ik_for_pose_com_safe if config.enable_com_init_gate else _ik_for_pose_no_gate
```

In `ParticleInitializer.__call__`, replace both call sites
`ik_result = _ik_for_pose_com_safe(world, world_from_ee, arm, context=...)` with:

```python
                ik_result = _select_init_ik(config)(
                    world, world_from_ee, arm,
                    context=f"pick({arm})",
                )
```

(and `context=f"place({arm})"` at the place site).

- [ ] **Step 4: Run tests (new + init-related CUDA)**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_ablation_flags.py cutamp/tests/test_com_aware_seed_ik.py cutamp/tests/test_com_polygon_ik.py -q`
Expected: all pass (defaults unchanged: `_ik_for_pose_com_safe` still the default path)

- [ ] **Step 5: Commit**

```bash
git add cutamp/particle_initialization.py cutamp/tests/test_ablation_flags.py
git commit -m "feat(ablation): gate CoM init retries (L2) behind flag

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `force_adam` and `soft_in_adam` in the optimizer

**Files:**
- Modify: `cutamp/optimize_plan.py` (the `skip_adam = ...` line ~505 and the `consider_types` block ~470; extract two pure helpers at module level)
- Test: append to `cutamp/tests/test_ablation_flags.py`

**Interfaces:**
- Consumes: `config.force_adam`, `config.soft_in_adam` (Task 2).
- Produces module-level helpers (unit-testable, used by `ParticleOptimizer.__call__`):
  `_should_skip_adam(init_num: int, config) -> bool`,
  `_adam_consider_types(config) -> set[str]`.

- [ ] **Step 1: Write the failing tests (CPU)**

```python
def test_should_skip_adam_logic():
    from cutamp.config import TAMPConfiguration
    from cutamp.optimize_plan import _should_skip_adam
    # Today's behavior at defaults (coupled_reik=True): never skip.
    assert not _should_skip_adam(10, TAMPConfiguration())
    # Uncoupled + init satisfied: skips today...
    c = TAMPConfiguration(coupled_reik=False)
    assert _should_skip_adam(10, c)
    assert not _should_skip_adam(0, c)
    # ...unless force_adam (the baseline fairness fix).
    assert not _should_skip_adam(10, TAMPConfiguration(coupled_reik=False, force_adam=True))


def test_adam_consider_types_soft_gate():
    from cutamp.config import TAMPConfiguration
    from cutamp.optimize_plan import _adam_consider_types
    assert _adam_consider_types(TAMPConfiguration()) == {"constraint", "cost"}
    assert _adam_consider_types(TAMPConfiguration(coupled_reik=False)) == {"constraint"}
    assert _adam_consider_types(
        TAMPConfiguration(coupled_reik=False, soft_in_adam="always")
    ) == {"constraint", "cost"}
    assert _adam_consider_types(
        TAMPConfiguration(optimize_soft_costs=False, soft_in_adam="always")
    ) == {"constraint"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_ablation_flags.py -q -k "skip_adam or consider_types"`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement helpers and rewire**

At module level in `cutamp/optimize_plan.py` (below `_build_ik_deps`):

```python
def _should_skip_adam(init_num: int, config: TAMPConfiguration) -> bool:
    """Skip Phase 1 only when init already satisfies, re-IK is off, and the
    ablation fairness flag hasn't forced optimization (upstream never
    skipped; skipping is our shortcut for the uncoupled configuration)."""
    return init_num > 0 and not config.coupled_reik and not config.force_adam


def _adam_consider_types(config: TAMPConfiguration) -> set:
    """Cost types in the Adam loss. Soft costs join when coupled re-IK makes
    them trainable ("auto") or unconditionally ("always" — the ablation's
    upstream-faithful attempt; historically destroys 21-DOF configs)."""
    consider = {"constraint"}
    if config.optimize_soft_costs and (
        config.coupled_reik or config.soft_in_adam == "always"
    ):
        consider.add("cost")
    return consider
```

In `ParticleOptimizer.__call__`, replace the existing lines:

```python
        consider_types = {"constraint"}
        # (comment block)
        if self.config.optimize_soft_costs and self.config.coupled_reik:
            consider_types.add("cost")
```

with:

```python
        consider_types = _adam_consider_types(self.config)
```

and replace:

```python
        skip_adam = (init_num > 0) and not self.config.coupled_reik
```

with:

```python
        skip_adam = _should_skip_adam(init_num, self.config)
```

Keep the deleted comment's content by moving it onto the helper docstrings (already done above).

- [ ] **Step 4: Run the tests + full suite**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/ -q`
Expected: all pass (defaults produce identical behavior)

- [ ] **Step 5: Commit**

```bash
git add cutamp/optimize_plan.py cutamp/tests/test_ablation_flags.py
git commit -m "feat(ablation): force_adam and soft_in_adam fairness switches

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Measure-only end-of-run metrics (the shared yardstick)

**Files:**
- Modify: `cutamp/optimize_plan.py` — the final-check block (after the final re-IK / before best-particle selection; near the existing `com_in_polygon_per_conf` diagnostic)
- Test: modify `cutamp/tests/test_coupled_reik_smoke.py` (assert new metric keys)

**Interfaces:**
- Consumes: `compute_com_polygon_mask` (existing), `compute_com_margin` (Task 1).
- Produces metric keys in `opt_metrics` (consumed by Task 9's analysis):
  `num_satisfying_com_feasible: int`, `best_com_margin: float | None`
  (margin of the selected best particle, min over its conf tensors),
  `best_soft_cost_com_feasible: float | None`.

- [ ] **Step 1: Extend the smoke test (failing first)**

In `cutamp/tests/test_coupled_reik_smoke.py`, after the existing end-of-run
assertions (the test already runs `cutamp_demo` and reads the experiment
dir's `optimization/opt_*.json`), add:

```python
    # Ablation yardstick: measure-only metrics recorded in EVERY arm.
    assert "num_satisfying_com_feasible" in opt_metrics
    assert "best_com_margin" in opt_metrics
    assert "best_soft_cost_com_feasible" in opt_metrics
    assert opt_metrics["num_satisfying_com_feasible"] <= opt_metrics["num_satisfying_final"]
```

If the smoke test does not currently load the opt metrics JSON, add (mirroring
how `analyze`-style code reads it):

```python
    import glob, json
    opt_files = sorted(glob.glob(str(tmp_path / "**" / "optimization" / "opt_*.json"), recursive=True))
    assert opt_files, "no optimization metrics written"
    opt_metrics = json.load(open(opt_files[-1]))
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_coupled_reik_smoke.py -q`
Expected: FAIL on the new asserts

- [ ] **Step 3: Implement in the final-check block**

In `cutamp/optimize_plan.py`, inside the existing
`try:` block that computes `com_in_polygon_per_conf` (it already imports
`compute_com_polygon_mask` and builds `conf_to_mask` for every conf tensor),
extend it:

```python
                # Ablation yardstick (measure-only, recorded in EVERY arm):
                # a particle is CoM-feasible iff ALL its confs are in-polygon.
                all_conf_ok = torch.stack(list(conf_to_mask.values()), dim=0).all(dim=0)
                sat_and_com = satisfying_mask & all_conf_ok
                opt_metrics["num_satisfying_com_feasible"] = int(sat_and_com.sum().item())
                soft_np = soft_costs[sat_and_com]
                opt_metrics["best_soft_cost_com_feasible"] = (
                    soft_np.min().item() if sat_and_com.any() else None
                )
                # Margin of the selected best particle: min over its confs.
                from cutamp.com_polygon_cost import compute_com_margin
                margins = [
                    compute_com_margin(rollout_fn.world, particles[cn][best_idx : best_idx + 1]).item()
                    for cn in conf_names
                ]
                opt_metrics["best_com_margin"] = min(margins) if margins else None
```

Placement note: `best_idx` is computed *after* this diagnostic today (the
best-particle selection block). Move the `best_idx` selection **above** the
diagnostic try-block (it only needs `soft_costs`, `costs`, `satisfying_mask`,
`indices`, all already computed), then compute the metrics. Keep the
diagnostic's existing behavior (log line + `com_in_polygon_per_conf`) intact.
If `sat_and_com` is empty, `best_com_margin` still records the selected
best particle's margin (selection falls back to lowest cost as today).

- [ ] **Step 4: Run smoke + full suite**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_coupled_reik_smoke.py -q` then the full suite.
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add cutamp/optimize_plan.py cutamp/tests/test_coupled_reik_smoke.py
git commit -m "feat(ablation): measure-only CoM yardstick metrics in every arm

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Trajectory CoM audit (measure-only)

**Files:**
- Modify: `cutamp/motion_solver.py` (new function `audit_trajectory_com` at module level, after `solve_curobo`)
- Modify: `cutamp/algorithm.py` (call the audit after a successful `_motion_plan_with_retries`, store in `overall_metrics`)
- Test: modify `cutamp/tests/test_motion_anchor.py` (the existing end-to-end test already produces a plan; assert the audit keys)

**Interfaces:**
- Consumes: `compute_com_polygon_mask` (existing).
- Produces: `audit_trajectory_com(world, curobo_plan) -> tuple[float, int]` —
  (fraction of trajectory frames with CoM outside the polygon, total frames).
  Stored as `overall_metrics["traj_com_violation_frac"]`,
  `overall_metrics["traj_com_frames"]`, and
  `overall_metrics["motion_plan_success"]` (consumed by Task 9). `run_cutamp`
  returns `(curobo_plan, num_satisfying, overall_metrics)` from this task on.

- [ ] **Step 1: Write the failing test**

`cutamp/tests/test_motion_anchor.py::test_all_segment_arrivals_com_in_hull`
runs the pipeline via `_generate_plan_segments`, which builds a
`TAMPConfiguration` with `enable_experiment_logging=False` — so
`overall_metrics.json` is never written there. Make the audit observable to
the test by threading it through the return value instead:
`run_cutamp` already returns `(curobo_plan, num_satisfying)`; extend it to
`(curobo_plan, num_satisfying, overall_metrics)` (a dict — it exists locally
in `run_cutamp`) and update ALL callers found by
`grep -rn "run_cutamp(" cutamp/ --include="*.py"` (at minimum:
`cutamp/scripts/run_cutamp.py::cutamp_demo` and
`cutamp/tests/test_motion_anchor.py::_generate_plan_segments`).

Then in `_generate_plan_segments`, also return the metrics dict, and assert in
the test body:

```python
    world, segments, overall = _generate_plan_segments(seed=seed)
    ...
    # Trajectory audit is recorded (measure-only) whenever a plan is produced.
    assert overall["motion_plan_success"] is True
    assert 0.0 <= overall["traj_com_violation_frac"] <= 1.0
    assert overall["traj_com_frames"] > 0
    # Full system: trajectories should be (near-)fully inside the polygon.
    assert overall["traj_com_violation_frac"] < 0.05
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_motion_anchor.py -q -k arrivals`
Expected: FAIL (unpack error or KeyError on `traj_com_violation_frac`)

- [ ] **Step 3: Implement the audit inside solve_curobo**

The conversion planner-cspace → full-cspace needs the planner's kinematics,
which is in scope inside `solve_curobo`. In `cutamp/motion_solver.py`, just
before `return accum_plans`, add:

```python
    # Measure-only trajectory CoM audit (recorded in every ablation arm).
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    total_frames, violation_frames = 0, 0
    with torch.no_grad():
        for entry in accum_plans:
            plan_js = entry["plan"]
            pos = plan_js.position.reshape(-1, plan_js.position.shape[-1])
            js = JointState.from_position(pos, joint_names=list(plan_js.joint_names))
            full = planner.kinematics.get_full_js(js).reorder(
                list(world.kinematics.joint_names)
            ).position
            ok = compute_com_polygon_mask(world, full)
            total_frames += int(ok.numel())
            violation_frames += int((~ok).sum().item())
    audit = {
        "traj_com_violation_frac": (violation_frames / total_frames) if total_frames else 0.0,
        "traj_com_frames": total_frames,
    }
    return accum_plans, audit
```

Change `solve_curobo`'s return contract to `(accum_plans, audit)` and update
its consumers:

1. `cutamp/algorithm.py::_motion_plan_with_retries` — unpack
   `curobo_plan, audit = solve_curobo(...)`; extend its return tuple to
   `(curobo_plan, audit, skip_plan)`; failure paths return `(None, None, ...)`.
2. The call site in `run_cutamp`'s loop:

```python
                curobo_plan, traj_audit, skip_plan = _motion_plan_with_retries(
                    plan_info, candidates, world, config, timer, visualizer,
                    get_planner=get_planner,
                )
                overall_metrics["motion_plan_success"] = bool(curobo_plan)
                if skip_plan or curobo_plan is None:
                    continue
                overall_metrics.update(traj_audit)
```

3. `run_cutamp`'s return statement: return `overall_metrics` as the third
   element, and update `cutamp_demo` + `_generate_plan_segments` to unpack
   three values.
4. Any other direct `solve_curobo(` callers found by the grep in Step 1.

- [ ] **Step 4: Run the motion tests + full suite**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_motion_anchor.py cutamp/tests/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add cutamp/motion_solver.py cutamp/algorithm.py cutamp/tests/test_motion_anchor.py
git commit -m "feat(ablation): measure-only trajectory CoM audit

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: The study runner

**Files:**
- Create: `cutamp/scripts/run_ablation.py`
- Test: `cutamp/tests/test_run_ablation.py` (create; CPU — tests manifest/plan logic with a stubbed executor)

**Interfaces:**
- Consumes: the CLI flags of Task 2 (builds `run_cutamp` invocations).
- Produces: `CELLS: dict[str, list[str]]` (cell name → extra CLI args),
  `build_run_plan(heights, seeds) -> list[dict]` (each: `cell`, `height`,
  `seed`, `experiment_id`, `args`), manifest JSON at
  `<out_dir>/manifest.json` (list of run dicts + `status` + `experiment_path`).
  Consumed by Task 9.

- [ ] **Step 1: Write the failing tests (CPU)**

```python
# cutamp/tests/test_run_ablation.py
"""Runner plan/manifest logic (no GPU: executor stubbed)."""
import json

from cutamp.scripts.run_ablation import CELLS, build_run_plan, run_study


def test_cells_match_spec():
    assert set(CELLS) == {"full", "no_com", "no_reik", "standard"}
    assert CELLS["full"] == []
    assert "--no_com_all" in CELLS["no_com"]
    for cell in ("no_reik", "standard"):
        assert "--no_coupled_reik" in CELLS[cell]
        assert "--force_adam" in CELLS[cell]
        assert "--soft_in_adam" in CELLS[cell] and "always" in CELLS[cell]
    assert "--no_com_all" in CELLS["standard"]


def test_build_run_plan_grid_and_ids():
    plan = build_run_plan(heights=[0.0, 0.5], seeds=[0, 1])
    assert len(plan) == 4 * 2 * 2
    ids = [r["experiment_id"] for r in plan]
    assert len(set(ids)) == len(ids)  # unique
    r = plan[0]
    assert {"cell", "height", "seed", "experiment_id", "args"} <= set(r)
    assert f"--seed" in r["args"] and "--surface_height" in r["args"]


def test_run_study_is_resumable(tmp_path):
    calls = []
    def fake_exec(run):  # stub executor: pretend every run succeeds
        calls.append(run["experiment_id"])
        return 0
    plan = build_run_plan(heights=[0.0], seeds=[0])
    run_study(plan, out_dir=tmp_path, execute=fake_exec)
    assert len(calls) == 4
    manifest = json.load(open(tmp_path / "manifest.json"))
    assert all(r["status"] == "done" for r in manifest)
    # Second invocation: nothing re-runs.
    calls.clear()
    run_study(plan, out_dir=tmp_path, execute=fake_exec)
    assert calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_run_ablation.py -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement the runner**

```python
# cutamp/scripts/run_ablation.py
"""Runner for the standard-cuTAMP 2x2 ablation study.

Sweeps 4 cells x heights x seeds, one sequential GPU run each, writing
standard experiment dirs plus a resumable manifest. See
docs/superpowers/specs/2026-08-05-standard-cutamp-baseline-design.md.

Usage:
    python -m cutamp.scripts.run_ablation --out /tmp/ablation-study
    python -m cutamp.scripts.run_ablation --out ... --smoke   # tiny pilot
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

CELLS = {
    "full": [],
    "no_com": ["--no_com_all"],
    "no_reik": ["--no_coupled_reik", "--force_adam", "--soft_in_adam", "always"],
    "standard": ["--no_com_all", "--no_coupled_reik", "--force_adam",
                 "--soft_in_adam", "always"],
}
HEIGHTS = [0.0, 0.25, 0.50, 0.65]
SEEDS = [0, 1, 2, 3, 4]
BASE_ARGS = ["-n", "512", "--disable_visualizer", "--motion_plan"]
SMOKE_ARGS = ["-n", "64", "--num_opt_steps", "30", "--disable_visualizer"]


def build_run_plan(heights=HEIGHTS, seeds=SEEDS, smoke=False):
    plan = []
    for cell, cell_args in CELLS.items():
        for h in heights:
            for s in seeds:
                exp_id = f"ablation_{cell}_h{int(h * 100):03d}_s{s}"
                args = (SMOKE_ARGS if smoke else BASE_ARGS) + [
                    "--surface_height", str(h),
                    "--seed", str(s),
                    "--experiment_id", exp_id,
                ] + cell_args
                plan.append({
                    "cell": cell, "height": h, "seed": s,
                    "experiment_id": exp_id, "args": args,
                })
    return plan


def _subprocess_exec(run):
    cmd = [sys.executable, "-m", "cutamp.scripts.run_cutamp"] + run["args"]
    print(f"[{run['experiment_id']}] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env={"PYTHONPATH": "", **_clean_env()})


def _clean_env():
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    return env


def run_study(plan, out_dir, execute=_subprocess_exec, experiment_root="/tmp/cutamp-experiments"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else []
    done = {r["experiment_id"] for r in manifest if r.get("status") == "done"}
    for run in plan:
        if run["experiment_id"] in done:
            continue
        code = execute(run)
        manifest = [r for r in manifest if r["experiment_id"] != run["experiment_id"]]
        manifest.append({
            **{k: run[k] for k in ("cell", "height", "seed", "experiment_id")},
            "status": "done" if code == 0 else f"failed({code})",
            "experiment_path": str(Path(experiment_root) / run["experiment_id"]),
        })
        manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="study output dir (manifest lives here)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny pilot: 64 particles, 30 steps, 1 height, 1 seed, no motion plan")
    a = ap.parse_args()
    plan = build_run_plan(
        heights=[0.0] if a.smoke else HEIGHTS,
        seeds=[0] if a.smoke else SEEDS,
        smoke=a.smoke,
    )
    manifest = run_study(plan, a.out)
    failed = [r for r in manifest if r["status"] != "done"]
    print(f"{len(manifest) - len(failed)}/{len(manifest)} runs done; {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_run_ablation.py -q`
Expected: 3 passed

- [ ] **Step 5: One real smoke cell on GPU**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_ablation --out /tmp/ablation-smoke --smoke`
Expected: exit 0, `/tmp/ablation-smoke/manifest.json` has 4 entries all `done`.
This is the first time the −CoM / standard cells ever execute — if a cell
crashes, fix the flag plumbing before continuing (likely spots: cost dict
consumers assuming `ComPolygon.type` exists; the checker's multiplier lookup).

- [ ] **Step 6: Commit**

```bash
git add cutamp/scripts/run_ablation.py cutamp/tests/test_run_ablation.py
git commit -m "feat(ablation): resumable study runner (cells x heights x seeds)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Analysis — table + figures

**Files:**
- Create: `cutamp/scripts/analyze_ablation.py`
- Test: `cutamp/tests/test_analyze_ablation.py` (create; CPU with synthetic experiment dirs)

**Interfaces:**
- Consumes: manifest (Task 8) + per-run `optimization/opt_*.json` (Task 6 keys)
  and `overall_metrics.json` (Task 7 keys).
- Produces: `load_study(manifest_path) -> list[dict]` (one flat record per run:
  cell, height, seed, num_satisfying_final, num_satisfying_com_feasible,
  best_soft_cost_com_feasible, best_com_margin, traj_com_violation_frac,
  loop_s), `aggregate(records) -> dict[(cell, height), dict[metric, (mean, std)]]`,
  and `main()` writing `results.csv`, `results.md`, and `figures/*.png` to
  `--out`.

- [ ] **Step 1: Write the failing tests (CPU, synthetic data)**

```python
# cutamp/tests/test_analyze_ablation.py
"""Analysis pipeline on synthetic experiment dirs (no GPU)."""
import json
from pathlib import Path

from cutamp.scripts.analyze_ablation import aggregate, load_study


def _fake_run(root: Path, cell, height, seed, sat=100, sat_com=90):
    exp = root / f"ablation_{cell}_h{int(height*100):03d}_s{seed}"
    (exp / "optimization").mkdir(parents=True)
    opt = {
        "num_satisfying_final": sat,
        "num_satisfying_com_feasible": sat_com,
        "best_soft_cost_com_feasible": 20.0 + seed,
        "best_com_margin": 0.05,
        "elapsed": [0.1, 30.0],
        "num_satisfying": [1, sat],
    }
    (exp / "optimization" / "opt_0001.json").write_text(json.dumps(opt))
    (exp / "overall_metrics.json").write_text(json.dumps({
        "traj_com_violation_frac": 0.0 if cell in ("full", "no_reik") else 0.2,
        "traj_com_frames": 500,
    }))
    return {"cell": cell, "height": height, "seed": seed,
            "experiment_id": exp.name, "status": "done",
            "experiment_path": str(exp)}


def test_load_and_aggregate(tmp_path):
    manifest = [
        _fake_run(tmp_path, c, 0.0, s)
        for c in ("full", "standard") for s in (0, 1)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    records = load_study(tmp_path / "manifest.json")
    assert len(records) == 4
    assert records[0]["num_satisfying_com_feasible"] == 90
    agg = aggregate(records)
    mean, std = agg[("full", 0.0)]["best_soft_cost_com_feasible"]
    assert mean == 20.5 and round(std, 2) == 0.71  # seeds 0,1 -> 20,21


def test_main_writes_outputs(tmp_path, monkeypatch):
    manifest = [
        _fake_run(tmp_path, c, h, s)
        for c in ("full", "no_com", "no_reik", "standard")
        for h in (0.0, 0.5) for s in (0, 1)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    import sys
    from cutamp.scripts import analyze_ablation
    monkeypatch.setattr(sys, "argv", [
        "analyze_ablation", "--manifest", str(tmp_path / "manifest.json"),
        "--out", str(tmp_path / "out"),
    ])
    analyze_ablation.main()
    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "results.md").exists()
    figs = list((tmp_path / "out" / "figures").glob("*.png"))
    assert len(figs) >= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_analyze_ablation.py -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# cutamp/scripts/analyze_ablation.py
"""Turn an ablation-study manifest into the paper table and figures.

Usage:
    python -m cutamp.scripts.analyze_ablation \
        --manifest /tmp/ablation-study/manifest.json --out docs/ablation
"""
import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = [
    "num_satisfying_final",
    "num_satisfying_com_feasible",
    "best_soft_cost_com_feasible",
    "best_com_margin",
    "traj_com_violation_frac",
    "motion_plan_success",
    "loop_s",
]
CELL_ORDER = ["full", "no_com", "no_reik", "standard"]
CELL_LABEL = {
    "full": "Full system",
    "no_com": "−CoM",
    "no_reik": "−re-IK",
    "standard": "Standard cuTAMP",
}


def load_study(manifest_path):
    manifest = json.load(open(manifest_path))
    records = []
    for run in manifest:
        if run.get("status") != "done":
            continue
        exp = Path(run["experiment_path"])
        opt_files = sorted(glob.glob(str(exp / "optimization" / "opt_*.json")))
        if not opt_files:
            continue
        opt = json.load(open(opt_files[-1]))
        overall = {}
        if (exp / "overall_metrics.json").exists():
            overall = json.load(open(exp / "overall_metrics.json"))
        records.append({
            "cell": run["cell"], "height": run["height"], "seed": run["seed"],
            "num_satisfying_final": opt.get("num_satisfying_final"),
            "num_satisfying_com_feasible": opt.get("num_satisfying_com_feasible"),
            "best_soft_cost_com_feasible": opt.get("best_soft_cost_com_feasible"),
            "best_com_margin": opt.get("best_com_margin"),
            "traj_com_violation_frac": overall.get("traj_com_violation_frac"),
            "motion_plan_success": (
                float(overall["motion_plan_success"])
                if "motion_plan_success" in overall else None
            ),
            "loop_s": (opt.get("elapsed") or [None])[-1],
        })
    return records


def aggregate(records):
    cells = defaultdict(lambda: defaultdict(list))
    for r in records:
        for m in METRICS:
            if r.get(m) is not None:
                cells[(r["cell"], r["height"])][m].append(r[m])
    out = {}
    for key, metrics in cells.items():
        out[key] = {}
        for m, vals in metrics.items():
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            out[key][m] = (mean, std)
    return out


def _write_csv(agg, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "height"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")])
        for (cell, height), metrics in sorted(agg.items()):
            row = [cell, height]
            for m in METRICS:
                mean, std = metrics.get(m, (None, None))
                row += [mean, std]
            w.writerow(row)


def _write_md(agg, path):
    lines = ["| cell | height | satisfying | CoM-feasible | best soft (CoM-feas.) | margin [m] | traj viol. | loop [s] |",
             "|---|---|---|---|---|---|---|---|"]
    for (cell, height), m in sorted(agg.items(), key=lambda kv: (kv[0][1], CELL_ORDER.index(kv[0][0]))):
        def fmt(name, digits=1):
            if name not in m:
                return "—"
            mean, std = m[name]
            return f"{mean:.{digits}f} ± {std:.{digits}f}"
        lines.append(
            f"| {CELL_LABEL[cell]} | {height:.2f} | {fmt('num_satisfying_final', 0)} "
            f"| {fmt('num_satisfying_com_feasible', 0)} | {fmt('best_soft_cost_com_feasible')} "
            f"| {fmt('best_com_margin', 3)} | {fmt('traj_com_violation_frac', 3)} | {fmt('loop_s', 0)} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def _figures(agg, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    heights = sorted({h for (_, h) in agg})
    plots = [
        ("num_satisfying_com_feasible", "CoM-feasible satisfying particles"),
        ("num_satisfying_final", "Satisfying particles (own rules)"),
        ("best_soft_cost_com_feasible", "Best soft cost (CoM-feasible subset)"),
        ("traj_com_violation_frac", "Trajectory frames outside polygon (fraction)"),
    ]
    for metric, title in plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        for cell in CELL_ORDER:
            xs, ys, es = [], [], []
            for h in heights:
                if (cell, h) in agg and metric in agg[(cell, h)]:
                    mean, std = agg[(cell, h)][metric]
                    xs.append(h); ys.append(mean); es.append(std)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=CELL_LABEL[cell])
        ax.set_xlabel("surface height [m]")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load_study(a.manifest)
    agg = aggregate(records)
    _write_csv(agg, out / "results.csv")
    _write_md(agg, out / "results.md")
    _figures(agg, out / "figures")
    print(f"{len(records)} runs -> {out}/results.csv, results.md, figures/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/test_analyze_ablation.py -q`
Expected: 2 passed

- [ ] **Step 5: Full suite**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add cutamp/scripts/analyze_ablation.py cutamp/tests/test_analyze_ablation.py
git commit -m "feat(ablation): analysis - CSV/markdown table and figures

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Four-cell GPU smoke + pilot readiness

**Files:**
- No new files. Uses Tasks 2-9.

**Interfaces:**
- Consumes: the runner (`--smoke`) and analysis end to end.

- [ ] **Step 1: Run the four-cell smoke**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_ablation --out /tmp/ablation-smoke2 --smoke`
Expected: 4/4 done. Inspect each cell's log for sanity:
- `no_com` / `standard`: log line `COM-in-polygon (per conf)` still prints
  (diagnostic is measure-only) and typically shows counts < batch for `*_q3`
  confs; `num_satisfying_com_feasible < num_satisfying_final` is plausible.
- `no_reik` / `standard`: no `re-IK refresh:` log lines (subsystem inactive);
  Phase 1 ran (progress bar advanced past step 0) — `force_adam` working.

- [ ] **Step 2: Analyze the smoke study**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.analyze_ablation --manifest /tmp/ablation-smoke2/manifest.json --out /tmp/ablation-smoke2/analysis`
Expected: table + figures produced; no crashes on cells lacking motion-plan
metrics (smoke runs skip `--motion_plan`; `traj_com_violation_frac` shows "—").

- [ ] **Step 3: Full suite one last time**

Run: `PYTHONPATH= /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/ -q`
Expected: all pass.

- [ ] **Step 4: Commit any fixes and mark the branch ready**

```bash
git status --short   # commit whatever the smoke shook out, then:
git log --oneline main..ablation-baseline
```

Expected: a clean sequence of feat(ablation) commits on top of the spec.
The full 80-run study is then launched manually (it is a 3-hour GPU job, not
part of implementation):

```bash
PYTHONPATH= python -m cutamp.scripts.run_ablation --out /tmp/ablation-study
PYTHONPATH= python -m cutamp.scripts.analyze_ablation \
    --manifest /tmp/ablation-study/manifest.json --out docs/ablation
```
