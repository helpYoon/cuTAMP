# COM-polygon & MPC-consumer review fixes (A/B/D/E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four root causes from the `/code-review max` pass — the COM penalty/mask/tol corner disagreement (A), the dead 0.0625 trunk-X offset references (B), missing numpy `.copy()` in the MPC example (D), and four small hardening items (E) — without touching the vendored `curobo/` tree.

**Architecture:** Source-of-truth refactor for A (one `com_polygon_penalty` geometry + module constants that the cost, mask, tol, and robot cfgs all reference), pure doc/comment edits for B, defensive copies for D, and isolated one-spot fixes for E. Each task is independently landable and committed separately.

**Tech Stack:** Python, PyTorch (CUDA tensors), cuRobo v0.8, pytest. Project python is `/home/yoonwoo/miniconda3/envs/tamp/bin/python`. All pytest runs use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

**Spec:** `docs/superpowers/specs/2026-05-29-com-and-mpc-review-fixes-design.md`. Cause C (velocity/acceleration derivation) is intentionally **out of scope** — it lives in `docs/superpowers/specs/2026-05-29-plan-velocity-acceleration-derivation-design.md` and is held until its C4 investigation resolves.

---

## Hard constraints (apply to EVERY task)

- **Never edit anything under `curobo/`** (vendored, read-only). All edits are in `cutamp/`, `examples/`, or `docs/`.
- **Stage only the specific files named in each commit.** Never `git add -A` / `git add .`.
- **The working tree already has uncommitted changes** to `cutamp/utils/plan_processor.py`, `examples/load_motion_plan_for_mpc.py`, and `data/motion_plan.pkl` from a prior session. Tasks B and D modify the first two of those files; their commits will include those pre-existing edits. Do **not** commit `data/motion_plan.pkl` (it is a regenerable binary artifact, unrelated to this plan).
- **Anchor edits on the quoted old code, not just line numbers.** Line numbers in this plan are hints; the working tree may have drifted. If a quoted snippet is not found verbatim, stop and re-read the file before editing.
- Every pytest command is prefixed `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

---

## File structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| `cutamp/com_polygon_cost.py` | COM-over-polygon penalty, mask, per-conf penalties, cost cfg; **new home of the COM constants** | A (Task 1, 2) |
| `cutamp/scripts/utils.py` | constraint tol / multiplier tables | A (Task 2) |
| `cutamp/robots/t1.py` | planner + IK solver construction, COM cost cfgs | A (Task 2) |
| `cutamp/tests/test_com_polygon_ik.py` | COM penalty/mask unit tests | A (Task 1) |
| `cutamp/cost_reduction.py` | `CostReducer.get_cost` weighted-sum reducer | E3 (Task 3) |
| `cutamp/tests/test_cost_reduction.py` | CostReducer unit tests | E3 (Task 3) |
| `cutamp/utils/plan_processor.py` | raw plan → MPC schema (docstrings/comments only) | B (Task 4) |
| `examples/load_motion_plan_for_mpc.py` | consumer recipe; schema check | B (Task 4), D (Task 5), E4 (Task 7) |
| `docs/sim_to_real_mapping.md` | sim↔real discrepancy doc | B (Task 4) |
| `cutamp/algorithm.py` | `make_arm_affinity_priority_fn` BFS priority | E1 (Task 6) |
| `cutamp/tests/test_arm_affinity.py` | arm-affinity priority unit tests | E1 (Task 6) |
| `cutamp/particle_initialization.py` | IK seed diversification | E2 (Task 8) |

Suggested order (from the spec's Sequencing): **Task 1 → 2 (A) → 3 (E3) → 4 (B) → 5 (D) → 6 (E1) → 7 (E4) → 8 (E2)**.

---

## Task 1: A — Fix the COM penalty geometry + add module constants (TDD core)

This is the behavioral heart of cause A. The penalty currently **sums** the inside-margin barrier over both axes, so a COM near a support-rectangle *corner* scores ~2× the tolerance and is wrongly rejected. Switching the inside barrier to **max-over-axes** makes a corner score the same as a single edge (`= tol`, satisfied). We also introduce the COM constants here so later tasks can reference them.

**Files:**
- Modify: `cutamp/com_polygon_cost.py` (add constants; change `com_polygon_penalty`; route `compute_com_polygon_mask` through the penalty)
- Test: `cutamp/tests/test_com_polygon_ik.py`

- [ ] **Step 1: Write the failing corner-parity test**

Append to `cutamp/tests/test_com_polygon_ik.py` (the module imports `pytest` at the top; follow the file's convention of importing `torch` and the cost helpers *inside* each test). These two tests run on CPU — **no `@needs_cuda` marker** — because they call `com_polygon_penalty` directly with hand-built tensors:

```python
def test_com_polygon_penalty_corner_equals_single_edge():
    # A COM ~1mm inside BOTH edges (near a corner) must score the same as
    # being on a single edge — NOT double. With the buggy summed barrier it
    # scores ~7.2e-4 (> tol 4e-4) and gets wrongly rejected; with the correct
    # max-over-axes barrier it scores ~3.6e-4 (< tol).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = torch.eye(4).unsqueeze(0)
    e = 0.001  # 1mm inside each edge
    com_world = torch.tensor([[0.10 - e, 0.15 - e, 0.0]])
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight)
    # nearest-edge barrier: weight * (margin - e)**2 = 1 * 0.019**2 = 3.61e-4
    assert pen.item() == pytest.approx((margin - e) ** 2, rel=1e-3)
    assert pen.item() < 4e-4  # passes the COM tol; the summed bug gave 7.22e-4


def test_com_polygon_penalty_corner_on_boundary_equals_tol():
    # COM exactly on the corner -> penalty == weight*margin**2 == COM tol 4e-4.
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = torch.eye(4).unsqueeze(0)
    com_world = torch.tensor([[0.10, 0.15, 0.0]])
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight)
    assert pen.item() == pytest.approx(weight * margin ** 2, rel=1e-4)  # 4e-4


def test_com_polygon_penalty_single_edge_equals_tol():
    # COM exactly on ONE edge (X), deep inside the other (Y) -> penalty == tol.
    # This is the calibration case the tol was derived from; max-over-axes must
    # leave it unchanged from the old summed behavior (only the X axis is active).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = torch.eye(4).unsqueeze(0)
    com_world = torch.tensor([[0.10, 0.0, 0.0]])  # on X edge, centered in Y
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight)
    assert pen.item() == pytest.approx(weight * margin ** 2, rel=1e-4)  # 4e-4


def test_com_polygon_penalty_just_outside_exceeds_tol():
    # COM 1mm past one edge -> penalty > tol (outside quadratic + saturated barrier).
    import torch
    from cutamp.com_polygon_cost import com_polygon_penalty
    half = torch.tensor([0.10, 0.15])
    margin, weight = 0.02, 1.0
    base_T = torch.eye(4).unsqueeze(0)
    com_world = torch.tensor([[0.101, 0.0, 0.0]])  # 1mm past X edge
    pen = com_polygon_penalty(com_world, base_T, half, margin, weight)
    # outside=0.001 -> 0.001**2; inside barrier saturates at (0.001+0.02)**2.
    # Sum: 1e-6 + 4.41e-4 ≈ 4.42e-4 > tol 4e-4.
    assert pen.item() > 4e-4
    assert pen.item() == pytest.approx(0.001 ** 2 + (0.001 + margin) ** 2, rel=1e-3)
```

- [ ] **Step 2: Run the new tests; confirm the corner tests FAIL**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py -k "com_polygon_penalty_corner or com_polygon_penalty_single_edge or com_polygon_penalty_just_outside" -v
```
Expected (against the current summed code):
- `test_com_polygon_penalty_corner_equals_single_edge` → **FAIL** (summed penalty returns `2 * 0.019**2 = 7.22e-4`, not `3.61e-4`, and not `< 4e-4`).
- `test_com_polygon_penalty_corner_on_boundary_equals_tol` → **FAIL** (returns `2 * 0.02**2 = 8e-4`, not `4e-4`).
- `test_com_polygon_penalty_single_edge_equals_tol` → **PASS** already (only the X axis is active, so sum == max == 4e-4 — this is an invariance guard that the sum→max change must not break).
- `test_com_polygon_penalty_just_outside_exceeds_tol` → **PASS** already (single active axis; same invariance guard).

The two corner FAILs are the red tests that Step 4 turns green; the two PASSes confirm single-axis behavior is preserved.

- [ ] **Step 3: Add the COM constants to `cutamp/com_polygon_cost.py`**

Immediately after the imports block (after the line `from curobo._src.cost.cost_base_cfg import BaseCostCfg`) and before `@dataclass\nclass ComOverBasePolygonCostCfg`, insert:

```python
# --- Single source of truth for the COM-over-base-polygon geometry ---
# Two-foot support hull per actual_robot.urdf: 22.3 cm foot length (X, fore/aft)
# × 31.2 cm stance width (Y, lateral). The cost, the post-IK mask, the per-conf
# penalties helper, the ComPolygon tolerance, and the planner/IK cost cfgs all
# reference these so they cannot drift apart.
COM_HALF_EXTENTS = (0.1115, 0.156)   # (X, Y) half-extents in mobile_base_link frame, meters
COM_INSIDE_MARGIN = 0.02             # inside-edge soft-barrier band width, meters
COM_INSIDE_WEIGHT = 1.0              # inside-barrier scale vs the outside quadratic
COM_TOL = 4e-4                       # == COM_INSIDE_WEIGHT * COM_INSIDE_MARGIN**2 (on-boundary penalty)
```

- [ ] **Step 4: Change the inside barrier to max-over-axes in `com_polygon_penalty`**

Find this exact return line in `com_polygon_penalty`:

```python
    return (outside ** 2).sum(dim=-1) + inside_weight * (inside ** 2).sum(dim=-1)
```

Replace it with:

```python
    # Inside barrier uses MAX over axes (distance to the NEAREST edge), not sum:
    # near a corner the nearest-edge margin — not the sum of both — is what
    # bounds tip-over. Summing double-counts corners and wrongly rejects
    # COM-feasible configs (see test_com_polygon_penalty_corner_*).
    return (outside ** 2).sum(dim=-1) + inside_weight * (inside ** 2).max(dim=-1).values
```

- [ ] **Step 5: Run the new tests; confirm all four PASS**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py -k "com_polygon_penalty_corner or com_polygon_penalty_single_edge or com_polygon_penalty_just_outside" -v
```
Expected: all four PASS (the two corner tests are now green; the two single-axis invariance guards stay green).

- [ ] **Step 6: Write the failing mask/penalty-equivalence test**

The mask currently uses a separate geometric `abs(com) <= half` test; we will route it through the penalty so it equals `penalty <= COM_TOL`. Add a test that the mask agrees with the penalty gate, using the file's **existing** `_make_world()` helper and `@needs_cuda` marker (both already defined at the top of `cutamp/tests/test_com_polygon_ik.py` — do NOT add a new fixture). Append:

```python
@needs_cuda
def test_mask_matches_penalty_gate():
    # The post-IK mask must equal (penalty <= COM_TOL) elementwise, so the IK
    # COM-safe retry loop and the ConstraintChecker can never disagree.
    import torch
    from cutamp.com_polygon_cost import (
        COM_TOL, compute_com_polygon_mask, compute_com_polygon_penalties,
    )
    world = _make_world(enable_com_polygon=True)
    # Spread a batch of configs around home so some land near the polygon edge.
    torch.manual_seed(0)
    home = world.q_init.detach().clone()
    q = home.unsqueeze(0).repeat(32, 1).clone()
    q[:, 3:7] += 0.3 * torch.randn(32, 4, device=q.device)   # perturb body DOFs
    mask = compute_com_polygon_mask(world, q)
    pens = compute_com_polygon_penalties(world, {"q": q})["q"]
    gate = pens <= COM_TOL
    assert torch.equal(mask, gate)
```

- [ ] **Step 7: Run the equivalence test; confirm it FAILS (or skips without CUDA)**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py::test_mask_matches_penalty_gate -v
```
Expected: SKIP if no CUDA (the `@needs_cuda` marker handles that). On a machine with CUDA, it may FAIL (the mask still uses the geometric test, which can disagree with the new max-penalty gate at the float boundary / near corners) or, if no perturbed config happens to land in the disagreement band, it may PASS already — either way Step 8 makes the equality hold by construction. If it SKIPs, note that and proceed — Step 9's smoke test exercises the real path.

- [ ] **Step 8: Route `compute_com_polygon_mask` through the penalty**

In `compute_com_polygon_mask`, find this exact block at the end of the function:

```python
    R = base_T[:, :3, :3]
    t = base_T[:, :3, 3]
    com_in_base = (R.transpose(-1, -2) @ (com - t).unsqueeze(-1)).squeeze(-1)[:, :2]
    half = torch.tensor(half_extents, device=device, dtype=q_batch.dtype)
    return (torch.abs(com_in_base) <= half).all(dim=-1)
```

Replace it with:

```python
    # Gate through the SAME penalty the hard ComPolygon constraint uses, so the
    # post-IK COM-safe check and ConstraintChecker can never diverge. With the
    # max-over-axes barrier, (penalty <= COM_TOL) == "COM inside-or-on the hull".
    half = torch.tensor(half_extents, device=device, dtype=q_batch.dtype)
    pen = com_polygon_penalty(com, base_T, half, COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT)
    return pen <= COM_TOL
```

(`com` and `base_T` are already computed earlier in the function; `com_polygon_penalty`, `COM_INSIDE_MARGIN`, `COM_INSIDE_WEIGHT`, `COM_TOL` are all in this same module.)

- [ ] **Step 9: Run the equivalence test; confirm it PASSES (or skips)**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py::test_mask_matches_penalty_gate -v
```
Expected: PASS (or SKIP without CUDA).

- [ ] **Step 10: Run the whole COM test file; fix any stale corner assumptions**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py -v
```
Expected: all pass. If a pre-existing test (e.g. one named like `*matches_mask_at_tolerance` or `*filters_com_violators`) asserted a *summed* two-axis corner value, update its expected number to the max-over-axes value (penalty at a near-corner = `weight * max(inside_axis)**2`, not the sum). Single-axis tests (`*_inside_margin_barrier`, `*_outside_positive`, `*_zero_inside`) are unaffected by the sum→max change and must still pass unchanged.

- [ ] **Step 11: Commit**

```bash
git add cutamp/com_polygon_cost.py cutamp/tests/test_com_polygon_ik.py
git commit -m "$(cat <<'EOF'
fix: COM polygon barrier uses max-over-axes; unify mask with penalty gate

The inside-margin barrier summed over both axes, so a COM near a support-
rectangle corner scored ~2x the 4e-4 tol and was wrongly rejected. Switch
to max-over-axes (distance to the nearest edge) so a corner scores the same
as a single edge. Route compute_com_polygon_mask through com_polygon_penalty
<= COM_TOL so the IK COM-safe check and ConstraintChecker use one function.
Add COM_HALF_EXTENTS / COM_INSIDE_MARGIN / COM_INSIDE_WEIGHT / COM_TOL as the
single source of truth.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: A — Point every COM site at the constants (resolve finding 14)

The half-extents are currently documented/defaulted three different ways (`[0.10,0.15]`, `[0.05,0.10]`, `[0.1115,0.156]`). Wire every site to the Task 1 constants so they agree by construction.

**Files:**
- Modify: `cutamp/com_polygon_cost.py` (cfg default + docstrings, mask default + docstring, penalties defaults)
- Modify: `cutamp/scripts/utils.py` (ComPolygon tol → `COM_TOL`)
- Modify: `cutamp/robots/t1.py` (both cost cfgs → constants)

- [ ] **Step 1: Cfg default + docstring in `com_polygon_cost.py`**

In `ComOverBasePolygonCostCfg`, find the docstring line:
```python
    #: meters. Default ``(0.10, 0.15)`` matches a 20 cm fore/aft × 30 cm
    #: lateral support polygon.
```
Replace with:
```python
    #: meters. Default ``COM_HALF_EXTENTS`` = ``(0.1115, 0.156)`` — the two-foot
    #: support hull per actual_robot.urdf (22.3 cm fore/aft × 31.2 cm lateral).
```
Then find in `__post_init__`:
```python
        if self.half_extents is None:
            self.half_extents = [0.10, 0.15]
```
Replace with:
```python
        if self.half_extents is None:
            self.half_extents = list(COM_HALF_EXTENTS)
```

- [ ] **Step 2: Mask default + docstring in `com_polygon_cost.py`**

In `compute_com_polygon_mask`, find the docstring line referencing the default:
```python
        half_extents: ``[2]``-list of (X, Y) half-sizes in meters. Default
            ``[0.05, 0.10]`` matches the safety rectangle used by the
            planner-side COM cost.
```
Replace with:
```python
        half_extents: ``[2]``-list of (X, Y) half-sizes in meters. Default
            ``COM_HALF_EXTENTS`` = ``[0.1115, 0.156]`` — the same two-foot
            support hull the cost and per-conf penalties use.
```
Then find the default assignment (a `[0.1115, 0.156]` literal under an `if half_extents is None:` with a multi-line comment) and replace the literal with the constant:
```python
    if half_extents is None:
        half_extents = COM_HALF_EXTENTS
```
(Delete the now-redundant inline comment block that re-derived `[0.1115, 0.156]`; the constant's definition carries that rationale.)

- [ ] **Step 3: Penalties defaults in `com_polygon_cost.py`**

In `compute_com_polygon_penalties`, change the signature defaults:
```python
def compute_com_polygon_penalties(
    world,
    particles: "dict[str, torch.Tensor]",
    *,
    half_extents: Optional[List[float]] = None,
    inside_margin: float = 0.02,
    inside_weight: float = 1.0,
) -> "dict[str, torch.Tensor]":
```
to:
```python
def compute_com_polygon_penalties(
    world,
    particles: "dict[str, torch.Tensor]",
    *,
    half_extents: Optional[List[float]] = None,
    inside_margin: float = COM_INSIDE_MARGIN,
    inside_weight: float = COM_INSIDE_WEIGHT,
) -> "dict[str, torch.Tensor]":
```
Then find the body default:
```python
    if half_extents is None:
        half_extents = [0.1115, 0.156]
```
Replace with:
```python
    if half_extents is None:
        half_extents = COM_HALF_EXTENTS
```

- [ ] **Step 4: ComPolygon tol → `COM_TOL` in `cutamp/scripts/utils.py`**

At the top of `cutamp/scripts/utils.py`, add to the imports (after the existing `from cutamp.task_planning.costs import TrajectoryLength` line):
```python
from cutamp.com_polygon_cost import COM_TOL
```
Then in `default_constraint_to_tol`, find:
```python
    ComPolygon.type: {"default": 4e-4},
```
Replace with:
```python
    ComPolygon.type: {"default": COM_TOL},
```
And update the comment block just above it — change the line that says `tol 4e-4 ⇒ "at the edge or inside" satisfies` to:
```python
    # ComPolygon per-conf values are continuous penalty in m² from
    # compute_com_polygon_penalties. With the max-over-axes barrier, the
    # on-boundary penalty (edge OR corner) == COM_TOL == inside_weight *
    # inside_margin² = 4e-4, so "at the boundary or inside" satisfies and any
    # excursion past an edge fails. "default" applies to every per-conf name
    # (left_q0, right_q1, ...).
```

- [ ] **Step 5: Both COM cost cfgs → constants in `cutamp/robots/t1.py`**

There are two `ComOverBasePolygonCostCfg(...)` instantiations — one in `get_t1_motion_planner` (planner-side, cfg at ~line 258) and one in `get_t1_ik_solver` (IK-side, cfg at ~line 330). Each already has its own **multi-line** lazy import (at ~line 235 and ~line 318) that currently reads:
```python
        from cutamp.com_polygon_cost import (
            ComOverBasePolygonCost,
            ComOverBasePolygonCostCfg,
        )
```
In **both** places, extend the import to also pull the constants:
```python
        from cutamp.com_polygon_cost import (
            ComOverBasePolygonCost,
            ComOverBasePolygonCostCfg,
            COM_HALF_EXTENTS,
            COM_INSIDE_MARGIN,
            COM_INSIDE_WEIGHT,
        )
```
Then in **both** cfg constructions, replace the three literal lines (each carries a trailing `#` comment — replace the whole line including the comment):
```python
            half_extents=[0.1115, 0.156],  # two-foot support hull: foot length (22.3cm) × stance width (31.2cm), per actual_robot.urdf
            inside_margin=0.02,           # 2cm band inside edge gets center-pulling gradient
            inside_weight=1.0,            # same scale as outside-quadratic; tunable
```
with:
```python
            half_extents=list(COM_HALF_EXTENTS),
            inside_margin=COM_INSIDE_MARGIN,
            inside_weight=COM_INSIDE_WEIGHT,
```
(Leave `weight=...`, `device_cfg=...` unchanged — `weight` is the outer cost multiplier (planner uses `[5.0e5]`, IK its own value), distinct from the geometry constants. The planner-side cfg has no `base_link_name` override; do not add one.)

- [ ] **Step 6: Verify imports resolve and there's no circular import**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -c \
  "import cutamp.com_polygon_cost as c, cutamp.scripts.utils as u, cutamp.robots.t1 as t; \
   print('COM_TOL', c.COM_TOL, 'half', c.COM_HALF_EXTENTS); \
   print('tol table', u.default_constraint_to_tol['ComPolygon'])"
```
Expected: prints `COM_TOL 0.0004 half (0.1115, 0.156)` and `tol table {'default': 0.0004}` with no ImportError.

- [ ] **Step 7: Re-run the COM test file**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_com_polygon_ik.py -v
```
Expected: all pass (or CUDA-dependent ones skip).

- [ ] **Step 8: End-to-end smoke (the behavioral check for A)**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 64 --num_opt_steps 50 --motion_plan
```
Expected: completes with "Total num satisfying" ≥ 1 and no new tracebacks. (If a GPU is unavailable in this environment, note that the smoke could not run and rely on the unit tests + Step 6 import check.)

- [ ] **Step 9: Commit**

```bash
git add cutamp/com_polygon_cost.py cutamp/scripts/utils.py cutamp/robots/t1.py
git commit -m "$(cat <<'EOF'
refactor: point all COM-polygon sites at shared COM_* constants

Resolve the three conflicting documented half-extents ([0.10,0.15] /
[0.05,0.10] / [0.1115,0.156]) by referencing COM_HALF_EXTENTS / _MARGIN /
_WEIGHT / _TOL everywhere: cost cfg default, mask default, penalties
defaults, ComPolygon tol, and both planner- and IK-side cost cfgs in t1.py.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: E3 — Guard `CostReducer.get_cost` against an all-one-type cost_dict

`get_cost` initializes `cost = None` and returns `None` when no entry matches `consider_types`; the optimizer then does `hard_costs(...) + soft_costs(...)`, which would raise `TypeError` if a side is empty. Return a zero tensor instead. (Note: the review's claim that `soft_costs` defaults to `("constraint",)` is **wrong** — it correctly uses `{"cost"}`; no change there.)

**Files:**
- Modify: `cutamp/cost_reduction.py` (`get_cost`)
- Test: `cutamp/tests/test_cost_reduction.py`

- [ ] **Step 1: Write the failing test**

Append to `cutamp/tests/test_cost_reduction.py` (it already imports `torch` and `CostReducer`):

```python
def test_get_cost_zero_tensor_when_no_matching_type():
    # cost_dict has only a 'cost' entry; asking for 'constraint' must yield a
    # zero tensor shaped to the particle batch, not None (so the optimizer's
    # hard_costs(...) + soft_costs(...) never hits None + tensor).
    reducer = CostReducer({})
    cost_dict = {"traj": {"type": "cost", "values": {"traj_length": torch.ones(5)}}}
    out = reducer.get_cost(cost_dict, consider_types={"constraint"})
    assert out is not None
    assert out.shape == (5,)
    assert torch.equal(out, torch.zeros(5))


def test_get_cost_normal_path_unchanged():
    reducer = CostReducer({"soft": {"traj_length": 2.0}})
    cost_dict = {"soft": {"type": "cost", "values": {"traj_length": torch.ones(5)}}}
    out = reducer.get_cost(cost_dict, consider_types={"cost"})
    assert torch.equal(out, torch.full((5,), 2.0))
```

- [ ] **Step 2: Run; confirm the first test FAILS**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_cost_reduction.py::test_get_cost_zero_tensor_when_no_matching_type -v
```
Expected: FAIL with `AssertionError` on `assert out is not None` (current code returns `None`).

- [ ] **Step 3: Implement the zero-tensor fallback**

In `cutamp/cost_reduction.py`, find the end of `get_cost`:
```python
                cost = values if cost is None else cost + values
        return cost
```
Replace with:
```python
                cost = values if cost is None else cost + values
        if cost is None:
            # No entry matched consider_types. Return a zero objective shaped
            # to the particle batch (inferred from any entry) so callers can
            # safely do hard_costs(...) + soft_costs(...) without hitting
            # None + tensor. Falls through to None only if cost_dict is empty.
            for entry in cost_dict.values():
                for values in entry["values"].values():
                    return torch.zeros(
                        values.shape[0], device=values.device, dtype=values.dtype
                    )
        return cost
```

- [ ] **Step 4: Run; confirm both new tests PASS**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_cost_reduction.py -v
```
Expected: all pass (the 4 pre-existing `_get_multiplier` tests + the 2 new ones).

- [ ] **Step 5: Commit**

```bash
git add cutamp/cost_reduction.py cutamp/tests/test_cost_reduction.py
git commit -m "$(cat <<'EOF'
fix: CostReducer.get_cost returns zero tensor (not None) on empty side

An all-one-type cost_dict made get_cost return None, which would raise
TypeError in the optimizer's hard_costs(...) + soft_costs(...). Return a
zero tensor shaped to the particle batch instead.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: B — Purge the dead 0.0625 m trunk-X offset from docs/comments

The 0.0625 m sim↔real Trunk-origin difference no longer exists (the MPC-side `actual_robot.urdf` was modified to share sim's Trunk origin). v3 already stores raw sim Trunk FK, which now equals the real Trunk pose. This task removes only the stale offset *language* — **no behavior changes, no data changes, no schema bump**. Preserve lines that *quote URDF joint origins* (e.g. `<origin xyz="0.0625 0 -0.1155"/>`) — those still truthfully explain the joint geometry.

**Files:**
- Modify: `cutamp/utils/plan_processor.py` (docstring + dead comment)
- Modify: `examples/load_motion_plan_for_mpc.py` (header docstring + the inline "no compensation needed" label)
- Modify: `docs/sim_to_real_mapping.md` (section 1 + TL;DR)

- [ ] **Step 1: `plan_processor.py` — module docstring v3 note (lines ~8-12)**

Find:
```
v3 vs v2: removes the -0.0625 X compensation on ``trunk_xyz``. Saved
``trunk_xyz`` is now the raw sim FK output (sim ``t1_simplified.urdf``
Trunk world pose). Consumers using ``t1_simplified.urdf`` (or a matching
URDF) read it directly. Consumers using ``actual_robot.urdf`` must
subtract 0.0625 from ``trunk_xyz[:, 0]`` themselves.
```
Replace with:
```
v3 stores ``trunk_xyz`` as the raw sim FK output (sim ``t1_simplified.urdf``
Trunk world pose). The on-robot ``actual_robot.urdf`` now shares the same
Trunk origin, so the saved value is the real Trunk world pose directly —
no compensation needed on the consumer side.
```

- [ ] **Step 2: `plan_processor.py` — "Why this schema" bullet (lines ~92-95)**

Find:
```
* ``trunk_xyz`` has the +0.0625 m X offset (sim vs real URDF, see
  docs/sim_to_real_mapping.md #1) **already subtracted** so the saved
  value represents real-URDF's Trunk world pose. No compensation needed
  downstream.
```
Replace with:
```
* ``trunk_xyz`` is the Trunk world pose, directly usable on the real robot:
  the sim and on-robot URDFs share the Trunk origin (see
  docs/sim_to_real_mapping.md #1), so no X compensation is needed.
```

- [ ] **Step 3: `plan_processor.py` — fix the "we don't emit leg joints" bullet (lines ~99-102)**

This is finding 5 (unrelated to the offset but in the same docstring): v3 **does** emit `ankle_pitch`/`knee_pitch`. Find:
```
* We don't emit ankle_pitch, knee_pitch, or any other leg joint —
  expectation is that the MPC solves leg IK to match the saved Trunk
  world pose, choosing the missing 3 DOFs (Hip_Roll, Hip_Yaw, Ankle_Roll)
  for balance per its own logic.
```
Replace with:
```
* We emit ``ankle_pitch`` and ``knee_pitch`` (each a single sim joint,
  broadcast to both legs on the real robot). The remaining 3 leg DOFs per
  side (Hip_Roll, Hip_Yaw, Ankle_Roll) are NOT emitted — the MPC chooses
  them for balance, matching the saved Trunk world pose.
```

- [ ] **Step 4: `plan_processor.py` — delete the dead compensation comment (lines ~312-314)**

Find and delete this entire comment block (there is **no code** under it; v3 removed the compensation):
```python
        # Apply -0.0625 X compensation to Trunk world pose: saved value
        # represents real-URDF's Trunk world pose (not sim's), so the MPC
        # consumer needs no compensation.
```
Also find the inline label on the trunk_xyz assignment:
```python
            # Trunk world pose (real-URDF-native)
            "trunk_xyz": trunk_xyz_w,
```
Replace the comment with one that no longer implies a frame conversion:
```python
            # Trunk world pose (raw sim FK; URDFs share Trunk origin → real-native)
            "trunk_xyz": trunk_xyz_w,
```

- [ ] **Step 5: `examples/load_motion_plan_for_mpc.py` — header docstring (lines ~14-21)**

Find:
```
This example expects ``schema_version=3`` pickles. Older plans (v1 with
Trunk-frame hand poses, v2 with -0.0625 X compensation on trunk_xyz)
will be rejected with a regenerate-with-current-code message.

v3 saves ``trunk_xyz`` as the raw sim FK output (sim
``t1_simplified.urdf`` Trunk world pose). If your MPC uses
``actual_robot.urdf`` instead, subtract 0.0625 from
``trunk_xyz[:, 0]`` on the consumer side.
```
Replace with:
```
This example expects ``schema_version=3`` pickles. Older plans (v1 with
Trunk-frame hand poses, v2 with a -0.0625 X compensation on trunk_xyz)
will be rejected with a regenerate-with-current-code message.

v3 saves ``trunk_xyz`` as the raw sim FK output (sim
``t1_simplified.urdf`` Trunk world pose). The on-robot ``actual_robot.urdf``
shares the same Trunk origin, so this value is the real Trunk world pose
directly — no compensation needed.
```

- [ ] **Step 6: `examples/load_motion_plan_for_mpc.py` — the inline "no compensation needed" label (~line 81)**

Find (inside the `segment_to_mpc_commands` docstring schema comment):
```python
                "xyz":                  [T, 3],   # real-URDF-native (no compensation needed)
```
Replace with:
```python
                "xyz":                  [T, 3],   # Trunk world pose (real-native; URDFs share Trunk origin)
```

- [ ] **Step 7: `docs/sim_to_real_mapping.md` — section 1 + TL;DR**

Open `docs/sim_to_real_mapping.md`. In **Section 1** (the "Trunk link frame is offset 6.25 cm" section), change the `**Status**` line and the "What changed in schema v2" paragraph to state the offset is resolved by a URDF change, not by compensation. Concretely:
- Change the Section-1 `**Status**` block from text claiming "plan_processor.py applies -0.0625 X" to:
  ```
  **Status**: ✅ Resolved — the on-robot ``actual_robot.urdf`` was modified to
  share the sim Trunk origin, so the saved (raw sim FK) ``trunk_xyz`` is the
  real Trunk world pose directly. No compensation in plan_processor.py and none
  on the consumer side.
  ```
- In the "**What changed in schema v2**" paragraph, replace the sentence "plan_processor.py subtracts `0.0625` from saved `trunk_xyz[:, 0]`" with "the URDFs now share the Trunk origin, so no subtraction is applied or required."
- Update the TL;DR/summary line that says trunk needs no compensation to reference schema v3 and the shared-origin reason.
- **Keep** the XML blocks that quote joint origins (`<origin xyz="0.0625 0 -0.1155"/>` and `<origin xyz="0 0 0.1155"/>`) and the surrounding explanation of why the sim joint origin is shaped that way — those remain true.

- [ ] **Step 8: Verify no offset survives as a required action / applied compensation**

Run:
```bash
grep -rn '0\.0625' cutamp examples docs
```
Expected: the **only** remaining matches are (a) XML/joint-origin quotes in `docs/sim_to_real_mapping.md` (e.g. `<origin xyz="0.0625 0 -0.1155"/>`), and (b) historical mentions inside other dated files under `docs/superpowers/` (older specs/plans — leave those; they are point-in-time records). There must be **no** match in `cutamp/utils/plan_processor.py` or `examples/load_motion_plan_for_mpc.py` that states the offset as a required consumer action or an applied compensation. Eyeball the list to confirm.

- [ ] **Step 9: Verify the processor still imports and the example still parses**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -c \
  "import cutamp.utils.plan_processor; import ast; ast.parse(open('examples/load_motion_plan_for_mpc.py').read()); print('ok')"
```
Expected: prints `ok` (docstring/comment-only edits keep both files valid).

- [ ] **Step 10: Commit**

```bash
git add cutamp/utils/plan_processor.py examples/load_motion_plan_for_mpc.py docs/sim_to_real_mapping.md
git commit -m "$(cat <<'EOF'
docs: purge dead 0.0625 trunk-X offset from plan-processor docs & example

actual_robot.urdf was modified to share sim's Trunk origin, so the saved
raw sim Trunk FK is the real Trunk pose directly — no compensation. Remove
the contradictory "must subtract 0.0625" / "already subtracted" language and
the dead compensation comment; correct the "we don't emit leg joints" bullet
(v3 does emit ankle/knee pitch). URDF-origin quotes are kept. No behavior or
schema change.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: D — Defensive `.copy()` on shared numpy views in the MPC example

`segment_to_mpc_commands` hands out numpy **views** into the pickled segment arrays for `trunk_world_pose["xyz"]` and the 14 per-arm joint columns, while deliberately `.copy()`-ing the broadcast joints just above. Any in-place edit downstream would silently corrupt the shared segment. Make all emitted arrays independent.

**Files:**
- Modify: `examples/load_motion_plan_for_mpc.py` (`segment_to_mpc_commands`)
- Test: `examples/test_load_motion_plan_for_mpc.py` (new)

- [ ] **Step 1: Write the failing aliasing test**

Create `examples/test_load_motion_plan_for_mpc.py`:

```python
"""Aliasing-safety test for the MPC consumer example."""
import importlib.util
from pathlib import Path

import numpy as np

_MOD_PATH = Path(__file__).resolve().parent / "load_motion_plan_for_mpc.py"
_spec = importlib.util.spec_from_file_location("load_motion_plan_for_mpc", _MOD_PATH)
mpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpc)


def _vel_acc_block(T, xyz_keys):
    # Joint channels: scalar [T] (broadcast joints) or [T,7] (arms).
    d = {}
    for k in ("trunk_pitch", "trunk_yaw", "ankle_pitch", "knee_pitch"):
        d[k] = np.zeros(T)
    for k in ("left_arm", "right_arm"):
        d[k] = np.zeros((T, 7))
    for k in xyz_keys:
        d[k] = np.zeros((T, 3))
    return d


def _fake_segment(T=4):
    pos = {
        "trunk_xyz": np.zeros((T, 3)),
        "trunk_quat_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)),
        "trunk_pitch": np.zeros(T),
        "trunk_yaw": np.zeros(T),
        "ankle_pitch": np.zeros(T),
        "knee_pitch": np.zeros(T),
        "left_arm": np.zeros((T, 7)),
        "right_arm": np.zeros((T, 7)),
        "right_hand_xyz": np.zeros((T, 3)),
        "right_hand_quat_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)),
        "left_hand_xyz": np.zeros((T, 3)),
        "left_hand_quat_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)),
    }
    vel = _vel_acc_block(T, (
        "trunk_xyz_dot", "right_hand_xyz_dot", "left_hand_xyz_dot",
        "trunk_angular_velocity_world", "right_hand_angular_velocity_world",
        "left_hand_angular_velocity_world",
    ))
    acc = _vel_acc_block(T, (
        "trunk_xyz_ddot", "right_hand_xyz_ddot", "left_hand_xyz_ddot",
        "trunk_angular_acceleration_world", "right_hand_angular_acceleration_world",
        "left_hand_angular_acceleration_world",
    ))
    return {"dt": 0.02, "T": T, "position": pos, "velocity": vel,
            "acceleration": acc, "held_objs": {}}


def test_outputs_do_not_alias_source_segment():
    seg = _fake_segment()
    cmd = mpc.segment_to_mpc_commands(seg)
    # Mutate every emitted command stream in place; the source seg must not change.
    cmd["trunk_world_pose"]["xyz"][:] += 1.0
    for d in (cmd["joint_commands"], cmd["joint_velocities"], cmd["joint_accelerations"]):
        for arr in d.values():
            arr[:] += 1.0
    assert np.all(seg["position"]["trunk_xyz"] == 0.0)
    assert np.all(seg["position"]["left_arm"] == 0.0)
    assert np.all(seg["position"]["right_arm"] == 0.0)
```

(Note: `_fake_segment` must mirror the exact keys `segment_to_mpc_commands` reads from `seg["position"]`/`["velocity"]`/`["acceleration"]` and `JOINT_MAP`. If the example's schema has drifted, re-read `segment_to_mpc_commands` and adjust the keys here. Confirmed against the current example: it reads the four broadcast joints, `left_arm`/`right_arm` as `[:, i]`, trunk + hand `xyz`/`quat_xyzw` and their `_dot`/`_ddot`/`angular_*` channels.)

- [ ] **Step 2: Run; confirm it FAILS**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  examples/test_load_motion_plan_for_mpc.py -v
```
Expected: FAIL — `seg["position"]["left_arm"]` becomes non-zero (the per-arm columns alias the source), and/or `trunk_xyz` becomes non-zero.

- [ ] **Step 3: Add `.copy()` to the aliasing assignments**

In `segment_to_mpc_commands`, find the trunk world pose dict assignment:
```python
    trunk_world_pose = {
        "xyz": P["trunk_xyz"],
```
Replace the first line with a copy:
```python
    trunk_world_pose = {
        "xyz": P["trunk_xyz"].copy(),
```
Then find the per-arm loop:
```python
    for sim_field in ("left_arm", "right_arm"):
        for i, rn in enumerate(JOINT_MAP[sim_field]):
            joint_commands[rn] = P[sim_field][:, i]
            joint_velocities[rn] = V[sim_field][:, i]
            joint_accelerations[rn] = A[sim_field][:, i]
```
Replace the three assignment lines with copies:
```python
    for sim_field in ("left_arm", "right_arm"):
        for i, rn in enumerate(JOINT_MAP[sim_field]):
            joint_commands[rn] = P[sim_field][:, i].copy()
            joint_velocities[rn] = V[sim_field][:, i].copy()
            joint_accelerations[rn] = A[sim_field][:, i].copy()
```

- [ ] **Step 4: Run; confirm it PASSES**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  examples/test_load_motion_plan_for_mpc.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/load_motion_plan_for_mpc.py examples/test_load_motion_plan_for_mpc.py
git commit -m "$(cat <<'EOF'
fix: copy trunk-xyz and per-arm columns in MPC example to avoid aliasing

segment_to_mpc_commands handed out numpy views into the pickled segment for
trunk_world_pose["xyz"] and the 14 per-arm joint columns, so an in-place
edit downstream would corrupt the shared segment. Copy them, matching the
broadcast-joint handling already present.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: E1 — Arm-affinity priority returns `inf` (not `0.0`) for the sentinel branches

`make_arm_affinity_priority_fn` returns `0.0` for non-pick / unresolved-block / no-pose ops, but resolved picks return a strictly-positive distance. With ascending BFS sibling sort, `0.0` sorts the sentinels **first**, contradicting the intent that same-side picks bubble up and cross-body groundings enumerate later. Return `float("inf")` so sentinels sort last.

**Files:**
- Modify: `cutamp/algorithm.py` (`make_arm_affinity_priority_fn`)
- Test: `cutamp/tests/test_arm_affinity.py` (extend; the file already exists)

- [ ] **Step 1: Write the failing test**

Append to `cutamp/tests/test_arm_affinity.py`:

```python
import math
import types

from cutamp.algorithm import make_arm_affinity_priority_fn


def _op(action_type, arm, values):
    return types.SimpleNamespace(
        operator=types.SimpleNamespace(
            metadata=types.SimpleNamespace(action_type=action_type, arm=arm)
        ),
        values=values,
    )


def test_priority_non_pick_is_inf():
    # world is never touched for a non-pick op.
    fn = make_arm_affinity_priority_fn(world=None)
    assert fn(_op("place", None, [])) == math.inf


def test_priority_unresolved_block_is_inf():
    class _World:
        def get_object(self, name):
            raise KeyError(name)
    fn = make_arm_affinity_priority_fn(_World())
    assert fn(_op("pick", "left", ["no_such_block"])) == math.inf


def test_priority_sentinel_sorts_after_resolved_pick():
    import numpy as np

    class _Obj:
        pose = [0.4, 0.2, 0.5, 0, 0, 0, 1]
    class _World:
        arm_home_ee_world = {"left": __import__("torch").tensor([0.0, 0.0, 0.0])}
        def get_object(self, name):
            return _Obj()
    fn = make_arm_affinity_priority_fn(_World())
    resolved = _op("pick", "left", ["block"])      # finite distance
    sentinel = _op("place", None, [])               # inf
    ordered = sorted([sentinel, resolved], key=fn)
    assert ordered[0] is resolved and ordered[1] is sentinel
```

- [ ] **Step 2: Run; confirm the inf tests FAIL**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_arm_affinity.py::test_priority_non_pick_is_inf \
  cutamp/tests/test_arm_affinity.py::test_priority_unresolved_block_is_inf \
  cutamp/tests/test_arm_affinity.py::test_priority_sentinel_sorts_after_resolved_pick -v
```
Expected: FAIL (current code returns `0.0`, so the asserts on `math.inf` and the ordering both fail).

- [ ] **Step 3: Change the three sentinel returns to `float("inf")`**

In `make_arm_affinity_priority_fn`, inside the inner `priority(ground_op)` function, there are three `return 0.0` statements (the non-pick/no-arm guard, the `except (KeyError, ValueError)` branch, and the `if obj is None or obj.pose is None` branch). Change **all three** to:
```python
            return float("inf")
```
Then update the docstring sentence:
```python
    For non-pick operators or
    pick operators whose block doesn't resolve (e.g., placeholder name
    not yet bound to a real scene object), returns 0.0.
```
to:
```python
    For non-pick operators or
    pick operators whose block doesn't resolve (e.g., placeholder name
    not yet bound to a real scene object), returns ``float("inf")`` so they
    sort AFTER all resolved picks in the ascending BFS order.
```

- [ ] **Step 4: Run; confirm all three PASS**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_arm_affinity.py -v
```
Expected: all pass (new + any pre-existing).

- [ ] **Step 5: Commit**

```bash
git add cutamp/algorithm.py cutamp/tests/test_arm_affinity.py
git commit -m "$(cat <<'EOF'
fix: arm-affinity priority returns inf for sentinel branches

Non-pick / unresolved-block ops returned 0.0, which sorts FIRST in the
ascending BFS sibling order — ahead of resolved same-side picks, defeating
the affinity bias. Return float("inf") so they enumerate last as intended.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: E4 — Distinct diagnostic when `schema_version` key is absent

`load_for_mpc` defaults a missing `schema_version` to `1`, so a keyless/legacy/corrupt pickle reports the misleading "got 1". Emit a distinct "key absent" message.

**Files:**
- Modify: `examples/load_motion_plan_for_mpc.py` (`load_for_mpc`)
- Test: `examples/test_load_motion_plan_for_mpc.py` (extend the file from Task 5)

- [ ] **Step 1: Write the failing test**

Append to `examples/test_load_motion_plan_for_mpc.py`:

```python
import pickle
import pytest


def test_missing_schema_version_distinct_message(tmp_path):
    p = tmp_path / "plan.pkl"
    with open(p, "wb") as f:
        pickle.dump({"segments": []}, f)  # no schema_version key
    with pytest.raises(RuntimeError, match="(?i)schema_version.*(absent|missing|no )"):
        mpc.load_for_mpc(p)


def test_wrong_schema_version_says_got(tmp_path):
    p = tmp_path / "plan.pkl"
    with open(p, "wb") as f:
        pickle.dump({"schema_version": 2, "segments": []}, f)
    with pytest.raises(RuntimeError, match="got 2"):
        mpc.load_for_mpc(p)
```

- [ ] **Step 2: Run; confirm the first test FAILS**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  examples/test_load_motion_plan_for_mpc.py::test_missing_schema_version_distinct_message -v
```
Expected: FAIL — the current code raises "expected schema_version=3, got 1" (no "absent"/"missing" wording).

- [ ] **Step 3: Distinguish absent from wrong-version in `load_for_mpc`**

Find:
```python
    schema_version = plan.get("schema_version", 1)
    if schema_version != 3:
        raise RuntimeError(
            f"This example expects schema_version=3, got {schema_version}. "
            f"Regenerate the plan with the current code:\n"
            f"  python -m cutamp.scripts.run_cutamp --motion_plan --save_plan {path}"
        )
```
Replace with:
```python
    schema_version = plan.get("schema_version")
    if schema_version is None:
        raise RuntimeError(
            f"motion_plan.pkl at {path} has no 'schema_version' key — it is a "
            f"legacy/unversioned or corrupt plan, not a versioned one. "
            f"Regenerate with the current code:\n"
            f"  python -m cutamp.scripts.run_cutamp --motion_plan --save_plan {path}"
        )
    if schema_version != 3:
        raise RuntimeError(
            f"This example expects schema_version=3, got {schema_version}. "
            f"Regenerate the plan with the current code:\n"
            f"  python -m cutamp.scripts.run_cutamp --motion_plan --save_plan {path}"
        )
```

- [ ] **Step 4: Run; confirm both PASS**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  examples/test_load_motion_plan_for_mpc.py -v
```
Expected: all pass (Task 5's aliasing tests + these two).

- [ ] **Step 5: Commit**

```bash
git add examples/load_motion_plan_for_mpc.py examples/test_load_motion_plan_for_mpc.py
git commit -m "$(cat <<'EOF'
fix: distinct error when motion_plan schema_version key is absent

A keyless/legacy/corrupt pickle defaulted to schema_version=1 and reported
the misleading "got 1". Emit a distinct "no 'schema_version' key" message
and keep "got N" for genuinely-wrong versions.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: E2 — Clamp the noise-perturbed IK seed to the robot's joint limits

In `_ik_for_pose_com_safe`, the COM-safe retry adds gaussian noise (`noise_std = 0.1·(attempt+1)`, up to ~1.5 rad) to the home seed without re-clamping, so large-σ retries can hand cuRobo an out-of-limits seed. Clamp the perturbed seed to the joint limits before the IK call. This task begins with a small discovery step because the exact cuRobo joint-limits accessor must be confirmed against the installed version.

**Files:**
- Modify: `cutamp/particle_initialization.py` (`_ik_for_pose_com_safe`)

- [ ] **Step 1: Discover the joint-limits accessor (run the probe; record the result)**

The seed is a full-cspace tensor in `world.q_init` order (`[B, full_dof]`). We need per-DOF lower/upper limits in that same order. Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python - <<'PY'
import os, torch
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.tamp_world import TAMPWorld
from cutamp.robots import load_robot_container
from cutamp.robots.t1 import t1_home
from curobo.types import DeviceCfg
env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
dc = DeviceCfg()
robot = load_robot_container("t1", dc)
q0 = torch.as_tensor(t1_home, dtype=torch.float32, device=dc.device)
world = TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=q0, enable_com_polygon=True)
kin = world.kinematics
print("full joint_names:", list(kin.joint_names))
print("q_init dof:", world.q_init.shape)
# Probe candidate accessors:
for attr in ("get_joint_limits", "joint_limits"):
    obj = getattr(kin, attr, None)
    print(attr, "->", type(obj))
jl = kin.get_joint_limits() if hasattr(kin, "get_joint_limits") else getattr(kin, "joint_limits", None)
print("limits repr:", type(jl))
pos = getattr(jl, "position", jl)
print("limits.position shape:", getattr(pos, "shape", None))
print("limits.position:", pos)
PY
```

Record from the output: (a) the accessor that works (`kin.get_joint_limits()` vs `kin.joint_limits`), (b) the limits tensor shape (expected `[2, dof]`: row 0 lower, row 1 upper) and its **joint order** (must match `world.kinematics.joint_names`, the same order as `world.q_init`). If the limits are in a different joint order or a different DOF count than `world.q_init`, the clamp must reorder — note this and adapt Step 2 accordingly (reorder the limit rows into `world.kinematics.joint_names` order before clamping).

- [ ] **Step 2: Implement the clamp**

In `_ik_for_pose_com_safe` (`cutamp/particle_initialization.py`), find the seed-construction block:

```python
        seed = world.q_init.unsqueeze(0).expand(batch_size, -1).clone()
        noise = torch.randn(batch_size, full_dof, device=device) * noise_std
        noise[:, :3] = 0.0  # base locked
        if arm == "left":
            noise[:, 14:21] = 0.0  # right arm locked
        elif arm == "right":
            noise[:, 7:14] = 0.0  # left arm locked
        seed = seed + noise
        retry_result = _ik_for_pose(world, world_from_ee, arm, current_state_q=seed)
```

Insert a clamp between `seed = seed + noise` and the `_ik_for_pose(...)` call. Use the accessor confirmed in Step 1 — the example below assumes `world.kinematics.get_joint_limits().position` is `[2, full_dof]` in `world.kinematics.joint_names` order (matching `world.q_init`); **adjust the two lines that fetch `lower`/`upper` to match what the probe found**:

```python
        seed = seed + noise
        # Clamp the perturbed seed to joint limits so large-sigma retries don't
        # hand cuRobo an out-of-limits current_state. Limits come from the same
        # full-cspace kinematics that defines world.q_init's joint order.
        _limits = world.kinematics.get_joint_limits().position  # [2, full_dof]
        _lower = _limits[0].to(seed.device, seed.dtype)
        _upper = _limits[1].to(seed.device, seed.dtype)
        seed = torch.clamp(seed, min=_lower, max=_upper)
        retry_result = _ik_for_pose(world, world_from_ee, arm, current_state_q=seed)
```

- [ ] **Step 3: Verify the clamp keeps the seed in-bounds (probe)**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python - <<'PY'
import os, torch
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.tamp_world import TAMPWorld
from cutamp.robots import load_robot_container
from cutamp.robots.t1 import t1_home
from curobo.types import DeviceCfg
env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
dc = DeviceCfg(); robot = load_robot_container("t1", dc)
q0 = torch.as_tensor(t1_home, dtype=torch.float32, device=dc.device)
world = TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=q0, enable_com_polygon=True)
lim = world.kinematics.get_joint_limits().position
lower, upper = lim[0], lim[1]
# Simulate a huge-noise seed and clamp the way the code now does:
B, dof = 8, world.q_init.shape[-1]
seed = world.q_init.unsqueeze(0).expand(B, -1).clone() + 5.0 * torch.randn(B, dof, device=dc.device)
clamped = torch.clamp(seed, min=lower.to(seed), max=upper.to(seed))
assert torch.all(clamped >= lower.to(seed) - 1e-6) and torch.all(clamped <= upper.to(seed) + 1e-6)
print("clamp OK: all", B*dof, "entries within limits")
PY
```
Expected: prints `clamp OK: ...`. If `get_joint_limits()` is not the right accessor, this probe fails fast and Step 2 must be corrected before committing.

- [ ] **Step 4: Smoke test that particle init still runs**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 64 --num_opt_steps 50 --motion_plan
```
Expected: completes, "Total num satisfying" ≥ 1, no new tracebacks. (If no GPU, note the smoke couldn't run and rely on Step 3's probe.)

- [ ] **Step 5: Commit**

```bash
git add cutamp/particle_initialization.py
git commit -m "$(cat <<'EOF'
fix: clamp noise-perturbed IK seed to joint limits in COM-safe retry

Large-sigma diversification retries (noise_std up to ~1.5 rad) could hand
cuRobo an out-of-limits current_state. Clamp the perturbed full-cspace seed
to the kinematics' joint limits before the IK call.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (after all tasks)

- [ ] **Step 1: Full test suite**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest cutamp/tests/ examples/ -v
```
Expected: all pass (CUDA-dependent COM tests may SKIP if no GPU; nothing fails).

- [ ] **Step 2: End-to-end smoke (behavioral confirmation for A)**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 64 --num_opt_steps 50 --motion_plan
```
Expected: "Total num satisfying" ≥ 1; near-corner COM-feasible particles that were previously dropped are now retained (count ≥ the pre-fix baseline); plan regenerates with no new tracebacks.

- [ ] **Step 3: Confirm `curobo/` was never touched**

Run:
```bash
git diff --name-only HEAD~8..HEAD | grep '^curobo/' && echo "VIOLATION: curobo edited" || echo "OK: no curobo edits"
```
Expected: `OK: no curobo edits`.

- [ ] **Step 4: Finish the branch**

Use the **superpowers:finishing-a-development-branch** skill to verify tests, then choose merge / PR / keep.
