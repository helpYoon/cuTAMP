# Differentiable ComPolygon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the non-differentiable `(~mask).float()` values in the ComPolygon hard constraint with a continuous differentiable penalty so Adam receives a real gradient toward COM feasibility, while sharing per-conf computation with the existing `--soft_cost com_polygon` path.

**Architecture:** Add one helper `compute_com_polygon_penalties` to `cutamp/com_polygon_cost.py` that returns per-conf differentiable penalty tensors. Both `cost_function.com_polygon_constraint()` (hard) and `cost_function._compute_soft_cost("com_polygon")` (soft) call the helper, with a per-`__call__` cache on the `CostFunction` instance so the FK runs once even when both are active. Recalibrate `default_constraint_to_tol[ComPolygon.type]` to `4e-4` (the at-edge penalty value) and add `default_constraint_to_mult[ComPolygon.type] = {"default": 10.0}` so Adam's COM gradient is on par with KinematicConstraint pos_err. Requires extending `CostReducer._get_multiplier` with the same `"default"` fallback `ConstraintChecker._get_tol` already has.

**Tech Stack:** PyTorch, cuRobo (vendored at `curobo/`), cuTAMP's `CostFunction` / `CostReducer` / `ConstraintChecker` triad.

**Reference spec:** `docs/superpowers/specs/2026-05-27-differentiable-com-polygon-design.md`

---

## File map

| File | What changes |
| --- | --- |
| `cutamp/cost_reduction.py` | `_get_multiplier` adds `(cost_type, "default")` fallback |
| `cutamp/com_polygon_cost.py` | Add `compute_com_polygon_penalties` helper |
| `cutamp/cost_function.py` | Add `_com_polygon_penalties_cache` field; rewrite `com_polygon_constraint`; rewrite `_compute_soft_cost("com_polygon")` branch |
| `cutamp/scripts/utils.py` | Tolerance 0.5 → 4e-4; add ComPolygon entry in `default_constraint_to_mult` |
| `cutamp/tests/test_com_polygon_ik.py` | Update `test_constraint_checker_filters_com_violators`; add 2 new tests |
| `cutamp/tests/test_cost_reduction.py` | New file — test the `"default"` fallback in CostReducer |

---

## Tracking

- [ ] Task 1 — CostReducer `"default"` multiplier fallback
- [ ] Task 2 — `compute_com_polygon_penalties` helper
- [ ] Task 3 — Hard `com_polygon_constraint` uses helper + cache
- [ ] Task 4 — Soft `--soft_cost com_polygon` shares the cache
- [ ] Task 5 — Recalibrate tolerance and add multiplier
- [ ] Task 6 — Smoke verification

---

## Task 1 — CostReducer `"default"` multiplier fallback

**Why first:** Task 5's multiplier registration `default_constraint_to_mult[ComPolygon.type] = {"default": 10.0}` is useless unless `CostReducer._get_multiplier` knows to consult the `"default"` key. This task makes the fallback work, with no behavior change to existing constraints (none use `"default"`).

**Files:**
- Create: `cutamp/tests/test_cost_reduction.py`
- Modify: `cutamp/cost_reduction.py` (lines 28–29)

- [ ] **Step 1.1: Write the failing test**

Create `cutamp/tests/test_cost_reduction.py`:

```python
"""Tests for CostReducer multiplier lookup semantics."""
import torch

from cutamp.cost_reduction import CostReducer


def test_default_multiplier_fallback_used_when_name_missing():
    """When (cost_type, name) is missing but (cost_type, 'default') exists,
    the default multiplier applies. Mirrors ConstraintChecker._get_tol's
    existing fallback so plan-skeleton-dependent constraint names
    (left_q0, right_q3, ...) can share a single default weight."""
    cost_config = {"MyCostType": {"default": 7.0}}
    reducer = CostReducer(cost_config)
    cost_dict = {
        "MyCostType": {
            "type": "constraint",
            "constraints": [],
            "values": {
                "some_specific_name": torch.tensor([1.0, 2.0, 3.0]),
            },
        },
    }
    cost = reducer.get_cost(cost_dict, consider_types={"constraint"})
    # value × 7.0 = [7, 14, 21]
    assert torch.allclose(cost, torch.tensor([7.0, 14.0, 21.0])), (
        f"Expected default multiplier 7.0 to apply; got cost={cost}"
    )


def test_exact_name_multiplier_takes_precedence_over_default():
    """If both (cost_type, name) and (cost_type, 'default') exist, the
    exact match wins. Default is only a fallback."""
    cost_config = {"MyCostType": {"default": 7.0, "exact_name": 100.0}}
    reducer = CostReducer(cost_config)
    cost_dict = {
        "MyCostType": {
            "type": "constraint",
            "constraints": [],
            "values": {
                "exact_name": torch.tensor([1.0, 2.0]),
            },
        },
    }
    cost = reducer.get_cost(cost_dict, consider_types={"constraint"})
    # value × 100.0 = [100, 200]
    assert torch.allclose(cost, torch.tensor([100.0, 200.0])), (
        f"Exact-name multiplier 100.0 should win over default; got cost={cost}"
    )


def test_no_multiplier_when_neither_exact_nor_default_exists():
    """When neither (cost_type, name) nor (cost_type, 'default') is in
    the config, values pass through unmultiplied (multiplier == None
    path). This is the pre-existing behavior for constraints with no
    explicit mult (e.g. Collision)."""
    cost_config = {"OtherCostType": {"default": 99.0}}  # different type
    reducer = CostReducer(cost_config)
    cost_dict = {
        "MyCostType": {
            "type": "constraint",
            "constraints": [],
            "values": {
                "name": torch.tensor([1.0, 2.0, 3.0]),
            },
        },
    }
    cost = reducer.get_cost(cost_dict, consider_types={"constraint"})
    assert torch.allclose(cost, torch.tensor([1.0, 2.0, 3.0])), (
        f"Unmultiplied passthrough expected; got cost={cost}"
    )
```

- [ ] **Step 1.2: Run test to verify it fails**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python \
  -m pytest cutamp/tests/test_cost_reduction.py -v
```

Expected: `test_default_multiplier_fallback_used_when_name_missing` FAILS (default fallback not implemented; multiplier resolves to `None`; values pass through unscaled; expected `[7, 14, 21]` ≠ actual `[1, 2, 3]`).
`test_exact_name_multiplier_takes_precedence_over_default` PASSES (exact lookup already works).
`test_no_multiplier_when_neither_exact_nor_default_exists` PASSES.

- [ ] **Step 1.3: Implement the fallback**

Edit `cutamp/cost_reduction.py`, replace lines 28–29:

Before:
```python
    def _get_multiplier(self, cost_type: str, name: str) -> Optional[float]:
        return self.cost_to_multiplier.get((cost_type, name))
```

After:
```python
    def _get_multiplier(self, cost_type: str, name: str) -> Optional[float]:
        direct = self.cost_to_multiplier.get((cost_type, name))
        if direct is not None:
            return direct
        # Mirror ConstraintChecker._get_tol's "default" fallback so
        # constraints with plan-skeleton-dependent inner names (e.g.
        # ComPolygon's per-conf entries left_q0, right_q3, ...) can
        # share a single weight without enumerating every possible name.
        return self.cost_to_multiplier.get((cost_type, "default"))
```

- [ ] **Step 1.4: Run tests to verify all pass**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python \
  -m pytest cutamp/tests/test_cost_reduction.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add cutamp/cost_reduction.py cutamp/tests/test_cost_reduction.py
git commit -m "feat: CostReducer falls back to (type, 'default') multiplier

Mirrors ConstraintChecker._get_tol's existing fallback semantics so
constraints with plan-skeleton-dependent inner names (per-conf entries
like ComPolygon's left_q0, right_q3) can register a single default
weight. Backwards-compatible: no existing constraint uses 'default' as
a name field, so exact-name lookups continue to take precedence."
```

---

## Task 2 — `compute_com_polygon_penalties` helper

**Why:** The differentiable per-conf penalty primitive that both the hard constraint and the soft cost will consume. Isolating it in `com_polygon_cost.py` keeps the FK + penalty math in one place and lets us test differentiability and per-conf parity-with-the-mask directly.

**Files:**
- Modify: `cutamp/com_polygon_cost.py` (append after `compute_com_polygon_mask`, end of file)
- Test: `cutamp/tests/test_com_polygon_ik.py` (add two new tests; both need CUDA)

- [ ] **Step 2.1: Write the failing differentiability test**

Append to `cutamp/tests/test_com_polygon_ik.py` (after `test_curobo_batched_com_kernel_returns_per_batch_distinct`):

```python
@needs_cuda
def test_compute_com_polygon_penalties_is_differentiable():
    """The shared helper must return tensors that carry an autograd
    connection back to the input joint positions. This is the property
    that makes the hard ComPolygon constraint actually usable as an
    Adam gradient source (the prior (~mask).float() lost this)."""
    import torch
    from cutamp.com_polygon_cost import compute_com_polygon_penalties

    world = _make_world(enable_com_polygon=True)
    home = world.q_init.detach().clone()
    B = 4
    q = home.unsqueeze(0).expand(B, -1).contiguous().clone().requires_grad_(True)
    particles = {"left_q1": q}

    pens = compute_com_polygon_penalties(world, particles)
    assert "left_q1" in pens, f"Expected 'left_q1' key in result; got {list(pens.keys())}"
    p = pens["left_q1"]
    assert p.shape == (B,), f"Expected shape ({B},), got {p.shape}"
    assert p.requires_grad, "Penalty tensor must carry an autograd connection."

    # Backward through the sum; q.grad must be populated and non-trivial.
    p.sum().backward()
    assert q.grad is not None, "Expected q.grad to be populated after backward."
    # Particles at home pose should have low penalty (COM inside polygon)
    # but the gradient through the FK chain is non-zero for at least some DOFs.
    assert q.grad.abs().sum().item() > 0, (
        f"Expected non-zero gradient on q; got all-zero grad."
    )


@needs_cuda
def test_compute_com_polygon_penalties_matches_mask_at_tolerance():
    """Parity between the mask helper and the penalty helper: a
    particle's penalty ≤ inside_weight * inside_margin² iff its COM
    is inside the polygon. Calibrates the hard constraint's tolerance
    (4e-4) against the mask's definition of 'inside'."""
    import torch
    from cutamp.com_polygon_cost import (
        compute_com_polygon_mask,
        compute_com_polygon_penalties,
    )

    world = _make_world(enable_com_polygon=True)
    full_names = list(world.kinematics.joint_names)
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    home = world.q_init.detach().clone()
    bent = home.clone()
    bent[name_to_idx["Torso_Pitch"]] = -1.5  # COM far forward → outside polygon
    bent[name_to_idx["knee_pitch"]] = +0.6

    q = torch.stack([home, bent], dim=0)  # [2, full_dof]

    mask = compute_com_polygon_mask(world, q)  # [B] Bool
    pens = compute_com_polygon_penalties(world, {"left_q1": q})
    p = pens["left_q1"]  # [B] float
    tol = 1.0 * (0.02) ** 2  # inside_weight * inside_margin² = 4e-4

    # mask True ↔ COM inside polygon ↔ penalty ≤ tol
    inside_mask = (p <= tol)
    assert torch.equal(mask, inside_mask), (
        f"Penalty-vs-mask disagree.\n  mask={mask.tolist()}\n"
        f"  penalties={p.tolist()}  tol={tol}\n  inside_mask={inside_mask.tolist()}"
    )
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python \
  -m pytest cutamp/tests/test_com_polygon_ik.py::test_compute_com_polygon_penalties_is_differentiable \
  cutamp/tests/test_com_polygon_ik.py::test_compute_com_polygon_penalties_matches_mask_at_tolerance -v
```

Expected: both tests FAIL with `ImportError: cannot import name 'compute_com_polygon_penalties' from 'cutamp.com_polygon_cost'`.

- [ ] **Step 2.3: Implement the helper**

Append to `cutamp/com_polygon_cost.py` (end of file, after `compute_com_polygon_mask`):

```python
def compute_com_polygon_penalties(
    world,
    particles: "dict[str, torch.Tensor]",
    *,
    half_extents: Optional[List[float]] = None,
    inside_margin: float = 0.02,
    inside_weight: float = 1.0,
) -> "dict[str, torch.Tensor]":
    """Per-conf differentiable COM-polygon penalty.

    For every Conf-shaped particle in ``particles`` (name starts with
    ``left_q``/``right_q``/``q`` and ``ndim == 2``), runs
    ``world.kinematics_with_com.compute_kinematics`` once and computes
    the per-particle COM-over-polygon penalty via
    :func:`com_polygon_penalty`. Returns a dict mapping conf name to a
    ``[B]`` tensor of penalties in m². Tensors are differentiable
    through cuRobo's FK chain — calling ``.backward()`` on a function of
    these values populates ``.grad`` on the input Conf particles.

    Used by both the hard ``ComPolygon`` constraint (via
    :meth:`CostFunction.com_polygon_constraint`) and the soft
    ``--soft_cost com_polygon`` path (via
    :meth:`CostFunction._compute_soft_cost`). When both are active in
    the same ``CostFunction.__call__``, the result is cached on the
    ``CostFunction`` instance so the FK runs once per conf.

    Args:
        world: TAMPWorld instance. ``world.kinematics_with_com`` must
            be available (cached property — first call builds it).
        particles: dict mapping particle name to ``[B, full_dof]``
            tensor. Non-Conf particles (poses, grasps) are skipped.
        half_extents: ``[2]``-list of (X, Y) half-sizes in meters.
            Default ``[0.1115, 0.156]`` matches the cost-side polygon.
        inside_margin: width of the inside-edge barrier band (meters).
        inside_weight: relative scale of the inside barrier vs the
            outside quadratic.

    Returns:
        ``{conf_name: [B] float tensor}`` of penalty values in m².
        Empty dict if no Conf particles are in ``particles``.
    """
    from curobo.types import JointState
    if half_extents is None:
        half_extents = [0.1115, 0.156]

    kin = world.kinematics_with_com
    device = kin.device_cfg.device
    joint_names = list(world.kinematics.joint_names)
    half = torch.tensor(half_extents, device=device, dtype=torch.float32)

    conf_names = [
        p for p in particles
        if p.startswith(("left_q", "right_q", "q")) and particles[p].ndim == 2
    ]
    out: "dict[str, torch.Tensor]" = {}
    for name in conf_names:
        q = particles[name].to(device)
        js = JointState.from_position(q, joint_names=joint_names)
        active_js = kin.get_active_js(js)
        ks = kin.compute_kinematics(active_js)
        com_world = ks.robot_com[..., :3].reshape(-1, 3)
        base_T = (
            ks.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True)
            .get_matrix()
            .reshape(-1, 4, 4)
        )
        out[name] = com_polygon_penalty(
            com_world, base_T, half, inside_margin, inside_weight,
        )
    return out
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python \
  -m pytest cutamp/tests/test_com_polygon_ik.py::test_compute_com_polygon_penalties_is_differentiable \
  cutamp/tests/test_com_polygon_ik.py::test_compute_com_polygon_penalties_matches_mask_at_tolerance -v
```

Expected: both tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add cutamp/com_polygon_cost.py cutamp/tests/test_com_polygon_ik.py
git commit -m "feat: add compute_com_polygon_penalties helper

Per-conf differentiable COM-polygon penalty primitive. Both the hard
ComPolygon constraint and the --soft_cost com_polygon path will call
this helper; sharing it lets the FK run once per conf even when both
paths are active. Two regression tests: differentiability (the
property the old (~mask).float() lost) and parity with
compute_com_polygon_mask at tolerance inside_weight * inside_margin²."
```

---

## Task 3 — Hard `com_polygon_constraint` uses helper + cache

**Why:** Replaces the non-differentiable `(~mask).float()` values with the helper's differentiable penalties. Adds a `__call__`-scoped cache on `CostFunction` so when the soft cost (Task 4) also uses the helper, the FK runs once.

**Files:**
- Modify: `cutamp/cost_function.py` (`com_polygon_constraint`, `__call__`)

Note: The synthetic filter test (`test_constraint_checker_filters_com_violators`) keeps its existing `0.0`/`1.0` values for this task. Tolerance is still `0.5` until Task 5 lowers it to `4e-4`, so the test passes unchanged. The value update to penalty-style (`0.0`/`1e-2`) lands in Task 5 alongside the tol change so both reflect the new continuous-penalty semantics.

- [ ] **Step 3.1: Add cache field and reset point**

In `cutamp/cost_function.py`, find the start of `__call__` (around line 539). Locate:

```python
    def __call__(self, rollout: Rollout, particles: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, dict]:
        self._validate_rollout(rollout)
        cost_dict = {}
        
        # Store particles for soft cost computation
        self._particles = particles
```

Add the cache reset immediately after `self._particles = particles`:

```python
    def __call__(self, rollout: Rollout, particles: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, dict]:
        self._validate_rollout(rollout)
        cost_dict = {}
        
        # Store particles for soft cost computation
        self._particles = particles
        # Reset per-call cache for the shared COM-polygon penalty helper.
        # com_polygon_constraint() and _compute_soft_cost("com_polygon")
        # both consult this; lazy-populated on first use so we only run
        # the FK once per __call__ even when both are active.
        self._com_polygon_penalties_cache: Optional[Dict[str, torch.Tensor]] = None
```

- [ ] **Step 3.2: Rewrite `com_polygon_constraint` to use the helper + cache**

In `cutamp/cost_function.py`, find the existing `com_polygon_constraint` method (added at commit `b2ce8ab`, just before `soft_costs`). Replace its body with:

```python
    def com_polygon_constraint(self) -> Union[dict, None]:
        """Hard COM-over-base-polygon filter, per conf.

        Mirrors how Collision works: per-conf continuous penalty values
        (units: m²), ANDed into the overall satisfying mask by
        ``ConstraintChecker.get_mask``. The penalty is the same
        differentiable inside-barrier function used by the soft cost
        (sharing the FK computation via
        ``self._com_polygon_penalties_cache``), so Adam receives a real
        gradient toward COM-feasibility instead of a dead-weight 0/1
        Boolean.
        """
        if not self.config.enable_com_polygon or self._particles is None:
            return None

        if self._com_polygon_penalties_cache is None:
            from cutamp.com_polygon_cost import compute_com_polygon_penalties
            self._com_polygon_penalties_cache = compute_com_polygon_penalties(
                self.world, self._particles,
            )
        pens = self._com_polygon_penalties_cache
        if not pens:
            return None

        return {
            "type": "constraint",
            "constraints": [],
            "values": dict(pens),  # shallow copy: ConstraintChecker iterates this dict
        }
```

- [ ] **Step 3.3: Run the existing filter test to confirm it still passes**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python \
  -m pytest cutamp/tests/test_com_polygon_ik.py::test_constraint_checker_filters_com_violators -v
```

Expected: PASS (test body unchanged; ConstraintChecker semantics unchanged; values 0.0/1.0 still filter correctly under tol=0.5).

- [ ] **Step 3.4: Run the new helper tests to confirm they still pass**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python \
  -m pytest cutamp/tests/test_com_polygon_ik.py -v
```

Expected: all 8 tests PASS (no regression from Task 2's additions; constraint test still works at the old tol since the dict literal values don't change).

- [ ] **Step 3.5: Commit**

```bash
git add cutamp/cost_function.py
git commit -m "refactor: ComPolygon hard constraint uses differentiable helper

Replaces the (~mask).float() 0/1 values with per-conf continuous
penalties from compute_com_polygon_penalties. Adds a per-__call__
cache (_com_polygon_penalties_cache) so when the soft cost also uses
the helper in the next commit, FK runs once. Behavior change: values
are now in m² (penalty), not 0/1. Tolerance recalibration is in a
follow-up commit; with the current tol=0.5 the filter is essentially
inactive (penalty values are O(1e-4 to 1e-2) so almost everything
passes). This is intentional — tol change ships in Task 5 alongside
the multiplier."
```

---

## Task 4 — Soft `--soft_cost com_polygon` shares the cache

**Why:** Eliminates the duplicate FK pass when both hard and soft ComPolygon are active. Functionally equivalent to the prior soft-cost code (same penalty function, same params, same sum-over-confs) — just routed through the shared helper.

**Files:**
- Modify: `cutamp/cost_function.py` (`_compute_soft_cost`, the `cost_name == "com_polygon"` branch only)

- [ ] **Step 4.1: Rewrite the soft cost branch**

In `cutamp/cost_function.py`, find the `elif cost_name == "com_polygon":` branch in `_compute_soft_cost` (around line 502). Replace its entire body with:

```python
        elif cost_name == "com_polygon":
            if self._particles is None:
                raise RuntimeError("Particles must be provided for com_polygon soft cost")
            if self._com_polygon_penalties_cache is None:
                from cutamp.com_polygon_cost import compute_com_polygon_penalties
                self._com_polygon_penalties_cache = compute_com_polygon_penalties(
                    self.world, self._particles,
                )
            pens = self._com_polygon_penalties_cache
            if not pens:
                return torch.zeros(num_particles, device=device)
            # Sum over confs to match the prior per-particle scalar shape.
            return torch.stack(list(pens.values()), dim=0).sum(dim=0)
```

- [ ] **Step 4.2: Run the full test file to confirm no regression**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python \
  -m pytest cutamp/tests/test_com_polygon_ik.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 4.3: Quick CLI sanity-check (soft cost still computes)**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 -n 8 --num_opt_steps 5 --disable_visualizer \
  --optimize_soft_costs --soft_cost com_polygon 2>&1 | grep -E "(soft|com_polygon|Loss)" | head -10
```

Expected: at least one line mentioning soft cost computation; no tracebacks. The run will end with 0 satisfying (tol still at 0.5, penalty values O(1e-3); filter is loose) but it must complete cleanly.

- [ ] **Step 4.4: Commit**

```bash
git add cutamp/cost_function.py
git commit -m "refactor: --soft_cost com_polygon shares per-call FK cache

_compute_soft_cost('com_polygon') now reads from the same
_com_polygon_penalties_cache as the hard constraint. When both are
active, FK runs once per conf instead of twice. Result is identical
to the prior code: sum over per-conf penalties, same penalty params
(inside_margin=0.02, inside_weight=1.0)."
```

---

## Task 5 — Recalibrate tolerance and add multiplier

**Why:** Activates the hard filter at the correct threshold (`4e-4` = at-edge penalty) and sets the COM gradient strength on par with KinematicConstraint pos_err. Both are required to make Adam meaningfully pull particles toward COM-feasibility — Task 3+4 wired the gradient through, but without the right tolerance the filter doesn't filter, and without the right multiplier the gradient is ~15× weaker than pos_err.

**Files:**
- Modify: `cutamp/scripts/utils.py` (`default_constraint_to_tol`, `default_constraint_to_mult`)
- Modify: `cutamp/tests/test_com_polygon_ik.py` (`test_constraint_checker_filters_com_violators`)

- [ ] **Step 5.1: Update tolerance and add multiplier in `scripts/utils.py`**

In `cutamp/scripts/utils.py`, find the ComPolygon entry in `default_constraint_to_tol` (added at commit `b2ce8ab`, near the end of the dict):

Before:
```python
    # ComPolygon per-conf values are 0 (inside) or 1 (outside); tol 0.5
    # accepts inside, rejects outside. "default" applies to every conf name
    # (left_q0, right_q1, ...) since they're plan-skeleton-dependent.
    ComPolygon.type: {"default": 0.5},
}
```

After:
```python
    # ComPolygon per-conf values are now continuous penalty in m² from
    # compute_com_polygon_penalties (inside_margin=0.02, inside_weight=1.0):
    # at the polygon edge, penalty = inside_weight * inside_margin² = 4e-4.
    # tol 4e-4 ⇒ "at the edge or inside" satisfies; any excursion past
    # the edge fails (1mm outside has penalty ≈ 4.4e-4). "default"
    # applies to every per-conf name (left_q0, right_q1, ...).
    ComPolygon.type: {"default": 4e-4},
}
```

In the same file, find `default_constraint_to_mult` (top of file). Add a ComPolygon entry. The existing dict ends with:
```python
    "soft": {
        ...
        "com_polygon": 10,  # Penalize COM projection outside the base rectangle
    },
}
```

Insert the ComPolygon entry above the `"soft":` block:
```python
default_constraint_to_mult = {
    KinematicConstraint.type: {"pos_err": 1.0, "rot_err": 5.0},
    StablePlacement.type: {"goal_support": 2.0},
    TrajectoryLength.type: {"traj_length": 1e-3},
    # ComPolygon penalty values are O(1e-4 to 1e-3) in m²; multiplier 10
    # puts the COM gradient contribution on par with KinematicConstraint
    # pos_err (mult 1.0 × value O(1e-2)). "default" applies to every
    # per-conf name — relies on CostReducer's (type, "default") fallback.
    ComPolygon.type: {"default": 10.0},
    "soft": {
        ...
        "com_polygon": 10,
    },
}
```

- [ ] **Step 5.2: Update the filter test to use realistic penalty values**

In `cutamp/tests/test_com_polygon_ik.py`, the values `0.0` and `1.0` in `test_constraint_checker_filters_com_violators` still work (1.0 > 4e-4 fails, 0.0 passes), but the test should reflect penalty-style values for clarity. Replace `1.0` with `1e-2` in the cost_dict literal:

Before:
```python
            "values": {
                "left_q1":  torch.tensor([0.0, 1.0, 0.0, 0.0]),
                "right_q3": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "left_q0":  torch.tensor([0.0, 0.0, 0.0, 0.0]),
            },
```

After:
```python
            "values": {
                # Penalty values: 0.0 = inside polygon; 1e-2 ≈ 1cm outside.
                # tol from default_constraint_to_tol[ComPolygon.type] = 4e-4.
                "left_q1":  torch.tensor([0.0, 1e-2, 0.0, 0.0]),
                "right_q3": torch.tensor([0.0, 0.0, 0.0, 1e-2]),
                "left_q0":  torch.tensor([0.0, 0.0, 0.0, 0.0]),
            },
```

- [ ] **Step 5.3: Run all ComPolygon tests to verify nothing breaks**

Run:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yoonwoo/miniconda3/envs/tamp/bin/python \
  -m pytest cutamp/tests/test_com_polygon_ik.py cutamp/tests/test_cost_reduction.py -v
```

Expected: all tests PASS. The mask-vs-penalty parity test from Task 2 specifically validates tol=4e-4 against the polygon-membership definition.

- [ ] **Step 5.4: Commit**

```bash
git add cutamp/scripts/utils.py cutamp/tests/test_com_polygon_ik.py
git commit -m "feat: recalibrate ComPolygon tolerance to 4e-4 and add mult=10

Tolerance 0.5 was correct for the prior 0/1 Boolean values; now that
values are continuous penalties from compute_com_polygon_penalties
(units m²), the natural at-edge threshold is
inside_weight * inside_margin² = 4e-4. Multiplier 10 puts the COM
gradient contribution to Adam's loss on par with KinematicConstraint
pos_err. Update the synthetic filter test to use realistic penalty
values for clarity."
```

---

## Task 6 — Smoke verification

**Why:** End-to-end checks that the full pipeline behaves as the spec predicts. The unit tests confirm differentiability, parity, and CostReducer fallback in isolation; this task confirms the integration doesn't have surprises.

**Files:**
- No code changes. Two CLI invocations + result interpretation.

- [ ] **Step 6.1: No-regression smoke (`--no_enable_com_polygon`)**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 -n 64 --num_opt_steps 50 --motion_plan --disable_visualizer \
  --no_enable_com_polygon 2>&1 | grep -E "(satisfying after|Total num satisfying|Best cost|Motion plan)" | tail -10
```

Pass: at least one `Opt N` reports `≥1/64 satisfying after optimization`; `Total num satisfying ≥ 1`. Same baseline as commits prior to this PR. (ComPolygon is disabled, so this run is fully independent of the new code paths.)

- [ ] **Step 6.2: Differentiable-ComPolygon smoke (the prior failing case)**

Run:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 -n 128 --num_opt_steps 200 --motion_plan \
  --optimize_soft_costs --soft_cost com_polygon 2>&1 | tee /tmp/diff_compolygon_smoke.log | tail -80
```

After the run completes, inspect the trace with these targeted greps:

```bash
grep -E "Loss:.*satisfying" /tmp/diff_compolygon_smoke.log | head -1   # initial loss
grep -E "Loss:.*satisfying" /tmp/diff_compolygon_smoke.log | tail -1   # final loss
grep -E "satisfying after optimization" /tmp/diff_compolygon_smoke.log
grep -E "\[ComPolygon\] left_q1" /tmp/diff_compolygon_smoke.log
grep -E "\[ComPolygon\] right_q1" /tmp/diff_compolygon_smoke.log
grep -E "Total num satisfying" /tmp/diff_compolygon_smoke.log
```

Pass criteria (relative to the prior failing run dumped at session start):
- **Loss trajectory does NOT explode 17×.** Prior run: 3.3 → 57. Expected after this PR: loss starts at a different (smaller) magnitude (penalty values O(1e-3) × mult 10 = ~1e-2 per violating conf, vs prior 1.0 per violating conf). Final loss should be of similar order to initial, not 17×.
- **At least one `Opt N` reports `≥1/128 satisfying`** (vs prior 0/128 across all 4 plans).
- **Per-conf `[ComPolygon] left_q1 <= 4e-4` counts improve during Adam** (vs prior 0/128 unchanged across 200 steps).

If pass criteria are NOT met but the smoke completes without tracebacks, this fix has removed the necessary blocker (no gradient) but the orthogonal Adam-first-step-destruction issue still dominates. Document the result; the next PR in line is the `skip_adam` adjustment (out of scope here).

- [ ] **Step 6.3: Record outcomes in the spec doc**

Append to `docs/superpowers/specs/2026-05-27-differentiable-com-polygon-design.md` under a new "Implementation outcome" section:

```markdown
## Implementation outcome

(Filled in after Task 6 smoke runs)

- No-regression smoke (`--no_enable_com_polygon`): pass / fail
- Differentiable smoke (n=128, 200 steps, --optimize_soft_costs --soft_cost com_polygon):
  - Initial loss: ...
  - Final loss:   ...
  - Best `≥X/128 satisfying after optimization` across plans: X
  - `[ComPolygon] left_q1` final count: X/128 (vs 0/128 pre-PR)
  - Total num satisfying: X
- If gradient alone was sufficient: PR complete. Else: follow-up PR for `skip_adam`/`conf_lr` is the next step.
```

- [ ] **Step 6.4: Commit verification outcome**

```bash
git add docs/superpowers/specs/2026-05-27-differentiable-com-polygon-design.md
git commit -m "docs: record differentiable-ComPolygon smoke verification outcome"
```

---

## Self-review notes (writer's pass)

**Spec coverage:** Every section of the spec maps to a task —
- "Penalty formulation" → Task 2 (helper using `com_polygon_penalty`)
- "Shared computation helper" → Task 2 (helper definition) + Task 3 (cache field) + Task 4 (soft cost uses cache)
- "Hard constraint rewrite" → Task 3
- "Soft cost rewrite" → Task 4
- "Tolerance" → Task 5
- "Multiplier (cost reducer)" → Task 5
- "CostReducer `default` fallback" → Task 1
- "Files" → covered across Tasks 1–5
- "Verification 1 (filter test)" → Task 3.3 / Task 5.3
- "Verification 2 (differentiability)" → Task 2.1
- "Verification 3 (no-regression smoke)" → Task 6.1
- "Verification 4 (diff smoke)" → Task 6.2
- "Verification 5 (parity)" → Task 2.1 (parity test)

**Type consistency:** `compute_com_polygon_penalties` signature is identical across Tasks 2 (definition), 3 (hard call site), and 4 (soft call site). Cache field name `_com_polygon_penalties_cache` consistent across Tasks 3 and 4.

**No placeholders:** All code blocks contain complete content. All commands are runnable as-given.
