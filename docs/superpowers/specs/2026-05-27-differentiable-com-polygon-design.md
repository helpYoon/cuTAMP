# Differentiable ComPolygon Hard Constraint — Design

**Status:** Proposed
**Date:** 2026-05-27
**Author:** yoonwoo (with Claude)

## Context

ComPolygon was registered as a hard constraint at commit `b2ce8ab` (`feat: enforce COM-over-base-polygon as a hard constraint`), mirroring the wiring pattern of Collision. The values it produces are `(~mask).float()` — 0.0 (COM inside polygon) or 1.0 (outside) — derived from a Boolean tensor.

A diagnostic run (`n=128`, 200 Adam steps, `--optimize_soft_costs --soft_cost com_polygon`) showed Adam destroying every other constraint:

| Constraint | Pre-Adam | Post-Adam |
| --- | --- | --- |
| KinematicConstraint pos_err | 64/128 satisfying | 0/128 |
| Motion joint_limit | 103/128 | 0/128 |
| Motion self_collision | 128/128 | 0/128 |
| ComPolygon (any conf) | 0/128 at `left_q1` | 0/128 at `left_q1` |

Three causes interact:
1. **No gradient.** `(~mask).float()` is a Boolean→float cast — no autograd connection. Adam never sees a COM-correcting gradient.
2. **`skip_adam` bypass.** `optimize_plan.py:378` skips Adam when IK already produced satisfying particles. With ComPolygon zeroing the IK-init satisfying count, the guard turned off and Adam ran on near-optimal IK initializations.
3. **Adam first-step destruction.** Bias-corrected first step is `lr · sign(g)` regardless of `|g|`. Tiny gradients (noise level at IK optimum) still produce `lr`-sized steps. `conf_lr = 2.226e-2 rad ≈ 1.27°/step`; cumulative drift over 5–10 steps is large enough to ruin sub-mm IK solutions.

This design addresses cause (1) — making ComPolygon's values differentiable so Adam has a real COM-feasibility gradient when it does run. Causes (2) and (3) are out of scope (separate spec).

## Goal

Replace the non-differentiable hard-constraint values with the existing continuous `com_polygon_penalty()`. Share the per-conf computation with the existing `--soft_cost com_polygon` path so a single FK pass per conf serves both. Preserve the hard-filter semantics in `ConstraintChecker.get_mask`.

## Architecture

### Penalty formulation

Reuse `com_polygon_penalty(com_world, base_T, half_extents, inside_margin, inside_weight)` from `cutamp/com_polygon_cost.py:63`. It is already differentiable through the cuRobo FK chain.

Parameters (matching the existing soft cost):
- `half_extents = [0.1115, 0.156]` — two-foot support hull from `actual_robot.urdf`.
- `inside_margin = 0.02` m — barrier band starts 2 cm inside the edge.
- `inside_weight = 1.0` — barrier same magnitude as outside-quadratic.

Penalty shape (per particle per conf):
- Center of polygon: `penalty = 0`
- Within `inside_margin` of edge: `penalty ∈ (0, inside_weight · inside_margin²]`
- At the edge: `penalty = inside_weight · inside_margin² = 4 × 10⁻⁴` m²
- 1 mm outside: `penalty ≈ 4.4 × 10⁻⁴` m² (strictly greater than at-edge)
- Beyond: strictly monotonically increasing in distance from polygon

Inside-barrier behavior chosen so Adam pulls COM toward center even for already-feasible particles (`Pull toward center always` decision during brainstorming).

### Shared computation helper

New top-level helper in `cutamp/com_polygon_cost.py`:

```python
def compute_com_polygon_penalties(
    world,
    particles: Dict[str, torch.Tensor],
    *,
    half_extents: Optional[List[float]] = None,
    inside_margin: float = 0.02,
    inside_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Per-conf differentiable COM-polygon penalty.

    For every Conf-shaped particle (name starts with "left_q"/"right_q"/"q",
    ndim == 2), runs world.kinematics_with_com once and computes the
    COM-over-polygon penalty via com_polygon_penalty. Returns a dict mapping
    conf name to a [B] tensor of penalties in m². Differentiable through
    cuRobo's FK chain.
    """
```

Defaults: `half_extents=[0.1115, 0.156]` (set in body if None, same as `compute_com_polygon_mask`).

Both `cost_function.com_polygon_constraint()` and `cost_function._compute_soft_cost("com_polygon")` call this helper. When both are active in the same `__call__` (hard constraint always on with `enable_com_polygon`, soft cost on with `--soft_cost com_polygon`), the helper runs **once** and the result is shared — no double FK pass.

### Hard constraint rewrite

`cost_function.com_polygon_constraint()` becomes:

```python
def com_polygon_constraint(self) -> Union[dict, None]:
    if not self.config.enable_com_polygon or self._particles is None:
        return None
    if self._com_polygon_penalties_cache is None:
        from cutamp.com_polygon_cost import compute_com_polygon_penalties
        self._com_polygon_penalties_cache = compute_com_polygon_penalties(
            self.world, self._particles,
        )
    if not self._com_polygon_penalties_cache:
        return None
    return {
        "type": "constraint",
        "constraints": [],
        "values": dict(self._com_polygon_penalties_cache),  # shallow copy
    }
```

`_com_polygon_penalties_cache` is set to `None` at the top of `__call__` (alongside `self._particles = particles`) and lazily populated on first use. Soft cost path checks the same cache.

### Soft cost rewrite

`_compute_soft_cost("com_polygon")` becomes:

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
    return torch.stack(list(pens.values()), dim=0).sum(dim=0)
```

Cache layer ensures shared computation without coupling the two methods' bodies.

### Tolerance

`cutamp/scripts/utils.py`:
```python
default_constraint_to_tol[ComPolygon.type] = {"default": 4e-4}
```

Was `0.5` (correct for the old 0/1 values). New value `4e-4` matches the at-edge penalty: particles with penalty ≤ 4e-4 satisfy (inside or exactly on the edge); particles even slightly outside fail.

Float32 stability: `(0.02)² = 4 × 10⁻⁴` is exactly representable in float32, so the boundary comparison `penalty <= 4e-4` is numerically stable at the edge.

### Multiplier (cost reducer)

`cutamp/scripts/utils.py`:
```python
default_constraint_to_mult[ComPolygon.type] = {"default": 10.0}
```

Brings COM gradient strength into parity with KinematicConstraint pos_err. Reasoning:
- KinematicConstraint pos_err (mult 1.0) at 1 cm error contributes `0.01` to loss.
- ComPolygon penalty at 1 cm outside is `~1 × 10⁻³` m².
- Multiplier 10 makes the COM contribution `~1 × 10⁻²` — same magnitude as pos_err.

Matches the existing `default_constraint_to_mult["soft"]["com_polygon"] = 10` weight (which was tuned for the soft cost path).

### CostReducer `"default"` fallback

`cutamp/cost_reduction.py:_get_multiplier` currently:
```python
def _get_multiplier(self, cost_type: str, name: str) -> Optional[float]:
    return self.cost_to_multiplier.get((cost_type, name))
```

Extended to:
```python
def _get_multiplier(self, cost_type: str, name: str) -> Optional[float]:
    direct = self.cost_to_multiplier.get((cost_type, name))
    if direct is not None:
        return direct
    return self.cost_to_multiplier.get((cost_type, "default"))
```

Mirrors `ConstraintChecker._get_tol`'s existing fallback semantics. Backwards-compatible — no current entry uses `"default"` as the name field (verified by reading `default_constraint_to_mult`), so behavior of existing constraints is unchanged.

Required because hard ComPolygon values are keyed by conf name (`left_q0`, `right_q1`, etc.) which is plan-skeleton-dependent and cannot be enumerated statically.

## Data flow

```
particles (Dict[name, Tensor])
    │
    ▼
compute_com_polygon_penalties(world, particles)
    │   ← world.kinematics_with_com.compute_kinematics  (per conf, differentiable)
    │   ← com_polygon_penalty                            (per conf)
    ▼
{conf_name: [B] penalty tensor}    ─── cached on CostFunction._com_polygon_penalties_cache
    │
    ├─► com_polygon_constraint() → {"type": "constraint", "values": pens}
    │       │
    │       ▼
    │   cost_dict[ComPolygon.type] = {"type": "constraint", "values": {...}}
    │       │
    │       ├─► ConstraintChecker.get_mask: per-conf values ≤ 4e-4 → AND into satisfying mask
    │       └─► CostReducer (consider_types={"constraint"}):
    │               values × 10.0 (via "default" fallback) → summed into Adam loss
    │
    └─► _compute_soft_cost("com_polygon") → sum over confs → [B] scalar
            │
            ▼
        cost_dict["soft"]["values"]["com_polygon"]
            │
            ├─► CostReducer (consider_types={"cost"}): used by Phase 2 LBFGS
            └─► (excluded from Adam unless upstream_style or coupled_reik)
```

## Files

| File | Change | Approx LOC |
| --- | --- | --- |
| `cutamp/com_polygon_cost.py` | Add `compute_com_polygon_penalties` helper | +30 |
| `cutamp/cost_function.py` | Add `_com_polygon_penalties_cache` field; rewrite `com_polygon_constraint`; rewrite `_compute_soft_cost("com_polygon")` branch | +15 / −20 |
| `cutamp/scripts/utils.py` | Tolerance 0.5 → 4e-4; add `default_constraint_to_mult[ComPolygon.type]` | +3 / −1 |
| `cutamp/cost_reduction.py` | `_get_multiplier` `"default"` fallback | +3 |
| `cutamp/tests/test_com_polygon_ik.py` | Update `test_constraint_checker_filters_com_violators` to use penalty-style values; add `test_compute_com_polygon_penalties_is_differentiable` | +25 / −10 |

Net: ~80 LOC of focused changes.

## Verification

1. **`test_constraint_checker_filters_com_violators` (updated)**: synthesize a per-conf dict with values `{0.0, 1e-2, 0.0, 1e-2}` (small inside-band values for the "passing" particles, large penalty for "failing"). Assert mask is `[True, False, True, False]` with tol `4e-4`.

2. **`test_compute_com_polygon_penalties_is_differentiable` (new)**: feed the helper a particle dict with one Conf parameter (`left_q1 = [B, full_dof]` with `requires_grad=True`). Assert the returned tensor has `requires_grad=True`; call `result.sum().backward()`; assert input has non-zero `.grad`. CUDA-gated like the rest of the file.

3. **No-regression smoke** (`--no_enable_com_polygon`):
   ```bash
   PYTORCH_ALLOC_CONF=expandable_segments:True \
     /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
     --env blocks_t1 -n 64 --num_opt_steps 50 --motion_plan --disable_visualizer \
     --no_enable_com_polygon
   ```
   Pass: same satisfying count as before this PR.

4. **Differentiable-ComPolygon smoke** (the failing case from the diagnosis):
   ```bash
   PYTORCH_ALLOC_CONF=expandable_segments:True \
     /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
     --env blocks_t1 -n 128 --num_opt_steps 200 --motion_plan \
     --optimize_soft_costs --soft_cost com_polygon \
     --save_plan /home/yoonwoo/cuTAMP/data/motion_plan.pkl
   ```
   Pass criteria (relative to the run dumped in the diagnostic):
   - `Loss:` displayed during Adam does NOT explode 17× (3 → 57). Should grow at most modestly (say, ≤ 3×) or decrease.
   - Per-conf `[ComPolygon] left_q1 <= 4e-4 has X/128` count should improve during Adam, not stay at 0.
   - At least one `Opt N` reports `≥1/128 satisfying after optimization` (vs the diagnostic's 0/128 across all 4 plans).

5. **`compute_com_polygon_penalties` parity test** (new, optional): for a synthetic particle batch where COM is computed both via `compute_com_polygon_mask` (Bool) and `compute_com_polygon_penalties` (float), assert that `(penalties <= 4e-4) == mask` for all particles. Verifies tolerance is set correctly.

## Out of scope (explicit non-goals)

These are diagnosed problems that are NOT addressed by this design:

- **`skip_adam` logic**: Adam should arguably skip when all *other* constraints satisfy, regardless of ComPolygon. Even with a perfect gradient, Adam's first-step destruction can ruin near-optimal IK initializations. Tracked separately.
- **Lowering `conf_lr`**: cumulative drift is the deeper issue. Separate tuning concern.
- **Reformulating ComPolygon as outside-only** (no inside barrier): brainstorming explicitly chose `Pull toward center always`.
- **Deprecating `--soft_cost com_polygon`**: chose to keep both with shared computation.

## Risk

- **CostReducer fallback breaking existing constraints**: Mitigated. Grep of `default_constraint_to_mult` shows no entry currently uses `"default"` as a name field. New fallback only activates for `(cost_type, "default")` lookups when the exact `(cost_type, name)` is absent. All existing entries provide exact name lookups, so they take precedence.
- **Tolerance set wrong relative to penalty magnitudes**: Mitigated by parity test (verification 5). If `(penalties <= 4e-4) == mask` holds for representative particles, tolerance is correct.
- **Cache invalidation**: Cache is keyed on `self._particles` identity, reset to `None` at the top of each `CostFunction.__call__`. If `_compute_soft_cost` is called outside `__call__` (it shouldn't be — `_compute_soft_cost` is `self.`-prefixed and called only from `soft_costs()` which is called only from `__call__`), the cache could be stale. No current call path bypasses `__call__`, so the invariant holds.
- **Adam first-step destruction persists**: Acknowledged in "Out of scope". This fix alone may not get the smoke test to "≥1 satisfying" — it removes a necessary blocker (no gradient) but the lr/destruction issue may still dominate. Verification 4 will reveal whether this fix alone is sufficient.
