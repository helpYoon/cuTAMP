# CoM-aware IK (seed-IK LM residual) — design

**Date:** 2026-06-09
**Status:** adversarially reviewed (6 claim-verifiers + 2 cleanup mappers against code) — pending user approval
**Branch:** `main` (local)

## Problem

When the T1 reaches forward to pick or place, the IK that produces the endpoint
configuration is COM-blind: it resolves body redundancy the laziest way (hip fold,
straight legs), leaving the center of mass at the very edge of the support polygon
(audited across 10 plans: arrivals up to 99% of the tipping tolerance `COM_TOL`).
Every existing protection is a pass/fail wall (hard `ComPolygon` gate, post-IK
mask+retry, COM-anchored endpoints): they guarantee *inside* but never pull *toward*
stable. The robot CAN stand stably for these reaches — bending knee/ankle within
real joint limits centers the COM (measured: knee gives −0.072 m COM-x per +0.3 rad,
with 2.34 rad of range) — but no solver ever asks it to. The previously attempted
remedies fail structurally: the IK soft cost via `add_extra_cost` is inert (see
"Why previous attempts failed"), and post-IK null-space refinement cannot escape
the hip-fold basin within grasp tolerance at far reach.

**Goal:** make IK itself trade "reach the hand pose" against "keep the COM well
inside the support region," recruiting the legs within joint limits, so pick/place
endpoint confs come out centered-with-margin — without regressing plan success,
grasp accuracy (5 mm), placement accuracy (1 mm xy), or any hard guarantee.

**Objective shape (user-chosen hybrid):** penalize outside the support rectangle +
inside-margin barrier near the edge (hard-margin behavior), plus a small
pull-to-center term active everywhere inside (tiebreak).

## Literature backing

The method is canonical whole-body IK: CoM (COG) Jacobian as an IK task
(Sugihara & Nakamura, IROS 2002/2003); LM IK with robust damping (Sugihara, T-RO
2011 — cuRobo's seed-IK *is* this solver family); CoM-inside-support-polygon as an
inequality task (Kanoun, Lamiraux, Wieber, T-RO 2011); CoM + postural objectives
stacked into a single LM residual (HJCD-IK, 2025; Stack-of-Tasks). cuRobo itself
ships an orphan `CostSupportPolygon` (referenced nowhere) — precedent that this
belongs in the solver. We use a *weighted* (soft-priority) stack, the common form;
strict-hierarchy HQP (Escande et al., IJRR 2014) is the documented escalation.

## Where the IK actually computes cost (investigation result)

cuRobo v0.8 `solve_pose` = **seed-IK LM stage** (`curobo/_src/solver/seed_ik/`)
followed by a **conditional MPPI→LBFGS refinement stage**:

- Seed stage: `SeedIKSolver._optimize` — pure LM; residuals computed in Python by
  `SeedIKErrorCalculator` (pose + joint-limit + optional velocity/accel blocks),
  combined in `_combine_errors` (jTerror summed, jacobian rows concatenated,
  error_norm summed). The LM kernel solves `(JᵀJ + λI)δ = −jTerror` with `jacobian`
  and `jTerror` supplied independently.
- Early exit (`exit_early=True`, `exit_early_batch_success_threshold=1.0` defaults):
  if ALL problems in the (padded-to-64) batch meet pose tolerance + feasibility at
  the metrics check, seed solutions are returned directly (the regime observed in
  all our probes). Otherwise `MultiStageOptimizer` (MPPI→LBFGS) runs, seeded by LM
  results, and the *refined* solutions are returned.

**Why previous attempts failed:** `add_extra_cost` wraps
`RobotCostManager.compute_costs`, but the seed-IK error calculator never calls the
cost managers on the optimization path (verified: 0 calls during a solve; only the
final metrics pass calls it once). The IK soft COM cost was therefore inert. This
supersedes the earlier "fused CUDA LBFGS" explanation in project memory.

## Design

### Fork edits (vendored cuRobo — Python only, no CUDA; tracked as fork edits per the no-upstream-push constraint)

1. **`seed_ik_solver_cfg.py`** — add fields (defaults = off, byte-identical):
   `com_support_weight: float = 0.0`, `com_half_extents: Optional[List[float]] = None`,
   `com_inside_margin: float = 0.02`, `com_inside_weight: float = 1.0`,
   `com_center_weight: float = 0.0`, `com_base_link_name: str = "mobile_base_link"`.

2. **`seed_ik_solver.py`** — construct the LM robot model with COM when enabled:
   `Kinematics(robot_model_config, compute_spheres=False, compute_jacobian=True,
   compute_com=(config.com_support_weight > 0))`. **Verified necessary:** seed-IK
   builds its own Kinematics (`seed_ik_solver.py:82-87`) ignoring the transition
   dict; today its `robot_com` is allocated but **all zeros** (probed:
   `max|robot_com| = 0.0`, `compute_com=False`). Thread the `com_*` cfg into
   `SeedIKErrorCalculator`.

3. **`seed_ik_error_calculator.py`** — CoM term inside the pose block (NOT an
   independent stream block):
   - In `_compute_pose_errors` (shares its FK and stream): when
     `com_support_weight > 0`, read `kin_state.robot_com[..., :3]` and the
     `com_base_link_name` pose from `kin_state.tool_poses`; evaluate cuTAMP's
     `com_polygon_penalty` (rectangle outside-quadratic + inside-margin max-over-axes
     barrier + `com_center_weight` pull-to-center; function re-gains the optional
     `center_weight` parameter, default 0, so gate semantics are untouched).
   - **Single combined backward** replacing the pose-only backward:
     `torch.autograd.backward([pose_cost, w*com_cost], [cost_shape, com_ones])` —
     `joint_position.grad` then carries pose + CoM jTerror together. A second
     separate backward is **forbidden** (freed graph / fused-kinematics grad-buffer
     aliasing). `com_ones` preallocated next to `_cost_shape`.
   - Add `w*com_cost` into the returned `error_norm` so LM trust-ratio/step
     acceptance accounts for it.
   - **v1 residual convention: rhs-only (zero jacobian rows).** The CoM term steers
     via `−jTerror` under λ-damping; `jacobian`, `_calculate_n_residuals`, and the
     `(dof, n_residuals)`-templated LM kernel are untouched — no respecialization,
     trivially byte-identical when off. Documented upgrade if leg recruitment needs
     curvature: sqrt-residual GN row (`r=√(w·c)`, row=`w∇c/(2√(w·c))`) + row-count
     update in `_calculate_n_residuals` gated on the same `weight>0` condition
     (mirror the velocity/accel gating, NOT joint-limit's always-concat mismatch).

### cuTAMP edits

4. **`cutamp/com_polygon_cost.py`** — re-add optional `center_weight: float = 0.0`
   parameter to `com_polygon_penalty` (used only by the IK residual; all gate/mask
   callers stay at 0 → `COM_TOL` semantics unchanged).
5. **`cutamp/robots/t1.py::get_t1_ik_solver`** — new kwarg
   `enable_com_aware_ik: bool`; when True, set the `com_*` cfg fields from the
   `COM_*` constants (single source of truth with the hard gate) + a starting
   `com_support_weight` (see tuning). Delete the inert
   `add_extra_cost(ik_solver, "com_polygon", ...)` block.
6. **`cutamp/config.py` / `tamp_world.py` / `run_cutamp.py`** — thread
   `enable_com_aware_ik` (default **False** until validated; flip to True after the
   acceptance gates pass). `enable_com_polygon` remains the umbrella for planner
   cost + hard gate.
7. **Close an adjacent hole found in review:**
   `_ik_solution_to_full_q` stores IK solutions without success-masking — confs
   flagged `success=False` (which may be out-of-limit: limits are soft residual +
   filter, nothing clamps) can silently enter particles. Gate stored q on
   `ik_result.success` (keep prior/home value for failures).
   Also fix-or-remove the dormant `num_seeds` kwarg in `_ik_for_pose`
   (`solve_pose` has no such parameter → latent TypeError).

### Fallback policy (refinement stage is CoM-blind)

When the seed stage misses 100% batch success, MPPI→LBFGS refinement runs without
the CoM term and returns refined solutions. **v1 policy: accept partial coverage.**
The retained Layer-2 mask+retry and the hard `ComPolygon` gate catch CoM-out refined
solutions exactly as today. Instrument and log fallback engagement (early-exit gate
fail count) in the A/B validation; if it engages often with the flag on (the CoM
trade pushing pose past tolerance is a self-triggering risk), first lower
`com_support_weight`; escalate options: `run_optimizer=False` (seed-only IK) or a
CoM term in the LBFGS rollout cost (larger fork).

### Success semantics & guarantees (unchanged)

IK success remains pose-tolerance + feasibility only — the CoM term shapes the
solution basin but can never gate success, and collision stays a post-solve
feasibility filter (verified untouched). The hard `ComPolygon` gate, post-IK
mask+retry, COM-anchored endpoints, planner mid-trajectory COM cost, and Adam-side
soft cost all remain the enforcement layers.

### Weight tuning protocol

`joint_limit_weight=1.0` and pose weights are O(1); the planner-cost weight (5e5)
is NOT transferable — an overpowering CoM term would trade pose/limits and collapse
IK success. Sweep `com_support_weight ∈ {0.1, 0.3, 1.0, 3.0}` on the forward-reach
A/B grid; pick the largest weight with: no IK success drop, no fallback-engagement
increase, pose errors within tolerance. Metrics per setting: mean/max `|com_x|`,
in-hull fraction, success&in-hull count, leg recruitment (knee/ankle deltas),
fallback engagements. Start `com_center_weight` at 0; enable last, smallest.

### CUDA-graph safety (recorded for future `use_cuda_graph=True`)

Our config runs the Python error path eagerly (verified: `use_cuda_graph=False`
propagates; executors never built). The residual must still be capture-safe:
config-scalar branching only (`if weight > 0` fixed at construction); no
tensor-value branching, `.item()`, `.cpu()`, prints, or syncs in the path
(branch-free `clamp`/`where` math — `com_polygon_penalty` already is); static
shapes sized in `setup_batch_tensors`; support-rectangle params as preallocated
device buffers updated via `copy_`; no new positional inputs to
`_compute_pose_error_and_jacobian` (GraphExecutor copies the exact captured tuple).

## Validity review (adversarial, against code)

| claim | verdict | consequence folded into design |
|---|---|---|
| Seed-IK LM is the whole solve | **partial** | MPPI→LBFGS fallback live iff seed batch <100% success → fallback policy section |
| Residual blocks stack cleanly | **confirmed** | v1 rhs-only convention; error_norm must carry the CoM scalar; success is pose-only |
| FK/backward sharable, robot_com available | **partial** | robot_com is ZEROS today → fork edit 2; single-backward requirement; in-pose-block placement |
| Joint limits enforced | **partial** | soft residual + filters, nothing clamps → weight balance + success-masking fix (edit 7) |
| Collision orthogonal | **confirmed** | post-solve filter untouched; two-sided success-rate risk → instrument A/B |
| CUDA-graph safe in our config | **confirmed** | eager today; capture-safety requirements recorded |

## Testing

New `cutamp/tests/test_com_aware_seed_ik.py`:
1. **Off-by-default byte-identical**: default cfg has `com_support_weight==0`;
   monkeypatch-count the residual (0 calls, flag off); fixed-seed A/B
   `torch.equal` on `solution` and `success`.
2. **Residual correctness**: cost/jTerror vs `com_polygon_penalty` + autograd
   reference on home + outside-polygon configs; shapes `[B]`, `[B,18]`.
3. **Stacking sanity**: mixed inside/outside batch; total jTerror = pose + CoM
   elementwise; one LM step finite.
4. **Centering A/B (integration)**: forward-reach grid both flags; mean `|com_x|`
   reduced ≥20% (or ≤ `half_x − margin`); in-hull fraction not lower; successful
   solutions stay within pose tolerances.
5. **Leg recruitment within limits**: hard-assert every returned solution within
   URDF bounds; soft-assert knee/ankle actually recruited on far reaches.
6. **Success-rate regression**: success&in-hull counts, 3–5 seeds, mean-based with
   one-target slack (nondeterminism-aware). Full-pipeline `num_satisfying` checked
   manually over repeated runs, not asserted in CI.
Plus: hoist `_free_cuda_between_tests` to `cutamp/tests/conftest.py` (A/B tests
build two worlds); replace the two inert registration tests in
`test_com_polygon_ik.py` with flag-wiring tests; keep all penalty-math, mask/gate
parity, batched-COM-kernel, and motion-anchor tests (the COM-kernel regression test
becomes a direct precondition of the residual).

## Post-implementation cleanup (reviewed map)

- **Delete:** inert IK `add_extra_cost` block (`t1.py:319-342`) + its two
  registration-presence tests (replaced above). Only after the residual is validated.
- **Shrink:** `ik_com_retry_max` 15 → ~3 once A/B confirms retries are rare
  (wrapper itself stays as Layer-2 backstop).
- **Replace:** `enable_com_polygon` threading TAMPWorld→`get_t1_ik_solver` becomes
  `enable_com_aware_ik` (the umbrella flag keeps gating planner cost + hard gate).
- **Keep (each still guards a distinct frontier):** hard ComPolygon gate (endpoint
  guarantee), COM-anchor (terminal = gated conf), planner soft cost (mid-trajectory),
  Adam soft cost (particle optimization), `_curobo_internals.add_extra_cost`
  (planner still uses it; docstring's IK claim removed), `_ik_transition_dict_with_compute_com`
  (other IK consumers; docstring corrected — it does NOT reach seed-IK).
- **Doc sweep:** every "two-layer / Layer 1 / convergence basin" comment that claims
  the IK rollout cost is active (`t1.py`, `particle_initialization.py`, `config.py`,
  `run_cutamp.py` help, `test_com_polygon_ik.py` header) → rewritten to reference
  the seed-IK residual. Project memory `project_ik_com_cost_inert.md` updated with
  the corrected mechanism (seed-IK LM bypasses cost managers; not "fused LBFGS").

## Risks

1. **Soft priority:** a weighted stack cannot guarantee pose dominance — mitigated
   by barrier shape (≈0 when comfortably inside), conservative weight, pose-only
   success criteria, and the retained hard gate. HQP is the literature escalation.
2. **Self-triggering fallback:** CoM pull degrading pose convergence triggers the
   CoM-blind refinement — mitigated by weight tuning + engagement instrumentation.
3. **Collision funneling:** centering may steer legs/torso into self-collision pairs;
   only the post-filter resists — watch converged-but-infeasible counts in A/B.
4. **Fork maintenance:** three Python files in vendored cuRobo — comment-tagged like
   the existing `compute_com` plumbing edits; diffable on upgrade; never pushed
   upstream (fork-only per project constraint).
5. **Far reaches stay task-limited:** at full arm extension with hip at its limit,
   no in-limit posture is both on-pose and centered; those confs remain edge-adjacent
   and the walls keep them safe. The residual fixes the *recruitable* cases.
