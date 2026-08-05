# Standard-cuTAMP baseline: 2×2 ablation of the CoM and re-IK subsystems

**Date:** 2026-08-05 · **Branch:** `ablation-baseline` · **Purpose:** paper/report
figures showing, separately, (1) how the support-polygon machinery changes
planning outcomes and (2) how the re-IK refresh subsystem changes them.

## Goal

Run the same domain and problem settings under four conditions forming a 2×2
factorial, where the "neither" cell approximates standard (upstream-mechanism)
cuTAMP on the T1, and produce publication-grade tables and figures with error
bars.

|                  | re-IK ON        | re-IK OFF                        |
|------------------|-----------------|----------------------------------|
| **CoM ON**       | full system     | −reIK ablation                   |
| **CoM OFF**      | −CoM ablation   | standard cuTAMP (both off)       |

Effect of the support polygon = compare rows. Effect of re-IK = compare
columns. Interaction = the fourth cell.

## Conditions as flags (no code deletion)

All four conditions run the same code; a condition is a flag set. New flags
default to today's behavior, so the branch is merge-safe.

New configuration surface:

| flag | default | gates |
|---|---|---|
| `enable_com_hard_constraint` | `True` | registration of the `ComPolygon` hard constraint in the cost function (and therefore the satisfying mask) — layer L3 |
| `enable_com_init_gate` | `True` | particle init uses `_ik_for_pose_com_safe` (CoM gate + escalating retries, layer L2); `False` uses plain `_ik_for_pose` |
| `force_adam` | `False` | disables the `skip_adam` shortcut. Today, with `coupled_reik=False`, Phase 1 is skipped whenever init already satisfies — upstream never skipped optimization; without this fix the baseline arms do no optimization on easy tasks |
| `soft_in_adam` | `"auto"` | `"auto"` = today's rule (soft costs join the Adam loss only when coupled-reik is on); `"always"` = soft costs join regardless. **Decision (a), approved:** −reIK arms run `"always"` — the upstream-faithful attempt whose measured failure is the re-IK evidence |
| `--no_com_all` (CLI umbrella) | — | sets `enable_com_aware_ik=False` (L1), `enable_com_init_gate=False` (L2), `enable_com_hard_constraint=False` (L3), `enable_com_polygon=False` (L5) |

Cell definitions:

- **full**: defaults.
- **−CoM**: `--no_com_all`.
- **−reIK**: `--no_coupled_reik --force_adam --soft_in_adam always`.
- **standard cuTAMP**: `--no_com_all --no_coupled_reik --force_adam --soft_in_adam always`.

Notes:
- L4 (cspace-anchored terminals in motion planning) stays ON in all arms: it is
  the planner interface, not a tunable mechanism.
- The GN projection, warm 4-seed fallback, and L-BFGS-40 budget live inside the
  re-IK subsystem and vanish automatically in −reIK arms. No extra flags.
- `--no_coupled_reik` already restores the upstream mechanism: covered confs
  stay in Adam, coupled to poses by the `KinematicConstraint` penalty.

## Fairness controls

Identical across all arms and cells:

- 512 particles (max-quality setting; runs are sequential, no GPU sharing).
- Early-stop patience 150 + best-snapshot (shared harness; kept in all arms —
  the study attributes effects to the two mechanisms, so all other
  infrastructure must be identical).
- Checker tolerances, `reik_interval=5` (where applicable), `blocks_t1`.
- **Same seed list `[0..4]` in every cell**, via a new `--seed` flag
  (`torch.manual_seed` at run start; drives grasp/placement sampling and retry
  noise). Enables paired per-seed comparisons. cuRobo's internal
  `random_seed=123` stays constant everywhere.

## Evaluation: one yardstick for every arm

Each arm runs by its own rules but is scored identically post-hoc:

1. **Satisfying (own rules)** — the arm's final satisfying count.
2. **CoM-feasible satisfying** — `compute_com_polygon_mask` applied
   measure-only to final particles in every arm. Headline metric for CoM-off
   arms: how many of their "solutions" would tip.
3. **CoM margin** of the selected best particle: min over the two axes of
   (h − |c|), i.e. distance to the nearest rectangle edge in the base frame,
   negative when outside — McGhee & Frank's static stability margin.
4. **Best soft cost** on the satisfying ∧ CoM-feasible subset.
5. **Motion planning** (all runs use `--motion_plan`): success rate, retries
   used, and a trajectory audit — fraction of trajectory frames with CoM
   outside the polygon (exposes the L5 ablation between stable endpoints).
6. **Runtime**: loop seconds, ms/step.

Grid: 4 conditions × 4 surface heights (0.0, 0.25, 0.50, 0.65 m) × 5 seeds =
80 runs ≈ 3 h GPU, sequential. Aggregation: mean ± std per cell + paired
per-seed deltas.

## Runner and analysis

- `cutamp/scripts/run_ablation.py`: iterates cells × heights × seeds, invokes
  the pipeline with the right flags, writes standard experiment dirs plus a
  study manifest (JSON: cell, height, seed, experiment path, exit status).
  Resumable: skips runs whose manifest entry is complete.
- `cutamp/scripts/analyze_ablation.py`: reads manifests + experiment metrics,
  emits `docs/ablation/results.csv`, a markdown table, and matplotlib figures
  (satisfying & CoM-feasible vs height per condition; tipping-fraction bars;
  soft cost; runtime). Style follows `docs/plot_p_com.py`.

## Testing

- Unit: flag plumbing (constraint absent from the cost dict when
  `enable_com_hard_constraint=False`; plain IK path when
  `enable_com_init_gate=False`; `force_adam` runs Phase 1 despite satisfying
  init; `soft_in_adam="always"` adds costs to `consider_types` without
  coupled-reik).
- Smoke: one tiny run per cell (64 particles, 30 steps) asserting completion
  and manifest write.
- Full suite must stay green with defaults (flags default to current behavior).

## Risks / known asymmetries

- −reIK arms with soft-in-Adam may collapse (historically: Adam destroys
  21-DoF configurations through the stiff kinematic coupling). This is the
  expected, reportable outcome of decision (a) — not a bug in the study.
- CoM-off arms report higher "own-rules" satisfying counts because their mask
  has fewer constraints; the CoM-feasible metric is the equalizer and must be
  the headline comparison.
- Per-height difficulty changes init success rates; `force_adam` ensures all
  arms still optimize.

## Out of scope

- Comparing against the literal upstream repo (Panda robot, different domain).
- A stripped-code branch (rejected in favor of flags; revisit only if a
  code-size exhibit is wanted).
- New task domains beyond the height sweep.
