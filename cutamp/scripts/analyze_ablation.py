"""Turn an ablation-study manifest into the paper table and figures.

With ``--gate``, a ``run_ablation --gate`` study's manifest becomes the speed
gate's gate.json and gate.md instead (design section 8).

Usage:
    python -m cutamp.scripts.analyze_ablation \
        --manifest /tmp/ablation-study/manifest.json --out docs/ablation
    python -m cutamp.scripts.analyze_ablation --manifest /tmp/gate/manifest.json --out /tmp/gate --gate
"""
import argparse
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from cutamp.scripts.run_ablation import GATE_N, HEIGHTS, SEEDS

METRICS = [
    "num_satisfying_final",
    "num_satisfying_com_feasible",
    "best_soft_cost_com_feasible",
    "best_com_margin",
    "traj_com_violation_frac",
    "motion_plan_success",
    "found_solution",
    "loop_s",
    "ms_per_step",
    "stable_plan",
    "reik_curobo_calls",
    "adam_gain",
    "restored",
]
CELL_ORDER = ["full", "default_reik", "no_com", "no_reik", "standard"]
# Label arms by what they HAVE, not what was removed — "−CoM" reads as
# "the CoM thing" in figures when it is precisely the arm WITHOUT it.
CELL_LABEL = {
    "full": "Full (CoM + balance-first re-IK)",
    "default_reik": "Full, default re-IK",
    "no_com": "re-IK only (no CoM)",
    "no_reik": "CoM only (no re-IK)",
    "standard": "Standard cuTAMP (neither)",
}
# The top-level timers of algorithm.run_cutamp: their totals sum to a run's planning time (design section 8). Every
# other timer nests inside one of them, and first_solution overlaps start_optimization.
TOP_TIMERS = ("load_tamp_world", "warmup_ik_solver", "get_plan_generator", "sample_initial_plans", "sort_plans",
              "start_optimization")
# The stage times (design section 8) beside ms_per_step, each a timer's total: seed IK, trajopt, the final re-IK
# repair, and the world's load (the timing check's).
STAGE_TIMERS = {"seed_ik_s": "initialize_particles", "trajopt_s": "curobo_planning", "final_repair_s": "final_repair",
                "load_tamp_world_s": "load_tamp_world"}
# The speed gate (design section 8), on run_ablation's GATE_CELLS: decided at its one n (64), the real-leg model passes if
# all GATE_RUNS of its runs find a solution, a motion plan and a balanced plan, and its median and p90 planning times
# are at most GATE_MEDIAN_MAX and GATE_P90_MAX times the same-session simplified model's. Both models need all
# GATE_RUNS runs, each with a planning time: a missing run of either would move the ratios.
GATE_BASELINE, GATE_MODEL = "simplified", "real_legs"
GATE_RUNS = len(HEIGHTS) * len(SEEDS)  # 40 per model and n
GATE_DECIDED_AT = GATE_N[0]
GATE_MEDIAN_MAX, GATE_P90_MAX = 1.5, 2.0
GATE_COUNTS = ("found_solution", "motion_plan_success", "stable_plan")
GATE_STAGES = ("seed_ik_s", "ms_per_step", "trajopt_s", "final_repair_s", "load_tamp_world_s")


def _total_opt_steps(opt_data):
    """Total optimization steps behind ``loop_s``, for ms/step.

    Counts steps actually EXECUTED: ``elapsed`` gains one entry per real
    optimization iteration (including the one that trips early stopping),
    so ``len(elapsed)`` is the true step count. ``num_steps`` is only the
    configured budget (``optimize_plan.py`` stamps it before the loop) —
    dividing by it would systematically understate ms/step whenever
    ``early_stop_patience`` fires, which it does in every real study run.
    A file with ``elapsed=[]`` (a skeleton started after the
    ``--max_duration`` budget) or none ran no step, so it adds none, as it
    adds no time to ``loop_s`` (known-bugs N6).
    """
    return sum(len(o.get("elapsed") or []) for o in opt_data)


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
        opt_data = [json.load(open(f)) for f in opt_files]
        # Multi-skeleton runs write one opt_*.json per attempted plan
        # skeleton, so loop_s (the run's total optimization wall-clock)
        # must sum each file's final `elapsed` entry across all of them,
        # not just read the last one. num_satisfying_final (and the rest
        # of `opt` below) intentionally keeps reading only the LAST
        # file: that's the skeleton the run actually finished on, so its
        # final state is the semantically right one to report per-run.
        elapsed_finals = [o["elapsed"][-1] for o in opt_data if o.get("elapsed")]
        loop_s = sum(elapsed_finals) if elapsed_finals else None
        total_steps = _total_opt_steps(opt_data)
        ms_per_step = (
            1000.0 * loop_s / total_steps if loop_s is not None and total_steps else None
        )
        opt = opt_data[-1]
        overall = {}
        if (exp / "overall_metrics.json").exists():
            overall = json.load(open(exp / "overall_metrics.json"))
        timers = {}
        if (exp / "timer_metrics.json").exists():
            timers = {name: t["total"] for name, t in json.load(open(exp / "timer_metrics.json")).items()}
        # A sum missing a top-level timer would understate the run, so it has none.
        planning_s = sum(timers[k] for k in TOP_TIMERS) if all(k in timers for k in TOP_TIMERS) else None
        failures = overall.get("motion_plan_failures")
        # A balanced plan: a motion plan whose every audited trajectory frame
        # keeps the CoM in the polygon. None when the audit is missing.
        mps, viol = overall.get("motion_plan_success"), overall.get("traj_com_violation_frac")
        stable_plan = None if mps is None or (mps and viol is None) else float(bool(mps) and viol == 0)
        # cuRobo calls during Adam: default-mode refreshes whose GN missed rows.
        # Balance-first logs fallback=False; the final repair isn't logged.
        refreshes = [r for o in opt_data for r in o.get("reik_refresh", [])]
        curobo_calls = sum(bool(r.get("fallback")) for r in refreshes) if refreshes else None
        records.append({
            "cell": run["cell"], "height": run["height"], "seed": run["seed"],
            # The manifest's n (gate studies), else the run's own.
            "n": run.get("n", overall.get("num_particles")),
            "planning_s": planning_s,
            **{k: timers.get(timer) for k, timer in STAGE_TIMERS.items()},
            # Failed motion-plan candidates whose trajectory passed a joint's effort (design 7.4a): trajopt's last
            # result, on its horizon or between its frames (the reason ": torque"), or a segment the balance gate's
            # torque check failed at every time scale or on the frames it returns (its raise names the torque).
            "torque_failures": None if failures is None else sum("torque" in m for m in failures),
            "num_satisfying_final": opt.get("num_satisfying_final"),
            "num_satisfying_com_feasible": opt.get("num_satisfying_com_feasible"),
            "best_soft_cost_com_feasible": opt.get("best_soft_cost_com_feasible"),
            # The planned particle's margin. Runs with no motion plan have only rank 0's, and so do runs
            # before known-bugs N3 (G1), where after a retry rank 0 isn't the planned particle.
            "best_com_margin": overall.get("planned_com_margin", opt.get("best_com_margin")),
            "traj_com_violation_frac": overall.get("traj_com_violation_frac"),
            "motion_plan_success": (
                float(overall["motion_plan_success"])
                if "motion_plan_success" in overall else None
            ),
            "found_solution": (
                float(overall["found_solution"])
                if "found_solution" in overall else None
            ),
            "loop_s": loop_s,
            "ms_per_step": ms_per_step,
            "stable_plan": stable_plan,
            "reik_curobo_calls": curobo_calls,
            # Adam's best count over the IK init, and whether the reported particles are a
            # restored snapshot: gain 0 with restored 1 means the run reports its IK init.
            "adam_gain": opt.get("adam_gain"),
            "restored": None if opt.get("restored") is None else float(opt["restored"]),
        })
    return records


def _percentiles(values, qs):
    """numpy's linear percentiles ``qs`` of ``values``, as floats; Nones when there are no values."""
    return [float(x) for x in np.percentile(values, qs, method="linear")] if values else [None] * len(qs)


def gate_summary(records) -> dict:
    """The speed gate (design section 8) over a ``run_ablation --gate`` study's records.

    ``by_n[n]``, for each particle count n: ``cells``, per model, its runs, how many of them have a planning time
    (``timed``), its counts of found_solution, motion_plan_success and stable_plan, its total torque_failures (failed
    candidates whose trajectory passed an effort), its planning time's median and p90 over the timed runs and its
    median stage times; ``ratio``, the real-leg / simplified ratios of the median and the p90; and ``flags``:
    ``success`` (both models ran GATE_RUNS runs, every one timed, and every real-leg run counts on all three),
    ``median`` (ratio at most GATE_MEDIAN_MAX), ``p90`` (at most GATE_P90_MAX) and ``pass`` (all three). The gate is
    decided at n = ``decided_at``: ``pass`` is that n's flags, or None without runs there."""
    by_n = {}
    for n in sorted({r["n"] for r in records if r["cell"] in (GATE_BASELINE, GATE_MODEL) and r.get("n") is not None}):
        cells = {}
        for cell in (GATE_BASELINE, GATE_MODEL):
            runs = [r for r in records if r["cell"] == cell and r.get("n") == n]
            times = [r["planning_s"] for r in runs if r.get("planning_s") is not None]
            median, p90 = _percentiles(times, [50, 90])
            cells[cell] = {
                "runs": len(runs),
                "timed": len(times),
                **{k: sum(r.get(k) == 1.0 for r in runs) for k in GATE_COUNTS},
                "torque_failures": sum(r.get("torque_failures") or 0 for r in runs),
                "planning_s": {"median": median, "p90": p90},
                "stage_medians": {k: _percentiles([r[k] for r in runs if r.get(k) is not None], [50])[0]
                                  for k in GATE_STAGES},
            }
        base, model = cells[GATE_BASELINE]["planning_s"], cells[GATE_MODEL]["planning_s"]
        ratio = {q: model[q] / base[q] if model[q] is not None and base[q] else None for q in ("median", "p90")}
        real = cells[GATE_MODEL]
        flags = {
            "success": (all(c["runs"] == c["timed"] == GATE_RUNS for c in cells.values())
                        and all(real[k] == GATE_RUNS for k in GATE_COUNTS)),
            "median": ratio["median"] is not None and ratio["median"] <= GATE_MEDIAN_MAX,
            "p90": ratio["p90"] is not None and ratio["p90"] <= GATE_P90_MAX,
        }
        flags["pass"] = all(flags.values())
        by_n[n] = {"cells": cells, "ratio": ratio, "flags": flags}
    decided = by_n.get(GATE_DECIDED_AT)
    return {"decided_at": GATE_DECIDED_AT, "pass": decided["flags"] if decided else None, "by_n": by_n}


def _write_gate_md(summary, path):
    def num(x, digits=2):
        return "—" if x is None else f"{x:.{digits}f}"

    def yes(flag):
        return "yes" if flag else "no"

    def short(cells):
        """``" (not 40 runs: ...)"`` naming each model whose run count is not GATE_RUNS, else ``""``."""
        counts = [f"{cell} {c['runs']}" for cell, c in cells.items() if c["runs"] != GATE_RUNS]
        return f" (not {GATE_RUNS} runs: {', '.join(counts)})" if counts else ""

    decided, at = summary["pass"], summary["decided_at"]
    lines = ["# Speed gate (design section 8)", ""]
    if decided is None:
        lines.append(f"Not decided: no runs at n = {at}.")
    else:
        lines.append(
            f"Decided at n = {at}: **{'PASS' if decided['pass'] else 'FAIL'}**{short(summary['by_n'][at]['cells'])}. "
            f"All {GATE_RUNS} runs of each model timed, and every {GATE_MODEL} run solved, motion-planned and "
            f"balanced: {yes(decided['success'])}; median ratio at most {GATE_MEDIAN_MAX}: {yes(decided['median'])}; "
            f"p90 ratio at most {GATE_P90_MAX}: {yes(decided['p90'])}."
        )
    for n, entry in summary["by_n"].items():
        lines += [
            "", f"## n = {n}", "",
            "| model | runs | timed | solved | motion plan | balanced | failed past an effort | planning median [s] "
            "| planning p90 [s] | seed IK [s] | ms/step | trajopt [s] | final repair [s] | load world [s] |",
            "|" + "---|" * 14,
        ]
        for cell, c in entry["cells"].items():
            stage = c["stage_medians"]
            lines.append(
                f"| {cell} | {c['runs']} | {c['timed']} | {c['found_solution']} | {c['motion_plan_success']} "
                f"| {c['stable_plan']} | {c['torque_failures']} | {num(c['planning_s']['median'])} "
                f"| {num(c['planning_s']['p90'])} | {num(stage['seed_ik_s'])} | {num(stage['ms_per_step'], 1)} "
                f"| {num(stage['trajopt_s'])} | {num(stage['final_repair_s'])} | {num(stage['load_tamp_world_s'])} |"
            )
        ratio, flags = entry["ratio"], entry["flags"]
        lines += [
            "",
            f"{GATE_MODEL} / {GATE_BASELINE}: median {num(ratio['median'])} (at most {GATE_MEDIAN_MAX}: "
            f"{yes(flags['median'])}), p90 {num(ratio['p90'])} (at most {GATE_P90_MAX}: {yes(flags['p90'])}); all "
            f"{GATE_RUNS} runs of each model timed and every {GATE_MODEL} run on all three counts: "
            f"{yes(flags['success'])}{short(entry['cells'])}.",
        ]
    lines += ["", "Planning time is the sum of the top-level timers, over the timed runs (a run missing a top-level "
                  "timer has none). Each stage time is the median over a model's runs of the run's timer total; "
                  "ms/step is the wall-clock per Adam step. Failed past an effort counts the motion-plan candidates "
                  "that failed with a trajectory past a joint's effort: trajopt's last result, on its horizon or "
                  "between its frames, or a segment the balance gate's torque check failed at every time scale up "
                  "to its cap or on the frames it returns."]
    Path(path).write_text("\n".join(lines) + "\n")


def wilson(k, n, z=1.96):
    """Wilson score interval for ``k`` successes in ``n`` trials (95% at z=1.96)."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def stable_rates(records):
    """``{(cell, height or "all"): (k, n)}`` balanced plans over runs with a known outcome."""
    out = defaultdict(lambda: [0, 0])
    for r in records:
        if r.get("stable_plan") is None:
            continue
        for key in ((r["cell"], r["height"]), (r["cell"], "all")):
            out[key][0] += int(r["stable_plan"])
            out[key][1] += 1
    return {k: tuple(v) for k, v in out.items()}


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


def compute_deltas(records):
    """Paired per-(height, seed) deltas of each non-``full`` cell vs. ``full``.

    The study shares seeds across cells precisely so paired comparisons
    are possible: for every (height, seed) present in BOTH a cell and
    ``full``, delta = metric_cell − metric_full, for every numeric
    metric both records have. Pairing on the shared seed controls for
    per-seed variance, unlike comparing independent cell means.

    That holds only for runs made with known-bugs N4's fix. Before it, the
    blocks' collision spheres came from numpy's unseeded global RNG, so
    same-seed runs planned against different geometry and diverged within a
    few steps. G1's seeds are therefore unpaired: its deltas carry the same
    run-to-run spread as differences of independent runs.

    Returns a flat list of dict rows: cell, height, seed, metric, delta.
    """
    by_key = {(r["cell"], r["height"], r["seed"]): r for r in records}
    full_keys = {(h, s) for (c, h, s) in by_key if c == "full"}
    rows = []
    for (cell, height, seed), r in by_key.items():
        if cell == "full" or (height, seed) not in full_keys:
            continue
        full_r = by_key[("full", height, seed)]
        for m in METRICS:
            v_cell, v_full = r.get(m), full_r.get(m)
            if v_cell is None or v_full is None:
                continue
            rows.append({
                "cell": cell, "height": height, "seed": seed,
                "metric": m, "delta": v_cell - v_full,
            })
    return rows


def aggregate_deltas(delta_rows):
    cells = defaultdict(lambda: defaultdict(list))
    for row in delta_rows:
        cells[(row["cell"], row["height"])][row["metric"]].append(row["delta"])
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


def _write_deltas_csv(delta_rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "height", "seed", "metric", "delta"])
        for row in sorted(delta_rows, key=lambda r: (r["cell"], r["height"], r["seed"], r["metric"])):
            w.writerow([row["cell"], row["height"], row["seed"], row["metric"], row["delta"]])


def _write_md(agg, delta_agg, path, rates=None):
    lines = ["| cell | height | satisfying | CoM-feasible | best soft (CoM-feas.) | margin [m] | motion plan | solved | traj viol. | loop [s] | ms/step | cuRobo calls | Adam gain | restored |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (cell, height), m in sorted(agg.items(), key=lambda kv: (kv[0][1], CELL_ORDER.index(kv[0][0]))):
        def fmt(name, digits=1):
            if name not in m:
                return "—"
            mean, std = m[name]
            return f"{mean:.{digits}f} ± {std:.{digits}f}"
        lines.append(
            f"| {CELL_LABEL[cell]} | {height:.2f} | {fmt('num_satisfying_final', 0)} "
            f"| {fmt('num_satisfying_com_feasible', 0)} | {fmt('best_soft_cost_com_feasible')} "
            f"| {fmt('best_com_margin', 3)} | {fmt('motion_plan_success', 2)} | {fmt('found_solution', 2)} "
            f"| {fmt('traj_com_violation_frac', 3)} | {fmt('loop_s', 0)} | {fmt('ms_per_step', 1)} "
            f"| {fmt('reik_curobo_calls', 1)} | {fmt('adam_gain', 0)} | {fmt('restored', 2)} |"
        )

    if rates:
        lines += [
            "",
            "## Balanced-plan rate (motion plan with every trajectory frame CoM-inside; Wilson 95%)",
            "",
            "| cell | height | k/n | rate | 95% CI |",
            "|---|---|---|---|---|",
        ]
        order = lambda kv: (CELL_ORDER.index(kv[0][0]), (1, 0) if kv[0][1] == "all" else (0, kv[0][1]))
        for (cell, height), (k, n) in sorted(rates.items(), key=order):
            lo, hi = wilson(k, n)
            h = height if height == "all" else f"{height:.2f}"
            lines.append(f"| {CELL_LABEL[cell]} | {h} | {k}/{n} | {k / n:.2f} | [{lo:.2f}, {hi:.2f}] |")

    lines += [
        "",
        "## Per-seed deltas vs. `full` (cell − full, matched by height + seed; paired only for runs with known-bugs N4's fix, so not for G1)",
        "",
        "| cell | height | metric | delta mean | delta std |",
        "|---|---|---|---|---|",
    ]
    for (cell, height), metrics in sorted(delta_agg.items(), key=lambda kv: (kv[0][1], CELL_ORDER.index(kv[0][0]))):
        for m in METRICS:
            if m not in metrics:
                continue
            mean, std = metrics[m]
            lines.append(f"| {CELL_LABEL[cell]} | {height:.2f} | {m} | {mean:.3f} | {std:.3f} |")

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
    ap.add_argument("--gate", action="store_true",
                    help="a run_ablation --gate study: write the speed gate's gate.json and gate.md instead")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load_study(a.manifest)
    if a.gate:
        summary = gate_summary(records)
        (out / "gate.json").write_text(json.dumps(summary, indent=1))
        _write_gate_md(summary, out / "gate.md")
        print(f"{len(records)} runs -> {out}/gate.json, gate.md")
        return
    agg = aggregate(records)
    delta_rows = compute_deltas(records)
    delta_agg = aggregate_deltas(delta_rows)
    _write_csv(agg, out / "results.csv")
    _write_deltas_csv(delta_rows, out / "deltas.csv")
    _write_md(agg, delta_agg, out / "results.md", stable_rates(records))
    _figures(agg, out / "figures")
    print(f"{len(records)} runs -> {out}/results.csv, deltas.csv, results.md, figures/")


if __name__ == "__main__":
    main()
