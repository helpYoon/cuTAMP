"""Turn an ablation-study manifest into the paper table and figures.

Usage:
    python -m cutamp.scripts.analyze_ablation \
        --manifest /tmp/ablation-study/manifest.json --out docs/ablation
"""
import argparse
import csv
import glob
import json
import math
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
            "num_satisfying_final": opt.get("num_satisfying_final"),
            "num_satisfying_com_feasible": opt.get("num_satisfying_com_feasible"),
            "best_soft_cost_com_feasible": opt.get("best_soft_cost_com_feasible"),
            # The planned particle's margin. Runs before known-bugs N3 (G1) only have rank 0's, which a
            # motion-plan retry didn't plan.
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
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load_study(a.manifest)
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
