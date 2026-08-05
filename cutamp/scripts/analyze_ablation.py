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
