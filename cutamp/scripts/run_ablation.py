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
