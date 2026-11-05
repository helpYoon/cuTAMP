"""Runner for the standard-cuTAMP 2x2 ablation study.

Sweeps the cells x heights x seeds, one sequential GPU run each, writing
standard experiment dirs plus a resumable manifest. See
docs/superpowers/specs/2026-08-05-standard-cutamp-baseline-design.md.
The 2x2 is balance (CoM) x re-IK on the balance-first refresh, plus
``default_reik``, the full system on the default refresh. Balance-first
and CUDA graphs are run_cutamp's defaults, so ``default_reik`` turns both
off explicitly; with no flags it would be a second ``full`` arm.

``--gate`` runs the speed gate instead (design section 8): the real-leg
model against the simplified one at n 256 and 512, interleaved, every run
replayable from its seed.

Usage:
    python -m cutamp.scripts.run_ablation --out /tmp/ablation-study
    python -m cutamp.scripts.run_ablation --out ... --smoke   # tiny pilot
    python -m cutamp.scripts.run_ablation --out ... --gate    # the speed gate
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

BALANCE_FIRST = ["--reik_balance_first", "--cuda_graphs"]
CELLS = {
    "full": BALANCE_FIRST,
    "default_reik": ["--no_reik_balance_first", "--no_cuda_graphs"],
    "no_com": ["--no_com_all"] + BALANCE_FIRST,
    "no_reik": ["--no_coupled_reik", "--force_adam", "--soft_in_adam", "always"],
    "standard": ["--no_com_all", "--no_coupled_reik", "--force_adam",
                 "--soft_in_adam", "always"],
}
HEIGHTS = [0.0, 0.25, 0.50, 0.65]
SEEDS = list(range(10))
SOFT_ARGS = ["--soft_cost", "joint_limit_margin"]  # one soft-cost setting for every arm
BASE_ARGS = ["-n", "512", "--disable_visualizer", "--motion_plan", "--save_plan", "auto"] + SOFT_ARGS
SMOKE_ARGS = ["-n", "64", "--num_opt_steps", "30", "--disable_visualizer"] + SOFT_ARGS
# The speed gate (design section 8): section 8's soft-cost list on both models, plus the real legs' own default,
# minimize_body_movement (call 2), at HEAD's n and G1's.
GATE_CELLS = {"simplified": ["--robot", "simplified", "--soft_cost", "joint_limit_margin"],
              "real_legs": ["--robot", "real_legs", "--soft_cost", "joint_limit_margin", "minimize_body_movement"]}
GATE_N = (256, 512)


def build_run_plan(heights=HEIGHTS, seeds=SEEDS, smoke=False):
    base = SMOKE_ARGS if smoke else BASE_ARGS
    n = int(base[base.index("-n") + 1])
    plan = []
    for cell, cell_args in CELLS.items():
        for h in heights:
            for s in seeds:
                exp_id = f"ablation_{cell}_h{int(h * 100):03d}_s{s}"
                args = base + [
                    "--surface_height", str(h),
                    "--seed", str(s),
                    "--experiment_id", exp_id,
                ] + cell_args
                plan.append({
                    "cell": cell, "n": n, "height": h, "seed": s,
                    "experiment_id": exp_id, "args": args,
                })
    return plan


def build_gate_plan(heights=HEIGHTS, seeds=SEEDS) -> List[dict]:
    """The speed gate's runs (design section 8): every height, seed, n in GATE_N and model, in that loop order, so the
    two models alternate through one session. Each run's environment sets PYTHONHASHSEED to its seed, so with its
    --seed any run replays exactly."""
    plan = []
    for h in heights:
        for s in seeds:
            for n in GATE_N:
                for cell, cell_args in GATE_CELLS.items():
                    exp_id = f"gate_{cell}_n{n}_h{int(h * 100):03d}_s{s}"
                    args = ["-n", str(n), "--disable_visualizer", "--motion_plan", "--save_plan", "auto",
                            "--surface_height", str(h), "--seed", str(s), "--experiment_id", exp_id] + cell_args
                    plan.append({"cell": cell, "n": n, "height": h, "seed": s, "experiment_id": exp_id,
                                 "args": args, "env": {"PYTHONHASHSEED": str(s)}})
    return plan


def _subprocess_exec(run, experiment_root):
    cmd = [sys.executable, "-m", "cutamp.scripts.run_cutamp"] + run["args"] + [
        "--experiment_root", str(experiment_root),
    ]
    env = run.get("env", {})  # the gate's PYTHONHASHSEED
    print(f"[{run['experiment_id']}] {' '.join([f'{k}={v}' for k, v in env.items()] + cmd)}", flush=True)
    return subprocess.call(cmd, env={"PYTHONPATH": "", **_clean_env(), **env})


def _clean_env():
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    return env


def run_study(plan, out_dir, execute=None, experiment_root=None):
    """Run (or resume) a study's plan, writing a resumable manifest.

    ``experiment_root`` defaults to ``<out_dir>/experiments`` so each
    study is self-contained under its own ``--out`` dir — two studies (or
    a real run followed by a ``--smoke`` pilot) never share an
    experiment_root and can't silently overwrite each other's experiment
    dirs, even though experiment IDs alone (``ablation_full_h000_s0``)
    aren't study-discriminated.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if experiment_root is None:
        experiment_root = out_dir / "experiments"
    experiment_root = Path(experiment_root)
    if execute is None:
        def execute(run, _experiment_root=experiment_root):
            return _subprocess_exec(run, _experiment_root)
    manifest_path = out_dir / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else []
    done = {r["experiment_id"] for r in manifest if r.get("status") == "done"}
    for run in plan:
        if run["experiment_id"] in done:
            continue
        code = execute(run)
        manifest = [r for r in manifest if r["experiment_id"] != run["experiment_id"]]
        manifest.append({
            **{k: run[k] for k in ("cell", "n", "height", "seed", "experiment_id")},
            "status": "done" if code == 0 else f"failed({code})",
            "experiment_path": str(experiment_root / run["experiment_id"]),
        })
        tmp_path = manifest_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=1))
        os.replace(tmp_path, manifest_path)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="study output dir (manifest lives here)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="tiny pilot: 64 particles, 30 steps, 1 height, 1 seed, no motion plan")
    mode.add_argument("--gate", action="store_true",
                      help="the speed gate (design section 8): the simplified and real-leg models at n 256 and 512, "
                           "interleaved, each run replayable (--seed and PYTHONHASHSEED)")
    ap.add_argument("--heights", type=float, nargs="+", help="override the heights")
    ap.add_argument("--seeds", type=int, nargs="+", help="override the seeds")
    a = ap.parse_args()
    if a.gate:
        plan = build_gate_plan(heights=a.heights or HEIGHTS, seeds=a.seeds or SEEDS)
    else:
        plan = build_run_plan(
            heights=a.heights or ([0.0] if a.smoke else HEIGHTS),
            seeds=a.seeds or ([0] if a.smoke else SEEDS),
            smoke=a.smoke,
        )
    manifest = run_study(plan, a.out)
    failed = [r for r in manifest if r["status"] != "done"]
    print(f"{len(manifest) - len(failed)}/{len(manifest)} runs done; {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
