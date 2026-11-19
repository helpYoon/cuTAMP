"""Compare fixed-feet planning and automatic step search on the shipped two-block floor scene.

Run in the cuTAMP environment with local CUDA access. The first run for each mode and
request warms dependencies. The subsequent AB/BA pairs use the same initial seed and
identical geometric, particle and trajectory checks. Every call uses public run_cutamp.
"""

import argparse
import hashlib
import json
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cutamp import algorithm, grasps
from cutamp.com_polygon_cost import _fk_com_and_base, margin_from_support_planes
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import load_env, resolve_env_path
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol


def measure(directory, name, cfg, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = load_env(resolve_env_path("blocks_t1"))
    reducer = CostReducer(default_constraint_to_mult)
    checker = ConstraintChecker(default_constraint_to_tol)
    observations = {"initially_cached_shapes": len(grasps._SETS)}
    shape_work = []
    chosen = {}
    optimize = algorithm.ParticleOptimizer.__call__
    solve = algorithm.solve_curobo
    draw = algorithm.grasp_sets

    def timed_optimize(self, *args, **kwargs):
        result = optimize(self, *args, **kwargs)
        observations.setdefault("optimizer_feasible_counts", []).append(result[1]["num_satisfying_final"])
        if result[0]:
            observations.setdefault("first_feasible_particles_s", perf_counter() - start)
        return result

    def timed_motion(info, particle, world, *args, **kwargs):
        result = solve(info, particle, world, *args, **kwargs)
        if result[0] is not None:
            observations.setdefault("first_accepted_motion_s", perf_counter() - start)
            chosen.update(info=info, particle=particle, world=world)
        return result

    def timed_grasps(objects, device):
        before = len(grasps._SETS)
        begin = perf_counter()
        pools = draw(objects, device)
        shape_work.append(
            {
                "seconds": perf_counter() - begin,
                "fresh_shapes": len(grasps._SETS) - before,
                "grasps_by_object": {obj: len(pool.grasps) for obj, pool in pools.items()},
            }
        )
        return pools

    with (
        patch.object(algorithm.ParticleOptimizer, "__call__", timed_optimize),
        patch.object(algorithm, "solve_curobo", timed_motion),
        patch.object(algorithm, "grasp_sets", timed_grasps),
    ):
        torch.cuda.synchronize()
        start = perf_counter()
        motion, _, metrics = algorithm.run_cutamp(env, cfg, reducer, checker, experiment_id=name)
        torch.cuda.synchronize()
        end = perf_counter()
    attempt = next(iter(metrics.get("step_attempts", [])), {})
    if chosen:
        info, particle, world = (chosen[k] for k in ("info", "particle", "world"))
        q = torch.stack([particle[q] for q in algorithm.pick_place_confs(info["plan_skeleton"])])
        with torch.no_grad():
            com, _ = _fk_com_and_base(world, q, world.site_kinematics("com"))
            xy = com[:, :2]
            observations["selected_keyframe_margin_m"] = margin_from_support_planes(
                xy, *info["cost_fn"].support_planes
            ).tolist()
            observations["selected_keyframe_center_distance_m"] = (
                (xy - info["cost_fn"].support_centroid).norm(dim=-1).tolist()
            )
        observations["selected_keyframes_sha256"] = hashlib.sha256(q.detach().cpu().numpy().tobytes()).hexdigest()
    timers = json.loads((directory / name / "timer_metrics.json").read_text())
    return {
        "name": name,
        "mode": cfg.step_mode,
        "requested_motion": cfg.curobo_plan,
        "seed": seed,
        "num_particles": cfg.num_particles,
        "num_initial_plans": cfg.num_initial_plans,
        "schedule_cap": cfg.max_step_schedules,
        "centroid_weight": cfg.com_center_weight,
        "com_margin_m": cfg.com_comfort_margin_m,
        "wall_s": end - start,
        **observations,
        "shape_work": shape_work,
        "feasible_rows": attempt.get("particle_rows", metrics.get("num_satisfying_final")),
        "selected": metrics["found_solution"],
        "motion_segments": len(motion) if motion is not None else 0,
        "motion_attempts": metrics.get("motion_plan_attempts", 0),
        "motion_duration_s": metrics.get("plan_duration_s"),
        "schedule": metrics.get("gap_schedule", attempt.get("orders")),
        "rank_s": attempt.get("rank_seconds"),
        "atlas_load_s": metrics.get("step_search", {}).get("load_seconds"),
        "skeleton": metrics.get("final_plan_skeleton"),
        "timers_s": {key: value["total"] for key, value in timers.items()},
        "evidence_file": str(directory / name / "overall_metrics.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--particles", default=64, type=int)
    parser.add_argument("--pairs", default=2, type=int)
    parser.add_argument("--request", choices=("both", "particles", "motion"), default="both")
    parser.add_argument("--output", type=Path, default=ROOT / "data/t1-reachability/standing-mode-comparison")
    args = parser.parse_args()
    if args.particles < 1 or args.pairs < 1:
        parser.error("Use positive particle and measured pair counts")
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=args.output)).resolve()
    base = TAMPConfiguration(
        num_particles=args.particles,
        num_initial_plans=1,
        max_step_schedules=1,
        enable_visualizer=False,
        experiment_root=str(directory),
    )
    report = {"device": torch.cuda.get_device_name(), "scene": "blocks_t1", "runs": []}
    out = directory / "summary.json"
    for motion in (False, True):
        request = "motion" if motion else "particles"
        if args.request not in ("both", request):
            continue
        for i in range(args.pairs + 1):
            # Alternating order keeps a single mode from consistently getting the first turn.
            for mode in ("off", "model") if i % 2 == 0 else ("model", "off"):
                name = f"{request}-{'warmup' if i == 0 else i}-{mode}"
                row = measure(directory, name, replace(base, step_mode=mode, curobo_plan=motion), args.seed)
                row["warmup"] = i == 0
                report["runs"].append(row)
                tmp = out.with_suffix(".tmp")
                tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                tmp.replace(out)
                print(name, round(row["wall_s"], 3), row["feasible_rows"], row["motion_segments"], flush=True)
    print(out)


if __name__ == "__main__":
    main()
