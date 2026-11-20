"""Run the measured atlas and public T1 planner comparisons without a simulator.

    uv run python experiments/benchmark_t1_step_planner.py --section atlas
    uv run python experiments/benchmark_t1_step_planner.py --section search
    uv run python experiments/benchmark_t1_step_planner.py --section orders

Each run gets its own experiment directory. The section summaries and the full per-row
planner outputs preserve unresolved and unavailable candidates.
"""

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.landing_accuracy import LANDING_ACCURACY, LANDING_BUNDLE
from cutamp.reachability import ORDERS, REACH_MAP, ReachMap, apply_command
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.step_planner import StepInputs
from cutamp.step_search import StepSearch, TargetRegion
from cutamp.tests.paired_grasp_scene import paired_setup

DATA = ROOT / "data/t1-reachability"
SEED = 641


def atlas_benchmark():
    t0 = perf_counter()
    atlas = ReachMap.load(REACH_MAP, nominal_only=True)
    profile = json.loads(LANDING_ACCURACY.read_text())
    all_depths = StepSearch(atlas, accuracy=profile, lag=8)
    standing = StepSearch(atlas, accuracy=profile, lag=8, postures=("standing",))
    load_s = perf_counter() - t0
    out = {
        "offline_build_seconds": atlas.metadata["build_seconds"],
        "map_bytes": REACH_MAP.stat().st_size,
        "map_profile_and_indices_load_seconds": load_s,
        "postures": [p["posture"] for p in atlas.layers(all_depths.home, "left")],
        "queries": {},
    }
    for x in (0.30, 0.48):
        region = TargetRegion("left", (x, 0.18, 0.025), (0.05,) * 3, solid_pick=True)
        rows = {}
        for name, search in (("standing", standing), ("all_postures", all_depths)):
            t = perf_counter()
            edges = search.edges(search.home, region)
            cold = perf_counter() - t
            repeats = []
            for _ in range(64):
                t = perf_counter()
                search.edges(search.home, region)
                repeats.append(perf_counter() - t)
            zero = next(e for e in edges if e.order == "none")
            rows[name] = {
                "fixed_foot_gap_m": zero.distance_m,
                "fixed_foot_posture": zero.posture,
                "overlapping_clear_commands": [sum(e.order == o and e.distance_m == 0 for e in edges) for o in ORDERS],
                "accurate_overlapping_commands": [sum(e.order == o and e.cost[0] == 0 for e in edges) for o in ORDERS],
                "cold_query_ms": cold * 1000,
                "cached_query_median_us": np.median(repeats) * 1e6,
            }
        out["queries"][str(x)] = rows
    order = ("L", ((0.24, 0.21, 0.0),))
    n = 1000
    t = perf_counter()
    for _ in range(n):
        apply_command(all_depths.home, "L", order[1][0])
    out["single_command_mean_us"] = (perf_counter() - t) * 1e6 / n
    t = perf_counter()
    for _ in range(n):
        all_depths._accuracy(*order, True, False)
    out["accuracy_lookup_mean_us"] = (perf_counter() - t) * 1e6 / n
    return out


def run_case(directory, name, env, q, inputs, cfg):
    torch.manual_seed(SEED)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    _, _, m = run_cutamp(
        env,
        cfg,
        CostReducer(default_constraint_to_mult.copy()),
        ConstraintChecker(default_constraint_to_tol.copy()),
        q_init=q,
        experiment_id=name,
        step_inputs=inputs,
    )
    torch.cuda.synchronize()
    attempt_keys = (
        "orders",
        "heuristic_cost",
        "status",
        "reason",
        "accepted_rows",
        "particle_rows",
        "rank_seconds",
        "furthest_endpoint",
    )
    stepped = cfg.for_refinement(stepped=True)
    standing = cfg.for_refinement(stepped=False)
    report = {
        "scene": env.name,
        "device": torch.cuda.get_device_name(),
        "timing_scope": "end-to-end public planner; includes initialization; concurrent GPU work is possible",
        "requested_motion": cfg.curobo_plan,
        "require_modeled_load": cfg.require_modeled_load,
        "com_center_weight": stepped.com_center_weight,
        "standing_com_center_weight": standing.com_center_weight,
        "step_derivative_method": stepped.step_derivative_method,
        "step_trust_scale": stepped.step_trust_scale,
        "seed": SEED,
        "mode": cfg.step_mode,
        "strategy": cfg.step_search_order,
        "budget": cfg.max_step_schedules,
        "time_budget_s": m.get("search_budget_s", cfg.max_loop_dur),
        "num_particles": cfg.num_particles,
        "step_refine_iterations": stepped.step_refine_iterations,
        **{
            key: m.get(key)
            for key in (
                "status",
                "model_evidence_path",
                "load_status",
                "num_optimized_plans",
                "grasp_generation_calls",
                "elapsed_s",
                "time_to_first_accepted_s",
                "total_steps",
                "best_soft_cost",
                "min_com_margin_m",
                "com_center_distances_m",
                "best_com_margin",
                "row_index",
                "gap_schedule",
                "time_limit_reached",
            )
        },
        "wall_seconds": perf_counter() - start,
        "peak_device_bytes_above_start": torch.cuda.max_memory_allocated() - baseline,
        "attempts": [{key: a.get(key) for key in attempt_keys} for a in m["step_attempts"]],
        "metrics_file": str((directory / name / "overall_metrics.json").resolve()),
    }
    if "gaps" in m:
        report["gaps"] = [
            {
                "order": g["order"],
                "command": g["commands"],
                "margin_m": g["margin_m"],
                "initial_margin_m": g["initial_margin_m"],
                "center_distance_m": g.get("center_distance_m"),
                "initial_center_distance_m": g.get("initial_center_distance_m"),
                "precision": [(s["xy_error_m"], s["yaw_error_deg"]) for s in g["steps"]],
            }
            for g in m["gaps"]
        ]
    for i, attempt in enumerate(m["step_attempts"]):
        diagnostics = attempt.get("diagnostics", {})
        for key in ("failed_constraints", "attempted_seed_commands"):
            if key in diagnostics:
                report["attempts"][i][key] = diagnostics[key]
    print(name, report["status"], report["gap_schedule"], round(report["wall_seconds"], 3), flush=True)
    return report


def main():
    from cutamp.envs.utils import load_env, resolve_env_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("atlas", "search", "orders"), required=True)
    parser.add_argument("--output", type=Path, default=DATA / "planner-benchmark")
    parser.add_argument("--case", help="Run one named search or order comparison and save a separate summary.")
    parser.add_argument("--particles", type=int, default=1, help="Particle rows used per schedule.")
    parser.add_argument("--max-seconds", type=float, default=180.0, help="Public search time limit.")
    parser.add_argument(
        "--autodiff-reference", action="store_true", help="Use the historical four-update model settings."
    )
    parser.add_argument(
        "--center-weight",
        type=float,
        help="Centroid preference coefficient. Use 0 for a margin-only ablation.",
    )
    args = parser.parse_args()
    if args.particles < 1:
        parser.error("--particles must be positive")
    search_names = {f"{scene}-{strategy}" for scene in ("low", "multi") for strategy in ("astar", "count")}
    order_names = {f"{order}-{mode}" for order in ORDERS[1:] for mode in ("geometric", "model")} | {
        "loaded",
        "loaded-diagnostic",
    }
    cases = {"atlas": set(), "search": search_names, "orders": order_names}[args.section]
    if args.case is not None and args.case not in cases:
        parser.error(f"Expected a case in {sorted(cases)} for {args.section}")
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=args.section + "-", dir=args.output))
    cfg = TAMPConfiguration(
        step_mode="model", curobo_plan=False,
        step_derivative_method="autodiff" if args.autodiff_reference else "finite_difference",
        com_center_weight=args.center_weight,
        enable_visualizer=False,
        experiment_root=str(directory),
        step_lag=2,
        max_step_schedules=2,
        num_initial_plans=2,
        num_particles=args.particles,
        max_loop_dur=args.max_seconds,
        num_opt_steps=20,
        cuda_graphs=False,
        landing_bundle=str(LANDING_BUNDLE),
    )
    report = {}
    suffix = "-" + args.case if args.case else ""
    if args.particles != 1:
        suffix += f"-p{args.particles}"
    if args.max_seconds != 180.0:
        suffix += f"-{args.max_seconds:g}s"
    if args.section != "atlas" and args.center_weight is not None:
        suffix += f"-cw{args.center_weight:g}"
    if args.section != "atlas" and not args.autodiff_reference:
        suffix += "-fast"
    output = args.output / (args.section + suffix + ".json")

    def save():
        tmp = directory / "summary.tmp"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(output)

    if args.section == "atlas":
        report = atlas_benchmark()
        save()
        return
    env, q, low = paired_setup("far")
    if args.section == "search":
        for scene, e, base_q, bind in (
            ("low", env, q, replace(low, orders=None, commands=None)),
            ("multi", load_env(resolve_env_path("blocks_t1")), None, StepInputs()),
        ):
            for strategy in ("astar", "count"):
                name = f"{scene}-{strategy}"
                if args.case is not None and args.case != name:
                    continue
                report[name] = run_case(directory, name, e, base_q, bind, replace(cfg, step_search_order=strategy))
                save()
    else:
        for order in ("L", "R", "LR", "RL"):
            signed = tuple(
                (0.12 if k == 0 else 0.10, 0.21 if side == "L" else -0.21, 0.0) for k, side in enumerate(order)
            )
            fixed = replace(low, orders=(order,), commands=(signed,))
            for mode in ("geometric", "model"):
                name = f"{order}-{mode}"
                if args.case is not None and args.case != name:
                    continue
                report[name] = run_case(
                    directory, name, env, q, fixed, replace(cfg, step_mode=mode, max_step_schedules=1)
                )
                save()
        if args.case is not None and args.case not in ("loaded", "loaded-diagnostic"):
            return
        env, q, bind = paired_setup("far", place=True)
        fixed = replace(bind, orders=("L", "R"), commands=(((0.12, 0.21, 0),), ((0.1, -0.21, 0),)))
        for name, require_load in (("loaded", True), ("loaded-diagnostic", False)):
            if args.case is not None and args.case != name:
                continue
            report[name] = run_case(
                directory, name, env, q, fixed, replace(cfg, require_modeled_load=require_load, max_step_schedules=1)
            )
            save()


if __name__ == "__main__":
    main()
