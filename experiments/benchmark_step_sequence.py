"""Time the public planner with a learned sequence objective and dense recurrent verification."""

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="two_steps_floor_transfer", help="Shipped scene name or a YAML path")
    parser.add_argument("--seed", type=int, default=641)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--particle-batch-size", type=int, help="Particles optimized together (default: all)")
    parser.add_argument("--plateau-mm", type=float, help="Minimum material improvement in worst-endpoint centering")
    parser.add_argument("--patience", type=int, default=5, help="Updates without material improvement before stopping")
    parser.add_argument("--min-updates", type=int)
    parser.add_argument("--budget-seconds", type=float, help="Public planner search budget, in seconds")
    args = parser.parse_args()
    if (
        args.runs < 1
        or (args.iterations is not None and args.iterations < 1)
        or (args.particles is not None and args.particles < 1)
        or (args.particle_batch_size is not None and args.particle_batch_size < 1)
        or (args.budget_seconds is not None and args.budget_seconds <= 0)
    ):
        parser.error("runs, updates, particle counts and time limits must be positive")

    import torch

    from cutamp import algorithm
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs.utils import load_env, resolve_env_path
    from cutamp.grasps import _SETS
    from cutamp.robots import REAL_LEGS
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.utils.plan_processor import raw_motion_plan

    args.output.mkdir(parents=True, exist_ok=True)
    env = load_env(resolve_env_path(args.env))
    cfg = {
        "enable_visualizer": False,
        "curobo_plan": True,
        "experiment_root": str(args.output),
        "max_loop_dur": args.budget_seconds,
    }
    overrides = {
        "num_particles": args.particles,
        "step_particle_batch_size": args.particle_batch_size,
        "step_refine_iterations": args.iterations,
        "step_adam_plateau_mm": args.plateau_mm,
        "step_adam_plateau_min_updates": args.min_updates,
    }
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    cfg.update(step_adam_plateau_patience=args.patience)
    cfg = TAMPConfiguration(**cfg)
    solver = cfg.for_refinement()
    plateau = solver.adam_plateau()
    reports = []
    for i in range(args.runs):
        _SETS.clear()
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        start = perf_counter()
        motion, _, metrics = algorithm.run_cutamp(
            env,
            cfg,
            CostReducer(default_constraint_to_mult.copy()),
            ConstraintChecker(default_constraint_to_tol.copy()),
            experiment_id=f"planner-{i + 1}",
        )
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        phases = [part["phase"] for part in motion or ()]
        grip = [part["gripper"] for part in motion or () if part["gripper"]]
        complete = bool(phases and phases[-1] == "back_out" and grip == ["close", "open"])
        max_xy = metrics.get("max_predicted_landing_xy_m")
        report = {
            "scene": args.env,
            "seed": args.seed,
            "run": i + 1,
            "gpu": torch.cuda.get_device_name(),
            "particles": cfg.num_particles,
            "particle_batch_size": cfg.step_particle_batch_size,
            "max_updates": solver.step_refine_iterations,
            "plateau_mm": solver.step_adam_plateau_mm,
            "plateau_patience": solver.step_adam_plateau_patience,
            "plateau_min_updates": plateau.min_updates if plateau is not None else None,
            "complete": complete,
            "search_budget_s": metrics.get("search_budget_s"),
            "public_wall_s": elapsed,
            "status": metrics["status"],
            "solver_first_accepted_s": metrics.get("time_to_first_accepted_s"),
            "phases": phases,
            "n_steps": sum(p in ("model_step_L", "model_step_R") for p in phases),
            "gripper_events": grip,
            "orders": metrics.get("gap_schedule"),
            "row": metrics.get("row_index"),
            "chosen_iteration": metrics.get("selected_adam_iteration"),
            "winning_batch_updates": metrics.get("adam_updates_completed"),
            "winning_batch_stop_reason": metrics.get("adam_stop_reason"),
            "centers_cm": [100 * x for x in metrics.get("com_center_distances_m", [])],
            "margins_cm": [100 * x for x in metrics.get("com_margins_m", [])],
            "worst_xy_cm": 100 * max_xy if max_xy is not None else None,
            "worst_yaw_deg": metrics.get("max_predicted_landing_yaw_deg"),
            "attempts": [
                {
                    "orders": a["orders"],
                    "status": a["status"],
                    "reason": a.get("reason"),
                    "exact_particle_rows": a.get("particle_rows"),
                    "furthest_endpoint": a.get("furthest_endpoint"),
                    "exact_motion_attempts": a.get("diagnostics", {}).get("motion_plan_attempts"),
                }
                for a in metrics["step_attempts"]
            ],
            "grasp_proposals": metrics.get("grasp_proposals_by_object"),
        }
        if complete and report["n_steps"]:
            assert report["worst_xy_cm"] < 2 and report["worst_yaw_deg"] < 1 and min(report["margins_cm"]) >= 5
        if complete and i == 0:
            file = args.output / "planner-motion.pkl"
            with file.open("wb") as handle:
                pickle.dump(raw_motion_plan(motion, REAL_LEGS), handle)
            report["motion_file"] = str(file)
        reports.append(report)
        print(json.dumps(report), flush=True)
        (args.output / "planner-results.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
