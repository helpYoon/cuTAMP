"""Time the public planner with a learned sequence objective and dense recurrent verification."""

import argparse
import json
import pickle
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["planner", "student", "native", "screen"],
                        help="Use the public defaults, compare learned predictions with replay, or run a reference solver")
    parser.add_argument("--checkpoint", type=Path)
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
    parser.add_argument("--trace-center", action="store_true", help="Record cumulative best predicted centering by update")
    parser.add_argument("--warm-start-ik", action="store_true", help="Start each subsequent learned Adam endpoint IK fit from the previous posture")
    parser.add_argument("--ik-objective-tol", type=float, help="Projected-gradient tolerance during the learned objective only (native default: 2e-9)")
    parser.add_argument(
        "--max-step-schedules",
        type=int,
        default=0,
        help="Maximum ranked schedules to try in Adam modes (default: uncapped)",
    )
    parser.add_argument("--budget-seconds", type=float, help="Public planner search budget, in seconds")
    args = parser.parse_args()
    if args.mode == "student" and args.checkpoint is None:
        parser.error("student mode needs --checkpoint")
    if (args.warm_start_ik or args.ik_objective_tol is not None or args.checkpoint is not None) and args.mode not in ("student", "planner"):
        parser.error("A learned checkpoint or inner IK settings require student or planner mode")
    if args.mode == "screen" and (args.plateau_mm is not None or args.trace_center):
        parser.error("plateau stopping and centering traces require an Adam mode")
    if (
        args.runs < 1
        or (args.iterations is not None and args.iterations < 1)
        or (args.particles is not None and args.particles < 1)
        or (args.particle_batch_size is not None and args.particle_batch_size < 1)
        or args.max_step_schedules < 0
        or (args.budget_seconds is not None and args.budget_seconds <= 0)
    ):
        parser.error("runs, updates, particle counts and time limits must be positive; max-step-schedules must be nonnegative")

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
        "max_step_schedules": args.max_step_schedules or None,
    }
    if args.mode == "planner":
        overrides = {
            "num_particles": args.particles,
            "step_particle_batch_size": args.particle_batch_size,
            "step_refine_iterations": args.iterations,
            "step_adam_plateau_mm": args.plateau_mm,
            "step_adam_plateau_min_updates": args.min_updates,
            "step_ik_objective_tol": args.ik_objective_tol,
            "step_sequence_bundle": str(args.checkpoint) if args.checkpoint else None,
        }
        cfg.update({key: value for key, value in overrides.items() if value is not None})
    else:
        cfg.update(num_particles=args.particles or 64, step_particle_batch_size=args.particle_batch_size or args.particles or 64)
    if args.mode in ("student", "native"):
        cfg.update(
            step_optimizer="adam",
            step_refine_iterations=args.iterations or 2,
            step_adam_plateau_mm=args.plateau_mm,
            step_adam_plateau_min_updates=args.min_updates or 0,
        )
    if args.mode == "screen":
        cfg["step_optimizer"] = "screen"
    else:
        cfg.update(step_adam_plateau_patience=args.patience, step_adam_trace_center=args.trace_center)
    cfg = TAMPConfiguration(**cfg)
    solver = cfg.for_refinement(stepped=True)
    plateau = solver.adam_plateau()
    reports = []
    student = None
    for i in range(args.runs):
        _SETS.clear()
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        start = perf_counter()
        load_s = 0
        if args.mode == "student" and student is None:
            from experiments.step_sequence_adapter import load_student, student_objectives

            student, _ = load_student(args.checkpoint)
            torch.cuda.synchronize()
            load_s = perf_counter() - start
        context = (student_objectives(student, warm_start_ik=args.warm_start_ik, ik_objective_tol=args.ik_objective_tol)
                   if student is not None else nullcontext([]))
        with context as diagnostics:
            motion, _, metrics = algorithm.run_cutamp(
                env,
                cfg,
                CostReducer(default_constraint_to_mult.copy()),
                ConstraintChecker(default_constraint_to_tol.copy()),
                experiment_id=f"{args.mode}-{i + 1}",
            )
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        phases = [part["phase"] for part in motion or ()]
        grip = [part["gripper"] for part in motion or () if part["gripper"]]
        complete = bool(phases and phases[-1] == "back_out" and grip == ["close", "open"])
        max_xy = metrics.get("max_predicted_landing_xy_m")
        report = {
            "mode": args.mode,
            "scene": args.env,
            "seed": args.seed,
            "run": i + 1,
            "gpu": torch.cuda.get_device_name(),
            "particles": cfg.num_particles,
            "particle_batch_size": cfg.step_particle_batch_size,
            "step_optimizer": solver.step_optimizer,
            "max_updates": solver.step_refine_iterations,
            "warm_start_ik": solver.step_ik_warm_start if args.mode == "planner" else args.warm_start_ik,
            "ik_objective_tol": solver.step_ik_objective_tol if args.mode == "planner" else args.ik_objective_tol,
            "plateau_mm": solver.step_adam_plateau_mm,
            "plateau_patience": solver.step_adam_plateau_patience,
            "plateau_min_updates": plateau.min_updates if plateau is not None else None,
            "max_step_schedules": cfg.max_step_schedules,
            "complete": complete,
            "search_budget_s": metrics.get("search_budget_s"),
            "public_wall_s": elapsed,
            "student_load_s": load_s,
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
            "diagnostics": diagnostics,
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
            file = args.output / f"{args.mode}-motion.pkl"
            with file.open("wb") as handle:
                pickle.dump(raw_motion_plan(motion, REAL_LEGS), handle)
            report["motion_file"] = str(file)
        reports.append(report)
        print(json.dumps(report), flush=True)
        (args.output / f"{args.mode}-results.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
