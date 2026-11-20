"""Time the public T1 planner on the recorded floor transfer without supplying task choices."""

import argparse
import json
import pickle
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

START = perf_counter()

import torch

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import load_env, resolve_env_path
from cutamp.grasps import _SETS
from cutamp.robots import REAL_LEGS
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--profile", action="store_true", help="Collect synchronized GPU phase timings on each request.")
    parser.add_argument(
        "--cli-home",
        action="store_true",
        help="Start at the robot's nominal home instead of the scene's synthetic standing start.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cfg = TAMPConfiguration(enable_visualizer=False, experiment_root=str(args.output), max_loop_dur=args.max_duration)
    env = load_env(resolve_env_path("two_steps_floor_transfer"))
    q = REAL_LEGS.home if args.cli_home else None
    reports = []
    for trial in range(args.runs):
        # Retain loaded weights between warm requests, regenerate the complete grasp set each time.
        _SETS.clear()
        torch.manual_seed(env.demo_seed)
        with ExitStack() as stack:
            if args.profile:
                from experiments.profile_run_cutamp import Timings
                from cutamp import algorithm, step_particles
                from cutamp.step_planner import ScheduleSearch

                phases = Timings()
                for obj, name, label in (
                    (algorithm, "setup_cutamp", "world_setup"),
                    (algorithm, "_load_step_search", "reach_and_landing_tables"),
                    (algorithm, "_load_predictor", "load_predictor"),
                    (algorithm, "grasp_sets", "grasp_generation"),
                    (ScheduleSearch, "next", "schedule_ranking"),
                    (step_particles, "_refine_batch", "particle_refinement"),
                    (step_particles, "_validate_motion_rows", "connected_motion"),
                ):
                    phases.wrap(stack, obj, name, label)
            torch.cuda.synchronize()
            start = perf_counter()
            motion, _, metrics = run_cutamp(
                env,
                cfg,
                CostReducer(default_constraint_to_mult.copy()),
                ConstraintChecker(default_constraint_to_tol.copy()),
                q_init=q,
                experiment_id=f"trial-{trial + 1}",
            )
            torch.cuda.synchronize()
            finished = perf_counter()
        events = [s["gripper"] for s in motion or () if s["gripper"]]
        complete = events == ["close", "open"] and motion[-1]["phase"] == "back_out"
        report = dict(
            run=trial + 1,
            public_wall_s=finished - start,
            status=metrics["status"],
            num_particles=metrics["num_particles"],
            grasp_proposals_by_object=metrics["grasp_proposals_by_object"],
            complete=complete,
            first_full_motion_s=metrics["time_to_first_accepted_s"] if complete else None,
            segments=len(motion or []),
            step_phases=[s["phase"] for s in motion or () if s["phase"] in ("model_step_L", "model_step_R")],
            gripper_events=events,
            orders=metrics.get("gap_schedule"),
            attempts=[
                {"orders": a["orders"], "status": a["status"], "reason": a.get("reason")}
                for a in metrics["step_attempts"]
            ],
            motion_attempts=metrics["motion_plan_attempts"],
            grasp_generation_calls=metrics["grasp_generation_calls"],
            search_budget_s=metrics["search_budget_s"],
            improvement_budget_s=metrics.get("improvement_budget_s"),
        )
        if args.profile:
            report["phase_seconds"] = dict(phases.seconds)
            report["phase_calls"] = dict(phases.calls)
            report["phase_samples_seconds"] = dict(phases.samples)
        if complete:
            from cutamp.utils.plan_processor import raw_motion_plan

            plan_file = args.output / f"trial-{trial + 1}" / "motion_plan.pkl"
            with plan_file.open("wb") as handle:
                pickle.dump(raw_motion_plan(motion, REAL_LEGS), handle)
            report.update(
                saved_motion=str(plan_file.relative_to(args.output)),
                com_center_distances_cm=[100 * value for value in metrics.get("com_center_distances_m", [])],
                max_predicted_landing_xy_cm=(
                    100 * metrics["max_predicted_landing_xy_m"] if "max_predicted_landing_xy_m" in metrics else None
                ),
                max_predicted_landing_yaw_deg=metrics.get("max_predicted_landing_yaw_deg"),
            )
        if trial == 0:
            report["process_to_return_s"] = finished - START
        print(json.dumps(report), flush=True)
        reports.append(report)
        (args.output / "timings.json").write_text(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "starting_state": "planner home" if args.cli_home else "recording's synthetic standing history",
                    "task_inputs": "scene and On(part, pad) goal only",
                    "runs": reports,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
