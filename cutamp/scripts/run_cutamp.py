# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import get_env_dir, load_env

from cutamp.scripts.utils import (
    default_constraint_to_mult,
    default_constraint_to_tol,
    setup_logging,
)

_log = logging.getLogger(__name__)


def load_demo_env(name: str) -> TAMPEnvironment:
    if name == "blocks_t1":
        env_path = os.path.join(get_env_dir(), "blocks_t1.yml")
        return load_env(env_path)
    raise ValueError(f"Unknown environment name: {name}")


def cutamp_demo(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    experiment_id: Optional[str] = None,
    save_plan: Optional[str] = None,
):
    """Run the cuTAMP demo.

    ``save_plan``:
        ``None`` — don't save the motion plan (default).
        ``"auto"`` — save to ``<experiment_root>/<experiment_id>/motion_plan.pkl``.
        Any other string — save to that path (parent directory auto-created).
    """
    setup_logging()

    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    # Resolve experiment_id up front so the --save_plan output path lines up
    # with the directory ExperimentLogger writes its other artifacts to (it
    # would otherwise generate its own timestamp inside run_cutamp).
    if experiment_id is None:
        from datetime import datetime
        experiment_id = datetime.now().isoformat().split(".")[0]

    curobo_plan, _num_satisfying = run_cutamp(
        env, config, cost_reducer, constraint_checker, experiment_id=experiment_id,
    )

    if save_plan is not None and curobo_plan is not None:
        if save_plan == "auto":
            out_path = Path(config.experiment_root) / experiment_id / "motion_plan.pkl"
        else:
            out_path = Path(save_plan)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Process into the structured [trunk, arms, hands] schema and pickle.
        # See cutamp/utils/plan_processor.py for the output schema.
        from cutamp.utils.plan_processor import process_motion_plan
        processed = process_motion_plan(curobo_plan)
        with open(out_path, "wb") as f:
            pickle.dump(processed, f)
        n_traj = sum(1 for e in curobo_plan if e.get("type") == "trajectory")
        _log.info(f"Saved processed motion plan ({n_traj} segments) to {out_path}")
    elif save_plan is not None and curobo_plan is None:
        _log.warning(f"--save_plan requested but no motion plan was generated (curobo_plan is None)")




def entrypoint():
    import argparse

    from cutamp.config import SUPPORTED_SOFT_COSTS
    _defaults = TAMPConfiguration()

    parser = argparse.ArgumentParser(
        description="Run cuTAMP demo. We do not expose all the configs so check cutamp/config.py for additional configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--env",
        help="Environment name to run",
        default="blocks_t1",
        choices=["blocks_t1"],
    )
    parser.add_argument(
        "-n", "--num_particles", type=int, default=_defaults.num_particles, help="Number of particles to use (i.e. batch size)"
    )

    # Soft costs
    parser.add_argument(
        "--no_optimize_soft_costs", dest="optimize_soft_costs", action="store_false",
        help="Disable soft-cost optimization (default: on, with place_close_to_base).",
    )
    parser.set_defaults(optimize_soft_costs=_defaults.optimize_soft_costs)
    parser.add_argument(
        "--upstream_style_optimize", action="store_true",
        help="Diagnostic: mimic NVlabs/cuTAMP main — Adam runs always with soft costs in loss, no Phase 2 LBFGS.",
    )
    parser.add_argument(
        "--no_coupled_reik", dest="coupled_reik", action="store_false",
        help="Disable re-IK-coupled Adam (default: on). Coupled mode is required for "
             "pose-class soft costs (place_close_to_base, dist_from_origin, ...) on T1's 21-DOF cspace.",
    )
    parser.set_defaults(coupled_reik=_defaults.coupled_reik)
    parser.add_argument(
        "--reik_interval", type=int, default=_defaults.reik_interval,
        help="Re-IK cadence K when --coupled_reik is on. Smaller K bounds KinematicConstraint drift but does more IK calls.",
    )
    parser.add_argument(
        "--soft_cost",
        nargs="*",
        default=list(_defaults.soft_cost or []),
        choices=list(SUPPORTED_SOFT_COSTS),
        help="Soft cost(s) to optimize (default: from TAMPConfiguration). Can specify multiple: --soft_cost retract_close_to_home minimize_body_movement",
    )

    # Grasp
    parser.add_argument(
        "--grasp_dof",
        type=int,
        default=_defaults.grasp_dof,
        choices=[4, 6],
        help="Grasp DOF to use.",
    )

    # Approach
    parser.add_argument(
        "--approach",
        default=_defaults.approach,
        choices=["optimization", "sampling"],
        help="Approach to use. Optimization is cuTAMP (sampling + optimization), while sampling is just resampling."
        "If using sampling, you may need to modify the --num_resampling_attempts.",
    )
    parser.add_argument(
        "--num_resampling_attempts",
        type=int,
        default=100,
        help="Number of resampling attempts per skeleton if using sampling approach.",
    )
    parser.add_argument(
        "--num_opt_steps", type=int, default=_defaults.num_opt_steps, help="Number of optimization steps to run for each skeleton."
    )
    parser.add_argument(
        "--lr", type=float, default=_defaults.lr, help="Adam LR for non-Conf particle params (poses, grasps).",
    )
    parser.add_argument(
        "--conf_lr", type=float, default=_defaults.conf_lr,
        help="Adam LR for Conf particle params (robot q). Lowering this reduces Adam first-step destruction on T1's 21-DOF cspace.",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=_defaults.max_loop_dur,
        help="Maximum duration for optimization or sampling in seconds. Overrides --num_resampling_attempts and --num_opt_steps if set.",
    )
    parser.add_argument(
        "--num_initial_plans", type=int, default=_defaults.num_initial_plans, help="Number of initial plans to sample with task planner."
    )
    parser.add_argument("--cache_subgraphs", action="store_true", help="Whether to cache subgraph samples for reuse.")
    parser.add_argument(
        "--motion_plan",
        action="store_true",
        help="Whether to plan for full motions after using cuTAMP.",
    )
    parser.add_argument(
        "--no_enable_com_polygon",
        dest="enable_com_polygon",
        action="store_false",
        help="Disable the COM-over-base-polygon soft cost on the motion planner. "
             "By default the cost is enabled (5e5 weight) and keeps the planner's "
             "configurations from tipping over the wheelbase.",
    )
    parser.set_defaults(enable_com_polygon=_defaults.enable_com_polygon)
    parser.add_argument(
        "--no_enable_com_aware_ik",
        dest="enable_com_aware_ik",
        action="store_false",
        help="Disable the CoM-aware seed-IK residual (cuRobo fork). By default "
             "IK trades hand-pose vs COM and recruits the legs to center the "
             "COM (validated 2026-06-09; see config.enable_com_aware_ik).",
    )
    parser.set_defaults(enable_com_aware_ik=_defaults.enable_com_aware_ik)

    # Visualization and logging
    parser.add_argument(
        "--disable_visualizer",
        action="store_true",
        help="Disable the rerun visualizer. Note if you want accurate timing information, you should disable the visualizer.",
    )
    parser.add_argument(
        "--viz_interval", type=int, default=_defaults.opt_viz_interval, help="Interval for visualizing optimization state in steps."
    )
    parser.add_argument(
        "--disable_robot_mesh",
        action="store_true",
        help="Disable robot mesh visualization to save visualization bandwidth.",
    )
    parser.add_argument(
        "--experiment_root", type=str, default=_defaults.experiment_root, help="Root directory for experiment logging."
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Experiment ID for logging. Results will be saved in <experiment_root>/<experiment_id>",
    )
    parser.add_argument(
        "--save_plan",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Save the processed motion plan as a pickle (no arg: "
            "<experiment_root>/<experiment_id>/motion_plan.pkl; or pass a path). "
            "See cutamp/utils/plan_processor.py for the schema."
        ),
    )

    # We only expose a subset of the full TAMPConfiguration. Check config.py for the full configuration.
    args = parser.parse_args()
    config = TAMPConfiguration(
        num_particles=args.num_particles,
        grasp_dof=args.grasp_dof,
        approach=args.approach,
        num_resampling_attempts=args.num_resampling_attempts,
        num_opt_steps=args.num_opt_steps,
        lr=args.lr,
        conf_lr=args.conf_lr,
        max_loop_dur=args.max_duration,
        optimize_soft_costs=args.optimize_soft_costs,
        upstream_style_optimize=args.upstream_style_optimize,
        coupled_reik=args.coupled_reik,
        reik_interval=args.reik_interval,
        soft_cost=args.soft_cost,
        num_initial_plans=args.num_initial_plans,
        cache_subgraphs=args.cache_subgraphs,
        curobo_plan=args.motion_plan,
        enable_com_polygon=args.enable_com_polygon,
        enable_com_aware_ik=args.enable_com_aware_ik,
        enable_visualizer=not args.disable_visualizer,
        opt_viz_interval=args.viz_interval,
        viz_robot_mesh=not args.disable_robot_mesh,
        experiment_root=args.experiment_root,
    )
    validate_tamp_config(config)

    # Load env and run demo
    env = load_demo_env(args.env)
    cutamp_demo(env, config, experiment_id=args.experiment_id, save_plan=args.save_plan)


if __name__ == "__main__":
    entrypoint()
