# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp     -n 64 --num_opt_steps 50  --surface_height 0.50 --soft_cost place_close_to_base joint_limit_margin minimize_body_movement  --motion_plan --save_plan data/motion_plan.pkl
"""

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
from cutamp.envs.utils import get_env_dir, load_env, raise_surface_height

from cutamp.scripts.utils import (
    default_constraint_to_mult,
    default_constraint_to_tol,
    setup_logging,
)

_log = logging.getLogger(__name__)


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
    The plan is saved raw (``"stage3-raw"``); see cutamp/utils/plan_processor.py.
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

    curobo_plan, _num_satisfying, _overall_metrics = run_cutamp(
        env, config, cost_reducer, constraint_checker, experiment_id=experiment_id,
    )

    if save_plan is not None and curobo_plan is not None:
        if save_plan == "auto":
            out_path = Path(config.experiment_root) / experiment_id / "motion_plan.pkl"
        else:
            out_path = Path(save_plan)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # The plan is saved raw, in its model's joint order, until stage 6's schema 4 (decision 9). See
        # cutamp/utils/plan_processor.py.
        from cutamp.robots import REAL_LEGS
        from cutamp.utils.plan_processor import raw_motion_plan
        processed = raw_motion_plan(curobo_plan, REAL_LEGS)
        with open(out_path, "wb") as f:
            pickle.dump(processed, f)
        n_traj = len(curobo_plan)
        _log.info(f"Saved the motion plan ({n_traj} segments, schema_version {processed['schema_version']!r}) "
                  f"to {out_path}")
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
        "-n", "--num_particles", type=int, default=_defaults.num_particles, help="Number of particles to use (i.e. batch size)"
    )
    parser.add_argument(
        "--surface_height", type=float, default=0.0,
        help="Elevate the manipulation surface (boxes + goal pad) by this many METRES. "
             "0.0 = ground variant (default). cuTAMP is kinematic, so objects sit at the "
             "new z with no support geometry added.",
    )

    # Soft costs
    parser.add_argument(
        "--reik_interval", type=int, default=_defaults.reik_interval,
        help="Periodic IK projection cadence K. Smaller K bounds KinematicConstraint drift but does more IK calls.",
    )
    parser.add_argument(
        "--cuda_graphs", dest="cuda_graphs", action="store_true",
        help="Replay the Adam step and re-IK refresh as CUDA graphs. On by default; align_yaw runs stay eager.",
    )
    parser.add_argument(
        "--no_cuda_graphs", dest="cuda_graphs", action="store_false",
        help="Run the Adam step and re-IK refresh eagerly.",
    )
    parser.set_defaults(cuda_graphs=_defaults.cuda_graphs)
    parser.add_argument(
        "--soft_cost",
        nargs="*",
        default=None,
        choices=list(SUPPORTED_SOFT_COSTS),
        help="Soft cost(s) to optimize (default: place_close_to_base, joint_limit_margin and "
             "minimize_body_movement; a bare --soft_cost runs none). "
             "Can specify multiple: --soft_cost retract_close_to_home minimize_body_movement",
    )

    parser.add_argument(
        "--num_opt_steps", type=int, default=_defaults.num_opt_steps, help="Number of optimization steps to run for each skeleton."
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=_defaults.early_stop_patience,
        help="Stop Adam early after this many steps without meaningful progress (0 disables).",
    )
    parser.add_argument(
        "--lr", type=float, default=_defaults.lr, help="Adam LR for non-Conf particle params (poses, grasps).",
    )
    parser.add_argument(
        "--conf_lr", type=float, default=_defaults.conf_lr,
        help="Adam LR for Conf particle params (robot q).",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=_defaults.max_loop_dur,
        help="Maximum duration for optimization in seconds. Overrides --num_opt_steps if set.",
    )
    parser.add_argument(
        "--num_initial_plans", type=int, default=_defaults.num_initial_plans, help="Number of initial plans to sample with task planner."
    )
    parser.add_argument(
        "--motion_plan",
        action="store_true",
        help="Whether to plan for full motions after using cuTAMP.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="torch.manual_seed for reproducible sampling.",
    )

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
            "Save the motion plan as a pickle (no arg: "
            "<experiment_root>/<experiment_id>/motion_plan.pkl; or pass a path): the raw "
            "'stage3-raw' plan. See cutamp/utils/plan_processor.py."
        ),
    )

    # We only expose a subset of the full TAMPConfiguration. Check config.py for the full configuration.
    args = parser.parse_args()
    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)
    config = TAMPConfiguration(
        num_particles=args.num_particles,
        num_opt_steps=args.num_opt_steps,
        early_stop_patience=args.early_stop_patience,
        lr=args.lr,
        conf_lr=args.conf_lr,
        max_loop_dur=args.max_duration,
        reik_interval=args.reik_interval,
        cuda_graphs=args.cuda_graphs,
        soft_cost=args.soft_cost,
        num_initial_plans=args.num_initial_plans,
        curobo_plan=args.motion_plan,
        enable_visualizer=not args.disable_visualizer,
        opt_viz_interval=args.viz_interval,
        viz_robot_mesh=not args.disable_robot_mesh,
        experiment_root=args.experiment_root,
    )
    validate_tamp_config(config)

    # Load env and run demo
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    raise_surface_height(env, args.surface_height)  # no-op at 0.0
    cutamp_demo(env, config, experiment_id=args.experiment_id, save_plan=args.save_plan)


if __name__ == "__main__":
    entrypoint()
