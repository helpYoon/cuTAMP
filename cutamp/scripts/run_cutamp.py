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
        from cutamp.robots.t1 import get_model
        from cutamp.utils.plan_processor import raw_motion_plan
        processed = raw_motion_plan(curobo_plan, get_model(config.robot))
        with open(out_path, "wb") as f:
            pickle.dump(processed, f)
        n_traj = len(curobo_plan)
        _log.info(f"Saved the motion plan ({n_traj} segments, schema_version {processed['schema_version']!r}) "
                  f"to {out_path}")
    elif save_plan is not None and curobo_plan is None:
        _log.warning(f"--save_plan requested but no motion plan was generated (curobo_plan is None)")




def entrypoint():
    import argparse

    from cutamp.config import SUPPORTED_ROBOTS, SUPPORTED_SOFT_COSTS
    _defaults = TAMPConfiguration()

    parser = argparse.ArgumentParser(
        description="Run cuTAMP demo. We do not expose all the configs so check cutamp/config.py for additional configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--robot", choices=SUPPORTED_ROBOTS, default=_defaults.robot,
        help="Planning model: real_legs (the default), bfm_wm's 33-DOF T1 standing on its home stance, which runs "
             "only the full system (no ablation flags), or simplified, the 21-DOF T1 the paper's ablation arms run on.",
    )
    parser.add_argument(
        "-n", "--num_particles", type=int, default=_defaults.num_particles, help="Number of particles to use (i.e. batch size)"
    )
    parser.add_argument(
        "--surface_height", type=float, default=0.0,
        help="Elevate the manipulation surface (boxes + goal pad) by this many METRES. "
             "0.0 = ground variant (default). cuTAMP is kinematic, so objects sit at the "
             "new z with no support geometry added. Used for the height sweep.",
    )

    # Soft costs
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
    # --reik_balance_first and --cuda_graphs only restate the defaults; the
    # paper's reproduction commands pass them.
    parser.add_argument(
        "--reik_balance_first", dest="reik_balance_first", action="store_true",
        help="Balance-first GN refresh (CoM in polygon, then pose, then joint-limit margins), "
             "persistent iterates, no cuRobo during Adam. On by default.",
    )
    parser.add_argument(
        "--no_reik_balance_first", dest="reik_balance_first", action="store_false",
        help="Use the pose-only GN refresh with cuRobo fallbacks instead; joint-limit margins "
             "are then not enforced on covered Confs.",
    )
    parser.set_defaults(reik_balance_first=_defaults.reik_balance_first)
    parser.add_argument(
        "--cuda_graphs", dest="cuda_graphs", action="store_true",
        help="Replay the balance-first Adam step and re-IK refresh as CUDA graphs. On by default; "
             "runs without coupled re-IK or balance-first, or with align_yaw, stay eager.",
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
        help="Soft cost(s) to optimize (default: TAMPConfiguration's for --robot: place_close_to_base and "
             "joint_limit_margin, plus minimize_body_movement on real_legs; a bare --soft_cost runs none). "
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
        help="Adam LR for Conf particle params (robot q). Lowering this reduces Adam first-step destruction on T1's 21-DOF cspace.",
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
    parser.add_argument(
        "--no_com_all", action="store_true",
        help="Ablation umbrella: disable ALL CoM machinery — seed-IK residual "
             "(L1), init gate+retries (L2), ComPolygon hard constraint (L3), "
             "trajopt cost (L5). Equivalent to the four individual flags.",
    )
    parser.add_argument(
        "--force_adam", action="store_true",
        help="Run Phase 1 even when init already satisfies with coupled_reik "
             "off (fairness fix for the ablation baseline).",
    )
    parser.add_argument(
        "--soft_in_adam", choices=["auto", "always"], default=None,
        help="'always' puts soft costs in the Adam loss even without "
             "coupled_reik (upstream-faithful attempt for -reIK arms).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="torch.manual_seed for reproducible sampling (ablation study "
             "uses seeds 0..4).",
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
        robot=args.robot,
        num_particles=args.num_particles,
        num_opt_steps=args.num_opt_steps,
        early_stop_patience=args.early_stop_patience,
        lr=args.lr,
        conf_lr=args.conf_lr,
        max_loop_dur=args.max_duration,
        coupled_reik=args.coupled_reik,
        reik_interval=args.reik_interval,
        reik_balance_first=args.reik_balance_first,
        cuda_graphs=args.cuda_graphs,
        soft_cost=args.soft_cost,
        num_initial_plans=args.num_initial_plans,
        curobo_plan=args.motion_plan,
        enable_com_polygon=args.enable_com_polygon and not args.no_com_all,
        enable_com_aware_ik=args.enable_com_aware_ik and not args.no_com_all,
        enable_com_init_gate=not args.no_com_all,
        enable_com_hard_constraint=not args.no_com_all,
        force_adam=args.force_adam,
        soft_in_adam=args.soft_in_adam or TAMPConfiguration.soft_in_adam,
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
