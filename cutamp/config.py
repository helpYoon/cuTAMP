# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import List, Optional


# The CLI choices (run_cutamp.py) and validate_tamp_config key off this tuple.
# Note: cost_function.py's _compute_soft_cost hard-codes the same names and
# must be kept in sync by hand.
SUPPORTED_SOFT_COSTS = (
    "dist_from_origin", "place_close_to_base", "max_obj_dist", "min_obj_dist",
    "min_y", "max_y", "align_yaw", "retract_close_to_home",
    "minimize_body_movement", "joint_limit_margin",
)

# The planning models (cutamp.robots.t1.get_model); the CLI's --robot choices.
SUPPORTED_ROBOTS = ("simplified", "real_legs")

# The paper's ablation arms (design call 15), which only the simplified model runs: each of these must keep its
# default on the real legs. They reproduce from the paper's tag.
_REAL_LEGS_REQUIRED = {
    "enable_com_polygon": True, "enable_com_aware_ik": True, "enable_com_init_gate": True,
    "enable_com_hard_constraint": True, "coupled_reik": True, "force_adam": False, "reik_balance_first": True,
}


@dataclass(frozen=True)
class TAMPConfiguration:
    # The planning model (SUPPORTED_ROBOTS): "simplified", the 21-DOF CoM-fixed model, or "real_legs", bfm_wm's
    # 33-DOF T1 on the home stance (design 7.2). It also sets soft_cost's default.
    robot: str = "simplified"

    # Number of particles to initialize and optimize over. The IK solver's
    # buffers are sized to this (ik_max_batch_size), so it bounds GPU memory:
    # 1024 OOMs a 24 GB card at solver build, 512 peaks near ~17 GB, 256
    # leaves room for two concurrent runs. Default lowered 1024 -> 256
    # (2026-08-05); pass -n for larger sweeps.
    num_particles: int = 256

    # Optimization hyperparams
    num_opt_steps: int = 1_000
    # Stop Adam Phase 1 after this many steps without meaningful progress
    # (new-best satisfying count or best-particle loss drop). 0 disables.
    # The best-seen particle snapshot is kept regardless of this setting.
    # Default 150: calibrated on blocks_t1 traces (max record gap 35) and
    # verified on the raised-surface variant (2026-08-05): a full 1000-step
    # run produced zero new records after step 50 — patience 150 matched its
    # quality (435 vs 433 satisfying, soft 0.80 vs 0.81) at 1/3 the loop time.
    early_stop_patience: int = 150
    lr: float = 7e-3  # default LR for optimizer
    conf_lr: float = 2.226e-2  # LR for robot configurations

    ## Advanced args - for soft cost experiments. Warning! Might cause unexpected behavior if not used correctly.
    # Maximum time for optimization in seconds before breaking
    max_loop_dur: Optional[float] = None

    ## Soft Costs
    # place_close_to_base is validated with coupled_reik (commit 0c94674,
    # with soft-cost optimization, then a switch, on) — don't change one
    # without considering the other. joint_limit_margin (added 2026-06-11)
    # is independent of that pair.
    # Must be members of SUPPORTED_SOFT_COSTS (checked in validate_tamp_config).
    # None (the default) resolves in __post_init__ to place_close_to_base and
    # joint_limit_margin, plus minimize_body_movement on the real legs, whose
    # legs move freely (design call 2). [] means no soft costs. A resolved list
    # stays when the robot changes: dataclasses.replace(cfg, robot=...) needs
    # soft_cost=None too, or build a fresh TAMPConfiguration(robot=...).
    soft_cost: Optional[List[str]] = None
    # Re-IK-coupled Adam (bilevel): outer Adam optimizes Pose/Grasp/free-Conf;
    # covered Confs (q_pick/q_place/etc.) are dropped from the optimizer and
    # refreshed by an exact IK call every reik_interval steps. Makes pose-class
    # soft costs trainable on T1 by removing the kinematic-constraint coupling
    # that destroys 21-DOF Adam runs.
    coupled_reik: bool = True
    reik_interval: int = 5
    # >0: coupled-reIK refreshes use a dedicated low-seed IK solver, warm-
    # started from the current particle q (the previous solution is a
    # near-answer, so the 32-seed global search is wasted there). Cost scales
    # ~linearly with seeds. Particle init and the final pre-check refresh
    # always use the full 32-seed solver. 0 = refreshes use the main solver.
    # Neither this nor reik_lbfgs_iters has an effect with reik_balance_first
    # beyond validation: balance-first runs no cuRobo during Adam.
    # Default 4 (= current q + 3 random): A/B on blocks_t1 @512 particles
    # showed 407 -> 251 ms/step with best-satisfying / best-soft-cost equal
    # or better than the 32-seed refresh (2026-08-04).
    reik_num_seeds: int = 4
    # Balance-first GN refresh (gn_project_balanced): CoM-in-polygon is GN's
    # first priority, the pose is least squares inside it, and joint-limit
    # margins use what is left; every row's iterate is written back, and no
    # cuRobo runs during Adam (the final pre-check refresh still repairs).
    # Default ON (2026-10-30): it is the only path that keeps covered Confs
    # out of the joint_limit_margin bands (they are out of Adam, so the soft
    # cost never moves them) — check_plan_margins failed 7/10 runs without
    # it, 0/10 with it (n=64, seeds 0-4). False = pose-only GN with the
    # reik_num_seeds cuRobo fallback for rows that miss checker tolerance or
    # the CoM polygon (the paper's default-refresh arm).
    reik_balance_first: bool = True
    # Replay the balance-first Adam step and re-IK refresh as CUDA graphs
    # (cutamp/cuda_graphs.py). Only runs that use them can capture them (see
    # uses_cuda_graphs); the rest stay eager. Default ON (2026-10-31): 3.5x
    # faster steps at n=64, 2.1x at 256, replays bitwise equal to eager. A
    # failed capture ends the run: no switch to eager.
    cuda_graphs: bool = True
    # LBFGS iteration budget for the refresh/fallback solver (0 = cuRobo's
    # default 100, sized for cold global solves). The fallback is warm-started
    # repair work. Default 40: full A/B (2026-08-04, blocks_t1 @512, two runs
    # + user verification) showed fallback 282 -> 168 ms, per-step 142 -> 95 ms
    # with satisfying / soft / CoM-in-polygon all at parity. 20 was REJECTED:
    # the CoM re-centering inside LBFGS doesn't converge (CoM-in-polygon
    # collapsed to 407/512). Must be a multiple of the LBFGS inner_iters
    # (20, lbfgs_ik.yml). Init and the final pre-check refresh keep the
    # full 100-iteration schedule.
    reik_lbfgs_iters: int = 40

    ## Task Planning
    # Number of initial plans to sample
    num_initial_plans: int = 30

    ## Collision Checking
    # Movable object collision sphere representation
    coll_n_spheres: int = 50
    coll_sphere_radius: float = 0.005
    # Distance at which collision checking is activated between gripper and movables
    gripper_activation_distance: float = 0.0
    # Distance at which collision checking is activated between movables and movables
    movable_activation_distance: float = 0.0

    ## Trajectories and cuRobo
    # Pass through to MotionPlanner's COM-over-base-polygon soft cost.
    enable_com_polygon: bool = True
    # CoM-aware seed-IK residual (cuRobo fork): IK natively trades hand-pose vs
    # COM, recruiting legs within limits. The post-IK mask stays as the
    # backstop. Validation: commits 217d4b6 / 5b9ae65. Also gates the re-IK
    # refresh's CoM terms (balance-first level 1, the GN merge gate, the final
    # repair's CoM check), so --no_com_all leaves no balance channel in re-IK.
    enable_com_aware_ik: bool = True
    # --- Ablation-study flags (2x2 baseline; docs/superpowers/specs/
    # 2026-08-05-standard-cutamp-baseline-design.md). All default to the
    # full system; the baseline cells turn them off/on explicitly. ---
    # L3: register the ComPolygon hard constraint (satisfying-mask member).
    enable_com_hard_constraint: bool = True
    # L2: particle init uses the CoM-safe retry wrapper; False = plain IK.
    enable_com_init_gate: bool = True
    # Fairness: run Phase 1 even when init already satisfies and coupled_reik
    # is off (today's skip_adam shortcut would let the baseline skip
    # optimization entirely on easy tasks — upstream never skipped).
    force_adam: bool = False
    # "auto" = soft costs join the Adam loss only when coupled_reik is on
    # (today's rule). "always" = join regardless — the upstream-faithful
    # attempt for the -reIK arms (spec decision (a)).
    soft_in_adam: str = "auto"
    # Motion plan with cuRobo after optimization
    curobo_plan: bool = False
    # On RuntimeError from solve_curobo, retry on the next-best satisfying particle
    # up to this many total attempts (1 = no retry). Mitigates per-particle trajopt
    # failures that look feasible but never converge — see motion_solver retry helper.
    motion_plan_max_retries: int = 5

    ## Visualizer Args
    # Whether to use visualizer, if set to False a Mock is used
    enable_visualizer: bool = True
    # Number of steps between visualizations of optimization state (note: visualization takes non-trivial time)
    opt_viz_interval: int = 10
    # Whether to visualize the robot mesh, set to False if you want to save network bandwidth (~10MB)
    viz_robot_mesh: bool = True

    ## Logging Args
    enable_experiment_logging: bool = True
    # Root directory for logging experiments
    experiment_root: str = "/tmp/cutamp-experiments"

    def __post_init__(self):
        if self.soft_cost is None:
            default = ["place_close_to_base", "joint_limit_margin"]
            if self.robot == "real_legs":
                default.append("minimize_body_movement")
            object.__setattr__(self, "soft_cost", default)


def uses_cuda_graphs(config: TAMPConfiguration) -> bool:
    """Whether this run captures CUDA graphs: they record the balance-first
    Adam step and refresh, so they need coupled re-IK and balance-first, and
    align_yaw can't be captured (roma's rotmat_to_euler syncs the host)."""
    return (
        config.cuda_graphs and config.coupled_reik and config.reik_balance_first
        and "align_yaw" not in (config.soft_cost or [])
    )


def validate_tamp_config(config: TAMPConfiguration):
    if config.robot not in SUPPORTED_ROBOTS:
        raise ValueError(f"Unknown robot {config.robot!r}; supported: {SUPPORTED_ROBOTS}")
    if config.robot == "real_legs":
        changed = [f"{name}={getattr(config, name)}" for name, value in _REAL_LEGS_REQUIRED.items()
                   if getattr(config, name) != value]
        if changed:
            raise ValueError(
                f"robot real_legs runs only the full system, not the paper's ablation paths ({', '.join(changed)}); "
                "they reproduce on the simplified model (design call 15)"
            )

    if config.num_particles <= 0:
        raise ValueError(f"num_particles must be positive, not {config.num_particles}")

    # Optimization hyperparams
    if config.num_opt_steps <= 0:
        raise ValueError(f"num_opt_steps must be positive, not {config.num_opt_steps}")
    if config.early_stop_patience < 0:
        raise ValueError(f"early_stop_patience must be >= 0, not {config.early_stop_patience}")
    if config.lr <= 0:
        raise ValueError(f"Learning rate (lr) must be positive, not {config.lr}")
    if config.conf_lr <= 0:
        raise ValueError(f"Configuration learning rate (conf_lr) must be positive, not {config.conf_lr}")
    if config.reik_interval <= 0:
        raise ValueError(f"reik_interval must be positive, not {config.reik_interval}")
    if config.reik_num_seeds < 0:
        raise ValueError(f"reik_num_seeds must be >= 0, not {config.reik_num_seeds}")
    if config.reik_lbfgs_iters < 0 or config.reik_lbfgs_iters % 20 != 0:
        # cuRobo's LBFGS requires num_iters to be a multiple of inner_iters
        # (20 in lbfgs_ik.yml); validate here so a bad value fails at startup,
        # not at the first mid-loop refresh-solver build.
        raise ValueError(
            f"reik_lbfgs_iters must be a non-negative multiple of 20, not {config.reik_lbfgs_iters}"
        )
    if config.soft_in_adam not in ("auto", "always"):
        raise ValueError(
            f"soft_in_adam must be 'auto' or 'always', not {config.soft_in_adam!r}"
        )
    if config.cuda_graphs and not uses_cuda_graphs(config):
        import logging
        logging.getLogger(__name__).info(
            "CUDA graphs off for this run: they need coupled_reik and reik_balance_first, and not align_yaw"
        )
    for name in (config.soft_cost or []):
        if name not in SUPPORTED_SOFT_COSTS:
            raise ValueError(f"Unknown soft cost {name!r}; supported: {SUPPORTED_SOFT_COSTS}")

    # Advanced args
    if config.max_loop_dur is not None and config.max_loop_dur <= 0:
        raise ValueError(f"max_loop_dur must be positive or None, not {config.max_loop_dur}")

    # Task Planning and subgraph caching
    if config.num_initial_plans <= 0:
        raise ValueError(f"num_initial_plans must be positive, not {config.num_initial_plans}")

    # Collision checking
    if config.coll_n_spheres <= 0:
        raise ValueError(f"coll_n_spheres must be positive, not {config.coll_n_spheres}")
    if config.coll_sphere_radius <= 0:
        raise ValueError(f"coll_sphere_radius must be positive, not {config.coll_sphere_radius}")
    if config.movable_activation_distance < 0:
        raise ValueError(f"movable_activation_distance must be non-negative, not {config.movable_activation_distance}")
