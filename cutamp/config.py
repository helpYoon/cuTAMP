# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# The CLI choices (run_cutamp.py) and validate_tamp_config key off this tuple.
# Note: cost_function.py's _compute_soft_cost hard-codes the same names and
# must be kept in sync by hand.
SUPPORTED_SOFT_COSTS = (
    "dist_from_origin", "place_close_to_base", "max_obj_dist", "min_obj_dist",
    "min_y", "max_y", "align_yaw", "retract_close_to_home",
    "minimize_body_movement", "joint_limit_margin",
)


@dataclass(frozen=True)
class TAMPConfiguration:
    # Number of particles to initialize and optimize over
    num_particles: int = 1024

    # Grasp and Placements
    grasp_dof: Literal[4, 6] = 4

    # Optimization hyperparams
    num_opt_steps: int = 1_000
    # Stop Adam Phase 1 after this many steps without meaningful progress
    # (new-best satisfying count or best-particle loss drop). 0 disables.
    # The best-seen particle snapshot is kept regardless of this setting.
    early_stop_patience: int = 0
    lr: float = 7e-3  # default LR for optimizer
    conf_lr: float = 2.226e-2  # LR for robot configurations

    ## Advanced args - for soft cost experiments. Warning! Might cause unexpected behavior if not used correctly.
    # Maximum time for optimization in seconds before breaking
    max_loop_dur: Optional[float] = None

    ## Soft Costs
    # Default ON: validated as a trio with coupled_reik + place_close_to_base
    # (commit 0c94674) — don't toggle one without considering the others.
    # joint_limit_margin (added 2026-06-11) is independent of that trio.
    optimize_soft_costs: bool = True
    # Must be members of SUPPORTED_SOFT_COSTS (checked in validate_tamp_config).
    soft_cost: Optional[List[str]] = field(
        default_factory=lambda: ["place_close_to_base", "joint_limit_margin"]
    )
    # Re-IK-coupled Adam (bilevel): outer Adam optimizes Pose/Grasp/free-Conf;
    # covered Confs (q_pick/q_place/etc.) are dropped from the optimizer and
    # refreshed by an exact IK call every reik_interval steps. Makes pose-class
    # soft costs trainable on T1 by removing the kinematic-constraint coupling
    # that destroys 21-DOF Adam runs.
    coupled_reik: bool = True
    reik_interval: int = 5
    # Drift gate for re-IK refreshes: a particle is "provably fresh" when its
    # IK target moved less than this (metres) since its last successful solve
    # AND achieved-IK-error + drift still clears the checker's
    # KinematicConstraint tolerance (triangle inequality — see _ReikDriftGate).
    # A dep's refresh solve is skipped only when EVERY particle is fresh
    # (cuRobo pads sub-capacity batches to max_batch_size, so per-particle
    # subset solves save nothing). Rotation budget scales at 10 rad/m, the
    # checker's rot/pos tolerance ratio. 0 disables gating; drift stats are
    # recorded either way under opt_metrics["reik_refresh"].
    reik_drift_tol: float = 0.0
    # >0: coupled-reIK refreshes use a dedicated low-seed IK solver, warm-
    # started from the current particle q (the previous solution is a
    # near-answer, so the 32-seed global search is wasted there). Cost scales
    # ~linearly with seeds. Particle init and the final pre-check refresh
    # always use the full 32-seed solver. 0 = refreshes use the main solver.
    # Default 4 (= current q + 3 random): A/B on blocks_t1 @512 particles
    # showed 407 -> 251 ms/step with best-satisfying / best-soft-cost equal
    # or better than the 32-seed refresh (2026-08-04).
    reik_num_seeds: int = 4
    # Gauss-Newton refresh (cutamp/gn_refresh.py): periodic refreshes project
    # the warm q onto the moved pose target with batched damped GN (~2-5 ms
    # per dep) instead of a cuRobo solve; rows that miss checker tolerance or
    # the CoM polygon fall back to the reik_num_seeds cuRobo solve that
    # refresh. Init and the final pre-check refresh always use the full
    # 32-seed solver. Default ON: A/B on blocks_t1 @512 (2026-08-04) vs the
    # cuRobo-only refresh: loop 98.0 -> 63.1 s, best soft 27.9 -> 21.3,
    # final satisfying 349 vs 341, CoM-in-polygon parity (worst 499 vs 502).
    reik_gn: bool = True
    # Add the CoM residual row to the GN projection (gn_refresh.py): violated
    # support-rectangle axes get an active-set pull back inside. A/B verdict
    # (2026-08-04, blocks_t1 @512): mechanically sound (halves com-misses;
    # 1-iter recovery) but NO runtime win — place-dep refreshes have ~70-90
    # conflicted rows that trade pose for CoM, so >=1 miss always remains and
    # the all-or-nothing cuRobo fallback fires regardless (pick-deps: 0%
    # fallbacks; place-deps: 100%, both with and without the row). Quality
    # parity. Default OFF; revisit if fallbacks become rare/cheap or for K=1.
    reik_gn_com: bool = False
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

    ## Task Planning and subgraph caching
    # Number of initial plans to sample
    num_initial_plans: int = 30
    # Cache particles for reuse - this is coupled to our current domains so the implementation is not general
    cache_subgraphs: bool = False

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
    # Pass through to MotionPlanner's joint-limit margin trajopt cost
    # (trajectory insurance; endpoint configs are handled by the
    # "joint_limit_margin" soft cost). Margins: robots/t1.py.
    enable_joint_limit_margin: bool = True
    # CoM-aware seed-IK residual (cuRobo fork): IK natively trades hand-pose vs
    # COM, recruiting legs within limits. The post-IK mask stays as the
    # backstop. Validation: commits 217d4b6 / 5b9ae65.
    enable_com_aware_ik: bool = True
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


def validate_tamp_config(config: TAMPConfiguration):
    if config.num_particles <= 0:
        raise ValueError(f"num_particles must be positive, not {config.num_particles}")
    if config.grasp_dof not in {4, 6}:
        raise ValueError(f"Invalid grasp_dof: {config.grasp_dof}")

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
    if config.reik_drift_tol < 0:
        raise ValueError(f"reik_drift_tol must be >= 0, not {config.reik_drift_tol}")
    if config.reik_num_seeds < 0:
        raise ValueError(f"reik_num_seeds must be >= 0, not {config.reik_num_seeds}")
    if config.reik_lbfgs_iters < 0 or config.reik_lbfgs_iters % 20 != 0:
        # cuRobo's LBFGS requires num_iters to be a multiple of inner_iters
        # (20 in lbfgs_ik.yml); validate here so a bad value fails at startup,
        # not at the first mid-loop refresh-solver build.
        raise ValueError(
            f"reik_lbfgs_iters must be a non-negative multiple of 20, not {config.reik_lbfgs_iters}"
        )
    if config.reik_gn and not config.coupled_reik:
        # GN refresh only runs inside the coupled-reIK loop. Now that reik_gn
        # defaults ON, this combination just means "no refreshes at all" —
        # warn (don't raise) so --no_coupled_reik keeps working, but a
        # benchmark comparing GN on/off knows the flag was inert.
        import logging
        logging.getLogger(__name__).warning(
            "reik_gn has no effect without coupled_reik (no re-IK refreshes run)"
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
