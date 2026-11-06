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


@dataclass(frozen=True)
class TAMPConfiguration:
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
    # Must be members of SUPPORTED_SOFT_COSTS (checked in validate_tamp_config).
    # None (the default, and run_cutamp's without --soft_cost) resolves in
    # __post_init__ to place_close_to_base, joint_limit_margin and
    # minimize_body_movement, which holds the legs, free to move (design call
    # 2), in that order: CostReducer sums them in it. [] means no soft costs.
    soft_cost: Optional[List[str]] = None
    # The periodic IK projection (coupled re-IK): Adam optimizes the Poses,
    # Grasps and free Confs, and the Confs an IK target covers (q_pick,
    # q_place, ...) leave the optimizer. Every reik_interval steps the
    # balance-first GN refresh (gn_project_balanced) projects them onto their
    # moved targets: the CoM in the support rectangle and both soles on the
    # stance first, the pose least squares inside it. Every row's
    # iterate is written back, and no cuRobo runs during Adam; the final
    # pre-check refresh repairs the rows that still miss.
    reik_interval: int = 5
    # Replay the Adam step and the re-IK refresh as CUDA graphs
    # (cutamp/cuda_graphs.py); align_yaw runs stay eager (see
    # uses_cuda_graphs). Default ON (2026-10-31): 3.5x faster steps at n=64,
    # 2.1x at 256, replays bitwise equal to eager. A failed capture ends the
    # run: no switch to eager.
    cuda_graphs: bool = True

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
    # Motion plan with cuRobo after optimization
    curobo_plan: bool = False
    # On RuntimeError from solve_curobo, retry on the next-best satisfying particle
    # up to this many total attempts (1 = no retry). Mitigates per-particle trajopt
    # failures that look feasible but never converge — see motion_solver retry helper.
    # 10 (the user, 2026-11-05): a stage 3 gate run on the real legs first plans on its 6th candidate.
    motion_plan_max_retries: int = 10

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
            default = ["place_close_to_base", "joint_limit_margin", "minimize_body_movement"]
            object.__setattr__(self, "soft_cost", default)


def uses_cuda_graphs(config: TAMPConfiguration) -> bool:
    """Whether this run captures CUDA graphs: align_yaw can't be captured
    (roma's rotmat_to_euler syncs the host)."""
    return config.cuda_graphs and "align_yaw" not in (config.soft_cost or [])


def validate_tamp_config(config: TAMPConfiguration):
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
    if config.cuda_graphs and not uses_cuda_graphs(config):
        import logging
        logging.getLogger(__name__).info("CUDA graphs off for this run: align_yaw can't be captured")
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
