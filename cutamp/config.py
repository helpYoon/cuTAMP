# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TAMPConfiguration:
    # Number of particles to initialize and optimize over.
    num_particles: int = 128

    # Optimization hyperparams
    num_opt_steps: int = 1_000
    # Stop Adam after this many steps without meaningful progress: before any particle is valid, a new-best
    # satisfying count or a best-particle loss drop; after, a better-ranked valid plan (_BestSnapshot). 0 disables.
    # Each particle's best valid state is kept regardless of this setting.
    # Default 50, still above the max record gap of 35 on the blocks_t1 traces
    # it was calibrated on. On a full 1000-step raised-surface run (2026-08-05)
    # no record came after step 50, and on 15 blocks_t1 problems (2026-11-07)
    # patience 150 ran 177-687 steps only to restore the same snapshot: the
    # same plans as 50, at 2-3.5x the loop time.
    early_stop_patience: int = 50
    lr: float = 7e-3  # default LR for optimizer
    conf_lr: float = 2.226e-2  # LR for robot configurations

    # The whole search's budget in seconds, timed from planner entry (_run_step_planner's deadline). None: no cap.
    max_loop_dur: Optional[float] = None

    # The refresh interval: every reik_interval Adam steps the plan refreshes (SteppedPlan.refresh): the world model
    # runs at the current commands, and each pick and place posture goes through the balance-first IK projection
    # (gn_project_balanced) on the stance it then stands on, the CoM in the support region, both soles on the stance
    # and the robot clear of collision first, the pose least squares inside it. A plan's particles are judged only
    # on the Adam step right after a refresh.
    reik_interval: int = 5
    # Replay the Adam step as a CUDA graph (cutamp/cuda_graphs.py), put-back included (2026-11-24), and run the
    # landing model and the stance IK on their own graphs. Default ON (2026-10-31). A failed capture ends the run: no
    # switch to eager. The refresh and the Adam update stay eager.
    cuda_graphs: bool = True

    ## Task Planning
    # The most distinct pick-and-place orders the plan search hands out (plan_search's max_base_plans)
    num_initial_plans: int = 30
    step_lag: int = 2  # actuator delay in physics substeps

    ## Trajectories and cuRobo
    # Motion plan with cuRobo after optimization
    curobo_plan: bool = True
    # When a row's motion fails, try the next satisfying row in _validate_motion_rows' order: the rank (whole units of
    # side lean, then of centring), then joint effort; round the grasp groups with com_center_weight 0. Up to this
    # many rows in all (1 = no retry). Mitigates per-particle trajopt failures that look feasible but never converge.
    # 10 (the user, 2026-11-05): a stage 3 gate run on the real legs first planned on its 6th candidate.
    motion_plan_max_retries: int = 10

    ## Visualizer Args
    # Whether to use visualizer, if set to False a Mock is used
    enable_visualizer: bool = True
    # Number of steps between visualizations of optimization state (note: visualization takes non-trivial time)
    opt_viz_interval: int = 10
    # Whether to visualize the robot mesh, set to False if you want to save network bandwidth (~10MB)
    viz_robot_mesh: bool = True

    ## Logging Args
    # Root directory for logging experiments
    experiment_root: str = "/tmp/cutamp-experiments"

    # Minimum projected CoM clearance from the physical sole hull at each pick and place.
    com_comfort_margin_m: float = 0.05
    # Positive values enable centered pick/place stance IK and center-first plan selection.
    # Their magnitude weights the outer Adam objective. IK uses its own calibrated pose/CoM ratio.
    # Zero restores step-count/effort selection.
    com_center_weight: float = 1.0


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

    # Time budget
    if config.max_loop_dur is not None and config.max_loop_dur <= 0:
        raise ValueError(f"max_loop_dur must be positive or None, not {config.max_loop_dur}")

    # Task Planning and subgraph caching
    if config.num_initial_plans <= 0:
        raise ValueError(f"num_initial_plans must be positive, not {config.num_initial_plans}")
    if not math.isfinite(config.com_comfort_margin_m) or config.com_comfort_margin_m < 0:
        raise ValueError(f"com_comfort_margin_m must be finite and >= 0, got {config.com_comfort_margin_m}")

    weight = config.com_center_weight
    if weight is None or not math.isfinite(weight) or weight < 0:
        raise ValueError(f"com_center_weight must be finite and nonnegative, got {config.com_center_weight}")
    if type(config.step_lag) is not int or config.step_lag < 0:
        raise ValueError("step_lag must be a nonnegative integer physics-step count")
