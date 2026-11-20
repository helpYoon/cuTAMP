# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
from dataclasses import dataclass, replace
from typing import List, Optional


# The CLI choices (run_cutamp.py) and validate_tamp_config key off this tuple.
# Note: cost_function.py's _compute_soft_cost hard-codes the same names and
# must be kept in sync by hand.
SUPPORTED_SOFT_COSTS = (
    "place_close_to_base", "retract_close_to_home", "minimize_body_movement", "joint_limit_margin",
)


@dataclass(frozen=True)
class TAMPConfiguration:
    # Number of particles to initialize and optimize over; pass -n for larger sweeps.
    num_particles: int = 64

    # Optimization hyperparams
    num_opt_steps: int = 1_000
    # Stop Adam after this many steps without meaningful progress
    # (new-best satisfying count or best-particle loss drop). 0 disables.
    # The best-seen particle snapshot is kept regardless of this setting.
    # Default 50, still above the max record gap of 35 on the blocks_t1 traces
    # it was calibrated on. On a full 1000-step raised-surface run (2026-08-05)
    # no record came after step 50, and on 15 blocks_t1 problems (2026-11-07)
    # patience 150 ran 177-687 steps only to restore the same snapshot: the
    # same plans as 50, at 2-3.5x the loop time.
    early_stop_patience: int = 50
    lr: float = 7e-3  # default LR for optimizer
    conf_lr: float = 2.226e-2  # LR for robot configurations

    # Maximum time for optimization in seconds before breaking
    max_loop_dur: Optional[float] = None

    ## Soft Costs
    # Must be members of SUPPORTED_SOFT_COSTS (checked in validate_tamp_config).
    # None (the default, and run_cutamp's without --soft_cost) resolves in
    # __post_init__ to place_close_to_base, joint_limit_margin and
    # minimize_body_movement, which holds the legs, free to move (design call
    # 2), in that order: CostReducer sums them in it. [] means no soft costs.
    soft_cost: Optional[List[str]] = None
    # The periodic IK projection (coupled re-IK): Adam optimizes the Poses
    # and free Confs, and the Confs an IK target covers (q_pick,
    # q_place, ...) leave the optimizer. Every reik_interval steps the
    # balance-first GN refresh (gn_project_balanced) projects them onto their
    # moved targets: the CoM in the support rectangle, both soles on the
    # stance and the robot clear of collision first, the pose least squares
    # inside it. Every row's iterate is written back; the satisfaction check
    # after the loop drops the rows that still miss.
    reik_interval: int = 5
    # Replay the Adam step and the re-IK refresh as CUDA graphs
    # (cutamp/cuda_graphs.py). Default ON (2026-10-31): 3.5x faster steps at n=64,
    # 2.1x at 256, replays bitwise equal to eager. A failed capture ends the
    # run: no switch to eager.
    cuda_graphs: bool = True

    ## Task Planning
    # Number of initial plans to sample
    num_initial_plans: int = 30
    step_mode: str = "model"  # off, rank only, geometric ablation, or frozen model
    step_search_order: str = "astar"
    max_step_schedules: int | None = None
    step_lag: int = 2  # actuator delay in physics substeps
    landing_bundle: str | None = None
    step_reach_map: str | None = None
    step_accuracy_table: str | None = None
    step_derivative_method: str | None = None
    # None selects the solver settings for the actual schedule (see for_refinement).
    step_refine_iterations: int | None = None
    step_trust_scale: float | None = None
    step_proposal_count: int | None = None  # Explicit counts apply to both pick and place.
    step_place_proposal_count: int | None = None
    step_particle_batch_size: int = 64
    require_modeled_load: bool = False

    ## Trajectories and cuRobo
    # Motion plan with cuRobo after optimization
    curobo_plan: bool | None = None
    step_motion_validation: str = "model"  # independent mode requires a transition evidence provider
    # On RuntimeError from solve_curobo, retry on the next satisfying particle, going round the grasp groups,
    # up to this many total attempts (1 = no retry). Mitigates per-particle trajopt
    # failures that look feasible but never converge — see motion_solver retry helper.
    # 10 (the user, 2026-11-05): a stage 3 gate run on the real legs first plans on its 6th candidate.
    # It also sets how many grasps each pick chooses at init (choose_grasps, G5).
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
    # Positive values enable center-first plan selection and weight the Adam centering term.
    # Zero restores step-count/effort selection.
    com_center_weight: float | None = None

    def __post_init__(self):
        if self.curobo_plan is None:
            object.__setattr__(self, "curobo_plan", self.step_mode == "model")
        if self.soft_cost is None:
            default = ["place_close_to_base", "joint_limit_margin", "minimize_body_movement"]
            object.__setattr__(self, "soft_cost", default)

    def _model_proposal_count(self):
        if self.step_proposal_count is not None:
            return self.step_proposal_count
        if self.step_derivative_method is not None or self.step_refine_iterations is not None or self.step_trust_scale is not None:
            return 0
        return 192

    def for_refinement(self, *, stepped: bool) -> "TAMPConfiguration":
        """Resolve optional overrides once a candidate's actual footstep count is known."""
        derivative = self.step_derivative_method or "finite_difference"
        proposals = self._model_proposal_count() if stepped and self.step_mode == "model" else 0
        fast = stepped and self.step_mode == "model" and derivative == "finite_difference"
        updates, trust_scale, weight = (2, 2.0, 4.0) if fast else (4, 1.0, 1.0)
        place_proposals = proposals
        if proposals:
            updates = 0
            if self.step_place_proposal_count is not None:
                place_proposals = self.step_place_proposal_count
            elif self.step_proposal_count is None:
                place_proposals = 128
        return replace(
            self,
            step_proposal_count=proposals,
            step_place_proposal_count=place_proposals,
            step_derivative_method=derivative,
            step_refine_iterations=updates if self.step_refine_iterations is None else self.step_refine_iterations,
            step_trust_scale=trust_scale if self.step_trust_scale is None else self.step_trust_scale,
            com_center_weight=weight if self.com_center_weight is None else self.com_center_weight,
        )


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
    for name in (config.soft_cost or []):
        if name not in SUPPORTED_SOFT_COSTS:
            raise ValueError(f"Unknown soft cost {name!r}; supported: {SUPPORTED_SOFT_COSTS}")

    # Time budget
    if config.max_loop_dur is not None and config.max_loop_dur <= 0:
        raise ValueError(f"max_loop_dur must be positive or None, not {config.max_loop_dur}")

    # Task Planning and subgraph caching
    if config.num_initial_plans <= 0:
        raise ValueError(f"num_initial_plans must be positive, not {config.num_initial_plans}")
    if not math.isfinite(config.com_comfort_margin_m) or config.com_comfort_margin_m < 0:
        raise ValueError(f"com_comfort_margin_m must be finite and >= 0, got {config.com_comfort_margin_m}")

    weight = config.com_center_weight
    if weight is not None and (not math.isfinite(weight) or weight < 0):
        raise ValueError(f"com_center_weight must be finite and nonnegative, got {config.com_center_weight}")
    if config.step_mode not in ("off", "rank", "model", "geometric"):
        raise ValueError("step_mode must be off, rank, model or geometric")
    if config.step_motion_validation not in ("model", "independent"):
        raise ValueError("step_motion_validation must be model or independent")
    if config.step_search_order not in ("astar", "count"):
        raise ValueError("step_search_order must be astar or count")
    if config.max_step_schedules is not None and config.max_step_schedules <= 0:
        raise ValueError("max_step_schedules must be positive when specified")
    if config.step_derivative_method not in (None, "finite_difference", "autodiff"):
        raise ValueError("step_derivative_method must be finite_difference or autodiff")
    scale = config.step_trust_scale
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError("step_trust_scale must be finite and positive")
    if config.step_proposal_count is not None and (type(config.step_proposal_count) is not int or config.step_proposal_count < 0):
        raise ValueError("step_proposal_count must be a nonnegative integer")
    if config.step_place_proposal_count is not None and (
        type(config.step_place_proposal_count) is not int or config.step_place_proposal_count < 0
    ):
        raise ValueError("step_place_proposal_count must be a nonnegative integer")
    if config.step_proposal_count and config.step_mode != "model":
        raise ValueError("Forward command proposals require step_mode=model")
    updates = config.step_refine_iterations
    if (updates is not None and (type(updates) is not int or updates < 0)) or config.step_particle_batch_size < 1:
        raise ValueError("Invalid step refinement iteration or batch count")
    if type(config.step_lag) is not int or config.step_lag < 0:
        raise ValueError("step_lag must be a nonnegative integer physics-step count")
