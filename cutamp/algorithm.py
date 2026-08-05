# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Core cuTAMP algorithm implementation."""

import logging
from datetime import datetime
from typing import Callable, List, Union, Optional, Tuple
from unittest.mock import Mock

import numpy as np
import torch
from curobo.types import DeviceCfg as TensorDeviceType

from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import TAMPEnvironment
from cutamp.experiment_logger import ExperimentLogger
from cutamp.motion_solver import solve_curobo
from cutamp.optimize_plan import ParticleOptimizer
from cutamp.particle_initialization import ParticleInitializer
from cutamp.robots import load_t1_container, t1_home
from cutamp.rollout import RolloutFunction
from cutamp.t1_domain import all_t1_operators
from cutamp.tamp_world import TAMPWorld, check_tamp_world_not_in_collision
from cutamp.task_planning import PlanSkeleton, task_plan_generator
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import RerunVisualizer, Visualizer

_log = logging.getLogger(__name__)


def heuristic_fn(
    plan_skeleton: PlanSkeleton, cost_dict: dict, constraint_checker: ConstraintChecker, verbose: bool = True
) -> float:
    """
    Get a single heuristic value for a cost dict corresponding to a rollout.

    We first compute the success rate of each constraint. If the constraint has zero success, we assign it a penalty
    of -num_particles. We then compute the mean success rate across all constraints, and use the failure rate as the
    heuristic (lower the better).
    """
    full_mask = constraint_checker.get_full_mask(cost_dict)
    successes = []
    num_particles = None
    for con_type, con_info in full_mask.items():
        for name, mask in con_info.items():
            num_particles = mask.shape[0]
            satisfying = mask.reshape(num_particles, -1).sum(0)
            satisfying[satisfying == 0] = -num_particles
            successes.extend(satisfying.tolist())
            if verbose:
                _log.debug(f"{con_type} {name} {satisfying.tolist()}")
    success_mean = sum(successes) / len(successes)
    success_rate = success_mean / num_particles
    failure_rate = 1 - success_rate
    heuristic = 100 * failure_rate

    # We have a preference for shorter plans
    heuristic += len(plan_skeleton)
    return heuristic


def make_arm_affinity_priority_fn(world) -> Callable[[object], float]:
    """Build a priority function for BFS sibling ordering.

    For pick operators, returns the 3D Euclidean distance from the arm's
    home end-effector position to the block's world pose. Closer = lower
    priority = explored first by biased BFS. For non-pick operators or
    pick operators whose block doesn't resolve (e.g., placeholder name
    not yet bound to a real scene object), returns ``float("inf")`` so they
    sort AFTER all resolved picks in the ascending BFS order.

    Same-side picks bubble to the top of the BFS exploration order, so
    the first satisfying plan skeleton tends to be the same-side assignment.
    Cross-body groundings are still enumerated (just later); BFS naturally
    backtracks if same-side fails feasibility.
    """
    def priority(ground_op) -> float:
        meta = ground_op.operator.metadata
        if meta.action_type != "pick" or meta.arm is None:
            return float("inf")
        # LeftPick/RightPick parameter order is (obj, grasp, q). Block is values[0].
        block_name = ground_op.values[0]
        try:
            obj = world.get_object(block_name)
        except (KeyError, ValueError):
            return float("inf")
        if obj is None or obj.pose is None:
            return float("inf")
        block_xyz = np.asarray(obj.pose[:3], dtype=np.float64)
        arm_home_xyz = world.arm_home_ee_world[meta.arm].cpu().numpy().astype(np.float64)
        return float(np.linalg.norm(block_xyz - arm_home_xyz))
    return priority


def get_top_satisfying_particles(
    plan_info: dict,
    config: TAMPConfiguration,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    n: int,
) -> List[dict]:
    """Top-N satisfying particles ranked by ascending soft cost. May return < n
    if fewer particles satisfy constraints. Used by the motion-plan retry loop:
    if solve_curobo fails on the best particle, retry on the next-best.
    """
    particles = plan_info["particles"]
    with torch.no_grad():
        rollout = plan_info["rollout_fn"](particles)
        cost_dict = plan_info["cost_fn"](rollout, particles)
    satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    if not satisfying_mask.any():
        raise RuntimeError("No satisfying particles found")
    soft_costs = cost_reducer.soft_costs(cost_dict)
    sat_indices = torch.arange(config.num_particles, device=soft_costs.device)[satisfying_mask]
    order = torch.argsort(soft_costs[satisfying_mask])
    top = sat_indices[order[: min(n, order.numel())]]
    return [{k: v[i].detach().clone() for k, v in particles.items()} for i in top]


def sample_plan_skeleton(
    plan_gen,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    plan_count: int,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    particle_initializer: ParticleInitializer,
) -> Tuple[Union[dict, None], bool]:
    """
    Try sampling a plan skeleton (if any remain), then its particles and compute the heuristic.
    Returns the plan_info dict and whether any satisfying particles were found upon initialization.
    """
    plan_skeleton = next(plan_gen)
    plan_str = [op.name for op in plan_skeleton]
    _log.debug(f"[Plan {plan_count + 1}] Sampled plan {plan_str}")

    # Sample particles
    with timer.time("initialize_particles"):
        plan_particles = particle_initializer(plan_skeleton)
    if plan_particles is None:  # failed subgraph
        return None, False

    # Rollout particles and compute costs
    rollout_fn = RolloutFunction(plan_skeleton, world, config)
    cost_fn = CostFunction(plan_skeleton, world, config)
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = rollout_fn(plan_particles)
        cost_dict = cost_fn(rollout, plan_particles)
        heuristic = heuristic_fn(plan_skeleton, cost_dict, constraint_checker)

    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict)
    num_satisfying = satisfying_mask.sum().item()

    # Best cost initially
    with timer.time("compute_best_cost"):
        consider_types = {"constraint"}
        if config.optimize_soft_costs:
            consider_types.add("cost")
        costs = cost_reducer(cost_dict, consider_types=consider_types)
        if satisfying_mask.any():
            best_cost = costs[satisfying_mask].min().item()
            best_soft_cost = cost_reducer.soft_costs(cost_dict)[satisfying_mask].min().item()
        else:
            best_cost, best_soft_cost = float("inf"), float("inf")

    plan_info = {
        "idx": plan_count,
        "plan_skeleton": plan_skeleton,
        "particles": plan_particles,
        "rollout_fn": rollout_fn,
        "cost_fn": cost_fn,
        "heuristic": heuristic,
        "num_satisfying": num_satisfying,
        "best_cost": best_cost,
        "best_soft_cost": best_soft_cost,
    }

    _log.debug(
        f"[Plan {plan_count + 1}] {plan_info['num_satisfying']}/{config.num_particles} satisfying, "
        f"heuristic = {plan_info['heuristic']}"
    )
    return plan_info, num_satisfying > 0


def _motion_plan_with_retries(
    plan_info: dict,
    candidates: list,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer,
) -> Tuple[Optional[dict], bool]:
    """Run ``solve_curobo`` on the top-N satisfying candidates; return on first success.

    cuRobo's plan_cspace can fail on a specific (start, goal) pair even after
    its internal retries; a different particle gives a different pair and
    usually succeeds.

    Returns ``(curobo_plan, skip_plan)``:
      - ``curobo_plan``: the successful plan, or ``None`` if all candidates
        failed.
      - ``skip_plan``: ``True`` if a ``NotImplementedError`` was raised — the
        caller should skip this plan_skeleton entirely and move on.
    """
    last_err = None
    for rank, particle in enumerate(candidates):
        try:
            curobo_plan = solve_curobo(
                plan_info, particle, world, config, timer, visualizer,
            )
            if rank > 0:
                _log.warning(
                    f"Motion plan succeeded on retry rank {rank}/{len(candidates) - 1}"
                )
            return curobo_plan, False
        except NotImplementedError as e:
            _log.warning(
                f"Motion planning not supported for this plan: {e}. Trying next plan..."
            )
            return None, True
        except RuntimeError as e:
            _log.warning(
                f"Motion plan failed on particle rank {rank}/{len(candidates) - 1}: {e}"
            )
            last_err = e
    _log.error(
        f"Exhausted {len(candidates)} particle retries; giving up on this plan. "
        f"Last error: {last_err}"
    )
    return None, False


def setup_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
):
    # Validate args and setup experiment logger
    validate_tamp_config(config)
    if experiment_id is None:
        experiment_id = datetime.now().isoformat().split(".")[0]

    exp_logger = ExperimentLogger(name=experiment_id, config=config) if config.enable_experiment_logging else Mock()
    exp_logger.save_env(env)

    # Loading robot can be done offline, so doesn't count towards timing
    tensor_args = TensorDeviceType()
    robot_container = load_t1_container(device_cfg=tensor_args)

    # v0.8 single-MotionPlanner: q_init is a single full-DOF vector for every robot.
    if q_init is None:
        q_init = t1_home
    q_init = tensor_args.to_device(q_init)

    # Load TAMP world and warmup IK solver
    timer = TorchTimer()
    with timer.time("load_tamp_world", log_callback=_log.info):
        world = TAMPWorld(
            env=env,
            device_cfg=tensor_args,
            robot=robot_container,
            q_init=q_init,
            collision_activation_distance=0.0,
            coll_n_spheres=config.coll_n_spheres,
            coll_sphere_radius=config.coll_sphere_radius,
            enable_com_aware_ik=config.enable_com_aware_ik,
            ik_max_batch_size=config.num_particles,
            reik_num_seeds=config.reik_num_seeds,
            reik_lbfgs_iters=config.reik_lbfgs_iters,
        )
        check_tamp_world_not_in_collision(world)

    with timer.time("warmup_ik_solver", log_callback=_log.info):
        world.warmup_ik_solver(config.num_particles)

    # Setup visualizer (doesn't count towards timing)
    visualizer = (
        RerunVisualizer(config, q_init, application_id=env.name, recording_id=experiment_id, spawn=True)
        if config.enable_visualizer
        else Visualizer(config, None)
    )
    visualizer.log_tamp_world(world)
    return exp_logger, visualizer, timer, world


def run_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
):
    """Overall cuTAMP algorithm implementation."""
    exp_logger, visualizer, timer, world = setup_cutamp(env, config, q_init, experiment_id)
    particle_initializer = ParticleInitializer(world, config)

    # Task plan generator
    _log.info(f"Initial State: {world.initial_state}")
    _log.info(f"Goal State: {world.goal_state}")
    all_operators = all_t1_operators
    with timer.time("get_plan_generator", log_callback=_log.info):
        arm_affinity_priority = make_arm_affinity_priority_fn(world)
        plan_gen = task_plan_generator(
            world.initial_state,
            world.goal_state,
            operators=all_operators,
            ground_op_priority_fn=arm_affinity_priority,
        )

    # Sample initial plans and particles
    found_solution_initially = False
    num_skipped_plans = 0
    with timer.time("sample_initial_plans", log_callback=_log.info):
        plan_queue: List[dict] = []
        plan_count = 0
        for idx in range(config.num_initial_plans):
            try:
                plan_info, has_solution = sample_plan_skeleton(
                    plan_gen, world, config, timer, idx, constraint_checker, cost_reducer, particle_initializer
                )
                if plan_info is None:
                    _log.debug("failed subgraph, skipping...")
                    num_skipped_plans += 1
                    continue
            except StopIteration:
                _log.info("Ran out of plans to sample")
                break
            plan_queue.append(plan_info)
            if has_solution:
                found_solution_initially = True
                break
            plan_count += 1

    def sort_plans():
        with timer.time("sort_plans"):
            plan_queue.sort(key=lambda x: x["heuristic"])

    sort_plans()
    _log.info(f"Num plans: {len(plan_queue)}, num skipped: {num_skipped_plans}")
    overall_metrics = {
        "num_optimized_plans": 0,
        "num_initial_plans": plan_count,
        "num_skipped_plans": num_skipped_plans,
        "num_satisfying_final": 0,
        "num_particles": config.num_particles,
        "best_cost": float("inf"),
        "best_soft_cost": float("inf"),
    }
    curobo_plan = None
    found_solution = False
    particle_optimizer = ParticleOptimizer(config, cost_reducer, constraint_checker)
    timer.start("first_solution")
    if found_solution_initially:
        found_solution = True
        timer.stop("first_solution")

    # Optimization loop for each skeleton and its particles
    timer.start("start_optimization")
    for idx, plan_info in enumerate(plan_queue):
        opt_iter = idx + 1
        should_break = False
        plan_skeleton = plan_info["plan_skeleton"]
        _log.info(
            f"[Opt {opt_iter}] Optimizing plan {[op.name for op in plan_skeleton]}, plan idx = {plan_info['idx']}, "
            f"heuristic = {plan_info['heuristic']:.2f}"
        )
        has_satisfying, metrics, time_exceeded = particle_optimizer(plan_info, timer, visualizer)
        if metrics["best_cost"] is not None:
            overall_metrics["best_cost"] = min(overall_metrics["best_cost"], metrics["best_cost"])
        if metrics["best_soft_cost"] is not None:
            overall_metrics["best_soft_cost"] = min(overall_metrics["best_soft_cost"], metrics["best_soft_cost"])
        if time_exceeded:
            _log.info(f"Max loop duration reached, stopping optimization")
            should_break = True
        exp_logger.log_dict(f"optimization/opt_{opt_iter:04d}", metrics)

        overall_metrics["num_optimized_plans"] += 1
        if has_satisfying:
            if config.curobo_plan:
                candidates = get_top_satisfying_particles(
                    plan_info, config, constraint_checker, cost_reducer,
                    n=config.motion_plan_max_retries,
                )
                curobo_plan, skip_plan = _motion_plan_with_retries(
                    plan_info, candidates, world, config, timer, visualizer,
                )
                if skip_plan or curobo_plan is None:
                    continue
            found_solution = True
            overall_metrics["num_satisfying_final"] = metrics["num_satisfying_final"]
            overall_metrics["final_plan_skeleton"] = [str(op) for op in plan_skeleton]
            _log.debug(f"Total num satisfying {metrics['num_satisfying_final']}")
            should_break = True

        if should_break:
            break

        # TODO: complete version of our algorithm that adds additional skeletons to the queue, resorts, revisits
        #  skeletons, etc.
        # new_plan_info = sample_plan_skeleton()
        # if new_plan_info is not None:
        #     plan_queue.append(new_plan_info)
        #     sort_plans()

    opt_elapsed = timer.stop("start_optimization")
    _log.debug(f"Optimization loop took roughly {opt_elapsed:.2f}s")
    if not found_solution:
        _log.warning("No satisfying particles found after optimizing all plans")
    _log.debug(f"Best cost: {overall_metrics['best_cost']:.4f}, soft cost: {overall_metrics['best_soft_cost']:.4f}")

    # Dump metrics out
    overall_metrics["found_solution"] = found_solution
    exp_logger.log_dict("overall_metrics", overall_metrics)
    exp_logger.log_dict("timer_metrics", timer.get_summaries())

    # Log constraint and cost multipliers
    exp_logger.log_dict("multipliers", cost_reducer.cost_config)
    exp_logger.log_dict("tolerances", constraint_checker.constraint_config)
    return curobo_plan, overall_metrics["num_satisfying_final"]
