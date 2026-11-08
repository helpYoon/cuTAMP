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
from typing import Callable, List, Union, Optional, Tuple

import numpy as np
import torch
from curobo.types import DeviceCfg

from cutamp.com_polygon_cost import particle_com_margin
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import TAMPEnvironment
from cutamp.experiment_logger import ExperimentLogger
from cutamp.motion_solver import build_motion_planner, solve_curobo
from cutamp.optimize_plan import ParticleOptimizer
from cutamp.particle_initialization import ParticleInitializer
from cutamp.robots import REAL_LEGS, load_t1_container
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
    rollout_fn = RolloutFunction(plan_skeleton, world)
    cost_fn = CostFunction(plan_skeleton, world, config)
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = rollout_fn(plan_particles)
        cost_dict = cost_fn(rollout, plan_particles)
        heuristic = heuristic_fn(plan_skeleton, cost_dict, constraint_checker)

    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict)
    num_satisfying = satisfying_mask.sum().item()

    plan_info = {
        "idx": plan_count,
        "plan_skeleton": plan_skeleton,
        "particles": plan_particles,
        "rollout_fn": rollout_fn,
        "cost_fn": cost_fn,
        "heuristic": heuristic,
        "num_satisfying": num_satisfying,
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
    get_planner=None,
    ik_tol=None,
) -> Tuple[Optional[dict], Optional[dict], int, Optional[int], List[str]]:
    """Run ``solve_curobo`` on the top-N satisfying candidates; return on first success. ``get_planner`` and
    ``ik_tol`` (the run's IK tolerances, ``ParticleInitializer.ik_tol``) go to ``solve_curobo``.

    cuRobo's plan_cspace can fail on a specific (start, goal) pair even after
    its internal retries; a different particle gives a different pair and
    usually succeeds. The same pair fails the same way (plan_cspace is only given
    the same PRM random state, which build_motion_planner's
    random.seed(0) fixes per process), so a candidate whose values for the failing op and every
    op before it equal a failed candidate's is skipped: particles often share
    confs (on blocks_t1 seed 1, every satisfying particle had the same
    ``left_q1``).

    Returns ``(curobo_plan, traj_metrics, attempts, rank, failures)``:
      - ``curobo_plan``: the successful plan, or ``None`` if all candidates
        failed.
      - ``traj_metrics``: the plan's metrics dict from ``solve_curobo`` (its
        duration and the balance gate's retimes and margins), or ``None`` if
        all candidates failed.
      - ``attempts``: number of candidates actually planned (skips excluded).
      - ``rank``: the index in ``candidates`` of the planned particle, or
        ``None`` if none was planned.
      - ``failures``: each planned candidate's failure message, in order. A
        failed trajopt's message names its reason (design 7.4a). The caller
        records them in ``overall_metrics["motion_plan_failures"]``.
    """
    skeleton = plan_info["plan_skeleton"]
    last_err, tried, failed, failures = None, 0, [], []
    for rank, particle in enumerate(candidates):
        if any(all(torch.equal(particle[k], v) for k, v in f.items()) for f in failed):
            _log.info(f"Skipping particle rank {rank}/{len(candidates) - 1}: repeats a failed plan")
            continue
        tried += 1
        try:
            curobo_plan, traj_metrics = solve_curobo(
                plan_info, particle, world, config, timer, visualizer,
                get_planner=get_planner, ik_tol=ik_tol,
            )
            if rank > 0:
                _log.warning(
                    f"Motion plan succeeded on retry rank {rank}/{len(candidates) - 1}"
                )
            return curobo_plan, traj_metrics, tried, rank, failures
        except RuntimeError as e:
            _log.warning(
                f"Motion plan failed on particle rank {rank}/{len(candidates) - 1}: {e}"
            )
            last_err = e
            failures.append(str(e))
            # ponytail: keys on every value read up to the failing op (all ops if untagged), so
            # a candidate differing only in values that op never sees (a pick conf before a
            # shared retract) still replans; key on the op's own start/goal/world if that matters.
            ops = skeleton[: getattr(e, "op_idx", len(skeleton)) + 1]
            used = {"left_q0"} | {v for op in ops for v in op.values if v in particle}
            failed.append({k: particle[k] for k in used})
    _log.error(
        f"Exhausted {len(candidates)} particle retries ({tried} planned); giving up on this plan. "
        f"Last error: {last_err}"
    )
    return None, None, tried, None, failures


def _planned_metrics(world: TAMPWorld, candidates: List[dict], rank: int) -> dict:
    """The planned particle's rank and CoM margin. ``best_com_margin`` is rank 0's; after a retry the plan comes
    from another particle (known-bugs N3)."""
    return {"motion_plan_rank": rank, "planned_com_margin": particle_com_margin(world, candidates[rank])}


def setup_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: Optional[List[float]] = None,
    *,
    experiment_id: str,
):
    # Validate args and setup experiment logger
    validate_tamp_config(config)
    exp_logger = ExperimentLogger(name=experiment_id, config=config)
    exp_logger.save_env(env)

    # Loading robot can be done offline, so doesn't count towards timing
    device_cfg = DeviceCfg()
    model = REAL_LEGS
    robot_container = load_t1_container(model, device_cfg=device_cfg)

    # Single-MotionPlanner: q_init is a single full-DOF vector of the model's cspace.
    if q_init is None:
        q_init = model.home
    q_init = device_cfg.to_device(q_init)

    # Load TAMP world
    timer = TorchTimer()
    with timer.time("load_tamp_world", log_callback=_log.info):
        world = TAMPWorld(
            env=env,
            device_cfg=device_cfg,
            robot=robot_container,
            q_init=q_init,
        )
        check_tamp_world_not_in_collision(world)

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
    *,
    experiment_id: str,
):
    """Overall cuTAMP algorithm implementation."""
    exp_logger, visualizer, timer, world = setup_cutamp(env, config, q_init, experiment_id=experiment_id)
    particle_initializer = ParticleInitializer(world, config, constraint_checker)

    # Build the trajopt planner on a background thread, overlapped with
    # particle init and the Adam loop. The build runs no solves, so no CUDA graph
    # capture overlaps the main thread (see build_motion_planner). Concurrent
    # cuRobo use is only safe with the per-stream warp wrapper cache
    # (_curobo_internals._patch_warp_streams): warp's stream registry is unlocked, and a
    # fresh wrapper per launch from two threads corrupts the heap. solve_curobo
    # joins via get_planner; its warmup timer then measures only the residual wait.
    # CUDA graphs capture only after this thread ends (planner_busy): a capture fails on another thread's CUDA work.
    get_planner = None
    planner_busy = None
    if config.curobo_plan:
        import threading

        planner_box: dict = {}

        def _build_planner_bg():
            try:
                planner_box["planner"] = build_motion_planner(world)
            except BaseException as e:  # noqa: BLE001 — surfaced at join time
                planner_box["error"] = e

        planner_thread = threading.Thread(
            target=_build_planner_bg, daemon=True, name="motion-planner-warmup",
        )
        planner_thread.start()
        planner_busy = planner_thread.is_alive

        def get_planner():
            planner_thread.join()
            if "error" in planner_box:
                raise planner_box["error"]
            return planner_box["planner"]

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
                    plan_gen, world, config, timer, idx, constraint_checker, particle_initializer
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

    with timer.time("sort_plans"):
        plan_queue.sort(key=lambda x: x["heuristic"])
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
    if config.curobo_plan:
        # Default to False so a run that never reaches motion planning
        # (e.g. no satisfying particles found for any skeleton) still
        # reports motion_plan_success=False instead of an absent key, so
        # every curobo_plan run reports it. Runs without curobo_plan never
        # had a chance to plan, so the key stays absent for them.
        overall_metrics["motion_plan_success"] = False
        overall_metrics["motion_plan_failures"] = []  # every failed candidate's message, over the skeletons planned
    curobo_plan = None
    found_solution = False
    particle_optimizer = ParticleOptimizer(config, cost_reducer, constraint_checker, planner_busy=planner_busy)
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
                curobo_plan, traj_metrics, attempts, rank, failures = _motion_plan_with_retries(
                    plan_info, candidates, world, config, timer, visualizer,
                    get_planner=get_planner, ik_tol=particle_initializer.ik_tol,
                )
                overall_metrics["motion_plan_success"] = bool(curobo_plan)
                overall_metrics["motion_plan_attempts"] = attempts
                overall_metrics["motion_plan_failures"] += failures
                if curobo_plan is None:
                    continue
                overall_metrics.update(traj_metrics)
                overall_metrics.update(_planned_metrics(world, candidates, rank))
            found_solution = True
            overall_metrics["num_satisfying_final"] = metrics["num_satisfying_final"]
            overall_metrics["final_plan_skeleton"] = [str(op) for op in plan_skeleton]
            _log.debug(f"Total num satisfying {metrics['num_satisfying_final']}")
            should_break = True

        if should_break:
            break

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
    return curobo_plan, overall_metrics["num_satisfying_final"], overall_metrics
