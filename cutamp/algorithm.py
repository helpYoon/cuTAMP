# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Core cuTAMP algorithm implementation."""

import atexit
import gc
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
from cutamp.grasps import GRASP_SCORE_CUTOFF, grasp_sets
from cutamp.motion_solver import build_motion_planner, solve_curobo
from cutamp.optimize_plan import ParticleOptimizer
from cutamp.particle_initialization import (
    IKTol, ParticleInitializer, choice_stats, pick_place_confs, worst_posture,
)
from cutamp.robots import REAL_LEGS, load_t1_container
from cutamp.rollout import RolloutFunction
from cutamp.t1_domain import all_t1_operators
from cutamp.tamp_world import TAMPWorld, check_tamp_world_not_in_collision
from cutamp.task_planning import PlanSkeleton, task_plan_generator
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import RerunVisualizer, Visualizer

_log = logging.getLogger(__name__)
# setup_cutamp freezes the heap. Unfrozen at exit, the interpreter's collection before module teardown frees what frozen
# garbage still holds: a MotionPlanner, whose __del__ raises ImportError ("Exception ignored") once teardown has begun.
atexit.register(gc.unfreeze)


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


def grasp_group_order(pool_idx, soft, all_hi, tilt, travel, margin, n: int) -> List[int]:
    """Try order, as positions into the ``s`` satisfying particles, of ``pool_idx`` ``[s, P]`` (each particle's pool
    row per pick); ``soft``, ``all_hi``, ``tilt``, ``travel`` and ``margin`` are ``[s]``. A group is the particles with
    equal ``pool_idx`` rows, ordered inside by ``(soft, position)``. Groups go by their first particle's
    ``(not all_hi, tilt, travel, -margin)``, then by ``tuple(pool_idx)``. The tries are every group's first particle in
    that order, then every group's second, and so on, cut to ``n`` (G7, G8)."""
    soft, hi, tilt, travel, margin = (t.tolist() for t in (soft, all_hi, tilt, travel, margin))
    groups = {}
    for i, row in enumerate(pool_idx.tolist()):
        groups.setdefault(tuple(row), []).append(i)
    ranked = []
    for row, members in groups.items():
        members.sort(key=lambda i: (soft[i], i))
        f = members[0]
        ranked.append(((not hi[f], tilt[f], travel[f], -margin[f]), row, members))
    ranked.sort(key=lambda g: g[:2])
    rounds = max(len(g[2]) for g in ranked)
    return [g[2][r] for r in range(rounds) for g in ranked if r < len(g[2])][:n]


def get_top_satisfying_particles(
    plan_info: dict,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    n: int,
) -> List[dict]:
    """Up to ``n`` satisfying particles in motion-plan try order: grouped by grasp and taken round the groups, each
    group's particles by ascending soft cost (``grasp_group_order``). The posture measures are the particle's after
    the optimizer. May return < n if fewer particles satisfy constraints. Used by the motion-plan retry loop: if
    solve_curobo fails on a try, retry on the next.
    """
    particles = plan_info["particles"]
    with torch.no_grad():
        rollout = plan_info["rollout_fn"](particles)
        cost_dict = plan_info["cost_fn"](rollout, particles)
        satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
        if not satisfying_mask.any():
            raise RuntimeError("No satisfying particles found")
        soft_costs = cost_reducer.soft_costs(cost_dict)
        sat = {k: v[satisfying_mask] for k, v in particles.items()}
        s = int(satisfying_mask.sum())
        rows, all_hi = [], torch.ones(s, dtype=torch.bool, device=soft_costs.device)
        for c in plan_info["grasp_choice"]:
            pool = c.grasps.to(sat[c.grasp])
            match = (sat[c.grasp][:, None, :] == pool[None]).all(-1)  # [s, m]
            if not match.any(-1).all():
                raise RuntimeError(f"A satisfying particle's {c.grasp} matches no row of its grasp pool")
            row = match.long().argmax(-1)  # the first matching row
            rows.append(row)
            all_hi &= (c.scores.to(row.device)[row] >= GRASP_SCORE_CUTOFF).to(all_hi.device)
        pool_idx = torch.stack(rows, -1) if rows else torch.zeros(s, 0, dtype=torch.long)
        confs = pick_place_confs(plan_info["plan_skeleton"])
        if confs:
            tilt, travel, margin = worst_posture(plan_info["rollout_fn"].world, [sat[name] for name in confs])
        else:
            tilt = travel = margin = torch.zeros(s)
        order = grasp_group_order(pool_idx, soft_costs[satisfying_mask], all_hi, tilt, travel, margin, n)
    return [{k: v[i].detach().clone() for k, v in sat.items()} for i in order]


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
        "grasp_choice": particle_initializer.grasp_choice,
    }

    _log.debug(
        f"[Plan {plan_count + 1}] {plan_info['num_satisfying']}/{config.num_particles} satisfying, "
        f"heuristic = {plan_info['heuristic']}"
    )
    return plan_info, num_satisfying > 0


# plan_cspace attempts per segment, one pass over the candidates each. On blocks_t1 (2026-11-07, 13 problems) every
# segment that planned did so on its first attempt (223 of 223), while each failing one spent all of cuRobo's 20 at
# 1.1-4.0 s, against 0.06 s for a success. The 20 run only when no candidate plans on 2 (one of them graph-seeded).
_PLAN_ATTEMPTS = (2, 20)


def _motion_plan_with_retries(
    plan_info: dict,
    candidates: list,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer,
    get_planner=None,
    *,
    ik_tol: IKTol,
) -> Tuple[Optional[dict], Optional[dict], int, Optional[int], List[str]]:
    """Run ``solve_curobo`` on the top-N satisfying candidates; return on first success. ``get_planner`` and
    ``ik_tol`` (the run's IK tolerances, ``ParticleInitializer.ik_tol``) go to ``solve_curobo``.

    cuRobo's plan_cspace can fail on a specific (start, goal) pair even after
    its internal retries; a different particle gives a different pair and
    usually succeeds. The same pair usually fails again (its straight-line attempt
    repeats; only the RRT-seeded one draws new samples, the next of the RRT's fixed
    Halton sequence), so a candidate whose values for the failing op and every
    op before it equal a failed candidate's is skipped: particles often share
    confs (on blocks_t1 seed 1, every satisfying particle had the same
    ``left_q1``).

    The candidates get two passes (``_PLAN_ATTEMPTS``): each segment first with
    2 plan_cspace attempts, then, only if no candidate planned, with cuRobo's
    20 for the candidates whose plan_cspace ran out of attempts
    (``e.out_of_attempts``). Any other failure (a balance-gate rejection, an
    approach-IK or line miss) is not planned again: plan_cspace returns its
    first success, so a bigger budget replans the same trajectory.

    Returns ``(curobo_plan, traj_metrics, attempts, rank, failures)``:
      - ``curobo_plan``: the successful plan, or ``None`` if all candidates
        failed.
      - ``traj_metrics``: the plan's metrics dict from ``solve_curobo`` (its
        duration and the balance gate's retimes and margins), or ``None`` if
        all candidates failed.
      - ``attempts``: number of candidates actually planned (skips excluded),
        over both passes.
      - ``rank``: the index in ``candidates`` of the planned particle, or
        ``None`` if none was planned.
      - ``failures``: each planned candidate's failure message, in order, over
        both passes: a failed trajopt's reads "cspace-anchored plan failed for
        <op> <phase>", and only the balance gate fails a segment on torque,
        naming the joint. The caller records them in
        ``overall_metrics["motion_plan_failures"]``.
    """
    skeleton = plan_info["plan_skeleton"]
    last_err, tried, failures, rejected = None, 0, [], []
    for max_attempts in _PLAN_ATTEMPTS:
        failed = list(rejected)
        for rank, particle in enumerate(candidates):
            if any(all(torch.equal(particle[k], v) for k, v in f.items()) for f in failed):
                _log.info(f"Skipping particle rank {rank}/{len(candidates) - 1}: repeats a failed plan")
                continue
            tried += 1
            try:
                curobo_plan, traj_metrics = solve_curobo(
                    plan_info, particle, world, config, timer, visualizer,
                    get_planner=get_planner, ik_tol=ik_tol, max_attempts=max_attempts,
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
                # No budget changes the rest (a gate rejection, an approach-IK or line miss): plan_cspace returns
                # its first success, so more attempts replan the same trajectory.
                # ponytail: trajopt's success test leaves torque to the gate (t1._trajopt_metrics_dict), so a path
                # the gate can't slow enough (past RETIME_MAX for the ZMP or an effort) ends the candidate here,
                # where cuRobo's torque bound used to send it on to graph-seeded attempts. Unseen on the 13
                # 2026-11-09 checks (at most 3.45x); if studies show "slowdown cap" rejections, replan that segment
                # once with graph seeds first (plan_cspace's enable_graph_attempt=0).
                if not getattr(e, "out_of_attempts", False):
                    rejected.append(failed[-1])
        if max_attempts != _PLAN_ATTEMPTS[-1]:
            _log.warning(f"No candidate planned with {max_attempts} plan_cspace attempts per segment; "
                         f"retrying those that ran out with {_PLAN_ATTEMPTS[-1]}")
    _log.error(
        f"Exhausted {len(candidates)} particle retries ({tried} planned); giving up on this plan. "
        f"Last error: {last_err}"
    )
    return None, None, tried, None, failures


def _planned_metrics(world: TAMPWorld, plan_skeleton: PlanSkeleton, candidates: List[dict], rank: int) -> dict:
    """The planned particle's rank, CoM margin (over all its confs) and the Trunk's tilt (radians to degrees) and travel
    (metres), the worst over its pick and place confs (G7; left out when the skeleton has none). ``best_com_margin``
    is the lowest-soft-cost satisfying particle's (optimize_plan), which need not be the planned one: the tries go
    round the grasp groups, and a retry plans another particle (known-bugs N3)."""
    particle = candidates[rank]
    metrics = {"motion_plan_rank": rank, "planned_com_margin": particle_com_margin(world, particle)}
    confs = pick_place_confs(plan_skeleton)
    if confs:
        tilt, travel, _ = worst_posture(world, [particle[name][None] for name in confs])
        metrics.update(planned_trunk_tilt_deg=float(torch.rad2deg(tilt[0])), planned_trunk_travel=float(travel[0]))
    return metrics


def setup_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: Optional[List[float]] = None,
    *,
    experiment_id: str,
):
    # cuRobo runs a full gc.collect() before every CUDA graph capture (cuda_graph_util.py:197), 11-27 per problem, and
    # each walks the whole heap: 2.2-2.6 s per problem. Frozen here and again once the world is built (GraspGenX's import
    # and grasps are most of the heap), those collections skip it. Frozen objects are still freed by refcount, but frozen
    # garbage, and what it holds, waits for the next problem's unfreeze below or for the one at exit (atexit, above).
    gc.unfreeze()
    gc.collect()
    gc.freeze()

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
    # GraspGenX's grasps, once per object shape and process, timed apart from planning. grasp_sets turns process-wide
    # settings on while it runs, so it comes before the motion planner's warmup thread starts (run_cutamp).
    with timer.time("generate_grasps", log_callback=_log.info):
        grasps = grasp_sets(env.movables, device_cfg.device)
    with timer.time("load_tamp_world", log_callback=_log.info):
        world = TAMPWorld(
            env=env,
            device_cfg=device_cfg,
            robot=robot_container,
            q_init=q_init,
            grasps=grasps,
        )
        check_tamp_world_not_in_collision(world)
    gc.collect()
    gc.freeze()  # the grasps and the world too, before run_cutamp's planner thread starts capturing graphs

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
        overall_metrics["grasp_choice"] = [choice_stats(c) for c in plan_info["grasp_choice"]]
        if has_satisfying:
            if config.curobo_plan:
                candidates = get_top_satisfying_particles(
                    plan_info, constraint_checker, cost_reducer,
                    n=config.motion_plan_max_retries,
                )
                curobo_plan, traj_metrics, attempts, rank, failures = _motion_plan_with_retries(
                    plan_info, candidates, world, config, timer, visualizer,
                    get_planner=get_planner, ik_tol=particle_initializer.ik_tol,
                )
                overall_metrics["motion_plan_success"] = bool(curobo_plan)
                # Over the skeletons planned, like the failures; absent until a skeleton is motion-planned.
                overall_metrics["motion_plan_attempts"] = overall_metrics.get("motion_plan_attempts", 0) + attempts
                overall_metrics["motion_plan_failures"] += failures
                if curobo_plan is None:
                    continue
                overall_metrics.update(traj_metrics)
                overall_metrics.update(_planned_metrics(world, plan_skeleton, candidates, rank))
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
