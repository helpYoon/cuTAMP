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
import json
import logging
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from threading import local
from time import monotonic
from typing import Callable, List, Optional, Tuple

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
from cutamp.grasps import GRASP_SCORE_CUTOFF, grasp_sets, start_sampler_load, wait_for_sampler_load
from cutamp.motion_solver import build_motion_planner, play_curobo_motion, solve_curobo
from cutamp.optimize_plan import ParticleOptimizer
from cutamp.particle_initialization import (
    IKTol, NoGraspCandidates, ParticleInitializer, choice_stats, pick_place_confs, worst_posture,
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
) -> Tuple[List[dict], List[int]]:
    """Order feasible motion candidates by worst manipulation CoM distance, then secondary effort; return them with
    their particle indices.

    Disabling centering selects the grasp-group retry order.
    """
    particles = plan_info["particles"]
    with torch.no_grad():
        rollout = plan_info["rollout_fn"](particles)
        cost_dict = plan_info["cost_fn"](rollout, particles)
        satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
        if not satisfying_mask.any():
            raise RuntimeError("No satisfying particles found")
        sat = {k: v[satisfying_mask] for k, v in particles.items()}
        s = int(satisfying_mask.sum())
        rows, all_hi = [], torch.ones(s, dtype=torch.bool, device=satisfying_mask.device)
        for c in plan_info["grasp_choice"]:
            pool = c.grasps.to(sat[c.grasp])
            match = (sat[c.grasp][:, None, :] == pool[None]).all(-1)  # [s, m]
            if not match.any(-1).all():
                raise RuntimeError(f"A satisfying particle's {c.grasp} matches no row of its grasp pool")
            row = match.long().argmax(-1)  # the first matching row
            rows.append(row)
            all_hi &= (c.scores.to(row.device)[row] >= GRASP_SCORE_CUTOFF).to(all_hi.device)
        center = cost_dict.get("soft", {}).get("max_center_distance_m")
        if center is not None:
            worst = center[satisfying_mask]
            effort = cost_reducer.secondary_effort(cost_dict)[satisfying_mask]
            by_effort = torch.argsort(effort, stable=True)
            order = by_effort[torch.argsort(worst[by_effort], stable=True)[:n]].tolist()
        else:
            pool_idx = torch.stack(rows, -1) if rows else torch.zeros(s, 0, dtype=torch.long)
            confs = pick_place_confs(plan_info["plan_skeleton"])
            if confs:
                tilt, travel, margin = worst_posture(plan_info["rollout_fn"].world, [sat[name] for name in confs])
            else:
                tilt = travel = margin = torch.zeros(s)
            soft = cost_reducer.soft_costs(cost_dict)[satisfying_mask]
            order = grasp_group_order(pool_idx, soft, all_hi, tilt, travel, margin, n)
    candidates = [{k: v[i].detach().clone() for k, v in sat.items()} for i in order]
    return candidates, satisfying_mask.nonzero(as_tuple=True)[0][order].tolist()


def sample_plan_skeleton(
    plan_gen,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    plan_count: int,
    constraint_checker: ConstraintChecker,
    particle_initializer: ParticleInitializer,
) -> Tuple[dict, bool]:
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


# Start plan_cspace with two attempts per segment. Allow twenty after a budget-limited failure.
# With centering enabled each candidate gets both budgets before a lower-ranked candidate can win.
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
    deadline: float | None = None,
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

    A candidate first gets 2 plan_cspace attempts per segment. If it runs out
    (``e.out_of_attempts``), centering gives it the full budget of 20 before
    trying the next candidate. Disabling centering gives all candidates their
    cheap pass first. Any other failure (a balance-gate rejection, an
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
    last_err, tried, failures = None, 0, []
    ranks = range(len(candidates))
    if getattr(config, "com_center_weight", 0) != 0:
        checks = ((rank, budget) for rank in ranks for budget in _PLAN_ATTEMPTS)
    else:
        checks = ((rank, budget) for budget in _PLAN_ATTEMPTS for rank in ranks)
    failed_prefixes = {budget: [] for budget in _PLAN_ATTEMPTS}
    for rank, max_attempts in checks:
        if deadline is not None and monotonic() >= deadline:
            return None, None, tried, None, failures
        particle = candidates[rank]
        if any(all(torch.equal(particle[k], v) for k, v in f.items()) for f in failed_prefixes[max_attempts]):
            _log.info(f"Skipping particle rank {rank}/{len(candidates) - 1}: repeats a failed plan")
            continue
        tried += 1
        try:
            curobo_plan, traj_metrics = solve_curobo(
                plan_info, particle, world, config, timer, visualizer,
                get_planner=get_planner, ik_tol=ik_tol, max_attempts=max_attempts,
            )
            if rank > 0:
                _log.warning(f"Motion plan succeeded on retry rank {rank}/{len(candidates) - 1}")
            return curobo_plan, traj_metrics, tried, rank, failures
        except RuntimeError as e:
            _log.warning(f"Motion plan failed on particle rank {rank}/{len(candidates) - 1}: {e}")
            last_err = e
            failures.append(str(e))
            ops = skeleton[: getattr(e, "op_idx", len(skeleton)) + 1]
            used = {"left_q0"} | {v for op in ops for v in op.values if v in particle}
            prefix = {k: particle[k] for k in used}
            # A matching prefix can benefit from a larger budget only after exhausting attempts.
            for budget in _PLAN_ATTEMPTS:
                if budget <= max_attempts or not getattr(e, "out_of_attempts", False):
                    failed_prefixes[budget].append(prefix)
    _log.error(
        f"Exhausted {len(candidates)} particle retries ({tried} planned); giving up on this plan. "
        f"Last error: {last_err}"
    )
    return None, None, tried, None, failures


def _planned_metrics(world: TAMPWorld, plan_skeleton: PlanSkeleton, candidates: List[dict], rank: int) -> dict:
    """The planned particle's rank, CoM margin (over all its confs) and the Trunk's tilt (radians to degrees) and travel
    (metres), the worst over its pick and place confs (G7; left out when the skeleton has none). ``best_com_margin``
    describes the preferred kinematic candidate. A motion retry can succeed on a different particle."""
    particle = candidates[rank]
    metrics = {"motion_plan_rank": rank, "planned_com_margin": particle_com_margin(world, particle)}
    confs = pick_place_confs(plan_skeleton)
    if confs:
        tilt, travel, _ = worst_posture(world, [particle[name][None] for name in confs])
        metrics.update(planned_trunk_tilt_deg=float(torch.rad2deg(tilt[0])), planned_trunk_travel=float(travel[0]))
    return metrics


def ungrasped(world: TAMPWorld, plan_skeleton: PlanSkeleton) -> list:
    """The objects ``plan_skeleton`` picks that have no grasps in ``world.grasps`` yet, each once, in plan order. A
    sampler runs for the parameters of a skeleton, so a movable that no skeleton picks never gets grasps."""
    names = dict.fromkeys(op.values[0] for op in plan_skeleton if op.operator.metadata.action_type == "pick")
    return [world.get_object(name) for name in names if name not in world.grasps]


def setup_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: Optional[List[float]] = None,
    *,
    experiment_id: str,
):
    # cuRobo runs a full gc.collect() before every CUDA graph capture (cuda_graph_util.py:197), 11-27 per problem, and
    # each walks the whole heap: 2.2-2.6 s per problem. Frozen here, once the world is built, and once a skeleton's
    # grasps are made (run_cutamp; GraspGenX's import and grasps are most of the heap), those collections skip it.
    # Frozen objects are still freed by refcount, but frozen garbage, and what it holds, waits for the next problem's
    # unfreeze below or for the one at exit (atexit, above).
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

    # Load TAMP world. It starts with no grasps: run_cutamp makes them for the objects a skeleton picks.
    timer = TorchTimer()
    with timer.time("load_tamp_world", log_callback=_log.info):
        world = TAMPWorld(
            env=env,
            device_cfg=device_cfg,
            robot=robot_container,
            q_init=q_init,
            grasps={},
        )
        check_tamp_world_not_in_collision(world)
    gc.collect()
    gc.freeze()  # the world too, before run_cutamp's planner thread starts capturing graphs

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
    step_inputs=None,
):
    """Return the highest-ranked feasible motion from the first successful skeleton.

    Keyframe-only searches can compare feasible solutions across skeletons.
    """
    if config.step_mode != "off":
        preload = config.curobo_plan
        if preload:
            # Only sampler construction overlaps world setup and search. Grasp generation itself
            # changes process-wide CUDA settings and runs synchronously for the selected objects.
            start_sampler_load()
        try:
            return _run_step_planner(env, config, cost_reducer, constraint_checker, q_init, experiment_id, step_inputs)
        finally:
            if preload:
                wait_for_sampler_load()  # also when no skeleton was reached or setup raised
    if step_inputs is not None:
        raise ValueError("Explicit step inputs require step_mode model")
    # GraspGenX's model loads on a background thread while the robot and the world load and the task planner runs.
    start_sampler_load()
    exp_logger, visualizer, timer, world = setup_cutamp(env, config, q_init, experiment_id=experiment_id)
    # The fixed-feet baseline has no centring: for_refinement resolves its weight to 0. Resolved after setup_cutamp,
    # which validates and logs the configuration as given.
    config = config.for_refinement(stepped=False)
    particle_initializer = ParticleInitializer(world, config, constraint_checker)

    # Task plan generator
    _log.info(f"Initial State: {world.initial_state}")
    _log.info(f"Goal State: {world.goal_state}")
    all_operators = all_t1_operators
    with timer.time("get_plan_generator", log_callback=_log.info):
        arm_affinity_priority = make_arm_affinity_priority_fn(world)
        skeletons = task_plan_generator(
            world.initial_state,
            world.goal_state,
            operators=all_operators,
            ground_op_priority_fn=arm_affinity_priority,
        )
        first = next(skeletons, None)

    # GraspGenX's grasps, for the objects a skeleton picks and for no other: once per object shape and process, timed
    # apart from planning (the timer covers the wait for the model's background load, then the sampling). grasp_sets
    # turns process-wide settings on while it runs, so the first skeleton's grasps are made here, before the motion
    # planner's warmup thread starts, and a later skeleton's wait for that thread to end.
    planner_thread = None
    grasp_counts = {}

    def make_grasps(plan_skeleton: PlanSkeleton) -> None:
        objs = ungrasped(world, plan_skeleton)
        if not objs:
            return
        if planner_thread is not None:
            planner_thread.join()
        with timer.time("generate_grasps", log_callback=_log.info):
            world.grasps.update(grasp_sets(objs, world.device, counts=grasp_counts))  # the full reference pool
        gc.collect()
        gc.freeze()  # GraspGenX's import and the grasps (setup_cutamp), before the next graph capture

    def plan_skeletons():
        if first is not None:
            yield first
        for plan_skeleton in skeletons:
            make_grasps(plan_skeleton)
            yield plan_skeleton

    if first is not None:
        make_grasps(first)
    wait_for_sampler_load()  # also when the first skeleton picks nothing: a capture fails on another thread's CUDA work
    plan_gen = plan_skeletons()

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

    # Sample initial plans and particles
    found_solution_initially = False
    plan_count = 0

    def initialized_skeletons():
        nonlocal plan_count
        for idx in range(config.num_initial_plans):
            try:
                plan_info, has_solution = sample_plan_skeleton(
                    plan_gen, world, config, timer, idx, constraint_checker, particle_initializer
                )
            except StopIteration:
                return
            plan_count += 1
            yield plan_info, has_solution

    pending_plans = initialized_skeletons()
    with timer.time("sample_initial_plans", log_callback=_log.info):
        plan_queue: List[dict] = []
        for plan_info, has_solution in pending_plans:
            plan_queue.append(plan_info)
            if has_solution:
                found_solution_initially = True
                break

    with timer.time("sort_plans"):
        plan_queue.sort(key=lambda x: x["heuristic"])
    _log.info(f"Num plans: {len(plan_queue)}")
    overall_metrics = {
        "grasp_proposals_by_object": grasp_counts,
        "num_optimized_plans": 0,
        "num_initial_plans": plan_count,
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
        found_solution = not config.curobo_plan
        timer.stop("first_solution")

    # Optimization loop for each skeleton and its particles
    timer.start("start_optimization")
    # Only the Adam loop checks max_loop_dur (time since start_optimization): the search ends after the skeleton whose
    # loop runs out.
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
                candidates, _ = get_top_satisfying_particles(
                    plan_info, constraint_checker, cost_reducer, n=config.motion_plan_max_retries,
                )
                curobo_plan, traj_metrics, attempts, rank, failures = _motion_plan_with_retries(
                    plan_info, candidates, world, config, timer, visualizer,
                    get_planner=get_planner, ik_tol=particle_initializer.ik_tol,
                )
                overall_metrics["motion_plan_attempts"] = overall_metrics.get("motion_plan_attempts", 0) + attempts
                overall_metrics["motion_plan_failures"] += failures
                if curobo_plan is None:
                    continue
                overall_metrics.update(traj_metrics)
                overall_metrics.update(_planned_metrics(world, plan_skeleton, candidates, rank))
                overall_metrics["motion_plan_success"] = True
            overall_metrics["num_satisfying_final"] = metrics["num_satisfying_final"]
            overall_metrics["final_plan_skeleton"] = [str(op) for op in plan_skeleton]
            found_solution = True
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


_PREDICTOR_CACHE = local()  # A captured graph's mutable replay buffers belong to its worker.


def _file_version(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path.resolve()), None)
    return (str(path.resolve()), (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size))


def _load_step_search(config, *, applicable=True):
    from cutamp.landing_accuracy import LANDING_ACCURACY
    from cutamp.reachability import REACH_MAP

    accuracy = _file_version(LANDING_ACCURACY) if applicable else None
    tables, notes = _step_tables(_file_version(REACH_MAP), accuracy, config.step_lag)
    return tables, (
        dict(notes) if applicable else dict(notes, accuracy_status="dynamics outside the profiled condition")
    )


@lru_cache(maxsize=4)
def _step_tables(map_version, accuracy_version, lag):
    from cutamp.reachability import ReachMap
    from cutamp.step_search import StepSearch

    path = Path(map_version[0])
    if map_version[1] is None:
        raise FileNotFoundError(f"Reach map is unavailable at {path}")
    try:
        atlas = ReachMap.load(path, nominal_only=True)
    except (ValueError, KeyError, OSError) as e:
        raise ValueError(f"Reach map at {path} failed to load: {e}") from e
    accuracy = Path(accuracy_version[0]) if accuracy_version else None
    notes = {"map": str(path), "accuracy": str(accuracy) if accuracy is not None else None}
    records = None
    if accuracy is not None and accuracy.is_file():
        try:
            records = json.loads(accuracy.read_text())
            return StepSearch(atlas, accuracy=records, lag=lag), notes
        except (ValueError, KeyError) as e:
            notes["accuracy_status"] = str(e)
    else:
        notes["accuracy_status"] = "unavailable"
    return StepSearch(atlas, lag=lag), notes


def _load_predictor(config):
    from cutamp.landing_accuracy import LANDING_BUNDLE
    from cutamp.landing_predictor import LandingPredictor
    from cutamp.step_sequence_model import DEFAULT_STEP_SEQUENCE_MODEL, load_step_sequence_model

    path, sequence_path = LANDING_BUNDLE, DEFAULT_STEP_SEQUENCE_MODEL
    version = _file_version(path)
    if version[1] is None or not path.is_file():
        raise FileNotFoundError(f"Landing bundle is unavailable at {path}")
    sequence_version = _file_version(sequence_path)
    if sequence_version[1] is None or not sequence_path.is_file():
        raise FileNotFoundError(f"Step sequence bundle is unavailable at {sequence_path}")
    if config.cuda_graphs:
        stream = torch.cuda.current_stream()
        key = (version, sequence_version, stream.device, stream.cuda_stream)
        if getattr(_PREDICTOR_CACHE, "key", None) == key:
            return _PREDICTOR_CACHE.model
    predictor = LandingPredictor(path, use_cuda_graph=config.cuda_graphs)
    predictor.sequence_model = load_step_sequence_model(
        sequence_path, predictor.device, expected_deployment=predictor.metadata["deployment_sha256"]
    )
    if config.cuda_graphs:
        _PREDICTOR_CACHE.model = predictor
        _PREDICTOR_CACHE.key = key
    return predictor


def _refine_step_schedule(proposal, world, config, *args, **kwargs):
    # The model and its CUDA deployment are optional for fixed-feet runs.
    from cutamp.step_ik import stance_ik_graphs
    from cutamp.step_particles import refine_step_particles

    with stance_ik_graphs(config.cuda_graphs):
        return refine_step_particles(proposal, world, config, *args, **kwargs)


def _refine_standing_schedule(
    proposal, world, config, reducer, checker, timer, visualizer, initializer, optimizer, get_planner, *, deadline=None
):
    from cutamp.com_polygon_cost import margin_from_support_planes
    from cutamp.step_planner import StepResult, StepRow, standing_endpoint_flags

    plan = proposal.skeleton
    try:
        info, _ = sample_plan_skeleton(iter((plan,)), world, config, timer, 0, checker, initializer)
    except NoGraspCandidates as e:
        return StepResult([], None, "initialization_failed", str(e))
    if deadline is not None and monotonic() >= deadline:
        return StepResult([], None, "budget_exhausted", "standing initialization reached the time limit")
    optimizer(info, timer, visualizer, deadline=deadline)
    particles = info["particles"]
    with torch.no_grad():
        rollout = info["rollout_fn"](particles)
        cost_fn = info["cost_fn"]
        costs = cost_fn(rollout, particles)
        masks = checker.get_full_mask(costs)
        margin = margin_from_support_planes(rollout["com_xy"], *cost_fn.support_planes)
        center_distances = torch.linalg.vector_norm(rollout["com_xy"] - cost_fn.support_centroid, dim=-1)
        endpoint = standing_endpoint_flags(masks, rollout, plan, cost_fn)
        good = checker.get_mask(costs, verbose=False)
        if not good.any():
            return StepResult([], endpoint, "unresolved")
        soft = reducer.soft_costs(costs)
        effort = reducer.secondary_effort(costs)
        total = reducer(costs)
        indices = good.nonzero(as_tuple=True)[0].tolist()
        rows = [
            StepRow(
                i,
                {name: v[i].detach().clone() for name, v in particles.items()},
                float(soft[i]),
                {
                    "com_margins_m": margin[i].tolist(),
                    "com_center_distances_m": center_distances[i].tolist(),
                    "support_centroid_xy_m": cost_fn.support_centroid.tolist(),
                    "min_com_margin_m": float(margin[i].min()),
                    "secondary_effort": float(effort[i]),
                    "cost": float(total[i]),
                },
            )
            for i in indices
        ]
    diagnostics = {}
    if config.curobo_plan:
        candidates, row_ids = get_top_satisfying_particles(info, checker, reducer, config.motion_plan_max_retries)
        motion, traj, tries, rank, failed = _motion_plan_with_retries(
            info,
            candidates,
            world,
            config,
            timer,
            None if config.com_center_weight != 0 else visualizer,
            get_planner=get_planner,
            ik_tol=initializer.ik_tol,
            deadline=deadline,
        )
        diagnostics = {"motion_plan_attempts": tries, "motion_plan_failures": failed}
        if motion is not None:
            winning = next(row for row in rows if row.index == row_ids[rank])
            winning.motion_plan = motion
            winning.metrics.update(traj or {}, **_planned_metrics(world, plan, candidates, rank))
    return StepResult(rows, endpoint, "standing_feasible", diagnostics=diagnostics)


def _run_step_planner(env, config, reducer, checker, q_init, experiment_id, inputs):
    start = monotonic()
    from cutamp.step_planner import StepInputs, StepResult, furthest_endpoint

    validate_tamp_config(config)
    inputs = inputs if inputs is not None else StepInputs()
    predictor = None
    if (getattr(env, "robot_start", "home") == "synthetic_standing"
            and q_init is None and inputs.initial_state is None):
        from cutamp.landing_predictor import nominal_landing_dynamics
        from cutamp.step_validation import StepStateAdapter

        predictor = _load_predictor(config)
        context = nominal_landing_dynamics(predictor.model.home_joints, 1, config.step_lag)
        state = predictor.standing_start(context)
        q, _ = StepStateAdapter(predictor).planner(state.tensors[:6])
        q_init = q[0].tolist()
        inputs = replace(inputs, initial_state=state)
    budget = config.max_loop_dur
    logger, visualizer, timer, world = setup_cutamp(env, config, q_init, experiment_id=experiment_id)
    deadline = start + budget if budget is not None else None
    timer.start("start_optimization")
    incoming = None

    def proposals():
        nonlocal incoming
        from cutamp.task_planning.arm_step_search import arm_step_skeletons

        load_start = monotonic()
        search, cache_info = _load_step_search(config, applicable=inputs.matches_accuracy_context(config.step_lag))
        metrics["step_search"].update(cache_info, load_seconds=monotonic() - load_start)
        if deadline is not None and monotonic() >= deadline:
            return
        soles = world.stance_soles
        yaw = torch.atan2(soles[..., 1, 0], soles[..., 0, 0])
        incoming = torch.cat((soles[..., :2, 3], yaw[:, None]), -1).detach().cpu().tolist()
        admitted = {}
        for proposal in arm_step_skeletons(search, world, all_t1_operators, incoming,
                                           max_base_plans=config.num_initial_plans, admitted=admitted, deadline=deadline):
            metrics["num_symbolic_skeletons"] = len(admitted)
            if step_cap is None or proposal.step_count <= step_cap:
                yield proposal

    metrics = {
        "found_solution": False,
        "status": "unresolved",
        "num_satisfying_final": 0,
        "num_particles": config.num_particles,
        "num_optimized_plans": 0,
        "num_symbolic_skeletons": 0,
        "step_attempts": [],
        "step_search": {},
        "search_budget_s": budget,
        "improvement_budget_s": None,
        "search_budget_start": "planner_entry",
        "grasp_generation_calls": 0,
        "grasp_proposals_by_object": {},
        "best_cost": None,
        "best_soft_cost": None,
        "time_to_first_accepted_s": None,
    }
    if config.curobo_plan:
        metrics.update(motion_plan_success=False, motion_plan_failures=[], motion_plan_attempts=0)
    incumbent = None
    center_first = config.com_center_weight != 0

    def plan_key(row, step_count):
        if center_first:
            return row.max_center_distance_m, step_count, row.effort
        return 0.0, step_count, row.soft_cost

    winning_motion = None
    winning_evidence = None
    selected_metrics = {}
    initializer = optimizer = planner = None
    unavailable = None

    def get_planner():
        nonlocal planner
        if planner is None:
            planner = build_motion_planner(world)
        return planner

    def make_grasps(skeleton):
        objs = ungrasped(world, skeleton)
        if objs:
            try:
                with timer.time("generate_grasps"):
                    world.grasps.update(grasp_sets(objs, world.device, max_grasps=min(64, config.num_particles),
                                                  counts=metrics["grasp_proposals_by_object"]))
                metrics["grasp_generation_calls"] += 1
            finally:
                wait_for_sampler_load()  # no background CUDA work when the selected solve captures graphs
            gc.collect()
            gc.freeze()
        else:
            wait_for_sampler_load()  # a caller may have already started a process-wide preload

    candidates = proposals()
    while True:
        if deadline is not None and monotonic() >= deadline:
            break
        rank_start = monotonic()
        # A shorter schedule can rule out longer ones only at the lower bound of the primary objective.
        step_cap = incumbent[1] if incumbent is not None and incumbent[0] == 0 else None
        proposal = next(candidates, None)
        if proposal is None:
            break
        if deadline is not None and monotonic() >= deadline:
            break
        attempt = {
            "skeleton": [op.name for op in proposal.skeleton],
            "orders": list(proposal.orders),
            "step_count": proposal.step_count,
            "heuristic_cost": proposal.cost,
            "rank_seconds": monotonic() - rank_start,
        }
        attempt["arm_approach_m"] = proposal.arm_approach_m
        distances = [e.distance_m for e in proposal.seed.schedule.edges]
        attempt["missing_reach_evidence"] = None in distances
        worst_gap = max((distance for distance in distances if distance is not None), default=0.)
        attempt["search_priority"] = [proposal.arm_approach_m + worst_gap, *proposal.cost]
        attempt["gaps"] = [
            {
                "incoming_soles": incoming if i == 0 else proposal.seed.schedule.edges[i - 1].soles,
                "commands": e.commands,
                "outgoing_soles": e.soles,
                "posture": e.posture,
                "distance_m": e.distance_m,
                "accuracy": e.accuracy,
                "passes": e.passes,
                "trials": e.trials,
                "cost": e.cost,
            }
            for i, e in enumerate(proposal.seed.schedule.edges)
        ]
        metrics["step_attempts"].append(attempt)
        if proposal.step_count and predictor is None and unavailable is None:
            try:
                predictor = _load_predictor(config)
            except (FileNotFoundError, RuntimeError, ValueError, KeyError, OSError) as e:
                unavailable = str(e)
        if proposal.step_count and unavailable is not None:
            result = StepResult([], None, "unavailable", unavailable)
        else:
            if deadline is not None and monotonic() >= deadline:
                attempt["status"] = "budget_exhausted"
                break
            make_grasps(proposal.skeleton)
            if deadline is not None and monotonic() >= deadline:
                attempt["status"] = "budget_exhausted"
                break
            metrics["num_optimized_plans"] += 1
            if proposal.step_count:
                try:
                    result = _refine_step_schedule(
                        proposal,
                        world,
                        config,
                        reducer,
                        checker,
                        inputs=inputs,
                        predictor=predictor,
                        timer=timer,
                        visualizer=None if center_first else visualizer,
                        deadline=deadline,
                    )
                except NoGraspCandidates as e:
                    result = StepResult([], None, "initialization_failed", str(e))
            else:
                if initializer is None:
                    initializer = ParticleInitializer(world, config, checker)
                    optimizer = ParticleOptimizer(config, reducer, checker)
                result = _refine_standing_schedule(
                    proposal,
                    world,
                    config,
                    reducer,
                    checker,
                    timer,
                    visualizer,
                    initializer,
                    optimizer,
                    get_planner,
                    deadline=deadline,
                )
        attempt.update(
            status=result.status,
            reason=result.reason,
            diagnostics=result.diagnostics,
            furthest_endpoint=furthest_endpoint(result.endpoint_ok),
            particle_rows=len(result.rows),
        )
        if config.curobo_plan:
            metrics["motion_plan_attempts"] += result.diagnostics.get("motion_plan_attempts", 0)
            metrics["motion_plan_failures"] += result.diagnostics.get("motion_plan_failures", [])
        required_level = "standing_feasible" if not proposal.step_count else "model_feasible"
        accepted = [
            row
            for row in result.rows
            if result.status == required_level and (not config.curobo_plan or row.motion_plan is not None)
        ]
        attempt["accepted_rows"] = len(accepted)
        if not accepted:
            if result.rows:
                # Keep evidence for a feasible model prediction whose subsequent motion was unavailable.
                best_model = min(result.rows, key=lambda row: plan_key(row, proposal.step_count))
                attempt["particle_candidate"] = dict(
                    best_model.metrics, row_index=best_model.index, soft_cost=best_model.soft_cost
                )
            continue
        row = min(accepted, key=lambda r: plan_key(r, proposal.step_count))
        key = plan_key(row, proposal.step_count)
        if incumbent is None or key < incumbent:
            if incumbent is None:
                metrics["time_to_first_accepted_s"] = monotonic() - start
            incumbent = key
            winning_motion = row.motion_plan
            winning_evidence = row.evidence
            selected = {
                name: value.detach().cpu().tolist() if torch.is_tensor(value) else value
                for name, value in row.particle.items()
            }
            selected_metrics = dict(row.metrics)
            if center_first:
                selected_metrics["max_center_distance_m"] = row.max_center_distance_m
            if "best_com_margin" in selected_metrics:
                selected_metrics.setdefault("min_com_margin_m", selected_metrics["best_com_margin"])
            status = result.status
            if config.curobo_plan:
                status = "model_predicted_motion" if row.metrics.get("motion_validation") == "model_predicted" else "motion_feasible"
            selected_metrics.update(
                found_solution=True,
                status=status,
                row_index=row.index,
                best_cost=row.metrics.get("cost", row.soft_cost),
                best_soft_cost=row.soft_cost,
                num_satisfying_final=len(accepted),
                gap_schedule=list(proposal.orders),
                total_steps=proposal.step_count,
                final_plan_skeleton=attempt["skeleton"],
                selected_particle=selected,
            )
            if config.curobo_plan:
                metrics["motion_plan_success"] = True
                break
    search_end = monotonic()
    timer.stop("start_optimization")
    metrics.update(selected_metrics)
    if winning_evidence is not None:
        artifact = logger.log_dict("model_evidence", winning_evidence())
        metrics["model_evidence_path"] = str(artifact.resolve())
        del winning_evidence
    metrics["time_limit_reached"] = deadline is not None and search_end >= deadline
    metrics["effective_search_budget_s"] = budget
    if not metrics["time_limit_reached"] and metrics["step_attempts"] and all(
        a["status"] == "unavailable" for a in metrics["step_attempts"]
    ):
        metrics["status"] = "unavailable"
    metrics["search_elapsed_s"] = search_end - start
    metrics["elapsed_s"] = monotonic() - start
    logger.log_dict("overall_metrics", metrics)
    logger.log_dict("timer_metrics", timer.get_summaries())
    if center_first and winning_motion is not None:
        play_curobo_motion(world, winning_motion, visualizer)
    return winning_motion, metrics["num_satisfying_final"], metrics
