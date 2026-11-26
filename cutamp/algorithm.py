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
from dataclasses import replace
from threading import local
from time import monotonic
from typing import List, Optional

import torch
from curobo.types import DeviceCfg

from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import TAMPEnvironment
from cutamp.experiment_logger import ExperimentLogger
from cutamp.grasps import grasp_sets, start_sampler_load, wait_for_sampler_load
from cutamp.motion_solver import play_curobo_motion
from cutamp.robots import REAL_LEGS, load_t1_container
from cutamp.t1_domain import all_t1_operators
from cutamp.tamp_world import TAMPWorld, check_tamp_world_not_in_collision
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import file_version
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import RerunVisualizer, Visualizer

_log = logging.getLogger(__name__)
# setup_cutamp freezes the heap. Unfrozen at exit, the interpreter's collection before module teardown frees what frozen
# garbage still holds: a MotionPlanner, whose __del__ raises ImportError ("Exception ignored") once teardown has begun.
atexit.register(gc.unfreeze)


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
    gc.freeze()  # the world too, before the first graph capture

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


_PREDICTOR_CACHE = local()  # A captured graph's mutable replay buffers belong to its worker.


def _load_predictor(config):
    from cutamp.landing_accuracy import LANDING_BUNDLE
    from cutamp.landing_predictor import LandingPredictor

    path = LANDING_BUNDLE
    version = file_version(path)
    if version[1] is None or not path.is_file():
        raise FileNotFoundError(f"Landing bundle is unavailable at {path}")
    if config.cuda_graphs:
        stream = torch.cuda.current_stream()
        key = (version, stream.device, stream.cuda_stream)
        if getattr(_PREDICTOR_CACHE, "key", None) == key:
            return _PREDICTOR_CACHE.model
    predictor = LandingPredictor(path, use_cuda_graph=config.cuda_graphs)
    if config.cuda_graphs:
        _PREDICTOR_CACHE.model = predictor
        _PREDICTOR_CACHE.key = key
    return predictor


def _refine_step_schedule(proposal, world, config, *args, **kwargs):
    from cutamp.step_ik import stance_ik_graphs
    from cutamp.step_loop import refine_stepped_plan

    with stance_ik_graphs(config.cuda_graphs):
        return refine_stepped_plan(proposal, world, config, *args, **kwargs)


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
        from cutamp.task_planning.plan_search import plan_search
        from cutamp.task_planning.landing_options import load_landing_options

        load_start = monotonic()
        options, cache_info = load_landing_options(config, applicable=inputs.matches_accuracy_context(config.step_lag))
        metrics["step_search"].update(cache_info, load_seconds=monotonic() - load_start)
        if deadline is not None and monotonic() >= deadline:
            return
        soles = world.stance_soles
        yaw = torch.atan2(soles[..., 1, 0], soles[..., 0, 0])
        incoming = torch.cat((soles[..., :2, 3], yaw[:, None]), -1).detach().cpu().tolist()
        admitted = {}
        for proposal in plan_search(options, world, all_t1_operators, incoming,
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
    unavailable = None

    def make_grasps(skeleton):
        objs = ungrasped(world, skeleton)
        if objs:
            try:
                with timer.time("generate_grasps"):
                    world.grasps.update(grasp_sets(objs, world.device, max_grasps=config.num_particles,
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
        distances = [e.distance_m for e in proposal.schedule.edges]
        attempt["missing_reach_evidence"] = None in distances
        worst_gap = max((distance for distance in distances if distance is not None), default=0.)
        attempt["search_priority"] = [proposal.arm_approach_m + worst_gap, *proposal.cost]
        attempt["gaps"] = [
            {
                "incoming_soles": incoming if i == 0 else proposal.schedule.edges[i - 1].soles,
                "commands": e.commands,
                "outgoing_soles": e.soles,
                "posture": e.posture,
                "distance_m": e.distance_m,
                "accuracy": e.accuracy,
                "passes": e.passes,
                "trials": e.trials,
                "cost": e.cost,
            }
            for i, e in enumerate(proposal.schedule.edges)
        ]
        metrics["step_attempts"].append(attempt)
        # Every plan, with or without steps, runs the stepped route, which needs the landing model.
        if predictor is None and unavailable is None:
            try:
                predictor = _load_predictor(config)
            except (FileNotFoundError, RuntimeError, ValueError, KeyError, OSError) as e:
                unavailable = str(e)
        if unavailable is not None:
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
            result = _refine_step_schedule(
                proposal,
                world,
                config,
                reducer,
                checker,
                inputs=inputs,
                predictor=predictor,
                timer=timer,
                visualizer=visualizer,
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
        accepted = [
            row
            for row in result.rows
            if result.status == "model_feasible" and (not config.curobo_plan or row.motion_plan is not None)
        ]
        attempt["accepted_rows"] = len(accepted)
        if not accepted:
            if result.rows:
                # Keep evidence for a feasible model prediction whose subsequent motion was unavailable.
                if center_first:  # picked as the accepted rows are, below
                    best_model = min(result.rows, key=lambda row: row.order)
                else:
                    best_model = min(result.rows, key=lambda row: plan_key(row, proposal.step_count))
                attempt["particle_candidate"] = dict(
                    best_model.metrics, row_index=best_model.index, soft_cost=best_model.soft_cost
                )
            continue
        if center_first:  # within a task plan: 2026-11-23 design 4.5
            row = min(accepted, key=lambda r: r.order)
        else:
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
