"""Refine the shared grasps, placements and commanded landings of a grounded T1 skeleton."""

from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from functools import partial
from itertools import islice
from time import monotonic

import roma
import torch

from cutamp.cost_function import CostFunction
from cutamp.grasps import GRASP_SCORE_CUTOFF
from cutamp.landing_predictor import LandingState, nominal_landing_dynamics, prediction_precision
from cutamp.particle_initialization import IKTol, ParticleInitializer, _draw_placements, pick_place_confs, worst_posture
from cutamp.rollout import RolloutFunction
from cutamp.samplers import project_supported_placement
from cutamp.stance import abs_pose, stance_joint_limits
from cutamp.step_ik import solve_stance_ik, stance_ik_tolerance
from cutamp.step_planner import StepResult, StepRow
from cutamp.step_refinement import _check_deadline, refine_gap, stance_center_residual, stance_ik_rows
from cutamp.step_sequence import predict_sequence
from cutamp.step_validation import StepStateAdapter, sweep_constraint_rows, sweep_penetration_rows, validate_step_geometry
from cutamp.utils.common import action_4dof_to_mat4x4, grasp_7dof_to_mat4x4


def _batch(value, first, stop, tail, like):
    x = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if x.shape[1:] != tail or (len(x) != 1 and len(x) < stop):
        raise ValueError(f"Expected particle rows of shape {tail}; got {tuple(x.shape)}")
    return x.expand(stop - first, *tail) if len(x) == 1 else x[first:stop]


def _take(value, i, device):
    """Detach a selected trajectory, retaining its batch axis and metadata."""
    if torch.is_tensor(value):
        return value[i : i + 1].detach().to(device)
    if is_dataclass(value):
        return type(value)(**{f.name: _take(getattr(value, f.name), i, device) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(_take(v, i, device) for v in value)
    return value


def _evidence(adapter, predictions):
    from cutamp.step_validation import model_step_record

    def render():
        gaps = {}
        for j, (start, commands, pred, preparation) in predictions.items():
            gaps[str(j)] = []
            for k, step in enumerate(pred.steps):
                gaps[str(j)].append(model_step_record(adapter, start, step, commands[:, k], preparation=preparation))
                start, preparation = step.terminal, None
        return gaps

    return render


def _list(x):
    return x.detach().cpu().tolist()


def _groups(skeleton):
    steps, groups = [], []
    for op in skeleton:
        kind = op.operator.metadata.action_type
        if kind == "step":
            steps.append(op)
        elif kind in ("pick", "place"):
            groups.append((steps, op))
            steps = []
    if steps:
        raise ValueError("Dangling step after final manipulation")
    return groups


def _planar(soles):
    return torch.cat((soles[..., :2, 3], torch.atan2(soles[..., 1, 0], soles[..., 0, 0]).unsqueeze(-1)), dim=-1)


def _carry_witness(q, soles):
    # Offline hand maps store the representative configuration in its mid-sole frame.
    mid = soles[:, :, :2].mean(1)
    half = (soles[:, 1, 2] - soles[:, 0, 2]) / 2
    yaw = soles[:, 0, 2] + torch.atan2(torch.sin(2 * half), torch.cos(2 * half)) / 2
    R = roma.euler_to_rotmat("Z", yaw[:, None])
    out = q.clone()
    out[:, :2] = (R[:, :2, :2] @ q[:, :2, None]).squeeze(-1) + mid
    out[:, 3:6] = roma.rotmat_to_euler("YXZ", R @ roma.euler_to_rotmat("YXZ", q[:, 3:6]))
    return out


def _nominal(incoming, order, commands):
    feet = incoming
    for j, side in enumerate(order if order != "none" else ""):
        index = "LR".index(side)
        swing = abs_pose(feet[:, 1 - index], commands[:, j])
        feet = torch.stack((swing, feet[:, 1]) if index == 0 else (feet[:, 0], swing), dim=1)
    return feet


def _retract_on_stance(soles, arrived, arm, limits, *, other_arm_reference=None, kinematics=None):
    """Project the post-manipulation rest onto planted soles and planner joint limits.

    A learned handoff may lie between the planner's soft limit and the physical
    joint stop. Preserve that handoff. The subsequent retract gets its own
    supported keyframe and the motion planner must connect the two poses.
    """
    from cutamp.step_ik import _kinematics

    bounds = stance_joint_limits(soles, *limits) if kinematics is None else limits
    bounded = arrived.clamp(*bounds)
    kin = _kinematics(arrived.device) if kinematics is None else kinematics
    fk, _ = kin.fk(bounded.double(), arm, jacobian=False)
    hand = fk.reshape(len(arrived), 3, 12)[:, -1]
    wrist = torch.eye(4, device=arrived.device, dtype=hand.dtype).repeat(len(arrived), 1, 1)
    wrist[:, :3, 3] = hand[:, :3]
    wrist[:, :3, :3] = (hand[:, 3:] / 0.2).reshape(-1, 3, 3).mT
    return solve_stance_ik(soles, wrist, bounded, arm, other_arm_reference=other_arm_reference, kinematics=kinematics)


def _inputs_for_gap(seeds, gap, op, incoming, cmds, q_rest):
    by_order = {}
    has_fallback = False
    for i, seed in enumerate(seeds):
        edge = seed.schedule.edges[gap] if seed.schedule.edges else None
        if edge is None or edge.witness is None:
            has_fallback = True
        else:
            by_order.setdefault(edge.order, []).append((i, edge.witness))
    if not by_order:
        return q_rest.clone()
    # A grounded skeleton normally gives all particles the same foot order.
    # Keep the original row bindings for mixed orders and missing reach witnesses.
    if len(by_order) == 1 and not has_fallback:
        order, group = next(iter(by_order.items()))
        witnesses = q_rest.new_tensor([pose for _, pose in group])
        return _carry_witness(witnesses, _nominal(incoming, order, cmds))
    out = q_rest.clone() if has_fallback else q_rest.new_empty(q_rest.shape)
    for order, group in by_order.items():
        rows, poses = zip(*group)
        witnesses = q_rest.new_tensor(poses)
        indices = torch.tensor(rows, device=q_rest.device)
        nominal = _nominal(incoming[indices], order, cmds[indices])
        out[indices] = _carry_witness(witnesses, nominal)
    return out


def _command_inputs(seeds, steps, like):
    if not steps:
        return like.new_empty(len(seeds), 0, 3)
    return like.new_tensor([[seed.commands[step.arg("sole_to")] for step in steps] for seed in seeds])


def _generated_grasp_indices(pool, scores, objects, tool, arm, seeds, gap, ranks):
    """Bind generated grasps to the particle's approximate pick posture before running landing rollouts."""
    by_score = scores.argsort(descending=True, stable=True)
    choice = by_score[ranks].clone()
    rows, poses, soles = [], [], []
    for i, seed in enumerate(seeds):
        edges = seed.schedule.edges
        if edges and edges[gap].witness is not None:
            rows.append(i)
            poses.append(edges[gap].witness)
            soles.append(edges[gap].soles)
    if not rows:
        return choice
    from cutamp.step_ik import _kinematics

    # Both translations and rotations use the same units/weight as the grasp-and-stance IK objective.
    witness = _carry_witness(pool.new_tensor(poses), pool.new_tensor(soles))
    hand = _kinematics(pool.device).fk(witness.double(), arm, False)[0][:, -12:]
    targets = objects[rows, None] @ (grasp_7dof_to_mat4x4(pool) @ tool)[None]
    desired = torch.cat((targets[..., :3, 3], 0.2 * targets[..., :3, :3].mT.flatten(-2)), dim=-1)
    error = (desired - hand[:, None]).square().sum(-1)
    # Higher confidence wins ties. Every below-cutoff grasp stays available in score order.
    error = error.masked_fill(scores < GRASP_SCORE_CUTOFF, torch.inf)
    order = by_score[error[:, by_score].argsort(stable=True)]
    choice[rows] = order.gather(1, ranks[rows, None]).squeeze(1)
    return choice


def _placements_and_grasps(groups, world, pools, first, stop, seeds, bindings=None, grasp_indices=None):
    n = stop - first
    like = world.q_init
    particles = {k: like.expand(n, -1).clone() for k in ("left_q0", "right_q0")}
    objects = {obj.name: world.get_object_pose(obj).expand(n, -1, -1) for obj in world.movables}
    snapshots, wrists, held, held_before = [], [], {}, []
    for gap, (_, op) in enumerate(groups):
        arm = op.operator.metadata.arm
        obj, grasp = op.arg("obj"), op.arg("grasp")
        snapshots.append(objects.copy())
        held_before.append(held.copy())
        if op.operator.metadata.action_type == "pick":
            pool, scores = pools[grasp]
            idx = torch.arange(first, stop, device=like.device) % len(pool)
            if grasp_indices is not None:
                choice = grasp_indices[grasp + "_idx"]
            else:
                tool = world.tool_from_ee[world.tool_frame_for_arm(arm)]
                choice = _generated_grasp_indices(pool, scores, objects[obj], tool, arm, seeds, gap, idx)
            particles[grasp], particles[grasp + "_idx"] = pool[choice], choice
        g = grasp_7dof_to_mat4x4(particles[grasp])
        if op.operator.metadata.action_type == "place":
            key = op.arg("placement")
            if bindings is not None and key in bindings:
                particles[key] = project_supported_placement(bindings[key], world.get_collision_spheres(obj),
                                                            world.get_aabb(op.arg("surface")))
            else:
                particles[key] = _draw_placements(
                    world,
                    world.get_object(obj),
                    world.get_collision_spheres(obj),
                    world.get_object(op.arg("surface")),
                    g,
                    surface_bounds=world.get_aabb(op.arg("surface")),
                )
            objects[obj] = action_4dof_to_mat4x4(particles[key])
        tool = g @ world.tool_from_ee[world.tool_frame_for_arm(arm)]
        wrists.append(objects[obj] @ tool)
        if op.operator.metadata.action_type == "pick":
            held[arm] = (obj, torch.linalg.inv(tool))
        else:
            held.pop(arm)
    return particles, snapshots, wrists, held_before


def _summary(result, commands, order, initial, validation, i, *, incoming, wrist, initialization,
             placement=None, placement_seed=None):
    ik = result.ik
    record = dict(
        order=order,
        seed_commands=_list(commands[i]),
        commands=_list(result.commands[i]),
        achieved=_list(result.achieved[i]),
        center_distance_m=float(result.center_distance[i]),
        com_xy=_list(ik.com[i, :2]),
        hull_centroid_xy=_list((ik.com[:, :2] - stance_center_residual(ik))[i]),
        margin_m=float(result.margin[i]),
        wrist_position_error_m=float(ik.wrist_pos_error[i]),
        wrist_rotation_error_rad=float(ik.wrist_rot_error[i]),
        history_source=initial.history_source if isinstance(initial, LandingState) else None,
        failed_rows=[k for k, v in result.rows.items() if (v[i] > 0).any() or not v[i].isfinite().all()],
        steps=[],
        incoming=_list(incoming[i]),
        wrist_world=_list(wrist[i]),
    )
    if placement is not None:
        record.update(placement_seed=_list(placement_seed[i]), placement=_list(placement[i]))
    if initialization is not None:
        record["initialization"] = {k: _list(v[i]) if torch.is_tensor(v) else v for k, v in initialization.items()}
    if result.prediction is not None:
        pred = result.prediction
        record.update(
            final_xy_error_m=_list(pred.final_xy_error[i]),
            final_yaw_error_deg=_list(pred.final_yaw_error[i].rad2deg()),
            tracking_ok=bool(pred.tracking_ok[i]),
        )
        for s in pred.steps:
            record["steps"].append(
                dict(
                    foot=s.side,
                    clock=int(s.clock[i]),
                    hold_ticks=int(s.valid_ticks[i].sum() - s.clock[i]),
                    reference=_list(s.reference[i]),
                    achieved=_list(s.achieved[i]),
                    xy_error_m=_list(s.xy_error[i]),
                    yaw_error_deg=_list(s.yaw_error[i].rad2deg()),
                    max_joint_rms=None if s.max_joint_rms is None else float(s.max_joint_rms[i]),
                    max_root_speed=None if s.max_root_speed is None else float(s.max_root_speed[i]),
                    near_rest=None if s.near_rest is None else bool(s.near_rest[i]),
                    deadline_ok=bool(s.phases.deadline_ok[i]),
                    support_ok=bool(s.phases.support_ok[i]),
                    hold_ok=bool(s.phases.hold_ok[i]),
                )
            )
    if validation is not None:
        record.update(
            load_status=validation["load_status"],
            sweep_ok=bool(validation["valid"][i]),
            penetration_m={k: float(v[i]) for k, v in validation["penetration"].items()},
        )
    return record


def _refine_batch(
    proposal, world, config, reducer, checker, inputs, predictor, pools, seeds, first, stop, motion_inputs, deadline,
    *, bindings=None, grasp_indices=None, objective=False, ik_starts=None,
):
    _check_deadline(deadline)
    groups = _groups(proposal.skeleton)
    arm_skeleton = [op for op in proposal.skeleton if op.operator.metadata.action_type != "step"]
    cost_function = CostFunction(arm_skeleton, world, config)
    n = stop - first
    like = world.q_init
    particles, objects, wrists, held_before = _placements_and_grasps(
        groups, world, pools, first, stop, seeds, bindings, grasp_indices
    )
    if config.require_modeled_load and any(steps and held_before[j] for j, (steps, _) in enumerate(groups)):
        return StepResult([], None, "unavailable", "seed-3 dynamics have no payload model")
    placement_seeds = {
        op.arg("placement"): particles[op.arg("placement")].clone()
        for _, op in groups
        if op.operator.metadata.action_type == "place"
    }
    if bindings is None:
        from cutamp.step_adam import optimize_particles

        initial = dict(placement_seeds)
        bound_grasps = {key: value for key, value in particles.items() if key.endswith("_idx")}
        for j, (steps, _) in enumerate(groups):
            if steps:
                initial[f"foot{j}"] = _command_inputs(seeds, steps, like)

        def evaluate(values, objective=True, *, ik_starts=None):
            return _refine_batch(proposal, world, config, reducer, checker, inputs, predictor, pools, seeds, first,
                                 stop, motion_inputs, deadline, bindings=values, grasp_indices=bound_grasps,
                                 objective=objective, ik_starts=ik_starts)

        options = {}
        if config.step_adam_plateau_mm is not None or config.step_adam_trace_center:
            options.update(plateau=config.adam_plateau(), trace_center=config.step_adam_trace_center,
                           center_weight=config.com_center_weight)
        objective_fn = partial(evaluate, ik_starts={}) if config.step_ik_warm_start else evaluate
        tolerance = nullcontext()
        if config.step_ik_objective_tol is not None:
            tolerance = stance_ik_tolerance(config.step_ik_objective_tol)
        with tolerance:
            best, report = optimize_particles(initial, objective_fn, config.step_refine_iterations, config.lr, **options)
        # Rebuild each selected row from scratch using the dense teacher and native IK precision.
        with torch.no_grad():
            verified = evaluate(best, objective=False)
        for row in verified.rows:
            idx = row.index - first
            row.metrics["adam_updates_completed"] = report["updates_completed"]
            row.metrics["adam_stop_reason"] = report["stop_reason"]
            row.metrics["selected_adam_iteration"] = int(report["iteration"][idx])
            for j, gap in enumerate(row.metrics["gaps"]):
                key = f"foot{j}"
                if key in initial:
                    gap["adam_seed_commands"] = _list(initial[key][idx])
                    grad = report.get("initial_gradient", {}).get(key)
                    gap["initial_adam_loss_gradient"] = _list(grad[idx]) if grad is not None else None
        verified.diagnostics["adam"] = {
            "first_index": first, "iterations": report["updates_completed"], "max_updates": config.step_refine_iterations,
            "stop_reason": report["stop_reason"],
            "selected_iteration": _list(report["iteration"]), "violation": _list(report["violation"]),
            "initial_violation": _list(report["initial_violation"]),
            "invalid_gradients": _list(report["invalid_gradients"]),
            "initial_gradient": {key: _list(value) for key, value in report.get("initial_gradient", {}).items()},
            "initial": {key: _list(value) for key, value in initial.items()},
            "selected": {key: _list(value) for key, value in best.items()},
        }
        if "best_feasible_center_m" in report:
            verified.diagnostics["adam"]["best_feasible_center_m"] = report["best_feasible_center_m"]
        return verified
    penalties, center_costs = [], []
    rest = particles["left_q0"]
    incoming = _planar(world.stance_soles).expand(n, -1, -1)
    d = nominal_landing_dynamics(like, n, config.step_lag)
    observed = None
    if inputs.initial_state is not None:
        full = inputs.initial_state
        ts = tuple(_batch(x, first, stop, tuple(x.shape[1:]), like) for x in full.tensors)
        observed = LandingState(ts, _batch(full.dynamics, first, stop, (38,), like), full.history_source)
        d = observed.dynamics
    adapter = StepStateAdapter(predictor)
    stances = {"left_q0": incoming, "right_q0": incoming}
    summaries = []

    def records(i):
        return [summarize(i) for summarize in summaries]
    endpoints = []
    predictions = {}
    rest_at_gap = []
    stance_at_gap = []
    retracted = set()
    gap = -1
    for movement in proposal.skeleton:
        if movement.operator.metadata.action_type in ("pick", "place"):
            gap += 1
        elif movement.operator.metadata.action_type == "retract":
            retracted.add(gap)

    def occupied_reference(k, previous):
        opposite = "right" if groups[k][1].operator.metadata.arm == "left" else "left"
        return previous[:, getattr(world.model, opposite + "_arm")] if opposite in held_before[k] else None

    solve_endpoint = partial(solve_stance_ik, center_posture=bool(config.com_center_weight))
    posture_references = {}

    def endpoint_solver(k, previous, witness):
        action = groups[k][1]
        carrying = occupied_reference(k, previous)
        if carrying is not None:
            solve = partial(solve_endpoint, other_arm_reference=carrying)
        else:
            idle = world.model.right_arm if action.operator.metadata.arm == "left" else world.model.left_arm
            nominal = witness.new_tensor(world.model.home[idle]).expand_as(witness[:, idle])
            # The coupled scene rows below already check the nominal idle arm.
            solve = (solve_endpoint if torch.equal(witness[:, idle], nominal)
                     else partial(solve_endpoint, collides=partial(endpoint_collisions, k)))
        if ik_starts is None:
            return solve

        def fit(soles, wrist, seed, arm):
            initial = None
            if k in ik_starts:
                prior_q, usable = ik_starts[k]
                initial = torch.where(usable[:, None], prior_q, seed)
            endpoint = solve(soles, wrist, seed, arm, initial=initial)
            # Reuse only the starting point. Bounds, posture prior, and implicit
            # derivative all belong to the new particle evaluation.
            ik_starts[k] = (endpoint.q.detach(), endpoint.optimality.detach() < 2e-6)
            return endpoint

        return fit

    def endpoint_collisions(k, q, owners=None):
        action = groups[k][1]

        def select(value):
            return value if owners is None else value[owners]

        binding = action.arg("grasp" if action.operator.metadata.action_type == "pick" else "placement")
        variables = {action.arg("q"): q, binding: select(particles[binding])}
        scene = {name: select(value) for name, value in objects[k].items()}
        carried = {obj: (arm, select(transform)) for arm, (obj, transform) in held_before[k].items()}
        values = cost_function.stepped_scene_costs(variables, [action], scene, carried)["values"]
        blocked = values.pop("self_collision").reshape(len(q), -1).amax(1) > 0
        for key, value in values.items():
            blocked |= value.reshape(len(q), -1).amax(1) > checker._get_tol("Collision", key)
        return blocked

    for j, (steps, op) in enumerate(groups):
        _check_deadline(deadline)
        order = proposal.orders[j]
        cmds = bindings[f"foot{j}"] if steps else _command_inputs(seeds, steps, like)
        seed = _inputs_for_gap(seeds, j, op, incoming, cmds, rest)
        arm = op.operator.metadata.arm
        held = held_before[j]
        other = occupied_reference(j, rest)
        margin = config.com_comfort_margin_m
        start = incoming
        usable = torch.ones(n, dtype=torch.bool, device=like.device)
        active_predictor = None
        init = None
        if steps:
            _check_deadline(deadline)
            obs = observed if j == 0 else None
            if inputs.initial_state is not None and j == 0:
                _, qd = adapter.planner(obs.tensors[:6])
            else:
                qd = torch.zeros_like(rest)
            if objective and obs is None:
                start = adapter.objective_state(rest, d, qd)
            else:
                init = adapter.initialize(rest, d, qd, observed=obs, world=world, object_poses=objects[j], held=held)
                start, usable = init.state, init.compatible
                if not objective and not usable.any():
                    return StepResult(
                        [],
                        None,
                        "unavailable",
                        "landing history does not match the incoming state",
                        {k: _list(v) if torch.is_tensor(v) else v for k, v in init.diagnostics.items()},
                    )
            incoming = predictor.model.feet(start.tensors[:6])[0]
            active_predictor = predictor
            if j == 0:
                stances.update(left_q0=incoming, right_q0=incoming)
        # Later manipulations without another step must also stand on these same feet.
        last = next((k for k in range(j + 1, len(groups)) if groups[k][0]), len(groups))
        validation = None
        def coupled_rows(pred, ik):
            nonlocal validation
            rows = {}
            centers = []
            frames = [op]
            variables = dict(particles)
            variables[op.arg("q")] = ik.q
            if last > j + 1 and steps:
                after = adapter.planner(pred.terminal.tensors[:6])[0] if pred is not None else ik.q
                if j in retracted:
                    after = _retract_on_stance(ik.soles, after, arm, world.joint_limits, other_arm_reference=other).q
                for k in range(j + 1, last):
                    future_op = groups[k][1]
                    prior = _inputs_for_gap(seeds, k, future_op, ik.soles, like.new_empty(n, 0, 3), after)
                    future = endpoint_solver(k, ik.q, prior)(ik.soles, wrists[k], prior, future_op.operator.metadata.arm)
                    rows.update({f"future_{k}_{key}": v for key, v in stance_ik_rows(future, margin).items()})
                    centers.append(stance_center_residual(future))
                    frames.append(future_op)
                    variables[future_op.arg("q")] = future.q
            carried = {name: (owner, transform) for owner, (name, transform) in held.items()}
            scene = cost_function.stepped_scene_costs(variables, frames, objects[j], carried)["values"]
            rows["manipulation_self_collision"] = scene.pop("self_collision").amax(1) / 0.02
            limits = [(v - checker._get_tol("Collision", key)) / 0.02 for key, v in scene.items()]
            rows["manipulation_collision"] = torch.cat(limits, dim=1).amax(1)
            if pred is not None and pred.steps:
                if not objective:
                    validation = validate_step_geometry(world, adapter, pred, object_poses=objects[j], held=held)
                    rows.update(sweep_penetration_rows(validation["penetration"]))
                else:
                    rows.update(sweep_constraint_rows(world, adapter, pred, objects[j], held))
            return rows, centers

        if objective:
            from cutamp.step_refinement import _evaluate

            # Each replay starts from the shared initial state. The next gap includes
            # the retraction and controller history prepared from this gap's prediction.
            prediction = None
            if active_predictor is not None:
                prediction = predict_sequence(active_predictor.sequence_model, active_predictor, start, order, cmds)
            ev = _evaluate(active_predictor, start, incoming, order.replace("none", ""), cmds, arm, wrists[j],
                           seed, margin, coupled_rows, solve_ik=endpoint_solver(j, rest, seed), prediction=prediction)
            penalties.append(ev.matrix().relu().square().sum(-1))
            center_costs.append(ev.worst_center_squared())
            incoming = ev.ik.soles
            if prediction is not None:
                rest = adapter.planner(prediction.terminal.tensors[:6])[0]
            if j in retracted and j + 1 < len(groups):
                rest = _retract_on_stance(incoming, rest, arm, world.joint_limits, other_arm_reference=other).q
            continue

        result = refine_gap(
            active_predictor,
            start,
            order,
            cmds,
            arm,
            wrists[j],
            seed,
            margin_m=margin,
            solve_ik=endpoint_solver(j, rest, seed),
            deadline=deadline,
            extra_rows=coupled_rows,
        )
        usable = usable & result.ok
        final = result.achieved.detach().to(like)
        if steps:
            prediction = result.prediction
            usable = usable & validation["valid"]
            predictions[j] = (start, result.commands, prediction, init.preparation)
            for k, step in enumerate(steps):
                foot = "LR".index(order[k])
                pair = result.prediction.steps[k].achieved.detach()
                rest = adapter.planner(result.prediction.steps[k].terminal.tensors[:6])[0].detach()
                name = step.arg("q_land")
                particles[name] = rest
                stances[name] = pair
                particles[step.arg("sole_to")] = pair[:, foot]
        particles[op.arg("q")] = result.posture.detach()
        if config.com_center_weight:
            posture_references[op.arg("q") + "_ik_reference"] = seed.detach()
        stances[op.arg("q")] = final
        if j in retracted:
            with torch.no_grad():
                retraction = _retract_on_stance(final, rest, arm, world.joint_limits, other_arm_reference=other)
            rest = retraction.q
            usable &= retraction.ok
        rest_at_gap.append(rest)
        stance_at_gap.append(final)
        endpoints.append(usable)
        # Copy per-frame diagnostics to the host only for exported feasible or failed rows.
        place_key = op.arg("placement") if op.operator.metadata.action_type == "place" else None
        summaries.append(partial(_summary, result, cmds, order, start, validation, incoming=incoming, wrist=wrists[j],
                                 initialization=None if init is None else init.diagnostics,
                                 placement_seed=placement_seeds.get(place_key), placement=particles.get(place_key)))
        incoming = final

    if objective:
        return torch.stack(penalties).sum(0), config.com_center_weight * torch.stack(center_costs).amax(0)

    j = -1
    for op in arm_skeleton:
        if op.operator.metadata.action_type in ("pick", "place"):
            j += 1
        elif op.operator.metadata.action_type == "retract":
            name = op.arg("q_retract")
            particles[name] = rest_at_gap[j]
            stances[name] = stance_at_gap[j]
    with torch.no_grad():
        rolled = RolloutFunction(arm_skeleton, world)(particles)
        costs = cost_function(rolled, particles, achieved_stances=stances)
        masks = checker.get_full_mask(costs)
        passed = checker.get_mask(costs, verbose=False)
        soft = reducer.soft_costs(costs)
        effort = reducer.secondary_effort(costs)
        hard = reducer(costs)
        ep = torch.stack(endpoints, dim=1)
        for key, value in masks["KinematicConstraint"].items():
            if key.startswith("sole_"):
                value = value.reshape(n, len(groups), 2).all(-1)
            ep = ep & value
        ep = ep & masks["ComComfort"]["margin_shortfall_squared"]
        m = -1
        retract_index = 0
        placed_count = {}
        timeline = [op for op in arm_skeleton if op.operator.metadata.action_type in ("pick", "place", "retract")]
        for t, op in enumerate(timeline):
            if op.operator.metadata.action_type != "retract":
                m += 1
                for key in ("joint_limit", "self_collision"):
                    ep[:, m] &= masks["Motion"][key][:, m]
            else:
                for key in ("retract_joint_limit", "retract_self_collision"):
                    ep[:, m] &= masks["Motion"][key][:, retract_index]
                retract_index += 1
            if op.operator.metadata.action_type == "place":
                surface = op.arg("surface")
                k = placed_count.get(surface, 0)
                for suffix in ("_in_xy", "_support"):
                    ep[:, m] &= masks["StablePlacement"][surface + suffix][:, k]
                placed_count[surface] = k + 1
            ep[:, m] &= masks["Collision"][f"{t}_movables"][:, 0] & masks["Collision"]["robot_to_world"][:, t]
            name = op.arg("q_retract" if op.operator.metadata.action_type == "retract" else "q")
            ep[:, m] &= masks["ComPolygon"][name]
        passed = passed & ep.all(1)
    rows = []
    for i in torch.where(passed)[0].tolist():
        particle = {key: v[i].detach().clone() for key, v in particles.items()}
        particle.update({key: value[i].clone() for key, value in posture_references.items()})
        gaps = records(i)
        report = dict(
            gaps=gaps,
            cost=float(hard[i]),
            secondary_effort=float(effort[i]),
            best_com_margin=min(g["margin_m"] for g in gaps),
            sole_box_m=list(world.model.sole_box),
            com_center_weight=config.com_center_weight,
            kinematic_candidate=True,
            model_feasible=True,
            load_status=(
                "load_unmodeled" if any(g.get("load_status") == "load_unmodeled" for g in gaps) else "unloaded"
            ),
            independent_motion="untested",
            deployment_sha256=predictor.metadata["deployment_sha256"],
            checkpoint_sha256=predictor.metadata["checkpoint_sha256"],
            step_refine_iterations=config.step_refine_iterations,
            normalized_dynamics=_list(d[i]),
            step_lag=config.step_lag,
        )
        saved = {gap: _take(trace, i, "cpu") for gap, trace in predictions.items()}
        evidence = _evidence(adapter, saved)
        if config.curobo_plan:
            motion_inputs[first + i] = (adapter, saved)
            report["motion_validation"] = "unattempted"
        rows.append(StepRow(first + i, particle, float(soft[i]), report, evidence=evidence))
    # Keep same-row details for a failed ablation or an exhausted search.
    best = int((ep.sum(1) * -1e6 + hard.nan_to_num(1e6)).argmin())
    failures = [f"{kind}.{key}" for kind, items in masks.items() for key, mask in items.items() if not mask[best].all()]
    diagnostics = {"row": first + best, "gaps": records(best), "failed_constraints": failures}
    status = "model_feasible" if rows else "unresolved"
    return StepResult(rows, ep, status, None if rows else "no particle met all applicable constraints", diagnostics)


def _validate_motion_rows(rows, pools, skeleton, world, config, attempt, *, deadline):
    """Spend one retry budget across all particle chunks of this grounded schedule."""
    from cutamp.algorithm import grasp_group_order

    diagnostics = {"motion_plan_attempts": 0, "motion_plan_failures": []}
    if not rows or (deadline is not None and monotonic() >= deadline):
        return diagnostics
    if config.com_center_weight != 0:
        ranked = sorted(rows, key=lambda row: (row.max_center_distance_m, row.effort))
        candidates = ranked[:config.motion_plan_max_retries]
    else:
        if pools:
            indices = torch.stack([torch.stack([r.particle[name + "_idx"] for name in pools]) for r in rows])
            hi = torch.stack([
                scores.to(indices.device)[indices[:, j]] >= GRASP_SCORE_CUTOFF
                for j, (_, scores) in enumerate(pools.values())
            ]).all(0)
        else:
            indices = torch.empty(len(rows), 0, dtype=torch.long)
            hi = torch.ones(len(rows), dtype=torch.bool)
        soft = torch.tensor([r.soft_cost for r in rows], dtype=torch.float64)
        margin = torch.tensor([r.metrics["best_com_margin"] for r in rows], dtype=torch.float64)
        confs = [torch.stack([r.particle[q] for r in rows]) for q in pick_place_confs(skeleton)]
        tilt, travel, _ = worst_posture(world, confs)
        order = grasp_group_order(indices, soft, hi, tilt, travel, margin, config.motion_plan_max_retries)
        candidates = [rows[i] for i in order]
    for rank, row in enumerate(candidates):
        if deadline is not None and monotonic() >= deadline:
            break
        row.motion_plan, info = attempt(row)
        row.metrics.update(info)
        diagnostics["motion_plan_attempts"] += 1
        if row.motion_plan is not None:
            row.metrics["motion_plan_rank"] = rank
            break
        diagnostics["motion_plan_failures"].append(info.get("reason", info["motion_validation"]))
    return diagnostics


@prediction_precision()
def refine_step_particles(
    proposal, world, config, cost_reducer, constraint_checker, *, inputs, predictor, timer, visualizer, deadline
):
    if predictor is None:
        return StepResult([], None, "unavailable", "missing trained landing predictor")
    config = config.for_refinement(stepped=True)
    try:
        pools = ParticleInitializer(world, config, constraint_checker)._pick_pools(proposal.skeleton)
    except ValueError as e:
        return StepResult([], None, "unresolved", str(e))
    n = config.num_particles
    seeds = [proposal.seed]
    seeds.extend(islice(proposal.alternative_seeds, n - 1))
    rows, feedback, diagnostic = [], [], {}
    status, reason = "unresolved", "no particle met all applicable constraints"
    attempted_seeds, motion_inputs, adam_batches = [], {}, []
    groups = _groups(proposal.skeleton)
    for first in range(0, n, config.step_particle_batch_size):
        stop = min(first + config.step_particle_batch_size, n)
        batch_seeds = [seeds[i % len(seeds)] for i in range(first, stop)]
        commands = [_list(_command_inputs(batch_seeds, steps, world.q_init)) for steps, _ in groups]
        for signature in zip(*commands):
            if signature not in attempted_seeds:
                attempted_seeds.append(signature)
        try:
            batch = _refine_batch(
                proposal,
                world,
                config,
                cost_reducer,
                constraint_checker,
                inputs,
                predictor,
                pools,
                batch_seeds,
                first,
                stop,
                motion_inputs,
                deadline,
            )
        except TimeoutError as exc:
            if not rows:
                status, reason = "budget_exhausted", str(exc)
            break
        rows.extend(batch.rows)
        if "adam" in batch.diagnostics:
            adam_batches.append(batch.diagnostics.pop("adam"))
        if batch.endpoint_ok is not None:
            feedback.append(batch.endpoint_ok)
        if not diagnostic or batch.rows:
            diagnostic = batch.diagnostics
        if not rows or batch.rows:
            status, reason = batch.status, batch.reason
        if deadline is not None and monotonic() >= deadline:
            break
    if motion_inputs:
        from cutamp.step_preview import plan_predicted_step_motion

        def attempt(row):
            adapter, saved = motion_inputs[row.index]
            traces = {gap: _take(trace, 0, world.q_init.device) for gap, trace in saved.items()}
            return plan_predicted_step_motion(
                proposal.skeleton, row, traces, adapter, world, config, timer=timer, visualizer=visualizer,
                ik_tol=IKTol.from_checker(constraint_checker), deadline=deadline,
            )

        eligible = [row for row in rows if row.index in motion_inputs]
        diagnostic.update(
            _validate_motion_rows(eligible, pools, proposal.skeleton, world, config, attempt, deadline=deadline)
        )
    diagnostic["attempted_seed_commands"] = attempted_seeds
    if adam_batches:
        diagnostic["adam_batches"] = adam_batches
    return StepResult(rows, torch.cat(feedback) if feedback else None, status, reason, diagnostic)
