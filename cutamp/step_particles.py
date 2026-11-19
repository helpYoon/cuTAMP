"""Refine the shared grasps, placements and commanded landings of a grounded T1 skeleton."""

from dataclasses import fields, is_dataclass
from itertools import islice
from time import monotonic

import roma
import torch

from cutamp.cost_function import CostFunction
from cutamp.landing_predictor import LandingState, prediction_precision
from cutamp.particle_initialization import ParticleInitializer, _draw_placements
from cutamp.rollout import RolloutFunction
from cutamp.sole_collision import geometric_sweep_rows
from cutamp.stance import abs_pose
from cutamp.step_ik import solve_stance_ik
from cutamp.step_planner import StepResult, StepRow
from cutamp.step_refinement import refine_gap, stance_center_residual, stance_ik_rows
from cutamp.step_validation import StepStateAdapter, sweep_constraint_rows, validate_step_geometry
from cutamp.utils.common import action_4dof_to_mat4x4, grasp_7dof_to_mat4x4, transform_spheres


def _batch(value, first, stop, tail, like):
    x = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if tuple(x.shape) == tail:
        return x.expand(stop - first, *tail)
    if x.shape[1:] != tail or (len(x) != 1 and len(x) < stop):
        raise ValueError(f"Expected {tail} or particle rows of that shape; got {tuple(x.shape)}")
    return x.expand(stop - first, *tail) if len(x) == 1 else x[first:stop]


def _take(value, i, *, cpu=False):
    """Detach a selected trajectory, retaining its batch axis and metadata."""
    if torch.is_tensor(value):
        tensor = value[i : i + 1].detach()
        return tensor.cpu() if cpu else tensor
    if is_dataclass(value):
        return type(value)(**{f.name: _take(getattr(value, f.name), i, cpu=cpu) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(_take(v, i, cpu=cpu) for v in value)
    return value


def _evidence(adapter, predictions, i):
    from cutamp.step_validation import model_step_record

    gaps = {}
    for j, (initial, commands, pred, preparation) in predictions.items():
        steps = []
        for k, step in enumerate(pred.steps):
            steps.append((_take(initial, i, cpu=True), _take(step, i, cpu=True), commands[i : i + 1, k].detach().cpu()))
            initial = step.terminal
        gaps[j] = (steps, _take(preparation, i, cpu=True))

    def render():
        return {
            str(j): [
                model_step_record(adapter, start, step, cmd, preparation=preparation if k == 0 else None)
                for k, (start, step, cmd) in enumerate(steps)
            ]
            for j, (steps, preparation) in gaps.items()
        }

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


def _retract_on_stance(soles, arrived, arm, limits):
    """Project the post-manipulation rest onto planted soles and planner joint limits.

    A learned handoff may lie between the planner's soft limit and the physical
    joint stop. Preserve that handoff. The subsequent retract gets its own
    supported keyframe and the motion planner must connect the two poses.
    """
    from cutamp.step_ik import _kinematics

    bounded = arrived.clamp(limits[0], limits[1])
    fk, _ = _kinematics(arrived.device).fk(bounded.double(), arm, jacobian=False)
    hand = fk.reshape(len(arrived), 3, 12)[:, -1]
    wrist = torch.eye(4, device=arrived.device, dtype=hand.dtype).repeat(len(arrived), 1, 1)
    wrist[:, :3, 3] = hand[:, :3]
    wrist[:, :3, :3] = (hand[:, 3:] / 0.2).reshape(-1, 3, 3).mT
    return solve_stance_ik(soles, wrist, bounded, arm)


def _inputs_for_gap(inputs, seeds, gap, op, incoming, cmds, q_rest, first, stop):
    if inputs.posture_seeds is not None and inputs.posture_seeds[gap] is not None:
        return _batch(inputs.posture_seeds[gap], first, stop, (q_rest.shape[-1],), q_rest)
    out = []
    for i, seed in enumerate(seeds):
        edge = seed.schedule.edges[gap] if seed is not None and seed.schedule.edges else None
        if edge is None or edge.witness is None:
            out.append(q_rest[i])
        else:
            nominal = _nominal(incoming[i : i + 1], edge.order, cmds[i : i + 1])
            out.append(_carry_witness(q_rest.new_tensor(edge.witness)[None], nominal)[0])
    return torch.stack(out)


def _command_inputs(inputs, seeds, j, steps, like):
    if not steps:
        return like.new_empty(len(seeds), 0, 3)
    if inputs.commands is not None:
        return like.new_tensor(inputs.commands[j]).expand(len(seeds), -1, -1).clone()
    # Leave every unscored symbolic order a legal starting proposal for refinement.
    rows = []
    for seed in seeds:
        rows.append(
            [
                seed.commands[step.arg("sole_to")]
                if seed is not None
                else (0.12, 0.21 if step.operator.metadata.foot == "left" else -0.21, 0.0)
                for step in steps
            ]
        )
    return like.new_tensor(rows)


def _placements_and_grasps(groups, world, pools, inputs, first, stop):
    n = stop - first
    like = world.q_init
    particles = {k: like.expand(n, -1).clone() for k in ("left_q0", "right_q0")}
    objects = {obj.name: world.get_object_pose(obj).expand(n, -1, -1) for obj in world.movables}
    snapshots, wrists = [], []
    for _, op in groups:
        arm = op.operator.metadata.arm
        obj, grasp = op.arg("obj"), op.arg("grasp")
        snapshots.append(objects.copy())
        if op.operator.metadata.action_type == "pick":
            pool, scores = pools[grasp]
            idx = torch.arange(first, stop, device=like.device) % len(pool)
            choice = torch.argsort(scores, descending=True, stable=True)[idx]
            particles[grasp], particles[grasp + "_idx"] = pool[choice], choice
        g = grasp_7dof_to_mat4x4(particles[grasp])
        if op.operator.metadata.action_type == "place":
            key = op.arg("placement")
            fixed = inputs.placements or {}
            particles[key] = (
                _batch(fixed[key], first, stop, (4,), like).clone()
                if key in fixed
                else _draw_placements(
                    world,
                    world.get_object(obj),
                    world.get_collision_spheres(obj),
                    world.get_object(op.arg("surface")),
                    g,
                )
            )
            objects[obj] = action_4dof_to_mat4x4(particles[key])
        wrists.append(objects[obj] @ g @ world.tool_from_ee[world.tool_frame_for_arm(arm)])
    return particles, snapshots, wrists


def _place_update(world, op, pose, soles, seed, grasp, requested_margin, centering_weight):
    """A fresh stance IK and a projected update of the particle's placement on its surface."""
    arm = op.operator.metadata.arm
    tool = grasp_7dof_to_mat4x4(grasp) @ world.tool_from_ee[world.tool_frame_for_arm(arm)]
    local = world.get_collision_spheres(op.arg("obj"))
    bounds = world.get_aabb(op.arg("surface"))

    def score(p):
        T = action_4dof_to_mat4x4(p)
        wrist = T @ tool
        ik = solve_stance_ik(soles, wrist, seed, arm)
        spheres = transform_spheres(local, T)
        lo = (bounds[0, :2] - (spheres[..., :2] - spheres[..., 3:])).clamp_min(0).amax((1, 2))
        hi = ((spheres[..., :2] + spheres[..., 3:]) - bounds[1, :2]).clamp_min(0).amax((1, 2))
        hard = list(stance_ik_rows(ik, requested_margin).values()) + [lo / 0.02, hi / 0.02]
        loss = torch.stack(hard, dim=-1).clamp_min(0).square().sum(-1)
        center = stance_center_residual(ik).square().sum(-1)
        return loss, centering_weight * center, wrist

    with torch.enable_grad():
        current = pose.detach().requires_grad_(True)
        loss, center, wrist = score(current)
        # Satisfy the hard rows first. Once feasible, descend on centering
        # without giving up feasible wrist, support or placement constraints.
        feasible = loss == 0
        objective = torch.where(feasible, center, loss)
        (grad,) = torch.autograd.grad(objective.sum(), current)
    # Z is established by supporting contact. Vary planar position and yaw jointly.
    grad = grad.detach() * pose.new_tensor([1.0, 1.0, 0.0, 0.3])
    delta = -grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-10) * 0.015
    best, best_loss, best_center, best_wrist = pose.detach(), loss.detach(), center.detach(), wrist.detach()
    if (grad.norm(dim=-1) > 1e-7).any():
        for alpha in (1.0, 0.5, 0.25):
            trial = pose.detach() + alpha * delta
            with torch.no_grad():
                trial_loss, trial_center, trial_wrist = score(trial)
            improved = ((best_loss > 0) & (trial_loss < best_loss)) | (
                (best_loss == 0) & (trial_loss == 0) & (trial_center < best_center)
            )
            best = torch.where(improved[:, None], trial, best)
            best_wrist = torch.where(improved[:, None, None], trial_wrist, best_wrist)
            best_loss = torch.where(improved, trial_loss, best_loss)
            best_center = torch.where(improved, trial_center, best_center)
    return best, best_wrist


def _summary(result, commands, order, initial, validation, i):
    ik = result.ik
    record = dict(
        order=order,
        seed_commands=_list(commands[i]),
        commands=_list(result.commands[i]),
        initial_achieved=_list(result.initial_achieved[i]),
        achieved=_list(result.achieved[i]),
        initial_margin_gradient=_list(result.initial_gradient[i]),
        initial_margin_m=float(result.initial_margin[i]),
        initial_center_distance_m=float(result.initial_center_distance[i]),
        initial_center_residual_xy=_list(result.initial_center_residual_xy[i]),
        center_distance_m=float(result.center_distance[i]),
        initial_center_gradient=_list(result.initial_center_gradient[i]),
        com_xy=_list(ik.com[i, :2]),
        hull_centroid_xy=_list((ik.com[:, :2] - stance_center_residual(ik))[i]),
        margin_m=float(result.margin[i]),
        wrist_position_error_m=float(ik.wrist_pos_error[i]),
        wrist_rotation_error_rad=float(ik.wrist_rot_error[i]),
        iterations=result.iterations,
        history_source=initial.history_source if isinstance(initial, LandingState) else None,
        failed_rows=[k for k, v in result.rows.items() if (v[i] > 0).any() or not v[i].isfinite().all()],
        steps=[],
    )
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
                    reference=_list(s.reference[i]),
                    achieved=_list(s.achieved[i]),
                    xy_error_m=_list(s.xy_error[i]),
                    yaw_error_deg=_list(s.yaw_error[i].rad2deg()),
                    max_joint_rms=float(s.max_joint_rms[i]),
                    max_root_speed=float(s.max_root_speed[i]),
                    near_rest=bool(s.near_rest[i]),
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
    proposal, world, config, reducer, checker, inputs, predictor, pools, seeds, first, stop, timer, visualizer, deadline
):
    groups = _groups(proposal.skeleton)
    arm_skeleton = [op for op in proposal.skeleton if op.operator.metadata.action_type != "step"]
    cost_function = CostFunction(arm_skeleton, world, config)
    n = stop - first
    like = world.q_init
    particles, objects, wrists = _placements_and_grasps(groups, world, pools, inputs, first, stop)
    placement_seeds = {
        op.arg("placement"): particles[op.arg("placement")].clone()
        for _, op in groups
        if op.operator.metadata.action_type == "place"
    }
    rest = particles["left_q0"]
    incoming = _planar(world.stance_soles).expand(n, -1, -1)
    d = like.new_zeros(n, 38)
    d[:, 0] = 0.25 / 0.45  # friction 1.0; all other continuous parameters nominal
    d[:, -5:] = (config.step_lag - 5) / 3
    observed = None
    if inputs.initial_state is not None:
        full = inputs.initial_state
        ts = tuple(_batch(x, first, stop, tuple(x.shape[1:]), like) for x in full.tensors)
        observed = LandingState(ts, _batch(full.dynamics, first, stop, (38,), like), full.history_source)
        d = observed.dynamics
    if inputs.dynamics is not None:
        d = _batch(inputs.dynamics, first, stop, (38,), like)
    adapter = StepStateAdapter(predictor) if predictor is not None else None
    stances = {"left_q0": incoming, "right_q0": incoming}
    records = [[] for _ in seeds]
    endpoints = []
    held = {}
    predictions = {}
    rest_at_gap = []
    stance_at_gap = []
    prepared_places = set()
    retracted = set()
    gap = -1
    for movement in proposal.skeleton:
        if movement.operator.metadata.action_type in ("pick", "place"):
            gap += 1
        elif movement.operator.metadata.action_type == "retract":
            retracted.add(gap)
    for j, (steps, op) in enumerate(groups):
        if deadline is not None and monotonic() >= deadline:
            return StepResult([], None, "unavailable", "continuous refinement time budget expired")
        order = proposal.orders[j]
        cmds = _command_inputs(inputs, seeds, j, steps, like)
        seed = _inputs_for_gap(inputs, seeds, j, op, incoming, cmds, rest, first, stop)
        arm = op.operator.metadata.arm
        margin = config.com_comfort_margin_m
        start = incoming
        usable = torch.ones(n, dtype=torch.bool, device=like.device)
        active_predictor = None
        init = None
        if steps and predictor is not None:
            obs = observed if j == 0 else None
            if inputs.initial_state is not None and j == 0:
                _, qd = adapter.planner(obs.tensors[:6])
            else:
                qd = torch.zeros_like(rest)
            init = adapter.initialize(rest, d, qd, observed=obs, world=world, object_poses=objects[j], held=held)
            start, usable = init.state, init.compatible
            if not usable.any():
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
        if steps and held and config.require_modeled_load:
            return StepResult([], None, "unavailable", "seed-3 dynamics have no payload model")
        # Bind free placements before differentiating this stance. A later place
        # on these same feet must use exactly that binding in its final IK too.
        last = next((k for k in range(j + 1, len(groups)) if groups[k][0]), len(groups))
        for k in range(j, last):
            place = groups[k][1]
            if place.operator.metadata.action_type != "place" or k in prepared_places:
                continue
            key = place.arg("placement")
            if key not in (inputs.placements or {}):
                nominal = _nominal(incoming, order, cmds)
                place_seed = (
                    seed
                    if j == k
                    else _inputs_for_gap(inputs, seeds, k, place, nominal, like.new_empty(n, 0, 3), rest, first, stop)
                )
                placed, wrists[k] = _place_update(
                    world,
                    place,
                    particles[key],
                    nominal,
                    place_seed,
                    particles[place.arg("grasp")],
                    margin,
                    config.com_center_weight,
                )
                particles[key] = placed
                obj = place.arg("obj")
                transformation = action_4dof_to_mat4x4(placed)
                for m in range(k + 1, len(groups)):
                    future_op = groups[m][1]
                    objects[m][obj] = transformation
                    if future_op.arg("obj") == obj:
                        if future_op.operator.metadata.action_type == "place":
                            break
                        grasp = grasp_7dof_to_mat4x4(particles[future_op.arg("grasp")])
                        frame = world.tool_frame_for_arm(future_op.operator.metadata.arm)
                        wrists[m] = transformation @ grasp @ world.tool_from_ee[frame]
            prepared_places.add(k)

        # Later manipulations without another step must also stand on these same feet.
        def coupled_rows(pred, ik):
            rows = {}
            centers = []
            frames = [op]
            variables = dict(particles)
            variables[op.arg("q")] = ik.q
            if last > j + 1 and steps:
                after = adapter.planner(pred.terminal.tensors[:6])[0] if pred is not None else ik.q
                if j in retracted and any(
                    inputs.posture_seeds is None or inputs.posture_seeds[k] is None for k in range(j + 1, last)
                ):
                    after = _retract_on_stance(ik.soles, after, arm, world.joint_limits).q
                for k in range(j + 1, last):
                    future_op = groups[k][1]
                    prior = _inputs_for_gap(
                        inputs, seeds, k, future_op, ik.soles, like.new_empty(n, 0, 3), after, first, stop
                    )
                    future = solve_stance_ik(ik.soles, wrists[k], prior, future_op.operator.metadata.arm)
                    rows.update({f"future_{k}_{key}": v for key, v in stance_ik_rows(future, margin).items()})
                    centers.append(stance_center_residual(future))
                    frames.append(future_op)
                    variables[future_op.arg("q")] = future.q
            carried = {name: (owner, transform) for owner, (name, transform) in held.items()}
            scene = cost_function.stepped_scene_costs(variables, frames, objects[j], carried)["values"]
            limits = [(v - checker._get_tol("Collision", key)) / 0.02 for key, v in scene.items()]
            rows["manipulation_collision"] = torch.cat(limits, dim=1).amax(1)
            if pred is not None and pred.steps:
                rows.update(sweep_constraint_rows(world, adapter, pred, objects[j], held))
            return rows, centers

        result = refine_gap(
            active_predictor,
            start,
            order,
            cmds,
            arm,
            wrists[j],
            seed,
            margin_m=margin,
            centering_weight=config.com_center_weight,
            max_iterations=config.step_refine_iterations,
            deadline=deadline,
            extra_rows=coupled_rows,
            geometry_rows=geometric_sweep_rows(world, objects[j], held) if steps and predictor is None else None,
        )
        usable = usable & result.ok
        final = result.achieved.detach().to(like)
        validation = None
        if steps:
            if active_predictor is not None:
                prediction = result.prediction
                validation = validate_step_geometry(world, adapter, prediction, object_poses=objects[j], held=held)
                usable = usable & validation["valid"]
                predictions[j] = (start, result.commands, prediction, init.preparation)
            pair = incoming
            for k, step in enumerate(steps):
                foot = "LR".index(order[k])
                pair = (
                    _nominal(pair, order[k], result.commands[:, k : k + 1]).detach()
                    if validation is None
                    else result.prediction.steps[k].achieved.detach()
                )
                rest = (
                    _retract_on_stance(pair, rest, arm, world.joint_limits)
                    if validation is None
                    else adapter.planner(result.prediction.steps[k].terminal.tensors[:6])[0].detach()
                )
                name = step.arg("q_land")
                if validation is None:
                    usable &= rest.ok & (stance_ik_rows(rest, 0)["supported"] <= 0)
                    rest = rest.q.detach()
                    carried = {obj: (owner, transform) for owner, (obj, transform) in held.items()}
                    at_step = cost_function.stepped_scene_costs({name: rest}, [step], objects[j], carried)
                    for key, v in at_step["values"].items():
                        usable &= (v <= checker._get_tol("Collision", key)).all(1)
                particles[name] = rest
                stances[name] = pair
                particles[step.arg("sole_to")] = pair[:, foot]
        particles[op.arg("q")] = result.posture.detach()
        stances[op.arg("q")] = final
        if j in retracted:
            with torch.no_grad():
                retraction = _retract_on_stance(final, rest, arm, world.joint_limits)
            rest = retraction.q
            usable &= retraction.ok
        rest_at_gap.append(rest)
        stance_at_gap.append(final)
        endpoints.append(usable)
        for i in range(n):
            records[i].append(_summary(result, cmds, order, start, validation, i))
            records[i][-1]["incoming"] = _list(incoming[i])
            records[i][-1]["wrist_world"] = _list(wrists[j][i])
            if op.operator.metadata.action_type == "place":
                key = op.arg("placement")
                records[i][-1].update(placement_seed=_list(placement_seeds[key][i]), placement=_list(particles[key][i]))
            if init is not None:
                records[i][-1]["initialization"] = {
                    k: _list(v[i]) if torch.is_tensor(v) else v for k, v in init.diagnostics.items()
                }
        if op.operator.metadata.action_type == "pick":
            g = grasp_7dof_to_mat4x4(particles[op.arg("grasp")])
            held[arm] = (op.arg("obj"), torch.linalg.inv(g @ world.tool_from_ee[world.tool_frame_for_arm(arm)]))
        else:
            held.pop(arm)
        incoming = final

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
        effort = reducer.soft_costs(costs)
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
        model_feasible = predictor is not None
        report = dict(
            gaps=records[i],
            cost=float(hard[i]),
            best_com_margin=min(g["margin_m"] for g in records[i]),
            sole_box_m=list(world.model.sole_box),
            com_center_weight=config.com_center_weight,
            kinematic_candidate=True,
            model_feasible=model_feasible,
            load_status=(
                "load_unmodeled" if any(g.get("load_status") == "load_unmodeled" for g in records[i]) else "unloaded"
            ),
            independent_motion="untested",
        )
        if model_feasible:
            report.update(
                deployment_sha256=predictor.metadata["deployment_sha256"],
                checkpoint_sha256=predictor.metadata["checkpoint_sha256"],
                normalized_dynamics=_list(d[i]),
                step_lag=config.step_lag,
            )
        motion = None
        if config.curobo_plan:
            if model_feasible and inputs.transition_validator is not None:
                from cutamp.motion_solver import plan_validated_step_motion
                from cutamp.particle_initialization import IKTol

                motion, motion_info = plan_validated_step_motion(
                    proposal.skeleton,
                    particle,
                    {k: _take(p, i) for k, p in predictions.items()},
                    adapter,
                    world,
                    config,
                    inputs.transition_validator,
                    timer=timer,
                    visualizer=visualizer,
                    ik_tol=IKTol.from_checker(checker),
                )
                report.update(motion_info)
            else:
                report.update(
                    motion_validation="unavailable", motion_reason="independent transition validator required"
                )
        evidence = _evidence(adapter, predictions, i) if model_feasible else None
        rows.append(StepRow(first + i, particle, float(effort[i]), report, motion, evidence))
    # Keep same-row details for a failed ablation or an exhausted search.
    best = int((ep.sum(1) * -1e6 + hard.nan_to_num(1e6)).argmin())
    failures = [f"{kind}.{key}" for kind, items in masks.items() for key, mask in items.items() if not mask[best].all()]
    diagnostics = {"row": first + best, "gaps": records[best], "failed_constraints": failures}
    status = ("model_feasible" if predictor is not None else "kinematic_candidate") if rows else "unresolved"
    return StepResult(rows, ep, status, None if rows else "no particle met all applicable constraints", diagnostics)


@prediction_precision()
def refine_step_particles(
    proposal, world, config, cost_reducer, constraint_checker, *, inputs, predictor, timer, visualizer, deadline
):
    if config.step_mode == "model" and predictor is None:
        return StepResult([], None, "unavailable", "missing trained landing predictor")
    try:
        pools = ParticleInitializer(world, config, constraint_checker)._pick_pools(proposal.skeleton)
    except ValueError as e:
        return StepResult([], None, "unresolved", str(e))
    n = config.num_particles
    seeds = [proposal.seed]
    if proposal.seed is not None and inputs.commands is None:
        seeds.extend(islice(proposal.alternative_seeds, n - 1))
    rows, feedback, diagnostic = [], [], {}
    status, reason = "unresolved", "no particle met all applicable constraints"
    attempted_seeds = []
    groups = _groups(proposal.skeleton)
    for first in range(0, n, config.step_particle_batch_size):
        stop = min(first + config.step_particle_batch_size, n)
        batch_seeds = [seeds[i % len(seeds)] for i in range(first, stop)]
        commands = [
            _list(_command_inputs(inputs, batch_seeds, j, steps, world.q_init)) for j, (steps, _) in enumerate(groups)
        ]
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
                timer,
                visualizer,
                deadline,
            )
        except TimeoutError as exc:
            if not rows:
                status, reason = "unavailable", str(exc)
            break
        rows.extend(batch.rows)
        if batch.endpoint_ok is not None:
            feedback.append(batch.endpoint_ok)
        if not diagnostic or batch.rows:
            diagnostic = batch.diagnostics
        if not rows or batch.rows:
            status, reason = batch.status, batch.reason
        if deadline is not None and monotonic() >= deadline:
            break
    diagnostic["attempted_seed_commands"] = attempted_seeds
    return StepResult(rows, torch.cat(feedback) if feedback else None, status, reason, diagnostic)
