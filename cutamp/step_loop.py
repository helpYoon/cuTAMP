"""A plan with steps inside the Adam loop (design sections 4 to 6): its starting values, its refresh on the world model
and controller, and the helpers the stepped route shares with the motion stage."""

import math
from dataclasses import fields, is_dataclass
from functools import partial
from time import monotonic

import torch

from cutamp.com_polygon_cost import physical_support_centroid
from cutamp.cost_function import CostFunction
from cutamp.grasps import GRASP_SCORE_CUTOFF
from cutamp.landing_predictor import LandingState, nominal_landing_dynamics, prediction_precision
from cutamp.particle_initialization import _draw_placements, pick_place_confs, worst_posture
from cutamp.robots.t1 import REAL_LEGS
from cutamp.rollout import RolloutFunction
from cutamp.samplers import project_supported_placement
from cutamp.stance import STEP_TRAVEL, abs_pose, rel_pose, stance_joint_limits
from cutamp.step_bound import planner_step_bound, travel
from cutamp.step_ik import StanceIK, solve_stance_ik
from cutamp.step_refresh import StandIn, draw_commands, refresh_gap
from cutamp.step_validation import StepStateAdapter, validate_step_geometry
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


def _placements_and_grasps(groups, world, pools, first, stop, bindings=None, grasp_indices=None):
    n = stop - first
    like = world.q_init
    particles = {k: like.expand(n, -1).clone() for k in ("left_q0", "right_q0")}
    objects = {obj.name: world.get_object_pose(obj).expand(n, -1, -1) for obj in world.movables}
    snapshots, wrists, held, held_before = [], [], {}, []
    for _, op in groups:
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
                # Each particle takes the next of the pick's grasps in score order.
                choice = scores.argsort(descending=True, stable=True)[idx]
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


def stance_center_residual(ik: StanceIK) -> torch.Tensor:
    """Signed xy displacement of the projected robot CoM from the physical hull's area centroid."""
    return ik.com[:, :2] - physical_support_centroid(ik.soles.double(), REAL_LEGS.sole_box)


def landing_constraint_rows(predictor, incoming, order, u, pred):
    """Command domain, phase, per-landing and composed-stance checks on this actual issue state."""
    before = incoming
    rows = {}
    hard = torch.ones(len(u), dtype=torch.bool, device=u.device)
    if order:
        bounds = planner_step_bound(u.device, u.dtype)
    for j, side in enumerate(order):
        prefix = f"step{j + 1}_"
        c = u[:, j]
        s = "LR".index(side)
        foot = torch.full((len(c),), s == 0, device=c.device)
        start = rel_pose(before[:, 1 - s], before[:, s])
        t = travel(start, c)
        rows[prefix + "min_travel"] = (STEP_TRAVEL[0] - t) / STEP_TRAVEL[0]
        rows[prefix + "max_travel"] = t / STEP_TRAVEL[1] - 1
        rows[prefix + "command_bound"] = bounds.g(c, foot) / 0.02
        dest = abs_pose(before[:, 1 - s], c)
        pair = torch.stack((dest, before[:, 1]) if s == 0 else (before[:, 0], dest), dim=1)
        if pred is not None:
            step = pred.steps[j]
            pair = step.achieved
            ph = step.phases
            profile = predictor.profile
            rows[prefix + "xy_precision"] = step.xy_error.square() / 0.020**2 - 1
            rows[prefix + "yaw_precision"] = step.yaw_error.square() / math.radians(1) ** 2 - 1
            for key in (
                "deadline_xy",
                "deadline_yaw",
                "deadline_height",
                "support_xy",
                "support_yaw",
                "hold_xy",
                "hold_yaw",
            ):
                threshold = getattr(profile, key)
                rows[prefix + key] = (getattr(ph, key) - threshold) / max(threshold, 1e-6)
            rows[prefix + "terminal_sole_height"] = step.sole_height[:, -1].abs() / 0.02 - 1
            rows[prefix + "terminal_sole_tilt"] = step.sole_tilt[:, -1] / 0.10 - 1
            hard = hard & ph.deadline_ok & ph.hold_ok & ph.support_ok
        actual = rel_pose(pair[:, 1 - s], pair[:, s])
        rows[prefix + "achieved_bound"] = bounds.g(actual, foot) / 0.02
        before = pair
    if pred is not None:
        hard = hard & pred.tracking_ok
        if len(order) > 1:
            rows["final_xy_precision"] = pred.final_xy_error.square() / 0.020**2 - 1
            rows["final_yaw_precision"] = pred.final_yaw_error.square() / math.radians(1) ** 2 - 1
    return rows, hard, before


def _check_deadline(deadline):
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("T1 step refinement exceeded the planner deadline")


def _gap_stances(standin, j, before, u):
    """Gap ``j`` of ``standin`` placed on the pair ``before`` it starts from: its ``(after_steps, final)`` entries, as
    :meth:`StandIn.stances` gives them."""
    gap = slice(j, j + 1)
    one = StandIn(standin.orders[gap], standin.relative[gap], standin.slope[gap], standin.at[gap])
    after, final = one.stances(before, [u])
    return after[0], final[0]


def _write(particles, key, value):
    """A new key takes a copy of ``value``; an existing one is written in place, so the loop's tensor stays the same
    object."""
    if key in particles:
        particles[key].copy_(value)
    else:
        particles[key] = value.clone()


class SteppedPlan:
    """A plan with steps inside the Adam loop (design 4 to 6).

    A particle's commands, ``[n, s, 3]`` per gap with steps, are Adam variables drawn uniformly inside the controller's
    limit; its pick, place and retract postures are Adam variables that start from one solve on the stance the first
    refresh gives. A refresh runs the world model and controller on every particle's commands, gap by gap, writes the
    landing postures (never optimized) and sets the stand-in the costs read until the next refresh. The starting pass is
    the first refresh.
    """

    def __init__(self, proposal, world, config, predictor, inputs, pools, checker, *, deadline=None, generator=None):
        began = monotonic()
        self.world, self.config, self.predictor, self.pools = world, config, predictor, pools
        self.skeleton = list(proposal.skeleton)
        self.arm_skeleton = [op for op in self.skeleton if op.operator.metadata.action_type != "step"]
        self.groups = _groups(self.skeleton)
        self.orders = tuple("" if order == "none" else order for order in proposal.orders)
        self.command_names = {j: f"step_commands_{j}" for j, (steps, _) in enumerate(self.groups) if steps}
        self.landing_confs = frozenset(step.arg("q_land") for steps, _ in self.groups for step in steps)
        self.rollout_fn = RolloutFunction(self.arm_skeleton, world)
        self.adapter = StepStateAdapter(predictor)
        empty = [None] * len(self.groups)
        self.standin = StandIn(self.orders, list(empty), list(empty), list(empty))
        self.last, self.motion_inputs, self.references, self.start_postures = {}, {}, {}, {}
        self.stats = {"refresh_s": [], "standin_error_m": [], "command_move_m": []}
        # Each pick, place and retract posture's gap; the posture each gap starts from: the plan's start for the first,
        # else the retract after the gap before it (without one, the posture the robot stood in at that pick or place).
        self.gap_of, retract_of, gap = {}, {}, -1
        for op in self.arm_skeleton:
            kind = op.operator.metadata.action_type
            if kind in ("pick", "place"):
                gap += 1
                self.gap_of[op.arg("q")] = gap
            elif kind == "retract":
                self.gap_of[op.arg("q_retract")] = gap
                retract_of[gap] = op.arg("q_retract")
        self._start_of = ["left_q0"]
        for j, (steps, _) in enumerate(self.groups[:-1]):
            self._start_of.append(retract_of.get(j, steps[-1].arg("q_land") if steps else self._start_of[j]))

        n, like = config.num_particles, world.q_init
        with torch.no_grad(), prediction_precision():
            particles, objects, wrists, held_before = _placements_and_grasps(self.groups, world, pools, 0, n)
            d = nominal_landing_dynamics(like, n, config.step_lag)
            observed = None
            if inputs.initial_state is not None:
                full = inputs.initial_state
                ts = tuple(_batch(x, 0, n, tuple(x.shape[1:]), like) for x in full.tensors)
                observed = LandingState(ts, _batch(full.dynamics, 0, n, (38,), like), full.history_source)
                d = observed.dynamics
            self._dynamics = d
            # The first gap's hold starts from the plan's fixed start, so it is run once, here; the draw starts from
            # the feet it ends on.
            _check_deadline(deadline)
            self._first = None
            self.incoming = _planar(world.stance_soles).expand(n, -1, -1).clone()
            if self.groups[0][0]:
                rest = particles["left_q0"]
                qd = self.adapter.planner(observed.tensors[:6])[1] if observed is not None else torch.zeros_like(rest)
                self._first = self.adapter.initialize(rest, d, qd, observed=observed, world=world,
                                                      object_poses=objects[0], held=held_before[0])
                self.incoming = predictor.model.feet(self._first.state.tensors[:6])[0]
            if generator is None:
                generator = (torch.cuda.default_generators[like.device.index or 0] if like.device.type == "cuda"
                             else torch.default_generator)
            for j, u in enumerate(draw_commands(self.orders, self.incoming, n, generator)):
                if u is not None:
                    particles[self.command_names[j]] = u

            cost_function = CostFunction(self.arm_skeleton, world, config)
            groups = self.groups

            def occupied_reference(k, previous):
                opposite = "right" if groups[k][1].operator.metadata.arm == "left" else "left"
                return previous[:, getattr(world.model, opposite + "_arm")] if opposite in held_before[k] else None

            solve_endpoint = partial(solve_stance_ik, center_posture=bool(config.com_center_weight))

            def endpoint_solver(k, previous, seed):
                action = groups[k][1]
                carrying = occupied_reference(k, previous)
                if carrying is not None:
                    return partial(solve_endpoint, other_arm_reference=carrying)
                idle = world.model.right_arm if action.operator.metadata.arm == "left" else world.model.left_arm
                nominal = seed.new_tensor(world.model.home[idle]).expand_as(seed[:, idle])
                # A nominal idle arm is left to the loop's collision costs.
                return (solve_endpoint if torch.equal(seed[:, idle], nominal)
                        else partial(solve_endpoint, collides=partial(endpoint_collisions, k)))

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

            ok, head, before = torch.ones(n, dtype=torch.bool, device=like.device), None, self.incoming
            for j, (steps, op) in enumerate(self.groups):
                _check_deadline(deadline)
                rest, arm = particles[self._start_of[j]], op.operator.metadata.arm
                seed = rest  # the posture the robot stands in at this pick or place: the start, or where it landed
                if steps:
                    passed, before, head = self._run_gap(j, particles, head, before, objects[j], held_before[j])
                    ok &= passed
                    seed = particles[steps[-1].arg("q_land")]
                ik = endpoint_solver(j, rest, seed)(before, wrists[j], seed, arm)
                particles[op.arg("q")] = ik.q
                self.references[op.arg("q") + "_ik_reference"] = seed.clone()
                self.start_postures[op.arg("q")] = ik.q.clone()
                if j in retract_of:
                    other = occupied_reference(j, rest)
                    retract = _retract_on_stance(before, seed, arm, world.joint_limits, other_arm_reference=other).q
                    particles[retract_of[j]] = retract
                    self.start_postures[retract_of[j]] = retract.clone()
            self.tests_ok = ok
        self.particles = particles
        self.stats["standin_error_m"].append(None)  # no stand-in before the first refresh
        self.stats["command_move_m"].append(None)
        self.stats["refresh_s"].append(monotonic() - began)

    def _run_gap(self, j, particles, head, before, objects, held):
        """Gap ``j``'s hold, its rollout and slope at the current commands, the stand-in set, its landing postures and
        ``sole_to`` written and ``last[j]`` kept. ``head`` ``[n, 2]`` (None: zero) are the head angles the earlier
        rollout ended with; ``before`` ``[n, 2, 3]`` is the stand-in's pair the gap starts on. Returns its tests' pass
        ``[n]``, the pair after it and the head angles its rollout ends with."""
        steps, order = self.groups[j][0], self.orders[j]
        if j == 0:
            start = self._first
        else:
            self.adapter.head = head
            start = self.adapter.initialize(particles[self._start_of[j]], self._dynamics, world=self.world,
                                            object_poses=objects, held=held)
        u = particles[self.command_names[j]]
        anchor = before[:, 1 - "LR".index(order[0])]
        refreshed = refresh_gap(self.predictor, start.state, order, u, anchor, self.standin.slope[j])
        self.standin.set(j, refreshed)
        prediction = refreshed.prediction
        for k, step in enumerate(steps):
            landed = prediction.steps[k]
            _write(particles, step.arg("q_land"), self.adapter.planner(landed.terminal.tensors[:6])[0])
            _write(particles, step.arg("sole_to"), landed.achieved[:, "LR".index(order[k])])
        self.last[j] = (start.state, refreshed.commands, prediction, start.preparation)
        feet = self.predictor.model.feet(start.state.tensors[:6])[0]
        rows, hard, _ = landing_constraint_rows(self.predictor, feet, order, refreshed.commands, prediction)
        passed = start.compatible & hard & (torch.cat([v.reshape(len(u), -1) for v in rows.values()], 1) <= 0).all(1)
        passed &= validate_step_geometry(self.world, self.adapter, prediction, object_poses=objects, held=held)["valid"]
        _, after = _gap_stances(self.standin, j, before, u)
        return passed, after, prediction.terminal.tensors[4][:, self.adapter.columns]

    def refresh(self, particles: dict) -> None:
        """One refresh (design 6) at the particles' current commands, retract postures, placements and grasps: every gap
        with steps in order, values only. ``q_land`` and ``sole_to`` are written in place; ``standin``, ``last``,
        ``tests_ok`` and ``stats`` follow."""
        began = monotonic()
        n = len(particles["left_q0"])
        with torch.no_grad(), prediction_precision():
            commands = [particles[self.command_names[j]] if j in self.command_names else None
                        for j in range(len(self.groups))]
            read, _ = self.standin.stances(self.incoming, commands)  # what the costs read until now
            moved = torch.stack([(u - self.standin.at[j])[..., :2].norm(dim=-1).amax()
                                 for j, u in enumerate(commands) if u is not None]).amax()
            _, objects, _, held_before = _placements_and_grasps(self.groups, self.world, self.pools, 0, n,
                                                                bindings=particles, grasp_indices=particles)
            ok, head, before = torch.ones(n, dtype=torch.bool, device=self.incoming.device), None, self.incoming
            errors = {}
            for j, (steps, _) in enumerate(self.groups):
                if not steps:
                    continue  # it stands on the pair before it
                passed, before, head = self._run_gap(j, particles, head, before, objects[j], held_before[j])
                ok &= passed
                landed = torch.stack([step.achieved for step in self.last[j][2].steps], 1)
                distance = (read[j][..., :2] - landed[..., :2]).norm(dim=-1).amax((1, 2))
                errors[j] = torch.stack((distance.nanmedian(), distance.nanquantile(0.9)))
            self.tests_ok = ok
            self.stats["standin_error_m"].append({j: e.tolist() for j, e in errors.items()})
            self.stats["command_move_m"].append(moved.item())
        self.stats["refresh_s"].append(monotonic() - began)
