"""A plan with steps inside the Adam loop (design sections 4 to 7): its starting values, its refresh on the world model
and controller, the route that runs it through the loop, and the helpers the stepped route shares with the motion
stage."""

from dataclasses import fields, is_dataclass, replace
from functools import partial
from time import monotonic

import torch
from curobo.types import JointState

from cutamp.com_polygon_cost import physical_support_centroid, physical_support_margin
from cutamp.cost_function import CostFunction
from cutamp.costs import curobo_pose_error
from cutamp.grasps import GRASP_SCORE_CUTOFF
from cutamp.landing_predictor import LandingState, nominal_landing_dynamics, prediction_precision
from cutamp.optimize_plan import ParticleOptimizer
from cutamp.particle_initialization import IKTol, ParticleInitializer, _draw_placements, pick_place_confs, worst_posture
from cutamp.robots.t1 import REAL_LEGS
from cutamp.rollout import RolloutFunction
from cutamp.samplers import project_supported_placement
from cutamp.stance import STEP_TRAVEL, abs_pose, rel_pose, stance_joint_limits
from cutamp.step_bound import planner_step_bound, travel
from cutamp.step_ik import StanceIK, solve_stance_ik
from cutamp.step_planner import StepResult, StepRow
from cutamp.step_refresh import NoStartingPath, StandIn, command_rows, draw_grid_paths, grid_commands, refresh_gap
from cutamp.step_validation import StepStateAdapter, _foot_half_extents, _object_box, validate_step_geometry
from cutamp.task_planning.constraints import ComComfort, KinematicConstraint
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


def _placements_and_grasps(groups, world, pools, first, stop, bindings=None, grasp_indices=None, project=True):
    """``project`` False takes the ``bindings`` placements as they are, without the support projection."""
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
            if bindings is not None and key in bindings and not project:
                particles[key] = bindings[key]
            elif bindings is not None and key in bindings:
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


def _foot_scene(world, poses: dict, held: dict, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The refresh's foot scene (:func:`cutamp.step_validation._step_geometry`) without its carried objects, the scene
    the starting draw keeps the feet clear of: boxes ``[n, b, 4, 4]`` and half extents ``[b, 3]``, float64 on the CPU,
    of ``world.movables + world.statics`` except the objects ``held`` (arm -> (name, grasp)) carries, each posed by
    ``poses`` (name -> ``[n, 4, 4]``) or else ``world.get_object_pose`` and boxed by :func:`_object_box`. The refresh
    also tests the object in the hand; the draw cannot, as the hand's pose is not known there."""
    carried = {name for name, _ in held.values()}
    boxes, halves = [], []
    for obj in world.movables + world.statics:
        if obj.name in carried:
            continue
        pose = poses[obj.name] if obj.name in poses else world.get_object_pose(obj)
        box, half = _object_box(obj, pose.detach().to("cpu", torch.float64).reshape(-1, 4, 4).expand(n, -1, -1))
        boxes.append(box)
        halves.append(half)
    if not boxes:
        return torch.empty(n, 0, 4, 4, dtype=torch.float64), torch.empty(0, 3, dtype=torch.float64)
    return torch.stack(boxes, 1), torch.stack(halves)


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
            for key in (
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


def _finite_rows(state):
    """``[B]`` bool: the rows of a physical state (p, R, v, w, q, qd) whose every value is finite."""
    return torch.stack([t.reshape(len(t), -1).isfinite().all(-1) for t in state]).all(0)


def _zero_unless(finite, state):
    """``state`` with the rows not ``finite`` set to zero, so a diverged row cannot make a batched solve raise."""
    return tuple(torch.where(finite.reshape(-1, *[1] * (t.ndim - 1)), t, 0) for t in state)


def _write(particles, key, value, rows=None):
    """A new key takes a copy of ``value``; an existing one is written in place, so the loop's tensor stays the same
    object: only its rows ``rows`` (``[n]`` bool) when given."""
    if key not in particles:
        particles[key] = value.clone()
    elif rows is None:
        particles[key].copy_(value)
    else:
        particles[key][rows] = value


class SteppedPlan:
    """A plan with steps inside the Adam loop (design 4 to 6).

    A particle's commands, ``[n, s, 3]`` per gap with steps, are Adam variables that start on a half-foot grid path over
    the task planner's foot orders that reaches each target and keeps the feet off objects, with noise within half a
    cell (:func:`cutamp.step_refresh.draw_grid_paths`); when no grid path reaches, the particles start on legal,
    feet-clear paths that do not reach. A particle's pick, place and retract postures are Adam variables that start from
    one solve on the stance the first refresh gives. A refresh runs the world model and controller on every particle's
    commands, gap by gap, writes the landing postures (never optimized), sets the stand-in the costs read until the next
    refresh and re-solves the pick, place and retract postures on the pairs they stand on (design 13). The starting pass
    is the first refresh.
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
        self.stats = {"refresh_s": [], "resolve_s": [], "standin_error_m": [], "command_move_m": []}
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
            # From the first gap with steps on (before it the feet are fixed), a gap whose landing the task planner
            # measured tests reach: a stance passes when its hand-map gap to the target is at most that landing's.
            # Each gap with steps keeps its feet clear of its own scene.
            search = proposal.search
            grid = grid_commands(search.sole_half, search.home[0][1] - search.home[1][1])
            j0 = next(j for j, order in enumerate(self.orders) if order)
            reach, scenes = [], []
            for j, order in enumerate(self.orders):
                bar = proposal.schedule.edges[j].distance_m
                reach.append(None if j < j0 or bar is None
                             else lambda pairs, t=proposal.targets[j], b=bar: search.reaches(pairs, t, b))
                scenes.append(_foot_scene(world, objects[j], held_before[j], n) if order else None)
            foot_half = torch.tensor(_foot_half_extents(world.model), dtype=torch.float64)
            drawing = monotonic()
            commands, self.stats["draw"] = draw_grid_paths(self.orders, self.incoming, generator, grid, reach, scenes,
                                                           foot_half, deadline=deadline)
            self.stats["draw"]["seconds"] = monotonic() - drawing
            for j, u in enumerate(commands):
                if u is not None:
                    particles[self.command_names[j]] = u

            self.cost_function = CostFunction(self.arm_skeleton, world, config)
            self.checker, self._retract_of, resolve = checker, retract_of, 0.0
            ok, head, before = torch.ones(n, dtype=torch.bool, device=like.device), None, self.incoming
            for j, (steps, op) in enumerate(self.groups):
                _check_deadline(deadline)
                seed = particles[self._start_of[j]]  # the posture the robot stands in here: the start, or its landing
                if steps:
                    passed, before, head = self._run_gap(j, particles, head, before, objects[j], held_before[j])
                    ok &= passed
                    seed = particles[steps[-1].arg("q_land")]
                solving = monotonic()
                self._solve_postures(j, particles, before, wrists[j], objects[j], held_before[j], seed)
                resolve += monotonic() - solving
                self.references[op.arg("q") + "_ik_reference"] = seed.clone()
                self.start_postures[op.arg("q")] = particles[op.arg("q")].clone()
                if j in retract_of:
                    self.start_postures[retract_of[j]] = particles[retract_of[j]].clone()
            self.tests_ok = ok
        self.particles = particles
        self.stats["standin_error_m"].append(None)  # no stand-in before the first refresh
        self.stats["command_move_m"].append(None)
        self.stats["resolve_s"].append(resolve)
        self.stats["refresh_s"].append(monotonic() - began)

    def _solve_postures(self, j, particles, before, wrist, objects, held, seed=None):
        """Gap ``j``'s pick or place posture on the pair ``before`` ``[n, 2, 3]`` it stands on, by
        :func:`solve_stance_ik` with ``wrist`` ``[n, 4, 4]`` as the hand target, then its retract on the same pair by
        :func:`_retract_on_stance` (design 4 and 13). An arm holding an object keeps the pose it has in the posture the
        gap starts from; an idle arm away from home makes the solve check collisions in ``objects`` with ``held``
        carried, the scene before gap ``j``.

        ``seed`` given (the start): it seeds both solves and their keys are made. ``seed`` None (a refresh): each
        posture is seeded by its current value and written in place; a row whose pair is not finite is not handed to
        the solves and keeps its postures."""
        world, op = self.world, self.groups[j][1]
        arm, name, retract = op.operator.metadata.arm, op.arg("q"), self._retract_of.get(j)
        rows = None  # every row
        if seed is None:
            finite = before.isfinite().all((1, 2))
            if not finite.any():
                return
            rows = None if finite.all() else finite

        def take(value):
            return value if rows is None else value[rows]

        before, wrist, rest = take(before), take(wrist), take(particles[self._start_of[j]])
        posture_seed = take(particles[name]) if seed is None else seed
        opposite = "right" if arm == "left" else "left"
        carrying = rest[:, getattr(world.model, opposite + "_arm")] if opposite in held else None
        solve = partial(solve_stance_ik, center_posture=bool(self.config.com_center_weight))
        if carrying is not None:
            solve = partial(solve, other_arm_reference=carrying)
        else:
            idle = world.model.right_arm if arm == "left" else world.model.left_arm
            nominal = posture_seed.new_tensor(world.model.home[idle]).expand_as(posture_seed[:, idle])
            # A nominal idle arm is left to the loop's collision costs.
            if not torch.equal(posture_seed[:, idle], nominal):
                solve = partial(solve, collides=partial(self._collides, op, particles, objects, held, owners=rows))
        _write(particles, name, solve(before, wrist, posture_seed, arm).q, rows)
        if retract is not None:
            arrived = take(particles[retract]) if seed is None else seed
            q = _retract_on_stance(before, arrived, arm, world.joint_limits, other_arm_reference=carrying).q
            _write(particles, retract, q, rows)

    def _collides(self, action, particles, objects, held, q, owners=None):
        """``[len(q)]`` bool: posture ``q`` of ``action`` collides, by the loop's collision costs against the checker's
        tolerances, in the scene ``objects`` with ``held`` carried; ``owners`` (``[n]`` bool, None: all) the rows of
        the scene ``q`` is for."""

        def select(value):
            return value if owners is None else value[owners]

        binding = action.arg("grasp" if action.operator.metadata.action_type == "pick" else "placement")
        variables = {action.arg("q"): q, binding: select(particles[binding])}
        scene = {name: select(value) for name, value in objects.items()}
        carried = {obj: (arm, select(transform)) for arm, (obj, transform) in held.items()}
        values = self.cost_function.stepped_scene_costs(variables, [action], scene, carried)["values"]
        blocked = values.pop("self_collision").reshape(len(q), -1).amax(1) > 0
        for key, value in values.items():
            blocked |= value.reshape(len(q), -1).amax(1) > self.checker._get_tol("Collision", key)
        return blocked

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
            start = self.adapter.initialize(particles[self._start_of[j]].detach(), self._dynamics, world=self.world,
                                            object_poses=objects, held=held)
        u = particles[self.command_names[j]]
        anchor = before[:, 1 - "LR".index(order[0])]
        refreshed = refresh_gap(self.predictor, start.state, order, u, anchor, self.standin.slope[j])
        self.standin.set(j, refreshed)
        prediction = refreshed.prediction
        # A diverged row (a state not finite anywhere in its rollout) is garbage and fails the tests. It is zeroed where
        # the planner's batched solve reads it, so the solve still answers for the other rows.
        finite = torch.stack([_finite_rows(s.trajectory) & _finite_rows(s.terminal.tensors[:6])
                              for s in prediction.steps]).all(0)
        checked = replace(prediction, steps=tuple(replace(s, trajectory=_zero_unless(finite, s.trajectory))
                                                  for s in prediction.steps))
        for k, step in enumerate(steps):
            landed = prediction.steps[k]
            q_land = self.adapter.planner(_zero_unless(finite, landed.terminal.tensors[:6]))[0]
            _write(particles, step.arg("q_land"), q_land)
            _write(particles, step.arg("sole_to"), landed.achieved[:, "LR".index(order[k])])
        self.last[j] = (start.state, refreshed.commands, prediction, start.preparation)
        feet = self.predictor.model.feet(start.state.tensors[:6])[0]
        rows, hard, _ = landing_constraint_rows(self.predictor, feet, order, refreshed.commands, prediction)
        passed = start.compatible & hard & (torch.cat([v.reshape(len(u), -1) for v in rows.values()], 1) <= 0).all(1)
        passed &= validate_step_geometry(self.world, self.adapter, checked, object_poses=objects, held=held)["valid"]
        _, after = _gap_stances(self.standin, j, before, u)
        head = prediction.terminal.tensors[4][:, self.adapter.columns]
        return passed & finite, after, torch.where(finite[:, None], head, 0)  # a diverged row hands on a zero head

    def refresh(self, particles: dict) -> None:
        """One refresh (design 6 and 13) at the particles' current commands, postures, placements and grasps, values
        only, gap by gap in order: its hold and rollout if it has steps, then its pick or place and retract postures
        re-solved on the pair it stands on, so the next gap's hold starts from the re-solved retract. ``q_land``,
        ``sole_to`` and the postures are written in place; ``standin``, ``last``, ``tests_ok`` and ``stats``
        follow."""
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
            # The re-solve's hand target and scene: the placements as Adam holds them, which the hand cost reads.
            _, scene, wrists, _ = _placements_and_grasps(self.groups, self.world, self.pools, 0, n, bindings=particles,
                                                         grasp_indices=particles, project=False)
            ok, head, before = torch.ones(n, dtype=torch.bool, device=self.incoming.device), None, self.incoming
            errors, resolve = {}, 0.0
            for j, (steps, _) in enumerate(self.groups):
                if steps:  # else it stands on the pair before it
                    passed, before, head = self._run_gap(j, particles, head, before, objects[j], held_before[j])
                    ok &= passed
                    landed = torch.stack([step.achieved for step in self.last[j][2].steps], 1)
                    distance = (read[j][..., :2] - landed[..., :2]).norm(dim=-1).amax((1, 2))
                    errors[j] = torch.stack((distance.nanmedian(), distance.nanquantile(0.9)))
                solving = monotonic()
                self._solve_postures(j, particles, before, wrists[j], scene[j], held_before[j])
                resolve += monotonic() - solving
            self.tests_ok = ok
            self.stats["standin_error_m"].append({j: e.tolist() for j, e in errors.items()})
            self.stats["command_move_m"].append(moved.item())
            self.stats["resolve_s"].append(resolve)
        self.stats["refresh_s"].append(monotonic() - began)

    def _commands(self, particles):
        return [particles[self.command_names[j]] if j in self.command_names else None for j in range(len(self.groups))]

    def _stances(self, particles):
        """:meth:`stance_table` and the stand-in's pairs after each step of every gap."""
        after, final = self.standin.stances(self.incoming, self._commands(particles))
        table = {"left_q0": self.incoming, "right_q0": self.incoming}
        table.update({name: final[j] for name, j in self.gap_of.items()})
        return table, after

    def stance_table(self, particles: dict) -> dict[str, torch.Tensor]:
        """Posture name to the ``[n, 2, 3]`` sole pair it stands on, from the stand-in at the particles' current
        commands: ``left_q0`` and ``right_q0`` the incoming pair, every pick, place and retract posture its gap's final
        pair. Differentiable in the commands."""
        return self._stances(particles)[0]

    def _retract_sole_errors(self, particles, table):
        """``retract_sole_pos_err`` and ``retract_sole_rot_err`` ``[n, 2 R]``: both soles of every retract posture
        against its stance, as :meth:`CostFunction.kinematic_costs` checks the picks and places, from one FK."""
        names = [name for name, _, _ in self.cost_function.retract_info]
        if not names:
            return {}
        q = torch.stack([particles[name] for name in names], 1)
        n, r = q.shape[:2]
        fk = self.world.site_kinematics("retract_soles").compute_kinematics(JointState.from_position(q.view(n * r, -1)))
        soles = torch.stack([fk.tool_poses.get_link_pose(frame, make_contiguous=True).get_matrix().view(n, r, 4, 4)
                             for frame in self.world.model.sole_frames], 2)
        x, y, yaw = torch.stack([table[name] for name in names], 1).unbind(-1)
        pos, rot = curobo_pose_error(soles, action_4dof_to_mat4x4(torch.stack((x, y, torch.zeros_like(x), yaw), -1)))
        return {"retract_sole_pos_err": pos.flatten(1), "retract_sole_rot_err": rot.flatten(1)}

    def __call__(self, rollout, particles: dict) -> dict:
        """The cost the Adam loop reads (design 5): today's cost on the stand-in's stances, the balance cost as the
        shortfall itself, the retracts' feet, the commands' rows, the last refresh's tests (read by the checker only)
        and staying near the starting postures."""
        table, after = self._stances(particles)
        inner = {key: value for key, value in particles.items() if key not in self.landing_confs}
        cost = self.cost_function(rollout, inner, achieved_stances=table)
        stances = torch.stack([table[name] for name in rollout["conf_params"]], 1)
        margin = physical_support_margin(rollout["com_xy"], stances, self.world.model.sole_box)
        shortfall = (self.config.com_comfort_margin_m - margin).clamp_min(0)
        cost[ComComfort.type] = {"type": "constraint", "values": {"margin_shortfall": shortfall}}
        cost[KinematicConstraint.type]["values"].update(self._retract_sole_errors(particles, table))
        rows = command_rows(self.orders, self.incoming, self._commands(particles), after)
        cost["StepCommand"] = {"type": "constraint", "values": rows}
        cost["StepTests"] = {"type": "constraint", "values": {"refresh_tests": (~self.tests_ok).to(margin.dtype)}}
        stay = sum((particles[name] - start).square().sum(-1) for name, start in self.start_postures.items())
        cost.setdefault("soft", {"type": "cost", "values": {}})["values"]["stay_near_start"] = stay
        return cost

    def _endpoints(self, masks, n):
        """``[n, gaps]`` bool: each pick or place's requirements met, as today's stepped route attributes them, with
        the balance row, the retract sole rows on their gap, each ``gap{j}_`` command row on gap ``j`` and the refresh
        tests on every gap."""
        ep = torch.ones(n, len(self.groups), dtype=torch.bool, device=self.incoming.device)
        retracts = [name for name, _, _ in self.cost_function.retract_info]
        for key, value in masks[KinematicConstraint.type].items():
            if key.startswith("retract_sole_"):
                value = value.reshape(n, len(retracts), 2).all(-1)
                for r, name in enumerate(retracts):
                    ep[:, self.gap_of[name]] &= value[:, r]
                continue
            if key.startswith("sole_"):
                value = value.reshape(n, len(self.groups), 2).all(-1)
            ep &= value
        ep &= masks[ComComfort.type]["margin_shortfall"]
        for key, value in masks["StepCommand"].items():
            ep[:, int(key[len("gap"):key.index("_")])] &= value.reshape(n, -1).all(1)
        ep &= masks["StepTests"]["refresh_tests"][:, None]
        m, retract_index, placed_count = -1, 0, {}
        timeline = [op for op in self.arm_skeleton if op.operator.metadata.action_type in ("pick", "place", "retract")]
        for t, op in enumerate(timeline):
            kind = op.operator.metadata.action_type
            if kind != "retract":
                m += 1
                for key in ("joint_limit", "self_collision"):
                    ep[:, m] &= masks["Motion"][key][:, m]
            else:
                for key in ("retract_joint_limit", "retract_self_collision"):
                    ep[:, m] &= masks["Motion"][key][:, retract_index]
                retract_index += 1
            if kind == "place":
                surface = op.arg("surface")
                k = placed_count.get(surface, 0)
                for suffix in ("_in_xy", "_support"):
                    ep[:, m] &= masks["StablePlacement"][surface + suffix][:, k]
                placed_count[surface] = k + 1
            ep[:, m] &= masks["Collision"][f"{t}_movables"][:, 0] & masks["Collision"]["robot_to_world"][:, t]
            ep[:, m] &= masks["ComPolygon"][op.arg("q_retract" if kind == "retract" else "q")]
        return ep

    def _gap_records(self, particles, i):
        """Per pick or place, row ``i``: its order as the proposal wrote it, its commands, and from the last refresh
        the pair it stands on and its final landing errors (a gap without steps stands on the pair before it)."""
        records, before = [], self.incoming[i]
        for j, order in enumerate(self.orders):
            record = {"order": order or "none", "commands": []}
            if j in self.last:
                _, _, prediction, _ = self.last[j]
                before = prediction.steps[-1].achieved[i]
                record.update(commands=particles[self.command_names[j]][i].tolist(),
                              final_xy_error_m=prediction.final_xy_error[i].tolist(),
                              final_yaw_error_deg=prediction.final_yaw_error[i].rad2deg().tolist())
            record["achieved"] = before.tolist()
            records.append(record)
        return records

    def result(self, particles: dict, reducer, checker) -> StepResult:
        """The rows handed to the motion stage, after the loop's last refresh (design 7): one evaluation without a
        gradient, one :class:`StepRow` per particle the checker passes. ``motion_inputs`` maps each row's index to the
        adapter and the row's saved refresh, for the motion stage; it stays out of ``diagnostics``."""
        n = len(particles["left_q0"])
        sole_box = self.world.model.sole_box
        with torch.no_grad():
            rollout = self.rollout_fn(particles)
            costs = self(rollout, particles)
            masks = checker.get_full_mask(costs)
            passed = checker.get_mask(costs, verbose=False)
            soft, effort, hard = reducer.soft_costs(costs), reducer.secondary_effort(costs), reducer(costs)
            ep = self._endpoints(masks, n)
            table = self.stance_table(particles)
            stances = torch.stack([table[name] for name in rollout["conf_params"]], 1)
            margin = physical_support_margin(rollout["com_xy"], stances, sole_box)
            center = (rollout["com_xy"] - physical_support_centroid(stances, sole_box)).norm(dim=-1)
            joints = range(len(self.world.model.joint_names))
            arms = [*joints[self.world.model.left_arm], *joints[self.world.model.right_arm]]
            move = torch.stack([(particles[name] - start)[:, arms].abs().amax(-1)
                                for name, start in self.start_postures.items()], 1).amax(1)
        adapter = StepStateAdapter(self.predictor)
        self.motion_inputs, rows = {}, []
        for i in torch.where(passed)[0].tolist():
            particle = {key: value[i].detach().clone() for key, value in particles.items()}
            particle.update({key: value[i].clone() for key, value in self.references.items()})
            margins = margin[i].tolist()
            report = dict(
                com_margins_m=margins,
                com_center_distances_m=center[i].tolist(),
                min_com_margin_m=min(margins),
                best_com_margin=min(margins),
                secondary_effort=float(effort[i]),
                cost=float(hard[i]),
                largest_arm_joint_move_rad=float(move[i]),
                gaps=self._gap_records(particles, i),
                model_feasible=True,
                sole_box_m=list(sole_box),
                com_center_weight=self.config.com_center_weight,
                independent_motion="untested",
                deployment_sha256=self.predictor.metadata["deployment_sha256"],
                checkpoint_sha256=self.predictor.metadata["checkpoint_sha256"],
                normalized_dynamics=self._dynamics[i].tolist(),
                step_lag=self.config.step_lag,
            )
            if self.config.curobo_plan:
                report["motion_validation"] = "unattempted"
            saved = {j: _take(trace, i, "cpu") for j, trace in self.last.items()}
            self.motion_inputs[i] = (adapter, saved)
            rows.append(StepRow(i, particle, float(soft[i]), report, evidence=_evidence(adapter, saved)))
        if rows:
            return StepResult(rows, ep, "model_feasible")
        best = int((ep.sum(1) * -1e6 + hard.nan_to_num(1e6)).argmin())
        failures = [f"{kind}.{key}" for kind, items in masks.items() for key, mask in items.items()
                    if not mask[best].all()]
        # How many particles fail each requirement, the requirements every particle meets left out.
        counts = {f"{kind}.{key}": int((~mask.reshape(n, -1).all(1)).sum()) for kind, items in masks.items()
                  for key, mask in items.items()}
        return StepResult(rows, ep, "unresolved", "no particle met all applicable constraints",
                          {"row": best, "failed_constraints": failures,
                           "failure_counts": {key: count for key, count in counts.items() if count}})


def refine_stepped_plan(proposal, world, config, cost_reducer, constraint_checker, *, inputs, predictor, timer,
                        visualizer, deadline) -> StepResult:
    """A plan with steps through the Adam loop (design 4 to 7): its starting values, the loop with its refreshes, the
    rows its last refresh passes and, with ``curobo_plan``, the motion stage on them.

    ``diagnostics["adam"]`` holds the Adam steps run, their rate over the optimizer's call, the step the loop stopped
    early at (None: it did not) and the plan's ``stats``. A deadline that passes before the loop, or during it with no
    row passing, answers ``budget_exhausted``."""
    if predictor is None:
        return StepResult([], None, "unavailable", "missing trained landing predictor")
    config = config.for_steps()
    try:
        pools = ParticleInitializer(world, config, constraint_checker)._pick_pools(proposal.skeleton)
    except ValueError as e:
        return StepResult([], None, "unresolved", str(e))
    try:
        plan = SteppedPlan(proposal, world, config, predictor, inputs, pools, constraint_checker, deadline=deadline)
    except TimeoutError as e:
        return StepResult([], None, "budget_exhausted", str(e))
    except NoStartingPath as e:
        return StepResult([], None, "initialization_failed", str(e))
    except ValueError as e:
        return StepResult([], None, "unresolved", str(e))
    info = {"plan_skeleton": plan.arm_skeleton, "particles": plan.particles, "rollout_fn": plan.rollout_fn,
            "cost_fn": plan, "steps": plan}
    began = monotonic()
    _, opt_metrics, time_exceeded = ParticleOptimizer(config, cost_reducer, constraint_checker)(
        info, timer, visualizer, deadline=deadline
    )
    seconds = monotonic() - began
    result = plan.result(plan.particles, cost_reducer, constraint_checker)
    if time_exceeded and not result.rows:
        result.status, result.reason = "budget_exhausted", "the Adam loop reached the planner deadline"
    run = len(opt_metrics["num_satisfying"])
    result.diagnostics["adam"] = {"steps_run": run, "seconds": seconds, "steps_per_s": run / seconds,
                                  "early_stopped_at": opt_metrics.get("early_stopped_at"), **plan.stats}
    if config.curobo_plan and result.rows:
        from cutamp.step_preview import plan_predicted_step_motion

        # Plans chosen by their centring show their motion after the search, as on the standing route.
        shown = None if config.com_center_weight != 0 else visualizer
        ik_tol = IKTol.from_checker(constraint_checker)

        def attempt(row):
            adapter, saved = plan.motion_inputs[row.index]
            traces = {gap: _take(trace, 0, world.q_init.device) for gap, trace in saved.items()}
            return plan_predicted_step_motion(proposal.skeleton, row, traces, adapter, world, config, timer=timer,
                                              visualizer=shown, ik_tol=ik_tol, deadline=deadline)

        with prediction_precision():  # the precision the motion stage ran at before the loop took stepped plans
            result.diagnostics.update(
                _validate_motion_rows(result.rows, pools, proposal.skeleton, world, config, attempt, deadline=deadline)
            )
    return result
