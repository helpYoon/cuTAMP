"""A plan with steps inside the Adam loop (design sections 4 to 7): its starting values, its refresh on the world model
and controller, the route that runs it through the loop, and the helpers the stepped route shares with the motion
stage."""

from collections.abc import Iterable
from dataclasses import replace
from functools import partial
from time import monotonic

import torch
from curobo.types import JointState

from cutamp import held
from cutamp.com_polygon_cost import physical_support_centroid, physical_support_margin
from cutamp.cost_function import CostFunction
from cutamp.costs import curobo_pose_error
from cutamp.grasps import GRASP_SCORE_CUTOFF
from cutamp.landing_predictor import LandingState, nominal_landing_dynamics, prediction_precision
from cutamp.optimize_plan import ParticleOptimizer, _finite_step
from cutamp.particle_initialization import IKTol, ParticleInitializer, _draw_placements, pick_place_confs, worst_posture
from cutamp.robots.t1 import REAL_LEGS
from cutamp.rollout import Rollout, RolloutFunction
from cutamp.samplers import project_supported_placement
from cutamp.stance import STEP_TRAVEL, UNIT_M, UNIT_RAD, abs_pose, rel_pose, stance_joint_limits
from cutamp.step_bound import planner_step_bound, travel
from cutamp.step_costs import GROWTH, WEIGHT_CAP, RequirementWeights, joint_effort, lines, plan_rank, side_lean
from cutamp.step_ik import StanceIK, solve_stance_ik
from cutamp.step_planner import StepResult, StepRow
from cutamp.step_refresh import (
    NoStartingPath,
    StandIn,
    _each,
    command_rows,
    draw_grid_paths,
    grid_commands,
    refresh_gap,
)
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
    return _each(value, lambda x: x[i : i + 1].detach().to(device))


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
        ranked = sorted(rows, key=lambda row: row.order)
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
    """Command domain, phase, per-landing and composed-stance checks on this actual issue state, each its excess over
    its threshold per unit (``UNIT_M`` for a length, ``UNIT_RAD`` for an angle), so met where at most zero."""
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
        rows[prefix + "min_travel"] = (STEP_TRAVEL[0] - t) / UNIT_M
        rows[prefix + "max_travel"] = (t - STEP_TRAVEL[1]) / UNIT_M
        rows[prefix + "command_bound"] = bounds.g(c, foot) / UNIT_M
        dest = abs_pose(before[:, 1 - s], c)
        pair = torch.stack((dest, before[:, 1]) if s == 0 else (before[:, 0], dest), dim=1)
        if pred is not None:
            step = pred.steps[j]
            pair = step.achieved
            ph = step.phases
            profile = predictor.profile
            for key, unit in (
                ("deadline_height", UNIT_M),
                ("support_xy", UNIT_M),
                ("support_yaw", UNIT_RAD),
                ("hold_xy", UNIT_M),
                ("hold_yaw", UNIT_RAD),
            ):
                rows[prefix + key] = (getattr(ph, key) - getattr(profile, key)) / unit
            rows[prefix + "terminal_sole_height"] = (step.sole_height[:, -1].abs() - 0.02) / UNIT_M
            rows[prefix + "terminal_sole_tilt"] = (step.sole_tilt[:, -1] - 0.10) / UNIT_RAD
            hard = hard & ph.deadline_ok & ph.hold_ok & ph.support_ok
        actual = rel_pose(pair[:, 1 - s], pair[:, s])
        rows[prefix + "achieved_bound"] = bounds.g(actual, foot) / UNIT_M
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


def _check_line(line, u):
    """``[n, K]``: a gap's checks on their straight line (2026-11-23 design 4.4) at its commands ``u`` ``[n, s, 3]``."""
    return line["value"] + torch.einsum("nkc,nc->nk", line["slope"], (u - line["at"]).reshape(len(u), -1))


CLEARANCE = 1e-6  # m and rad: the depth :func:`validate_step_geometry`'s verdict accepts


def _clearances(check, n, blocks, device):
    """A gap's geometry verdict and clearances over its refresh's rows: ``blocks`` blocks of ``n``, the particles' own
    rows, then one block per nudged command number. ``check(rows)`` is :func:`validate_step_geometry` on the rows
    ``rows`` of that rollout. The own rows are always checked; a particle's copies only where its own row has a
    clearance above ``CLEARANCE``, else they take its own row's values, so its clearances get no slope (2026-11-24
    design). Returns ``(valid [n], penetration {name: [blocks * n]})``."""
    own = check(torch.arange(n, device=device))
    penetration = {name: value.repeat(blocks) for name, value in own["penetration"].items()}
    violating = (torch.stack(tuple(own["penetration"].values()), 1) > CLEARANCE).any(1).nonzero().flatten()
    if len(violating):
        rows = (torch.arange(1, blocks, device=device)[:, None] * n + violating).flatten()
        for name, value in check(rows)["penetration"].items():
            penetration[name][rows] = value
    return own["valid"], penetration


def _finite_rows(state):
    """``[B]`` bool: the rows of a physical state (p, R, v, w, q, qd) whose every value is finite."""
    return torch.stack([t.reshape(len(t), -1).isfinite().all(-1) for t in state]).all(0)


def _zero_unless(finite, state):
    """``state`` with the rows not ``finite`` set to zero, so a diverged row cannot make a batched solve raise."""
    return tuple(torch.where(finite.reshape(-1, *[1] * (t.ndim - 1)), t, 0) for t in state)


def _write(particles, key, value):
    """A new key takes a copy of ``value``; an existing one is written in place, so the loop's tensor stays the same
    object."""
    if key not in particles:
        particles[key] = value.clone()
    else:
        particles[key].copy_(value)


def _keep_finite(value, new):
    """``value`` ``[n, ...]`` overwritten in place by ``new``, except the rows where ``new`` is not finite."""
    finite = new.reshape(len(new), -1).isfinite().all(1)
    value.copy_(torch.where(finite.view(-1, *[1] * (value.ndim - 1)), new, value))


class SteppedPlan:
    """A plan with steps inside the Adam loop (design 4 to 6).

    A particle's commands, ``[n, s, 3]`` per gap with steps, are Adam variables that start on a half-foot grid path over
    the task planner's foot orders that reaches each target and keeps the feet off objects, with noise within half a
    cell (:func:`cutamp.step_refresh.draw_grid_paths`); when no grid path reaches, the particles start on legal,
    feet-clear paths that do not reach. A particle's pick, place and retract postures are Adam variables that start from
    one solve on the stance the first refresh gives, and are held on their hand target and stance (2026-11-23 design
    4.1): each evaluation puts them back first (:meth:`rollout`), their slope along that hold reaches the commands and
    the placements (:meth:`hand_on`), and each Adam step ends with the postures inside their joint limits and the
    placements on their support (:meth:`step`). A refresh runs the world model and controller on every particle's
    commands, gap by gap, writes the landing postures (never optimized), sets the stand-in and the step's checks'
    straight lines (design 4.4) the costs read until the next refresh and puts the pick, place and retract postures
    back on the pairs they stand on; it solves none (2026-11-23 design 4.7). The starting pass is the first refresh,
    with one solve per posture.
    """

    def __init__(self, proposal, world, config, predictor, inputs, pools, checker, *, reducer, deadline=None, generator=None):
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
        self.weights = RequirementWeights(reducer, checker)
        # Per gap with steps, its checks' straight lines in its commands and their weights (2026-11-23 design 4.4).
        self.checks = {}
        self.stats = {"refresh_s": [], "standin_error_m": [], "command_move_m": [], "correction_capped": [],
                      "check_line_error": []}
        # Each held posture's J from its last put-back, and the rows the correction capped since the last refresh.
        self.held_jac, self.capped = {}, 0
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

            self.cost_function = CostFunction(self.arm_skeleton, world, config, self_collision_scales=True)
            self.checker, self._retract_of = checker, retract_of
            ok, head, before = torch.ones(n, dtype=torch.bool, device=like.device), None, self.incoming
            for j, (steps, op) in enumerate(self.groups):
                _check_deadline(deadline)
                seed = particles[self._start_of[j]]  # the posture the robot stands in here: the start, or its landing
                if steps:
                    passed, before, head, _ = self._run_gap(j, particles, head, before, objects[j], held_before[j])
                    ok &= passed
                    seed = particles[steps[-1].arg("q_land")]
                self._solve_postures(j, particles, before, wrists[j], objects[j], held_before[j], seed)
                self.references[op.arg("q") + "_ik_reference"] = seed.clone()
                self.start_postures[op.arg("q")] = particles[op.arg("q")].clone()
                if j in retract_of:
                    self.start_postures[retract_of[j]] = particles[retract_of[j]].clone()
            self.tests_ok = ok
        self.particles = particles
        self.stats["standin_error_m"].append(None)  # no stand-in before the first refresh
        self.stats["command_move_m"].append(None)
        self.stats["refresh_s"].append(monotonic() - began)

    def _solve_postures(self, j, particles, before, wrist, objects, held, seed):
        """Gap ``j``'s pick or place posture on the pair ``before`` ``[n, 2, 3]`` it stands on, by
        :func:`solve_stance_ik` with ``wrist`` ``[n, 4, 4]`` as the hand target, then its retract on the same pair by
        :func:`_retract_on_stance`, both seeded by ``seed`` and their keys made (design 4 and 13). An arm holding an
        object keeps the pose it has in the posture the gap starts from; an idle arm away from home makes the solve
        check collisions in ``objects`` with ``held`` carried, the scene before gap ``j``."""
        world, op = self.world, self.groups[j][1]
        arm, name, retract = op.operator.metadata.arm, op.arg("q"), self._retract_of.get(j)
        rest = particles[self._start_of[j]]
        opposite = "right" if arm == "left" else "left"
        carrying = rest[:, getattr(world.model, opposite + "_arm")] if opposite in held else None
        solve = partial(solve_stance_ik, center_posture=bool(self.config.com_center_weight))
        if carrying is not None:
            solve = partial(solve, other_arm_reference=carrying)
        else:
            idle = world.model.right_arm if arm == "left" else world.model.left_arm
            nominal = seed.new_tensor(world.model.home[idle]).expand_as(seed[:, idle])
            # A nominal idle arm is left to the loop's collision costs.
            if not torch.equal(seed[:, idle], nominal):
                solve = partial(solve, collides=partial(self._collides, op, particles, objects, held))
        _write(particles, name, solve(before, wrist, seed, arm).q)
        if retract is not None:
            q = _retract_on_stance(before, seed, arm, world.joint_limits, other_arm_reference=carrying).q
            _write(particles, retract, q)

    def _collides(self, action, particles, objects, held, q):
        """``[len(q)]`` bool: posture ``q`` of ``action`` collides, by the loop's collision costs against the checker's
        tolerances, in the scene ``objects`` with ``held`` carried."""
        binding = action.arg("grasp" if action.operator.metadata.action_type == "pick" else "placement")
        variables = {action.arg("q"): q, binding: particles[binding]}
        carried = {obj: (arm, transform) for arm, (obj, transform) in held.items()}
        values = self.cost_function.stepped_scene_costs(variables, [action], objects, carried)["values"]
        blocked = values.pop("self_collision").reshape(len(q), -1).amax(1) > 0
        for key, value in values.items():
            blocked |= value.reshape(len(q), -1).amax(1) > self.checker._get_tol("Collision", key)
        return blocked

    def _run_gap(self, j, particles, head, before, objects, held):
        """Gap ``j``'s hold, its rollout and slope at the current commands, the stand-in set, its landing postures and
        ``sole_to`` written, ``last[j]`` kept and its checks read (:meth:`_set_checks`): the landing's on every row of
        the rollout, the clearances as :func:`_clearances` reads them.
        ``head`` ``[n, 2]`` (None: zero) are the head angles the earlier rollout ended with; ``before`` ``[n, 2, 3]``
        is the stand-in's pair the gap starts on. Returns its tests' pass ``[n]`` (the particles' own rows), the pair
        after it, the head angles its rollout ends with and :meth:`_set_checks`'s misses."""
        steps, order = self.groups[j][0], self.orders[j]
        if j == 0:
            start = self._first
        else:
            self.adapter.head = head
            start = self.adapter.initialize(particles[self._start_of[j]].detach(), self._dynamics, world=self.world,
                                            object_poses=objects, held=held)
        u = particles[self.command_names[j]]
        n = len(u)
        anchor = before[:, 1 - "LR".index(order[0])]
        refreshed = refresh_gap(self.predictor, start.state, order, u, anchor, self.standin.slope[j])
        self.standin.set(j, refreshed)
        prediction, copies = refreshed.prediction, refreshed.copies
        blocks = len(refreshed.copy_commands) // n

        def per_copy(x):
            return x.repeat(blocks, *[1] * (x.ndim - 1))

        # A diverged row (a state not finite anywhere in its rollout) is garbage and fails the tests. It is zeroed where
        # the planner's batched solve reads it, so the solve still answers for the other rows.
        finite = torch.stack([_finite_rows(s.trajectory) & _finite_rows(s.terminal.tensors[:6])
                              for s in copies.steps]).all(0)
        own = finite[:n]
        checked = replace(copies, steps=tuple(replace(s, trajectory=_zero_unless(finite, s.trajectory))
                                              for s in copies.steps))
        for k, step in enumerate(steps):
            landed = prediction.steps[k]
            q_land = self.adapter.planner(_zero_unless(own, landed.terminal.tensors[:6]))[0]
            _write(particles, step.arg("q_land"), q_land)
            _write(particles, step.arg("sole_to"), landed.achieved[:, "LR".index(order[k])])
        self.last[j] = (start.state, refreshed.commands, prediction, start.preparation)
        feet = self.predictor.model.feet(start.state.tensors[:6])[0]
        rows, hard, _ = landing_constraint_rows(self.predictor, per_copy(feet), order, refreshed.copy_commands, copies)

        def check(chosen):
            # The geometry of those rows of the rollout, each in its own particle's scene. per_copy first: it lays a
            # pose out row by row, as this check received every pose before, where a held pose (an inverse) comes
            # column by column and a product's last bit follows that layout.
            return validate_step_geometry(
                self.world, self.adapter, _each(checked, lambda x: x[chosen]),
                object_poses={name: per_copy(pose)[chosen] for name, pose in objects.items()},
                held={arm: (name, per_copy(pose)[chosen]) for arm, (name, pose) in held.items()},
            )

        valid, penetration = _clearances(check, n, blocks, u.device)
        passed = start.compatible & hard[:n] & (torch.cat([v[:n].reshape(n, -1) for v in rows.values()], 1) <= 0).all(1)
        passed &= valid
        miss = self._set_checks(j, rows, penetration, refreshed, finite)
        _, after = _gap_stances(self.standin, j, before, u)
        head = prediction.terminal.tensors[4][:, self.adapter.columns]
        return passed & own, after, torch.where(own[:, None], head, 0), miss  # a diverged row hands on a zero head

    def _set_checks(self, j, rows, penetration, refreshed, finite):
        """``checks[j]`` from gap ``j``'s rows on every row of its refresh ``refreshed`` (2026-11-23 design 4.4): the
        :func:`landing_constraint_rows` ``rows`` other than the step's range, each the worst of its columns, then the
        clearances ``penetration`` (:func:`_clearances`), each counted from ``CLEARANCE`` and per unit;
        :func:`lines` gives their value and slope, ``finite`` ``[(1 + 3 s) n]`` the rows that did not
        diverge. The weights start at 1 and double, up to ``WEIGHT_CAP``, where a check is above zero at a later
        refresh. Returns how far the last refresh's lines missed the values read now, in units, over the checks either
        puts above zero on the particles whose own row did not diverge (None: no earlier line)."""
        names, columns = [], []
        for key, value in rows.items():
            if not key.endswith(("min_travel", "max_travel", "command_bound", "achieved_bound")):
                names.append(key)
                columns.append(value.reshape(len(value), -1).amax(1))
        for key, value in penetration.items():
            names.append("geometry_" + key)
            columns.append((value - CLEARANCE) / (UNIT_RAD if key.endswith("_rad") else UNIT_M))
        n = len(refreshed.commands)
        value, slope = lines(torch.stack(columns, 1), n, refreshed.ticks_changed, finite)
        old, miss = self.checks.get(j), None
        if old is not None and old["names"] == names:
            guess = _check_line(old, refreshed.commands)
            miss = (guess - value)[((guess > 0) | (value > 0)) & finite[:n, None]].abs()
            weight = torch.where(value > 0, old["weight"] * GROWTH, old["weight"]).clamp_max(WEIGHT_CAP)
        else:
            weight = torch.ones_like(value)
        self.checks[j] = {"names": names, "value": value, "slope": slope, "at": refreshed.commands, "weight": weight}
        return miss

    def refresh(self, particles: dict) -> None:
        """One refresh (design 6; 2026-11-23 design 4.7) at the particles' current commands, postures, placements and
        grasps, values only, gap by gap in order: its hold and rollout if it has steps, then its pick or place and
        retract postures put back (:meth:`put_back`) on the pair it stands on, so the next gap's hold starts from the
        retract put back. ``q_land``, ``sole_to`` and the postures are written in place; ``standin``, ``last``,
        ``checks``, ``tests_ok`` and ``stats`` follow, ``stats["correction_capped"]`` with the rows the correction
        capped since the last refresh and ``stats["check_line_error"]`` with the median miss of the checks' last lines
        over every gap (:meth:`_set_checks`; None where there is none)."""
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
            errors, misses = {}, []
            for j, (steps, _) in enumerate(self.groups):
                if steps:  # else it stands on the pair before it
                    passed, before, head, miss = self._run_gap(j, particles, head, before, objects[j], held_before[j])
                    ok &= passed
                    if miss is not None:
                        misses.append(miss)
                    landed = torch.stack([step.achieved for step in self.last[j][2].steps], 1)
                    distance = (read[j][..., :2] - landed[..., :2]).norm(dim=-1).amax((1, 2))
                    errors[j] = torch.stack((distance.nanmedian(), distance.nanquantile(0.9)))
                self.put_back(particles, (j,))
            self.tests_ok = ok
            self.stats["standin_error_m"].append({j: e.tolist() for j, e in errors.items()})
            self.stats["command_move_m"].append(moved.item())
            self.stats["correction_capped"].append(int(self.capped))
            self.capped = 0
            missed = torch.cat(misses) if misses else torch.empty(0)
            self.stats["check_line_error"].append(missed.median().item() if len(missed) else None)
        self.stats["refresh_s"].append(monotonic() - began)
        self.weights.grow()

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

    def hand_targets(self, particles: dict) -> list[torch.Tensor]:
        """Per gap, the hand's target ``[n, 4, 4]`` as the hand cost reads it: the placements as Adam holds them, not
        projected onto their support. Differentiable in the placements."""
        n = len(particles["left_q0"])
        return _placements_and_grasps(self.groups, self.world, self.pools, 0, n, bindings=particles,
                                      grasp_indices=particles, project=False)[2]

    def _holds_hand(self, name):
        """Whether posture ``name`` is a pick or place (held on its hand target and its soles), not a retract."""
        return name == self.groups[self.gap_of[name]][1].arg("q")

    def put_back(self, particles: dict, gaps: Iterable[int] | None = None) -> None:
        """Without a gradient, every pick, place and retract posture of the gaps ``gaps`` (None: all) put back in place
        on its stand-in stance by :func:`held.put_back` (2026-11-23 design 4.1): a pick or place with its hand target
        and arm, a retract with its soles only. Each posture's ``J`` is kept in ``held_jac``; the rows the correction
        capped are added to ``capped``."""
        gaps = set(range(len(self.groups)) if gaps is None else gaps)
        with torch.no_grad():
            table, targets = self.stance_table(particles), self.hand_targets(particles)
            for name, j in self.gap_of.items():
                if j not in gaps:
                    continue
                q = particles[name]
                if self._holds_hand(name):
                    hand = targets[j], self.groups[j][1].operator.metadata.arm
                else:
                    hand = None, None
                q_new, self.held_jac[name], capped = held.put_back(self.world, q.detach(), table[name], *hand)
                q.copy_(q_new)
                self.capped = self.capped + capped.sum()

    def rollout(self, particles: dict) -> Rollout:
        """:meth:`put_back` on every posture, then the arm skeleton's rollout: what the loop and :meth:`result`
        evaluate."""
        self.put_back(particles)
        return self.rollout_fn(particles)

    def hand_on(self, particles: dict) -> None:
        """After the backward (2026-11-23 design 4.2): each posture with a gradient and a kept ``J`` keeps the part of
        its gradient that moves neither its hand nor its feet (:func:`held.split_gradient`), and its multipliers reach
        the commands through the stand-in's stances (:func:`held.stance_share`) and, for a pick or place, the
        placements through its hand target (:func:`held.target_share`): one gradient of their sum, added to the
        commands' and placements' gradients with entries that are not finite as zero."""
        mus = {}
        for name, jac in self.held_jac.items():
            grad = particles[name].grad
            if grad is not None:
                kept, mus[name] = held.split_gradient(grad, jac)
                grad.copy_(kept)
        places = [op.arg("placement") for _, op in self.groups if op.operator.metadata.action_type == "place"]
        leaves = [particles[key] for key in (*self.command_names.values(), *places)
                  if particles[key].requires_grad and particles[key].grad is not None]
        if not mus or not leaves:
            return
        with torch.enable_grad():
            table, targets = self.stance_table(particles), self.hand_targets(particles)
            share = sum(held.stance_share(mu, table[name]) for name, mu in mus.items())
            share = share + sum(held.target_share(mu, targets[self.gap_of[name]])
                                for name, mu in mus.items() if self._holds_hand(name))
            if not share.requires_grad:
                return
            grads = torch.autograd.grad(share, leaves, allow_unused=True)
        for leaf, grad in zip(leaves, grads):
            if grad is not None:
                leaf.grad.add_(grad.nan_to_num(0.0, posinf=0.0, neginf=0.0))

    def step(self, optimizer, params, particles: dict) -> torch.Tensor:
        """One Adam step (2026-11-23 design 4.1): :meth:`hand_on`, :func:`_finite_step`, then without a gradient each
        posture clamped to its stand-in stance's joint limits (:func:`stance_joint_limits`) and each placement put
        back on its support (:func:`project_supported_placement`); a row that would not be finite keeps its value.
        Returns :func:`_finite_step`'s held rows."""
        self.hand_on(particles)
        rows = _finite_step(optimizer, params)
        with torch.no_grad():
            table = self.stance_table(particles)
            for name in self.gap_of:
                q = particles[name]
                lower, upper = stance_joint_limits(table[name].to(q), *self.world.joint_limits.to(q))
                _keep_finite(q, q.clamp(lower, upper))
            for _, op in self.groups:
                if op.operator.metadata.action_type == "place":
                    pose = particles[op.arg("placement")]
                    _keep_finite(pose, project_supported_placement(pose, self.world.get_collision_spheres(op.arg("obj")),
                                                                   self.world.get_aabb(op.arg("surface"))))
        return rows

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
        shortfall itself, the retracts' feet, the commands' rows, the last refresh's tests (read by the checker only),
        the step's checks on their straight lines in the commands (``StepChecks``, design 4.4) and, as the soft cost's
        only values, the preferences over the pick and place postures, each in its unit (2026-11-23 design 4.5): side
        lean (:func:`side_lean`), off centre and joint effort (:func:`joint_effort`); beside them, detached, each
        particle's ``rank`` (:func:`plan_rank`), its worst ``effort`` and ``cost_at_start_weights``, the total with
        every requirement at its starting weight and the check lines at weight 1, which the stop rule's countdown reads
        before any particle is valid (design 4.6). Every requirement row carries its weight per particle
        (:class:`RequirementWeights`, 2026-11-23 design 4.3); the hand and feet rows weigh zero: they are held, not
        costs (design 4.1), and the checker still reads them."""
        table, after = self._stances(particles)
        inner = {key: value for key, value in particles.items() if key not in self.landing_confs}
        cost = self.cost_function(rollout, inner, achieved_stances=table)
        stances = torch.stack([table[name] for name in rollout["conf_params"]], 1)
        margin = physical_support_margin(rollout["com_xy"], stances, self.world.model.sole_box)
        shortfall = (self.config.com_comfort_margin_m - margin).clamp_min(0)
        cost[ComComfort.type] = {"type": "constraint", "values": {"margin_shortfall": shortfall}}
        kinematic = cost[KinematicConstraint.type]
        kinematic["values"].update(self._retract_sole_errors(particles, table))
        rows = command_rows(self.orders, self.incoming, self._commands(particles), after)
        cost["StepCommand"] = {"type": "constraint", "values": rows}
        cost["StepTests"] = {"type": "constraint", "values": {"refresh_tests": (~self.tests_ok).to(margin.dtype)}}
        steer, at_one = margin.new_zeros(len(margin)), margin.new_zeros(len(margin))
        for j, line in self.checks.items():
            excess = _check_line(line, particles[self.command_names[j]]).clamp_min(0)
            steer = steer + (line["weight"] * excess).sum(1)
            at_one = at_one + excess.detach().sum(1)
        cost["StepChecks"] = {"type": "cost", "values": {"lines": steer}}
        q = torch.stack([particles[name] for name in rollout["conf_params"]], 1)
        lean = side_lean(q, stances, self.world.model)
        distance = (rollout["com_xy"] - physical_support_centroid(stances, self.world.model.sole_box)).norm(dim=-1)
        effort = joint_effort(self.world, q)
        weight = self.config.com_center_weight
        soft = cost.setdefault("soft", {"type": "cost"})
        soft["values"] = {"side_lean": (lean / UNIT_RAD).square().sum(1),
                          "off_centre": weight * (distance / UNIT_M).sum(1),
                          "joint_effort": effort.sum(1)}
        worst = distance.detach().amax(1) if weight else margin.new_zeros(len(margin))
        soft["rank"] = plan_rank(lean.detach().abs().clamp(max=1).asin().amax(1), worst)
        soft["effort"] = effort.detach().amax(1)
        self.weights.apply(cost)
        soft["cost_at_start_weights"] = self.weights.total_at_start(
            {**cost, "StepChecks": {"type": "cost", "values": {"lines": at_one}}})
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
        gradient, one :class:`StepRow` per particle the checker passes, its report with the side lean and joint effort
        of each pick and place and its place in the order of the preferences (``rank``, then the worst ``joint_effort``;
        2026-11-23 design 4.5). ``motion_inputs`` maps each row's index to the adapter and the row's saved refresh, for
        the motion stage; it stays out of ``diagnostics``."""
        n = len(particles["left_q0"])
        sole_box = self.world.model.sole_box
        with torch.no_grad():
            rollout = self.rollout(particles)
            costs = self(rollout, particles)
            masks = checker.get_full_mask(costs)
            passed = checker.get_mask(costs, verbose=False)
            soft, effort = reducer.soft_costs(costs), reducer.secondary_effort(costs)
            at_start = costs["soft"]["cost_at_start_weights"]  # the weights grow; reported at their start
            ep = self._endpoints(masks, n)
            table = self.stance_table(particles)
            stances = torch.stack([table[name] for name in rollout["conf_params"]], 1)
            margin = physical_support_margin(rollout["com_xy"], stances, sole_box)
            center = (rollout["com_xy"] - physical_support_centroid(stances, sole_box)).norm(dim=-1)
            q = torch.stack([particles[name] for name in rollout["conf_params"]], 1)
            lean = side_lean(q, stances, self.world.model).clamp(-1, 1).asin()
            efforts = joint_effort(self.world, q)
            rank, worst_effort = costs["soft"]["rank"], costs["soft"]["effort"]
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
                side_lean_rad=lean[i].tolist(),
                joint_efforts=efforts[i].tolist(),
                joint_effort=float(worst_effort[i]),
                rank=float(rank[i]),
                cost=float(at_start[i]),
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
        best = int((ep.sum(1) * -1e6 + at_start.nan_to_num(1e6)).argmin())
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
        plan = SteppedPlan(proposal, world, config, predictor, inputs, pools, constraint_checker,
                           reducer=cost_reducer, deadline=deadline)
    except TimeoutError as e:
        return StepResult([], None, "budget_exhausted", str(e))
    except NoStartingPath as e:
        return StepResult([], None, "initialization_failed", str(e))
    except ValueError as e:
        return StepResult([], None, "unresolved", str(e))
    info = {"plan_skeleton": plan.arm_skeleton, "particles": plan.particles, "rollout_fn": plan.rollout,
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
