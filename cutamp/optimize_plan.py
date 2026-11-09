# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import time
from dataclasses import dataclass
from typing import AbstractSet, Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, TypedDict

import torch
from torch.optim import Adam
from tqdm import tqdm

from cutamp import cuda_graphs
from cutamp.conf_locking import apply_grad_masks, build_grad_masks
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    grasp_7dof_to_mat4x4,
)
from cutamp.com_polygon_cost import particle_com_margin
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.gn_refresh import gn_gate, gn_project_balanced
from cutamp.robots.t1 import GRIPPER_CLOSED, GRIPPER_OPEN
from cutamp.rollout import RolloutFunction, movable_spheres_fn
from cutamp.t1_domain import Conf, Pose
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import KinematicConstraint
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


@dataclass
class IKDep:
    """Re-IK dependency: which Conf to refresh and how to rebuild its IK target.

    ``target_fn(particles)`` rebuilds ``world_from_ee`` from the current
    particle state. ``q_name`` is the Conf to overwrite (in-place) with the
    IK solution. ``arm`` selects the active tool frame, and so which arm is
    inactive (GN freezes its columns). ``movables_fn(particles)`` gives the object spheres the conf must clear,
    ``[B, M, 4]`` (:func:`~cutamp.rollout.movable_spheres_fn`).
    """
    q_name: str
    arm: Literal["left", "right"]
    target_fn: Callable[[dict], torch.Tensor]
    movables_fn: Callable[[dict], torch.Tensor]


def _build_ik_deps(
    plan_skeleton: PlanSkeleton, world: TAMPWorld,
) -> List[IKDep]:
    """Walk the plan skeleton and emit one IKDep per actionable op whose Conf
    is determined by Pose / Grasp particles via IK.

    Mirrors ``ParticleInitializer``'s init-time IK targets (Pick /
    Place) so re-IK reproduces what was done at init,
    just from updated particle values. Operators with no IK target
    (Retract, MoveFree, MoveHolding) are skipped.
    """
    deps: List[IKDep] = []
    for idx, ground_op in enumerate(plan_skeleton):
        meta = ground_op.operator.metadata
        if not getattr(meta, "is_actionable", False):
            continue
        action_type = meta.action_type
        arm = meta.arm
        if arm is None:
            continue
        params = ground_op.values
        tool_frame = world.tool_frame_for_arm(arm)
        tool_from_ee = world.tool_from_ee[tool_frame]

        if action_type == "pick":
            obj, grasp_name, q_name = params
            # Snapshot the object's INITIAL pose. If the skeleton ever picks
            # the same object after a place (Pick A → Place A → Pick A), the
            # IK target should follow A's placed pose, not its initial one —
            # not handled today; blocks_t1 has no such re-grasp pattern.
            world_from_obj = world.get_object_pose(obj)
            def target_fn(p, _wo=world_from_obj, _g=grasp_name, _t=tool_from_ee):
                return _wo @ grasp_7dof_to_mat4x4(p[_g]) @ _t
            deps.append(IKDep(q_name, arm, target_fn, movable_spheres_fn(world, plan_skeleton, idx)))
        elif action_type == "place":
            obj, grasp_name, placement, surface, q_name = params
            def target_fn(p, _pl=placement, _g=grasp_name, _t=tool_from_ee):
                return action_4dof_to_mat4x4(p[_pl]) @ grasp_7dof_to_mat4x4(p[_g]) @ _t
            deps.append(IKDep(q_name, arm, target_fn, movable_spheres_fn(world, plan_skeleton, idx)))
        # action_type in {"retract", None}: no IK target, skip.
    return deps


#: Cost types in the Adam loss: the hard constraints and the soft costs, which the periodic IK projection makes
#: trainable (the covered Confs follow the Poses that Adam moves; grasps stay as sampled).
ADAM_CONSIDER_TYPES = frozenset(("constraint", "cost"))


def _adam_param_groups(params: List[Tuple[torch.Tensor, bool]], conf_lr: float) -> List[dict]:
    """One Adam group per learning rate, non-Confs at ``lr`` then Confs at ``conf_lr``, empty ones dropped:
    Adam launches its foreach kernels once per group, and each element's update is unchanged."""
    non_confs = [p for p, is_conf in params if not is_conf]
    confs = [p for p, is_conf in params if is_conf]
    return [g for g in ({"params": non_confs}, {"params": confs, "lr": conf_lr}) if g["params"]]


@dataclass
class _StepBody:
    """One Adam step's device work, run eagerly or captured as G_step: the forward, the per-step scalars
    stacked into ``packed`` (satisfying count, loss, min cost, then the best satisfying soft cost), and the
    backward with the grad masks. Returns ``(packed, vis)``; ``vis`` is ``None`` unless ``keep_vis``."""
    particles: dict
    rollout_fn: RolloutFunction
    cost_fn: CostFunction
    cost_reducer: CostReducer
    get_mask: Callable[..., torch.Tensor]
    consider_types: AbstractSet[str]
    params: List[torch.Tensor]
    grad_masks: Dict[int, torch.Tensor]
    keep_vis: bool

    def __call__(self) -> Tuple[torch.Tensor, Optional[tuple]]:
        rollout = self.rollout_fn(self.particles)
        cost_dict = self.cost_fn(rollout, self.particles)
        costs = self.cost_reducer(cost_dict, consider_types=self.consider_types)
        satisfying_mask = self.get_mask(cost_dict, verbose=False)
        loss = costs.mean()
        # Every per-step scalar in one tensor, so the loop reads them in one
        # device-to-host copy: each .item() blocks the host until the GPU drains.
        scalars = [satisfying_mask.sum().to(costs.dtype), loss, costs.min()]
        soft_costs = self.cost_reducer.soft_costs(cost_dict)
        scalars.append(soft_costs.masked_fill(~satisfying_mask, float("inf")).min())
        packed = torch.stack(scalars).detach()
        # The backward runs before the loop's bookkeeping, so elapsed[s] and the first-solution times include
        # it. It writes only .grad, which the bookkeeping doesn't read.
        loss.backward()
        apply_grad_masks([{"params": self.params}], self.grad_masks)
        return packed, ((rollout, cost_dict, costs, satisfying_mask) if self.keep_vis else None)

    def clear_autograd(self) -> None:
        """Drop every param's ``.grad``: before a capture, so a deferred one's first warm-up doesn't accumulate
        into the last eager step's grads, and after G_step's release, when they hold its pool."""
        for p in self.params:
            p.grad = None


class PlanContainer(TypedDict):
    plan_skeleton: PlanSkeleton
    particles: Particles
    rollout_fn: RolloutFunction
    cost_fn: CostFunction
    heuristic: float


class _BestSnapshot:
    """Track and restore the best-seen particle state during the Adam loop.

    The satisfying count oscillates after convergence (±~50 particles with
    period ``reik_interval`` — re-IK repairs kinematic drift each refresh),
    so the final step's state is a lottery ticket. Snapshot on every new-best
    count; reset the early-stop patience clock only on *meaningful* progress:
    a count record beating the previous by ``MIN_DELTA``, or the best
    particle's total loss dropping by ``MIN_LOSS_DELTA`` (proxy for ongoing
    pose-class soft-cost descent).
    """

    # ponytail: constants calibrated on one blocks_t1 trace (max record gap 35,
    # sawtooth noise band ~±50); revisit if varying-heights traces disagree.
    MIN_DELTA = 5
    MIN_LOSS_DELTA = 0.05

    def __init__(self):
        self.best_sat = -1
        self.best_step = 0
        self.last_improve_step = 0
        self.state = None
        self._patience_sat = -1
        self._best_min_loss = float("inf")

    def update(self, step: int, num_satisfying: int, min_loss: float, particles: dict):
        if num_satisfying > self.best_sat:
            self.best_sat = num_satisfying
            self.best_step = step
            with torch.no_grad():
                self.state = {k: v.detach().clone() for k, v in particles.items()}
        meaningful = num_satisfying >= self._patience_sat + self.MIN_DELTA
        loss_improved = min_loss < self._best_min_loss - self.MIN_LOSS_DELTA
        if meaningful:
            self._patience_sat = num_satisfying
        if loss_improved:
            self._best_min_loss = min_loss
        if meaningful or loss_improved:
            self.last_improve_step = step

    def should_stop(self, step: int, patience: int) -> bool:
        return patience > 0 and (step - self.last_improve_step) > patience

    def restore_if_better(self, particles: dict, current_num_satisfying: int) -> bool:
        """Copy the best-seen state back into ``particles`` (in place) if it
        beats the current state's satisfying count. Ties keep the current
        state (it has had more soft-cost descent)."""
        if self.state is None or self.best_sat <= current_num_satisfying:
            return False
        with torch.no_grad():
            for k, v in particles.items():
                v.copy_(self.state[k])
        return True


class _QueuedRefresh(NamedTuple):
    """One balance-first refresh's stats and timing events, on the device until ``_ReikDriftGate.flush``."""
    step: int
    q_names: Tuple[str, ...]
    batch: int
    packed: torch.Tensor  # [D, 3]: _ReikDriftGate._pack
    start: torch.cuda.Event
    end: torch.cuda.Event


class _ReikDriftGate:
    """Per-refresh instrumentation for the re-IK subsystem, plus the checker
    tolerances the GN merge gate tests against.

    Records per dep-refresh: solve time and the merge-gate misses, split by pose and CoM. Lands in
    ``opt_metrics["reik_refresh"]``. Refreshes keep their stats on the device until :meth:`flush`.
    """

    def __init__(self, pos_tol: float, rot_tol: float):
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.stats: List[dict] = []
        self._queued: List[_QueuedRefresh] = []  # refreshes awaiting flush()

    def _append_record(self, step, q_name, batch, solve_s, gn_fail, gn_fail_pose, gn_fail_com):
        """Append one ``stats`` record."""
        # solve_s: the refresh's CUDA-event span (a G_refresh replay's also covers the stats packing) split evenly
        # over its deps.
        rec = {"step": step, "conf": q_name, "batch": batch, "solve_s": round(solve_s, 4)}
        # Rows the projection missed, split by which merge-gate criterion failed (a row can fail both).
        rec["gn_fail"] = gn_fail
        rec["gn_fail_pose"] = gn_fail_pose
        rec["gn_fail_com"] = gn_fail_com
        self.stats.append(rec)

    def _pack(self, D: int, batch: int, ok, pose_ok, com_ok) -> torch.Tensor:
        """A balance-first refresh's stats as a ``[D, 3]`` float32 device tensor: per dep, each mask's miss
        count."""
        cols = [(~m).view(D, batch).sum(-1) for m in (ok, pose_ok, com_ok)]
        return torch.stack([c.float() for c in cols], dim=-1)

    def log_batch(self, step: int, q_names: List[str], batch: int, ok, pose_ok, com_ok, start, end) -> None:
        """Queue one balance-first refresh's :meth:`_pack` and timing events for :meth:`flush`: no host read."""
        packed = self._pack(len(q_names), batch, ok, pose_ok, com_ok)
        self._queued.append(_QueuedRefresh(step, tuple(q_names), batch, packed, start, end))

    def flush(self) -> None:
        """Turn the queued refreshes into ``stats`` records, in refresh order, with one event wait and one
        device-to-host copy. ``solve_s`` splits each event time over its deps."""
        if not self._queued:
            return
        self._queued[-1].end.synchronize()  # elapsed_time needs both events done
        values = iter(torch.cat([r.packed.view(-1) for r in self._queued]).tolist())
        for r in self._queued:
            solve_s = r.start.elapsed_time(r.end) / 1e3 / len(r.q_names)  # ms -> s, per dep
            for q_name in r.q_names:
                gn_fail, gn_fail_pose, gn_fail_com = (int(next(values)) for _ in range(3))
                self._append_record(r.step, q_name, r.batch, solve_s, gn_fail, gn_fail_pose, gn_fail_com)
        self._queued.clear()


def _project_and_write_back(
    particles: dict, ik_deps: List[IKDep], world: TAMPWorld, world_from_ee: torch.Tensor, gate: _ReikDriftGate,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GN-project the covered Confs' dep-major stack toward ``world_from_ee`` (``[D*B, 4, 4]``), clear of each
    conf's objects (``IKDep.movables_fn``), gate the result on the ``"gate"`` FK site, and write every row back in
    place. Returns ``(pose_ok, com_ok)``: the gate checks no collision (the stats count pose and CoM misses)."""
    q_names = [dep.q_name for dep in ik_deps]
    batch = world_from_ee.shape[0] // len(ik_deps)
    arms = tuple(dep.arm for dep in ik_deps)
    movables = torch.cat([dep.movables_fn(particles).detach() for dep in ik_deps])
    q = gn_project_balanced(
        world, torch.cat([particles[name].detach() for name in q_names]), world_from_ee, arms, movables=movables,
    )
    pose_ok, com_ok = gn_gate(
        world, q, world_from_ee, arms, gate.pos_tol, gate.rot_tol, kin=world.site_kinematics("gate"),
    )
    with torch.no_grad():
        for name, q_dep in zip(q_names, q.split(batch)):
            particles[name].copy_(q_dep)
            # Dropped from the optimizer, so .grad has no consumer.
            particles[name].grad = None
    return pose_ok, com_ok


def _refresh_balanced(
    particles: dict, ik_deps: List[IKDep], world: TAMPWorld, gate: _ReikDriftGate, step: int,
) -> None:
    """Balance-first refresh: one :func:`gn_project_balanced` call over every dep's covered Conf.

    Every row's result is written back (a persistent lower-level iterate): an unconverged row resumes
    next refresh, and an unreachable one sits at its closest balanced pose, which the kinematic cost
    pulls the placement toward. The :func:`gn_gate` masks only feed the stats. No host syncs
    once gn_refresh's constants are cached, since the loop is launch-bound: CUDA events time it, and
    the stats stay on the device until ``gate.flush()``.
    """
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    q_names = [dep.q_name for dep in ik_deps]
    assert len(set(q_names)) == len(q_names), f"one dep per covered Conf, got {q_names}"
    # Detached: nothing may flow into the Pose / Grasp leaves.
    targets = [dep.target_fn(particles).detach() for dep in ik_deps]
    batch = targets[0].shape[0]
    assert all(t.shape[0] == batch for t in targets), [t.shape[0] for t in targets]

    pose_ok, com_ok = _project_and_write_back(particles, ik_deps, world, torch.cat(targets), gate)
    end.record()

    gate.log_batch(step, q_names, batch, pose_ok & com_ok, pose_ok, com_ok, start, end)


class _StaticRefresh:
    """G_refresh: :func:`_refresh_balanced` captured as a CUDA graph and replayed at each later refresh.

    Static I/O besides the particles: ``slot`` ``[D, 3]``, the packed stats, which :meth:`__call__` queues
    since a replay runs no Python. Captured right after an eager refresh, which sizes the ``"gate"`` site and
    ``world.kinematics_jac`` for the ``D*B`` stack and fills gn_refresh's cached constants (a miss is a
    host-to-device copy, which fails a capture).
    """

    def __init__(self, particles: dict, ik_deps: List[IKDep], world: TAMPWorld, gate: _ReikDriftGate):
        self.particles, self.ik_deps, self.world, self.gate = particles, ik_deps, world, gate
        self.q_names = tuple(dep.q_name for dep in ik_deps)
        self.batch = particles[self.q_names[0]].shape[0]
        self.slot = torch.zeros(len(ik_deps), 3, device=world.device)  # _pack's [D, 3]

        def warmup() -> None:  # on scratch copies: the body writes the confs
            self.body({k: v.detach().clone() for k, v in particles.items()}, self.slot.detach().clone())

        # Guard every Kinematics the body runs; the eager refresh built both.
        self.graph = cuda_graphs.CapturedGraph(
            "G_refresh", lambda: self.body(particles, self.slot), warmup,
            ptrs=lambda: cuda_graphs.graph_ptrs(
                [*particles.values(), self.slot],
                [world.site_kinematics("gate"), world.kinematics_jac],
            ),
        )

    def body(self, particles: dict, slot: torch.Tensor) -> None:
        """:func:`_refresh_balanced`'s device work on static I/O, sync-free so it can be captured."""
        targets = [dep.target_fn(particles).detach() for dep in self.ik_deps]
        pose_ok, com_ok = _project_and_write_back(particles, self.ik_deps, self.world, torch.cat(targets), self.gate)
        slot.copy_(self.gate._pack(len(targets), self.batch, pose_ok & com_ok, pose_ok, com_ok))

    def __call__(self, step: int) -> None:
        """Replay G_refresh as the refresh at ``step`` and queue its stats for ``gate.flush()``."""
        # Events recorded inside a capture are invalid, so these time the replay from outside.
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.graph.replay()
        end.record()
        # The next replay rewrites the slot, so the queue keeps a copy.
        self.gate._queued.append(_QueuedRefresh(step, self.q_names, self.batch, self.slot.clone(), start, end))


class ParticleOptimizer:
    """Attempts to optimize particles for a plan skeleton."""

    def __init__(self, config: TAMPConfiguration, cost_reducer: CostReducer, constraint_checker: ConstraintChecker,
                 planner_busy: Optional[Callable[[], bool]] = None):
        self.config = config
        self.cost_reducer = cost_reducer
        self.constraint_checker = constraint_checker
        self.get_satisfying_mask = self.constraint_checker.get_mask
        # True while the motion-planner build thread runs (None: no thread). Nothing is captured until it
        # returns False: a global-mode capture fails on another thread's CUDA work.
        self.planner_busy = planner_busy

        # Types to optimize
        self.types_to_optimize = {Pose, Conf}
        self.opt_counter = 0

    def _capture_step(self, body: _StepBody, world: TAMPWorld) -> cuda_graphs.CapturedGraph:
        """Capture ``body`` as G_step. Each warm-up drops every ``.grad``, so the captured AccumulateGrad
        gives each param a graph-owned ``.grad`` that every replay rewrites in place (never zero_grad it).
        The guard covers the particles, those grads and the step's FK sites: three, and the retract confs'
        motion site (cost_function.motion_costs)."""
        body.clear_autograd()

        def warmup() -> None:
            body()
            for p in body.params:
                p.grad = None

        return cuda_graphs.CapturedGraph(
            "G_step", body, warmup,
            ptrs=lambda: cuda_graphs.graph_ptrs(
                [*body.particles.values(), *(p.grad for p in body.params)],
                [world.site_kinematics(s) for s in ("rollout", "retract", "com", "retract_motion")],
            ),
        )

    def _visualize_step(self, visualizer: Visualizer, step: int, vis: tuple, num_satisfying: int,
                        loss_val: float, indices: torch.Tensor) -> None:
        """Log the best particle's last conf and object poses, the cost dict and the loss, from the step's ``vis``."""
        rollout, cost_dict, costs, satisfying_mask = vis
        # Visualize satisfying particles if we have any. Otherwise, just show the best particle so far.
        if num_satisfying > 0:
            best_satisfying_idx = costs[satisfying_mask].argmin()
            best_idx = indices[satisfying_mask][best_satisfying_idx]
        else:
            best_idx = costs.argmin()

        # Show last state after rolling out the best particle
        visualizer.set_time_sequence(f"opt_{self.opt_counter}", step)
        q_last = rollout["confs"][best_idx, -1].tolist()

        # Configs are the model's full cspace; T1RerunRobot maps them onto the URDF.
        visualizer.set_joint_positions(q_last)
        for obj in rollout["obj_to_pose"]:
            mat4x4_last = rollout["obj_to_pose"][obj][best_idx, -1]
            visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

        visualizer.log_cost_dict(cost_dict)
        visualizer.log_scalar("loss", loss_val)

    def __call__(self, plan_info: PlanContainer, timer: TorchTimer, visualizer: Visualizer) -> Tuple[bool, dict, bool]:
        """Optimize particles for the plan skeleton.

        Returns ``(solution_found, opt_metrics, time_exceeded)``.
        """
        plan_skeleton = plan_info["plan_skeleton"]
        particles = plan_info["particles"]
        rollout_fn = plan_info["rollout_fn"]
        cost_fn = plan_info["cost_fn"]
        opt_metrics = {
            "plan_skeleton": [str(op) for op in plan_skeleton],
            "num_particles": self.config.num_particles,
        }

        # Create parameter groups for the optimizer
        timer.start("setup_optimizer")
        param_to_type = {}
        for ground_op in plan_skeleton:
            param_names = ground_op.values
            types = [param.type for param in ground_op.operator.parameters]
            assert len(param_names) == len(types)
            for param_name, param_type in zip(param_names, types):
                param_to_type[param_name] = param_type

        # Re-IK-coupled Adam: build the IK dependency map and drop covered
        # Confs from the optimizer (they're refreshed by IK every K steps
        # instead of trained by gradient descent).
        world = rollout_fn.world
        ik_deps: List[IKDep] = _build_ik_deps(plan_skeleton, world)
        dropped_conf_names = {dep.q_name for dep in ik_deps}
        # Refresh instrumentation + the GN merge-gate tolerances, taken from
        # the live checker so the gate tracks whatever the run enforces.
        reik_gate = _ReikDriftGate(
            pos_tol=self.constraint_checker._get_tol(KinematicConstraint.type, "pos_err"),
            rot_tol=self.constraint_checker._get_tol(KinematicConstraint.type, "rot_err"),
        ) if ik_deps else None

        pairs = []  # (tensor, is a Conf) per optimized param, for _adam_param_groups
        # Initial configurations that we don't optimize
        initial_confs = {"q0", "left_q0", "right_q0"}
        for param, val in particles.items():
            # Skip particles that aren't used in this plan (e.g., right_q0 in a left-arm only plan)
            if param not in param_to_type:
                continue
            param_type = param_to_type[param]
            if param_type not in self.types_to_optimize:
                continue
            if param in initial_confs:
                continue  # we don't optimize the initial configuration(s)
            if param in dropped_conf_names:
                # Coupled re-IK: q is determined by IK from current Pose/Grasp,
                # not by Adam, so nothing reads its gradient; skip computing it.
                val.requires_grad = False
                continue
            pairs.append((val, param_type == Conf))
            val.requires_grad = True

        if not pairs:
            raise RuntimeError(f"No parameters to optimize! For plan skeleton: {plan_skeleton}")

        optimizer = Adam(params=_adam_param_groups(pairs, self.config.conf_lr), lr=self.config.lr)

        # Per-conf gradient mask: hard-zero DOFs that should not move during
        # this operator (e.g., base + inactive arm during arm operations).
        # See cutamp/conf_locking.py for the policy.
        grad_masks = build_grad_masks(
            plan_skeleton, particles, param_to_type, Conf, world.model,
        )

        indices = torch.arange(self.config.num_particles, device="cuda")
        consider_types = ADAM_CONSIDER_TYPES
        found_solution = False
        timer.stop("setup_optimizer")

        # Setup some more metrics
        opt_metrics["num_satisfying"] = []
        opt_metrics["loss"] = []
        opt_metrics["best_soft_costs"] = []
        opt_metrics["elapsed"] = []

        vis_every = self.config.opt_viz_interval
        num_steps = self.config.num_opt_steps
        opt_metrics["num_steps"] = num_steps

        # Adam: find satisfying particles and descend the soft
        # costs. Adam runs even when the IK init already satisfies, since the
        # goal is to actively move the Pose particles for the soft cost; the
        # init's satisfying particles are step 0's count.
        with torch.no_grad():
            init_rollout = rollout_fn(particles)
            init_cost_dict = cost_fn(init_rollout, particles)
            init_satisfying = self.get_satisfying_mask(init_cost_dict, verbose=False)
            init_num = init_satisfying.sum().item()
        if init_num > 0:
            # First-solution timing semantics: we already have a solution
            # from IK init, so stamp 0.0 even though Adam is about to run.
            if timer.has_timer("first_solution"):
                timer.stop("first_solution")
            opt_metrics["opt_start_to_first_sol"] = 0.0
            found_solution = True

        pbar = tqdm(range(num_steps))
        start_time = time.perf_counter()
        time_exceeded = False
        best_snapshot = _BestSnapshot()
        body = _StepBody(
            particles, rollout_fn, cost_fn, self.cost_reducer, self.get_satisfying_mask, consider_types,
            params=[p for g in optimizer.param_groups for p in g["params"]], grad_masks=grad_masks,
            # The no-op Visualizer skips the vis block, and with it the block's syncs.
            keep_vis=type(visualizer) is not Visualizer,
        )
        step_graph = refresh_graph = None
        use_graphs = self.config.cuda_graphs
        if use_graphs:
            opt_metrics.update(capture_s=None, capture_step=None, refresh_capture_s=None, refresh_capture_step=None)
        try:
            for step in pbar:
                if (self.config.max_loop_dur is not None
                        and timer.elapsed("start_optimization") >= self.config.max_loop_dur):
                    _log.warning(f"Optimization exceeded max time of {self.config.max_loop_dur:.2f}s")
                    time_exceeded = True
                    break
                timer.start("optimization_step")  # a capture's cost lands in its step
                if use_graphs and step_graph is None and not (self.planner_busy and self.planner_busy()):
                    step_graph = self._capture_step(body, world)  # the planner thread has ended
                    opt_metrics["capture_s"], opt_metrics["capture_step"] = step_graph.capture_s, step
                if step_graph is not None:
                    step_graph.replay()  # never zero_grad after the capture: the graph owns .grad
                    packed, vis = step_graph.outputs
                else:
                    optimizer.zero_grad()
                    packed, vis = body()
                num_satisfying, loss_val, min_cost, best_soft = packed.tolist()
                num_satisfying = int(num_satisfying)
                opt_metrics["num_satisfying"].append(num_satisfying)

                if num_satisfying > 0 and not found_solution:
                    if timer.has_timer("first_solution"):
                        time_to_first_sol = timer.stop("first_solution")
                        _log.info(f"Found first solution in {time_to_first_sol:.2f}s after sampling plans")

                    torch.cuda.synchronize()
                    opt_metrics["opt_start_to_first_sol"] = time.perf_counter() - start_time
                    found_solution = True

                opt_metrics["loss"].append(loss_val)

                # Soft costs are recorded even if we don't optimize them.
                opt_metrics["best_soft_costs"].append(best_soft if num_satisfying > 0 else None)

                torch.cuda.synchronize()
                opt_metrics["elapsed"].append(time.perf_counter() - start_time)

                # Best-seen snapshot + early-stop bookkeeping. Measured state is
                # pre-step, so the clone is consistent with num_satisfying.
                best_snapshot.update(step, num_satisfying, min_cost, particles)
                if best_snapshot.should_stop(step, self.config.early_stop_patience):
                    _log.info(
                        f"Early stop at step {step}: no meaningful progress for "
                        f"{self.config.early_stop_patience} steps (best "
                        f"{best_snapshot.best_sat} satisfying at step {best_snapshot.best_step})"
                    )
                    opt_metrics["early_stopped_at"] = step
                    # Close it: the next skeleton reuses this timer.
                    timer.stop("optimization_step")
                    break

                # Visualize the optimization progress. We do this before stepping the optimizer to see current state.
                if vis is not None and (step % vis_every == 0 or step == num_steps - 1):
                    with timer.time("visualize_opt_rollout"):
                        self._visualize_step(visualizer, step, vis, num_satisfying, loss_val, indices)

                # The body already ran the backward and the grad masks.
                optimizer.step()
                # Counts from 1 so the first refresh lands at step K-1 (after
                # the first batch of pose updates), not step 0.
                if ik_deps and (step + 1) % self.config.reik_interval == 0:
                    if refresh_graph is not None:
                        refresh_graph(step)
                    else:
                        _refresh_balanced(particles, ik_deps, world, reik_gate, step)
                        # G_step exists only once the planner thread has ended for good.
                        if step_graph is not None:
                            refresh_graph = _StaticRefresh(particles, ik_deps, world, reik_gate)
                            opt_metrics["refresh_capture_s"] = refresh_graph.graph.capture_s
                            opt_metrics["refresh_capture_step"] = step
                timer.stop("optimization_step")
                pbar.set_description(
                    f"Loss: {loss_val:.3f}, Min: {min_cost:.3f}, "
                    f"{num_satisfying}/{self.config.num_particles} satisfying"
                )
                vis = None  # no eager step leaves an autograd graph alive for a later capture
        finally:  # also on early stop and on errors: no graph outlives the loop
            if step_graph is not None:
                step_graph.release()
                body.clear_autograd()
            if refresh_graph is not None:
                refresh_graph.graph.release()

        # Restore the best-seen state if the loop ended on a dip of the
        # post-convergence oscillation (final state is otherwise a lottery).
        restored = best_snapshot.restore_if_better(
            particles, opt_metrics["num_satisfying"][-1] if opt_metrics["num_satisfying"] else 0
        )
        if restored:
            _log.info(
                f"Restored best-seen particles: {best_snapshot.best_sat} satisfying "
                f"(step {best_snapshot.best_step}) over final "
                f"{opt_metrics['num_satisfying'][-1]}"
            )
        opt_metrics["best_num_satisfying"] = best_snapshot.best_sat
        # What Adam added over step 0, which measures the untouched IK init (None if no step ran:
        # best_sat stays -1), and whether the reported particles are a restored snapshot.
        opt_metrics["adam_gain"] = (
            best_snapshot.best_sat - opt_metrics["num_satisfying"][0] if best_snapshot.best_sat >= 0 else None
        )
        opt_metrics["restored"] = restored

        # No re-IK after the loop: the satisfaction check below drops any covered conf that drifted off its
        # target in the steps since the last refresh (a final GN IK repair here changed no blocks_t1 plan, 2026-11-06).
        if reik_gate is not None:
            reik_gate.flush()  # the queued refresh records sit on the device until here
        if reik_gate is not None and reik_gate.stats:
            opt_metrics["reik_refresh"] = reik_gate.stats
            total_s = sum(r["solve_s"] for r in reik_gate.stats)
            _log.info(
                f"re-IK refresh: {len(reik_gate.stats)} dep-refreshes, {total_s:.1f}s IK time"
            )

        # Now we've finished the optimization loop. Check the satisfying particles and compute their costs.
        with torch.no_grad():
            rollout = rollout_fn(particles)
            cost_dict = cost_fn(rollout, particles)
            costs = self.cost_reducer(cost_dict, consider_types=consider_types)
            soft_costs = self.cost_reducer.soft_costs(cost_dict)
            satisfying_mask = self.get_satisfying_mask(cost_dict, verbose=True)
        num_satisfying = satisfying_mask.sum().item()
        _log.info(f"{num_satisfying}/{self.config.num_particles} satisfying after optimization")
        opt_metrics["num_satisfying_final"] = num_satisfying

        # Get the best costs from the satisfying particles. Note cost itself includes soft and hard (i.e., constraints)
        # costs, as the Adam loss does (ADAM_CONSIDER_TYPES).
        best_cost = costs[satisfying_mask].min().item() if num_satisfying > 0 else None
        best_soft_cost = soft_costs[satisfying_mask].min().item() if num_satisfying > 0 else None
        if best_cost is not None:
            _log.info(f"Best particle cost: {best_cost:04f}, best soft cost: {best_soft_cost:04f}")
        else:
            _log.info("No satisfying particles found at end of optimization")
        opt_metrics["best_cost"] = best_cost
        opt_metrics["best_soft_cost"] = best_soft_cost

        # Visualize rollout of the best particle in terms of soft costs, which is just trajectory length if no
        # other soft costs are specified. If no satisfying particles, just visualize lowest overall cost particle.
        if num_satisfying > 0:
            best_satisfying_idx = soft_costs[satisfying_mask].argmin()
            best_idx = indices[satisfying_mask][best_satisfying_idx]
        else:
            best_idx = costs.argmin()
        best_idx_int = int(best_idx.item())
        # Measure only: the selected best particle's CoM margin, the min over its confs.
        opt_metrics["best_com_margin"] = particle_com_margin(world, {k: v[best_idx_int] for k, v in particles.items()})

        # Log initial state
        timer.start("visualize_rollout")
        world = rollout_fn.world
        # The robot's conf in the model's cspace, and each hand's gripper state on the rerun robot (when one is
        # shown): a pick conf's hand closed, a place conf's open, the other hand as it was; both open at the start.
        rr_robot = getattr(visualizer, "robot", None)
        grippers = {"left": GRIPPER_OPEN, "right": GRIPPER_OPEN}

        def show(q: torch.Tensor) -> None:
            if hasattr(rr_robot, "set_grippers"):
                rr_robot.set_grippers(**grippers)
            visualizer.set_joint_positions(q.tolist())

        visualizer.set_time_sequence(f"rollout_{self.opt_counter}", 0)
        show(world.q_init)
        for obj in world.movables:
            obj_pose = world.get_object_pose(obj).cpu()
            visualizer.log_mat4x4(f"world/{obj.name}", obj_pose)

        for ts in range(len(rollout["conf_params"])):
            visualizer.set_time_sequence(f"rollout_{self.opt_counter}", ts + 1)
            arm = rollout_fn.conf_to_arm.get(rollout["conf_params"][ts])
            if arm is not None:
                grippers[arm] = GRIPPER_CLOSED if rollout["gripper_close"][ts] else GRIPPER_OPEN
            show(rollout["confs"][best_idx, ts])

            world_from_ee = rollout["world_from_ee"][best_idx, ts].cpu()
            visualizer.log_mat4x4("rollout/ee_pose", world_from_ee)

            robot_spheres = rollout["robot_spheres"][best_idx, ts].cpu()
            visualizer.log_spheres("rollout/robot_spheres", robot_spheres)

            pose_ts = rollout["ts_to_pose_ts"][ts]
            for obj in world.movables:
                obj_pose = rollout["obj_to_pose"][obj.name][best_idx, pose_ts].cpu()
                visualizer.log_mat4x4(f"world/{obj.name}", obj_pose)
        timer.stop("visualize_rollout")

        self.opt_counter += 1
        return num_satisfying > 0, opt_metrics, time_exceeded
