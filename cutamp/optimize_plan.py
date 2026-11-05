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
from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, TypedDict

import torch
from torch.optim import Adam
from tqdm import tqdm

from cutamp import cuda_graphs
from cutamp.conf_locking import apply_grad_masks, build_grad_masks
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
)
from cutamp.config import TAMPConfiguration, uses_cuda_graphs
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.costs import curobo_pose_error
from cutamp.gn_refresh import gn_gate, gn_project, gn_project_balanced, soles_on_stance, tilt_within_bound
from cutamp.particle_initialization import (
    _achieved_pose_errors,
    _ik_for_pose,
    _ik_solution_to_full_q,
    _ik_success_mask,
)
from cutamp.rollout import RolloutFunction
from cutamp.t1_domain import Conf, Pose
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import KinematicConstraint, Motion
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


@dataclass
class IKDep:
    """Re-IK dependency: which Conf to refresh and how to rebuild its IK target.

    ``target_fn(particles)`` rebuilds ``world_from_ee`` from the current
    particle state. ``q_name`` is the Conf to overwrite (in-place) with the
    IK solution. ``arm`` selects the active tool frame, and so which arm is
    inactive (matching ``_ik_for_pose``'s contract).
    """
    q_name: str
    arm: Literal["left", "right"]
    target_fn: Callable[[dict], torch.Tensor]


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
    grasp_to_mat = action_4dof_to_mat4x4
    for ground_op in plan_skeleton:
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
            def target_fn(p, _wo=world_from_obj, _g=grasp_name, _t=tool_from_ee, _gm=grasp_to_mat):
                return _wo @ _gm(p[_g]) @ _t
            deps.append(IKDep(q_name=q_name, arm=arm, target_fn=target_fn))
        elif action_type == "place":
            obj, grasp_name, placement, surface, q_name = params
            def target_fn(p, _pl=placement, _g=grasp_name, _t=tool_from_ee, _gm=grasp_to_mat):
                return action_4dof_to_mat4x4(p[_pl]) @ _gm(p[_g]) @ _t
            deps.append(IKDep(q_name=q_name, arm=arm, target_fn=target_fn))
        # action_type in {"retract", None}: no IK target, skip.
    return deps


def _should_skip_adam(init_num: int, config: TAMPConfiguration) -> bool:
    """Skip Phase 1 only when init already satisfies, re-IK is off, and the
    ablation fairness flag hasn't forced optimization (upstream never
    skipped; skipping is our shortcut for the uncoupled configuration)."""
    return init_num > 0 and not config.coupled_reik and not config.force_adam


def _adam_consider_types(config: TAMPConfiguration) -> set:
    """Cost types in the Adam loss. Soft costs join when coupled re-IK makes
    them trainable ("auto") or unconditionally ("always" — the ablation's
    upstream-faithful attempt; historically destroys 21-DOF configs)."""
    consider = {"constraint"}
    if config.coupled_reik or config.soft_in_adam == "always":
        consider.add("cost")
    return consider


def _adam_param_groups(params: List[Tuple[torch.Tensor, bool]], conf_lr: float) -> List[dict]:
    """One Adam group per learning rate, non-Confs at ``lr`` then Confs at ``conf_lr``, empty ones dropped:
    Adam launches its foreach kernels once per group, and each element's update is unchanged."""
    non_confs = [p for p, is_conf in params if not is_conf]
    confs = [p for p, is_conf in params if is_conf]
    return [g for g in ({"params": non_confs}, {"params": confs, "lr": conf_lr}) if g["params"]]


@dataclass
class _StepBody:
    """One Adam step's device work, run eagerly or captured as G_step: the forward, the per-step scalars
    stacked into ``packed`` (satisfying count, loss, min cost, then the best satisfying soft cost when
    ``with_soft``), and the backward with the grad masks. Returns ``(packed, vis)``; ``vis`` is ``None``
    unless ``keep_vis``."""
    particles: dict
    rollout_fn: RolloutFunction
    cost_fn: CostFunction
    cost_reducer: CostReducer
    get_mask: Callable[..., torch.Tensor]
    consider_types: set
    params: List[torch.Tensor]
    grad_masks: Dict[int, torch.Tensor]
    with_soft: bool
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
        if self.with_soft:
            soft_costs = self.cost_reducer.soft_costs(cost_dict)
            scalars.append(soft_costs.masked_fill(~satisfying_mask, float("inf")).min())
        packed = torch.stack(scalars).detach()
        # The backward runs before the loop's bookkeeping, so elapsed[s] and the first-solution times include
        # it. It writes only .grad, which the bookkeeping doesn't read.
        loss.backward()
        apply_grad_masks([{"params": self.params}], self.grad_masks)
        return packed, ((rollout, cost_dict, costs, satisfying_mask) if self.keep_vis else None)

    def clear_autograd(self) -> None:
        """Drop the CoM cache and every param's ``.grad``: before a capture, so a deferred one's first warm-up
        doesn't accumulate into the last eager step's grads, and after G_step's release, when both hold its pool."""
        self.cost_fn._com_polygon_penalties_cache = None
        for p in self.params:
            p.grad = None


class PlanContainer(TypedDict):
    plan_skeleton: PlanSkeleton
    particles: Particles
    rollout_fn: RolloutFunction
    cost_fn: CostFunction
    heuristic: float


class _BestSnapshot:
    """Track and restore the best-seen particle state during Adam Phase 1.

    The satisfying count oscillates after convergence (±~50 particles with
    period ``reik_interval`` — re-IK repairs kinematic drift each refresh),
    so the final step's state is a lottery ticket. Snapshot on every new-best
    count; reset the early-stop patience clock only on *meaningful* progress:
    a count record beating the previous by ``MIN_DELTA``, or the best
    particle's total loss dropping by ``MIN_LOSS_DELTA`` (proxy for ongoing
    pose-class soft-cost descent, which trains only in Phase 1).
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
    packed: torch.Tensor  # [D, k]: _ReikDriftGate._pack
    start: torch.cuda.Event
    end: torch.cuda.Event


class _ReikDriftGate:
    """Per-refresh instrumentation for the re-IK subsystem, plus the checker
    tolerances the GN merge gate tests against.

    Records per dep-refresh: target drift since the previous refresh
    (``curobo_pose_error`` — the checker's own metric; cuRobo's result
    error fields cross-pair metres/radians on multi-tool-frame robots and
    must not be used), solve time, failure counts, and in GN mode the
    pose/CoM split of merge-gate misses. Lands in
    ``opt_metrics["reik_refresh"]``. Balance-first refreshes keep their
    stats on the device until :meth:`flush`.

    NOTE three removed mechanisms (git history has the code): a
    triangle-inequality drift-SKIP path (cuRobo pads sub-capacity batches so
    per-particle skips save nothing, and the all-512-fresh dep-skip never
    fired in any run), a fallback cooldown (fallbacks are dominated by fresh
    churn misses; withholding repair starved the satisfying pool and the
    early-stop signal), and balance-first mode's stall-gated cuRobo fallback
    (never calling cuRobo during Adam matched or beat it).
    """

    # Drift histogram bucket edges (metres) recorded per refresh.
    _BUCKETS = (5e-4, 1e-3, 2e-3, 5e-3, 1e-2)

    def __init__(self, num_deps: int, pos_tol: float, rot_tol: float, balance: bool = True):
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.balance = balance  # config.enable_com_aware_ik: does the refresh see the CoM
        self.refs = [None] * num_deps  # [B, 4, 4] target at previous refresh; stale once G_refresh is captured
        self.stats: List[dict] = []
        self._queued: List[_QueuedRefresh] = []  # refreshes awaiting flush()

    def measure_drift(self, dep_idx: int, target: torch.Tensor):
        """``(pos_drift, rot_drift)`` vs the previous refresh's targets
        (``None`` on the first refresh); advances the reference."""
        ref = self.refs[dep_idx]
        self.refs[dep_idx] = target.clone()
        if ref is None:
            return None, None
        return curobo_pose_error(ref, target)

    def _drift_stats(self, pos_drift: torch.Tensor, rot_drift: torch.Tensor) -> List[torch.Tensor]:
        """Drift summary over the last dim: pos mean, pos max, rot max, then
        the fraction of pos drift under each ``_BUCKETS`` edge."""
        return [
            pos_drift.mean(-1), pos_drift.amax(-1), rot_drift.amax(-1),
            *((pos_drift < b).float().mean(-1) for b in self._BUCKETS),
        ]

    def _append_record(self, step, q_name, batch, n_fail, solve_s, drift,
                       gn_fail=None, gn_fail_pose=None, gn_fail_com=None, fallback=None):
        """Append one ``stats`` record. ``drift`` is :meth:`_drift_stats` as
        Python floats, or ``None`` on the first refresh."""
        # solve_s: pose-only mode, each dep's wall time; balance-first, the refresh's CUDA-event span (a G_refresh
        # replay's also covers the stats packing) split evenly over its deps. Per-run sums hold in all, but don't
        # compare solve_s across modes.
        rec = {
            "step": step, "conf": q_name, "batch": batch,
            "n_ik_fail": n_fail, "solve_s": round(solve_s, 4),
        }
        if fallback is not None:
            rec["fallback"] = fallback  # GN mode: did this refresh call cuRobo
        if gn_fail is not None:
            # GN refresh: rows the projection missed, split by which merge-gate
            # criterion failed (a row can fail both). "n_ik_fail" counts rows still
            # failing after GN + fallback (balance-first has no fallback, so it equals
            # gn_fail).
            rec["gn_fail"] = gn_fail
            rec["gn_fail_pose"] = gn_fail_pose
            rec["gn_fail_com"] = gn_fail_com
        if drift is not None:
            pos_mean, pos_max, rot_max, *below = drift
            rec["pos_drift_mean"] = pos_mean
            rec["pos_drift_max"] = pos_max
            rec["rot_drift_max"] = rot_max
            rec["frac_drift_below"] = {f"{b * 1e3:g}mm": f for b, f in zip(self._BUCKETS, below)}
        self.stats.append(rec)

    def log_stat(self, step, q_name, batch, n_fail, solve_s, pos_drift, rot_drift,
                 gn_fail=None, gn_fail_pose=None, gn_fail_com=None, fallback=None):
        drift = None if pos_drift is None else [s.item() for s in self._drift_stats(pos_drift, rot_drift)]
        self._append_record(step, q_name, batch, n_fail, solve_s, drift,
                            gn_fail, gn_fail_pose, gn_fail_com, fallback)

    def _pack(self, D: int, batch: int, ok, pose_ok, com_ok, pos_drift, rot_drift) -> torch.Tensor:
        """A balance-first refresh's stats as a ``[D, k]`` float32 device tensor: per dep, each mask's miss
        count, then :meth:`_drift_stats` when there is drift (``k = 3`` or ``11``)."""
        cols = [(~m).view(D, batch).sum(-1) for m in (ok, pose_ok, com_ok)]
        if pos_drift is not None:
            cols += self._drift_stats(pos_drift.view(D, batch), rot_drift.view(D, batch))
        return torch.stack([c.float() for c in cols], dim=-1)

    def log_batch(self, step: int, q_names: List[str], batch: int, ok, pose_ok, com_ok,
                  pos_drift, rot_drift, start, end) -> None:
        """Queue one balance-first refresh's :meth:`_pack` and timing events for :meth:`flush`: no host read."""
        packed = self._pack(len(q_names), batch, ok, pose_ok, com_ok, pos_drift, rot_drift)
        self._queued.append(_QueuedRefresh(step, tuple(q_names), batch, packed, start, end))

    def flush(self) -> None:
        """Turn the queued refreshes into ``stats`` records, in refresh order, with one event wait and one
        device-to-host copy. No cuRobo ran, so ``fallback=False``; ``solve_s`` splits each event time over its deps."""
        if not self._queued:
            return
        self._queued[-1].end.synchronize()  # elapsed_time needs both events done
        values = iter(torch.cat([r.packed.view(-1) for r in self._queued]).tolist())
        for r in self._queued:
            D, k = r.packed.shape
            solve_s = r.start.elapsed_time(r.end) / 1e3 / D  # ms -> s, per dep
            for q_name in r.q_names:
                gn_fail, gn_fail_pose, gn_fail_com, *drift = (next(values) for _ in range(k))
                gn_fail = int(gn_fail)
                self._append_record(r.step, q_name, r.batch, gn_fail, solve_s, drift or None,
                                    gn_fail, int(gn_fail_pose), int(gn_fail_com), fallback=False)
        self._queued.clear()


def _project_and_write_back(
    particles: dict, ik_deps: List[IKDep], world: TAMPWorld, world_from_ee: torch.Tensor, gate: _ReikDriftGate,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GN-project the covered Confs' dep-major stack toward ``world_from_ee`` (``[D*B, 4, 4]``), gate the
    result on the ``"gate"`` FK site, and write every row back in place. Returns ``(pose_ok, com_ok)``."""
    q_names = [dep.q_name for dep in ik_deps]
    batch = world_from_ee.shape[0] // len(ik_deps)
    arms = tuple(dep.arm for dep in ik_deps)
    q = gn_project_balanced(
        world, torch.cat([particles[name].detach() for name in q_names]), world_from_ee, arms,
        balance=gate.balance,
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
    pulls the placement toward. No cuRobo runs (see :class:`_ReikDriftGate`); the final pre-check
    refresh is the only cuRobo repair, and the :func:`gn_gate` masks only feed the stats. No host syncs
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
    pos_drifts, rot_drifts = zip(*(gate.measure_drift(i, t) for i, t in enumerate(targets)))
    first_refresh = pos_drifts[0] is None

    pose_ok, com_ok = _project_and_write_back(particles, ik_deps, world, torch.cat(targets), gate)
    end.record()

    gate.log_batch(
        step, q_names, batch, pose_ok & com_ok, pose_ok, com_ok,
        None if first_refresh else torch.cat(pos_drifts),
        None if first_refresh else torch.cat(rot_drifts),
        start, end,
    )


class _StaticRefresh:
    """G_refresh: :func:`_refresh_balanced` captured as a CUDA graph and replayed at each later refresh.

    Static I/O besides the particles: ``refs`` ``[D, B, 4, 4]``, advanced in place because a graph keeps
    reading the tensor ``measure_drift`` would rebind, and ``slot`` ``[D, 11]``, the packed stats, which
    :meth:`__call__` queues since a replay runs no Python. Captured right after an eager refresh, which
    sizes the ``"gate"`` site and ``world.kinematics_jac`` for the ``D*B`` stack, fills gn_refresh's
    cached constants (a miss is a host-to-device copy, which fails a capture) and sets ``gate.refs``.
    """

    def __init__(self, particles: dict, ik_deps: List[IKDep], world: TAMPWorld, gate: _ReikDriftGate):
        assert all(r is not None for r in gate.refs), "G_refresh is captured after an eager refresh sets gate.refs"
        self.particles, self.ik_deps, self.world, self.gate = particles, ik_deps, world, gate
        self.refs = torch.stack(gate.refs)
        self.slot = torch.zeros(len(ik_deps), 3 + 3 + len(gate._BUCKETS), device=self.refs.device)  # _pack's [D, 11]
        self.q_names = tuple(dep.q_name for dep in ik_deps)
        self.batch = self.refs.shape[1]

        def warmup() -> None:  # on scratch copies: the body writes the confs and the refs
            self.body({k: v.detach().clone() for k, v in particles.items()}, self.refs.detach().clone(),
                      self.slot.detach().clone())

        # Guard every Kinematics the body runs; the eager refresh built both.
        self.graph = cuda_graphs.CapturedGraph(
            "G_refresh", lambda: self.body(particles, self.refs, self.slot), warmup,
            ptrs=lambda: cuda_graphs.graph_ptrs(
                [*particles.values(), self.refs, self.slot],
                [world.site_kinematics("gate"), world.kinematics_jac],
            ),
        )

    def body(self, particles: dict, refs: torch.Tensor, slot: torch.Tensor) -> None:
        """:func:`_refresh_balanced`'s device work on static I/O, sync-free so it can be captured."""
        targets = [dep.target_fn(particles).detach() for dep in self.ik_deps]
        drifts = [curobo_pose_error(refs[i], t) for i, t in enumerate(targets)]  # measure_drift's metric
        for i, t in enumerate(targets):
            refs[i].copy_(t)
        pos_drifts, rot_drifts = zip(*drifts)
        pose_ok, com_ok = _project_and_write_back(particles, self.ik_deps, self.world, torch.cat(targets), self.gate)
        slot.copy_(self.gate._pack(len(targets), self.batch, pose_ok & com_ok, pose_ok, com_ok,
                                   torch.cat(pos_drifts), torch.cat(rot_drifts)))

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

    def _refresh_ik_deps(
        self,
        particles: dict,
        ik_deps: List[IKDep],
        world: TAMPWorld,
        gate: "_ReikDriftGate" = None,
        step: int = None,
    ) -> None:
        """Re-solve IK for each ``IKDep`` and write the result into the
        corresponding Conf particle in-place. Failed particles keep their
        previous q (per-particle masked write), except in a balance-first
        gated refresh, which writes every row back.

        When ``gate`` is given, refreshes run GN first, with a cuRobo fallback
        on ``world.ik_refresh_solver`` (a cheap warm-seeded low-seed solver when
        ``reik_num_seeds > 0``), and record per-dep drift/timing stats; in
        balance-first mode a gated refresh is :func:`_refresh_balanced` instead.
        ``gate=None`` repairs the rows that miss checker tolerance or the CoM
        polygon on the full-strength main solver — used for the final
        pre-check refresh.

        cuRobo's IK uses autograd internally (LM Jacobian via ``cost.backward``)
        so the IK call cannot run under ``torch.no_grad``; only the in-place
        ``copy_`` is wrapped. Covered Confs are out of the optimizer, so
        neither Adam nor ``apply_grad_masks`` touches them.
        """
        if not ik_deps:
            return
        if gate is not None and self.config.reik_balance_first:
            return _refresh_balanced(particles, ik_deps, world, gate, step)
        for i, dep in enumerate(ik_deps):
            # Detach the target so IK's backward doesn't try to flow into
            # Pose / Grasp particles (which are leaves in our autograd graph).
            world_from_ee = dep.target_fn(particles).detach()
            batch = world_from_ee.shape[0]
            if gate is not None:
                pos_drift, rot_drift = gate.measure_drift(i, world_from_ee)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n_gn_fail = n_gn_fail_pose = n_gn_fail_com = fallback = None
            if gate is not None:
                from cutamp.com_polygon_cost import compute_com_polygon_mask
                # Gauss-Newton projection from the warm q (see gn_refresh.py).
                # Merge only rows that converged to checker tolerance
                # (checker metric via _achieved_pose_errors) AND stayed
                # CoM-in-polygon. The CoM term is load-bearing: pose-only GN
                # has no CoM residual and the fallback is the only CoM-aware
                # channel during Adam (covered Confs are out of the optimizer),
                # so a pose-only gate would lock in confs that drift out of the
                # support polygon.
                q_gn = gn_project(
                    world, particles[dep.q_name].detach(), world_from_ee, dep.arm,
                )
                with torch.no_grad():
                    pos_err, rot_err = _achieved_pose_errors(
                        world, q_gn, world_from_ee, dep.arm,
                    )
                    pose_ok = (pos_err < gate.pos_tol) & (rot_err < gate.rot_tol)
                    com_ok = compute_com_polygon_mask(world, q_gn)
                    # No-CoM ablation: gate on pose alone (com_ok still feeds the stats).
                    ok = pose_ok & com_ok if self.config.enable_com_aware_ik else pose_ok
                    merged = torch.where(ok.view(-1, 1), q_gn, particles[dep.q_name])
                    particles[dep.q_name].copy_(merged)
                    particles[dep.q_name].grad = None
                n_gn_fail = int((~ok).sum().item())
                n_gn_fail_pose = int((~pose_ok).sum().item())
                n_gn_fail_com = int((~com_ok).sum().item())
                n_fail = n_gn_fail
                # Rows offered to cuRobo: every GN miss.
                repair = ~ok if n_gn_fail > 0 else None
                fallback = repair is not None
                if fallback:
                    # Full-batch call (cuRobo pads sub-capacity batches to
                    # max_batch_size, so subsets save nothing); only offered
                    # rows that cuRobo solved take its result.
                    cheap = world.ik_refresh_solver is not world.ik_solver
                    ik_result = _ik_for_pose(
                        world, world_from_ee, dep.arm,
                        current_state_q=particles[dep.q_name].detach(),
                        ik=world.ik_refresh_solver if cheap else None,
                        seed_from_current=cheap,
                    )
                    new_q = _ik_solution_to_full_q(ik_result, world).detach()
                    take = repair & _ik_success_mask(ik_result)
                    with torch.no_grad():
                        merged = torch.where(take.view(-1, 1), new_q, particles[dep.q_name])
                        particles[dep.q_name].copy_(merged)
                        particles[dep.q_name].grad = None
                    n_fail = int((~(ok | take)).sum().item())
            else:
                # Final pre-check refresh: REPAIR-ONLY, on the full-strength
                # main solver. Measure which rows actually fail the checker's
                # kinematic tolerances (the hand's, and on a model with soles
                # the soles' against the stance) or the CoM gate, and only
                # overwrite those — a blanket success-masked overwrite perturbs
                # already-good rows' soft costs (measured best_soft 18.6 -> 22.8
                # on one run). Full batch always: cuRobo pads sub-capacity
                # batches internally, so subsets save nothing.
                from cutamp.com_polygon_cost import _fk_state, compute_com_polygon_mask
                pos_tol = self.constraint_checker._get_tol(KinematicConstraint.type, "pos_err")
                rot_tol = self.constraint_checker._get_tol(KinematicConstraint.type, "rot_err")
                with torch.no_grad():
                    pos_err, rot_err = _achieved_pose_errors(
                        world, particles[dep.q_name].detach(), world_from_ee, dep.arm,
                    )
                    ok_now = (pos_err < pos_tol) & (rot_err < rot_tol)
                    if world.model.sole_frames:
                        ok_now &= soles_on_stance(
                            world, _fk_state(world, particles[dep.q_name].detach()),
                            self.constraint_checker._get_tol(KinematicConstraint.type, "sole_pos_err"),
                            self.constraint_checker._get_tol(KinematicConstraint.type, "sole_rot_err"),
                        )
                    if self.config.enable_com_aware_ik:  # off in the no-CoM ablation
                        ok_now &= compute_com_polygon_mask(world, particles[dep.q_name])
                if bool(ok_now.all()):
                    continue  # nothing to repair for this dep
                ik_result = _ik_for_pose(
                    world, world_from_ee, dep.arm,
                    current_state_q=particles[dep.q_name].detach(),
                )
                new_q = _ik_solution_to_full_q(ik_result, world).detach()
                success = _ik_success_mask(ik_result)
                n_fail = int((~success).sum().item())
                # Repair-only merge: already-good rows keep their q.
                success = success & ~ok_now
                if world.model.envelope is not None and world.model.envelope.trunk_tilt_max is not None:
                    # cuRobo's IK does not see an envelope's tilt bound, which GN holds: on blocks_t1's floor scene,
                    # under a 1.0 rad bound, it bowed 9 of 11 repaired rows past it, each trading a hand, sole or CoM
                    # failure for a tilt one, so a repair past the bound is not taken.
                    success &= tilt_within_bound(world, new_q,
                                                 self.constraint_checker._get_tol(Motion.type, "trunk_tilt"))
                with torch.no_grad():
                    merged = torch.where(success.view(-1, 1), new_q, particles[dep.q_name])
                    particles[dep.q_name].copy_(merged)
                    # Dropped from the optimizer, so .grad has no consumer.
                    particles[dep.q_name].grad = None
            torch.cuda.synchronize()
            if gate is not None:
                gate.log_stat(
                    step, dep.q_name, batch, n_fail,
                    time.perf_counter() - t0, pos_drift, rot_drift,
                    gn_fail=n_gn_fail, gn_fail_pose=n_gn_fail_pose, gn_fail_com=n_gn_fail_com,
                    fallback=fallback,
                )

    def _capture_step(self, body: _StepBody, world: TAMPWorld) -> cuda_graphs.CapturedGraph:
        """Capture ``body`` as G_step. Each warm-up drops every ``.grad``, so the captured AccumulateGrad
        gives each param a graph-owned ``.grad`` that every replay rewrites in place (never zero_grad it).
        The guard covers the particles, those grads and the step's FK sites: three, and on the real legs the
        retract confs' motion site (cost_function.motion_costs)."""
        body.clear_autograd()

        def warmup() -> None:
            body()
            for p in body.params:
                p.grad = None

        return cuda_graphs.CapturedGraph(
            "G_step", body, warmup,
            ptrs=lambda: cuda_graphs.graph_ptrs(
                [*body.particles.values(), *(p.grad for p in body.params)],
                [world.site_kinematics(s) for s in ("rollout", "retract", "com")
                 + (("retract_motion",) if world.model.sole_frames else ())],
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

        # v0.8: configs are full 21-DOF (T1) or single chain DOF (others).
        # T1RerunRobot maps 21 → 28 URDF actuated joints internally.
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
            "opt_params": {},
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
        ik_deps: List[IKDep] = (
            _build_ik_deps(plan_skeleton, world)
            if self.config.coupled_reik else []
        )
        dropped_conf_names = {dep.q_name for dep in ik_deps}
        # Refresh instrumentation + the GN merge-gate tolerances, taken from
        # the live checker so the gate tracks whatever the run enforces.
        reik_gate = _ReikDriftGate(
            len(ik_deps),
            pos_tol=self.constraint_checker._get_tol(KinematicConstraint.type, "pos_err"),
            rot_tol=self.constraint_checker._get_tol(KinematicConstraint.type, "rot_err"),
            balance=self.config.enable_com_aware_ik,
        ) if ik_deps else None

        pairs = []  # (tensor, is a Conf) per optimized param, for _adam_param_groups
        param_msg = []
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
            param_msg.append(f"(param: {param} = {tuple(val.shape)})")
            pairs.append((val, param_type == Conf))
            opt_metrics["opt_params"][param] = {"type": param_type, "shape": list(val.shape)}
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
        num_total_params = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
        opt_metrics["optimizer"] = {
            "optimizer_cls": optimizer.__class__.__name__,
            "lr": self.config.lr,
            "conf_lr": self.config.conf_lr,
            "num_params": num_total_params,
        }
        _log.debug(f"Optimizing {num_total_params} parameters, {', '.join(param_msg)}")
        total_dof = sum(p.shape[-1] for group in optimizer.param_groups for p in group["params"])
        _log.debug(f"Total DOF for each particle: {total_dof}")

        indices = torch.arange(self.config.num_particles, device="cuda")
        consider_types = _adam_consider_types(self.config)
        found_solution = False
        timer.stop("setup_optimizer")

        # Setup some more metrics
        opt_metrics["num_satisfying"] = []
        opt_metrics["loss"] = []
        if self.config.soft_cost is not None:
            opt_metrics["best_soft_costs"] = []
        opt_metrics["elapsed"] = []

        vis_every = self.config.opt_viz_interval
        num_steps = self.config.num_opt_steps
        opt_metrics["num_steps"] = num_steps

        # Phase 1 (Adam): find satisfying particles by minimizing the
        # hard-constraint penalty. Skip Adam entirely if the IK-init
        # already produced satisfying particles — Adam's bias-corrected
        # first step has magnitude ``lr`` per non-zero-gradient DOF,
        # independent of gradient size, which destroys near-optimal
        # initializations.
        # When coupled_reik is on, run Adam even if init satisfied, since
        # the goal is to actively move Pose particles for the soft cost.
        # --force_adam runs it without re-IK too; there it loses the init's
        # satisfying particles within a few steps, and the best-seen restore
        # below hands the init back (recorded as adam_gain and restored).
        with torch.no_grad():
            init_rollout = rollout_fn(particles)
            init_cost_dict = cost_fn(init_rollout, particles)
            init_satisfying = self.get_satisfying_mask(init_cost_dict, verbose=False)
            init_num = init_satisfying.sum().item()
        skip_adam = _should_skip_adam(init_num, self.config)
        if skip_adam:
            _log.info(
                f"IK-init already produced {init_num}/{self.config.num_particles} satisfying "
                f"particles; skipping Adam Phase 1."
            )
            if timer.has_timer("first_solution"):
                timer.stop("first_solution")
            opt_metrics["num_satisfying"].append(init_num)
            opt_metrics["loss"].append(
                self.cost_reducer(init_cost_dict, consider_types=consider_types).mean().item()
            )
            opt_metrics["elapsed"].append(0.0)
            opt_metrics["opt_start_to_first_sol"] = 0.0
        elif init_num > 0 and self.config.coupled_reik:
            # First-solution timing semantics: we already have a solution
            # from IK init, so stamp 0.0 even though Adam is about to run.
            if timer.has_timer("first_solution"):
                timer.stop("first_solution")
            opt_metrics["opt_start_to_first_sol"] = 0.0
            found_solution = True

        # Phase 1 (Adam): only run when IK init didn't already satisfy.
        pbar = tqdm(range(num_steps)) if not skip_adam else []
        start_time = time.perf_counter()
        time_exceeded = False
        best_snapshot = _BestSnapshot()
        body = _StepBody(
            particles, rollout_fn, cost_fn, self.cost_reducer, self.get_satisfying_mask, consider_types,
            params=[p for g in optimizer.param_groups for p in g["params"]], grad_masks=grad_masks,
            with_soft=self.config.soft_cost is not None,
            # The no-op Visualizer skips the vis block, and with it the block's syncs.
            keep_vis=type(visualizer) is not Visualizer,
        )
        step_graph = refresh_graph = None
        use_graphs = uses_cuda_graphs(self.config)
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
                num_satisfying, loss_val, min_cost, *best_soft = packed.tolist()
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
                if self.config.soft_cost is not None:
                    opt_metrics["best_soft_costs"].append(best_soft[0] if num_satisfying > 0 else None)

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
                        self._refresh_ik_deps(particles, ik_deps, world, gate=reik_gate, step=step)
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
        restored = not skip_adam and best_snapshot.restore_if_better(
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
        # best_sat stays -1), and whether the reported particles are a restored snapshot. Without
        # re-IK, Adam empties the satisfying set at every lr tried, so those arms report gain 0
        # with a restore: the IK init (known-bugs B13).
        opt_metrics["adam_gain"] = (
            best_snapshot.best_sat - opt_metrics["num_satisfying"][0] if best_snapshot.best_sat >= 0 else None
        )
        opt_metrics["restored"] = restored

        # Final re-IK before downstream satisfaction checks: Adam may have
        # moved poses since the last periodic refresh. Always FULL (no gate):
        # restore_if_better may have rewritten particles, making gate refs stale.
        if ik_deps:
            with timer.time("final_repair"):  # a stage time of the speed gate (design section 8)
                self._refresh_ik_deps(particles, ik_deps, world)
                reik_gate.flush()  # balance-first records sit on the device until here
        if reik_gate is not None and reik_gate.stats:
            opt_metrics["reik_refresh"] = reik_gate.stats
            total_s = sum(r["solve_s"] for r in reik_gate.stats)
            _log.info(
                f"re-IK refresh: {len(reik_gate.stats)} dep-refreshes, {total_s:.1f}s IK time"
            )

        # Now we've finished the optimization loop. Check the satisfying particles and compute soft/hard costs.
        with torch.no_grad():
            rollout = rollout_fn(particles)
            cost_dict = cost_fn(rollout, particles)
            costs = self.cost_reducer(cost_dict, consider_types=consider_types)
            soft_costs = self.cost_reducer.soft_costs(cost_dict)
            hard_costs = self.cost_reducer.hard_costs(cost_dict)
            satisfying_mask = self.get_satisfying_mask(cost_dict, verbose=True)
        num_satisfying = satisfying_mask.sum().item()
        _log.info(f"{num_satisfying}/{self.config.num_particles} satisfying after optimization")
        opt_metrics["num_satisfying_final"] = num_satisfying

        # Get the best costs from the satisfying particles. Note cost itself includes soft and hard (i.e., constraints)
        # costs when the soft costs join the Adam loss (_adam_consider_types), and only hard costs otherwise.
        best_cost = costs[satisfying_mask].min().item() if num_satisfying > 0 else None
        best_soft_cost = soft_costs[satisfying_mask].min().item() if num_satisfying > 0 else None
        best_hard_cost = hard_costs[satisfying_mask].min().item() if num_satisfying > 0 else None
        if best_cost is not None:
            _log.info(
                f"Best particle cost: {best_cost:04f}, best soft cost: {best_soft_cost:04f}, "
                f"best hard cost: {best_hard_cost:04f}"
            )
        else:
            _log.info("No satisfying particles found at end of optimization")
        opt_metrics["best_cost"] = best_cost
        opt_metrics["best_soft_cost"] = best_soft_cost
        opt_metrics["best_hard_cost"] = best_hard_cost

        # Visualize rollout of the best particle in terms of soft costs, which is just trajectory length if no
        # other soft costs are specified. If no satisfying particles, just visualize lowest overall cost particle.
        if num_satisfying > 0:
            best_satisfying_idx = soft_costs[satisfying_mask].argmin()
            best_idx = indices[satisfying_mask][best_satisfying_idx]
        else:
            best_idx = costs.argmin()
        best_idx_int = int(best_idx.item())

        # Post-optimization COM-polygon visibility: report per-conf how many
        # final particles' configurations are inside the foot-realistic
        # support polygon. Useful diagnostic — COM is a soft cost (not in
        # the constraint_checker output) so without this you'd only see the
        # aggregate soft cost number.
        try:
            from cutamp.com_polygon_cost import compute_com_polygon_mask
            conf_names = sorted(
                p for p in particles
                if p.startswith(("left_q", "right_q", "q")) and particles[p].ndim == 2
            )
            if conf_names:
                conf_to_mask = {
                    cn: compute_com_polygon_mask(rollout_fn.world, particles[cn])
                    for cn in conf_names
                }
                com_summary = [
                    f"{cn} {int(mask.sum().item())}/{mask.shape[0]}"
                    for cn, mask in conf_to_mask.items()
                ]
                _log.info("COM-in-polygon (per conf): " + ", ".join(com_summary))
                opt_metrics["com_in_polygon_per_conf"] = {
                    cn: int(mask.sum().item()) for cn, mask in conf_to_mask.items()
                }

                # Ablation yardstick (measure-only, recorded in EVERY arm):
                # a particle is CoM-feasible iff ALL its confs are in-polygon.
                all_conf_ok = torch.stack(list(conf_to_mask.values()), dim=0).all(dim=0)
                sat_and_com = satisfying_mask & all_conf_ok
                opt_metrics["num_satisfying_com_feasible"] = int(sat_and_com.sum().item())
                soft_np = soft_costs[sat_and_com]
                opt_metrics["best_soft_cost_com_feasible"] = (
                    soft_np.min().item() if sat_and_com.any() else None
                )
                # Margin of the selected best particle: min over its confs.
                from cutamp.com_polygon_cost import compute_com_margin
                margins = [
                    compute_com_margin(rollout_fn.world, particles[cn][best_idx_int : best_idx_int + 1]).item()
                    for cn in conf_names
                ]
                opt_metrics["best_com_margin"] = min(margins) if margins else None
        except Exception as e:
            _log.warning(f"COM-in-polygon summary skipped: {e}")

        # Log initial state
        timer.start("visualize_rollout")
        world = rollout_fn.world
        visualizer.set_time_sequence(f"rollout_{self.opt_counter}", 0)
        visualizer.set_joint_positions(world.q_init.tolist())
        for obj in world.movables:
            obj_pose = world.get_object_pose(obj).cpu()
            visualizer.log_mat4x4(f"world/{obj.name}", obj_pose)

        for ts in range(len(rollout["conf_params"])):
            visualizer.set_time_sequence(f"rollout_{self.opt_counter}", ts + 1)
            q = rollout["confs"][best_idx, ts]

            gripper_close = rollout["gripper_close"][ts]
            # T1 gripper: 4-bar parallel linkage, 4 joints per hand.
            # GRIPPER_OPEN = (0,0,0,0), GRIPPER_CLOSED = (1,-1,-1,1).
            gripper_joints = [1.0, -1.0, -1.0, 1.0] if gripper_close else [0.0, 0.0, 0.0, 0.0]
            visualizer.set_joint_positions(q.tolist() + gripper_joints)

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
