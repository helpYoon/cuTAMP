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
from typing import Callable, List, Literal, Tuple, TypedDict

import torch
from torch.optim import Adam
from tqdm import tqdm

from cutamp.conf_locking import apply_grad_masks, build_grad_masks
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
)
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.costs import curobo_pose_error
from cutamp.gn_refresh import gn_project
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
from cutamp.task_planning.constraints import KinematicConstraint
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


@dataclass
class IKDep:
    """Re-IK dependency: which Conf to refresh and how to rebuild its IK target.

    ``target_fn(particles)`` rebuilds ``world_from_ee`` from the current
    particle state. ``q_name`` is the Conf to overwrite (in-place) with the
    IK solution. ``arm`` selects the active tool frame and the inactive-arm
    cspace pin (matching ``_ik_for_pose``'s contract).
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


class _ReikDriftGate:
    """Per-refresh instrumentation for the re-IK subsystem, plus the checker
    tolerances the GN merge gate tests against.

    Records per dep-refresh: target drift since the previous refresh
    (``curobo_pose_error`` — the checker's own metric; cuRobo's result
    error fields cross-pair metres/radians on multi-tool-frame robots and
    must not be used), solve wall time, failure counts, and in GN mode the
    pose/CoM split of merge-gate misses. Lands in
    ``opt_metrics["reik_refresh"]``.

    NOTE two removed mechanisms, both measured dead 2026-08-04 (git history
    has the code): a triangle-inequality drift-SKIP path (cuRobo pads
    sub-capacity batches so per-particle skips save nothing, and the
    all-512-fresh dep-skip never fired in any run) and a fallback cooldown
    (fallbacks are dominated by fresh churn misses; withholding repair
    starved the satisfying pool and the early-stop signal).
    """

    # Drift histogram bucket edges (metres) recorded per refresh.
    _BUCKETS = (5e-4, 1e-3, 2e-3, 5e-3, 1e-2)

    def __init__(self, num_deps: int, pos_tol: float, rot_tol: float):
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.refs = [None] * num_deps  # [B, 4, 4] target at previous refresh
        self.stats: List[dict] = []

    def measure_drift(self, dep_idx: int, target: torch.Tensor):
        """``(pos_drift, rot_drift)`` vs the previous refresh's targets
        (``None`` on the first refresh); advances the reference."""
        ref = self.refs[dep_idx]
        self.refs[dep_idx] = target.clone()
        if ref is None:
            return None, None
        return curobo_pose_error(ref, target)

    def log_stat(self, step, q_name, batch, n_fail, solve_s, pos_drift, rot_drift,
                 gn_fail=None, gn_fail_pose=None, gn_fail_com=None):
        rec = {
            "step": step, "conf": q_name, "batch": batch,
            "n_ik_fail": n_fail, "solve_s": round(solve_s, 4),
        }
        if gn_fail is not None:
            # GN refresh: rows the projection missed (they went to the cuRobo
            # fallback), split by which merge-gate criterion failed (a row can
            # fail both). NOTE cross-mode stats caveat: in GN mode "n_ik_fail"
            # counts rows still failing after GN + fallback; in cuRobo mode it
            # counts solver success-flag failures. Presence of "gn_fail" marks
            # which semantic a record uses.
            rec["gn_fail"] = gn_fail
            rec["gn_fail_pose"] = gn_fail_pose
            rec["gn_fail_com"] = gn_fail_com
        if pos_drift is not None:
            rec["pos_drift_mean"] = pos_drift.mean().item()
            rec["pos_drift_max"] = pos_drift.max().item()
            rec["rot_drift_max"] = rot_drift.max().item()
            rec["frac_drift_below"] = {
                f"{b * 1e3:g}mm": (pos_drift < b).float().mean().item() for b in self._BUCKETS
            }
        self.stats.append(rec)


class ParticleOptimizer:
    """Attempts to optimize particles for a plan skeleton."""

    def __init__(self, config: TAMPConfiguration, cost_reducer: CostReducer, constraint_checker: ConstraintChecker):
        self.config = config
        self.cost_reducer = cost_reducer
        self.constraint_checker = constraint_checker
        self.get_satisfying_mask = self.constraint_checker.get_mask

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
        previous q (per-particle masked write).

        When ``gate`` is given, refreshes run GN-first on
        ``world.ik_refresh_solver`` (a cheap warm-seeded low-seed solver when
        ``reik_num_seeds > 0``) and record per-dep drift/timing stats.
        ``gate=None`` refreshes everything on the full-strength main solver —
        used for the final pre-check refresh.

        cuRobo's IK uses autograd internally (LM Jacobian via ``cost.backward``)
        so the IK call cannot run under ``torch.no_grad``; only the in-place
        ``copy_`` is wrapped. The ``copy_`` preserves tensor identity, which
        ``grad_masks`` and Adam state both depend on.
        """
        if not ik_deps:
            return
        for i, dep in enumerate(ik_deps):
            # Detach the target so IK's backward doesn't try to flow into
            # Pose / Grasp particles (which are leaves in our autograd graph).
            world_from_ee = dep.target_fn(particles).detach()
            batch = world_from_ee.shape[0]
            if gate is not None:
                pos_drift, rot_drift = gate.measure_drift(i, world_from_ee)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n_gn_fail = n_gn_fail_pose = n_gn_fail_com = None
            if gate is not None and self.config.reik_gn:
                from cutamp.com_polygon_cost import compute_com_polygon_mask
                # Gauss-Newton projection from the warm q (see gn_refresh.py);
                # merge only rows that converged to checker tolerance (checker
                # metric via _achieved_pose_errors) AND stayed CoM-in-polygon.
                # The CoM term is load-bearing: GN has no CoM residual and the
                # fallback is the only CoM-aware channel during Adam (covered
                # Confs are out of the optimizer), so a pose-only gate would
                # lock in confs that drift out of the support polygon.
                q_gn = gn_project(
                    world, particles[dep.q_name].detach(), world_from_ee, dep.arm,
                )
                with torch.no_grad():
                    pos_err, rot_err = _achieved_pose_errors(
                        world, q_gn, world_from_ee, dep.arm,
                    )
                    pose_ok = (pos_err < gate.pos_tol) & (rot_err < gate.rot_tol)
                    com_ok = compute_com_polygon_mask(world, q_gn)
                    ok = pose_ok & com_ok
                    merged = torch.where(ok.view(-1, 1), q_gn, particles[dep.q_name])
                    particles[dep.q_name].copy_(merged)
                    particles[dep.q_name].grad = None
                n_gn_fail = int((~ok).sum().item())
                n_gn_fail_pose = int((~pose_ok).sum().item())
                n_gn_fail_com = int((~com_ok).sum().item())
                n_fail = n_gn_fail
                if n_gn_fail > 0:
                    # cuRobo fallback for the rows GN missed. Full-batch call
                    # (cuRobo pads sub-capacity batches to max_batch_size, so
                    # subsets save nothing); only GN-missed rows that cuRobo
                    # solved take its result.
                    cheap = world.ik_refresh_solver is not world.ik_solver
                    ik_result = _ik_for_pose(
                        world, world_from_ee, dep.arm,
                        current_state_q=particles[dep.q_name].detach(),
                        ik=world.ik_refresh_solver if cheap else None,
                        seed_from_current=cheap,
                    )
                    new_q = _ik_solution_to_full_q(ik_result, world).detach()
                    take = (~ok) & _ik_success_mask(ik_result)
                    with torch.no_grad():
                        merged = torch.where(take.view(-1, 1), new_q, particles[dep.q_name])
                        particles[dep.q_name].copy_(merged)
                        particles[dep.q_name].grad = None
                    n_fail = int((~(ok | take)).sum().item())
            else:
                # cuRobo solve. Low-seed refresh solver + warm seed only when
                # gated; with reik_num_seeds=0 this path is byte-identical to
                # the ungated one (ik_refresh_solver falls back to ik_solver).
                # Full batch always: cuRobo pads sub-capacity batches
                # internally, so subsets save nothing.
                ok_now = None
                if gate is None:
                    # Final pre-check refresh: REPAIR-ONLY. Measure which rows
                    # actually fail the checker's kinematic tolerance or the
                    # CoM gate and only overwrite those — a blanket
                    # success-masked overwrite perturbs already-good rows'
                    # soft costs (measured best_soft 18.6 -> 22.8 on one run).
                    from cutamp.com_polygon_cost import compute_com_polygon_mask
                    pos_tol = self.constraint_checker._get_tol(KinematicConstraint.type, "pos_err")
                    rot_tol = self.constraint_checker._get_tol(KinematicConstraint.type, "rot_err")
                    with torch.no_grad():
                        pos_err, rot_err = _achieved_pose_errors(
                            world, particles[dep.q_name].detach(), world_from_ee, dep.arm,
                        )
                        ok_now = (
                            (pos_err < pos_tol)
                            & (rot_err < rot_tol)
                            & compute_com_polygon_mask(world, particles[dep.q_name])
                        )
                    if bool(ok_now.all()):
                        continue  # nothing to repair for this dep
                cheap = gate is not None and world.ik_refresh_solver is not world.ik_solver
                ik_result = _ik_for_pose(
                    world, world_from_ee, dep.arm,
                    current_state_q=particles[dep.q_name].detach(),
                    ik=world.ik_refresh_solver if cheap else None,
                    seed_from_current=cheap,
                )
                new_q = _ik_solution_to_full_q(ik_result, world).detach()
                success = _ik_success_mask(ik_result)
                n_fail = int((~success).sum().item())
                if ok_now is not None:
                    # Repair-only merge: already-good rows keep their q.
                    success = success & ~ok_now
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
                )

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
        ) if ik_deps else None

        param_groups = []
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
                # Coupled re-IK: q is determined by IK from current Pose/Grasp.
                # requires_grad stays True so cost evaluation differentiates
                # through it; we just don't update it via Adam.
                val.requires_grad = True
                continue
            param_msg.append(f"(param: {param} = {tuple(val.shape)})")
            group = {"params": val}
            if param_type == Conf:
                group["lr"] = self.config.conf_lr
            param_groups.append(group)
            opt_metrics["opt_params"][param] = {"type": param_type, "shape": list(val.shape)}
            val.requires_grad = True

        if not param_groups:
            raise RuntimeError(f"No parameters to optimize! For plan skeleton: {plan_skeleton}")

        optimizer = Adam(params=param_groups, lr=self.config.lr)

        # Per-conf gradient mask: hard-zero DOFs that should not move during
        # this operator (e.g., base + inactive arm during arm operations).
        # See cutamp/conf_locking.py for the policy.
        grad_masks = build_grad_masks(
            plan_skeleton, particles, param_to_type, Conf,
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
        consider_types = {"constraint"}
        # Coupled re-IK breaks the KinematicConstraint coupling that destroys
        # vanilla Adam on pose-class soft costs, so soft costs can safely
        # join the Adam loss when --optimize_soft_costs is on. (A separate
        # "Phase 2" LBFGS soft polish existed for the upstream Adam-only-hard
        # regime; measured null-to-negative under coupled re-IK and deleted
        # 2026-08-04 — soft descent lives entirely in this Adam loss now.)
        if self.config.optimize_soft_costs and self.config.coupled_reik:
            consider_types.add("cost")
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
        with torch.no_grad():
            init_rollout = rollout_fn(particles)
            init_cost_dict = cost_fn(init_rollout, particles)
            init_satisfying = self.get_satisfying_mask(init_cost_dict, verbose=False)
            init_num = init_satisfying.sum().item()
        skip_adam = (init_num > 0) and not self.config.coupled_reik
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
        for step in pbar:
            if self.config.max_loop_dur is not None and timer.elapsed("start_optimization") >= self.config.max_loop_dur:
                _log.warning(f"Optimization exceeded max time of {self.config.max_loop_dur:.2f}s")
                time_exceeded = True
                break
            timer.start("optimization_step")
            optimizer.zero_grad()

            rollout = rollout_fn(particles)
            cost_dict = cost_fn(rollout, particles)
            costs = self.cost_reducer(cost_dict, consider_types=consider_types)
            satisfying_mask = self.get_satisfying_mask(cost_dict, verbose=False)
            num_satisfying = satisfying_mask.sum().item()
            opt_metrics["num_satisfying"].append(num_satisfying)

            if num_satisfying > 0 and not found_solution:
                if timer.has_timer("first_solution"):
                    time_to_first_sol = timer.stop("first_solution")
                    _log.info(f"Found first solution in {time_to_first_sol:.2f}s after sampling plans")

                torch.cuda.synchronize()
                opt_metrics["opt_start_to_first_sol"] = time.perf_counter() - start_time
                found_solution = True

            loss = costs.mean()
            opt_metrics["loss"].append(loss.item())

            # Compute soft costs if required (even if we don't optimize them)
            if self.config.soft_cost is not None:
                soft_costs = self.cost_reducer.soft_costs(cost_dict)
                best_soft_cost = soft_costs[satisfying_mask].min().item() if num_satisfying > 0 else None
                opt_metrics["best_soft_costs"].append(best_soft_cost)

            torch.cuda.synchronize()
            opt_metrics["elapsed"].append(time.perf_counter() - start_time)

            # Best-seen snapshot + early-stop bookkeeping. Measured state is
            # pre-step, so the clone is consistent with num_satisfying.
            best_snapshot.update(step, num_satisfying, costs.min().item(), particles)
            if best_snapshot.should_stop(step, self.config.early_stop_patience):
                _log.info(
                    f"Early stop at step {step}: no meaningful progress for "
                    f"{self.config.early_stop_patience} steps (best "
                    f"{best_snapshot.best_sat} satisfying at step {best_snapshot.best_step})"
                )
                opt_metrics["early_stopped_at"] = step
                break

            # Visualize the optimization progress. We do this before stepping the optimizer to see current state.
            if step % vis_every == 0 or step == num_steps - 1:
                timer.start("visualize_opt_rollout")
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
                visualizer.log_scalar("loss", loss.item())
                timer.stop("visualize_opt_rollout")

            # Compute gradients and step the optimizer.
            loss.backward()
            apply_grad_masks(optimizer.param_groups, grad_masks)
            optimizer.step()
            # Counts from 1 so the first refresh lands at step K-1 (after
            # the first batch of pose updates), not step 0.
            if ik_deps and (step + 1) % self.config.reik_interval == 0:
                self._refresh_ik_deps(particles, ik_deps, world, gate=reik_gate, step=step)
            timer.stop("optimization_step")
            pbar.set_description(
                f"Loss: {loss:.3f}, Min: {costs.min():.3f}, {num_satisfying}/{self.config.num_particles} satisfying"
            )

        # Restore the best-seen state if the loop ended on a dip of the
        # post-convergence oscillation (final state is otherwise a lottery).
        if not skip_adam and best_snapshot.restore_if_better(
            particles, opt_metrics["num_satisfying"][-1] if opt_metrics["num_satisfying"] else 0
        ):
            _log.info(
                f"Restored best-seen particles: {best_snapshot.best_sat} satisfying "
                f"(step {best_snapshot.best_step}) over final "
                f"{opt_metrics['num_satisfying'][-1]}"
            )
        opt_metrics["best_num_satisfying"] = best_snapshot.best_sat

        # Final re-IK before downstream satisfaction checks: Adam may have
        # moved poses since the last periodic refresh. Always FULL (no gate):
        # restore_if_better may have rewritten particles, making gate refs stale.
        if ik_deps:
            self._refresh_ik_deps(particles, ik_deps, world)
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
        # costs if config.optimize_soft_costs is True, while it only includes hard costs otherwise.
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
        except Exception as e:
            _log.debug(f"COM-in-polygon summary skipped: {e}")

        # Visualize rollout of the best particle in terms of soft costs, which is just trajectory length if no
        # other soft costs are specified. If no satisfying particles, just visualize lowest overall cost particle.
        if num_satisfying > 0:
            best_satisfying_idx = soft_costs[satisfying_mask].argmin()
            best_idx = indices[satisfying_mask][best_satisfying_idx]
        else:
            best_idx = costs.argmin()

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
