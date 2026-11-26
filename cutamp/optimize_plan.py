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
from typing import AbstractSet, Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from torch.optim import Adam
from tqdm import tqdm

from cutamp import cuda_graphs
from cutamp.conf_locking import apply_grad_masks, build_grad_masks
from cutamp.utils.common import Particles
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.robots.t1 import GRIPPER_CLOSED, GRIPPER_OPEN
from cutamp.rollout import RolloutFunction, get_conf_to_arm
from cutamp.t1_domain import Conf, Pose
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


#: Cost types in the Adam loss: the hard constraints and the soft costs.
ADAM_CONSIDER_TYPES = frozenset(("constraint", "cost"))


def _adam_param_groups(params: List[Tuple[torch.Tensor, bool]], conf_lr: float) -> List[dict]:
    """One Adam group per learning rate, non-Confs at ``lr`` then Confs at ``conf_lr``, empty ones dropped:
    Adam launches its foreach kernels once per group, and each element's update is unchanged."""
    non_confs = [p for p, is_conf in params if not is_conf]
    confs = [p for p, is_conf in params if is_conf]
    return [g for g in ({"params": non_confs}, {"params": confs, "lr": conf_lr}) if g["params"]]


def _finite_step(optimizer: Adam, params: List[torch.Tensor]) -> torch.Tensor:
    """``optimizer.step()`` for a plan (design 4): a particle (a row, the first dimension) whose gradient is not
    finite in any parameter keeps its values, so a refresh never receives a command that is not a number. Its gradient
    rows are zeroed, so Adam's moments take no NaN, and its values are put back after the step, which the moments
    still move. A parameter without a gradient is left alone. Returns the held rows, ``[B]`` bool; no host sync."""
    graded = [p for p in params if p.grad is not None]
    held = torch.zeros(params[0].shape[0], dtype=torch.bool, device=params[0].device)
    for p in graded:
        held |= ~p.grad.reshape(p.shape[0], -1).isfinite().all(1)
    with torch.no_grad():
        rows = [held.view((-1,) + (1,) * (p.ndim - 1)) for p in graded]
        saved = [p.detach().clone() for p in graded]
        for p, row in zip(graded, rows):
            p.grad.masked_fill_(row, 0)
        optimizer.step()
        for p, row, value in zip(graded, rows, saved):
            p.copy_(torch.where(row, value, p))
    return held


class _FeasibleArchive:
    """Keep each particle's best complete feasible state across the Adam steps, by ``keys`` compared in order, smaller
    first (:func:`_quality`'s): ``score`` is the first key and ``effort`` the last, its worst joint effort."""

    def __init__(self, particles, *keys):
        self.keys = [torch.full_like(key, float("inf")) for key in keys]
        self.state = {name: value.detach().clone() for name, value in particles.items()}

    @property
    def score(self):
        return self.keys[0]

    @property
    def effort(self):
        return self.keys[-1]

    @staticmethod
    def _before(a, b):
        """``[n]`` bool: the keys ``a`` come before ``b``, compared in order."""
        before, tied = torch.zeros_like(a[0], dtype=torch.bool), torch.ones_like(a[0], dtype=torch.bool)
        for x, y in zip(a, b):
            before, tied = before | (tied & (x < y)), tied & (x == y)
        return before

    def best(self) -> list[float]:
        """The keys of the best particle on record, in order; all infinite while no particle has been valid."""
        best, tied = [], torch.ones_like(self.keys[0], dtype=torch.bool)
        for key in self.keys:
            value = key.masked_fill(~tied, float("inf"))
            best.append(value.min())
            tied = tied & (value == best[-1])
        return torch.stack(best).tolist()

    @torch.no_grad()
    def observe(self, particles, feasible, *keys):
        better = feasible & self._before(keys, self.keys)
        for saved, key in zip(self.keys, keys):
            saved.copy_(torch.where(better, key, saved))
        for name, saved in self.state.items():
            mask = better.view((-1,) + (1,) * (saved.ndim - 1))
            saved.copy_(torch.where(mask, particles[name], saved))

    @torch.no_grad()
    def restore(self, particles, feasible, *keys):
        better = self.score.isfinite() & (~feasible | self._before(self.keys, keys))
        if not better.any():
            return False
        for name, saved in self.state.items():
            mask = better.view((-1,) + (1,) * (saved.ndim - 1))
            value = particles[name]
            value.copy_(torch.where(mask, saved, value))
        return True


def _quality(cost_dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The numbers :class:`_FeasibleArchive` compares valid particles by, in order, smaller first: the score
    (:func:`cutamp.step_costs.plan_score`), the rank (:func:`cutamp.step_costs.plan_rank`) and the worst joint effort
    the plan's cost publishes (2026-11-25)."""
    soft = cost_dict["soft"]
    return soft["score"], soft["rank"], soft["effort"]


def _fixed_row_quality(cost_dict, total_cost, soft_cost, reducer):
    """Scores of the final rows: the costs, the centring distance where the cost publishes one, and the score, the
    rank and the worst joint effort, which choose the row."""
    quality = {"best_cost": total_cost, "best_soft_cost": soft_cost}
    center = cost_dict["soft"].get("max_center_distance_m")
    if center is not None:
        quality.update(max_center_distance_m=center, secondary_effort=reducer.secondary_effort(cost_dict))
    quality.update(score=cost_dict["soft"]["score"], rank=cost_dict["soft"]["rank"],
                   joint_effort=cost_dict["soft"]["effort"])
    return quality


def _best_fixed_row(quality, passing):
    """The passing row of the lowest score, then of the lowest rank, then of the lowest joint effort."""
    for key in ("score", "rank"):
        value = quality[key].masked_fill(~passing, float("inf"))
        passing = passing & (value == value.min())
    return int(quality["joint_effort"].masked_fill(~passing, float("inf")).argmin())


@dataclass
class _StepBody:
    """One Adam step's device work, run eagerly or captured as G_step: the put-back, the forward, the per-step
    scalars stacked into ``packed`` (satisfying count, loss, min cost, then the best satisfying soft cost), the backward
    with the grad masks and the gradients' hand-on. The min cost is the smallest ``cost_at_start_weights`` where the
    cost publishes it, since the weights grow. Returns ``(packed, vis)``; ``vis`` is ``None`` unless ``keep_vis``."""
    particles: dict
    rollout_fn: RolloutFunction
    cost_fn: CostFunction
    cost_reducer: CostReducer
    get_mask: Callable[..., torch.Tensor]
    consider_types: AbstractSet[str]
    params: List[torch.Tensor]
    grad_masks: Dict[int, torch.Tensor]
    keep_vis: bool
    #: The plan itself (:class:`~cutamp.step_loop.SteppedPlan`): its postures are put back before the forward and,
    #: after the backward, their gradients are handed on to the commands and placements (``put_back``,
    #: ``hand_on``), all in the one recording.
    steps: object
    archive: _FeasibleArchive
    #: Whether the archive reads this step, a 0-dim bool tensor: a plan's particles are judged only right after a
    #: refresh (call 11). The loop sets it before each step, since a recorded step takes the record at every replay
    #: and can only mask it.
    judge: torch.Tensor

    def __call__(self) -> Tuple[torch.Tensor, Optional[tuple]]:
        self.steps.put_back(self.particles)
        rollout = self.rollout_fn(self.particles)
        cost_dict = self.cost_fn(rollout, self.particles)
        costs = self.cost_reducer(cost_dict, consider_types=self.consider_types)
        satisfying_mask = self.get_mask(cost_dict, verbose=False)
        self.archive.observe(self.particles, satisfying_mask & self.judge, *_quality(cost_dict))
        loss = costs.mean()
        # Every per-step scalar in one tensor, so the loop reads them in one
        # device-to-host copy: each .item() blocks the host until the GPU drains.
        scalars = [satisfying_mask.sum().to(costs.dtype), loss, costs.min()]
        at_start = cost_dict["soft"].get("cost_at_start_weights")
        if at_start is not None:  # the stop rule's loss clock cannot read weights that double (2026-11-23 design 4.6)
            scalars[2] = at_start.nan_to_num(nan=float("inf")).min()
        else:  # a row whose refresh diverged holds NaN costs: else the min is NaN and the loss clock never resets
            scalars[2] = costs.nan_to_num(nan=float("inf")).min()
        soft_costs = self.cost_reducer.soft_costs(cost_dict)
        scalars.append(soft_costs.masked_fill(~satisfying_mask, float("inf")).min())
        packed = torch.stack(scalars).detach()
        # The backward runs before the loop's bookkeeping, so elapsed[s] and the first-solution times include
        # it. It writes only .grad, which the bookkeeping doesn't read.
        loss.backward()
        apply_grad_masks([{"params": self.params}], self.grad_masks)
        self.steps.hand_on(self.particles)
        return packed, ((rollout, cost_dict, costs, satisfying_mask) if self.keep_vis else None)

    def clear_autograd(self) -> None:
        """Drop every param's ``.grad``: before a capture, so its first warm-up starts from no grads, and after
        G_step's release, when they hold its pool."""
        for p in self.params:
            p.grad = None


class PlanContainer(TypedDict):
    plan_skeleton: PlanSkeleton
    particles: Particles
    rollout_fn: RolloutFunction
    cost_fn: CostFunction
    steps: object  # the plan itself (cutamp.step_loop.SteppedPlan)


class _BestSnapshot:
    """The Adam loop's early-stop bookkeeping.

    Before any particle has been valid, the patience clock resets only on *meaningful* progress: a satisfying-count
    record beating the previous by ``MIN_DELTA``, or the best particle's loss dropping by ``MIN_LOSS_DELTA``. Once
    the archive holds a valid plan, only a valid plan of a lower score resets the clock: the returned plan is the one
    of the lowest score, so neither the count nor the loss changes it. Once that plan has rank 0, no unit of side lean
    or off centre left, the loop ends at once: it does not wait for the forward lean, the last preference, to fall
    further (user, 2026-11-25). A plan of rank 0 that is not the best one, such as a centred fold, ends nothing.
    """

    # ponytail: constants calibrated on one blocks_t1 trace (max record gap 35,
    # sawtooth noise band ~±50); revisit if varying-heights traces disagree.
    MIN_DELTA = 5
    MIN_LOSS_DELTA = 0.05

    def __init__(self):
        self.best_sat = -1
        self.best_step = 0
        self.last_improve_step = 0
        self._patience_sat = -1
        self._best_min_loss = float("inf")
        self._best_score = float("inf")
        self._score_step = None  # the step of the last valid plan of a lower score; None until one exists
        self.centred = False  # the best plan on record has rank 0

    def update(self, step: int, num_satisfying: int, min_loss: float):
        if num_satisfying > self.best_sat:
            self.best_sat = num_satisfying
            self.best_step = step
        meaningful = num_satisfying >= self._patience_sat + self.MIN_DELTA
        loss_improved = min_loss < self._best_min_loss - self.MIN_LOSS_DELTA
        if meaningful:
            self._patience_sat = num_satisfying
        if loss_improved:
            self._best_min_loss = min_loss
        if meaningful or loss_improved:
            self.last_improve_step = step

    def should_stop(self, step: int, patience: int, archive: _FeasibleArchive) -> bool:
        if patience <= 0:
            return False
        score, rank = archive.best()[:2]  # inf while no particle has been valid
        if score < self._best_score:
            self._best_score, self._score_step = score, step
        self.centred = rank == 0
        if self.centred:
            return True
        if self._score_step is not None:
            return step - self._score_step > patience
        return step - self.last_improve_step > patience


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

    def _capture_step(self, body: _StepBody) -> cuda_graphs.CapturedGraph:
        """Capture ``body`` as G_step. Each warm-up drops every ``.grad``, so the captured AccumulateGrad
        gives each param a graph-owned ``.grad`` that every replay rewrites in place (never zero_grad it).
        The guard covers the particles, those grads and what the plan names (``body.steps.recorded()``): the tensors
        its refresh writes and the kinematics its cost runs."""
        steps = body.steps
        body.clear_autograd()

        def warmup() -> None:
            # A put-back moves the postures and counts capped rows: each warm-up undoes both.
            with torch.no_grad():
                kept = {name: body.particles[name].clone() for name in steps.gap_of}
                capped = steps.capped.clone()
            body()
            for p in body.params:
                p.grad = None
            with torch.no_grad():
                for name, q in kept.items():
                    body.particles[name].copy_(q)
                steps.capped.copy_(capped)

        def ptrs() -> tuple:
            tensors, kins = steps.recorded()
            return cuda_graphs.graph_ptrs([*body.particles.values(), *(p.grad for p in body.params), *tensors], kins)

        return cuda_graphs.CapturedGraph("G_step", body, warmup, ptrs=ptrs)

    def _visualize_step(self, visualizer: Visualizer, step: int, vis: tuple, num_satisfying: int,
                        loss_val: float) -> None:
        """Log the best particle's last conf and object poses, the cost dict and the loss, from the step's ``vis``."""
        rollout, cost_dict, costs, satisfying_mask = vis
        # Visualize satisfying particles if we have any: lowest score, then rank, then joint effort.
        # Otherwise, just show the best particle so far.
        if num_satisfying > 0:
            soft = cost_dict["soft"]
            best_idx = _best_fixed_row({"score": soft["score"], "rank": soft["rank"],
                                        "joint_effort": soft["effort"]}, satisfying_mask)
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

    def __call__(
        self, plan_info: PlanContainer, timer: TorchTimer, visualizer: Visualizer, *, deadline=None
    ) -> Tuple[bool, dict, bool]:
        """Optimize particles for the plan skeleton.

        Returns ``(solution_found, opt_metrics, time_exceeded)``.

        ``plan_info["steps"]`` is the plan (:class:`~cutamp.step_loop.SteppedPlan`, design 4 and 6), with or without
        steps: its commands and its postures are Adam variables, it refreshes every ``reik_interval`` steps, its
        particles are judged only on the step right after a refresh, and it refreshes once more after the loop. Its
        step body puts the postures back first (``steps.put_back``), then runs on ``steps.rollout_fn`` and hands the
        gradients on (``steps.hand_on``); with ``cuda_graphs`` the body is recorded and replayed (2026-11-24 design),
        guarded by what ``steps.recorded()`` names. The recording holds the whole step but the Adam update and the
        clamps after it: the put-back, the forward and backward, and the gradients' hand-on.
        """
        plan_skeleton = plan_info["plan_skeleton"]
        particles = plan_info["particles"]
        rollout_fn = plan_info["rollout_fn"]
        cost_fn = plan_info["cost_fn"]
        steps = plan_info["steps"]
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

        world = cost_fn.world  # the plan's rollout is a method of the plan (SteppedPlan.rollout), without a world
        pairs = []  # (tensor, is a Conf) per optimized param, for _adam_param_groups
        # Initial configurations that we don't optimize
        initial_confs = {"q0", "left_q0", "right_q0"}
        for param, val in particles.items():
            if param in steps.landing_confs:
                val.requires_grad = False  # the posture a step lands in is the world model's, written by a refresh
                continue
            # Skip particles that aren't used in this plan (e.g., right_q0 in a left-arm only plan)
            if param not in param_to_type:
                continue
            param_type = param_to_type[param]
            if param_type not in self.types_to_optimize:
                continue
            if param in initial_confs:
                continue  # we don't optimize the initial configuration(s)
            pairs.append((val, param_type == Conf))
            val.requires_grad = True
        for name in steps.command_names.values():  # the commands, in the placements' group (call 15)
            pairs.append((particles[name], False))
            particles[name].requires_grad = True

        optimizer = Adam(params=_adam_param_groups(pairs, self.config.conf_lr), lr=self.config.lr)

        # Per-conf gradient mask: hard-zero DOFs that should not move during
        # this operator (the inactive arm during arm operations).
        # See cutamp/conf_locking.py for the policy.
        grad_masks = build_grad_masks(plan_skeleton, particles, param_to_type, Conf, world.model)

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
        # costs. Adam runs even when the starting values already satisfy, since the
        # goal is to actively move the particles for the soft cost; the
        # start's satisfying particles are step 0's count.
        with torch.no_grad():
            init_rollout = rollout_fn(particles)
            init_cost_dict = cost_fn(init_rollout, particles)
            init_satisfying = self.get_satisfying_mask(init_cost_dict, verbose=False)
            init_num = init_satisfying.sum().item()
        if init_num > 0:
            # First-solution timing semantics: we already have a solution
            # from the starting values, so stamp 0.0 even though Adam is about to run.
            opt_metrics["opt_start_to_first_sol"] = 0.0
            found_solution = True

        pbar = tqdm(range(num_steps))
        start_time = time.perf_counter()
        time_exceeded = False
        best_snapshot = _BestSnapshot()
        archive = _FeasibleArchive({k: v for k, v in particles.items() if v.requires_grad}, *_quality(init_cost_dict))
        # The body puts the postures back, runs on the arm's rollout and hands the gradients on.
        body = _StepBody(
            particles, steps.rollout_fn, cost_fn, self.cost_reducer,
            self.get_satisfying_mask, consider_types,
            params=[p for g in optimizer.param_groups for p in g["params"]], grad_masks=grad_masks,
            # The no-op Visualizer skips the vis block, and with it the block's syncs.
            keep_vis=type(visualizer) is not Visualizer, steps=steps, archive=archive,
            judge=torch.zeros((), dtype=torch.bool, device=world.device),
        )
        step_graph = None
        use_graphs = self.config.cuda_graphs
        if use_graphs:
            opt_metrics.update(capture_s=None, capture_step=None)
        try:
            for step in pbar:
                if deadline is not None and time.monotonic() >= deadline:
                    time_exceeded = True
                    break
                timer.start("optimization_step")  # a capture's cost lands in its step
                # A plan is judged only right after a refresh; the first ran before the loop (call 11).
                judged = step % self.config.reik_interval == 0
                body.judge.fill_(judged)
                if use_graphs and step_graph is None:
                    step_graph = self._capture_step(body)
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

                if num_satisfying > 0 and not found_solution and judged:
                    torch.cuda.synchronize()
                    opt_metrics["opt_start_to_first_sol"] = time.perf_counter() - start_time
                    found_solution = True

                opt_metrics["loss"].append(loss_val)

                # Soft costs are recorded even if we don't optimize them.
                opt_metrics["best_soft_costs"].append(best_soft if num_satisfying > 0 else None)

                torch.cuda.synchronize()
                opt_metrics["elapsed"].append(time.perf_counter() - start_time)

                # Early-stop bookkeeping, on the judged steps only.
                if judged:
                    best_snapshot.update(step, num_satisfying, min_cost)
                if judged and best_snapshot.should_stop(step, self.config.early_stop_patience, archive):
                    ranked = ""
                    if best_snapshot._score_step is not None:
                        ranked = f"; best score {best_snapshot._best_score:.0f} at step {best_snapshot._score_step}"
                    why = f"no meaningful progress for {self.config.early_stop_patience} steps"
                    if best_snapshot.centred:
                        why = "the best plan has no unit of side lean or off centre left"
                    _log.info(
                        f"Early stop at step {step}: {why} (best "
                        f"{best_snapshot.best_sat} satisfying at step {best_snapshot.best_step}{ranked})"
                    )
                    opt_metrics["early_stopped_at"] = step
                    # Close it: the next skeleton reuses this timer.
                    timer.stop("optimization_step")
                    break

                # Visualize the optimization progress. We do this before stepping the optimizer to see current state.
                if vis is not None and (step % vis_every == 0 or step == num_steps - 1):
                    with timer.time("visualize_opt_rollout"):
                        self._visualize_step(visualizer, step, vis, num_satisfying, loss_val)

                # The body already ran the backward and the grad masks.
                steps.step(optimizer, body.params, particles)
                # Counts from 1 so the first refresh lands at step K-1 (after
                # the first batch of updates), not step 0.
                if (step + 1) % self.config.reik_interval == 0:
                    steps.refresh(particles)  # the world model at the current commands (call 7)
                timer.stop("optimization_step")
                pbar.set_description(
                    f"Loss: {loss_val:.3f}, Min: {min_cost:.3f}, "
                    f"{num_satisfying}/{self.config.num_particles} satisfying"
                )
                vis = None  # drop this step's rollout (and in eager mode its autograd graph) before the next step
        finally:  # also on early stop and on errors: no graph outlives the loop
            if step_graph is not None:
                step_graph.release()
                body.clear_autograd()

        # The current state was last judged on a stand-in, so every archived state goes back (call 11).
        unjudged = torch.zeros_like(archive.score, dtype=torch.bool)
        restored = archive.restore(particles, unjudged, *archive.keys)
        # The last refresh: the final check below reads the world model's values. No command moves after it, so
        # it measures no slope: the particles' own rows only, without their nudged copies (2026-11-24).
        steps.refresh(particles, slopes=False)
        opt_metrics["best_num_satisfying"] = best_snapshot.best_sat
        # What Adam added over step 0, which measures the untouched start (None if no step ran:
        # best_sat stays -1), and whether the reported particles are a restored record.
        opt_metrics["adam_gain"] = (
            best_snapshot.best_sat - opt_metrics["num_satisfying"][0] if best_snapshot.best_sat >= 0 else None
        )
        opt_metrics["restored"] = restored

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

        quality = _fixed_row_quality(cost_dict, costs, soft_costs, self.cost_reducer)
        best_idx = _best_fixed_row(quality, satisfying_mask) if num_satisfying else int(costs.argmin())
        opt_metrics.update({k: float(v[best_idx]) if num_satisfying else None for k, v in quality.items()})
        if num_satisfying and "max_center_distance_m" not in quality:
            opt_metrics["best_cost"] = float(costs[satisfying_mask].min())
        opt_metrics["steps"] = steps.stats

        # Log initial state
        timer.start("visualize_rollout")
        conf_to_arm = get_conf_to_arm(plan_skeleton)
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
            arm = conf_to_arm.get(rollout["conf_params"][ts])
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
