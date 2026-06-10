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
    action_6dof_to_mat4x4,
)
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.particle_initialization import _ik_for_pose, _ik_solution_to_full_q
from cutamp.rollout import RolloutFunction
from cutamp.t1_domain import Conf, Pose
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
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
    plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration,
) -> List[IKDep]:
    """Walk the plan skeleton and emit one IKDep per actionable op whose Conf
    is determined by Pose / Grasp particles via IK.

    Mirrors ``ParticleInitializer``'s init-time IK targets (Pick /
    Place / Push / PushStick) so re-IK reproduces what was done at init,
    just from updated particle values. Operators with no IK target
    (Retract, MoveFree, MoveHolding, Navigate) are skipped.
    """
    deps: List[IKDep] = []
    grasp_to_mat = action_4dof_to_mat4x4 if config.grasp_dof == 4 else action_6dof_to_mat4x4
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
        elif action_type == "push":
            button, push_pose, q_name = params
            def target_fn(p, _pp=push_pose, _t=tool_from_ee):
                return action_4dof_to_mat4x4(p[_pp]) @ _t
            deps.append(IKDep(q_name=q_name, arm=arm, target_fn=target_fn))
        elif action_type == "push_stick":
            button, stick_name, grasp_name, push_pose, q_name = params
            def target_fn(p, _pp=push_pose, _g=grasp_name, _t=tool_from_ee, _gm=grasp_to_mat):
                return action_4dof_to_mat4x4(p[_pp]) @ _gm(p[_g]) @ _t
            deps.append(IKDep(q_name=q_name, arm=arm, target_fn=target_fn))
        # action_type in {"retract", "navigate", None}: no IK target, skip.
    return deps


class PlanContainer(TypedDict):
    plan_skeleton: PlanSkeleton
    particles: Particles
    rollout_fn: RolloutFunction
    cost_fn: CostFunction
    heuristic: float


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

    def _run_phase2_lbfgs_refinement(
        self,
        particles: dict,
        sat_mask: torch.Tensor,
        rollout_fn,
        cost_fn,
        *,
        barrier_weight: float = 1e6,
        max_iter: int = 20,
    ) -> None:
        """Refine soft costs on satisfying particles via LBFGS + line search.

        Operates on a SUBSET of particles (those that already satisfy hard
        constraints). The barrier term ``barrier_weight * mean(hard_costs)``
        spikes the loss whenever a step would break satisfaction, so the
        strong-Wolfe line search backtracks before destruction. Refined
        values are written back into ``particles[*][sat_mask]``.
        """
        n_sat = int(sat_mask.sum().item())
        if n_sat == 0:
            return
        # Detach + clone so LBFGS owns the parameter tape independent of any
        # outer optimizer state (Adam, grad_masks, etc.).
        refined = {
            k: v[sat_mask].detach().clone().requires_grad_(True)
            for k, v in particles.items()
        }
        params = list(refined.values())
        lbfgs = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
        )

        def closure():
            lbfgs.zero_grad()
            rollout = rollout_fn(refined)
            cost_dict = cost_fn(rollout, refined)
            soft = self.cost_reducer.soft_costs(cost_dict).mean()
            hard = self.cost_reducer.hard_costs(cost_dict).mean()
            loss = soft + barrier_weight * hard
            loss.backward()
            return loss

        with torch.no_grad():
            soft_before = self.cost_reducer.soft_costs(
                cost_fn(rollout_fn(refined), refined)
            ).mean().item()
        try:
            lbfgs.step(closure)
        except Exception as e:
            _log.warning(f"Phase 2 LBFGS refinement failed: {e}; keeping pre-Phase-2 particles.")
            return
        with torch.no_grad():
            cd = cost_fn(rollout_fn(refined), refined)
            soft_after = self.cost_reducer.soft_costs(cd).mean().item()
            post_sat = self.get_satisfying_mask(cd, verbose=False)
        kept = int(post_sat.sum().item())
        _log.info(
            f"Phase 2 LBFGS: refined {n_sat} particles; soft {soft_before:.4f} → "
            f"{soft_after:.4f}; {kept}/{n_sat} still satisfying after refinement."
        )
        # Write refined values back. Particles that became unsatisfying during
        # refinement (rare with the barrier, but possible if Wolfe couldn't
        # find a backtracking step) get their original values restored.
        with torch.no_grad():
            for k in particles:
                # Build the refined-or-original tensor for the sat_mask slice.
                sub = refined[k].detach()
                # Restore to pre-refinement values where post_sat is False.
                if (~post_sat).any():
                    pre = particles[k][sat_mask]
                    sub = torch.where(
                        post_sat.view(-1, *([1] * (sub.dim() - 1))).expand_as(sub),
                        sub, pre,
                    )
                particles[k][sat_mask] = sub

    def _refresh_ik_deps(
        self,
        particles: dict,
        ik_deps: List[IKDep],
        world: TAMPWorld,
    ) -> None:
        """Re-solve IK for each ``IKDep`` and write the result into the
        corresponding Conf particle in-place. Failed particles keep their
        previous q (per-particle masked write).

        cuRobo's IK uses autograd internally (LM Jacobian via ``cost.backward``)
        so the IK call cannot run under ``torch.no_grad``; only the in-place
        ``copy_`` is wrapped. The ``copy_`` preserves tensor identity, which
        ``grad_masks`` and Adam state both depend on.
        """
        if not ik_deps:
            return
        for dep in ik_deps:
            # Detach the target so IK's backward doesn't try to flow into
            # Pose / Grasp particles (which are leaves in our autograd graph).
            world_from_ee = dep.target_fn(particles).detach()
            current_state_q = particles[dep.q_name].detach()
            ik_result = _ik_for_pose(
                world, world_from_ee, dep.arm,
                current_state_q=current_state_q,
            )
            new_q = _ik_solution_to_full_q(ik_result, world).detach()
            success = ik_result.success.view(-1, 1)
            with torch.no_grad():
                merged = torch.where(success, new_q, particles[dep.q_name])
                particles[dep.q_name].copy_(merged)
                # Dropped from the optimizer, so .grad has no consumer.
                particles[dep.q_name].grad = None

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
            _build_ik_deps(plan_skeleton, world, self.config)
            if self.config.coupled_reik else []
        )
        dropped_conf_names = {dep.q_name for dep in ik_deps}

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
        # Adam only ever minimizes hard-constraint penalties (Phase 1).
        # ``--optimize_soft_costs`` engages a separate Phase 2 (LBFGS with
        # strong-Wolfe line search + barrier on hard constraints) that
        # refines satisfying particles for soft costs without breaking
        # constraint satisfaction.
        consider_types = {"constraint"}
        # Upstream-style fallback: when set, mimic NVlabs/cuTAMP main —
        # Adam runs always with soft costs in the loss (statistical
        # exploration), no Phase 2. Useful for pose-class soft costs that
        # have no constraint-manifold tangent for LBFGS to refine along.
        upstream_style = self.config.upstream_style_optimize
        # Coupled re-IK breaks the KinematicConstraint coupling that destroys
        # vanilla Adam on pose-class soft costs, so soft costs can safely
        # join the Adam loss. Phase 2 LBFGS still runs for q-class refinement.
        if self.config.optimize_soft_costs and (upstream_style or self.config.coupled_reik):
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
        skip_adam = (
            (init_num > 0) and not upstream_style and not self.config.coupled_reik
        )
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
            # Skip per-conf gradient masking in upstream-style mode so the
            # comparison is honest (upstream has no grad masking).
            loss.backward()
            if not upstream_style:
                apply_grad_masks(optimizer.param_groups, grad_masks)
            optimizer.step()
            # Counts from 1 so the first refresh lands at step K-1 (after
            # the first batch of pose updates), not step 0.
            if ik_deps and (step + 1) % self.config.reik_interval == 0:
                self._refresh_ik_deps(particles, ik_deps, world)
            timer.stop("optimization_step")
            pbar.set_description(
                f"Loss: {loss:.3f}, Min: {costs.min():.3f}, {num_satisfying}/{self.config.num_particles} satisfying"
            )

        # Final re-IK before downstream satisfaction checks: Adam may have
        # moved poses since the last periodic refresh.
        if ik_deps:
            self._refresh_ik_deps(particles, ik_deps, world)

        # Phase 2 (LBFGS soft-cost refinement): when --optimize_soft_costs is
        # set, refine the satisfying particles' soft costs with LBFGS +
        # strong-Wolfe line search. A large barrier on the hard-constraint
        # penalty makes any step that breaks satisfaction increase the loss,
        # so LBFGS's line search backtracks before taking it. Result:
        # soft cost monotonically decreases on satisfying particles, and
        # hard-constraint satisfaction is preserved by construction.
        if self.config.optimize_soft_costs and not upstream_style:
            with torch.no_grad():
                pre_p2_rollout = rollout_fn(particles)
                pre_p2_cost_dict = cost_fn(pre_p2_rollout, particles)
                pre_p2_sat = self.get_satisfying_mask(pre_p2_cost_dict, verbose=False)
            if pre_p2_sat.any():
                self._run_phase2_lbfgs_refinement(
                    particles, pre_p2_sat, rollout_fn, cost_fn,
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
            conf_name = rollout["conf_params"][ts]

            gripper_close = rollout["gripper_close"][ts]
            # T1 gripper: 4-bar parallel linkage, 4 joints per hand.
            # GRIPPER_OPEN = (0,0,0,0), GRIPPER_CLOSED = (1,-1,-1,1).
            gripper_joints = [1.0, -1.0, -1.0, 1.0] if gripper_close else [0.0, 0.0, 0.0, 0.0]
            arm = "left" if conf_name.startswith("left_") else "right"
            visualizer.set_joint_positions(q.tolist() + gripper_joints, arm=arm)

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
