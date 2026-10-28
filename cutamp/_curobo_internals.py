# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Encapsulated cuRobo v0.8 internal-API workarounds.

Every helper here reaches past cuRobo's public surface to fix a bug or work
around a missing API. They are gathered in one module so a future cuRobo
upgrade has exactly one place to audit.

Documented public-API constraints (no helper needed; noted here so the next
upgrade audit catches them):

* CUDA graphs can't be re-captured ("CUDA graph reset is not available."), so
  a solver whose call structure changes after its first solve must run
  without them. The main IK solver in ``cutamp.robots.t1`` does (particle init
  mixes calls with and without ``current_state``); the coupled-reIK refresh
  solver and the motion planner each see one fixed call shape and use them.
  cuRobo captures in CUDA's global mode, where another thread's allocations
  and syncs fail mid-capture, so captures run only on the main thread while
  the planner-build thread is not running (``TAMPWorld.warmup_ik_solver``,
  ``motion_solver.build_motion_planner``).
"""

from typing import Dict, Iterable, List

import torch

# TODO: file upstream issue against cuRobo. ``MotionPlanner.attachment_manager``
# is a property that delegates to ``trajopt_solver.attachment_manager``; that
# attribute does not exist on v0.8. The actual manager lives on
# ``trajopt_solver.core.attachment_manager``.
def get_attachment_manager(planner):
    """Resolve the attachment_manager via the working access path."""
    am = getattr(planner, "attachment_manager", None)
    if am is not None:
        return am
    core = getattr(planner.trajopt_solver, "core", None)
    if core is not None and hasattr(core, "attachment_manager"):
        return core.attachment_manager
    return None


# Kept here for the upgrade audit: in cuRobo 0.8, ``result.success`` from
# plan_cspace is already the full feasibility test (TrajOptSolverResult.
# _process_metrics): last step converged on every tool frame, AND every
# constraint (scene/self collision, joint limits) satisfied over the
# optimized AND the interpolated trajectory.
# Don't second-guess it: ``result.cspace_error`` is declared but never set
# (always None), and a trajectory ending on the goal proves nothing because
# cuRobo pins the last waypoint to it. A rejected T1 plan is a real violation
# (e.g. ankle_pitch past its 0 rad upper limit mid-path); fix the planner.
def cspace_plan_succeeded(result) -> bool:
    """True iff cuRobo reports a successful (converged and feasible) plan."""
    return result is not None and bool(result.success.any())


# TODO: file upstream issue against cuRobo. The per-DOF target weight tensor
# lives on each optimizer rollout's cost_manager (one per rollout — sampling +
# gradient-based) with no public mutator. Direct ``.copy_`` is the only way to
# update the weights so the kernel picks them up; reassignment would break the
# device/dtype binding the optimizer captured.
def iter_rollouts(host) -> List:
    """Resolve the optimizer rollout list on either a MotionPlanner or an IKSolver.

    MotionPlanner exposes them under ``host.trajopt_solver.optimizer_rollouts``;
    cuRobo's IKSolver puts them on ``host.core.optimizer_rollouts``.
    """
    if hasattr(host, "trajopt_solver"):
        return list(getattr(host.trajopt_solver, "optimizer_rollouts", []))
    if hasattr(host, "core"):
        return list(getattr(host.core, "optimizer_rollouts", []))
    return list(getattr(host, "optimizer_rollouts", []))


# All mutators below share this walk: rollouts → cost_manager → costs["cspace"].
# The cspace cost reads ``config.joint_limits.position`` and ``self._weight``
# fresh every iteration (no caching), so in-place mutation takes effect
# immediately. ``cost._weight`` is the LIVE tensor; ``config.weight`` is the
# "armed" value ``enable_cost()`` restores from.
def _cspace_costs_from_rollouts(rollouts: Iterable) -> List:
    costs = []
    for r in rollouts:
        cm = getattr(r, "cost_manager", None)
        if cm is None:
            continue
        cspace_cost = cm.costs.get("cspace") if hasattr(cm, "costs") else None
        if cspace_cost is not None and hasattr(cspace_cost, "_weight"):
            costs.append(cspace_cost)
    return costs


def write_cspace_target_dof_weight(hosts: Iterable, weights: torch.Tensor) -> None:
    """In-place write of ``weights`` to every rollout's cspace target weight."""
    for h in hosts:
        for c in _cspace_costs_from_rollouts(iter_rollouts(h)):
            cfg = c.config
            cfg.cspace_target_dof_weight.copy_(weights.to(cfg.cspace_target_dof_weight))


# Snapshot key uses ``id(host)``: safe within a process for the lifetime of
# the host objects, not portable across pickling. Only ever held briefly
# (one IK call or one motion-plan operator), so this is fine.
def snapshot_cspace_target_dof_weight(hosts: Iterable) -> Dict[int, List[torch.Tensor]]:
    """Per-host snapshot of the current cspace target weights."""
    return {
        id(h): [c.config.cspace_target_dof_weight.clone()
                for c in _cspace_costs_from_rollouts(iter_rollouts(h))]
        for h in hosts
    }


def restore_cspace_target_dof_weight(
    hosts: Iterable, snapshot: Dict[int, List[torch.Tensor]],
) -> None:
    """Restore each host's cspace target weights from a prior snapshot."""
    for h in hosts:
        saved_list = snapshot.get(id(h))
        if saved_list is None:
            continue
        for c, saved in zip(_cspace_costs_from_rollouts(iter_rollouts(h)), saved_list):
            cfg = c.config
            cfg.cspace_target_dof_weight.copy_(saved.to(cfg.cspace_target_dof_weight))


def build_pin_weights(
    host, pin_names: Iterable[str], pin_weight: float = 1000.0, default_weight: float = 1.0,
) -> torch.Tensor:
    """Build a per-DOF cspace target weight tensor: ``default_weight``
    everywhere, ``pin_weight`` on ``pin_names``.

    Sized to the host's active cspace (post-lock_joints); names not in the
    active cspace are silently skipped. The returned tensor is on the same
    device/dtype as the host's cspace cost weight.
    """
    costs = _cspace_costs_from_rollouts(iter_rollouts(host))
    if not costs:
        raise RuntimeError("Host has no rollouts with a cspace cost.")
    ref = costs[0].config.cspace_target_dof_weight
    active_names = list(host.kinematics.joint_names)
    weights = torch.full((len(active_names),), default_weight, device=ref.device, dtype=ref.dtype)
    name_to_idx = {n: i for i, n in enumerate(active_names)}
    for n in pin_names:
        i = name_to_idx.get(n)
        if i is not None:
            weights[i] = pin_weight
    return weights


# TODO: file upstream issue against cuRobo. RobotCostManager.compute_costs
# (cost_manager_robot.py:195-286) hardcodes evaluation of exactly four cost
# names (tool_pose, cspace, self_collision, scene_collision); register_cost
# adds to the registry but compute_costs never calls the new cost. Until
# cuRobo grows a generic dispatch loop, monkey-patch each manager so any
# cost in ``_extra_costs`` is invoked after the hardcoded set with the full
# state passed through.
def _ensure_extra_cost_dispatch(manager) -> None:
    if hasattr(manager, "_extra_costs"):
        return
    manager._extra_costs = {}
    orig = manager.compute_costs

    def wrapped(state, cost_collection=None, goal=None, **kwargs):
        cc = orig(state, cost_collection=cost_collection, goal=goal, **kwargs)
        for name, cost in manager._extra_costs.items():
            if cost.enabled:
                cc.add(cost.forward(state), name)
        return cc

    manager.compute_costs = wrapped


def add_extra_cost(host, name: str, cost) -> None:
    """Register ``cost`` on every rollout's cost_manager and metrics
    cost_manager for ``host``. Each manager is wrapped (once) so its
    ``compute_costs`` invokes any extra costs registered through this helper.

    ``host`` may be a ``MotionPlanner`` (rollouts live at
    ``host.trajopt_solver.optimizer_rollouts``) or an ``IKSolver``
    (rollouts at ``host.core.optimizer_rollouts``). ``iter_rollouts``
    handles both paths.

    NOTE: for IK solve-time costs this path is INERT — the seed-IK LM
    error calculator never consults the rollout cost managers (use the
    seed-IK CoM residual fork instead). The planner is the supported
    target."""
    for rollout in iter_rollouts(host):
        for mgr in (
            getattr(rollout, "cost_manager", None),
            getattr(rollout, "metrics_cost_manager", None),
        ):
            if mgr is None:
                continue
            _ensure_extra_cost_dispatch(mgr)
            mgr._extra_costs[name] = cost
