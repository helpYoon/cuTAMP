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
  without them; the motion planner sees one fixed call shape and uses them.
  cuRobo captures in CUDA's global mode, where another thread's allocations
  and syncs fail mid-capture, so captures
  run only on the main thread while the planner-build thread is not running
  (``motion_solver.build_motion_planner``).
"""

from typing import List

# TODO: file upstream issue against cuRobo. ``MotionPlanner.attachment_manager`` delegates to
# ``trajopt_solver.attachment_manager``, which does not exist on v0.8: the manager lives on the solver's core.
def get_attachment_manager(planner):
    """The planner's AttachmentManager, by the access path that works on v0.8."""
    return planner.trajopt_solver.core.attachment_manager


# Kept here for the upgrade audit: in cuRobo 0.8, ``result.success`` from
# plan_cspace is already the full feasibility test (TrajOptSolverResult.
# _process_metrics): last step converged on every tool frame, AND every
# constraint (scene/self collision, joint limits) satisfied over the
# optimized AND the interpolated trajectory.
# Don't second-guess it: ``result.cspace_error`` is declared but never set
# (always None), and a trajectory ending on the goal proves nothing because
# cuRobo pins the last waypoint to it. A rejected T1 plan is a real violation
# (e.g. Left_Ankle_Pitch past a joint limit mid-path); fix the planner.
def cspace_plan_succeeded(result) -> bool:
    """True iff cuRobo reports a successful (converged and feasible) plan."""
    return result is not None and bool(result.success.any())


def iter_rollouts(planner) -> List:
    """The MotionPlanner's trajopt optimizer rollouts (one per optimizer: sampling and gradient-based)."""
    return list(planner.trajopt_solver.optimizer_rollouts)


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
    """Register ``cost`` on every trajopt rollout's cost_manager and metrics
    cost_manager for the MotionPlanner ``host``. Each manager is wrapped (once)
    so its ``compute_costs`` invokes any extra costs registered through this
    helper."""
    for rollout in iter_rollouts(host):
        for mgr in (
            getattr(rollout, "cost_manager", None),
            getattr(rollout, "metrics_cost_manager", None),
        ):
            if mgr is None:
                continue
            _ensure_extra_cost_dispatch(mgr)
            mgr._extra_costs[name] = cost
