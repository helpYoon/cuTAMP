# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Everything cuTAMP changes in, or reaches past the public surface of, NVlabs cuRobo v0.9.0 (``CUROBO_COMMIT``).

Every helper and patch here fixes a cuRobo bug or works around a missing API. They are gathered in one module so a
cuRobo upgrade has exactly one place to audit. ``install_curobo_patches`` (run on ``import cutamp``) first checks that
the cuRobo functions the patches wrap, and those whose behavior cuTAMP relies on (``_PINNED_SOURCES``), still have the
v0.9.0 source they were written against, so an upgrade, or the old fork, fails loudly instead of running with
half-applied fixes.

Documented public-API constraints (no helper needed; noted here so the next
upgrade audit catches them):

* CUDA graphs can't be re-captured ("CUDA graph reset is not available."), so
  a solver whose call structure changes after its first solve must run
  without them; the motion planner sees one fixed call shape and uses them.
  cuRobo captures in CUDA's global mode, where another thread's allocations
  and syncs fail mid-capture, so captures
  run only on the main thread while the planner-build thread is not running
  (``motion_solver.build_motion_planner``).
* cspace.velocity_scale is applied twice (open PR NVlabs/curobo#729), so build_planner_model.py writes the square
  root of each cap's scale, and the planner reads joint limits back from cuRobo.
* timeopt (on by default) bounds torque on its own RNEA, without the floor's push, so get_t1_motion_planner turns it
  off and the balance gate paces each segment.
"""

import hashlib
import importlib
import inspect
import threading
from typing import Callable, List

import numpy as np
import torch

CUROBO_COMMIT = "42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83"
"""NVlabs cuRobo v0.9.0, the version the patches below were written against."""

_PINNED_SOURCES = {
    # sha1[:12] of the source at CUROBO_COMMIT of each cuRobo function the patches wrap (the first three), then of each
    # one whose behavior cuTAMP relies on: FK reads compute_com on every call (install_floor_wrench flips it); RNEA's
    # f_ext layout and reused torque buffer (set_external_wrench_fn); trajopt's rollout list (install_floor_wrench,
    # add_extra_cost); the cost manager's six hardcoded costs (_ensure_extra_cost_dispatch); the attachment fit
    # (motion_solver._attach_object).
    "curobo._src.robot.parser.parser_urdf:UrdfRobotParser.get_link_parameters": "7da2bd10430b",
    "curobo._src.transition.robot_state_transition:RobotStateTransition.compute_augmented_state": "b850477475b9",
    "curobo._src.util.warp:get_warp_device_stream": "ba0736c58155",
    "curobo._src.robot.kinematics.kinematics:Kinematics._forward": "c77db1c18ed6",
    "curobo._src.robot.dynamics.dynamics:Dynamics.compute_inverse_dynamics": "88bc59d7b638",
    "curobo._src.solver.solver_trajopt:TrajOptSolver.get_all_rollout_instances": "284a926d8db7",
    "curobo._src.rollout.cost_manager.cost_manager_robot:RobotCostManager.compute_costs": "06ce73fe202f",
    "curobo._src.collision.attachment_manager:AttachmentManager.attach_from_scene": "c11743a31b72",
}


# Kept here for the upgrade audit: ``result.success`` from plan_cspace is already
# the full feasibility test: last step converged on every tool frame, AND every
# constraint (scene/self collision, joint limits) satisfied over the optimized
# AND the interpolated trajectory (v0.9.0 solver_trajopt.py: compute_metrics_from_action
# and _interpolate_and_compute_metrics both feed the result processor) — with timeopt off,
# as get_t1_motion_planner sets it; on, timeopt's own check replaces the interpolated one.
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
# (cost_manager_robot.py) hardcodes evaluation of six cost names (tool_pose,
# tool_relative_transform, tool_velocity_limit, cspace, self_collision,
# scene_collision); register_cost adds to the registry but compute_costs never
# calls the new cost. Until cuRobo grows a generic dispatch loop, monkey-patch
# each manager so any cost in ``_extra_costs`` is invoked after the hardcoded
# set with the full state passed through.
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


def set_external_wrench_fn(transition, fn: Callable) -> None:
    """Make one ``RobotStateTransition`` run RNEA twice, the second time with the links' external wrenches.

    ``compute_augmented_state`` first runs as cuRobo wrote it (free RNEA, FK); then, under no_grad,
    ``f_ext = fn(joint_state, kinematics_state, tau_free)`` ``[B, H, n_links, 6]`` (Dynamics' layout), and the torques
    of ``compute_inverse_dynamics(joint_state, f_ext=f_ext)`` replace the free ones. Gradients flow through the second
    pass with ``f_ext`` held fixed. cuRobo has the ``f_ext`` argument but no hook for it in the transition."""
    compute = type(transition).compute_augmented_state

    def two_pass(state_seq, idxs_env=None):
        state = compute(transition, state_seq, idxs_env)
        with torch.no_grad():  # Dynamics reuses its torque buffer, so fn runs before the second pass overwrites it
            f_ext = fn(state.joint_state, state.cuda_robot_model_state, state.joint_torque.detach())
        state.joint_torque = transition.robot_dynamics.compute_inverse_dynamics(state.joint_state, f_ext=f_ext)
        return state

    transition.compute_augmented_state = two_pass
    transition._external_wrench_fn = fn


def _patch_mass_floor() -> None:
    """cuRobo's URDF parser weighs a link with no inertial, or with mass <= 0, at 0.01 kg, and scales its default
    inertia to that mass. The T1 has 11 such links (``world`` and 10 gripper links), 0.11 kg that the CoM and RNEA would
    count. Here they weigh 0, with zero inertia unless the URDF gives one; YAML extra links keep cuRobo's values."""
    from curobo._src.robot.parser.parser_urdf import UrdfRobotParser

    get_link_parameters = UrdfRobotParser.get_link_parameters

    def no_mass_floor(self, link_name: str, base=False):
        params = get_link_parameters(self, link_name, base)
        if self.extra_links and link_name in self.extra_links:
            return params
        inertial = self._robot.link_map[link_name].inertial
        if inertial is None or inertial.mass <= 0.0:
            params.link_mass = 0.0
            if inertial is None or not np.any(inertial.inertia):
                params.link_inertia = np.zeros(6)
        return params

    no_mass_floor._cutamp_patch = True
    no_mass_floor.__wrapped__ = get_link_parameters  # what _check_curobo_sources fingerprints
    UrdfRobotParser.get_link_parameters = no_mass_floor


def _patch_warp_streams() -> None:
    """One ``wp.Stream`` per CUDA stream. ``wp.stream_from_torch`` registers each new wrapper in warp's process-global
    stream map, which has no lock, and the wrapper's ``__del__`` erases it; cuRobo makes one per kernel launch
    (``get_warp_device_stream``), so the planner-build thread launching beside the main thread corrupted the heap
    (warp 1.10.1: "malloc(): unaligned tcache chunk detected", SIGSEGV). Kept wrappers are never erased mid-run."""
    import warp as wp

    stream_from_torch = wp.stream_from_torch
    streams, lock = {}, threading.Lock()

    def cached(stream_or_device=None):
        if not isinstance(stream_or_device, torch.cuda.Stream):
            return stream_from_torch(stream_or_device)
        key = (stream_or_device.device_index, stream_or_device.cuda_stream)
        stream = streams.get(key)
        if stream is None:
            with lock:
                stream = streams.get(key)
                if stream is None:
                    stream = streams[key] = stream_from_torch(stream_or_device)
        return stream

    wp.stream_from_torch = cached


def _check_curobo_sources() -> None:
    for path, digest in _PINNED_SOURCES.items():
        module, qualname = path.split(":")
        obj = importlib.import_module(module)
        for name in qualname.split("."):
            obj = getattr(obj, name)
        found = hashlib.sha1(inspect.getsource(inspect.unwrap(obj)).encode()).hexdigest()[:12]
        if found != digest:
            raise RuntimeError(
                f"cuTAMP patches NVlabs cuRobo v0.9.0 ({CUROBO_COMMIT[:7]}), and this cuRobo's {qualname} is not that "
                f"version's (source sha1 {found}, expected {digest}). Install cuRobo v0.9.0, or re-audit "
                "cutamp/_curobo_internals.py against this version and update _PINNED_SOURCES.")


def install_curobo_patches() -> None:
    """Check the cuRobo sources, then apply the warp-stream and parser patches; a no-op once applied. The parser
    patch goes last: its marker says the install finished."""
    from curobo._src.robot.parser.parser_urdf import UrdfRobotParser

    if getattr(UrdfRobotParser.get_link_parameters, "_cutamp_patch", False):
        return
    _check_curobo_sources()
    _patch_warp_streams()
    _patch_mass_floor()
