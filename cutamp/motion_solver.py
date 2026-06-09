# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Motion-plan solver for cuTAMP under cuRobo v0.8 single-MotionPlanner.

Per plan-skeleton operator (T1):
  - Arm ops (Pick/Place/Move/Retract): ``pin_for_arm_action(arm)`` locks base
    + inactive arm so the active arm reaches the target without drifting.
  - MoveBaseTo (action_type="navigate"): ``pin_for_movebase()`` locks both
    arms + lift/torso so only the planar base moves.
  - Pick / Place use ``plan_single_arm_grasp`` (forked plan_grasp scoped to
    the active tool frame).
  - Retract is a cspace target pinning all joints toward the configured retract.
"""

import contextlib
import logging
from typing import List, Optional

import torch

from curobo.sphere_fit import SphereFitType
from curobo.types import GoalToolPose, JointState, Pose

from cutamp._curobo_internals import (
    cspace_plan_succeeded as _cspace_plan_succeeded,
    get_attachment_manager as _get_attachment_manager,
)
from cutamp.config import TAMPConfiguration
from cutamp.grasp_planning import plan_single_arm_grasp
from cutamp.optimize_plan import PlanContainer
from cutamp.robots.t1 import GRIPPER_CLOSED, GRIPPER_OPEN
from cutamp.t1_state import T1State
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    action_6dof_to_mat4x4,
)
from cutamp.utils.motion_diagnostics import log_cspace_failure_diagnostic
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


def _planner_js_from_full(planner, full_pos, full_joint_names):
    """Build a planner-native JointState from a full-cspace position vector.

    The mobile base is locked in the planner config (``get_t1_motion_planner``
    extends ``lock_joints`` with ``base_j_x/y/yaw``), so the planner's cspace
    is 18-DOF while particles + ``world.kinematics`` stay 21-DOF. cuRobo's
    ``get_active_js`` reorders + drops locked joints based on ``joint_names``,
    which is why we must set ``joint_names`` on the full-cspace JointState.
    """
    if full_pos.dim() == 1:
        full_pos = full_pos[None]
    full_js = JointState.from_position(full_pos, joint_names=list(full_joint_names))
    return planner.kinematics.get_active_js(full_js)


def _plan_arm_to_conf(
    planner,
    conf_full: torch.Tensor,
    last_js: JointState,
    full_joint_names,
    *,
    disable_obstacle: Optional[str],
    retries: int,
    ground_op,
):
    """Plan a collision-free trajectory whose TERMINAL is the COM-safe particle
    conf ``conf_full`` (full 21-DOF), via ``plan_cspace`` — so the executed
    arrival equals the hard-COM-checked conf instead of a free Cartesian
    redundancy solution. Mirrors the retract branch. ``disable_obstacle`` (if
    given) is temporarily removed from the world during planning, so terminal
    gripper-block / held-block-surface contact isn't rejected. Returns the
    cuRobo result; raises ``RuntimeError`` if no attempt converges.
    """
    target_js = _planner_js_from_full(planner, conf_full, full_joint_names)
    obstacle_ctx = (
        _disabled_world_obstacle(planner, disable_obstacle)
        if disable_obstacle is not None
        else contextlib.nullcontext()
    )
    result = None
    with obstacle_ctx:
        for _ in range(retries):
            result = planner.plan_cspace(
                target_js, last_js,
                max_attempts=CSPACE_MAX_ATTEMPTS,
                enable_graph_attempt=CSPACE_ENABLE_GRAPH_ATTEMPT,
            )
            if _cspace_plan_succeeded(result, target_js):
                return result
    raise RuntimeError(f"cspace-anchored plan failed for {ground_op}")


@contextlib.contextmanager
def _disabled_world_obstacle(planner, name):
    """Temporarily disable a named obstacle in the planner's scene collision.

    Used for pick (disable the block being grasped so gripper-block contact at
    the grasp pose isn't rejected) and place (disable the surface being placed
    on so the held block can sit on it). Re-enables on exit even if the body
    raises.
    """
    sc = getattr(planner, "scene_collision_checker", None)
    if sc is None or name is None:
        yield
        return
    sc.enable_obstacle(name=name, enable=False)
    try:
        yield
    finally:
        sc.enable_obstacle(name=name, enable=True)


def _interp_plan(result, interp=None, last_tstep=None):
    """Return the trimmed interpolated trajectory from a v0.8 planner result.

    cuRobo's ``interpolated_trajectory`` is padded to a fixed horizon by
    repeating the last pose. Without trimming, every trajectory's tail is
    seconds of the robot standing still — which appears as a "wait" between
    consecutive operators in playback. Mirrors v0.7 main's
    ``get_interpolated_plan()`` pattern.

    For results that own their interpolated trajectory directly, pass only
    ``result`` (uses ``result.get_interpolated_plan()`` when available).
    For ``plan_grasp`` sub-trajectories, pass ``result``,
    ``interp=approach_interpolated_trajectory``, and
    ``last_tstep=approach_interpolated_last_tstep`` (etc.) — those live on
    the parent ``GraspPlanResult`` rather than as a sub-result with its own
    ``get_interpolated_plan``.
    """
    from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
    if interp is None:
        if result is None:
            return None
        if hasattr(result, "get_interpolated_plan"):
            return result.get_interpolated_plan()
        interp = getattr(result, "interpolated_trajectory", None)
        if interp is None:
            return getattr(result, "js_solution", None)
        if last_tstep is None:
            last_tstep = getattr(result, "interpolated_last_tstep", None)
    if last_tstep is None or len(last_tstep) == 0:
        return interp
    return trim_joint_state_trajectory(interp, 0, last_tstep[0])


APPROACH_HEIGHT = 0.05

# attached_object_{left,right} extra_link names from t1_planar_base.yml.
# Each has 50 reserved sphere slots; we use only ~6 via VOXEL fit.
_ATTACH_LINK_FOR_ARM = {
    "left": "attached_object_left",
    "right": "attached_object_right",
}

# Retry budgets ported from the v0.7-era MotionGenPlanConfig that made the
# T1 blocks demo robust (commit b45a4bc — graph planner fallback). v0.8
# defaults are 5 attempts / graph at attempt 1, which is far too tight for
# constrained dual-arm trajopt.
CSPACE_MAX_ATTEMPTS = 5           # joint-space (retract / move-base): v0.8 default
CSPACE_ENABLE_GRAPH_ATTEMPT = 1   # v0.8 default: attempt 0 unseeded, attempts 1-4 graph-seeded
POSE_MAX_ATTEMPTS = 120           # Cartesian (place, grasp goalset)
POSE_ENABLE_GRAPH_ATTEMPT = 5
GRASP_RETRY = 4                   # outer retry around plan_grasp (since plan_grasp itself doesn't expose attempts)


def _plan_dt(result, default=0.05):
    """Extract the trajectory dt from a v0.8 planner result, falling back to
    ``default`` when no joint-space solution / dt is available."""
    js = getattr(result, "js_solution", None)
    return float(js.dt.item()) if js is not None and js.dt is not None else default


def _last_timestep_js(plan, planner=None) -> JointState:
    """Build a [1, active_dof] JointState from the last timestep of a trajectory.

    cuRobo trajectories arrive in different shapes (``[time, dof]``,
    ``[batch, time, dof]``, ``[1, 1, time, dof]``) and in either active-DOF
    or full-DOF form (full includes the locked joints). We want a 2D
    ``[1, active_dof]`` tensor for the next op's ``current_state``, since
    plan_cspace / plan_pose target tensors live in active-DOF space.
    """
    pos = plan.position
    while pos.dim() > 2:
        if pos.shape[0] == 1:
            pos = pos.squeeze(0)
        else:
            pos = pos[0]
    # Preserve joint_names from the source plan so get_active_js can reorder.
    js = JointState.from_position(pos[-1:], joint_names=getattr(plan, "joint_names", None))
    if planner is not None:
        active_dof = len(planner.kinematics.joint_names)
        if js.position.shape[-1] != active_dof:
            js = planner.trajopt_solver.get_active_js(js)
    return js


def _attach_object(
    planner,
    obj_name: str,
    last_js: JointState,
    link_name: str,
    num_spheres: int = 6,
):
    """Attach a scene obstacle to the gripper link via cuRobo's v0.8 attach.

    ``attach_from_scene`` does the principled thing: fits spheres for the
    obstacle, mounts them on ``link_name`` (which has reserved sphere slots
    in t1_planar_base.yml), and auto-disables the obstacle in the world
    collision checker so it isn't double-counted. The held block then
    travels with the robot in subsequent plans, retains collision against
    the rest of the world (table, other blocks), and the world-disable +
    re-enable bookkeeping is handled internally — no need for our own
    enable_obstacle hack.

    ``VOXEL`` (interior-voxel grid with inscribed radii) gives a tight,
    cube-shaped sphere set for the small block geometry — much less prone
    to over-extending into the wrist links than ``MORPHIT``'s optimization
    output. ``num_spheres=6`` is enough coverage for a 5cm cube.
    """
    am = _get_attachment_manager(planner)
    if am is None:
        _log.warning(f"attach({obj_name}) skipped: no attachment_manager")
        return
    try:
        am.attach_from_scene(
            joint_states=last_js,
            obstacle_names=[obj_name],
            link_name=link_name,
            num_spheres=num_spheres,
            sphere_fit_type=SphereFitType.VOXEL,
        )
    except Exception as e:
        _log.warning(f"attach({obj_name}) failed: {e}")


def _detach_object(planner, link_name: str):
    """Detach object from gripper link. Re-enables the world obstacle."""
    am = _get_attachment_manager(planner)
    if am is None:
        return
    try:
        am.detach(link_name=link_name)
    except Exception as e:
        _log.warning(f"detach failed: {e}")


def _reset_attachment_state(planner, world) -> None:
    """Reset gripper attachments and re-enable all movables in the scene.

    cuRobo's AttachmentManager tracks only the most recent attach (single
    ``_attached_link_name`` field). On the T1 we attach to both arms; a prior
    ``solve_curobo`` that failed mid-plan can leave stale link spheres on the
    arm whose attach was overwritten, plus a disabled world obstacle whose
    bookkeeping was lost. Force-detach both attachment links and re-enable
    every movable so the next solve starts from a clean planner state.
    """
    am = _get_attachment_manager(planner)
    if am is not None:
        for link in _ATTACH_LINK_FOR_ARM.values():
            try:
                am.detach(link_name=link)
            except Exception as e:
                _log.debug(f"detach({link}) during reset: {e}")
    sc = getattr(planner, "scene_collision_checker", None)
    if sc is not None:
        for obj in world.movables:
            try:
                sc.enable_obstacle(name=obj.name, enable=True)
            except Exception as e:
                _log.debug(f"enable_obstacle({obj.name}) during reset: {e}")


def _update_world_obstacle_pose(planner, name: str, pose_mat) -> None:
    """Update the stored world pose of an obstacle.

    Needed after place: ``attach_from_scene`` removed the obstacle from the
    world during carry, and ``detach`` re-enables it — but at its *original*
    stored pose. Calling this with the placement target tells the planner
    where the block actually lives now, so subsequent plans treat it as an
    obstacle at the new location.
    """
    sc = getattr(planner, "scene_collision_checker", None)
    if sc is None:
        return
    try:
        sc.update_obstacle_pose(name=name, w_obj_pose=Pose.from_matrix(pose_mat))
    except Exception as e:
        _log.warning(f"update_obstacle_pose({name}) failed: {e}")


def solve_curobo(
    plan_info: PlanContainer,
    best_particle: Particles,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer: Visualizer,
    timeline: str = "curobo",
):
    """Run trajopt across the plan skeleton, producing a list of trajectories."""
    plan_skeleton = plan_info["plan_skeleton"]

    # Disable CUDA graphs: solve_curobo invokes plan_pose / plan_grasp /
    # plan_cspace with varying batch+goalset shapes across operators, and
    # SolverCore.reset_cuda_graph raises "CUDA graph reset is not available."
    # CUDA graph capture would speed up identical-shape repeats, but our
    # per-operator shape changes invalidate that.
    planner = world.get_motion_planner(
        collision_activation_distance=config.world_activation_distance,
        use_cuda_graph=False,
        enable_com_polygon=config.enable_com_polygon,
    )

    # Defensive reset in case a prior solve_curobo aborted mid-plan and left the
    # planner with stale attached objects or disabled world obstacles. Required
    # by the outer-retry loop in algorithm.py — without this, a failure on the
    # best particle leaves attached spheres on an arm link and a disabled
    # movable, both of which corrupt the retry attempt.
    _reset_attachment_state(planner, world)

    # Build initial state. Planner is in active-cspace (mobile base locked);
    # particles are in full-cspace — convert at every boundary.
    full_joint_names = world.kinematics.joint_names
    last_js = _planner_js_from_full(
        planner, best_particle["left_q0"].clone(), full_joint_names,
    )
    state = T1State(
        planner=planner,
        kinematics=world.kinematics,
        tool_from_ee=world.tool_from_ee,
        current_js=last_js,
    )
    last_q_name = "left_q0"
    q_init = best_particle["left_q0"]

    if config.warmup_motion_gen:
        with timer.time("curobo_motion_gen_warmup", log_callback=_log.debug):
            planner.warmup()

    accum_plans: List[dict] = []
    obj_to_current_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}

    ts = 0.0
    visualizer.set_time_seconds(timeline, ts)
    visualizer.set_joint_positions(q_init)
    for obj, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj}", pose)

    def _active_tool_frame(arm: Optional[str]) -> str:
        return state.get_tool_frame(arm or "left")

    def _tool_from_ee(arm: Optional[str]) -> torch.Tensor:
        return world.tool_from_ee[_active_tool_frame(arm)]

    for idx, ground_op in enumerate(plan_skeleton):
        metadata = ground_op.operator.metadata
        arm = metadata.arm
        active_tool = _active_tool_frame(arm)
        tool_from_ee_mat = _tool_from_ee(arm)

        # Pin protocol: navigate locks body, arm ops lock base+inactive arm.
        if metadata.action_type == "navigate":
            state.pin_for_movebase()
        elif arm is not None:
            state.pin_for_arm_action(arm)

        try:
            if metadata.action_type == "navigate":
                # MoveBaseTo(obj, q_start, traj, q_end). Plan a cspace trajectory
                # to q_end while body cspace weights pin everything except base.
                # WARNING: this planner instance has the mobile base locked, so
                # navigate will not actually move the base. Envs that need
                # MoveBaseTo must build a second planner without the base lock
                # and dispatch by operator type. blocks_t1 has no navigate.
                q_end_name = ground_op.values[-1]
                target_q = best_particle[q_end_name].clone()
                with timer.time("curobo_planning"):
                    target_js = _planner_js_from_full(planner, target_q, full_joint_names)
                    result = planner.plan_cspace(
                        target_js, last_js,
                        max_attempts=CSPACE_MAX_ATTEMPTS,
                        enable_graph_attempt=CSPACE_ENABLE_GRAPH_ATTEMPT,
                    )
                if not _cspace_plan_succeeded(result, target_js):
                    raise RuntimeError(f"MoveBaseTo plan failed for {ground_op}")
                plan = _interp_plan(result)
                dt = _plan_dt(result)
                accum_plans.append({
                    "type": "trajectory", "plan": plan, "dt": dt, "arm": None,
                    "held_objs": state.compute_all_held_obj_poses(plan),
                })
                last_js = _last_timestep_js(plan, planner)
                state.current_js = last_js
                last_q_name = q_end_name

            elif metadata.is_motion and metadata.action_type is None:
                # MoveFree / MoveHolding — no trajectory; pick/place chains
                # from this last_q_name as their start state.
                q_start = ground_op.values[-3] if len(ground_op.values) == 5 else ground_op.values[0]
                last_q_name = q_start

            elif metadata.action_type == "retract":
                q_retract_name = ground_op.values[-1]
                target_q = best_particle[q_retract_name].clone()
                with timer.time("curobo_planning"):
                    target_js = _planner_js_from_full(planner, target_q, full_joint_names)
                    result = planner.plan_cspace(
                        target_js, last_js,
                        max_attempts=CSPACE_MAX_ATTEMPTS,
                        enable_graph_attempt=CSPACE_ENABLE_GRAPH_ATTEMPT,
                    )
                if not _cspace_plan_succeeded(result, target_js):
                    if config.debug_motion_failures:
                        log_cspace_failure_diagnostic(
                            planner, last_js, target_js, result, ground_op, _log,
                        )
                    raise RuntimeError(f"Retract plan failed for {ground_op}")
                plan = _interp_plan(result)
                dt = _plan_dt(result)

                accum_plans.append({
                    "type": "trajectory", "plan": plan, "dt": dt, "arm": arm,
                    "held_objs": state.compute_all_held_obj_poses(plan),
                })
                last_js = _last_timestep_js(plan, planner)
                state.current_js = last_js

            elif metadata.action_type == "pick":
                obj_name, grasp_name, q_name = ground_op.values
                obj_from_grasp = (
                    action_4dof_to_mat4x4 if config.grasp_dof == 4 else action_6dof_to_mat4x4
                )(best_particle[grasp_name].clone())

                # Anchor the grasp TERMINAL to the COM-safe particle conf
                # (best_particle[q_name]) via plan_cspace instead of a free
                # Cartesian-pose redundancy solve. The conf is an exact IK
                # solution for the grasp pose, so the gripper still arrives at
                # the grasp, with a COM-safe posture. Disable the target block
                # obstacle so gripper-block contact at the grasp isn't rejected;
                # gripper-table / gripper-other-block collision stays enforced.
                with timer.time("curobo_planning"):
                    grasp_result = _plan_arm_to_conf(
                        planner, best_particle[q_name].clone(), last_js,
                        full_joint_names,
                        disable_obstacle=obj_name, retries=GRASP_RETRY, ground_op=ground_op,
                    )

                # Attach AT the grasp conf. Block tracking from this point uses
                # inverse(obj_from_grasp) since the gripper is at the intended
                # grasp pose.
                grasp_plan = _interp_plan(grasp_result)
                dt = _plan_dt(grasp_result)
                accum_plans.append({
                    "type": "trajectory", "plan": grasp_plan, "dt": dt, "arm": arm,
                    "held_objs": state.compute_all_held_obj_poses(grasp_plan),
                })
                last_js = _last_timestep_js(grasp_plan, planner)

                attach_link = _ATTACH_LINK_FOR_ARM[arm]
                _attach_object(planner, obj_name, last_js, link_name=attach_link)
                grasp_from_obj = torch.inverse(obj_from_grasp).clone()
                state.current_js = last_js
                state.arm_holding[arm] = obj_name
                state.arm_grasp_transform[arm] = grasp_from_obj
                last_q_name = ground_op.values[-1]

            elif metadata.action_type == "place":
                obj_name, grasp_name, place_name, surface_name, q_name = ground_op.values
                world_from_obj_target = action_4dof_to_mat4x4(best_particle[place_name].clone())

                # Anchor the place TERMINAL to the COM-safe particle conf
                # (best_particle[q_name]) via plan_cspace, instead of a free
                # Cartesian-pose redundancy solve. The conf is an exact IK
                # solution for the same place hand pose but sits inside the COM
                # support hull (hard-checked), so the executed arrival is
                # COM-feasible. The held block sits ON the placement surface at
                # the terminal, so the surface obstacle is disabled during
                # planning (gripper stays collision-checked against everything
                # else).
                with timer.time("curobo_planning"):
                    place_result = _plan_arm_to_conf(
                        planner, best_particle[q_name].clone(), last_js,
                        full_joint_names,
                        disable_obstacle=surface_name, retries=1, ground_op=ground_op,
                    )

                plan = _interp_plan(place_result)
                dt = _plan_dt(place_result)

                accum_plans.append({
                    "type": "trajectory", "plan": plan, "dt": dt, "arm": arm,
                    "held_objs": state.compute_all_held_obj_poses(plan),
                })
                last_js = _last_timestep_js(plan, planner)
                obj_to_current_pose[obj_name] = world_from_obj_target

                state.current_js = last_js
                state.arm_holding[arm] = None
                state.arm_grasp_transform[arm] = None
                # Detach: resets the gripper link's spheres and re-enables
                # the world obstacle. But the obstacle's stored pose is
                # still the original pickup pose, so update it to the
                # placement target before continuing.
                attach_link = _ATTACH_LINK_FOR_ARM[arm]
                _detach_object(planner, link_name=attach_link)
                _update_world_obstacle_pose(planner, obj_name, world_from_obj_target)
                last_q_name = ground_op.values[-1]

            elif metadata.action_type in ("push", "push_stick"):
                raise NotImplementedError("Push operations not yet supported in v0.8 motion planning")

            else:
                raise NotImplementedError(f"Unsupported operator {ground_op.operator.name}")

        finally:
            if metadata.action_type == "navigate" or arm is not None:
                state.unpin()

        _log.info(f"{idx + 1}. {ground_op.name}")

    # Play back the accumulated trajectories on the visualizer's `timeline`
    # axis. Each entry's `plan.position` is a JointState in cuRobo full-DOF
    # form ([..., time, full_dof]); we squeeze leading length-1 dims, convert
    # to active DOF (which T1RerunRobot maps to URDF joints internally), and
    # log frame-by-frame so Rerun shows the leg/arms animating.
    #
    rr_robot = getattr(visualizer, "robot", None)
    grippers = {"left": list(GRIPPER_OPEN), "right": list(GRIPPER_OPEN)}

    def _set_gripper(arm_name, closed):
        if rr_robot is None or not hasattr(rr_robot, "set_grippers"):
            return
        grippers[arm_name] = list(GRIPPER_CLOSED if closed else GRIPPER_OPEN)
        rr_robot.set_grippers(left=grippers["left"], right=grippers["right"])

    cur_t = ts
    for entry in accum_plans:
        if entry.get("type") != "trajectory":
            continue
        plan = entry["plan"]
        dt = entry.get("dt", 0.05)
        traj_arm = entry.get("arm")
        held_objs = entry.get("held_objs", {})
        pos = plan.position
        while pos.dim() > 2:
            pos = pos.squeeze(0) if pos.shape[0] == 1 else pos[0]
        # Reproject to JOINT_NAMES_FULL (21-DOF) so T1RerunRobot can splice
        # in the locked base DOFs. Planner trajectories arrive in either
        # active cspace (18-DOF) or fully articulated form (31-DOF including
        # head + grippers) depending on which planner method produced them.
        full_dof = len(full_joint_names)
        if pos.shape[-1] != full_dof:
            src_names = getattr(plan, "joint_names", None)
            if src_names is None:
                src_names = (
                    list(planner.kinematics.joint_names)
                    if pos.shape[-1] == len(planner.kinematics.joint_names)
                    else list(planner.kinematics.all_articulated_joint_names)
                )
            traj_js = JointState.from_position(pos, joint_names=src_names)
            if pos.shape[-1] < full_dof:
                traj_js = planner.kinematics.get_full_js(traj_js)
            pos = traj_js.reorder(list(full_joint_names)).position

        if traj_arm in ("left", "right"):
            _set_gripper(traj_arm, traj_arm in held_objs)

        mat4x4s = {f"world/{obj_name}": poses for obj_name, poses in held_objs.values()}
        if mat4x4s:
            visualizer.log_joint_trajectory_with_mat4x4s(
                traj=pos, mat4x4s=mat4x4s, timeline=timeline,
                start_time=cur_t, dt=dt, arm=traj_arm,
            )
        else:
            visualizer.log_joint_trajectory(
                traj=pos, timeline=timeline, start_time=cur_t, dt=dt, arm=traj_arm,
            )
        cur_t += pos.shape[0] * dt

    return accum_plans
