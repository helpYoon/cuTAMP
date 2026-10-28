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
  - Pick / Place terminals are cspace-anchored via ``_plan_arm_to_conf``:
    we plan to the gate-approved particle conf with the grasp-object /
    placement-surface obstacle temporarily disabled.
  - Retract is a cspace target pinning all joints toward the configured retract.
"""

import contextlib
import logging
from typing import List, Optional

import torch

from curobo.sphere_fit import SphereFitType
from curobo.types import JointState, Pose

from cutamp._curobo_internals import (
    cspace_plan_succeeded as _cspace_plan_succeeded,
    get_attachment_manager as _get_attachment_manager,
)
from cutamp.config import TAMPConfiguration
from cutamp.optimize_plan import PlanContainer
from cutamp.robots.t1 import GRIPPER_CLOSED, GRIPPER_OPEN, TOOL_FRAME_FOR_ARM
from cutamp.t1_state import T1State
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
)
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
    ground_op,
):
    """Plan a collision-free trajectory whose TERMINAL is the COM-safe particle
    conf ``conf_full`` (full 21-DOF), via ``plan_cspace`` — so the executed
    arrival equals the hard-COM-checked conf instead of a free Cartesian
    redundancy solution. Mirrors the retract branch. ``disable_obstacle`` (if
    given) is temporarily removed from the world during planning, so terminal
    gripper-block / held-block-surface contact isn't rejected. Returns the
    cuRobo result; raises ``RuntimeError`` if it fails.

    One call: plan_cspace already makes 5 attempts (4 graph-seeded), and
    calling it again with the same start and goal returns the same trajectory
    (blocks_t1: 54 of 54 repeats bit-identical, none succeeded).
    """
    target_js = _planner_js_from_full(planner, conf_full, full_joint_names)
    # _disabled_world_obstacle no-ops on name=None.
    with _disabled_world_obstacle(planner, disable_obstacle):
        result = planner.plan_cspace(target_js, last_js)
    if not _cspace_plan_succeeded(result):
        raise RuntimeError(f"cspace-anchored plan failed for {ground_op}")
    return result


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


def _interp_plan(result):
    """Return the trimmed interpolated trajectory from a v0.8 planner result.

    cuRobo's ``interpolated_trajectory`` is padded to a fixed horizon by
    repeating the last pose; ``get_interpolated_plan`` trims that tail so
    playback doesn't show a "wait" between consecutive operators.
    """
    return None if result is None else result.get_interpolated_plan()


# attached_object_{left,right} extra_link names from t1_planar_base.yml.
# Each has 50 reserved sphere slots; we use only ~6 via VOXEL fit.
_ATTACH_LINK_FOR_ARM = {
    "left": "attached_object_left",
    "right": "attached_object_right",
}


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
    arm: str,
    num_spheres: int = 6,
):
    """Attach a scene obstacle to ``arm``'s attach link via cuRobo's v0.8 attach.

    ``attach_from_scene`` fits spheres for the obstacle, mounts them on the
    attach link (which has reserved sphere slots in t1_planar_base.yml), and
    auto-disables the obstacle in the world collision checker so it isn't
    double-counted; ``_detach_object`` undoes both. The held block then
    travels with the robot in subsequent plans and keeps collision against the
    rest of the world (table, other blocks).

    cuRobo fits the spheres to the mesh posed in WORLD frame but writes those
    centers unchanged as link-local offsets, so FK would put the held block
    ~0.5 m off the hand. We rewrite them as ``inv(world_from_link) @ p`` at
    the attach conf. (cuRobo's ``world_objects_pose_offset`` doesn't fix it:
    it resolves the link as ``tool_frames[0]``, the left hand, for both arms.)

    ``VOXEL`` (interior-voxel grid with inscribed radii) gives a tight,
    cube-shaped sphere set for the small block geometry — much less prone
    to over-extending into the wrist links than ``MORPHIT``'s optimization
    output. ``num_spheres=6`` is enough coverage for a 5cm cube.
    """
    am = _get_attachment_manager(planner)
    if am is None:
        _log.warning(f"attach({obj_name}) skipped: no attachment_manager")
        return
    link_name = _ATTACH_LINK_FOR_ARM[arm]
    try:
        am.attach_from_scene(
            joint_states=last_js,
            obstacle_names=[obj_name],
            link_name=link_name,
            num_spheres=num_spheres,
            sphere_fit_type=SphereFitType.VOXEL,
        )
        # The attach link is a FIXED identity child of the arm's tool frame
        # (t1_planar_base.yml), so the tool-frame pose is the link pose.
        world_from_link = planner.kinematics.compute_kinematics(last_js).tool_poses.get_link_pose(
            TOOL_FRAME_FOR_ARM[arm]
        )
        kp = am.kinematics_params
        idx = kp.get_sphere_index_from_link_name(link_name)
        idx = idx[kp.link_spheres[0, idx, 3] > 0]  # fitted spheres, not the radius -100 padding
        kp.link_spheres[0, idx, :3] = world_from_link.inverse().transform_points(
            kp.link_spheres[0, idx, :3]
        ).view(-1, 3)
    except Exception as e:
        _log.warning(f"attach({obj_name}) failed: {e}")


def _detach_object(planner, obj_name: str, arm: str):
    """Detach ``obj_name`` from ``arm``'s attach link and re-enable it as a
    world obstacle.

    cuRobo's own re-enable breaks with two held blocks: it re-enables the most
    recent attach's obstacles, which can be the other arm's block, and after
    one detach its env count is 0, so a named re-enable then does nothing.
    Naming ``obj_name`` stops the stale re-enable; we enable it ourselves.
    """
    am = _get_attachment_manager(planner)
    if am is None:
        return
    try:
        am.detach(link_name=_ATTACH_LINK_FOR_ARM[arm], enable_obstacle_names=[obj_name])
        planner.scene_collision_checker.enable_obstacle(name=obj_name, enable=True)
    except Exception as e:
        _log.warning(f"detach failed: {e}")


def _reset_attachment_state(planner, world) -> None:
    """Reset gripper attachments and put every movable back, enabled, at its
    original pose.

    cuRobo's AttachmentManager tracks only the most recent attach (single
    ``_attached_link_name`` field). On the T1 we attach to both arms; a prior
    ``solve_curobo`` that failed mid-plan can leave stale link spheres on the
    arm whose attach was overwritten, a disabled world obstacle whose
    bookkeeping was lost, and blocks its Places moved to the failed particle's
    placements (``_update_world_obstacle_pose``). Force-detach both attachment
    links and restore every movable so the next solve starts from a clean
    planner state.
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
            _update_world_obstacle_pose(planner, obj.name, world.get_object_pose(obj))


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


def build_motion_planner(world: TAMPWorld, config: TAMPConfiguration):
    """Build the trajopt planner, with CUDA graphs. Safe to run on a background
    thread concurrently with Phase 1 because it runs no solves.

    No ``planner.warmup()``: its solves would capture CUDA graphs on this
    thread, and cuRobo captures in CUDA's global mode, where the main thread's
    allocations and syncs fail mid-capture. The first ``plan_cspace`` captures
    instead (~1 s), on the main thread after the join; every later segment has
    the same shape and replays.
    """
    return world.get_motion_planner(
        collision_activation_distance=0.0,
        enable_com_polygon=config.enable_com_polygon,
        enable_joint_limit_margin=config.enable_joint_limit_margin,
    )


def solve_curobo(
    plan_info: PlanContainer,
    best_particle: Particles,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer: Visualizer,
    get_planner=None,
):
    """Run trajopt across the plan skeleton, producing a list of trajectories.

    ``get_planner`` (optional) returns a pre-built, warmed planner — the
    overlap path: algorithm.py starts ``build_motion_planner`` on a thread
    when the optimization loop starts, and the timer below then measures
    only the residual wait. ``None`` builds+warms inline.
    """
    plan_skeleton = plan_info["plan_skeleton"]

    with timer.time("curobo_motion_gen_warmup", log_callback=_log.debug):
        planner = get_planner() if get_planner is not None else build_motion_planner(world, config)

    # Defensive reset in case a prior solve_curobo aborted mid-plan and left the
    # planner with stale attached objects, disabled world obstacles, or placed
    # blocks. Required by the outer-retry loop in algorithm.py — without this,
    # a failure on the best particle leaves attached spheres on an arm link, a
    # disabled movable, and blocks at its placements, all of which corrupt the
    # retry attempt.
    _reset_attachment_state(planner, world)

    # Build initial state. Planner is in active-cspace (mobile base locked);
    # particles are in full-cspace — convert at every boundary.
    full_joint_names = world.kinematics.joint_names
    last_js = _planner_js_from_full(
        planner, best_particle["left_q0"].clone(), full_joint_names,
    )
    state = T1State(
        planner=planner,
        tool_from_ee=world.tool_from_ee,
    )
    q_init = best_particle["left_q0"]

    accum_plans: List[dict] = []
    obj_to_current_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}

    ts = 0.0
    visualizer.set_time_seconds("curobo", ts)
    visualizer.set_joint_positions(q_init)
    for obj, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj}", pose)

    for idx, ground_op in enumerate(plan_skeleton):
        metadata = ground_op.operator.metadata
        arm = metadata.arm

        # Pin protocol: arm ops lock base+inactive arm.
        if arm is not None:
            state.pin_for_arm_action(arm)

        try:
            if metadata.is_motion and metadata.action_type is None:
                # MoveFree / MoveHolding — no trajectory; pick/place chain
                # from the previous trajectory's terminal state via last_js.
                pass

            elif metadata.action_type == "retract":
                q_retract_name = ground_op.values[-1]
                target_q = best_particle[q_retract_name].clone()
                with timer.time("curobo_planning"):
                    target_js = _planner_js_from_full(planner, target_q, full_joint_names)
                    result = planner.plan_cspace(
                        target_js, last_js,
                    )
                if not _cspace_plan_succeeded(result):
                    raise RuntimeError(f"Retract plan failed for {ground_op}")
                plan = _interp_plan(result)
                dt = _plan_dt(result)

                accum_plans.append({
                    "plan": plan, "dt": dt, "arm": arm,
                    "held_objs": state.compute_all_held_obj_poses(plan),
                })
                last_js = _last_timestep_js(plan, planner)
                state.current_js = last_js

            elif metadata.action_type == "pick":
                obj_name, grasp_name, q_name = ground_op.values
                obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp_name].clone())

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
                        disable_obstacle=obj_name, ground_op=ground_op,
                    )

                # Attach AT the grasp conf. Block tracking from this point uses
                # inverse(obj_from_grasp) since the gripper is at the intended
                # grasp pose.
                grasp_plan = _interp_plan(grasp_result)
                dt = _plan_dt(grasp_result)
                accum_plans.append({
                    "plan": grasp_plan, "dt": dt, "arm": arm,
                    "held_objs": state.compute_all_held_obj_poses(grasp_plan),
                })
                last_js = _last_timestep_js(grasp_plan, planner)

                _attach_object(planner, obj_name, last_js, arm)
                grasp_from_obj = torch.inverse(obj_from_grasp).clone()
                state.current_js = last_js
                state.arm_holding[arm] = obj_name
                state.arm_grasp_transform[arm] = grasp_from_obj

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
                        disable_obstacle=surface_name, ground_op=ground_op,
                    )

                plan = _interp_plan(place_result)
                dt = _plan_dt(place_result)

                accum_plans.append({
                    "plan": plan, "dt": dt, "arm": arm,
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
                _detach_object(planner, obj_name, arm)
                _update_world_obstacle_pose(planner, obj_name, world_from_obj_target)

            else:
                raise NotImplementedError(f"Unsupported operator {ground_op.operator.name}")

        except RuntimeError as e:
            e.op_idx = idx  # lets _motion_plan_with_retries skip candidates that repeat ops 0..idx
            raise
        finally:
            if arm is not None:
                state.unpin()

        _log.info(f"{idx + 1}. {ground_op.name}")

    # Play back the accumulated trajectories on the visualizer's "curobo"
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
                traj=pos, mat4x4s=mat4x4s, timeline="curobo",
                start_time=cur_t, dt=dt,
            )
        else:
            visualizer.log_joint_trajectory(
                traj=pos, timeline="curobo", start_time=cur_t, dt=dt,
            )
        cur_t += pos.shape[0] * dt

    # Measure-only trajectory CoM audit (recorded in every ablation arm).
    #
    # Deviates from the task brief's illustrative snippet in two respects:
    #
    # 1. accum_plans entries carry trajectories in different DOF forms
    #    depending on which planner method produced them — either the
    #    planner's locked active-cspace (18-DOF, missing the locked base
    #    joints) or the fully articulated form (31-DOF, a superset of
    #    world.kinematics's 21 joints that already includes the "locked"
    #    base joints). Unconditionally calling get_full_js (as the brief's
    #    snippet does) raises ``ValueError: lock_joints is also listed in
    #    js.joint_names`` on the latter case. Mirror the same conditional
    #    the playback loop just above already uses for this exact DOF-form
    #    ambiguity: only augment with get_full_js when the trajectory is
    #    short the locked joints; a plain reorder (index-select) handles
    #    the already-fully-articulated case. That loop also defends against
    #    ``plan.joint_names`` being None (falling back to the planner's
    #    active or fully-articulated joint names by shape) — mirrored below
    #    too, matching the same defensive pattern used in
    #    ``_last_timestep_js``.
    #
    # 2. The whole block is exception-isolated: this audit is MEASURE-ONLY
    #    and must never change which plans are produced or accepted. Any
    #    failure here (e.g. an unexpected DOF/joint-name mismatch) is logged
    #    and swallowed, returning an empty audit dict so
    #    ``overall_metrics.update(audit)`` is a no-op and the keys are simply
    #    absent — downstream already reads them with ``.get(...)``.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    audit_full_dof = len(world.kinematics.joint_names)
    try:
        total_frames, violation_frames = 0, 0
        with torch.no_grad():
            for entry in accum_plans:
                plan_js = entry["plan"]
                pos = plan_js.position.reshape(-1, plan_js.position.shape[-1])
                src_names = getattr(plan_js, "joint_names", None)
                if src_names is None:
                    src_names = (
                        list(planner.kinematics.joint_names)
                        if pos.shape[-1] == len(planner.kinematics.joint_names)
                        else list(planner.kinematics.all_articulated_joint_names)
                    )
                else:
                    src_names = list(src_names)
                js = JointState.from_position(pos, joint_names=src_names)
                if pos.shape[-1] < audit_full_dof:
                    js = planner.kinematics.get_full_js(js)
                full = js.reorder(list(world.kinematics.joint_names)).position
                ok = compute_com_polygon_mask(world, full)
                total_frames += int(ok.numel())
                violation_frames += int((~ok).sum().item())
        audit = {
            "traj_com_violation_frac": (violation_frames / total_frames) if total_frames else 0.0,
            "traj_com_frames": total_frames,
        }
    except Exception as e:
        _log.warning(f"Trajectory CoM audit failed (measure-only, ignoring): {e}")
        audit = {}
    return accum_plans, audit
