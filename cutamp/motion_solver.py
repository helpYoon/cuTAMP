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
  - Arm ops (Pick/Place/Move/Retract): ``pin_for_arm_action(arm)`` pins the
    inactive arm's cspace weights, a no-op. The base is locked at
    construction; nothing holds the inactive hand between confs.
  - Pick / Place terminals are cspace-anchored via ``_plan_arm_to_conf``:
    we plan to the gate-approved particle conf with the grasp-object /
    placement-surface obstacle temporarily disabled.
  - Retract is a cspace target pinning all joints toward the configured retract.
"""

import contextlib
import logging
import math
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


def _plan_dt(planner) -> float:
    """The time step between the positions of ``get_interpolated_plan()``: cuRobo's
    ``interpolation_dt``. Not ``js_solution.dt``, the optimizer's step (0.0145-0.0246 s),
    which labelled plans 1.02-1.54x too fast."""
    return float(planner.trajopt_solver.config.interpolation_dt)


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
        # 2.5 mm, the trajopt margin in cuRobo's lbfgs_bspline_trajopt.yml (the
        # planner's own default, 0.01, is untested here). It only shapes the
        # optimizers' collision cost: cuRobo still judges success at 0
        # (metrics_base.yml). At 0 the optimum hugs obstacles, and retracts after
        # a place grazed the block just released by up to 0.2 mm.
        collision_activation_distance=0.0025,
        enable_com_polygon=config.enable_com_polygon,
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
    gate_stats: List[dict] = []

    def add_segment(plan, dt, arm):
        # The real robot's balance check (balance arms only): every frame's CoM and ZMP inside the support
        # rectangle, slowing the segment down if its ZMP needs it; a RuntimeError fails the candidate. A slowed
        # segment also keeps its motion planner's timing, with the held objects along it.
        entry = {"dt": dt, "arm": arm}
        if config.enable_com_polygon:
            plan, stats, fast = _balance_gate(world, planner, plan, dt)
            gate_stats.append(stats)
            if fast is not None:
                entry["planner_timing"] = {"plan": fast, "held_objs": state.compute_all_held_obj_poses(fast)}
        entry.update(plan=plan, held_objs=state.compute_all_held_obj_poses(plan))
        accum_plans.append(entry)
        return plan

    obj_to_current_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}

    ts = 0.0
    visualizer.set_time_seconds("curobo", ts)
    visualizer.set_joint_positions(q_init)
    for obj, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj}", pose)

    for idx, ground_op in enumerate(plan_skeleton):
        metadata = ground_op.operator.metadata
        arm = metadata.arm

        # Pin protocol: arm ops pin the inactive arm's cspace weights (the base
        # is locked at construction). The pin is a no-op and plan_cspace holds
        # nothing between endpoint confs, so the inactive hand is not held
        # along the segment (T1State.pin_for_arm_action). Both arms stay in the
        # collision checks, except against the obstacle a pick or place
        # segment disables for the whole robot (the grasped block, or the
        # support surface; _plan_arm_to_conf).
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
                plan = add_segment(_interp_plan(result), _plan_dt(planner), arm)
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
                grasp_plan = add_segment(_interp_plan(grasp_result), _plan_dt(planner), arm)
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

                plan = add_segment(_interp_plan(place_result), _plan_dt(planner), arm)
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

    metrics = _audit_traj_com(world, planner, accum_plans)
    metrics["plan_duration_s"] = sum((e["plan"].position.shape[-2] - 1) * e["dt"] for e in accum_plans)
    if gate_stats:
        metrics["retime_max"] = max(g["retime"] for g in gate_stats)
        metrics["traj_com_min_margin"] = min(g["com_min_margin"] for g in gate_stats)
        metrics["traj_zmp_min_margin"] = min(g["zmp_min_margin"] for g in gate_stats)
    return accum_plans, metrics


def _audit_traj_com(world: TAMPWorld, planner, accum_plans: List[dict]) -> dict:
    """Measure-only trajectory CoM audit (recorded in every ablation arm): the plan's frame count,
    ``traj_com_frames``, and the fraction of them whose CoM leaves the support rectangle the gate enforces,
    ``traj_com_violation_frac``."""
    # The whole block is exception-isolated: this audit is MEASURE-ONLY
    # and must never change which plans are produced or accepted. Any
    # failure here (e.g. an unexpected DOF/joint-name mismatch) is logged
    # and swallowed, returning an empty audit dict so
    # ``overall_metrics.update(audit)`` is a no-op and the keys are simply
    # absent — downstream already reads them with ``.get(...)``.
    from cutamp.com_polygon_cost import compute_support_mask
    try:
        total_frames, violation_frames = 0, 0
        with torch.no_grad():
            for i, entry in enumerate(accum_plans):
                # A later segment's frame 0 is the previous one's last (it is planned from last_js): count it
                # once, not once per segment (known-bugs N3).
                full = _full_positions(world, planner, entry["plan"])[1 if i else 0:]
                ok = compute_support_mask(world, full)
                total_frames += int(ok.numel())
                violation_frames += int((~ok).sum().item())
        return {
            "traj_com_violation_frac": (violation_frames / total_frames) if total_frames else 0.0,
            "traj_com_frames": total_frames,
        }
    except Exception as e:
        _log.warning(f"Trajectory CoM audit failed (measure-only, ignoring): {e}")
        return {}


def _full_positions(world: TAMPWorld, planner, plan_js) -> torch.Tensor:
    """``[T, full_dof]`` positions of a planner trajectory in ``world.kinematics.joint_names`` order.

    accum_plans entries carry trajectories in different DOF forms depending on which planner method
    produced them: the planner's locked active cspace (18-DOF, missing the locked base joints) or the
    fully articulated form (31-DOF, a superset of world.kinematics's 21 joints). Only the first is
    augmented with get_full_js (on the second it raises ``lock_joints is also listed``); a plain
    reorder handles both. ``plan.joint_names`` may be None, then the planner's names are picked by
    shape, as in ``_last_timestep_js``."""
    pos = plan_js.position.reshape(-1, plan_js.position.shape[-1])
    src_names = getattr(plan_js, "joint_names", None)
    if src_names is None:
        src_names = (
            list(planner.kinematics.joint_names)
            if pos.shape[-1] == len(planner.kinematics.joint_names)
            else list(planner.kinematics.all_articulated_joint_names)
        )
    js = JointState.from_position(pos, joint_names=list(src_names))
    if pos.shape[-1] < len(world.kinematics.joint_names):
        js = planner.kinematics.get_full_js(js)
    return js.reorder(list(world.kinematics.joint_names)).position


# The slowest the balance gate will make a segment: 4x longer, 16x lower accelerations.
RETIME_MAX = 4.0
# The balance gate's central-difference step along each frame's motion, s.
ZMP_STEP = 0.01


def _taylor_positions(world: TAMPWorld, planner, plan_js, h: float = ZMP_STEP) -> torch.Tensor:
    """``[3, T, full_dof]``: each frame's configuration ``h`` s before, at and after it along the plan's own
    velocity and acceleration, ``q + tau q' + tau^2 / 2 q''`` (com_zmp_in_support's input)."""
    q, v, a = plan_js.position, plan_js.velocity, plan_js.acceleration
    return torch.stack([
        _full_positions(world, planner, JointState.from_position(q + tau * v + 0.5 * tau**2 * a,
                                                                 joint_names=plan_js.joint_names))
        for tau in (-h, 0.0, h)
    ])


def _retime(plan_js: JointState, s: float, dt: float) -> JointState:
    """``plan_js`` slowed down uniformly by about ``s``, resampled every ``dt`` s.

    A C2 cubic spline through the frames, with the native end velocities as boundary conditions,
    is sampled at ``N' = ceil(s (N - 1)) + 1`` frames, so the slowdown is ``s_eff = (N' - 1) / (N - 1)
    >= s`` and the last frame is the original's. Velocities scale by ``1 / s_eff``, accelerations by
    ``1 / s_eff**2``, jerks by ``1 / s_eff**3``."""
    from scipy.interpolate import CubicSpline

    pos = plan_js.position
    lead, dof = pos.shape[:-2], pos.shape[-1]
    q = pos.reshape(-1, dof).double().cpu().numpy()
    n = q.shape[0]
    n_new = math.ceil(s * (n - 1)) + 1
    s_eff = (n_new - 1) / (n - 1)
    if plan_js.velocity is not None:
        v = plan_js.velocity.reshape(-1, dof).double().cpu().numpy()
        bc = ((1, v[0]), (1, v[-1]))
    else:
        bc = "clamped"
    spline = CubicSpline(dt * torch.arange(n).double().numpy(), q, axis=0, bc_type=bc)
    t = (dt / s_eff * torch.arange(n_new).double()).clamp(max=dt * (n - 1)).numpy()

    def sample(k):
        x = torch.as_tensor(spline(t, k) / s_eff**k, dtype=pos.dtype, device=pos.device)
        return x.reshape(*lead, n_new, dof)

    return JointState(position=sample(0), velocity=sample(1), acceleration=sample(2), jerk=sample(3),
                      joint_names=plan_js.joint_names, dt=plan_js.dt)


def _balance_gate(world: TAMPWorld, planner, plan_js: JointState, dt: float):
    """The real robot's balance check on every interpolated frame of a segment.

    The CoM must stay inside the support rectangle (``SUPPORT_HALF_EXTENTS``); a segment whose CoM leaves it
    raises RuntimeError, so the planner moves to its next candidate or skeleton. The multibody ZMP, from
    the plan's own velocities and accelerations, must stay inside too; a segment whose ZMP leaves it is slowed down uniformly until it fits, up to
    ``RETIME_MAX``, past which it raises. Returns ``(plan_js, stats, planner_timing)``: the plan, retimed if
    needed; ``{"retime", "com_min_margin", "zmp_min_margin"}`` (metres); and, for a slowed segment, the same
    spline sampled without a slowdown: the motion planner's positions exactly, with the spline's velocities and
    accelerations (the timing test runs both), else None."""
    from cutamp.com_polygon_cost import (
        SUPPORT_HALF_EXTENTS, com_margin_from_base, com_zmp_in_support, retime_factor,
    )

    def margins(js):
        com, zmp = com_zmp_in_support(world, _taylor_positions(world, planner, js), ZMP_STEP)
        return (com, zmp, com_margin_from_base(com, SUPPORT_HALF_EXTENTS).min().item(),
                com_margin_from_base(zmp, SUPPORT_HALF_EXTENTS).min().item())

    with torch.no_grad():
        com, zmp, com_m, zmp_m = margins(plan_js)
        if com_m < 0:
            raise RuntimeError(f"trajectory CoM leaves the support region by {-100 * com_m:.1f} cm")
        out, s, tried = plan_js, 1.05 * retime_factor(com, zmp), 1.0
        while zmp_m < 0:
            if tried >= RETIME_MAX:
                raise RuntimeError(f"trajectory ZMP stays outside the support region at the {RETIME_MAX}x slowdown cap")
            tried = min(s, RETIME_MAX)
            out = _retime(plan_js, tried, dt)
            _, _, com_m, zmp_m = margins(out)
            if com_m < 0:
                raise RuntimeError(f"retimed trajectory CoM leaves the support region by {-100 * com_m:.1f} cm")
            s = 1.25 * tried
    n, n_out = plan_js.position.shape[-2], out.position.shape[-2]
    stats = {"retime": (n_out - 1) / (n - 1) if n_out != n else 1.0, "com_min_margin": com_m, "zmp_min_margin": zmp_m}
    return out, stats, None if out is plan_js else _retime(plan_js, 1.0, dt)
