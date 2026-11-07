# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Motion-plan solver for cuTAMP under cuRobo's single MotionPlanner.

Per plan-skeleton operator (T1):
  - Arm ops (Pick/Place/Retract): every joint is free, the base included, and
    trajopt holds the soles on the stance (``get_t1_motion_planner``); nothing
    holds the inactive hand between confs.
  - Pick / Place terminals are cspace-anchored via ``_plan_arm_to_conf``:
    we plan to the gate-approved particle conf with the grasp-object /
    placement-surface obstacle temporarily disabled.
  - Retract is a cspace target pinning all joints toward the configured retract.
"""

import collections
import contextlib
import logging
import math
from typing import List, Optional, Tuple

import torch

from curobo.motion_planner import MotionPlanner
from curobo.sphere_fit import SphereFitType
from curobo.types import JointState, Pose

from cutamp.com_polygon_cost import soles_planted
from cutamp.config import TAMPConfiguration
from cutamp.optimize_plan import PlanContainer
from cutamp.robots.t1 import GRIPPER_CLOSED, GRIPPER_OPEN, TOOL_FRAME_FOR_ARM, get_t1_motion_planner
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.common import Particles, action_4dof_to_mat4x4, get_world_cfg
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


def _cspace_plan_succeeded(result) -> bool:
    """True iff ``plan_cspace`` returned a plan cuRobo calls converged and feasible over the optimized and the
    interpolated trajectory (timeopt off, as get_t1_motion_planner sets it). Trust it: ``cspace_error`` is never set,
    and the last waypoint is pinned to the goal, so neither shows a mid-path violation."""
    return result is not None and bool(result.success.any())


def _planner_js_from_full(planner, full_pos, full_joint_names):
    """Build a planner-native JointState from a full-cspace position vector.

    cuRobo's ``get_active_js`` reorders + drops locked joints based on
    ``joint_names``, which is why we must set ``joint_names`` on the
    full-cspace JointState (the model's 33 joints).
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
    world: Optional[TAMPWorld] = None,
):
    """Plan a collision-free trajectory whose TERMINAL is the COM-safe particle
    conf ``conf_full`` (the model's full cspace), via ``plan_cspace`` — so the executed
    arrival equals the hard-COM-checked conf instead of a free Cartesian
    redundancy solution. Mirrors the retract branch. ``disable_obstacle`` (if
    given) is temporarily removed from the world during planning, so terminal
    gripper-block / held-block-surface contact isn't rejected. Returns the
    cuRobo result; raises ``RuntimeError`` if it fails (``_plan_failed``: given
    ``world`` it names the reason).

    One call: plan_cspace already makes 5 attempts (4 graph-seeded), and
    calling it again with the same start and goal returns the same trajectory
    (blocks_t1: 54 of 54 repeats bit-identical, none succeeded).
    """
    target_js = _planner_js_from_full(planner, conf_full, full_joint_names)
    # _disabled_world_obstacle no-ops on name=None.
    with _disabled_world_obstacle(planner, disable_obstacle):
        result = planner.plan_cspace(target_js, last_js)
    if not _cspace_plan_succeeded(result):
        raise _plan_failed(world, result, f"cspace-anchored plan failed for {ground_op}")
    return result


def _failure_reason(world: TAMPWorld, result) -> str:
    """Why a trajopt failed (design 7.4a): ``"torque"`` when either trajectory ``result`` carries passes a
    joint's effort by cuRobo's two RNEA passes with the floor's push (``world.segment_torques``), else ``"other"``.
    The two are the trajopt horizon (``js_solution``) and the interpolated trajectory between its frames, which
    trajopt's success test re-checks (``get_interpolated_plan``). IK and Adam never see torque, so a conf can need
    more than a motor gives."""
    torques = world.segment_torques

    def over(js) -> bool:
        return js is not None and bool((torques(js).abs() > torques.effort).any())

    return "torque" if over(getattr(result, "js_solution", None)) or over(_interp_plan(result)) else "other"


def _plan_failed(world: Optional[TAMPWorld], result, message: str) -> RuntimeError:
    """The RuntimeError for a failed trajopt; given ``world`` its message ends with the reason
    (``_failure_reason``), which ``_motion_plan_with_retries`` keeps per candidate."""
    if world is not None:
        message = f"{message}: {_failure_reason(world, result)}"
    return RuntimeError(message)


@contextlib.contextmanager
def _disabled_world_obstacle(planner, name):
    """Temporarily disable a named obstacle in the planner's scene collision.

    Used for pick (disable the block being grasped so gripper-block contact at
    the grasp pose isn't rejected) and place (disable the surface being placed
    on so the held block can sit on it). Re-enables on exit even if the body
    raises.
    """
    if name is None:
        yield
        return
    sc = planner.scene_collision_checker
    sc.enable_obstacle(name=name, enable=False)
    try:
        yield
    finally:
        sc.enable_obstacle(name=name, enable=True)


def _interp_plan(result):
    """Return the trimmed interpolated trajectory from a planner result.

    cuRobo's ``interpolated_trajectory`` is padded to a fixed horizon by
    repeating the last pose; ``get_interpolated_plan`` trims that tail so
    playback doesn't show a "wait" between consecutive operators.
    """
    return None if result is None else result.get_interpolated_plan()


# attached_object_{left,right} extra_link names from t1_29dof.yml. Each has build_planner_model.py's
# HELD_OBJECT_SPHERES sphere slots, which _attach_object fills.
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


def _attach_object(planner, obj_name: str, last_js: JointState, arm: str):
    """Attach a scene obstacle to ``arm``'s attach link via cuRobo's attach.

    ``attach_from_scene`` fits spheres for the obstacle, mounts them on the
    attach link (which has reserved sphere slots in t1_29dof.yml), and
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
    output. It fits as many spheres as the link has slots: build_planner_model.py's
    ``HELD_OBJECT_SPHERES``, 6, enough coverage for a 5 cm cube.
    """
    am = planner.attachment_manager
    link_name = _ATTACH_LINK_FOR_ARM[arm]
    kp = am.kinematics_params
    idx = kp.get_sphere_index_from_link_name(link_name)
    am.attach_from_scene(
        joint_states=last_js,
        obstacle_names=[obj_name],
        link_name=link_name,
        num_spheres=len(idx),
        sphere_fit_type=SphereFitType.VOXEL,
    )
    # The attach link is a FIXED identity child of the arm's tool frame
    # (t1_29dof.yml), so the tool-frame pose is the link pose.
    world_from_link = planner.kinematics.compute_kinematics(last_js).tool_poses.get_link_pose(TOOL_FRAME_FOR_ARM[arm])
    idx = idx[kp.link_spheres[0, idx, 3] > 0]  # fitted spheres, not the radius -100 padding
    kp.link_spheres[0, idx, :3] = world_from_link.inverse().transform_points(kp.link_spheres[0, idx, :3]).view(-1, 3)


def _detach_object(planner, obj_name: str, arm: str):
    """Detach ``obj_name`` from ``arm``'s attach link and re-enable it as a
    world obstacle.

    cuRobo's own re-enable breaks with two held blocks: it re-enables the most
    recent attach's obstacles, which can be the other arm's block, and after
    one detach its env count is 0, so a named re-enable then does nothing.
    Naming ``obj_name`` stops the stale re-enable; we enable it ourselves.
    """
    planner.attachment_manager.detach(link_name=_ATTACH_LINK_FOR_ARM[arm], enable_obstacle_names=[obj_name])
    planner.scene_collision_checker.enable_obstacle(name=obj_name, enable=True)


def _reset_attachment_state(planner, world) -> None:
    """Reset gripper attachments and put every movable back, enabled, at its
    original pose.

    cuRobo's AttachmentManager tracks only the most recent attach (single
    ``_attached_link_name`` field). On the T1 we attach to both arms; a prior
    ``solve_curobo`` that failed mid-plan can leave stale link spheres on the
    arm whose attach was overwritten, a disabled world obstacle whose
    bookkeeping was lost, and blocks its Places moved to the failed particle's
    placements. Force-detach both attachment links and restore every movable
    so the next solve starts from a clean planner state.
    """
    am, sc = planner.attachment_manager, planner.scene_collision_checker
    for link in _ATTACH_LINK_FOR_ARM.values():
        am.detach(link_name=link)
    for obj in world.movables:
        sc.enable_obstacle(name=obj.name, enable=True)
        sc.update_obstacle_pose(name=obj.name, w_obj_pose=Pose.from_matrix(world.get_object_pose(obj)))


def _held_obj_poses(planner, tool_from_ee, held: dict, plan: JointState) -> dict:
    """``{arm: (obj_name, world_from_obj per timestep)}`` along ``plan`` for every arm in ``held``.

    ``held`` maps an arm to ``(obj_name, grasp_from_obj)``; ``tool_from_ee`` is ``{tool_frame: 4x4}``. The body
    joints (the legs and the Waist) are free during single-arm planning, so an inactive arm's held block still moves
    through the world with the body; visualization needs poses for both arms. ``plan`` is a planner-output JointState
    in the planner's active cspace (the head and grippers locked); one that is not is reprojected with
    ``get_active_js`` before FK through ``planner.kinematics``.
    """
    out = {}
    for arm in ("left", "right"):
        if arm not in held:
            continue
        obj_name, grasp_tf = held[arm]
        pos = plan.position
        while pos.dim() > 2:
            pos = pos.squeeze(0) if pos.shape[0] == 1 else pos[0]
        if pos.shape[-1] != len(planner.kinematics.joint_names):
            traj_js = JointState.from_position(pos, joint_names=plan.joint_names)
            pos = planner.trajopt_solver.get_active_js(traj_js).position
        js = JointState.from_position(pos.to(grasp_tf.device))
        kin_state = planner.kinematics.compute_kinematics(js)
        tool_frame = TOOL_FRAME_FOR_ARM[arm]
        world_from_ee = kin_state.tool_poses.get_link_pose(tool_frame, make_contiguous=True).get_matrix()
        out[arm] = (obj_name, world_from_ee @ torch.inverse(tool_from_ee[tool_frame]) @ grasp_tf)
    return out


def build_motion_planner(world: TAMPWorld, use_cuda_graph: bool = True) -> MotionPlanner:
    """Build the trajopt planner on ``world.model``, colliding with the world's Scene with the movables included
    (so attached objects are part of the collision world during planning). Safe to run on a background
    thread concurrently with the Adam loop because it runs no solves.

    No ``planner.warmup()``: its solves would capture CUDA graphs on this
    thread, and cuRobo captures in CUDA's global mode, where the main thread's
    allocations and syncs fail mid-capture. The first ``plan_cspace`` captures
    instead (~1 s), on the main thread after the join; every later segment has
    the same shape and replays.
    """
    return get_t1_motion_planner(
        world.model,
        get_world_cfg(world.env, include_movables=True),
        use_cuda_graph=use_cuda_graph,
        # 2.5 mm, the trajopt margin in cuRobo's lbfgs_bspline_trajopt.yml (the
        # planner's own default, 0.01, is untested here). It only shapes the
        # optimizers' collision cost: cuRobo still judges success at 0
        # (metrics_base.yml). At 0 the optimum hugs obstacles, and retracts after
        # a place grazed the block just released by up to 0.2 mm.
        collision_activation_distance=0.0025,
        device_cfg=world.device_cfg,
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
        planner = get_planner() if get_planner is not None else build_motion_planner(world)

    # Defensive reset in case a prior solve_curobo aborted mid-plan and left the
    # planner with stale attached objects, disabled world obstacles, or placed
    # blocks. Required by the outer-retry loop in algorithm.py — without this,
    # a failure on the best particle leaves attached spheres on an arm link, a
    # disabled movable, and blocks at its placements, all of which corrupt the
    # retry attempt.
    _reset_attachment_state(planner, world)

    # Build initial state. Planner is in active-cspace (head and grippers
    # locked); particles are in full-cspace — convert at every boundary.
    full_joint_names = world.kinematics.joint_names
    last_js = _planner_js_from_full(
        planner, best_particle["left_q0"].clone(), full_joint_names,
    )
    held = {}  # {arm: (obj_name, grasp_from_obj)}
    q_init = best_particle["left_q0"]

    accum_plans: List[dict] = []
    gate_stats: List[dict] = []

    def add_segment(plan, dt, arm):
        # The real robot's balance check: every frame's CoM and ZMP inside the support rectangle, and the
        # segment's pace from its speeds, torques and ZMP; a RuntimeError fails the candidate. A retimed segment
        # also keeps its motion planner's timing, with the held objects along it.
        entry = {"dt": dt, "arm": arm}
        plan, stats, fast = _balance_gate(world, plan, dt)
        gate_stats.append(stats)
        if fast is not None:
            entry["planner_timing"] = {
                "plan": fast, "held_objs": _held_obj_poses(planner, world.tool_from_ee, held, fast),
            }
        entry.update(plan=plan, held_objs=_held_obj_poses(planner, world.tool_from_ee, held, plan))
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

        # plan_cspace holds nothing between endpoint confs, so the inactive hand is not held along the segment.
        # Both arms stay in the collision checks, except against the obstacle a pick or place segment disables for
        # the whole robot (the grasped block, or the support surface; _plan_arm_to_conf).
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
                    raise _plan_failed(world, result, f"Retract plan failed for {ground_op}")
                plan = add_segment(_interp_plan(result), _plan_dt(planner), arm)
                last_js = _last_timestep_js(plan, planner)

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
                        disable_obstacle=obj_name, ground_op=ground_op, world=world,
                    )

                # Attach AT the grasp conf. Block tracking from this point uses
                # inverse(obj_from_grasp) since the gripper is at the intended
                # grasp pose.
                grasp_plan = add_segment(_interp_plan(grasp_result), _plan_dt(planner), arm)
                last_js = _last_timestep_js(grasp_plan, planner)

                _attach_object(planner, obj_name, last_js, arm)
                held[arm] = (obj_name, torch.inverse(obj_from_grasp).clone())

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
                        disable_obstacle=surface_name, ground_op=ground_op, world=world,
                    )

                plan = add_segment(_interp_plan(place_result), _plan_dt(planner), arm)
                last_js = _last_timestep_js(plan, planner)
                obj_to_current_pose[obj_name] = world_from_obj_target

                held.pop(arm, None)
                # Detach: resets the gripper link's spheres and re-enables
                # the world obstacle. But the obstacle's stored pose is
                # still the original pickup pose, so update it to the
                # placement target before continuing.
                _detach_object(planner, obj_name, arm)
                planner.scene_collision_checker.update_obstacle_pose(
                    name=obj_name, w_obj_pose=Pose.from_matrix(world_from_obj_target),
                )

            else:
                raise NotImplementedError(f"Unsupported operator {ground_op.operator.name}")

        except RuntimeError as e:
            e.op_idx = idx  # lets _motion_plan_with_retries skip candidates that repeat ops 0..idx
            raise

        _log.info(f"{idx + 1}. {ground_op.name}")

    # Play back the accumulated trajectories on the visualizer's "curobo" axis, frame by frame, in the model's full
    # cspace (its 33 joints, the form T1RerunRobot takes: _full_positions), so Rerun shows the legs and arms animating.
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
        pos = _full_positions(world, plan)

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

    metrics = {"plan_duration_s": sum((e["plan"].position.shape[-2] - 1) * e["dt"] for e in accum_plans)}
    if gate_stats:
        metrics["retime_max"] = max(g["retime"] for g in gate_stats)
        metrics["retime_min"] = min(g["retime"] for g in gate_stats)  # below 1 where the gate sped one up
        metrics["retime_binding"] = dict(collections.Counter(g["binding"] for g in gate_stats))
        metrics["traj_com_min_margin"] = min(g["com_min_margin"] for g in gate_stats)
        metrics["traj_zmp_min_margin"] = min(g["zmp_min_margin"] for g in gate_stats)
    return accum_plans, metrics


def _full_positions(world: TAMPWorld, plan_js) -> torch.Tensor:
    """``[T, full_dof]`` positions of a planner trajectory in ``world.kinematics.joint_names`` order.

    accum_plans entries carry trajectories in different DOF forms depending on which planner method
    produced them: the planner's active cspace or the fully articulated form (a superset of
    world.kinematics's 33 joints); a plain reorder by ``plan_js.joint_names``, which every planner trajectory
    carries, handles both."""
    pos = plan_js.position.reshape(-1, plan_js.position.shape[-1])
    js = JointState.from_position(pos, joint_names=list(plan_js.joint_names))
    return js.reorder(list(world.kinematics.joint_names)).position


# The slowest the balance gate will make a segment: 4x longer, 16x lower accelerations.
RETIME_MAX = 4.0
# The balance gate's central-difference step along each frame's motion, s.
ZMP_STEP = 0.01
# The gate times each segment at the smallest uniform time scale its checks pass, found to within this fraction
# above it: a search setting, not a limit (_balance_gate) ...
RETIME_TOL = 0.02
# ... testing this many time scales per batched round of the ZMP's search ...
RETIME_SEARCH_POINTS = 16
# ... and slowing a resampled segment that fails its re-check by this factor until it passes, up to RETIME_MAX.
RETIME_RECHECK_STEP = 1.05


def _gate_error(check: str, message: str) -> RuntimeError:
    """The balance gate's RuntimeError, whose ``check`` names the check that failed: ``"soles"``, ``"com"``,
    ``"zmp"``, ``"torque <joint>"`` or ``"velocity <joint>"``."""
    error = RuntimeError(message)
    error.check = check
    return error


def _taylor_positions(world: TAMPWorld, plan_js, h: float = ZMP_STEP) -> torch.Tensor:
    """``[3, T, full_dof]``: each frame's configuration ``h`` s before, at and after it along the plan's own
    velocity and acceleration, ``q + tau q' + tau^2 / 2 q''`` (com_zmp_in_support's input)."""
    q, v, a = plan_js.position, plan_js.velocity, plan_js.acceleration
    return torch.stack([
        _full_positions(world, JointState.from_position(q + tau * v + 0.5 * tau**2 * a,
                                                        joint_names=plan_js.joint_names))
        for tau in (-h, 0.0, h)
    ])


def _retime(plan_js: JointState, s: float, dt: float) -> JointState:
    """``plan_js`` retimed uniformly by about ``s`` (above 1 slower, below 1 faster), resampled every ``dt`` s.

    A C2 cubic spline through the frames, with the native end velocities as boundary conditions,
    is sampled at ``N' = ceil(s (N - 1)) + 1`` frames, so the time scale is ``s_eff = (N' - 1) / (N - 1)
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


def _check_planted(world: TAMPWorld, q: torch.Tensor) -> None:
    """Raise a RuntimeError, naming its check, unless every frame ``q`` ``[T, dof]`` keeps both soles on the stance
    (``world.stance_soles``) within the kinematic constraint's sole tolerances (``soles_planted``). A NaN fails the
    check."""
    from cutamp.com_polygon_cost import _fk_state
    from cutamp.costs import curobo_pose_error

    ks = _fk_state(world, q)
    soles = torch.stack([ks.tool_poses.get_link_pose(f, make_contiguous=True).get_matrix()
                         for f in world.model.sole_frames], dim=1)  # [T, 2, 4, 4]
    pos, rot = curobo_pose_error(soles, world.stance_soles.expand_as(soles))
    if not bool(soles_planted(pos, rot).all()):
        pos_max, rot_max = pos.max().item(), rot.max().item()
        raise _gate_error("soles", f"trajectory soles leave the stance by up to {1000 * pos_max:.1f} mm and "
                                   f"{rot_max:.3f} rad")


def _check_torques(world: TAMPWorld, plan_js: JointState) -> None:
    """Raise a RuntimeError naming the torque unless every frame of ``plan_js`` keeps every joint's |torque|, by
    cuRobo's two RNEA passes with the floor's push (``world.segment_torques``), within its effort. A NaN torque
    fails."""
    torques = world.segment_torques
    tau = torques(plan_js).abs()
    if bool((tau <= torques.effort).all()):
        return
    excess = (tau - torques.effort).nan_to_num(nan=math.inf).amax(0)  # [dof]
    j = int(excess.argmax())
    raise _gate_error(f"torque {world.model.joint_names[j]}",
                      f"retimed trajectory torque passes {world.model.joint_names[j]}'s "
                      f"{torques.effort[j].item():.1f} N m effort by {excess[j].item():.2f} N m")


def _model_frames(world: TAMPWorld, plan_js: JointState) -> JointState:
    """``plan_js``'s frames: ``[T, n]`` positions, velocities and accelerations in ``world.model.joint_names`` order, by
    its joint names. A plan may also carry the joints the planner locks (the head, the fingers)."""
    names = list(world.model.joint_names)
    dof = plan_js.position.shape[-1]
    cols = [list(plan_js.joint_names).index(j) for j in names]
    q, v, a = (x.reshape(-1, dof)[:, cols] for x in (plan_js.position, plan_js.velocity, plan_js.acceleration))
    return JointState(position=q, velocity=v, acceleration=a, joint_names=names)


def _check_velocity(world: TAMPWorld, js: JointState, s: float = 1.0) -> Tuple[float, str]:
    """``(s_v, joint)``: the smallest uniform time scale at which every actuated joint of ``js`` (``[T, n]``, the
    model's order) keeps |velocity| / s within cuRobo's limit (``world.segment_torques.velocity``), exactly, and the
    joint that sets it. The base's six joints are virtual and go unchecked. Raises a RuntimeError naming that joint if
    ``s`` is under s_v: on the frames the gate returns (``s`` 1), or at the slowdown cap. A NaN velocity fails."""
    torques, names = world.segment_torques, list(world.model.joint_names)
    base = range(len(names))[world.model.base]
    act = [j for j in range(len(names)) if j not in base]
    peak = (js.velocity[:, act].abs() / torques.velocity[act]).amax(0).nan_to_num(nan=math.inf)
    k = int(peak.argmax())
    s_v, j = float(peak[k]), act[k]
    if not s_v <= s:
        limit = f"{names[j]}'s {float(torques.velocity[j]):.2f} rad/s limit"
        raise _gate_error(f"velocity {names[j]}",
                          f"retimed trajectory velocity passes {limit} by {100 * (s_v - 1):.1f}%" if s == 1.0
                          else f"trajectory velocity passes {limit} at the {s}x slowdown cap")
    return s_v, names[j]


def _at_scales(js: JointState, scales: torch.Tensor) -> JointState:
    """``js``'s frames (``[T, n]``) under each uniform time scale s in ``scales`` (``[K]``), stacked ``[K T, n]``: the
    positions repeated, the velocities over s, the accelerations over s^2. That is each frame's whole state at that
    scale, so every per-frame check on the stack is exact for it."""
    T, n = js.position.shape
    s = scales.to(js.position).reshape(-1, 1, 1)
    K = s.shape[0]
    return JointState(position=js.position.expand(K, T, n).reshape(K * T, n),
                      velocity=(js.velocity / s).reshape(K * T, n),
                      acceleration=(js.acceleration / s**2).reshape(K * T, n), joint_names=js.joint_names)


def _torque_scale(world: TAMPWorld, js: JointState, lo: float, hi: float) -> Tuple[float, Optional[str]]:
    """The smallest uniform time scale s in [lo, hi] at which every frame of ``js`` (``[T, n]``, the model's order)
    keeps every joint's |torque| within its effort, and the joint that sets it (None where ``lo`` passes). Exact: the
    torque, cuRobo's two RNEA passes with the floor's push (``world.segment_torques``), is linear in the accelerations
    and in the products of the velocities, and the floor's split depends on the positions only, so at scale s it is
    tau_0 + (tau_1 - tau_0) / s^2, tau_0 at rest and tau_1 at the plan's own timing; each frame and joint keeps
    u = 1 / s^2 inside an interval. Raises a RuntimeError naming the joint when no s in [lo, hi] passes; a NaN torque
    fails."""
    torques, names = world.segment_torques, list(world.model.joint_names)
    q, v, a = js.position, js.velocity, js.acceleration
    T, rest = q.shape[0], torch.zeros_like(v)
    tau = torques(JointState(position=torch.cat([q, q]), velocity=torch.cat([rest, v]),
                             acceleration=torch.cat([rest, a]), joint_names=js.joint_names))
    tau0, d = tau[:T], tau[T:] - tau[:T]
    e = torques.effort
    # |tau0 + d u| <= e bounds u to [lower, upper] at each frame and joint; a joint whose torque does not move with s
    # allows every u, or none.
    b1, b2 = (-e - tau0) / d, (e - tau0) / d
    still, inside = d == 0, tau0.abs() <= e
    lower = torch.where(still, torch.where(inside, -math.inf, math.inf), torch.minimum(b1, b2))
    upper = torch.where(still, torch.where(inside, math.inf, -math.inf), torch.maximum(b1, b2)).amin(0)  # [n]
    u = min(float(upper.min()), lo**-2)
    if not u >= max(float(lower.max()), hi**-2):  # no s in [lo, hi] passes, or a NaN
        excess = ((tau0 + d * hi**-2).abs() - e).nan_to_num(nan=math.inf).amax(0)
        j = int(excess.argmax())
        raise _gate_error(f"torque {names[j]}", f"trajectory torque passes {names[j]}'s {float(e[j]):.1f} N m effort "
                                                f"by {float(excess[j]):.2f} N m at the {hi}x slowdown cap")
    if u == lo**-2:
        return lo, None
    return u**-0.5, names[int(upper.argmin())]


def _zmp_margins(world: TAMPWorld, js: JointState, scales: torch.Tensor) -> torch.Tensor:
    """``[K]``: the multibody ZMP's smallest margin to the gate's support rectangle over the frames of ``js`` under each
    uniform time scale in ``scales`` (negative outside, -inf where a frame would need the floor to pull; a NaN fails),
    from the scaled frames' Taylor positions."""
    from cutamp.com_polygon_cost import com_margin_from_base, com_zmp_in_support

    _, zmp = com_zmp_in_support(world, _taylor_positions(world, _at_scales(js, scales)), ZMP_STEP)
    return com_margin_from_base(zmp, world.model.support.gate_half).reshape(len(scales), -1).amin(-1)


def _zmp_scale(world: TAMPWorld, js: JointState, lo: float, hi: float) -> Tuple[float, bool]:
    """The smallest uniform time scale in [lo, hi] at which every frame's multibody ZMP stays inside the gate's support
    rectangle, to within RETIME_TOL above it, and whether the ZMP sets it (False where ``lo`` passes): a geometric
    bisection testing RETIME_SEARCH_POINTS scales per batched round, which assumes the ZMP's excursion falls as s
    grows. Raises a RuntimeError when ``hi`` fails."""
    def first_pass(grid: list) -> Optional[int]:
        ok = (_zmp_margins(world, js, torch.tensor(grid, dtype=torch.float64)) >= 0).tolist()
        return ok.index(True) if True in ok else None

    grid = [lo * (hi / lo) ** (k / (RETIME_SEARCH_POINTS - 1)) for k in range(RETIME_SEARCH_POINTS - 1)] + [hi]
    i = first_pass(grid)
    if i == 0:
        return lo, False
    if i is None:
        raise _gate_error("zmp", f"trajectory ZMP stays outside the support region at the {hi}x slowdown cap")
    a, b = grid[i - 1], grid[i]  # the ZMP fails at a and passes at b
    while b > a * (1 + RETIME_TOL):
        grid = [a * (b / a) ** (k / (RETIME_SEARCH_POINTS + 1)) for k in range(1, RETIME_SEARCH_POINTS + 1)]
        i = first_pass(grid)
        a, b = (grid[-1], b) if i is None else (grid[i - 1] if i else a, grid[i])
    return b, True


def _recheck(world: TAMPWorld, js: JointState, planted: bool = True) -> Tuple[float, float]:
    """The gate's checks on the frames it returns, from their own velocities and accelerations: both soles on the
    stance (``_check_planted``; ``planted`` False skips it on the planner's own frames, which
    passed it already), the CoM and the multibody ZMP inside the gate's support rectangle (a frame that would need the
    floor to pull has no ZMP and fails it), every joint's torque within its effort and every actuated joint's speed
    within its limit. Returns the CoM's and the ZMP's smallest margins; a failed check raises a RuntimeError whose
    ``check`` names it."""
    from cutamp.com_polygon_cost import com_margin_from_base, com_zmp_in_support

    half = world.model.support.gate_half
    if planted:
        _check_planted(world, _full_positions(world, js))
    com, zmp = com_zmp_in_support(world, _taylor_positions(world, js), ZMP_STEP)
    com_m, zmp_m = com_margin_from_base(com, half).min().item(), com_margin_from_base(zmp, half).min().item()
    if not com_m >= 0:
        raise _gate_error("com", f"retimed trajectory CoM leaves the support region by {-100 * com_m:.1f} cm")
    lifted = int(torch.isposinf(zmp).any(-1).sum())  # com_zmp_in_support: no ZMP where the floor would have to pull
    if lifted:
        raise _gate_error("zmp", f"the floor would have to pull on the retimed trajectory's soles at {lifted} "
                                 f"frames: there is no ZMP")
    if not zmp_m >= 0:
        raise _gate_error("zmp", f"retimed trajectory ZMP leaves the support region by {-100 * zmp_m:.1f} cm")
    _check_torques(world, js)
    _check_velocity(world, _model_frames(world, js))
    return com_m, zmp_m


def _balance_gate(world: TAMPWorld, plan_js: JointState, dt: float):
    """The real robot's balance check on every interpolated frame of a segment, and the segment at the pace the
    robot allows. Returns ``(plan_js, stats, planner_timing)``: the plan at that pace; its stats; and, for a retimed
    segment, the same spline sampled at the motion planner's own timing: its positions exactly, with the spline's
    velocities and accelerations (the timing test runs both), else None. Each check raises a RuntimeError that names
    it, so the planner moves to its next candidate or skeleton.

    Every frame must keep both soles on the stance (``_check_planted``) and its CoM inside the gate's support
    rectangle; no time scale changes either. The gate then
    finds the smallest uniform time scale s in [s_v, RETIME_MAX] (below 1 speeds the segment up, above 1 slows it down)
    at which every actuated joint keeps its speed within cuRobo's limit (s_v, exact: ``_check_velocity``), every joint
    its torque within its effort (exact: ``_torque_scale``) and the multibody ZMP stays inside (``_zmp_scale``, to
    within RETIME_TOL above the smallest). Each is evaluated on the planner's own frames with the velocities over s and
    the accelerations over s^2. Below one step of ``dt`` for the whole segment, the shortest ``_retime`` makes, no check
    binds (``"none"``). The segment is resampled at s (``_retime``, whose whole steps give s_eff >= s; at s_eff 1 it is
    the planner's own), its first and last frames kept the planner's bit for bit (the next segment starts from the
    last), and the frames it returns are checked again (``_recheck``); one whose speed, torque or ZMP fails is slowed
    by RETIME_RECHECK_STEP until it passes, and past RETIME_MAX, on any other failed check, or when no s passes, the
    gate raises, naming the check. The stats' ``"retime"`` is s_eff and ``"binding"`` the check that sets it:
    ``"velocity <joint>"``, ``"torque <joint>"``, ``"zmp"``, ``"none"``, or the re-check's that failed last. Their
    ``"com_min_margin"`` and ``"zmp_min_margin"`` are the smaller of the returned frames' and the planner's frames' at
    s_eff (the returned spline at its knots): a sped-up segment returns fewer frames than the planner's, which the
    search ran on."""
    from cutamp.com_polygon_cost import com_margin_from_base, com_zmp_in_support

    with torch.no_grad():
        _check_planted(world, _full_positions(world, plan_js))
        js = _model_frames(world, plan_js)
        com, _ = com_zmp_in_support(world, _taylor_positions(world, js), ZMP_STEP)
        com_planner = com_margin_from_base(com, world.model.support.gate_half).min().item()
        if not com_planner >= 0:
            raise _gate_error("com", f"trajectory CoM leaves the support region by {-100 * com_planner:.1f} cm")
        n = js.position.shape[0]
        step = 1.0 / (n - 1) if n > 1 else 1.0
        s_v, joint = _check_velocity(world, js, RETIME_MAX)
        s, binding = (s_v, f"velocity {joint}") if s_v >= step else (step, "none")
        s_t, joint = _torque_scale(world, js, s, RETIME_MAX)
        if joint is not None:
            s, binding = s_t, f"torque {joint}"
        s_z, zmp_binds = _zmp_scale(world, js, s, RETIME_MAX)
        if zmp_binds:
            s, binding = s_z, "zmp"
        while True:
            out = plan_js if math.ceil(s * (n - 1)) + 1 == n else _retime(plan_js, s, dt)
            if out is not plan_js:  # the spline's end values miss a joint that moves to 0 by a rounding error
                out.position[..., [0, -1], :] = plan_js.position[..., [0, -1], :]
            try:
                com_m, zmp_m = _recheck(world, out, planted=out is not plan_js)
                break
            except RuntimeError as e:
                check = getattr(e, "check", None) or ""
                if check.split(" ")[0] not in ("velocity", "torque", "zmp") or s >= RETIME_MAX:
                    raise  # the soles and the CoM follow the path, which no time scale changes
                s, binding = min(RETIME_RECHECK_STEP * s, RETIME_MAX), check
        n_out = out.position.shape[-2]
        s_eff = (n_out - 1) / (n - 1) if n_out != n else 1.0
        zmp_planner = float(_zmp_margins(world, js, torch.tensor([s_eff]))[0])
    stats = {"retime": s_eff, "binding": binding, "com_min_margin": min(com_m, com_planner),
             "zmp_min_margin": min(zmp_m, zmp_planner)}
    return out, stats, None if out is plan_js else _retime(plan_js, 1.0, dt)
