"""Single-arm grasp planner — fork of ``MotionPlanner.plan_grasp`` for cuTAMP.

The stock ``MotionPlanner.plan_grasp`` (curobo/_src/motion/motion_planner.py:412)
applies its approach/lift offsets uniformly to every frame in
``grasp_poses.tool_frames``. For T1 we want the offsets applied only to the
*active* arm; the inactive arm should stay put. This wrapper:

- Accepts a single active tool frame name.
- Builds a ``GoalToolPose`` covering ALL of the planner's tool frames — the
  active frame gets the requested target pose, inactive frames get their
  current FK pose so the IK's ``reorder_links`` check accepts the goal.
- The caller is expected to have called ``T1State.pin_for_arm_action(arm)``
  beforehand, which both pins the inactive arm's cspace weight and calls
  ``update_tool_pose_criteria({inactive: ToolPoseCriteria.disabled()})`` so
  the inactive frame's pose cost is zero — the trajopt won't try to drag
  the inactive arm to satisfy a pose target.
"""

from typing import Dict, List, Optional

import torch

from curobo.motion_planner import GraspPlanResult, MotionPlanner
from curobo.types import GoalToolPose, JointState, Pose


def _build_multi_frame_goal(
    host,
    active_tool_frame: str,
    target_pose: Pose,
    current_state: JointState,
) -> GoalToolPose:
    """Build a GoalToolPose covering all of ``host.kinematics.tool_frames``.

    ``host`` may be a ``MotionPlanner`` or an ``IKSolver`` — both expose
    ``kinematics.tool_frames`` and ``compute_kinematics(state)``.

    The active frame gets ``target_pose``; every inactive frame gets its
    current FK pose at ``current_state``. This satisfies cuRobo IK's
    ``reorder_links`` requirement that the goal cover every kinematics
    tool frame, while keeping the inactive frames at their "stay put"
    pose so they don't drag the trajopt away from the cspace pin.
    """
    all_frames: List[str] = list(host.kinematics.tool_frames)
    poses: Dict[str, Pose] = {active_tool_frame: target_pose}
    inactive = [f for f in all_frames if f != active_tool_frame]
    if inactive:
        kin_state = host.compute_kinematics(current_state)
        for f in inactive:
            poses[f] = kin_state.tool_poses.get_link_pose(f)
    return GoalToolPose.from_poses(poses, ordered_tool_frames=all_frames)


def plan_single_arm_grasp(
    planner: MotionPlanner,
    active_tool_frame: str,
    grasp_poses_dict: dict,
    current_state: JointState,
    *,
    grasp_approach_axis: str = "z",
    grasp_approach_offset: float = -0.15,
    grasp_lift_axis: str = "z",
    grasp_lift_offset: float = -0.15,
    plan_approach_to_grasp: bool = True,
    plan_grasp_to_lift: bool = True,
    disable_collision_links: Optional[List[str]] = None,
) -> GraspPlanResult:
    """Plan an approach→grasp→lift trajectory for one arm.

    Args:
        planner: The unified MotionPlanner (covers all tool frames).
        active_tool_frame: The tool frame that should execute the grasp.
        grasp_poses_dict: ``{active_tool_frame: Pose}`` for the candidate grasp(s).
        current_state: Current joint state (full robot DOF).
        grasp_approach_axis / grasp_approach_offset: how far to back off along
            the tool axis before approaching.
        grasp_lift_axis / grasp_lift_offset: how far to lift after grasping.
        plan_approach_to_grasp / plan_grasp_to_lift: which sub-trajectories to
            actually plan (matches stock ``plan_grasp``).
        disable_collision_links: links to disable collisions for during the
            grasp. Defaults to the planner's grasp_contact_link_names.
    """
    if active_tool_frame not in grasp_poses_dict:
        raise ValueError(f"grasp_poses_dict must contain {active_tool_frame}")

    # Multi-tool goal: active frame gets the grasp pose; inactive frames get
    # their current FK pose. cuRobo's plan_grasp will add the approach/lift
    # offset to ALL frames in the goal, so the inactive frames receive
    # offset versions of their current poses — but the inactive
    # ToolPoseCriteria has been zeroed by pin_for_arm_action, so the trajopt
    # ignores those targets and the inactive arm stays put under its cspace
    # pin.
    grasp_goal = _build_multi_frame_goal(
        planner, active_tool_frame, grasp_poses_dict[active_tool_frame], current_state,
    )

    return planner.plan_grasp(
        grasp_poses=grasp_goal,
        current_state=current_state,
        grasp_approach_axis=grasp_approach_axis,
        grasp_approach_offset=grasp_approach_offset,
        grasp_approach_in_tool_frame=True,
        grasp_lift_axis=grasp_lift_axis,
        grasp_lift_offset=grasp_lift_offset,
        grasp_lift_in_tool_frame=True,
        plan_approach_to_grasp=plan_approach_to_grasp,
        plan_grasp_to_lift=plan_grasp_to_lift,
        disable_collision_links=disable_collision_links,
    )
