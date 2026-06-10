"""Multi-frame goal construction for single-arm actions on T1.

cuRobo IK's ``reorder_links`` check requires a ``GoalToolPose`` that covers
every kinematics tool frame, even when only one arm is acting.
``_build_multi_frame_goal`` builds that goal: the active frame gets the
requested target pose, and every inactive frame gets its current FK pose at
the segment's start state. The inactive frame's ``ToolPoseCriteria`` stays
ENABLED — cuRobo's native world-frame ``tool_pose`` cost therefore holds the
inactive wrist at its start pose, complementing the inactive-arm cspace pin
applied by ``T1State.pin_for_arm_action``.
"""

from typing import Dict, List

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
