"""T1State — single-MotionPlanner state container for the T1 humanoid.

Cspace pinning protocols (TrajOpt-side only — see ``pin_for_arm_action``):

* ``pin_for_arm_action(active_arm)``: pin the inactive arm's cspace target
  weights (mostly informational for IK; trajopt's
  ``cspace_target_weight`` is 0). The inactive wrist is held in place by
  cuRobo's native world-frame ``tool_pose`` cost via the multi-frame goal
  built by ``cutamp/grasp_planning._build_multi_frame_goal`` — that goal
  sets the inactive frame's target pose to its FK at the segment's
  starting state, and the criterion stays ENABLED so trajopt drives the
  inactive wrist back to that world position throughout the trajectory.
* ``pin_for_movebase()``: lock both arms + lift/torso; only base DOFs free.

Both restore via ``unpin()``.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Iterable, List, Literal, Optional

import torch

from curobo.kinematics import Kinematics
from curobo.motion_planner import MotionPlanner
from curobo.types import JointState, ToolPoseCriteria

from cutamp._curobo_internals import (
    restore_cspace_target_dof_weight,
    snapshot_cspace_target_dof_weight,
    write_cspace_target_dof_weight,
)
from cutamp.robots.t1 import (
    BODY_INDICES,
    LEFT_ARM_JOINT_NAMES,
    LEFT_TOOL_FRAME,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_TOOL_FRAME,
    TOOL_FRAME_FOR_ARM,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
    JOINT_NAMES_FULL,
)


@dataclass
class T1State:
    """State for T1 single-MotionPlanner planning."""

    planner: MotionPlanner
    kinematics: Kinematics
    tool_from_ee: Dict[str, torch.Tensor]      # {tool_frame_name: 4x4 transform}
    current_js: JointState                      # planner-active-cspace joint state
    arm_holding: Dict[str, Optional[str]] = field(
        default_factory=lambda: {"left": None, "right": None}
    )
    arm_grasp_transform: Dict[str, Optional[torch.Tensor]] = field(
        default_factory=lambda: {"left": None, "right": None}
    )

    _saved_target_dof_weight: Optional[Dict[int, List[torch.Tensor]]] = None
    _saved_pin_hosts: Optional[List[Any]] = None
    _disabled_tool_pose_frames: Optional[List[str]] = None

    def get_tool_frame(self, arm: Literal["left", "right"]) -> str:
        return TOOL_FRAME_FOR_ARM[arm]

    def other_arm(self, arm: Literal["left", "right"]) -> Literal["left", "right"]:
        return "right" if arm == "left" else "left"

    @cached_property
    def _planner_joint_names(self) -> List[str]:
        """Joint names of the planner's active cspace (post-lock_joints)."""
        return list(self.planner.kinematics.joint_names)

    def _apply_pin(
        self,
        pin_joint_names: list,
        disabled_tool_frames: list,
        pin_weight: float,
        default_weight: float,
        hosts: Iterable[Any],
    ) -> None:
        """Set per-DOF cspace weights (joints addressed by NAME) on every host.

        Joint names that aren't in the planner's active cspace are silently
        skipped — e.g., a base DOF requested for pinning when the base is
        already locked has nothing to pin in the active cspace.

        Bookkeeping for ``_disabled_tool_pose_frames`` is set BEFORE the
        planner-mutating call so ``unpin`` can recover any partial state if
        ``update_tool_pose_criteria`` raises.
        """
        hosts = list(hosts)
        active_names = self._planner_joint_names
        weights = torch.full(
            (len(active_names),),
            default_weight,
            device=self.current_js.position.device,
            dtype=self.current_js.position.dtype,
        )
        name_to_idx = {n: i for i, n in enumerate(active_names)}
        for n in pin_joint_names:
            i = name_to_idx.get(n)
            if i is not None:
                weights[i] = pin_weight

        if self._saved_target_dof_weight is None:
            self._saved_target_dof_weight = snapshot_cspace_target_dof_weight(hosts)
            self._saved_pin_hosts = hosts

        # Record the disabled frames BEFORE mutating the planner so unpin
        # can re-enable them even if update_tool_pose_criteria raises
        # mid-mutation. Use list() to capture by value.
        self._disabled_tool_pose_frames = list(disabled_tool_frames)

        write_cspace_target_dof_weight(hosts, weights)

        if disabled_tool_frames:
            self.planner.update_tool_pose_criteria(
                {f: ToolPoseCriteria.disabled() for f in disabled_tool_frames}
            )

    def pin_for_arm_action(
        self,
        active_arm: Literal["left", "right"],
        pin_weight: float = 1000.0,
        default_weight: float = 1.0,
        *,
        hosts: Optional[Iterable[Any]] = None,
    ) -> None:
        """Pin the inactive arm's cspace target weight.

        The inactive wrist is held in place by cuRobo's native world-frame
        ``tool_pose`` cost — its criterion stays ENABLED and the multi-frame
        goal built by ``_build_multi_frame_goal`` sets the inactive frame's
        target pose to its FK at the segment's start state. The cspace
        weight write here is largely informational for trajopt (whose
        ``cspace_target_weight`` defaults to 0); it engages on the planner's
        internal IK seed solver where ``cspace_target_weight`` is nonzero.

        Idempotency: caller must ``unpin()`` before a second pin.
        """
        if self._saved_target_dof_weight is not None:
            raise RuntimeError(
                "pin_for_arm_action called while a pin is already active; "
                "call unpin() first."
            )
        inactive_names = (
            RIGHT_ARM_JOINT_NAMES if active_arm == "left" else LEFT_ARM_JOINT_NAMES
        )
        if hosts is not None:
            host_list = list(hosts)
        else:
            host_list = [self.planner]
            internal_ik = getattr(self.planner, "ik_solver", None)
            if internal_ik is not None:
                host_list.append(internal_ik)
        self._apply_pin(
            pin_joint_names=list(inactive_names),
            disabled_tool_frames=[],
            pin_weight=pin_weight,
            default_weight=default_weight,
            hosts=host_list,
        )

    def pin_for_movebase(
        self,
        pin_weight: float = 1000.0,
        default_weight: float = 1.0,
        *,
        hosts: Optional[Iterable[Any]] = None,
    ) -> None:
        """Lock body + both arms; only the base DOFs remain free.

        WARNING: the planner config locks the mobile base, so MoveBaseTo
        cannot actually move the base in this planner. Envs needing
        navigation must build a separate planner without the base lock.

        Idempotency: caller must ``unpin()`` before a second pin.
        """
        if self._saved_target_dof_weight is not None:
            raise RuntimeError(
                "pin_for_movebase called while a pin is already active; "
                "call unpin() first."
            )
        body_names = list(JOINT_NAMES_FULL[BODY_INDICES])
        self._apply_pin(
            pin_joint_names=body_names + list(LEFT_ARM_JOINT_NAMES) + list(RIGHT_ARM_JOINT_NAMES),
            disabled_tool_frames=[LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME],
            pin_weight=pin_weight,
            default_weight=default_weight,
            hosts=hosts if hosts is not None else [self.planner],
        )
        # `_disabled_tool_pose_frames` is now set inside `_apply_pin`.

    def unpin(self) -> None:
        """Restore default cspace weights and re-enable any disabled
        tool-pose criteria."""
        if self._saved_target_dof_weight is None:
            return
        hosts = self._saved_pin_hosts or [self.planner]
        restore_cspace_target_dof_weight(hosts, self._saved_target_dof_weight)
        self._saved_target_dof_weight = None
        self._saved_pin_hosts = None
        if self._disabled_tool_pose_frames:
            self.planner.update_tool_pose_criteria(
                {f: ToolPoseCriteria() for f in self._disabled_tool_pose_frames}
            )
            self._disabled_tool_pose_frames = None

    def gripper_state(self, arm: Literal["left", "right"]):
        return GRIPPER_CLOSED if self.arm_holding.get(arm) is not None else GRIPPER_OPEN

    def compute_held_obj_poses(
        self,
        arm: Optional[Literal["left", "right"]],
        plan: JointState,
    ) -> Optional[torch.Tensor]:
        """Per-timestep ``world_from_obj`` for a held object along ``plan``.

        ``plan`` is a planner-output JointState. Trajectories arrive in the
        planner's active cspace (e.g., 18-DOF with the mobile base locked) so
        FK runs through ``planner.kinematics`` after a ``get_active_js``
        reprojection. Returns ``None`` when ``arm`` is ``None`` or nothing is
        held on it.
        """
        if arm is None:
            return None
        grasp_tf = self.arm_grasp_transform.get(arm)
        if self.arm_holding.get(arm) is None or grasp_tf is None:
            return None
        pos = plan.position
        while pos.dim() > 2:
            pos = pos.squeeze(0) if pos.shape[0] == 1 else pos[0]
        active_dof = len(self.planner.kinematics.joint_names)
        if pos.shape[-1] != active_dof:
            traj_js = JointState.from_position(pos, joint_names=plan.joint_names)
            pos = self.planner.trajopt_solver.get_active_js(traj_js).position
        js = JointState.from_position(pos.to(grasp_tf.device))
        kin_state = self.planner.kinematics.compute_kinematics(js)
        tool_frame = self.get_tool_frame(arm)
        ee_pose = kin_state.tool_poses.get_link_pose(tool_frame, make_contiguous=True)
        world_from_ee = ee_pose.get_matrix()
        tool_from_ee_mat = self.tool_from_ee[tool_frame]
        return world_from_ee @ torch.inverse(tool_from_ee_mat) @ grasp_tf

    def compute_all_held_obj_poses(self, plan: JointState) -> Dict[str, tuple]:
        """``{arm: (obj_name, poses)}`` for every arm currently holding.

        The body joints (leg/torso/waist) are unlocked during single-arm
        planning so an inactive arm's held block still translates through
        world as the body moves; visualization needs poses for both arms.
        """
        out: Dict[str, tuple] = {}
        for held_arm in ("left", "right"):
            obj_name = self.arm_holding.get(held_arm)
            if obj_name is None:
                continue
            poses = self.compute_held_obj_poses(held_arm, plan)
            if poses is not None:
                out[held_arm] = (obj_name, poses)
        return out
