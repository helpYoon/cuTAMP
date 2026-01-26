"""Dual-arm state management for T1 robot motion planning."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import torch
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen

from cutamp.robots.t1 import (
    NUM_SHARED_JOINTS,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    t1_curobo_cfg,
    t1_home_head,
    GRIPPER_OPEN,
)
from cutamp.utils.visualizer import Visualizer


@dataclass
class DualArmState:
    """Encapsulates state for dual-arm motion planning.
    
    This class tracks the state of both arms during motion planning for the T1 robot,
    including joint states, configuration names, which arm is currently active,
    and what objects each arm is holding.
    """
    
    # Motion generators and kinematics
    left_motion_gen: MotionGen
    right_motion_gen: MotionGen
    left_kin_model: object
    right_kin_model: object
    left_tool_from_ee: torch.Tensor
    right_tool_from_ee: torch.Tensor
    
    # Joint state tracking
    left_js: JointState
    right_js: JointState
    left_q_name: str = "left_q0"
    right_q_name: str = "right_q0"
    
    # Current arm being used
    current_arm: Optional[Literal["left", "right"]] = None
    
    # Object tracking
    arm_holding: Dict[str, Optional[str]] = field(default_factory=lambda: {"left": None, "right": None})
    arm_grasp_transform: Dict[str, Optional[torch.Tensor]] = field(default_factory=lambda: {"left": None, "right": None})
    
    def get_state(self, arm: Literal["left", "right"]):
        """Get motion gen, kin model, tool transform, and joint state for arm."""
        if arm == "left":
            return self.left_motion_gen, self.left_kin_model, self.left_tool_from_ee, self.left_js
        return self.right_motion_gen, self.right_kin_model, self.right_tool_from_ee, self.right_js
    
    def get_js(self, arm: Literal["left", "right"]) -> JointState:
        """Get joint state for an arm."""
        return self.left_js if arm == "left" else self.right_js
    
    def set_js(self, arm: Literal["left", "right"], js: JointState):
        """Update joint state for an arm."""
        if arm == "left":
            self.left_js = js
        else:
            self.right_js = js
    
    def get_q_name(self, arm: Literal["left", "right"]) -> str:
        """Get configuration name for an arm."""
        return self.left_q_name if arm == "left" else self.right_q_name
    
    def set_q_name(self, arm: Literal["left", "right"], name: str):
        """Update configuration name for an arm."""
        if arm == "left":
            self.left_q_name = name
        else:
            self.right_q_name = name
    
    def other_arm(self, arm: Literal["left", "right"]) -> Literal["left", "right"]:
        """Get the other arm."""
        return "right" if arm == "left" else "left"
    
    def sync_shared_joints(self, from_arm: Literal["left", "right"], to_arm: Literal["left", "right"]):
        """Synchronize shared joints from one arm to another.
        
        T1 robot has 4 shared joints (lift + torso) that appear in both arms' 11-DOF configs.
        When switching arms, we need to ensure the new arm's shared joints match the previous arm's.
        """
        from_js = self.get_js(from_arm)
        to_js = self.get_js(to_arm)
        new_pos = to_js.position.clone()
        new_pos[:, :NUM_SHARED_JOINTS] = from_js.position[:, :NUM_SHARED_JOINTS]
        self.set_js(to_arm, JointState.from_position(new_pos))


def update_locked_arm_position(
    motion_gen: MotionGen,
    active_arm: Literal["left", "right"],
    locked_arm_js: JointState,
) -> None:
    """Update the locked arm's joint positions in the motion generator.
    
    When planning for one arm, cuRobo needs to know the actual position of the locked arm
    for accurate collision checking. This function updates the locked joints in the
    kinematics model to match the locked arm's current position.
    
    Args:
        motion_gen: The motion generator for the active arm.
        active_arm: Which arm is currently active ("left" or "right").
        locked_arm_js: Joint state of the locked arm (11 DOF).
    """
    # Determine which joints are locked based on active arm
    locked_arm = "right" if active_arm == "left" else "left"
    locked_joint_names = RIGHT_ARM_JOINT_NAMES if locked_arm == "right" else LEFT_ARM_JOINT_NAMES
    
    # Extract arm joint values (indices 4-10) from the locked arm's 11-DOF state
    locked_arm_values = locked_arm_js.position[0, NUM_SHARED_JOINTS:].tolist()
    
    # Build the lock_joints dict with all locked joints
    lock_joints = {}
    
    # Head joints (always locked at neutral)
    lock_joints["AAHead_yaw"] = t1_home_head[0]
    lock_joints["Head_pitch"] = t1_home_head[1]
    
    # Locked arm joints (7 DOF) - update to current position
    for name, value in zip(locked_joint_names, locked_arm_values):
        lock_joints[name] = value
    
    # Gripper joints (all at open position - 0.0)
    for prefix in ["left", "right"]:
        for suffix in ["Link1", "Link11", "Link2", "Link22"]:
            lock_joints[f"{prefix}_{suffix}"] = GRIPPER_OPEN[0]
    
    # Get the robot config for the active arm
    robot_cfg = t1_curobo_cfg(active_arm)
    
    # Update the locked joints in the motion generator
    motion_gen.update_locked_joints(lock_joints, robot_cfg)


def make_dual_arm_traj(traj: torch.Tensor, active_arm: Literal["left", "right"], state: DualArmState) -> torch.Tensor:
    """Combine active arm trajectory with inactive arm (shared joints synced).
    
    Args:
        traj: Trajectory for the active arm (N, 11)
        active_arm: Which arm is active ("left" or "right")
        state: Current dual-arm state
        
    Returns:
        Combined trajectory (N, 22) with both arms, where:
        - Active arm follows the trajectory
        - Inactive arm stays at its last position but with shared joints synced
    """
    n = traj.shape[0]
    traj_cpu = traj.cpu()
    other_js = state.get_js(state.other_arm(active_arm))
    other = other_js.position.cpu().expand(n, -1).clone()
    other[:, :NUM_SHARED_JOINTS] = traj_cpu[:, :NUM_SHARED_JOINTS]
    if active_arm == "left":
        return torch.cat([traj_cpu, other], dim=1)
    return torch.cat([other, traj_cpu], dim=1)


def compute_held_obj_poses(arm: Literal["left", "right"], traj: torch.Tensor, state: DualArmState) -> Optional[torch.Tensor]:
    """Compute held object poses for an arm given joint trajectory.
    
    Args:
        arm: Which arm to compute poses for
        traj: Joint trajectory (N, 11)
        state: Current dual-arm state
        
    Returns:
        Object poses (N, 4, 4) if the arm is holding an object, None otherwise
    """
    held = state.arm_holding.get(arm)
    grasp_tf = state.arm_grasp_transform.get(arm)
    if held is None or grasp_tf is None:
        return None
    km = state.right_kin_model if arm == "right" else state.left_kin_model
    tf = state.right_tool_from_ee if arm == "right" else state.left_tool_from_ee
    traj_cuda = traj.to(km.tensor_args.device)
    world_from_ee = km.get_state(traj_cuda).ee_pose.get_matrix()
    return world_from_ee @ torch.inverse(tf) @ grasp_tf


def log_dual_arm_traj(
    traj: torch.Tensor,
    active_arm: Literal["left", "right"],
    active_obj: Optional[str],
    active_obj_poses: Optional[torch.Tensor],
    start_time: float,
    dt: float,
    state: DualArmState,
    visualizer: Visualizer,
    timeline: str,
    obj_to_current_pose: Dict[str, torch.Tensor],
) -> float:
    """Log dual-arm trajectory to visualizer and update all held objects.
    
    Args:
        traj: Trajectory for the active arm (N, 11)
        active_arm: Which arm is active
        active_obj: Object name held by active arm (or None)
        active_obj_poses: Poses for the active arm's held object (or None)
        start_time: Start time for visualization
        dt: Time step between frames
        state: Current dual-arm state
        visualizer: Visualizer instance
        timeline: Timeline name for Rerun
        obj_to_current_pose: Dict to update with final object poses
        
    Returns:
        End time of the logged trajectory
    """
    viz_traj = make_dual_arm_traj(traj, active_arm, state)
    n = traj.shape[0]
    
    # Compute inactive arm's held object poses if any
    other_arm = state.other_arm(active_arm)
    other_js = state.get_js(other_arm)
    other_traj = other_js.position.cpu().expand(n, -1).clone()
    other_traj[:, :NUM_SHARED_JOINTS] = traj.cpu()[:, :NUM_SHARED_JOINTS]
    other_obj_poses = compute_held_obj_poses(other_arm, other_traj, state)
    other_obj = state.arm_holding.get(other_arm)
    
    # Determine what objects need updating
    objects_to_update = []
    if active_obj is not None and active_obj_poses is not None:
        objects_to_update.append((active_obj, active_obj_poses))
    if other_obj is not None and other_obj_poses is not None:
        objects_to_update.append((other_obj, other_obj_poses))
    
    if len(objects_to_update) == 0:
        return visualizer.log_joint_trajectory(viz_traj, timeline=timeline, start_time=start_time, dt=dt)
    elif len(objects_to_update) == 1:
        obj_name, obj_poses = objects_to_update[0]
        ts = visualizer.log_joint_trajectory_with_mat4x4(
            traj=viz_traj, mat4x4_key=f"world/{obj_name}", mat4x4=obj_poses,
            timeline=timeline, start_time=start_time, dt=dt
        )
        obj_to_current_pose[obj_name] = obj_poses[-1]
        return ts
    else:
        # Multiple objects - log frame by frame
        for i in range(n):
            visualizer.set_time_seconds(timeline, start_time + i * dt)
            visualizer.set_joint_positions(viz_traj[i])
            for obj_name, obj_poses in objects_to_update:
                visualizer.log_mat4x4(f"world/{obj_name}", obj_poses[i])
        for obj_name, obj_poses in objects_to_update:
            obj_to_current_pose[obj_name] = obj_poses[-1]
        return start_time + n * dt


def make_gripper_anim_traj(pos: torch.Tensor, gripper_interp: torch.Tensor, arm: Literal["left", "right"], state: DualArmState) -> torch.Tensor:
    """Create 26-DOF trajectory (22 arms + 4 gripper) for gripper animation.
    
    Args:
        pos: Current joint position of the active arm (1, 11)
        gripper_interp: Gripper interpolation trajectory (N, 4)
        arm: Which arm is active
        state: Current dual-arm state
        
    Returns:
        Combined trajectory (N, 26) with both arms and gripper joints
    """
    n = gripper_interp.shape[0]
    active = pos.cpu().expand(n, -1)
    other_js = state.get_js(state.other_arm(arm))
    other = other_js.position.cpu().clone()
    other[:, :NUM_SHARED_JOINTS] = pos.cpu()[:, :NUM_SHARED_JOINTS]
    other = other.expand(n, -1)
    if arm == "left":
        return torch.cat([active, other, gripper_interp], dim=1)
    return torch.cat([other, active, gripper_interp], dim=1)
