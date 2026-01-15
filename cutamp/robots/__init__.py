# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import Sequence, Union

import roma
import torch
from jaxtyping import Float

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.transform import quaternion_to_matrix
from curobo.types.base import TensorDeviceType
from .franka import (
    franka_curobo_cfg,
    get_franka_kinematics_model,
    get_franka_gripper_spheres,
    franka_neutral_joint_positions,
    load_franka_rerun,
)
from .ur5 import load_ur5_rerun, ur5_home, get_ur5_gripper_spheres, get_ur5_ik_solver, get_ur5_kinematics_model
from .utils import RerunRobot
from .t1 import (
    load_t1_rerun,
    t1_home_left,
    t1_home_right,
    get_t1_gripper_spheres,
    get_t1_kinematics_model,
    curobo_to_urdf_joints,
)


@dataclass(frozen=True)
class RobotContainer:
    name: str
    kin_model: CudaRobotModel
    joint_limits: Float[torch.Tensor, "2 d"]
    # Note: in tool frame, not end-effector
    gripper_spheres: Float[torch.Tensor, "n 4"]
    # Transformation from end-effector frame to tool/grasp frame (defined in cuRobo config)
    tool_from_ee: Float[torch.Tensor, "4 4"]


@dataclass(frozen=True)
class DualArmRobotContainer:
    """
    Container for dual-arm robot configuration (e.g., T1 humanoid).
    
    This container holds all the necessary models and configurations for
    a dual-arm robot where each arm can be controlled independently.
    
    Attributes:
        name: Robot identifier (e.g., "t1")
        left_kin_model: cuRobo kinematics model for left arm
        right_kin_model: cuRobo kinematics model for right arm
        left_joint_limits: Joint position limits for left arm, shape (2, DOF)
                          where [0] is lower limits and [1] is upper limits
        right_joint_limits: Joint position limits for right arm
        left_gripper_spheres: Collision spheres for left gripper, shape (N, 4)
                             where each row is [x, y, z, radius] in gripper frame
        right_gripper_spheres: Collision spheres for right gripper
        left_tool_from_ee: 4x4 homogeneous transform from left EE to tool frame
        right_tool_from_ee: 4x4 homogeneous transform from right EE to tool frame
    
    Note:
        For T1 robot, each arm has 11 DOF in the cuRobo model:
        - 2 DOF lifting column
        - 2 DOF torso
        - 7 DOF arm
    """
    name: str
    # Kinematic models for each arm
    left_kin_model: CudaRobotModel
    right_kin_model: CudaRobotModel
    # Joint limits for each arm, shape (2, DOF) - [lower, upper]
    left_joint_limits: Float[torch.Tensor, "2 d"]
    right_joint_limits: Float[torch.Tensor, "2 d"]
    # Gripper spheres for each arm, shape (N, 4) - [x, y, z, radius]
    left_gripper_spheres: Float[torch.Tensor, "n 4"]
    right_gripper_spheres: Float[torch.Tensor, "n 4"]
    # Tool frame transformations (EE frame → grasp/tool frame)
    left_tool_from_ee: Float[torch.Tensor, "4 4"]
    right_tool_from_ee: Float[torch.Tensor, "4 4"]
    

def load_panda_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_franka_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 7), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_franka_gripper_spheres(tensor_args)
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    gripper_down_quat = tensor_args.to_device([0.0, 1.0, 0.0, 0.0])
    tool_from_ee[:3, :3] = quaternion_to_matrix(gripper_down_quat[None])[0]
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.105])
    return RobotContainer("panda", kin_model, joint_limits, gripper_spheres, tool_from_ee)


def load_ur5_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_ur5_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 6), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_ur5_gripper_spheres(tensor_args)
    # See screenshot in assets/ur5_home.png to see gripper frame
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    rpy = tensor_args.to_device([torch.pi, 0, torch.pi / 2])
    tool_from_ee[:3, :3] = roma.euler_to_rotmat("XYZ", rpy)
    # The Robotiq gripper goes down when closing, so we move the tool frame up by 1cm
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.01])
    return RobotContainer("ur5", kin_model, joint_limits, gripper_spheres, tool_from_ee)


def load_t1_container(tensor_args: TensorDeviceType) -> DualArmRobotContainer:
    """
    Load T1 dual-arm robot container with all necessary configurations.
    
    Creates a DualArmRobotContainer with:
    - Kinematics models for left and right arms (11 DOF each)
    - Joint limits for each arm
    - Gripper collision spheres for grasp planning
    - Tool frame transformations for top-down grasping
    
    Tool Frame Convention (same as Panda and UR5):
        tool_from_ee transforms FROM the robot's EE frame TO the grasp/tool frame.
        Usage: world_from_ee = world_from_grasp @ tool_from_ee
        
        Grasp frame has +Z pointing UP (world vertical).
        For top-down grasps, the gripper approaches from above.
    
    T1 EE Frame Convention (left_base_link / right_base_link):
        - EE link is left_base_link (or right_base_link for right arm)
        - In the EE frame, +X points toward the fingertips
        - Ry(+90°) maps EE +X → Grasp -Z, so the gripper points DOWN
        
        Compare to Panda: uses Rx(180°) because Panda's EE has -Z toward fingertips
        Compare to UR5: uses Rx(180°) @ Rz(90°) for its gripper orientation
    
    Args:
        tensor_args: Device configuration for tensors
        
    Returns:
        DualArmRobotContainer configured for T1 robot
    """
    # Load kinematics models for both arms
    left_kin_model = get_t1_kinematics_model("left")
    right_kin_model = get_t1_kinematics_model("right")
    
    # Get joint limits (should be (2, 11) for T1: 2 lift + 2 torso + 7 arm)
    left_joint_limits = left_kin_model.kinematics_config.joint_limits.position
    right_joint_limits = right_kin_model.kinematics_config.joint_limits.position
    assert left_joint_limits.shape == (2, 11), f"Invalid left joint limits shape: {left_joint_limits.shape}"
    assert right_joint_limits.shape == (2, 11), f"Invalid right joint limits shape: {right_joint_limits.shape}"
    
    # Load gripper spheres for both arms (used for grasp collision checking)
    left_gripper_spheres = get_t1_gripper_spheres("left", tensor_args)
    right_gripper_spheres = get_t1_gripper_spheres("right", tensor_args)
    
    # Build tool_from_ee for left arm (left_base_link)
    # In left_base_link frame, +X points toward fingertips
    # Ry(+90°) rotation: EE +X → Grasp -Z (so gripper points DOWN for top-down grasps)
    rotvec_y_pos90 = torch.tensor([[0, torch.pi/2, 0]], device=tensor_args.device, dtype=torch.float32)
    grasp_from_ee = roma.rotvec_to_rotmat(rotvec_y_pos90)[0]
    
    left_tool_from_ee = torch.eye(4, device=tensor_args.device, dtype=torch.float32)
    left_tool_from_ee[:3, :3] = grasp_from_ee
    # NOTE: Zero translation assumes left_base_link origin is at the grasp point.
    # If T1 gripper has offset from left_base_link to fingertips, adjust this value.
    # Compare: Panda uses 0.105m, UR5 uses 0.01m
    left_tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.0])
    
    # Build tool_from_ee for right arm (right_base_link)
    # Although left_base_link and right_base_link have different yaw orientations in URDF
    # (+90° vs -90° relative to their hand_links), BOTH frames have the same local convention:
    # +X points toward fingertips (see finger joint origins: xyz="0.036 ...").
    # Therefore, both arms use the same tool_from_ee transformation.
    right_tool_from_ee = left_tool_from_ee.clone()
    
    return DualArmRobotContainer(
        name="t1",
        left_kin_model=left_kin_model,
        right_kin_model=right_kin_model,
        left_joint_limits=left_joint_limits,
        right_joint_limits=right_joint_limits,
        left_gripper_spheres=left_gripper_spheres,
        right_gripper_spheres=right_gripper_spheres,
        left_tool_from_ee=left_tool_from_ee,
        right_tool_from_ee=right_tool_from_ee,
    )



robot_to_fns = {
    "panda": {
        "rerun": load_franka_rerun,
        "q_home": franka_neutral_joint_positions[:7],  # exclude gripper joint
        "container": load_panda_container,
    },
    "ur5": {
        "rerun": load_ur5_rerun,
        "q_home": ur5_home[:6],
        "container": load_ur5_container,
    },
    "t1": {
        "rerun": load_t1_rerun,
        "q_home": t1_home_left,  # Default to left arm home position (11 DOF)
        "q_home_left": t1_home_left,
        "q_home_right": t1_home_right,
        "container": load_t1_container,
    },
}


def load_rerun_robot(robot: str, load_mesh: bool = True) -> RerunRobot:
    if robot not in robot_to_fns:
        raise ValueError(f"Unknown robot: {robot}. Supported robots: {list(robot_to_fns.keys())}")
    rerun_fn = robot_to_fns[robot]["rerun"]
    rerun_robot = rerun_fn(load_mesh)
    if not isinstance(rerun_robot, RerunRobot):
        raise TypeError(f"Expected RerunRobot, got {type(rerun_robot)}")
    return rerun_robot


def get_q_home(robot: str) -> Sequence[float]:
    """Get the home joint positions for the specified robot."""
    if robot not in robot_to_fns:
        raise ValueError(f"Unknown robot: {robot}. Supported robots: {list(robot_to_fns.keys())}")
    q_home = robot_to_fns[robot]["q_home"]
    return q_home


def load_robot_container(
    robot: str, tensor_args: TensorDeviceType
    ) -> Union[RobotContainer, DualArmRobotContainer]:
    """Load robot container which contains many helper classes and variables."""
    if robot not in robot_to_fns:
        raise ValueError(f"Unknown robot: {robot}. Supported robots: {list(robot_to_fns.keys())}")
    container_fn = robot_to_fns[robot]["container"]
    container = container_fn(tensor_args)
    if not isinstance(container, (RobotContainer, DualArmRobotContainer)):
        raise TypeError(f"Expected RobotContainer or DualArmRobotContainer, got {type(container)}")
    return container
