"""
T1 Robot Module - Dual-Arm Humanoid Support for cuTAMP
=======================================================

This module provides cuRobo integration for the Booster T1 humanoid robot.

Robot Configuration:
    - Full URDF: 28 actuated joints
    - cuRobo model: 11 DOF per arm (lifting column + torso + arm)
    - Gripper: 4-bar parallel linkage (4 joints, 1 effective DOF)

DOF Breakdown per arm:
    - Lifting column: 2 DOF (waist_lift_1, waist_lift_2)
    - Torso: 2 DOF (Torso_Pitch, Waist_Yaw)
    - Arm: 7 DOF (shoulder, elbow, wrist joints)
    - Total: 11 DOF for cuRobo motion planning

Configuration Files:
    - t1_left_11dof.yml: Left arm active, right arm locked
    - t1_right_11dof.yml: Right arm active, left arm locked

Gripper Mechanism:
    The gripper uses a 4-bar parallel linkage where fingers close by rotating
    around the local Z-axis (axis xyz="0 0 1" in URDF). All 4 gripper joints
    should be set to 0.0 for a closed gripper configuration.

Usage:
    from cutamp.robots.t1 import get_t1_kinematics_model, get_t1_ik_solver
    
    # Get kinematics model for left arm
    kin_left = get_t1_kinematics_model("left")
    
    # Get IK solver for right arm
    ik_right = get_t1_ik_solver("right", world_cfg=None)
"""

import logging
from pathlib import Path
from functools import lru_cache
from typing import Literal, Tuple

import torch
from jaxtyping import Float
from yourdfpy import URDF

from curobo.types.base import TensorDeviceType

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from curobo.geom.types import WorldConfig
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from cutamp.robots.utils import RerunRobot

_log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# DOF counts
CUROBO_DOF = 11  # DOF per arm in cuRobo model
URDF_DOF = 28    # Total actuated joints in full URDF

# Joint indices in URDF (28 joints total)
URDF_LIFT_IDX = (0, 1)           # Lifting column joints
URDF_TORSO_IDX = (2, 3)          # Torso joints
URDF_HEAD_IDX = (4, 5)           # Head joints (locked)
URDF_LEFT_ARM_IDX = (6, 12)      # Left arm joints (inclusive start, exclusive end)
URDF_LEFT_GRIP_IDX = (13, 16)    # Left gripper joints (inclusive start, exclusive end)
URDF_RIGHT_ARM_IDX = (17, 23)    # Right arm joints (inclusive start, exclusive end)
URDF_RIGHT_GRIP_IDX = (24, 27)   # Right gripper joints (inclusive start, exclusive end)

# Gripper joint configuration (4-bar parallel linkage)
# The gripper has 4 joints but only 1 effective DOF due to linkage:
#   Link11 mirrors -Link1 (finger 1 tip compensation)
#   Link22 mirrors -Link2 (finger 2 tip compensation)
GRIPPER_OPEN = (0.0, 0.0, 0.0, 0.0)
GRIPPER_CLOSED = (1.0, -1.0, -1.0, 1.0)
GRIPPER_JOINTS_PER_HAND = 4

# Home positions for 11 DOF arms
# Joint order: [lift1, lift2, torso_pitch, waist_yaw, shoulder_p, shoulder_r,
#               elbow_p, elbow_y, wrist_p, wrist_y, hand_r]
t1_home_left: Tuple[float, ...] = (0.15, 0.15, 0.0, 0.0, 0.5, -1.0, 0.0, -1.4, 0.0, 0.0, 0.0)
t1_home_right: Tuple[float, ...] = (0.15, 0.15, 0.0, 0.0, 0.5, 1.0, 0.0, 1.4, 0.0, 0.0, 0.0)
# Joints: AAHead_yaw, Head_pitch
t1_home_head: Tuple[float, ...] = (0.0, 0.0)

# Number of shared joints (lift + torso) in 11-DOF arm config
NUM_SHARED_JOINTS = 4

# Joint names for arm-specific joints (indices 4-10 in 11-DOF config)
# These are used for updating locked joints in dual-arm motion planning
LEFT_ARM_JOINT_NAMES: Tuple[str, ...] = (
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
    "Left_Elbow_Yaw", "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll"
)
RIGHT_ARM_JOINT_NAMES: Tuple[str, ...] = (
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
    "Right_Elbow_Yaw", "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll"
)

# Collision link names for each arm (used for disabling collision in dual-arm planning)
LEFT_ARM_COLLISION_LINKS: Tuple[str, ...] = (
    "AL1", "AL2", "AL3", "AL4", "AL5", "AL6",
    "left_hand_link", "left_base_link",
    "left_Link1", "left_Link11", "left_Link2", "left_Link22",
)
RIGHT_ARM_COLLISION_LINKS: Tuple[str, ...] = (
    "AR1", "AR2", "AR3", "AR4", "AR5", "AR6",
    "right_hand_link", "right_base_link",
    "right_Link1", "right_Link11", "right_Link2", "right_Link22",
)

# Path to T1 robot assets
T1_ASSETS_DIR = Path(__file__).parent / "assets" / "t1_description"


def _get_t1_config_path(arm: Literal["left", "right"]) -> str:
    """Get path to T1 config file for specified arm."""
    if arm == "left":
        return str(T1_ASSETS_DIR / "t1_left_11dof.yml")
    elif arm == "right":
        return str(T1_ASSETS_DIR / "t1_right_11dof.yml")
    else:
        raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")

@lru_cache(maxsize=2)
def t1_curobo_cfg(arm: Literal["left", "right"]) -> dict:
    """
    Load cuRobo configuration for T1 robot.
    
    Args:
        arm: Which arm config to load ("left" or "right")
        
    Returns:
        Configuration dictionary with robot_cfg key
    """
    config_path = _get_t1_config_path(arm)
    cfg = load_yaml(config_path)
    
    # Resolve all paths to absolute paths so cuRobo can find t1 assets
    kinematics = cfg["robot_cfg"]["kinematics"]
    
    # Resolve URDF path to absolute
    if isinstance(kinematics.get("urdf_path"), str):
        kinematics["urdf_path"] = str(T1_ASSETS_DIR / kinematics["urdf_path"])
    
    # Resolve collision_spheres to absolute path
    if isinstance(kinematics.get("collision_spheres"), str):
        kinematics["collision_spheres"] = str(T1_ASSETS_DIR / kinematics["collision_spheres"])
    
    # Set asset_root_path for mesh loading (will be derived from urdf_path dirname if not set)
    kinematics["asset_root_path"] = str(T1_ASSETS_DIR)
    
    return cfg

def t1_left_curobo_cfg() -> dict:
    """Load left arm cuRobo configuration."""
    return t1_curobo_cfg("left")

def t1_right_curobo_cfg() -> dict:
    """Load right arm cuRobo configuration."""
    return t1_curobo_cfg("right")

def _t1_cfg_dict(arm: Literal["left", "right"]) -> dict:
    """Get robot_cfg dictionary for specified arm."""
    return t1_curobo_cfg(arm)["robot_cfg"]

def get_t1_kinematics_model(arm: Literal["left", "right"]) -> CudaRobotModel:
    """
    Get cuRobo kinematics model for T1 robot.
    
    Args:
        arm: Which arm's kinematics model to get ("left" or "right")
        
    Returns:
        CudaRobotModel configured for the specified arm
    """
    if arm not in ("left", "right"):
        raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")
    
    robot_cfg = _t1_cfg_dict(arm)
    robot_cfg = RobotConfig.from_dict(robot_cfg)
    kinematics_model = CudaRobotModel(robot_cfg.kinematics)
    return kinematics_model

def get_t1_ik_solver(
    arm: Literal["left", "right"],
    world_cfg: WorldConfig,
    num_seeds: int = 32, 
    self_collision_opt: bool = True,
    self_collision_check: bool = True,
    use_particle_opt: bool = True,
) -> IKSolver:
    """
    Create cuRobo IK solver for T1 robot.
    
    Args:
        arm: Which arm's IK solver to create ("left" or "right")
        world_cfg: World configuration for collision checking (can be None)
        num_seeds: Number of IK seeds to use
        self_collision_opt: Enable self-collision optimization
        self_collision_check: Enable self-collision checking
        use_particle_opt: Enable particle-based optimization
        
    Returns:
        IKSolver configured for the specified arm
    """
    if arm not in ("left", "right"):
        raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")
    
    robot_cfg = _t1_cfg_dict(arm)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
    )
    ik_solver = IKSolver(ik_config)
    return ik_solver

def get_t1_gripper_spheres(
    arm: Literal["left", "right"],
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> Float[torch.Tensor, "num_spheres 4"]:
    """
    Get collision spheres for T1 gripper.
    
    The spheres are defined in the gripper base link frame (left_base_link or right_base_link).
    Each sphere is [x, y, z, radius].
    
    Args:
        arm: Which gripper's spheres to load ("left" or "right")
        tensor_args: Device configuration for the returned tensor
        
    Returns:
        Tensor of shape (N, 4) with sphere centers and radii
    """
    if arm not in ("left", "right"):
        raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")
    
    spheres_file = T1_ASSETS_DIR / f"{arm}_gripper_spheres.pt"
    
    if not spheres_file.exists():
        raise FileNotFoundError(
            f"T1 {arm} gripper spheres file not found at {spheres_file}. "
            "Please generate it using cutamp/scripts/gripper_sphere_editor.py"
        )
    
    spheres = torch.load(spheres_file, map_location=tensor_args.device, weights_only=True)
    assert spheres.ndim == 2 and spheres.shape[1] == 4, \
        f"Invalid shape for T1 {arm} gripper spheres: {spheres.shape}"
    
    # Filter out any spheres with non-positive radius
    spheres = spheres[spheres[:, 3] > 0]
    return spheres

def get_t1_home(arm: Literal["left", "right"]) -> tuple:
    """
    Get home joint positions for T1 robot.
    
    Args:
        arm: Which arm's home position to get ("left" or "right")
        
    Returns:
        Tuple of 11 joint positions
    """
    if arm == "left":
        return t1_home_left
    elif arm == "right":
        return t1_home_right
    else:
        raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")

def curobo_to_urdf_joints(
    q_curobo: tuple | list,
    arm: Literal["left", "right"] = "left",
    gripper: tuple = GRIPPER_OPEN,
) -> tuple:
    """
    Map cuRobo's 11-DOF joint configuration to the full URDF's 28-DOF configuration.
    
    This function is essential for visualization because:
    - cuRobo uses a simplified 11-DOF model for motion planning
    - Rerun visualization requires the full 28-DOF URDF configuration
    
    cuRobo 11-DOF order:
        [lift1, lift2, torso_pitch, waist_yaw, shoulder_p, shoulder_r, 
         elbow_p, elbow_y, wrist_p, wrist_y, hand_r]
    
    URDF 28-DOF order (see URDF_*_IDX constants):
        [0-1]   Lifting column: shared between both configs
        [2-3]   Torso: shared between both configs
        [4-5]   Head: locked (not in cuRobo model)
        [6-12]  Left arm: 7 DOF
        [13-16] Left gripper: 4 joints (4-bar linkage, closed)
        [17-23] Right arm: 7 DOF
        [24-27] Right gripper: 4 joints (4-bar linkage, closed)
    
    Args:
        q_curobo: 11-DOF joint configuration from cuRobo
        arm: Which arm the configuration is for ("left" or "right")
        
    Returns:
        28-DOF joint configuration for URDF visualization
        
    Raises:
        ValueError: If q_curobo doesn't have exactly 11 elements or arm is invalid
    """
    if len(q_curobo) != CUROBO_DOF:
        raise ValueError(f"Expected {CUROBO_DOF} DOF from cuRobo, got {len(q_curobo)}")
    
    # Extract components from cuRobo configuration
    lift = q_curobo[0:2]         # 2 DOF: lifting column
    torso = q_curobo[2:4]        # 2 DOF: torso
    active_arm = q_curobo[4:11]  # 7 DOF: arm joints
    
    if arm == "left":
        # Left arm active, right arm at neutral
        urdf_joints = (
            *lift,                                # 0-1: lifting column
            *torso,                               # 2-3: torso
            *t1_home_head,                        # 4-5: head (locked)
            *active_arm,                          # 6-12: left arm (active)
            *gripper,                             # 13-16: left gripper (open)
            *t1_home_right[4:11],                 # 17-23: right arm (neutral)
            *GRIPPER_OPEN,                        # 24-27: right gripper (open)
        )
    elif arm == "right":
        # Right arm active, left arm at home position
        urdf_joints = (
            *lift,                                # 0-1: lifting column
            *torso,                               # 2-3: torso
            *t1_home_head,                        # 4-5: head (locked)
            *t1_home_left[4:11],                  # 6-12: left arm (at home)
            *GRIPPER_OPEN,                        # 13-16: left gripper (open)
            *active_arm,                          # 17-23: right arm (active)
            *gripper,                             # 24-27: right gripper (open)
        )
    else:
        raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")
    
    assert len(urdf_joints) == URDF_DOF, f"Expected {URDF_DOF} joints, got {len(urdf_joints)}"
    return urdf_joints


def curobo_dual_arm_to_urdf_joints(
    q_left: tuple | list, 
    q_right: tuple | list,
    left_gripper: tuple = GRIPPER_OPEN,
    right_gripper: tuple = GRIPPER_OPEN,
) -> tuple:
    """
    Map both arms' cuRobo 11-DOF joint configurations to the full URDF's 28-DOF configuration.
    
    This function is essential for proper dual-arm visualization because it combines
    both arms' configurations while ensuring shared joints (lift + torso) are consistent.
    
    Note: The shared joints (lift + torso) from q_left are used. Caller should ensure
    q_left and q_right have consistent shared joint values (first 4 joints).
    
    Args:
        q_left: 11-DOF joint configuration for left arm from cuRobo
        q_right: 11-DOF joint configuration for right arm from cuRobo
        left_gripper: 4 gripper joint values for left hand (default: open)
        right_gripper: 4 gripper joint values for right hand (default: open)
    
    Returns:
        28-DOF joint configuration for URDF visualization
    
    Raises:
        ValueError: If configurations don't have exactly 11 elements
    """
    if len(q_left) != CUROBO_DOF:
        raise ValueError(f"Expected {CUROBO_DOF} DOF for left arm, got {len(q_left)}")
    if len(q_right) != CUROBO_DOF:
        raise ValueError(f"Expected {CUROBO_DOF} DOF for right arm, got {len(q_right)}")
    
    # Extract components from left arm (use its shared joints as canonical)
    lift = q_left[0:2]           # 2 DOF: lifting column (shared)
    torso = q_left[2:4]          # 2 DOF: torso (shared)
    left_arm = q_left[4:11]      # 7 DOF: left arm joints
    right_arm = q_right[4:11]    # 7 DOF: right arm joints
    
    urdf_joints = (
        *lift,                    # 0-1: lifting column
        *torso,                   # 2-3: torso
        *t1_home_head,            # 4-5: head (locked)
        *left_arm,                # 6-12: left arm
        *left_gripper,            # 13-16: left gripper
        *right_arm,               # 17-23: right arm
        *right_gripper,           # 24-27: right gripper
    )
    
    assert len(urdf_joints) == URDF_DOF, f"Expected {URDF_DOF} joints, got {len(urdf_joints)}"
    return urdf_joints


class T1RerunRobot(RerunRobot):
    """
    T1-specific RerunRobot that handles mapping from cuRobo's DOF to URDF's 28-DOF.
    
    Supports multiple input formats:
    - 11 DOF: Single arm cuRobo format (uses active_arm to determine which arm)
    - 15 DOF: Single arm (11) + gripper (4) - gripper applied to specified arm
    - 22 DOF: Dual arm cuRobo format (left[11] + right[11])
    - 28 DOF: Full URDF format (passed through directly)
    """
    
    def __init__(self, name: str, urdf: URDF, q_neutral, load_mesh: bool = True, arm: Literal["left", "right"] = "left"):
        self.active_arm = arm
        super().__init__(name, urdf, q_neutral, load_mesh)
    
    def set_joint_positions(self, joint_positions, arm: Literal["left", "right"] = None) -> None:
        """Override to handle cuRobo DOF to URDF's 28-DOF mapping."""
        if hasattr(joint_positions, 'tolist'):
            joint_positions = joint_positions.tolist()
        elif hasattr(joint_positions, '__iter__'):
            joint_positions = list(joint_positions)
        
        # Use provided arm or fall back to default active_arm
        active_arm = arm if arm is not None else self.active_arm
        
        # Handle different input sizes
        if len(joint_positions) == 11:
            # Single arm cuRobo format - use active_arm to determine mapping
            joint_positions = curobo_to_urdf_joints(joint_positions, active_arm)

        elif len(joint_positions) == 15:
            # Single arm (11) + gripper (4) - gripper applied to active arm
            q_arm = joint_positions[:11]
            gripper = tuple(joint_positions[11:15])
            joint_positions = curobo_to_urdf_joints(q_arm, active_arm, gripper=gripper)

        elif len(joint_positions) == 22:
            # Dual arm cuRobo format: left[11] + right[11]
            q_left = joint_positions[:11]
            q_right = joint_positions[11:]
            joint_positions = curobo_dual_arm_to_urdf_joints(q_left, q_right)
        # else: assume it's already 28-DOF URDF format
        
        # Call parent implementation
        super().set_joint_positions(joint_positions)
    
    def get_rr_columns(self, joint_positions: Float[torch.Tensor, "n d"], arm: Literal["left", "right"] = None):
        """
        Override to handle cuRobo DOF to URDF's 28-DOF mapping for batched positions.
        Supported input formats (same as set_joint_positions):
        - 11 DOF: Single arm cuRobo format
        - 15 DOF: Single arm (11) + gripper (4)
        - 22 DOF: Dual arm cuRobo format
        - 28 DOF: Full URDF format
        
        Args:
            joint_positions: Batched joint positions tensor
            arm: Which arm to use for mapping. If None, uses self.active_arm
        """
        # Use provided arm or fall back to default active_arm
        active_arm = arm if arm is not None else self.active_arm
        
        dof = joint_positions.shape[-1]
        
        if dof == 11:
            # Single arm: map each to 28-DOF
            mapped_positions = []
            for q in joint_positions:
                q_mapped = curobo_to_urdf_joints(q.tolist(), active_arm)
                mapped_positions.append(q_mapped)
            joint_positions = torch.tensor(mapped_positions, dtype=joint_positions.dtype, device=joint_positions.device)
        
        elif dof == 15:
            # Single arm (11) + gripper (4) - gripper applied to active arm
            mapped_positions = []
            for q in joint_positions:
                q_arm = q[:11].tolist()
                gripper = tuple(q[11:15].tolist())
                q_mapped = curobo_to_urdf_joints(q_arm, active_arm, gripper=gripper)
                mapped_positions.append(q_mapped)
            joint_positions = torch.tensor(mapped_positions, dtype=joint_positions.dtype, device=joint_positions.device)
        
        elif dof == 22:
            # Dual arm: map each pair to 28-DOF
            mapped_positions = []
            for q in joint_positions:
                q_left = q[:11].tolist()
                q_right = q[11:].tolist()
                q_mapped = curobo_dual_arm_to_urdf_joints(q_left, q_right)
                mapped_positions.append(q_mapped)
            joint_positions = torch.tensor(mapped_positions, dtype=joint_positions.dtype, device=joint_positions.device)
        
        elif dof == 26:
            # Dual arm (22) + gripper (4) for active arm
            mapped_positions = []
            for q in joint_positions:
                q_left = q[:11].tolist()
                q_right = q[11:22].tolist()
                gripper = tuple(q[22:26].tolist())
                if active_arm == "left":
                    q_mapped = curobo_dual_arm_to_urdf_joints(q_left, q_right, left_gripper=gripper)
                else:
                    q_mapped = curobo_dual_arm_to_urdf_joints(q_left, q_right, right_gripper=gripper)
                mapped_positions.append(q_mapped)
            joint_positions = torch.tensor(mapped_positions, dtype=joint_positions.dtype, device=joint_positions.device)
        # else: assume it's already 28-DOF URDF format
        
        return super().get_rr_columns(joint_positions)


def load_t1_rerun(load_mesh: bool = True, arm: Literal["left", "right"] = "left") -> T1RerunRobot:
    """
    Load T1 robot for Rerun visualization.
    
    Note: This loads the full robot URDF (not the simplified one with single arm active).
    For visualization, we show the complete robot.
    
    Args:
        load_mesh: Whether to load mesh geometries for visualization
        arm: Which arm is active ("left" or "right") - affects joint mapping
        
    Returns:
        T1RerunRobot configured for T1 with proper joint mapping
    """
    # Use the simplified URDF for visualization
    urdf_path = T1_ASSETS_DIR / "t1_simplified.urdf"
    
    if not urdf_path.exists():
        raise FileNotFoundError(f"T1 URDF not found at {urdf_path}")
    
    def _locate_t1_asset(fname: str) -> str:
        """Resolve asset paths for T1 robot."""
        if fname.startswith("package://"):
            # Handle package:// URIs
            return str(T1_ASSETS_DIR / fname.replace("package://", ""))
        
        # Try relative path from URDF location
        full_path = T1_ASSETS_DIR / fname
        if full_path.exists():
            return str(full_path)
        
        return fname
    
    urdf = URDF.load(str(urdf_path), filename_handler=_locate_t1_asset)
    
    full_neutral = curobo_to_urdf_joints(t1_home_left, "left")
    
    return T1RerunRobot("t1", urdf, q_neutral=full_neutral, load_mesh=load_mesh, arm=arm)