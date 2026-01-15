#!/usr/bin/env python3
"""
Debug script for T1 tool frame transformation.

This script verifies that the tool_from_ee transformation is correctly
configured for top-down grasping with the T1 robot.

Key Concept:
    tool_from_ee transforms FROM the robot's EE frame TO the grasp/tool frame.
    For top-down grasps:
        world_from_ee = world_from_grasp @ tool_from_ee

    Grasp Frame Convention:
        - +Z points UP (world vertical)
        - Origin is at the grasp point on the object
        - Yaw rotation controls gripper orientation around vertical axis

    Expected Result for Top-Down Grasp:
        - Both arms: EE +X (fingertips direction) should point DOWN in world frame
        - Both left_base_link and right_base_link have +X toward fingertips

Run with: python -m cutamp.tests.debug_t1_tool_frame
"""

import torch
import roma
import numpy as np
import rerun as rr
from curobo.types.base import TensorDeviceType
from curobo.geom.transform import quaternion_to_matrix

np.set_printoptions(precision=4, suppress=True)

AXIS_LENGTH = 0.05


def log_frame_to_rerun(path: str, transform_4x4: torch.Tensor, label: str = None):
    """Log a coordinate frame to rerun with axes visualization."""
    transform_4x4_np = transform_4x4.cpu().numpy()
    rr.log(
        path,
        rr.Transform3D(
            translation=transform_4x4_np[:3, 3],
            mat3x3=transform_4x4_np[:3, :3],
            axis_length=AXIS_LENGTH,
        ),
    )


def verify_top_down_grasp(tool_from_ee: torch.Tensor, world_from_grasp: torch.Tensor) -> bool:
    """
    Verify that a top-down grasp configuration has gripper pointing DOWN.
    
    Args:
        tool_from_ee: 4x4 transform from EE frame to grasp frame
        world_from_grasp: 4x4 transform from grasp frame to world frame
        
    Returns:
        True if gripper points DOWN (EE +X has Z component < -0.9)
    
    Note:
        T1's EE frame (left_base_link / right_base_link) has +X pointing
        toward the fingertips. Both arms use the same convention.
    """
    world_from_ee = world_from_grasp @ tool_from_ee
    ee_x_world = world_from_ee[:3, 0].cpu().numpy()
    fingertips_dir_world = ee_x_world  # +X points toward fingertips in T1 EE frame
    return fingertips_dir_world[2] < -0.9


def main():
    print("=" * 60)
    print("T1 Tool Frame Transformation Debug")
    print("=" * 60)
    
    # Initialize rerun
    rr.init("T1 Tool Frame Debug", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    tensor_args = TensorDeviceType()
    
    # =========================================================================
    # 1. Load T1 models
    # =========================================================================
    print("\n[1] Loading T1 models...")
    from cutamp.robots import load_t1_container
    from cutamp.robots.t1 import get_t1_kinematics_model, t1_home_left, load_t1_rerun
    from cutamp.utils.common import action_4dof_to_mat4x4
    
    container = load_t1_container(tensor_args)
    left_kin = get_t1_kinematics_model("left")
    rerun_robot = load_t1_rerun(load_mesh=True, arm="left")
    rerun_robot.set_joint_positions(t1_home_left)
    print("    Models loaded successfully")
    
    # =========================================================================
    # 2. Test top-down grasp transformation
    # =========================================================================
    print("\n[2] Testing top-down grasp transformation...")
    
    tool_from_ee = container.left_tool_from_ee
    print(f"\n    tool_from_ee rotation (Ry(+90°)):")
    print(tool_from_ee[:3, :3].cpu().numpy())
    
    # Test grasp at position [0.5, 0, 0.3] with yaw=0
    grasp_4dof = torch.tensor([[0.5, 0.0, 0.3, 0.0]], device=tensor_args.device, dtype=torch.float32)
    world_from_grasp = action_4dof_to_mat4x4(grasp_4dof)[0]
    world_from_ee = world_from_grasp @ tool_from_ee
    
    # Verify gripper direction
    # T1 EE frame (left_base_link / right_base_link) has +X pointing toward fingertips
    ee_x_world = world_from_ee[:3, 0].cpu().numpy()
    fingertips_dir = ee_x_world  # +X points toward fingertips
    gripper_down = fingertips_dir[2] < -0.9
    
    print(f"\n    Test grasp: position=[0.5, 0, 0.3], yaw=0")
    print(f"    EE +X (toward fingertips) in world: {fingertips_dir}")
    print(f"    Gripper points DOWN: {'PASS' if gripper_down else 'FAIL'}")
    
    # Visualize frames
    log_frame_to_rerun("test/grasp_frame", world_from_grasp)
    log_frame_to_rerun("test/desired_ee", world_from_ee)
    
    # =========================================================================
    # 3. IK feasibility test
    # =========================================================================
    print("\n[3] IK feasibility test...")
    
    try:
        from cutamp.robots.t1 import get_t1_ik_solver
        from curobo.types.math import Pose
        
        ik_solver = get_t1_ik_solver("left", world_cfg=None)
        
        # Test grasp at a reachable position
        grasp_4dof = torch.tensor([[0.5, 0.0, 0.5, 0.0]], device=tensor_args.device, dtype=torch.float32)
        world_from_grasp = action_4dof_to_mat4x4(grasp_4dof)[0]
        world_from_ee = world_from_grasp @ tool_from_ee
        
        # Extract position and quaternion for IK
        pos = world_from_ee[:3, 3][None]
        rotmat = world_from_ee[:3, :3]
        quat_xyzw = roma.rotmat_to_unitquat(rotmat[None])  # roma returns xyzw format
        # cuRobo expects wxyz format, so reorder: [x,y,z,w] -> [w,x,y,z]
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        
        goal_pose = Pose(pos, quat_wxyz)
        result = ik_solver.solve_batch(goal_pose)
        
        ik_success = result.success.any().item()
        print(f"\n    Test grasp at [0.5, 0, 0.5], yaw=0:")
        print(f"    IK Success: {'PASS' if ik_success else 'FAIL'}")
        print(f"    Position error: {result.position_error.min().item():.4f} m")
        print(f"    Rotation error: {result.rotation_error.min().item():.4f} rad")
        
        if ik_success:
            # Visualize IK solution
            q_sol = result.solution[result.success][0:1]
            achieved_state = left_kin.get_state(q_sol)
            achieved_rotmat = quaternion_to_matrix(achieved_state.ee_pose.quaternion)[0]
            achieved_pos = achieved_state.ee_pose.position[0]
            
            achieved_ee_transform = torch.eye(4, device=tensor_args.device)
            achieved_ee_transform[:3, :3] = achieved_rotmat
            achieved_ee_transform[:3, 3] = achieved_pos
            
            # Verify achieved pose also has gripper pointing down
            # T1 EE frame has +X pointing toward fingertips
            achieved_x = achieved_rotmat[:, 0].cpu().numpy()
            achieved_fingertips_dir = achieved_x  # +X points toward fingertips
            achieved_gripper_down = achieved_fingertips_dir[2] < -0.9
            
            print(f"    Achieved EE +X direction: {achieved_fingertips_dir}")
            print(f"    Achieved gripper DOWN: {'PASS' if achieved_gripper_down else 'FAIL'}")
            
            log_frame_to_rerun("ik_test/achieved_ee", achieved_ee_transform)
            log_frame_to_rerun("ik_test/desired_ee", world_from_ee)
            rerun_robot.set_joint_positions(q_sol[0].cpu().tolist())
            
    except Exception as e:
        print(f"\n    IK test skipped: {e}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nThe tool_from_ee transformation for T1 uses Ry(+90°) which maps:")
    print("  - EE +X → Grasp -Z (so gripper points DOWN for top-down grasps)")
    print("  - EE +Y → Grasp +Y (horizontal)")
    print("  - EE +Z → Grasp +X (horizontal)")
    print("\nT1's EE frame (left_base_link / right_base_link) has +X toward fingertips.")
    print("Both arms use the same tool_from_ee transformation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
