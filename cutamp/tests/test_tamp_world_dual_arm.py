"""
Unit tests for TAMPWorld dual-arm helper methods.

These tests verify that the arm-aware accessor methods work correctly
for both single-arm (Panda, UR5) and dual-arm (T1) robots.
"""

import pytest
import torch
from curobo.types.base import TensorDeviceType

from cutamp.robots import load_robot_container, DualArmRobotContainer, RobotContainer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tensor_args():
    return TensorDeviceType()


@pytest.fixture(scope="module")
def t1_container(tensor_args):
    """Load T1 dual-arm robot container."""
    return load_robot_container("t1", tensor_args)


# =============================================================================
# Tests for DualArmRobotContainer accessors
# =============================================================================

class TestDualArmContainerAccessors:
    """Test that DualArmRobotContainer has all required attributes."""

    def test_t1_is_dual_arm_container(self, t1_container):
        """Verify T1 is a DualArmRobotContainer."""
        assert isinstance(t1_container, DualArmRobotContainer)

    def test_t1_has_left_kin_model(self, t1_container):
        """Verify T1 has left kinematics model."""
        assert hasattr(t1_container, "left_kin_model")
        assert t1_container.left_kin_model is not None

    def test_t1_has_right_kin_model(self, t1_container):
        """Verify T1 has right kinematics model."""
        assert hasattr(t1_container, "right_kin_model")
        assert t1_container.right_kin_model is not None

    def test_t1_has_left_tool_from_ee(self, t1_container):
        """Verify T1 has left tool_from_ee transform."""
        assert hasattr(t1_container, "left_tool_from_ee")
        assert t1_container.left_tool_from_ee.shape == (4, 4)

    def test_t1_has_right_tool_from_ee(self, t1_container):
        """Verify T1 has right tool_from_ee transform."""
        assert hasattr(t1_container, "right_tool_from_ee")
        assert t1_container.right_tool_from_ee.shape == (4, 4)

    def test_t1_has_left_joint_limits(self, t1_container):
        """Verify T1 has left joint limits."""
        assert hasattr(t1_container, "left_joint_limits")
        assert t1_container.left_joint_limits.shape == (2, 11)

    def test_t1_has_right_joint_limits(self, t1_container):
        """Verify T1 has right joint limits."""
        assert hasattr(t1_container, "right_joint_limits")
        assert t1_container.right_joint_limits.shape == (2, 11)

    def test_t1_has_left_gripper_spheres(self, t1_container):
        """Verify T1 has left gripper spheres."""
        assert hasattr(t1_container, "left_gripper_spheres")
        assert t1_container.left_gripper_spheres.shape[1] == 4

    def test_t1_has_right_gripper_spheres(self, t1_container):
        """Verify T1 has right gripper spheres."""
        assert hasattr(t1_container, "right_gripper_spheres")
        assert t1_container.right_gripper_spheres.shape[1] == 4


class TestDualArmToolFromEE:
    """Test tool_from_ee transformations for both arms."""

    def test_tool_from_ee_same_for_both_arms(self, t1_container):
        """Both arms should have the same tool_from_ee (per URDF analysis)."""
        left = t1_container.left_tool_from_ee
        right = t1_container.right_tool_from_ee
        assert torch.allclose(left, right, atol=1e-5)

    def test_tool_from_ee_is_valid_transform(self, t1_container):
        """tool_from_ee should be a valid SE(3) transformation."""
        for arm in ["left", "right"]:
            if arm == "left":
                tool_from_ee = t1_container.left_tool_from_ee
            else:
                tool_from_ee = t1_container.right_tool_from_ee
            
            # Check last row is [0, 0, 0, 1]
            assert torch.allclose(tool_from_ee[3], torch.tensor([0., 0., 0., 1.], device=tool_from_ee.device))
            
            # Check rotation part is orthonormal
            R = tool_from_ee[:3, :3]
            assert torch.allclose(R @ R.T, torch.eye(3, device=R.device), atol=1e-5)
            assert torch.allclose(torch.det(R), torch.tensor(1.0, device=R.device), atol=1e-5)


class TestDualArmKinematicsModels:
    """Test kinematics models for both arms."""

    def test_left_arm_fk_at_home(self, t1_container, tensor_args):
        """Test left arm forward kinematics at home position."""
        from cutamp.robots.t1 import t1_home_left
        
        q = torch.tensor(t1_home_left, device=tensor_args.device, dtype=torch.float32)[None]
        state = t1_container.left_kin_model.get_state(q)
        
        assert state.ee_pose.position.shape == (1, 3)
        assert state.ee_pose.quaternion.shape == (1, 4)

    def test_right_arm_fk_at_home(self, t1_container, tensor_args):
        """Test right arm forward kinematics at home position."""
        from cutamp.robots.t1 import t1_home_right
        
        q = torch.tensor(t1_home_right, device=tensor_args.device, dtype=torch.float32)[None]
        state = t1_container.right_kin_model.get_state(q)
        
        assert state.ee_pose.position.shape == (1, 3)
        assert state.ee_pose.quaternion.shape == (1, 4)

    def test_left_right_home_positions_different(self, t1_container, tensor_args):
        """Left and right arms at home should have different EE positions."""
        from cutamp.robots.t1 import t1_home_left, t1_home_right
        
        q_left = torch.tensor(t1_home_left, device=tensor_args.device, dtype=torch.float32)[None]
        q_right = torch.tensor(t1_home_right, device=tensor_args.device, dtype=torch.float32)[None]
        
        left_state = t1_container.left_kin_model.get_state(q_left)
        right_state = t1_container.right_kin_model.get_state(q_right)
        
        left_pos = left_state.ee_pose.position[0]
        right_pos = right_state.ee_pose.position[0]
        
        # Y positions should be mirrored (opposite signs)
        assert left_pos[1].item() > 0  # Left arm on positive Y side
        assert right_pos[1].item() < 0  # Right arm on negative Y side


class TestDualArmIKSolvers:
    """Test IK solver creation for both arms."""

    def test_create_left_ik_solver(self):
        """Test creating left arm IK solver."""
        from cutamp.robots.t1 import get_t1_ik_solver
        
        ik_solver = get_t1_ik_solver("left", world_cfg=None)
        assert ik_solver is not None

    def test_create_right_ik_solver(self):
        """Test creating right arm IK solver."""
        from cutamp.robots.t1 import get_t1_ik_solver
        
        ik_solver = get_t1_ik_solver("right", world_cfg=None)
        assert ik_solver is not None

    def test_invalid_arm_raises_error(self):
        """Test that invalid arm name raises error."""
        from cutamp.robots.t1 import get_t1_ik_solver
        
        with pytest.raises(ValueError, match="Invalid arm"):
            get_t1_ik_solver("middle", world_cfg=None)


class TestDualArmGripperSpheres:
    """Test gripper collision spheres for both arms."""

    def test_left_gripper_spheres_shape(self, t1_container):
        """Left gripper spheres should have shape (N, 4)."""
        spheres = t1_container.left_gripper_spheres
        assert spheres.ndim == 2
        assert spheres.shape[1] == 4

    def test_right_gripper_spheres_shape(self, t1_container):
        """Right gripper spheres should have shape (N, 4)."""
        spheres = t1_container.right_gripper_spheres
        assert spheres.ndim == 2
        assert spheres.shape[1] == 4

    def test_gripper_spheres_have_positive_radius(self, t1_container):
        """All gripper spheres should have positive radius."""
        for spheres in [t1_container.left_gripper_spheres, t1_container.right_gripper_spheres]:
            radii = spheres[:, 3]
            assert (radii > 0).all()


class TestDualArmJointLimits:
    """Test joint limits for both arms."""

    def test_left_joint_limits_shape(self, t1_container):
        """Left joint limits should have shape (2, 11)."""
        limits = t1_container.left_joint_limits
        assert limits.shape == (2, 11)

    def test_right_joint_limits_shape(self, t1_container):
        """Right joint limits should have shape (2, 11)."""
        limits = t1_container.right_joint_limits
        assert limits.shape == (2, 11)

    def test_lower_limits_less_than_upper(self, t1_container):
        """Lower limits should be less than upper limits."""
        for limits in [t1_container.left_joint_limits, t1_container.right_joint_limits]:
            lower = limits[0]
            upper = limits[1]
            assert (lower < upper).all()
