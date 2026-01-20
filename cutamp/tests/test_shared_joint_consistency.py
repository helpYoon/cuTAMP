"""
Unit tests for shared joint consistency in dual-arm T1 robot operations.

The T1 robot has 4 shared joints (lift + torso) that appear in both arms' 11-DOF configs.
These tests verify that shared joints are properly propagated between arms.
"""

import pytest
import torch

from cutamp.particle_initialization import propagate_shared_joints, NUM_SHARED_JOINTS
from cutamp.tamp_domain import Pick, Place, MoveFree, MoveHolding, Push, PushStick
from cutamp.t1_domain import (
    LeftPick, RightPick, LeftPlace, RightPlace,
    LeftMoveFree, RightMoveFree, LeftMoveHolding, RightMoveHolding,
    LeftPush, RightPush, LeftPushStick, RightPushStick,
)


class TestOperatorMetadataArm:
    """Test that operator metadata correctly stores arm information."""

    def test_left_pick_metadata_arm(self):
        assert LeftPick.metadata.arm == "left"

    def test_right_pick_metadata_arm(self):
        assert RightPick.metadata.arm == "right"

    def test_left_move_free_metadata_arm(self):
        assert LeftMoveFree.metadata.arm == "left"

    def test_right_move_holding_metadata_arm(self):
        assert RightMoveHolding.metadata.arm == "right"

    def test_single_arm_pick_metadata_arm_is_none(self):
        assert Pick.metadata.arm is None

    def test_single_arm_place_metadata_arm_is_none(self):
        assert Place.metadata.arm is None

    def test_all_left_operators_have_left_arm(self):
        left_ops = [LeftPick, LeftPlace, LeftMoveFree, LeftMoveHolding, LeftPush, LeftPushStick]
        for op in left_ops:
            assert op.metadata.arm == "left", f"{op.name} should have arm='left'"

    def test_all_right_operators_have_right_arm(self):
        right_ops = [RightPick, RightPlace, RightMoveFree, RightMoveHolding, RightPush, RightPushStick]
        for op in right_ops:
            assert op.metadata.arm == "right", f"{op.name} should have arm='right'"

    def test_all_single_arm_operators_have_no_arm(self):
        single_ops = [Pick, Place, MoveFree, MoveHolding, Push, PushStick]
        for op in single_ops:
            assert op.metadata.arm is None, f"{op.name} should have arm=None"


class TestSharedJointPropagation:
    """Test shared joint propagation between arms."""

    @pytest.fixture
    def num_particles(self):
        return 10

    @pytest.fixture
    def dof(self):
        return 11  # T1 arm DOF

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def initial_particles(self, num_particles, dof, device):
        """Create initial particles with different configs for left and right arms."""
        return {
            "left_q0": torch.randn(num_particles, dof, device=device),
            "right_q0": torch.randn(num_particles, dof, device=device),
        }

    def test_propagate_left_to_right(self, initial_particles, num_particles, dof, device):
        """When left arm IK solves, right arm shared joints should be updated."""
        particles = {k: v.clone() for k, v in initial_particles.items()}
        
        # Create a new solution for left arm
        solved_q = torch.randn(num_particles, dof, device=device)
        original_right = particles["right_q0"].clone()
        
        # Propagate shared joints from left to right
        propagate_shared_joints(particles, "left", solved_q)
        
        # Shared joints (first 4) should match solved_q
        assert torch.allclose(
            particles["right_q0"][:, :NUM_SHARED_JOINTS],
            solved_q[:, :NUM_SHARED_JOINTS]
        )
        
        # Arm-specific joints (5-11) should remain unchanged
        assert torch.allclose(
            particles["right_q0"][:, NUM_SHARED_JOINTS:],
            original_right[:, NUM_SHARED_JOINTS:]
        )

    def test_propagate_right_to_left(self, initial_particles, num_particles, dof, device):
        """When right arm IK solves, left arm shared joints should be updated."""
        particles = {k: v.clone() for k, v in initial_particles.items()}
        
        # Create a new solution for right arm
        solved_q = torch.randn(num_particles, dof, device=device)
        original_left = particles["left_q0"].clone()
        
        # Propagate shared joints from right to left
        propagate_shared_joints(particles, "right", solved_q)
        
        # Shared joints (first 4) should match solved_q
        assert torch.allclose(
            particles["left_q0"][:, :NUM_SHARED_JOINTS],
            solved_q[:, :NUM_SHARED_JOINTS]
        )
        
        # Arm-specific joints (5-11) should remain unchanged
        assert torch.allclose(
            particles["left_q0"][:, NUM_SHARED_JOINTS:],
            original_left[:, NUM_SHARED_JOINTS:]
        )

    def test_propagate_to_all_inactive_configs(self, num_particles, dof, device):
        """Propagation should update ALL configurations of the inactive arm."""
        particles = {
            "left_q0": torch.randn(num_particles, dof, device=device),
            "left_q1": torch.randn(num_particles, dof, device=device),
            "left_q2": torch.randn(num_particles, dof, device=device),
            "right_q0": torch.randn(num_particles, dof, device=device),
        }
        original_left_q0_arm = particles["left_q0"][:, NUM_SHARED_JOINTS:].clone()
        original_left_q1_arm = particles["left_q1"][:, NUM_SHARED_JOINTS:].clone()
        original_left_q2_arm = particles["left_q2"][:, NUM_SHARED_JOINTS:].clone()
        
        # Right arm IK solves
        solved_q = torch.randn(num_particles, dof, device=device)
        propagate_shared_joints(particles, "right", solved_q)
        
        # ALL left arm configs should have updated shared joints
        for key in ["left_q0", "left_q1", "left_q2"]:
            assert torch.allclose(
                particles[key][:, :NUM_SHARED_JOINTS],
                solved_q[:, :NUM_SHARED_JOINTS]
            ), f"Shared joints not updated for {key}"
        
        # But arm-specific joints should be unchanged
        assert torch.allclose(particles["left_q0"][:, NUM_SHARED_JOINTS:], original_left_q0_arm)
        assert torch.allclose(particles["left_q1"][:, NUM_SHARED_JOINTS:], original_left_q1_arm)
        assert torch.allclose(particles["left_q2"][:, NUM_SHARED_JOINTS:], original_left_q2_arm)

    def test_active_arm_not_modified(self, initial_particles, num_particles, dof, device):
        """The active arm's configuration should not be modified by propagation."""
        particles = {k: v.clone() for k, v in initial_particles.items()}
        
        solved_q = torch.randn(num_particles, dof, device=device)
        
        # Propagate from left to right
        propagate_shared_joints(particles, "left", solved_q)
        
        # Left arm (active) should not be modified
        # Note: The function doesn't modify the active arm at all
        assert torch.allclose(particles["left_q0"], initial_particles["left_q0"])

    def test_propagation_preserves_batch_dimension(self, initial_particles, num_particles, dof, device):
        """Propagation should work correctly across batch dimension."""
        particles = {k: v.clone() for k, v in initial_particles.items()}
        
        solved_q = torch.randn(num_particles, dof, device=device)
        
        # Propagate from left to right
        propagate_shared_joints(particles, "left", solved_q)
        
        # Shape should be preserved
        assert particles["right_q0"].shape == (num_particles, dof)

    def test_missing_inactive_arm_no_error(self, num_particles, dof, device):
        """Propagation should handle missing inactive arm gracefully."""
        # Only left arm present
        particles = {"left_q0": torch.randn(num_particles, dof, device=device)}
        solved_q = torch.randn(num_particles, dof, device=device)
        
        # Should not raise error when no right configs exist
        propagate_shared_joints(particles, "left", solved_q)
        
        # Left arm should be unchanged
        assert "left_q0" in particles
    
    def test_non_config_keys_not_affected(self, num_particles, dof, device):
        """Non-configuration keys (grasps, poses) should not be affected."""
        particles = {
            "left_q0": torch.randn(num_particles, dof, device=device),
            "right_q0": torch.randn(num_particles, dof, device=device),
            "grasp0": torch.randn(num_particles, 4, device=device),
            "pose0": torch.randn(num_particles, 4, device=device),
        }
        original_grasp = particles["grasp0"].clone()
        original_pose = particles["pose0"].clone()
        
        solved_q = torch.randn(num_particles, dof, device=device)
        propagate_shared_joints(particles, "left", solved_q)
        
        # Grasp and pose should be unchanged
        assert torch.allclose(particles["grasp0"], original_grasp)
        assert torch.allclose(particles["pose0"], original_pose)


class TestSharedJointConsistencySequential:
    """Test shared joint consistency across sequential operations."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_sequential_left_right_operations(self, device):
        """Sequential left and right operations should maintain shared joint consistency."""
        num_particles = 5
        dof = 11
        
        # Initial particles
        particles = {
            "left_q0": torch.zeros(num_particles, dof, device=device),
            "right_q0": torch.zeros(num_particles, dof, device=device),
        }
        
        # First operation: Left arm moves (IK returns new config)
        left_solution_1 = torch.ones(num_particles, dof, device=device)
        left_solution_1[:, :NUM_SHARED_JOINTS] = 1.0  # Shared joints = 1
        left_solution_1[:, NUM_SHARED_JOINTS:] = 2.0  # Left arm joints = 2
        propagate_shared_joints(particles, "left", left_solution_1)
        
        # After left op, right's shared joints should be updated to 1.0
        assert torch.allclose(
            particles["right_q0"][:, :NUM_SHARED_JOINTS],
            torch.ones(num_particles, NUM_SHARED_JOINTS, device=device)
        )
        
        # Second operation: Right arm moves (IK returns new config)
        right_solution_1 = torch.ones(num_particles, dof, device=device) * 3.0
        right_solution_1[:, :NUM_SHARED_JOINTS] = 3.0  # New shared joints = 3
        right_solution_1[:, NUM_SHARED_JOINTS:] = 4.0  # Right arm joints = 4
        propagate_shared_joints(particles, "right", right_solution_1)
        
        # After right op, left's shared joints should be updated to 3.0
        assert torch.allclose(
            particles["left_q0"][:, :NUM_SHARED_JOINTS],
            torch.ones(num_particles, NUM_SHARED_JOINTS, device=device) * 3.0
        )
        
        # Left arm-specific joints should still be 0 (original value, never updated)
        assert torch.allclose(
            particles["left_q0"][:, NUM_SHARED_JOINTS:],
            torch.zeros(num_particles, dof - NUM_SHARED_JOINTS, device=device)
        )

    def test_interleaved_operations_with_multiple_configs(self, device):
        """Test interleaved operations where each arm accumulates multiple configs."""
        num_particles = 5
        dof = 11
        
        # Initial particles
        particles = {
            "left_q0": torch.zeros(num_particles, dof, device=device),
            "right_q0": torch.zeros(num_particles, dof, device=device),
        }
        
        # Step 1: LeftPick creates left_q1
        left_q1 = torch.ones(num_particles, dof, device=device)
        left_q1[:, :NUM_SHARED_JOINTS] = 1.0  # Shared = 1
        left_q1[:, NUM_SHARED_JOINTS:] = 10.0  # Left arm = 10
        particles["left_q1"] = left_q1.clone()
        propagate_shared_joints(particles, "left", left_q1)
        
        # right_q0 should have shared joints = 1
        assert torch.allclose(particles["right_q0"][:, :NUM_SHARED_JOINTS], 
                              torch.ones(num_particles, NUM_SHARED_JOINTS, device=device))
        
        # Step 2: RightPick creates right_q1
        right_q1 = torch.ones(num_particles, dof, device=device) * 2
        right_q1[:, :NUM_SHARED_JOINTS] = 2.0  # Shared = 2
        right_q1[:, NUM_SHARED_JOINTS:] = 20.0  # Right arm = 20
        particles["right_q1"] = right_q1.clone()
        propagate_shared_joints(particles, "right", right_q1)
        
        # BOTH left_q0 and left_q1 should have shared joints = 2
        assert torch.allclose(particles["left_q0"][:, :NUM_SHARED_JOINTS],
                              torch.ones(num_particles, NUM_SHARED_JOINTS, device=device) * 2)
        assert torch.allclose(particles["left_q1"][:, :NUM_SHARED_JOINTS],
                              torch.ones(num_particles, NUM_SHARED_JOINTS, device=device) * 2)
        
        # But arm-specific joints should be unchanged
        assert torch.allclose(particles["left_q0"][:, NUM_SHARED_JOINTS:],
                              torch.zeros(num_particles, dof - NUM_SHARED_JOINTS, device=device))
        assert torch.allclose(particles["left_q1"][:, NUM_SHARED_JOINTS:],
                              torch.ones(num_particles, dof - NUM_SHARED_JOINTS, device=device) * 10)
        
        # Step 3: LeftPlace creates left_q2
        left_q2 = torch.ones(num_particles, dof, device=device) * 3
        left_q2[:, :NUM_SHARED_JOINTS] = 3.0  # Shared = 3
        left_q2[:, NUM_SHARED_JOINTS:] = 30.0  # Left arm = 30
        particles["left_q2"] = left_q2.clone()
        propagate_shared_joints(particles, "left", left_q2)
        
        # BOTH right_q0 and right_q1 should have shared joints = 3
        assert torch.allclose(particles["right_q0"][:, :NUM_SHARED_JOINTS],
                              torch.ones(num_particles, NUM_SHARED_JOINTS, device=device) * 3)
        assert torch.allclose(particles["right_q1"][:, :NUM_SHARED_JOINTS],
                              torch.ones(num_particles, NUM_SHARED_JOINTS, device=device) * 3)
        
        # But right arm-specific joints should be unchanged
        assert torch.allclose(particles["right_q0"][:, NUM_SHARED_JOINTS:],
                              torch.zeros(num_particles, dof - NUM_SHARED_JOINTS, device=device))
        assert torch.allclose(particles["right_q1"][:, NUM_SHARED_JOINTS:],
                              torch.ones(num_particles, dof - NUM_SHARED_JOINTS, device=device) * 20)


class TestNumSharedJoints:
    """Test that NUM_SHARED_JOINTS constant is correct."""

    def test_num_shared_joints_is_four(self):
        """T1 has 4 shared joints: lift1, lift2, torso_pitch, waist_yaw."""
        assert NUM_SHARED_JOINTS == 4

    def test_shared_joints_within_dof(self):
        """Shared joints index should be less than total DOF (11)."""
        assert NUM_SHARED_JOINTS < 11
