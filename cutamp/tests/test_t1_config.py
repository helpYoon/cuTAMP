"""
Tests for T1 robot YAML configuration files.

Verify that t1_left_11dof.yml and t1_right_11dof.yml are correctly structured.
"""
import pytest
import os


class TestT1ConfigFiles:
    """Tests for T1 YAML configuration files."""

    def test_left_config_loads(self, t1_left_config):
        """Verify left arm config loads without errors."""
        assert "robot_cfg" in t1_left_config
        assert "kinematics" in t1_left_config["robot_cfg"]

    def test_right_config_loads(self, t1_right_config):
        """Verify right arm config loads without errors."""
        assert "robot_cfg" in t1_right_config
        assert "kinematics" in t1_right_config["robot_cfg"]

    def test_left_config_has_11_joints(self, t1_left_config):
        """Verify left arm has exactly 11 DOF."""
        joint_names = t1_left_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
        assert len(joint_names) == 11, f"Expected 11 joints, got {len(joint_names)}"

    def test_right_config_has_11_joints(self, t1_right_config):
        """Verify right arm has exactly 11 DOF."""
        joint_names = t1_right_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
        assert len(joint_names) == 11, f"Expected 11 joints, got {len(joint_names)}"

    def test_left_ee_link_is_correct(self, t1_left_config):
        """Verify left arm EE link is left_base_link."""
        ee_link = t1_left_config["robot_cfg"]["kinematics"]["ee_link"]
        assert ee_link == "left_base_link"

    def test_right_ee_link_is_correct(self, t1_right_config):
        """Verify right arm EE link is right_base_link."""
        ee_link = t1_right_config["robot_cfg"]["kinematics"]["ee_link"]
        assert ee_link == "right_base_link"

    def test_base_link_is_lifting_base(self, t1_left_config, t1_right_config):
        """Verify both configs use lifting_base_link as base."""
        left_base = t1_left_config["robot_cfg"]["kinematics"]["base_link"]
        right_base = t1_right_config["robot_cfg"]["kinematics"]["base_link"]
        assert left_base == "lifting_base_link"
        assert right_base == "lifting_base_link"

    def test_shared_joints_match(self, t1_left_config, t1_right_config):
        """Verify lifting column and torso joints are same in both configs."""
        left_joints = t1_left_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
        right_joints = t1_right_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]

        # First 4 joints should be identical (lift1, lift2, torso_pitch, waist_yaw)
        assert left_joints[:4] == right_joints[:4], \
            f"Shared joints don't match: {left_joints[:4]} vs {right_joints[:4]}"

    def test_left_has_left_arm_joints(self, t1_left_config):
        """Verify left config has left arm joints in cspace."""
        joint_names = t1_left_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
        left_arm_joints = [
            "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
            "Left_Elbow_Yaw", "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll"
        ]
        for joint in left_arm_joints:
            assert joint in joint_names, f"Left arm joint {joint} not in cspace"

    def test_right_has_right_arm_joints(self, t1_right_config):
        """Verify right config has right arm joints in cspace."""
        joint_names = t1_right_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
        right_arm_joints = [
            "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
            "Right_Elbow_Yaw", "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll"
        ]
        for joint in right_arm_joints:
            assert joint in joint_names, f"Right arm joint {joint} not in cspace"

    def test_right_arm_locked_in_left_config(self, t1_left_config):
        """Verify right arm joints are locked in left arm config."""
        lock_joints = t1_left_config["robot_cfg"]["kinematics"]["lock_joints"]
        right_arm_joints = [
            "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
            "Right_Elbow_Yaw", "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll"
        ]
        for joint in right_arm_joints:
            assert joint in lock_joints, f"Right arm joint {joint} should be locked"

    def test_left_arm_locked_in_right_config(self, t1_right_config):
        """Verify left arm joints are locked in right arm config."""
        lock_joints = t1_right_config["robot_cfg"]["kinematics"]["lock_joints"]
        left_arm_joints = [
            "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
            "Left_Elbow_Yaw", "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll"
        ]
        for joint in left_arm_joints:
            assert joint in lock_joints, f"Left arm joint {joint} should be locked"

    def test_collision_spheres_file_reference(self, t1_left_config, t1_right_config):
        """Verify collision spheres file reference is valid."""
        left_spheres = t1_left_config["robot_cfg"]["kinematics"]["collision_spheres"]
        right_spheres = t1_right_config["robot_cfg"]["kinematics"]["collision_spheres"]
        assert left_spheres == "t1_spheres.yml"
        assert right_spheres == "t1_spheres.yml"

    def test_retract_config_length(self, t1_left_config, t1_right_config):
        """Verify retract config has correct length (11 values)."""
        left_retract = t1_left_config["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
        right_retract = t1_right_config["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
        assert len(left_retract) == 11, f"Expected 11 retract values, got {len(left_retract)}"
        assert len(right_retract) == 11, f"Expected 11 retract values, got {len(right_retract)}"

    def test_null_space_weight_length(self, t1_left_config, t1_right_config):
        """Verify null_space_weight has correct length (11 values)."""
        left_nsw = t1_left_config["robot_cfg"]["kinematics"]["cspace"]["null_space_weight"]
        right_nsw = t1_right_config["robot_cfg"]["kinematics"]["cspace"]["null_space_weight"]
        assert len(left_nsw) == 11, f"Expected 11 null_space_weight values, got {len(left_nsw)}"
        assert len(right_nsw) == 11, f"Expected 11 null_space_weight values, got {len(right_nsw)}"

    def test_urdf_path_is_correct(self, t1_left_config, t1_right_config):
        """Verify URDF path points to t1_simplified.urdf."""
        left_urdf = t1_left_config["robot_cfg"]["kinematics"]["urdf_path"]
        right_urdf = t1_right_config["robot_cfg"]["kinematics"]["urdf_path"]
        assert left_urdf == "t1_simplified.urdf"
        assert right_urdf == "t1_simplified.urdf"

    def test_grippers_locked(self, t1_left_config, t1_right_config):
        """Verify all gripper joints are locked in both configs."""
        gripper_joints = [
            "left_Link1", "left_Link11", "left_Link2", "left_Link22",
            "right_Link1", "right_Link11", "right_Link2", "right_Link22"
        ]
        
        left_lock = t1_left_config["robot_cfg"]["kinematics"]["lock_joints"]
        right_lock = t1_right_config["robot_cfg"]["kinematics"]["lock_joints"]
        
        for joint in gripper_joints:
            assert joint in left_lock, f"Gripper joint {joint} should be locked in left config"
            assert joint in right_lock, f"Gripper joint {joint} should be locked in right config"

    def test_head_locked(self, t1_left_config, t1_right_config):
        """Verify head joints are locked in both configs."""
        head_joints = ["AAHead_yaw", "Head_pitch"]
        
        left_lock = t1_left_config["robot_cfg"]["kinematics"]["lock_joints"]
        right_lock = t1_right_config["robot_cfg"]["kinematics"]["lock_joints"]
        
        for joint in head_joints:
            assert joint in left_lock, f"Head joint {joint} should be locked in left config"
            assert joint in right_lock, f"Head joint {joint} should be locked in right config"


class TestT1ConfigWithCuRobo:
    """Tests that require cuRobo to load configs.
    
    Note: Full cuRobo loading tests are in test_t1_robot_module.py (Phase 2),
    which properly handles path resolution via the t1.py module.
    """

    def test_robot_config_from_dict_left(self):
        """Verify left config can be loaded into cuRobo RobotConfig via t1.py."""
        from cutamp.robots.t1 import t1_curobo_cfg
        from curobo.types.robot import RobotConfig
        
        # Load config via t1.py which handles path resolution
        cfg = t1_curobo_cfg("left")
        robot_cfg = RobotConfig.from_dict(cfg["robot_cfg"])
        
        assert robot_cfg is not None
        assert robot_cfg.kinematics is not None

    def test_robot_config_from_dict_right(self):
        """Verify right config can be loaded into cuRobo RobotConfig via t1.py."""
        from cutamp.robots.t1 import t1_curobo_cfg
        from curobo.types.robot import RobotConfig
        
        # Load config via t1.py which handles path resolution
        cfg = t1_curobo_cfg("right")
        robot_cfg = RobotConfig.from_dict(cfg["robot_cfg"])
        
        assert robot_cfg is not None
        assert robot_cfg.kinematics is not None
