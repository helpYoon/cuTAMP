"""Tests for the T1 v0.8 single-planner robot config (t1_planar_base.yml).

Verifies the YAML loads, has 21 active cspace joints (3 base + 2 leg + 2 torso +
7 left arm + 7 right arm), declares both tool frames, and locks 10 head/gripper
joints.
"""

from cutamp.robots.t1 import (
    BASE_DOF,
    BASE_INDICES,
    CUROBO_DOF,
    JOINT_NAMES_FULL,
    LEFT_ARM_INDICES,
    LEFT_TOOL_FRAME,
    RIGHT_ARM_INDICES,
    RIGHT_TOOL_FRAME,
    TOOL_FRAMES,
    URDF_DOF,
)


def test_yaml_loads(t1_planar_base_config):
    cfg = t1_planar_base_config["robot_cfg"]["kinematics"]
    assert cfg["base_link"] == "world"
    assert "extra_links" in cfg
    assert "lock_joints" in cfg


def test_extra_links_declares_planar_base(t1_planar_base_config):
    extra = t1_planar_base_config["robot_cfg"]["kinematics"]["extra_links"]
    assert extra["base_link_x"]["joint_type"] == "X_PRISM"
    assert extra["base_link_y"]["joint_type"] == "Y_PRISM"
    assert extra["base_link_yaw"]["joint_type"] == "Z_ROT"
    assert extra["base_link_yaw"]["child_link_name"] == "mobile_base_link"


def test_tool_frames(t1_planar_base_config):
    tools = t1_planar_base_config["robot_cfg"]["kinematics"]["tool_frames"]
    # Both arm tool frames must be present. mobile_base_link is also tracked
    # so the COM-over-base cost can read its world pose; not an IK target.
    assert LEFT_TOOL_FRAME in tools and RIGHT_TOOL_FRAME in tools
    assert TOOL_FRAMES == (LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME)


def test_cspace_has_21_joints_in_expected_order(t1_planar_base_config):
    joints = t1_planar_base_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
    assert len(joints) == CUROBO_DOF == 21
    assert tuple(joints) == JOINT_NAMES_FULL
    assert joints[BASE_INDICES] == ["base_j_x", "base_j_y", "base_j_yaw"]
    assert "ankle_pitch" in joints and "knee_pitch" in joints
    assert "Torso_Pitch" in joints and "Waist_Yaw" in joints
    assert "waist_lift_1" not in joints
    assert len(joints[LEFT_ARM_INDICES]) == 7
    assert len(joints[RIGHT_ARM_INDICES]) == 7


def test_cspace_distance_weight_uniform(t1_planar_base_config):
    weights = t1_planar_base_config["robot_cfg"]["kinematics"]["cspace"]["cspace_distance_weight"]
    assert len(weights) == CUROBO_DOF
    assert all(w == 1.0 for w in weights), "Revolute leg should not need outsized weights"


def test_lock_joints_has_head_and_grippers(t1_planar_base_config):
    locked = t1_planar_base_config["robot_cfg"]["kinematics"]["lock_joints"]
    assert "AAHead_yaw" in locked and "Head_pitch" in locked
    for arm in ("left", "right"):
        for k in (1, 11, 2, 22):
            assert f"{arm}_Link{k}" in locked
    assert len(locked) == 10
    # Removed lift/yaw entries should NOT be present.
    assert "waist_lift_1" not in locked
    assert "waist_lift_2" not in locked
    assert "Waist_Yaw" not in locked


def test_default_joint_position_length(t1_planar_base_config):
    djp = t1_planar_base_config["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"]
    assert len(djp) == CUROBO_DOF


def test_dof_constants():
    assert URDF_DOF == 28
    assert CUROBO_DOF == 21
    assert BASE_DOF == 3
