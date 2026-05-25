"""Tests for the T1 robot module under cuRobo v0.8 single-MotionPlanner.

The dual-arm-specific assertions from the v0.7 era (DualArmRobotContainer,
two kinematics models, etc.) are gone. This file now exercises the single-
container shape: one Kinematics, two tool frames, gripper-spheres dict.
"""

import pytest

import torch


@pytest.fixture
def t1_container(device_cfg):
    from cutamp.robots import load_t1_container
    return load_t1_container(device_cfg)


def test_t1_container_is_single_robot(t1_container):
    from cutamp.robots import RobotContainer
    assert isinstance(t1_container, RobotContainer)
    assert t1_container.name == "t1"


def test_t1_has_two_tool_frames(t1_container):
    from cutamp.robots.t1 import LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME
    assert set(t1_container.tool_frames) == {LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME}
    assert LEFT_TOOL_FRAME in t1_container.tool_from_ee
    assert RIGHT_TOOL_FRAME in t1_container.tool_from_ee
    assert LEFT_TOOL_FRAME in t1_container.gripper_spheres
    assert RIGHT_TOOL_FRAME in t1_container.gripper_spheres


def test_t1_joint_limits_shape(t1_container):
    from cutamp.robots.t1 import CUROBO_DOF
    assert t1_container.joint_limits.shape == (2, CUROBO_DOF)


def test_t1_kinematics_runs(t1_container):
    from cutamp.robots.t1 import CUROBO_DOF, t1_home
    from curobo.types import JointState

    q = torch.tensor(t1_home, device=t1_container.kinematics.device_cfg.device).unsqueeze(0)
    js = JointState.from_position(q)
    state = t1_container.kinematics.compute_kinematics(js)
    # ToolPose covers both arm tool frames plus mobile_base_link (tracked for
    # the COM-over-base cost; not an IK target).
    assert state.tool_poses.num_links >= 2
