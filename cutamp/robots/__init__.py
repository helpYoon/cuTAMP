"""Robot containers and loaders for cuTAMP (cuRobo v0.8 single-MotionPlanner).

T1 humanoid is the only supported robot. The ``RobotContainer`` exposes its
21-DOF kinematics + per-tool-frame gripper spheres / tool transforms.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from jaxtyping import Float

from curobo.kinematics import Kinematics
from curobo.types import DeviceCfg

from .t1 import (
    LEFT_TOOL_FRAME,
    RIGHT_TOOL_FRAME,
    TOOL_FRAMES as T1_TOOL_FRAMES,
    curobo_to_urdf_joints,
    get_t1_gripper_spheres,
    get_t1_kinematics,
    get_t1_tool_from_ee,
    load_t1_rerun,
    t1_home,
)
from .utils import RerunRobot


@dataclass(frozen=True)
class RobotContainer:
    """Container for robot kinematics, gripper data, and tool frames.

    T1 has two tool frames (``left_base_link``, ``right_base_link``);
    ``tool_from_ee`` and ``gripper_spheres`` are dicts keyed by tool-frame name.
    """

    name: str
    kinematics: Kinematics
    joint_limits: Float[torch.Tensor, "2 d"]
    tool_frames: List[str]
    tool_from_ee: Dict[str, Float[torch.Tensor, "4 4"]]
    gripper_spheres: Dict[str, Float[torch.Tensor, "n 4"]]


def load_t1_container(device_cfg: DeviceCfg = None) -> RobotContainer:
    """T1 humanoid as a single 21-DOF robot with two tool frames."""
    if device_cfg is None:
        device_cfg = DeviceCfg()
    kin = get_t1_kinematics(device_cfg)
    joint_limits = kin.get_joint_limits().position
    assert joint_limits.shape[0] == 2

    tool_from_ee = get_t1_tool_from_ee(device_cfg)
    spheres = {
        LEFT_TOOL_FRAME: get_t1_gripper_spheres("left", device_cfg),
        RIGHT_TOOL_FRAME: get_t1_gripper_spheres("right", device_cfg),
    }

    return RobotContainer(
        name="t1",
        kinematics=kin,
        joint_limits=joint_limits,
        tool_frames=list(T1_TOOL_FRAMES),
        tool_from_ee=tool_from_ee,
        gripper_spheres=spheres,
    )


def load_rerun_robot(robot: str = "t1", load_mesh: bool = True) -> RerunRobot:
    if robot != "t1":
        raise ValueError(f"Unknown robot: {robot}. Only 't1' is supported.")
    return load_t1_rerun(load_mesh)


def get_q_home(robot: str = "t1") -> Sequence[float]:
    if robot != "t1":
        raise ValueError(f"Unknown robot: {robot}. Only 't1' is supported.")
    return t1_home


def load_robot_container(
    robot: str = "t1", device_cfg: DeviceCfg = None,
) -> RobotContainer:
    if robot != "t1":
        raise ValueError(f"Unknown robot: {robot}. Only 't1' is supported.")
    return load_t1_container(device_cfg)
