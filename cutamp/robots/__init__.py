"""Robot containers and loaders for cuTAMP (cuRobo single-MotionPlanner).

T1 humanoid is the only supported robot, on its real-leg planning model
(``REAL_LEGS``, a :class:`~cutamp.robots.t1.T1Model`). The ``RobotContainer`` holds the model,
its kinematics + per-tool-frame gripper spheres / tool transforms.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch

from curobo.kinematics import Kinematics
from curobo.types import DeviceCfg

from .t1 import (
    LEFT_TOOL_FRAME,
    REAL_LEGS,
    RIGHT_TOOL_FRAME,
    TOOL_FRAMES as T1_TOOL_FRAMES,
    T1Model,
    get_t1_gripper_spheres,
    get_t1_kinematics,
    get_t1_tool_from_ee,
    load_t1_rerun,
)


@dataclass(frozen=True)
class RobotContainer:
    """Container for robot kinematics, gripper data, and tool frames.

    T1 has two tool frames (``left_base_link``, ``right_base_link``);
    ``tool_from_ee`` and ``gripper_spheres`` are dicts keyed by tool-frame
    name. The kinematics tracks more frames (the soles and stance).
    """

    model: T1Model
    kinematics: Kinematics
    joint_limits: torch.Tensor  # [2, d]
    tool_frames: List[str]
    tool_from_ee: Dict[str, torch.Tensor]  # each [4, 4]
    gripper_spheres: Dict[str, torch.Tensor]  # each [n, 4]


def load_t1_container(model: T1Model, device_cfg: DeviceCfg = None) -> RobotContainer:
    """T1 humanoid on ``model``, as a single robot with two tool frames."""
    if device_cfg is None:
        device_cfg = DeviceCfg()
    kin = get_t1_kinematics(model, device_cfg)
    joint_limits = kin.get_joint_limits().position
    assert joint_limits.shape[0] == 2

    tool_from_ee = get_t1_tool_from_ee(device_cfg)
    spheres = {
        LEFT_TOOL_FRAME: get_t1_gripper_spheres("left", device_cfg),
        RIGHT_TOOL_FRAME: get_t1_gripper_spheres("right", device_cfg),
    }

    return RobotContainer(
        model=model,
        kinematics=kin,
        joint_limits=joint_limits,
        tool_frames=list(T1_TOOL_FRAMES),
        tool_from_ee=tool_from_ee,
        gripper_spheres=spheres,
    )
