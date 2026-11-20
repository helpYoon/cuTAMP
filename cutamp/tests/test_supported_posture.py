"""Supported-posture fit on a translated stance with a displaced idle-arm seed."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from cutamp.robots.t1 import REAL_LEGS, t1_curobo_cfg
from cutamp.step_ik import _Kinematics
from experiments.t1_supported_posture import fit_centered_posture, set_stance_base_limits


def test_transported_base_chart_intersects_both_leg_workspaces():
    cfg = deepcopy(t1_curobo_cfg(REAL_LEGS))
    feet = np.array([[0.1, 0.15, 0.3], [0.08, -0.14, -0.1]])
    limits = set_stance_base_limits(cfg, feet)
    shifted = set_stance_base_limits(cfg, feet + [0.5, -0.2, 0])
    np.testing.assert_allclose(shifted, limits + [[0.5], [-0.2]], atol=1e-9)
    with pytest.raises(ValueError, match="shared virtual-base"):
        set_stance_base_limits(cfg, np.array([[0, 3, 0], [0, -3, 0]]))


@pytest.mark.parametrize("arm", ["left", "right"])
def test_supported_posture_uses_nominal_idle_and_preserves_private_config(arm):
    cfg = deepcopy(t1_curobo_cfg(REAL_LEGS))
    pristine = deepcopy(cfg)
    kin = _Kinematics(torch.device("cpu"), cfg["robot_cfg"]["kinematics"])
    seed = torch.tensor(REAL_LEGS.home, dtype=torch.float64)
    pose, _ = kin.fk(seed[None], arm, False)
    pose = pose.reshape(3, 12)
    feet = torch.cat((pose[:2, :2], torch.zeros(2, 1, dtype=pose.dtype)), dim=-1)
    wrist = torch.eye(4, dtype=pose.dtype)
    wrist[:3, :3] = pose[2, 3:].reshape(3, 3).T / 0.2
    wrist[:3, 3] = pose[2, :3]
    idle = REAL_LEGS.right_arm if arm == "left" else REAL_LEGS.left_arm
    seed[idle.start] -= 0.4
    solved = fit_centered_posture(feet, wrist, seed, arm, cfg=cfg)
    np.testing.assert_array_equal(solved[idle], REAL_LEGS.home[idle])
    assert cfg == pristine
    with pytest.raises(ValueError, match="fixed posture"):
        fit_centered_posture(feet, wrist, seed, arm, cfg=cfg, other_arm_reference=[10.0] * 7)
