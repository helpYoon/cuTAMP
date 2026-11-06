"""The balance gate's resampling (motion_solver._retime): a segment keeps its ends, and at no slowdown it is the
motion planner's own timing; and the bodies' config hygiene (get_t1_bodies). The gate itself has its tests on the real
legs (test_real_legs_motion.py), which build their segments with ``_min_jerk``."""
import math

import numpy as np
import pytest
import torch

from cutamp.robots.t1 import REAL_LEGS
from cutamp.tests.conftest import needs_cuda

DT = 0.025


def _min_jerk(T, n):
    """A rest-to-rest 0 -> 1 move over T s on n frames: (t, position, velocity, acceleration)."""
    t = np.linspace(0.0, T, n)
    u = t / T
    return (t, 10 * u**3 - 15 * u**4 + 6 * u**5, (30 * u**2 - 60 * u**3 + 30 * u**4) / T,
            (60 * u - 180 * u**2 + 120 * u**3) / T**2)


def test_retime_preserves_endpoints():
    from curobo.types import JointState

    from cutamp.motion_solver import _retime
    _, p, v, a = _min_jerk(1.0, 41)
    col = lambda x: torch.tensor(x, dtype=torch.float32).view(-1, 1)  # noqa: E731
    js = JointState(position=col(p), velocity=col(v), acceleration=col(a), joint_names=["j"])
    out = _retime(js, 2.3, DT)
    n = 1 + math.ceil(2.3 * 40)
    s_eff = (n - 1) / 40
    assert out.position.shape == (n, 1) and out.joint_names == ["j"]
    assert out.position[0].item() == pytest.approx(0.0, abs=1e-6)
    assert out.position[-1].item() == pytest.approx(1.0, abs=1e-6)
    assert out.velocity[-1].item() == pytest.approx(0.0, abs=1e-6)
    assert out.velocity.max().item() == pytest.approx(1.875 / s_eff, abs=1e-3)
    assert out.acceleration.max().item() == pytest.approx(a.max() / s_eff**2, rel=2e-2)


def test_retime_without_slowdown_is_the_motion_planners_timing():
    # The timing test runs a slowed segment at the motion planner's own timing: the gate's spline sampled
    # without a slowdown. That reproduces the planner's frames, and the slowed segment is the same curve
    # played s times slower (frame 2k of a 2x slowdown is frame k, with velocity / 2 and acceleration / 4).
    from curobo.types import JointState

    from cutamp.motion_solver import _retime
    _, p, v, a = _min_jerk(1.0, 41)
    col = lambda x: torch.tensor(x, dtype=torch.float32).view(-1, 1)  # noqa: E731
    js = JointState(position=col(p), velocity=col(v), acceleration=col(a), joint_names=["j"])
    fast, slow = _retime(js, 1.0, DT), _retime(js, 2.0, DT)
    assert fast.position.shape == js.position.shape
    torch.testing.assert_close(fast.position, js.position, rtol=0, atol=1e-6)
    torch.testing.assert_close(slow.position[::2], fast.position, rtol=0, atol=1e-6)
    torch.testing.assert_close(slow.velocity[::2], fast.velocity / 2, rtol=0, atol=1e-5)
    torch.testing.assert_close(slow.acceleration[::2], fast.acceleration / 4, rtol=0, atol=1e-4)


@needs_cuda
def test_bodies_leave_the_robot_config_alone():
    # The bodies' kinematics is built from the cached robot config, whose tool frames every later IK solver
    # and motion planner reads: rewritten to the bodies' links, the next run in the process found no IK solution.
    from cutamp.robots.t1 import get_t1_bodies, t1_curobo_cfg
    before = list(t1_curobo_cfg(REAL_LEGS)["robot_cfg"]["kinematics"]["tool_frames"])
    get_t1_bodies(REAL_LEGS)
    assert t1_curobo_cfg(REAL_LEGS)["robot_cfg"]["kinematics"]["tool_frames"] == before
