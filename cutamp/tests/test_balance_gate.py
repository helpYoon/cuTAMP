"""The trajectory balance gate (motion_solver._balance_gate): every interpolated frame's CoM and
cart-table ZMP stay inside the support rectangle; a fast segment is slowed, an unbalanced one fails."""
import math

import numpy as np
import pytest
import torch

from cutamp.com_polygon_cost import SUPPORT_HALF_EXTENTS, cart_table_zmp, com_margin_from_base, retime_factor
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda

DT = 0.025


def _min_jerk(T, n):
    """A rest-to-rest 0 -> 1 move over T s on n frames: (t, position, velocity, acceleration)."""
    t = np.linspace(0.0, T, n)
    u = t / T
    return (t, 10 * u**3 - 15 * u**4 + 6 * u**5, (30 * u**2 - 60 * u**3 + 30 * u**4) / T,
            (60 * u - 180 * u**2 + 120 * u**3) / T**2)


def test_cart_table_zmp_constant_acceleration():
    # A CoM accelerating at 1 m/s^2 at constant height 0.6 m has its ZMP z/g * a behind it, at every
    # frame: the one-sided edge differences are exact on a quadratic.
    t = torch.arange(41, dtype=torch.float64) * DT
    com = torch.stack([0.5 * t**2, torch.zeros_like(t), torch.full_like(t, 0.6)], dim=-1)
    zmp = cart_table_zmp(com, DT)
    assert torch.allclose(zmp[:, 0] - com[:, 0], torch.full_like(t, -0.6 / 9.81), atol=1e-6)
    assert torch.allclose(zmp[:, 1], torch.zeros_like(t), atol=1e-9)


def test_retime_factor_scales_with_square():
    # Slowing by s divides the ZMP's offset from the CoM by s^2.
    hx = SUPPORT_HALF_EXTENTS[0]
    com = torch.tensor([[hx - 0.01, 0.0]])
    assert retime_factor(com, com + torch.tensor([[0.04, 0.0]])) == pytest.approx(2.0)
    assert retime_factor(com, com + torch.tensor([[0.005, 0.0]])) == 1.0


def test_retime_factor_zero_room_is_inf():
    hx = SUPPORT_HALF_EXTENTS[0]
    com = torch.tensor([[hx, 0.0]])
    assert retime_factor(com, com + torch.tensor([[0.001, 0.0]])) == math.inf


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


def _torso_ramp(world, pitch, T):
    """A rest-to-rest Torso_Pitch move from 0 to ``pitch`` over T s at DT, other joints at home, as a
    full-dof JointState in world.kinematics order."""
    from curobo.types import JointState
    names = list(world.kinematics.joint_names)
    n = int(round(T / DT)) + 1
    _, p, v, a = _min_jerk(T, n)
    home = world.q_init.view(1, -1).float()
    pos, vel, acc = home.repeat(n, 1), torch.zeros(n, len(names)).to(home), torch.zeros(n, len(names)).to(home)
    j = names.index("Torso_Pitch")
    pos[:, j] = home[0, j] + pitch * pos.new_tensor(p)
    vel[:, j], acc[:, j] = pitch * pos.new_tensor(v), pitch * pos.new_tensor(a)
    return JointState(position=pos, velocity=vel, acceleration=acc, joint_names=names)


@needs_cuda
def test_gate_rejects_com_outside():
    # A deep bow puts the CoM past the toe (13.5 cm at Torso_Pitch -1.6, the front edge is 10.15 cm).
    from cutamp.motion_solver import _balance_gate
    world = make_blocks_t1_world()
    with pytest.raises(RuntimeError, match="CoM leaves"):
        _balance_gate(world, None, _torso_ramp(world, -1.6, 1.0), DT)


@needs_cuda
def test_gate_slows_fast_lean():
    # A bow to -0.6 is CoM-safe (8 cm), but in 0.5 s its ZMP swings far past the edges.
    from cutamp.com_polygon_cost import com_zmp_in_support
    from cutamp.motion_solver import _balance_gate, _full_positions
    world = make_blocks_t1_world()
    plan = _torso_ramp(world, -0.6, 0.5)
    com, zmp = com_zmp_in_support(world, _full_positions(world, None, plan), DT)
    margin = lambda xy: com_margin_from_base(xy, SUPPORT_HALF_EXTENTS).min()  # noqa: E731
    assert margin(com) >= 0 > margin(zmp)
    out, stats = _balance_gate(world, None, plan, DT)
    assert stats["retime"] > 1.0
    assert out.position.shape[0] == round((plan.position.shape[0] - 1) * stats["retime"]) + 1
    com, zmp = com_zmp_in_support(world, _full_positions(world, None, out), DT)
    assert margin(zmp) >= 0 and stats["zmp_min_margin"] >= 0
    assert torch.allclose(out.position[-1], plan.position[-1], atol=1e-5)


@needs_cuda
def test_gate_passes_slow_motion_untouched():
    from cutamp.motion_solver import _balance_gate
    world = make_blocks_t1_world()
    plan = _torso_ramp(world, -0.3, 2.0)
    out, stats = _balance_gate(world, None, plan, DT)
    assert out is plan and stats["retime"] == 1.0
