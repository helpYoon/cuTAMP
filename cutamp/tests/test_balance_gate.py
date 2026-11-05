"""The trajectory balance gate (motion_solver._balance_gate): every interpolated frame's CoM and
multibody ZMP stay inside the support rectangle; a fast segment is slowed, an unbalanced one fails."""
import functools
import math

import numpy as np
import pytest
import torch

from cutamp.com_polygon_cost import (
    COM_CENTER, SOLE_HALF_EXTENTS, SUPPORT_HALF_EXTENTS, com_margin_from_base, retime_factor,
)
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda

DT = 0.025


def _min_jerk(T, n):
    """A rest-to-rest 0 -> 1 move over T s on n frames: (t, position, velocity, acceleration)."""
    t = np.linspace(0.0, T, n)
    u = t / T
    return (t, 10 * u**3 - 15 * u**4 + 6 * u**5, (30 * u**2 - 60 * u**3 + 30 * u**4) / T,
            (60 * u - 180 * u**2 + 120 * u**3) / T**2)


@functools.lru_cache(maxsize=1)
def _simplified():
    import yourdfpy

    from cutamp.robots.t1 import T1_ASSETS_DIR
    return yourdfpy.URDF.load(str(T1_ASSETS_DIR / "t1_simplified.urdf"), load_meshes=False,
                              build_scene_graph=True, build_collision_scene_graph=False, load_collision_meshes=False)


def _reference_zmp(moves, T, t, h=1e-4, g=9.81):
    """``(zmp_xy, com_xy)`` about COM_CENTER at time ``t`` of the min-jerk move ``moves`` ({joint: delta}
    from home, over T s), from first principles: float64 yourdfpy on t1_simplified.urdf, every link's CoM
    acceleration by central differences of the true trajectory, its spin L = R I R^T w (w from R' R^T)
    differentiated the same way; sum m (g + z'') x - m z x'' - L'_y over the links (y: + L'_x)."""
    from cutamp.robots.t1 import JOINT_NAMES_FULL, t1_home
    u = _simplified()
    links = [(n, link.inertial) for n, link in u.link_map.items() if link.inertial is not None and link.inertial.mass]
    m = np.array([i.mass for _, i in links])
    inertia = np.stack([i.inertia for _, i in links])
    home = dict(zip(JOINT_NAMES_FULL, t1_home))

    def state(tau):
        x = min(max((t + tau) / T, 0.0), 1.0)
        s = 10 * x**3 - 15 * x**4 + 6 * x**5
        u.update_cfg({j: home.get(j, 0.0) + s * moves.get(j, 0.0) for j in u.actuated_joint_names})
        X = [u.get_transform(n) for n, _ in links]
        return (np.array([Xi[:3, :3] @ i.origin[:3, 3] + Xi[:3, 3] for Xi, (_, i) in zip(X, links)]),
                np.array([Xi[:3, :3] for Xi in X]))

    st = {k: state(k * h) for k in (-2, -1, 0, 1, 2)}
    x, _ = st[0]
    a = (st[1][0] - 2 * x + st[-1][0]) / h**2

    def spin(k):
        W = (st[k + 1][1] - st[k - 1][1]) / (2 * h) @ st[k][1].transpose(0, 2, 1)
        w = 0.5 * np.stack([W[:, 2, 1] - W[:, 1, 2], W[:, 0, 2] - W[:, 2, 0], W[:, 1, 0] - W[:, 0, 1]], -1)
        return np.einsum("lij,ljk,lmk,lm->li", st[k][1], inertia, st[k][1], w).sum(0)

    dL = (spin(1) - spin(-1)) / (2 * h)
    f = m * (g + a[:, 2])
    zmp = np.array([(f * x[:, 0] - m * x[:, 2] * a[:, 0]).sum() - dL[1],
                    (f * x[:, 1] - m * x[:, 2] * a[:, 1]).sum() + dL[0]]) / f.sum()
    return zmp - COM_CENTER, (m[:, None] * x).sum(0)[:2] / m.sum() - COM_CENTER


def _margin(xy, half):
    return float((np.asarray(half) - np.abs(np.asarray(xy))).min())


def test_retime_factor_scales_with_square():
    # Slowing by s divides the ZMP's offset from the CoM by s^2.
    hx = SUPPORT_HALF_EXTENTS[0]
    com = torch.tensor([[hx - 0.01, 0.0]])
    assert retime_factor(com, com + torch.tensor([[0.04, 0.0]]), SUPPORT_HALF_EXTENTS) == pytest.approx(2.0)
    assert retime_factor(com, com + torch.tensor([[0.005, 0.0]]), SUPPORT_HALF_EXTENTS) == 1.0


def test_retime_factor_zero_room_is_inf():
    hx = SUPPORT_HALF_EXTENTS[0]
    com = torch.tensor([[hx, 0.0]])
    assert retime_factor(com, com + torch.tensor([[0.001, 0.0]]), SUPPORT_HALF_EXTENTS) == math.inf


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


def _ramp(world, moves, T):
    """The rest-to-rest min-jerk move ``moves`` ({joint: delta} from home) over T s at DT, other joints at
    home, as a full-dof JointState in world.kinematics order."""
    from curobo.types import JointState
    names = list(world.kinematics.joint_names)
    n = int(round(T / DT)) + 1
    _, p, v, a = _min_jerk(T, n)
    home = world.q_init.view(1, -1).float()
    pos, vel, acc = home.repeat(n, 1), torch.zeros(n, len(names)).to(home), torch.zeros(n, len(names)).to(home)
    for name, delta in moves.items():
        j = names.index(name)
        pos[:, j] = home[0, j] + delta * pos.new_tensor(p)
        vel[:, j], acc[:, j] = delta * pos.new_tensor(v), delta * pos.new_tensor(a)
    return JointState(position=pos, velocity=vel, acceleration=acc, joint_names=names)


def _torso_ramp(world, pitch, T):
    return _ramp(world, {"Torso_Pitch": pitch}, T)


def _gate_view(world, plan):
    """The gate's own ``(com_xy, zmp_xy)`` of a plan."""
    from cutamp.com_polygon_cost import com_zmp_in_support
    from cutamp.motion_solver import ZMP_STEP, _taylor_positions
    return com_zmp_in_support(world, _taylor_positions(world, None, plan), ZMP_STEP)


def _reference_margins(moves, T):
    """Min over every DT frame of the reference ``(com, zmp)`` margins against SUPPORT and against the soles."""
    rows = [_reference_zmp(moves, T, k * DT) for k in range(int(round(T / DT)) + 1)]
    return {(name, rect): min(_margin(r[i], half) for r in rows)
            for i, name in ((1, "com"), (0, "zmp")) for rect, half in (("support", SUPPORT_HALF_EXTENTS),
                                                                        ("soles", SOLE_HALF_EXTENTS))}


@needs_cuda
def test_bodies_leave_the_robot_config_alone():
    # The bodies' kinematics is built from the cached robot config, whose tool frames every later IK solver
    # and motion planner reads: rewritten to the 36 links, the next run in the process found no IK solution.
    from cutamp.robots.t1 import SIMPLIFIED, get_t1_bodies, t1_curobo_cfg
    before = list(t1_curobo_cfg(SIMPLIFIED)["robot_cfg"]["kinematics"]["tool_frames"])
    get_t1_bodies(SIMPLIFIED)
    assert t1_curobo_cfg(SIMPLIFIED)["robot_cfg"]["kinematics"]["tool_frames"] == before


@needs_cuda
def test_zmp_matches_multibody_reference():
    # Torso, waist, both arms and the legs at once, so every link moves and spins: the gate's ZMP, from the
    # plan's own velocities and accelerations, matches the float64 first-principles one, at rest and at the
    # acceleration peaks.
    world = make_blocks_t1_world()
    moves = {"Torso_Pitch": -0.5, "Waist_Yaw": 0.4, "Left_Shoulder_Pitch": -1.0, "Right_Elbow_Pitch": -0.8,
             "knee_pitch": 0.3, "ankle_pitch": -0.15}
    com, zmp = _gate_view(world, _ramp(world, moves, 0.6))
    for k in (0, 3, 6, 12, 18, 24):
        ref_zmp, ref_com = _reference_zmp(moves, 0.6, k * DT)
        np.testing.assert_allclose(com[k].cpu().numpy(), ref_com, atol=2e-4)
        np.testing.assert_allclose(zmp[k].cpu().numpy(), ref_zmp, atol=3e-4)


@needs_cuda
def test_gate_rejects_com_outside():
    # A deep bow puts the CoM past the toe (13.5 cm at Torso_Pitch -1.6, the front edge is 10.15 cm).
    from cutamp.motion_solver import _balance_gate
    world = make_blocks_t1_world()
    with pytest.raises(RuntimeError, match="CoM leaves"):
        _balance_gate(world, None, _torso_ramp(world, -1.6, 1.0), DT)


@needs_cuda
def test_gate_rejects_com_between_support_and_soles():
    # A slow bow to -0.9 keeps its CoM over the real soles but past the 2 cm margin: rejected.
    from cutamp.motion_solver import _balance_gate
    ref = _reference_margins({"Torso_Pitch": -0.9}, 4.0)
    assert ref["com", "soles"] > 0 > ref["com", "support"]
    world = make_blocks_t1_world()
    with pytest.raises(RuntimeError, match="CoM leaves"):
        _balance_gate(world, None, _torso_ramp(world, -0.9, 4.0), DT)


@needs_cuda
def test_gate_slows_a_bow_start_by_the_bodies_zmp():
    # A bow to -0.5 in 0.8 s: the cart-table ZMP of the CoM alone stays inside the support rectangle, but the
    # trunk swinging forward above the hips puts the real ZMP 8 mm past it (still over the soles). The
    # gate slows it until the slowed bow's reference ZMP is inside the 2 cm margin.
    from cutamp.motion_solver import _balance_gate
    moves = {"Torso_Pitch": -0.5}
    ref = _reference_margins(moves, 0.8)
    assert ref["com", "support"] > 0 and ref["zmp", "soles"] > 0 > ref["zmp", "support"]
    world = make_blocks_t1_world()
    out, stats, _ = _balance_gate(world, None, _torso_ramp(world, -0.5, 0.8), DT)
    assert stats["retime"] > 1.0 and stats["zmp_min_margin"] >= 0
    assert _reference_margins(moves, 0.8 * stats["retime"])["zmp", "support"] >= 0


@needs_cuda
def test_gate_tries_the_retime_cap():
    # A bow to -0.6 in 0.3 s fits at a 3.76x slowdown. The first try (1.05 x retime_factor, 3.60x) falls
    # short and the next step (x1.25) overshoots the 4x cap: the gate still tries 4x before giving up.
    from cutamp.motion_solver import RETIME_MAX, _balance_gate
    world = make_blocks_t1_world()
    out, stats, _ = _balance_gate(world, None, _torso_ramp(world, -0.6, 0.3), DT)
    assert 3.6 < stats["retime"] <= RETIME_MAX and stats["zmp_min_margin"] >= 0


@needs_cuda
def test_gate_slows_fast_lean():
    # A bow to -0.6 is CoM-safe (8 cm), but in 0.5 s its ZMP swings far past the edges.
    from cutamp.motion_solver import _balance_gate, _retime
    world = make_blocks_t1_world()
    plan = _torso_ramp(world, -0.6, 0.5)
    com, zmp = _gate_view(world, plan)
    margin = lambda xy: com_margin_from_base(xy, SUPPORT_HALF_EXTENTS).min()  # noqa: E731
    assert margin(com) >= 0 > margin(zmp)
    out, stats, fast = _balance_gate(world, None, plan, DT)
    assert stats["retime"] > 1.0
    assert out.position.shape[0] == round((plan.position.shape[0] - 1) * stats["retime"]) + 1
    com, zmp = _gate_view(world, out)
    assert margin(zmp) >= 0 and stats["zmp_min_margin"] >= 0
    assert torch.allclose(out.position[-1], plan.position[-1], atol=1e-5)
    # ... and keeps the segment at the motion planner's timing: the planner's own frames, the gate's spline.
    assert fast.position.shape == plan.position.shape and fast.joint_names == plan.joint_names
    torch.testing.assert_close(fast.position, plan.position, rtol=0, atol=1e-5)
    torch.testing.assert_close(fast.acceleration, _retime(plan, 1.0, DT).acceleration)


@needs_cuda
def test_gate_passes_slow_motion_untouched():
    from cutamp.motion_solver import _balance_gate
    world = make_blocks_t1_world()
    plan = _torso_ramp(world, -0.3, 2.0)
    out, stats, fast = _balance_gate(world, None, plan, DT)
    assert out is plan and stats["retime"] == 1.0 and fast is None


@needs_cuda
def test_simplified_gate_only_slows_and_names_no_binding():
    # Only a model standing on soles has its pace set by the gate (velocity, torque, ZMP). The simplified model's gate
    # is unchanged: a segment with ZMP room stays at the planner's timing however slow it is, a fast one is slowed
    # for its ZMP, and the stats are the same three keys (no binding check), at s >= 1.
    from cutamp.motion_solver import _balance_gate
    world = make_blocks_t1_world()
    slow = _torso_ramp(world, -0.3, 4.0)
    out, stats, fast = _balance_gate(world, None, slow, DT)
    assert out is slow and fast is None and set(stats) == {"retime", "com_min_margin", "zmp_min_margin"}
    assert stats["retime"] == 1.0
    out, stats, fast = _balance_gate(world, None, _torso_ramp(world, -0.6, 0.5), DT)
    assert fast is not None and set(stats) == {"retime", "com_min_margin", "zmp_min_margin"}
    assert stats["retime"] > 1.0
