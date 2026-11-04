"""The planning model as data (design 7.2, 7.4; calls 11, 12, 15): ``T1Model``'s two instances, the ``--robot``
switch and its refusals, the support rectangles, and the blocks_t1 world on the real legs, standing on the home
stance. The GPU tests share conftest's module-scoped ``real_world``: one world per module."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cutamp.com_polygon_cost import (
    COM_CENTER,
    COM_HALF_EXTENTS,
    LOAD_GUARD_A,
    SUPPORT_HALF_EXTENTS,
    support_rectangles,
)
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.robots.t1 import BASE_INDICES, JOINT_NAMES_FULL, REAL_LEGS, SIMPLIFIED, get_model, t1_home
from cutamp.tests.conftest import needs_cuda

CUTAMP = Path(__file__).resolve().parents[1]
T1_29DOF = CUTAMP / "robots" / "assets" / "t1_29dof"
SIMPLIFIED_CONSTANTS = re.compile(
    r"\b(CUROBO_DOF|URDF_DOF|BASE_DOF|BASE_INDICES|BODY_INDICES|LEFT_ARM_INDICES|RIGHT_ARM_INDICES|"
    r"JOINT_NAMES_FULL|t1_home|T1_LIMIT_MARGIN_LOWER|T1_LIMIT_MARGIN_UPPER)\b"
)
#: The definitions, their re-exports, and a script that checks simplified plans only.
MAY_READ_SIMPLIFIED_CONSTANTS = {"robots/t1.py", "robots/__init__.py", "scripts/check_plan_margins.py"}
LEGS = {
    side: {f"{link}_{side.title()}" for link in ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Shank", "Ankle_Cross")}
    | {f"{side}_foot_link", f"{side}_sole_link"}
    for side in ("left", "right")
}
ARMS_AND_GRIPPERS = (
    {f"A{s}{k}" for s in "LR" for k in range(1, 7)}
    | {f"{side}_{link}" for side in ("left", "right")
       for link in ("hand_link", "base_link", "Link1", "Link11", "Link2", "Link22")}
    | {"attached_object_left", "attached_object_right"}
)


def _fk(kin, q: torch.Tensor):
    """``kin``'s kinematics state at the single conf ``q``, given in ``kin``'s joint order."""
    from curobo.types import JointState
    return kin.compute_kinematics(JointState.from_position(q.view(1, -1), joint_names=list(kin.joint_names)))


def _home(world) -> torch.Tensor:
    return torch.tensor(world.model.home, device=world.device)


def _matrices(state, links) -> torch.Tensor:
    """``[len(links), 4, 4]`` world poses of ``links`` in a single-conf state."""
    return torch.stack([state.tool_poses.get_link_pose(link, make_contiguous=True).get_matrix().reshape(4, 4)
                        for link in links])


def _self_collision(kin, q: torch.Tensor) -> float:
    """cuRobo's self-collision cost at ``q``, as cost_function.py builds it."""
    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
    from curobo.types import JointState
    cost = SelfCollisionCost(SelfCollisionCostCfg(weight=kin.device_cfg.to_device([1.0]), device_cfg=kin.device_cfg,
                                                  self_collision_kin_config=kin.get_self_collision_config()))
    spheres = kin.compute_kinematics(JointState.from_position(q.view(1, 1, -1))).robot_spheres
    cost.setup_batch_tensors(1, 1)
    return float(cost.forward(spheres).sum())


def _leg_gap(kin, q: torch.Tensor) -> float:
    """The smallest surface-to-surface distance between a left-leg sphere and a right-leg sphere at ``q``, metres."""
    kp = kin.config.kinematics_config
    link_of = {i: name for name, i in kp.link_name_to_idx_map.items()}
    sphere_link = [link_of[int(i)] for i in kp.link_sphere_idx_map.cpu()]
    spheres = _fk(kin, q).robot_spheres.reshape(-1, 4).double().cpu()
    left, right = ([k for k, link in enumerate(sphere_link) if link in LEGS[side]] for side in ("left", "right"))
    assert left and right
    a, b = spheres[left], spheres[right]
    return float((torch.cdist(a[:, :3], b[:, :3]) - a[:, 3:] - b[:, 3].unsqueeze(0)).min())


def hips_crossed(world) -> torch.Tensor:
    """Home with both Hip_Roll joints at their inward soft limits, so the thighs cross: a self-colliding conf
    (ruling R5, reused by Task 4's retract test)."""
    names, (lo, hi) = list(world.model.joint_names), world.joint_limits
    q = _home(world)
    q[names.index("Left_Hip_Roll")] = lo[names.index("Left_Hip_Roll")]  # +x roll swings a foot to +y: left is out
    q[names.index("Right_Hip_Roll")] = hi[names.index("Right_Hip_Roll")]
    return q


def _yaml_soft_limits() -> dict:
    """``{joint: (lower, upper)}`` as t1_29dof.yml sets them, by name: the planner URDF's range (the base's from its
    extra links) shrunk by the joint's ``position_limit_clip``."""
    kinematics = yaml.safe_load((T1_29DOF / "t1_29dof.yml").read_text())["robot_cfg"]["kinematics"]
    hard = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
            for j in ET.parse(T1_29DOF / "t1_planner.urdf").getroot().findall("joint") if j.find("limit") is not None}
    hard |= {link["joint_name"]: tuple(link["joint_limits"])
             for link in kinematics["extra_links"].values() if "joint_limits" in link}
    cspace = kinematics["cspace"]
    clip = dict(zip(cspace["joint_names"], cspace["position_limit_clip"]))
    return {name: (hard[name][0] + clip[name], hard[name][1] - clip[name]) for name in cspace["joint_names"]}


def test_real_legs_layout():
    m = REAL_LEGS
    assert len(m.joint_names) == 33 and not m.lock_base
    assert m.joint_names[m.base] == tuple(f"base_j_{n}" for n in ("x", "y", "z", "yaw", "pitch", "roll"))
    assert m.joint_names[m.left_arm][0] == "Left_Shoulder_Pitch" and m.joint_names[m.right_arm][-1] == "Right_Hand_Roll"
    assert set(m.joint_names[m.body]) == {"Waist"} | {f"{s}_{j}" for s in ("Left", "Right") for j in
        ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")}
    assert (m.base, m.left_arm, m.right_arm, m.body) == (slice(0, 6), slice(6, 13), slice(13, 20), slice(20, 33))
    assert not any(m.margin_lower + m.margin_upper) and len(m.margin_lower) == len(m.margin_upper) == 33
    assert len(m.home) == 33 and m.sole_frames == ("left_sole_link", "right_sole_link") and m.envelope is None
    assert SIMPLIFIED.home == t1_home and SIMPLIFIED.base == BASE_INDICES and SIMPLIFIED.joint_names == JOINT_NAMES_FULL
    assert SIMPLIFIED.lock_base and SIMPLIFIED.sole_frames == ()


def test_unknown_robot():
    assert get_model("simplified") is SIMPLIFIED and get_model("real_legs") is REAL_LEGS
    with pytest.raises(ValueError, match="t2"):
        get_model("t2")
    with pytest.raises(ValueError, match="t2"):
        validate_tamp_config(TAMPConfiguration(robot="t2"))


def test_soft_cost_default_by_robot():
    assert TAMPConfiguration().soft_cost == ["place_close_to_base", "joint_limit_margin"]
    assert TAMPConfiguration(robot="real_legs").soft_cost == ["place_close_to_base", "joint_limit_margin", "minimize_body_movement"]
    assert TAMPConfiguration(robot="real_legs", soft_cost=["joint_limit_margin"]).soft_cost == ["joint_limit_margin"]
    assert TAMPConfiguration(soft_cost=[]).soft_cost == []


@pytest.mark.parametrize("kw", [dict(enable_com_polygon=False), dict(enable_com_aware_ik=False),
    dict(enable_com_init_gate=False), dict(enable_com_hard_constraint=False), dict(coupled_reik=False),
    dict(force_adam=True), dict(reik_balance_first=False)])
def test_real_legs_refuses_ablation_paths(kw):
    validate_tamp_config(TAMPConfiguration(robot="real_legs"))
    validate_tamp_config(TAMPConfiguration(robot="simplified", **kw))
    with pytest.raises(ValueError, match="real_legs"):
        validate_tamp_config(TAMPConfiguration(robot="real_legs", **kw))


def test_no_module_reads_simplified_constants():
    # A simplified slice, home or margin read where world.model should be gives plausible, wrong numbers on the
    # real legs (Review Focus 1). Comments count too.
    hits = []
    for path in sorted(CUTAMP.rglob("*.py")):
        rel = path.relative_to(CUTAMP).as_posix()
        if rel.startswith("tests/") or rel in MAY_READ_SIMPLIFIED_CONSTANTS:
            continue
        hits += [f"{rel}:{k}: {line.strip()}" for k, line in enumerate(path.read_text().splitlines(), 1)
                 if SIMPLIFIED_CONSTANTS.search(line)]
    assert not hits, "\n".join(hits)


def test_support_rectangles():
    gate, key = support_rectangles((0.2249, 0.100), 0.10625, 0.1)
    assert gate == pytest.approx((0.09245, 0.135)) and key == pytest.approx((0.08745, 0.130))
    assert REAL_LEGS.support == ("stance", (0.0, 0.0), gate, key)
    assert SIMPLIFIED.support == ("mobile_base_link", COM_CENTER, SUPPORT_HALF_EXTENTS, COM_HALF_EXTENTS)


def test_load_guard_is_the_hull_of_its_32_sums():
    # Design 7.4: S_a = conv(a F_L + (1 - a) F_R  u  (1 - a) F_L + a F_R), over the two soles' corners.
    from scipy.spatial import ConvexHull
    assert LOAD_GUARD_A == 0.1
    a, (length, width), y0 = LOAD_GUARD_A, (0.2249, 0.100), 0.10625
    corners = np.array([(sx * length / 2, sy * width / 2) for sx in (-1, 1) for sy in (-1, 1)])
    left, right = corners + (0.0, y0), corners - (0.0, y0)
    sums = np.array([w * c_l + (1 - w) * c_r for c_l in left for c_r in right for w in (a, 1 - a)])
    assert sums.shape == (32, 2)
    hull = ConvexHull(sums)
    half = np.array([0.11245, 0.135])
    vertices = sums[hull.vertices]
    assert len(vertices) == 4 and np.abs(np.abs(vertices) - half).max() < 1e-9  # its vertices are the corners
    assert abs(hull.volume - 4 * half.prod()) < 1e-9                             # and it fills the rectangle


@needs_cuda
def test_home_soles_flat_on_the_floor(real_world):
    m = real_world.model
    home = _home(real_world)
    soles = _matrices(_fk(real_world.kinematics, home), m.sole_frames)
    p = soles[:, :3, 3].double().cpu()
    tilt = soles[:, :2, 2].double().norm(dim=-1)  # sine of the angle between each sole's z axis and the vertical
    assert p[:, 2].abs().max() <= 1e-6 and tilt.max() <= 1e-6, (p, tilt)
    assert abs(p[0, 1] - 0.10625) <= 1e-4 and abs(p[1, 1] + 0.10625) <= 1e-4 and abs(p[0, 0] - p[1, 0]) <= 1e-6, p
    ankles = _matrices(_fk(real_world.bodies.kinematics, home), ["Ankle_Cross_Left", "Ankle_Cross_Right"])
    assert ankles[:, :2, 3].double().mean(0).abs().max() <= 1e-6, ankles[:, :3, 3]  # decision 3: today's footprint
    assert abs(m.home[m.joint_names.index("base_j_z")] - 0.6791) <= 1e-4  # call 11: Trunk 0.6791 m above the soles
    fixture_names = [str(j) for j in np.load(T1_29DOF / "mujoco_fixture.npz")["joint_names"]]
    default_q = np.load(T1_29DOF / "t1_actuators.npz")["default_q"]
    body = [(name, value) for name, value in zip(m.joint_names, m.home) if not name.startswith("base_j_")]
    assert len(body) == 27 and all(value == float(default_q[fixture_names.index(name)]) for name, value in body)
    torch.testing.assert_close(real_world.stance_soles, soles, rtol=0.0, atol=1e-7)


@needs_cuda
def test_home_com_over_the_mid_sole(real_world):
    com = _fk(real_world.kinematics, _home(real_world)).robot_com[..., :3].reshape(3).double().cpu()
    mid_sole = real_world.stance_soles[:, :3, 3].double().mean(0).cpu()
    ahead, aside = (com - mid_sole)[:2].tolist()
    assert abs(ahead - 0.0120) <= 5e-4 and abs(aside) <= 5e-4, (ahead, aside)
    mass = float(real_world.kinematics.config.kinematics_config.link_masses_com[:, 3].sum())
    assert abs(mass - 34.6325) <= 1e-3, mass


@needs_cuda
def test_home_clears_itself(real_world):
    kin, home = real_world.kinematics, _home(real_world)
    assert _fk(kin, home).robot_spheres.shape[-2] == 167
    assert _self_collision(kin, home) == 0.0  # the gripper clears the thighs (call 11)
    gap = _leg_gap(kin, home)
    assert abs(gap - 0.102) <= 0.005, gap  # section 10: the shanks 0.102 m apart
    crossed = hips_crossed(real_world)
    assert _leg_gap(kin, crossed) < gap and _self_collision(kin, crossed) > 0.0  # the check fires
    from cutamp.robots.t1 import t1_curobo_cfg
    ignore = t1_curobo_cfg(real_world.model)["robot_cfg"]["kinematics"]["self_collision_ignore"]
    for link, others in ignore.items():
        for other in others:
            pair = {link, other}
            assert not (pair & LEGS["left"] and pair & LEGS["right"]), pair       # left leg against right leg
            assert not (pair & (LEGS["left"] | LEGS["right"]) and pair & ARMS_AND_GRIPPERS), pair  # legs against arms


@needs_cuda
def test_world_on_real_legs(real_world):
    m = real_world.model
    lo, hi = real_world.joint_limits
    home = _home(real_world)
    assert m is REAL_LEGS and real_world.joint_limits.shape == (2, 33)
    assert torch.equal(real_world.q_init, home) and bool(((lo <= home) & (home <= hi)).all())
    assert {"left_sole_link", "right_sole_link", "stance"} <= set(real_world.kinematics.tool_frames)
    assert real_world.tool_frames == ["left_base_link", "right_base_link"]
    assert real_world.stance_soles.shape == (2, 4, 4)


@needs_cuda
def test_factories_keep_the_soft_limits_on_their_joints(real_world):
    # Ruling R12: cuRobo never reorders position_limit_clip with the cspace, so a build whose joint order differs from
    # the YAML's puts the soft limits on the wrong joints, silently. Each factory asserts the order; this checks it
    # and every factory's limits, by name.
    from cutamp.robots.t1 import get_t1_motion_planner
    soft = _yaml_soft_limits()
    builds = {"kinematics": real_world.kinematics, "kinematics_jac": real_world.kinematics_jac,
              "bodies": real_world.bodies.kinematics, "ik": real_world.ik_solver.kinematics}
    for what, kin in builds.items():
        names = list(kin.joint_names)
        assert names == list(REAL_LEGS.joint_names), what
        lo, hi = kin.get_joint_limits().position.double().cpu().numpy()
        want_lo, want_hi = np.array([soft[n] for n in names]).T
        assert np.abs(lo - want_lo).max() < 1e-6 and np.abs(hi - want_hi).max() < 1e-6, what
    assert real_world.bodies.kinematics.tool_frames[0] == "stance" and float(real_world.bodies.mass[0]) == 0.0
    with pytest.raises(NotImplementedError):
        get_t1_motion_planner(REAL_LEGS, real_world.world_cfg)
