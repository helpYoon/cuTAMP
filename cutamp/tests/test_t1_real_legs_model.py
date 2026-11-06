"""The planning model as data (design 7.2, 7.4; calls 11, 12, 15): the real legs' layout, the soft-cost default and
the removed options (the ablation's and ``--robot``), the support rectangles, and the blocks_t1 world on the real legs,
standing on the home stance. The GPU tests share conftest's module-scoped ``real_world``: one world per module."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cutamp.com_polygon_cost import LOAD_GUARD_A, support_rectangles
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.robots.t1 import REAL_LEGS
from cutamp.tests.conftest import needs_cuda

CUTAMP = Path(__file__).resolve().parents[1]
T1_29DOF = CUTAMP / "robots" / "assets" / "t1_29dof"
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
    (test_real_legs_init's retract test reuses it)."""
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
    assert len(m.joint_names) == 33
    assert m.joint_names[m.base] == tuple(f"base_j_{n}" for n in ("x", "y", "z", "pitch", "roll", "yaw"))
    assert m.joint_names[m.left_arm][0] == "Left_Shoulder_Pitch" and m.joint_names[m.right_arm][-1] == "Right_Hand_Roll"
    assert set(m.joint_names[m.body]) == {"Waist"} | {f"{s}_{j}" for s in ("Left", "Right") for j in
        ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")}
    assert (m.base, m.left_arm, m.right_arm, m.body) == (slice(0, 6), slice(6, 13), slice(13, 20), slice(20, 33))
    assert len(m.home) == 33 and m.sole_frames == ("left_sole_link", "right_sole_link")
    assert m.envelope.name == "route_a_start"


def _cli_config(monkeypatch, *argv) -> TAMPConfiguration:
    """The TAMPConfiguration run_cutamp's CLI builds from ``argv``, with the run itself stubbed out."""
    import sys

    import cutamp.scripts.run_cutamp as rc
    seen = []
    monkeypatch.setattr(rc, "cutamp_demo", lambda env, config, **kw: seen.append(config))
    monkeypatch.setattr(sys, "argv", ["run_cutamp", *argv])
    rc.entrypoint()
    return seen[-1]


def test_soft_cost_default_and_choice(monkeypatch):
    # CostReducer sums the soft costs in this order, so the order is part of a same-seed run's result.
    default = ["place_close_to_base", "joint_limit_margin", "minimize_body_movement"]
    assert TAMPConfiguration().soft_cost == default and _cli_config(monkeypatch).soft_cost == default
    assert TAMPConfiguration(soft_cost=["joint_limit_margin"]).soft_cost == ["joint_limit_margin"]
    assert _cli_config(monkeypatch, "--soft_cost", "joint_limit_margin").soft_cost == ["joint_limit_margin"]
    assert TAMPConfiguration(soft_cost=[]).soft_cost == []


#: The paper's ablation options, which main fixes at the full system (they live on branch ablation-baseline): each
#: CLI spelling, with the argument it took, and each config field at its old default.
REMOVED_CLI_OPTIONS = (["--no_coupled_reik"], ["--reik_balance_first"], ["--no_reik_balance_first"],
                       ["--no_enable_com_polygon"], ["--no_enable_com_aware_ik"], ["--no_com_all"], ["--force_adam"],
                       ["--soft_in_adam", "always"], ["--robot", "real_legs"])
REMOVED_FIELDS = {"enable_com_polygon": True, "enable_com_aware_ik": True, "enable_com_init_gate": True,
                  "enable_com_hard_constraint": True, "coupled_reik": True, "force_adam": False,
                  "reik_balance_first": True, "soft_in_adam": "auto", "reik_num_seeds": 4, "reik_lbfgs_iters": 40,
                  "robot": "real_legs"}


@pytest.mark.parametrize("argv", REMOVED_CLI_OPTIONS, ids=lambda argv: argv[0])
def test_removed_options_are_rejected(monkeypatch, capsys, argv):
    # An old command line fails loudly, even one that only restated a default.
    with pytest.raises(SystemExit) as exit_info:
        _cli_config(monkeypatch, *argv)
    assert exit_info.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("name", REMOVED_FIELDS)
def test_removed_fields_are_rejected(name):
    with pytest.raises(TypeError, match=name):
        TAMPConfiguration(**{name: REMOVED_FIELDS[name]})


def test_joint_limit_margin_is_registered_and_weighted():
    # Unregistered names pass through UNWEIGHTED in CostReducer, so the weight entry is mandatory.
    from cutamp.config import SUPPORTED_SOFT_COSTS
    from cutamp.scripts.utils import default_constraint_to_mult
    assert "joint_limit_margin" in SUPPORTED_SOFT_COSTS
    validate_tamp_config(TAMPConfiguration(soft_cost=["joint_limit_margin"]))
    assert default_constraint_to_mult["soft"]["joint_limit_margin"] == 1e3


def test_shrunken_bounds_penalty_is_zero_strictly_inside():
    from cutamp.joint_limit_cost import shrunken_bounds_penalty
    lo, hi = torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0])
    q = torch.tensor([[0.0, 1.0], [-0.99, 1.99]])
    assert shrunken_bounds_penalty(q, lo, hi).sum() == 0.0


def test_shrunken_bounds_penalty_is_quadratic_in_penetration():
    # 0.2 below the lower bound -> 0.04; doubling the depth quadruples it.
    from cutamp.joint_limit_cost import shrunken_bounds_penalty
    lo, hi = torch.tensor([0.0]), torch.tensor([1.0])
    p1 = shrunken_bounds_penalty(torch.tensor([[-0.2]]), lo, hi)
    p2 = shrunken_bounds_penalty(torch.tensor([[-0.4]]), lo, hi)
    assert torch.allclose(p1, torch.tensor([[0.04]]))
    assert torch.allclose(p2, 4.0 * p1)


def test_shrunken_bounds_penalty_penalizes_both_sides():
    from cutamp.joint_limit_cost import shrunken_bounds_penalty
    lo, hi = torch.tensor([0.0]), torch.tensor([1.0])
    below = shrunken_bounds_penalty(torch.tensor([[-0.1]]), lo, hi)
    above = shrunken_bounds_penalty(torch.tensor([[1.1]]), lo, hi)
    assert below.item() > 0.0 and above.item() > 0.0
    assert torch.allclose(below, above)


WAIST = REAL_LEGS.joint_names.index("Waist")


def _real_limits() -> torch.Tensor:
    """The real legs' ``[2, 33]`` limits in the model's joint order, row 0 lower: t1_29dof.yml's, which every factory's
    kinematics reports (test_factories_keep_the_soft_limits_on_their_joints), so the soft cost's
    ``world.joint_limits``."""
    soft = _yaml_soft_limits()
    return torch.tensor([soft[n] for n in REAL_LEGS.joint_names], dtype=torch.float32).T.contiguous()


def _joint_limit_margin(particles: dict, n: int) -> torch.Tensor:
    """The joint_limit_margin soft cost as cost_function.py computes it: on the real legs' limits."""
    from cutamp.joint_limit_cost import joint_limit_margin_soft_cost
    return joint_limit_margin_soft_cost(particles, _real_limits(), num_particles=n, device=torch.device("cpu"))


def _real_home(n: int) -> torch.Tensor:
    return torch.tensor(REAL_LEGS.home, dtype=torch.float32).expand(n, -1).clone()


def test_joint_limit_margin_is_zero_inside_the_limits():
    # No margin band: home, and a conf 1 mrad inside the Waist's upper limit, cost exactly nothing.
    near = _real_home(4)
    near[:, WAIST] = _real_limits()[1, WAIST] - 0.001
    cost = _joint_limit_margin({"q_retract": _real_home(4), "left_q1": near}, 4)
    assert cost.shape == (4,)
    assert torch.all(cost == 0.0)


def test_joint_limit_margin_penalizes_a_conf_past_its_limit():
    # The Waist 0.08 rad past its upper limit: penalty 0.08^2 = 6.4e-3.
    q = _real_home(2)
    q[:, WAIST] = _real_limits()[1, WAIST] + 0.08
    assert torch.allclose(_joint_limit_margin({"q_pick": q}, 2), torch.full((2,), 0.08 ** 2), atol=1e-6)


def test_joint_limit_margin_excludes_the_initial_confs():
    # A violating conf under every initial-state name and a clean home under q_pick: the total is exactly 0 (the
    # exclusion works inside the stacked path, not just via the empty-dict fallback). Initial states must never
    # acquire limit gradients through the Adam soft loss.
    bad = _real_home(2)
    bad[:, WAIST] = _real_limits()[1, WAIST] + 0.08
    cost = _joint_limit_margin({"q0": bad, "left_q0": bad.clone(), "right_q0": bad.clone(), "q_pick": _real_home(2)}, 2)
    assert torch.all(cost == 0.0)


def test_joint_limit_margin_without_confs_is_zero():
    cost = _joint_limit_margin({"pose1": torch.zeros(3, 4)}, 3)
    assert cost.shape == (3,) and torch.all(cost == 0.0)


def test_joint_limit_margin_gradient_restores_and_skips_the_initial_confs():
    # (a) The gradient through the hinge^2 points back toward the interior; (b) an excluded initial-state conf gets
    # no gradient even when it is in the dict, the safety property the exclusion exists for.
    bad = _real_home(2)
    bad[:, WAIST] = _real_limits()[0, WAIST] - 0.08  # 0.08 below the lower limit
    q_pick, left_q0 = bad.clone().requires_grad_(True), bad.clone().requires_grad_(True)
    _joint_limit_margin({"q_pick": q_pick, "left_q0": left_q0}, 2).sum().backward()
    assert left_q0.grad is None or torch.all(left_q0.grad == 0.0)
    # d/dq of relu(lower - q)^2 is -2 relu(lower - q): pushing q up, toward the interior, lowers the cost.
    assert q_pick.grad is not None and bool((q_pick.grad[:, WAIST] < 0.0).all())
    others = [k for k in range(len(REAL_LEGS.joint_names)) if k != WAIST]
    assert torch.all(q_pick.grad[:, others] == 0.0)  # the untouched joints get no gradient


def test_support_rectangles():
    gate, key = support_rectangles((0.2249, 0.100), 0.10625, 0.1)
    assert gate == pytest.approx((0.09245, 0.135)) and key == pytest.approx((0.08745, 0.130))
    assert REAL_LEGS.support == ("stance", (0.0, 0.0), gate, key)


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
    # cuRobo never reorders position_limit_clip with the cspace, so a build whose joint order differs from
    # the YAML's puts the soft limits on the wrong joints, silently. Each factory asserts the order; this checks it
    # and every factory's limits, by name.
    from cutamp.robots.t1 import get_t1_motion_planner
    soft = _yaml_soft_limits()
    planner = get_t1_motion_planner(REAL_LEGS, real_world.world_cfg, use_cuda_graph=False)
    builds = {"kinematics": real_world.kinematics, "kinematics_jac": real_world.kinematics_jac,
              "bodies": real_world.bodies.kinematics, "ik": real_world.ik_solver.kinematics,
              "planner": planner.kinematics}
    for what, kin in builds.items():
        names = list(kin.joint_names)
        assert names == list(REAL_LEGS.joint_names), what
        lo, hi = kin.get_joint_limits().position.double().cpu().numpy()
        want_lo, want_hi = np.array([soft[n] for n in names]).T
        assert np.abs(lo - want_lo).max() < 1e-6 and np.abs(hi - want_hi).max() < 1e-6, what
    assert real_world.bodies.kinematics.tool_frames[0] == "stance" and float(real_world.bodies.mass[0]) == 0.0
    # The planner keeps the YAML's tool frames in the YAML's order; the floor hook and the CoM cost find theirs.
    yml = yaml.safe_load((T1_29DOF / "t1_29dof.yml").read_text())["robot_cfg"]["kinematics"]["tool_frames"]
    assert list(planner.kinematics.tool_frames) == list(planner.trajopt_solver.tool_frames) == yml


@needs_cuda
def test_retry_rounds_seed_from_retry_seed(real_world, monkeypatch):
    # Each CoM-gate retry round starts IK from _retry_seed's conf, at sigma 0.1,
    # 0.2 and 0.3, which stands the real legs on the stance (test_real_legs_init.py checks its seeds). The noise moves
    # the body and the active arm; the inactive arm holds q_init. IK and the CoM mask are stubbed: every row fails the
    # gate, so all three retries run, and only their seeds count.
    import cutamp.com_polygon_cost as com_polygon_cost
    import cutamp.particle_initialization as particle_initialization

    class _Result:
        def copy_at_batch_indices(self, other, idx):
            pass

    n, m, q0, seeds, made = 8, real_world.model, real_world.q_init, [], []
    retry_seed = particle_initialization._retry_seed

    def _ik(world, world_from_ee, arm, *, current_state_q=None):
        if current_state_q is not None:
            seeds.append(current_state_q)
        return _Result()

    def _seed(world, batch_size, arm, noise_std):
        made.append(((batch_size, arm, noise_std), retry_seed(world, batch_size, arm, noise_std)))
        return made[-1][1]

    monkeypatch.setattr(particle_initialization, "_ik_for_pose", _ik)
    monkeypatch.setattr(particle_initialization, "_retry_seed", _seed)
    monkeypatch.setattr(particle_initialization, "_ik_solution_to_full_q", lambda result, world: q0.expand(n, -1))
    monkeypatch.setattr(com_polygon_cost, "compute_com_polygon_mask",
                        lambda world, q: torch.zeros(len(q), dtype=torch.bool, device=q.device))
    torch.manual_seed(0)
    particle_initialization._ik_for_pose_com_safe(real_world, torch.eye(4, device=q0.device).expand(n, 4, 4), "left")
    assert [args for args, _ in made] == [(n, "left", pytest.approx(sigma)) for sigma in (0.1, 0.2, 0.3)]
    assert len(seeds) == 3 and all(seed is q for seed, (_, q) in zip(seeds, made))
    for seed in seeds:
        assert torch.equal(seed[:, m.right_arm], q0[m.right_arm].expand(n, -1))
        for noisy in (m.body, m.left_arm):
            assert not torch.equal(seed[:, noisy], q0[noisy].expand(n, -1))
