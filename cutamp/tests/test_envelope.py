"""Route A's envelope (design 7.10, call 22): the pinned file, the cut of the real legs' limits to it (a Trunk floor
on ``base_j_z``, the joint ranges through ``position_limit_clip``, never widening one), and the Trunk tilt bound, as a
Motion value over the confs and as a trajopt cost. Route A's envelope sets no Trunk floor or tilt bound, so the floor
and tilt tests replace it with one that does. The GPU tests share conftest's module-scoped ``real_world``: one world
per module."""

import dataclasses
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cutamp.robots.envelope import Envelope, load_envelope
from cutamp.robots.t1 import REAL_LEGS
from cutamp.tests.conftest import needs_cuda

ENVELOPES = Path(__file__).resolve().parents[1] / "robots" / "assets" / "envelopes"
PITCH, ROLL = REAL_LEGS.joint_names.index("base_j_pitch"), REAL_LEGS.joint_names.index("base_j_roll")
#: The tilt tests' bound, rad, in an envelope that replaces route A's.
TILT_MAX = 1.0
#: base_j_z's limits on the real legs, the legs' reach (build_planner_model.py), m: no envelope cut.
BASE_Z = (-0.046, 0.772)


def _with_envelope(**fields):
    """REAL_LEGS with its envelope's ``fields`` replaced: a new model, so t1_curobo_cfg's cache builds it afresh."""
    return dataclasses.replace(REAL_LEGS, envelope=REAL_LEGS.envelope._replace(**fields))


def _limits(model) -> dict:
    """``{joint: (lower, upper)}`` as cuRobo's Kinematics for ``model`` reports them (``get_joint_limits``)."""
    from cutamp.robots.t1 import get_t1_kinematics
    kin = get_t1_kinematics(model)
    lo, hi = kin.get_joint_limits().position.double().cpu().numpy()
    return {name: (lo[i], hi[i]) for i, name in enumerate(kin.joint_names)}


def test_route_a_start_values():
    path = ENVELOPES / "route_a_start.yml"
    envelope = load_envelope("route_a_start")
    assert envelope == Envelope(
        name="route_a_start",
        sha1=hashlib.sha1(path.read_bytes()).hexdigest(),
        trunk_height_min=None,
        trunk_tilt_max=None,
        joint_ranges={"Waist": (-0.911, 0.911)},
        speed_caps=None,
        acceleration_caps=None,
        trunk_speed_caps=None,
        payload_kg={"range": [0.0, 0.5], "tail_max": 1.0},
        stance_range={"xy_m": 2.0, "yaw_rad": math.pi},
        hold_s={"start_blend": 0.5, "grasp": 0.5, "end": 1.0},
    )
    assert REAL_LEGS.envelope == envelope and REAL_LEGS.envelope.name == "route_a_start"


def test_load_envelope_refuses_a_misnamed_file_or_another_field(tmp_path, monkeypatch):
    # The file's name is the one the planner asks for, and its fields are the Envelope's: a step-ranges field, deferred
    # with the step case, is refused.
    from cutamp.robots import envelope as envelope_module
    text = (ENVELOPES / "route_a_start.yml").read_text()
    monkeypatch.setattr(envelope_module, "ENVELOPES_DIR", tmp_path)
    (tmp_path / "other.yml").write_text(text)
    with pytest.raises(ValueError, match="route_a_start"):
        load_envelope("other")
    (tmp_path / "route_a_start.yml").write_text(text + "step_ranges: {travel_m: [0.10, 0.30]}\n")
    with pytest.raises(TypeError, match="step_ranges"):
        load_envelope("route_a_start")


def test_load_envelope_reads_a_set_floor_and_tilt(tmp_path, monkeypatch):
    # Route A's null floor and tilt load as None (test_route_a_start_values); an envelope that sets them loads floats.
    from cutamp.robots import envelope as envelope_module
    text = (ENVELOPES / "route_a_start.yml").read_text()
    assert text.count("trunk_height_min: null") == text.count("trunk_tilt_max: null") == 1
    monkeypatch.setattr(envelope_module, "ENVELOPES_DIR", tmp_path)
    text = text.replace("trunk_height_min: null", "trunk_height_min: 0.45")
    (tmp_path / "route_a_start.yml").write_text(text.replace("trunk_tilt_max: null", "trunk_tilt_max: 1"))
    envelope = load_envelope("route_a_start")
    assert (envelope.trunk_height_min, envelope.trunk_tilt_max) == (0.45, 1.0)
    assert type(envelope.trunk_tilt_max) is float


@needs_cuda
def test_envelope_cuts_never_widen():
    from cutamp.robots.t1 import t1_curobo_cfg
    real = _limits(REAL_LEGS)
    # Route A sets no Trunk floor, so the base keeps the legs' reach, as the robot YAML gives it; Waist keeps its soft
    # limit, which lies inside route A's +-0.911.
    import yaml
    extra = yaml.safe_load(REAL_LEGS.cfg_path.read_text())["robot_cfg"]["kinematics"]["extra_links"].values()
    base = {link["joint_name"]: tuple(link["joint_limits"]) for link in extra if "joint_limits" in link}
    assert sorted(base) == sorted(n for n in REAL_LEGS.joint_names if n.startswith("base_j_"))
    for joint, limits in base.items():
        np.testing.assert_allclose(real[joint], limits, atol=1e-6, err_msg=joint)
    np.testing.assert_allclose(real["base_j_z"], BASE_Z, atol=1e-6)
    np.testing.assert_allclose(real["Waist"], (-0.9108, 0.9108), atol=1e-6)
    for waist, want in (((-0.5, 0.5), (-0.5, 0.5)), ((-2.0, 2.0), (-0.9108, 0.9108))):
        got = _limits(_with_envelope(joint_ranges={"Waist": waist}))
        np.testing.assert_allclose(got["Waist"], want, atol=1e-6, err_msg=str(waist))
        assert {n: v for n, v in got.items() if n != "Waist"} == {n: v for n, v in real.items() if n != "Waist"}
    # A floor above the base's lower limit raises it, alone; one below leaves it; one at or above its upper raises.
    for floor, want in ((0.5, 0.5), (BASE_Z[0] - 0.2, BASE_Z[0])):
        got = _limits(_with_envelope(trunk_height_min=floor))
        np.testing.assert_allclose(got["base_j_z"], (want, BASE_Z[1]), atol=1e-6, err_msg=str(floor))
        assert {n: v for n, v in got.items() if n != "base_j_z"} == {n: v for n, v in real.items() if n != "base_j_z"}
    with pytest.raises(ValueError, match="base_j_z"):
        t1_curobo_cfg(_with_envelope(trunk_height_min=BASE_Z[1]))
    with pytest.raises(ValueError, match="Waist"):
        t1_curobo_cfg(_with_envelope(joint_ranges={"Waist": (2.0, 3.0)}))


@needs_cuda
def test_envelope_cut_moves_only_its_joint():
    # The cut writes position_limit_clip at the joint's index in cspace.joint_names, and cuRobo applies the clip
    # without reindexing it. A knee, far from Waist in the cspace order and with the other knee's clip, cut 0.1 rad
    # inside its soft limit at both ends moves cuRobo's limits on that knee alone.
    real = _limits(REAL_LEGS)
    lo, hi = real["Left_Knee_Pitch"]
    got = _limits(_with_envelope(joint_ranges={"Left_Knee_Pitch": (lo + 0.1, hi - 0.1)}))
    np.testing.assert_allclose(got["Left_Knee_Pitch"], (lo + 0.1, hi - 0.1), atol=1e-6)
    assert {n: v for n, v in got.items() if n != "Left_Knee_Pitch"} == {
        n: v for n, v in real.items() if n != "Left_Knee_Pitch"}


def test_cut_names_a_missing_trunk_floor_joint():
    # A floor needs base_j_z; without a floor (route A) the cut never looks for it.
    import copy
    from cutamp.robots.t1 import _cut_to_envelope, t1_curobo_cfg
    with pytest.raises(ValueError, match="base_j_z"):
        _cut_to_envelope({"extra_links": {}}, REAL_LEGS.envelope._replace(trunk_height_min=0.5))
    kinematics = copy.deepcopy(t1_curobo_cfg(REAL_LEGS)["robot_cfg"]["kinematics"])
    kinematics["extra_links"] = {}
    _cut_to_envelope(kinematics, REAL_LEGS.envelope)


def test_trunk_tilt_value():
    # The tilt is the angle between the Trunk's z axis and the vertical, arccos(cos(pitch) cos(roll)), for the base's
    # pitch, roll, then yaw (YXZ): 0, 1.0 (on the bound) and 0.946 rad pass, 1.006 rad at (0.75, 0.75) fails.
    from cutamp.joint_limit_cost import trunk_tilt_excess
    confs = torch.tensor(REAL_LEGS.home).repeat(2, 4, 1)  # [N, H, dof]
    for t, (pitch, roll) in enumerate(((0.0, 0.0), (1.0, 0.0), (0.7, 0.7), (0.75, 0.75))):
        confs[1, t, PITCH], confs[1, t, ROLL] = pitch, roll
    value = trunk_tilt_excess(confs, PITCH, ROLL, math.cos(TILT_MAX))
    assert value.shape == (2, 4)
    assert not bool(value[0].any()) and not bool(value[1, :3].any())
    assert math.isclose(float(value[1, 3]), math.cos(1.0) - math.cos(0.75) ** 2, rel_tol=1e-4)


def test_trunk_tilt_cost():
    from curobo.types import DeviceCfg
    from cutamp.joint_limit_cost import TrunkTiltCost, TrunkTiltCostCfg
    cos_max = math.cos(1.0)
    cost = TrunkTiltCost(TrunkTiltCostCfg(weight=[3.0], device_cfg=DeviceCfg(device=torch.device("cpu")),
                                          pitch_index=1, roll_index=2, cos_max=cos_max))
    q = torch.zeros(2, 3, 4)  # [B, H, dof]
    q[0, 1, 1], q[0, 2, 1:3] = 0.9, 0.7           # inside: a 0.9 rad bow, and 0.946 rad of bow and lean
    q[1, 0, 1], q[1, 1, 1], q[1, 2, 1:3] = -1.2, 1.2, 0.75  # outside: bent back, bent forward, both
    q.requires_grad_(True)
    out = cost.forward(SimpleNamespace(joint_state=SimpleNamespace(position=q)))
    assert out.shape == (2, 3, 1)
    assert not bool(out[0].any())
    want = 3.0 * (cos_max - torch.cos(q[1, :, 1]) * torch.cos(q[1, :, 2])).detach() ** 2
    torch.testing.assert_close(out[1, :, 0].detach(), want)
    out.sum().backward()
    assert not bool(q.grad[0].any())
    assert q.grad[1, 0, 1] < 0 < q.grad[1, 1, 1] and q.grad[1, 2, 1] > 0  # descent moves each bow toward 0


@needs_cuda
def test_motion_values_bound_the_tilt_on_the_real_legs(real_world, real_plan, monkeypatch):
    # Route A sets no tilt bound, so its Motion values have no trunk_tilt. Under an envelope that bounds the tilt at
    # 1.0 rad, the Motion value at every rollout conf, on the tolerance 1e-6: a conf tilted 1.006 rad fails the check,
    # and one on the bound (roll 1.0 rad) passes, whatever the rounding of cos(1.0).
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_function import CostFunction
    from cutamp.joint_limit_cost import trunk_tilt_excess
    from cutamp.rollout import RolloutFunction
    from cutamp.scripts.utils import default_constraint_to_tol
    from cutamp.task_planning.constraints import Motion
    from cutamp.tests.conftest import clone_particles
    skeleton, particles = real_plan
    config = TAMPConfiguration(num_particles=64)
    rollout_fn = RolloutFunction(skeleton, real_world, config)
    particles = clone_particles(particles)
    with torch.no_grad():
        route_a = CostFunction(skeleton, real_world, config)(rollout_fn(particles), particles)
    assert "trunk_tilt" not in route_a[Motion.type]["values"]
    model = _with_envelope(trunk_tilt_max=TILT_MAX)
    monkeypatch.setattr(real_world, "robot_container", dataclasses.replace(real_world.robot_container, model=model))
    with torch.no_grad():
        particles[rollout_fn.conf_params[0]][0, [PITCH, ROLL]] = 0.75
        particles[rollout_fn.conf_params[0]][1, PITCH] = 0.0
        particles[rollout_fn.conf_params[0]][1, ROLL] = 1.0
        rollout = rollout_fn(particles)
        cost_dict = CostFunction(skeleton, real_world, config)(rollout, particles)
    tilt = cost_dict[Motion.type]["values"]["trunk_tilt"]
    assert tilt.shape == rollout["confs"].shape[:2] and float(tilt[0, 0]) > 0.0
    assert torch.equal(tilt, trunk_tilt_excess(rollout["confs"], PITCH, ROLL, math.cos(TILT_MAX)))
    assert default_constraint_to_tol[Motion.type]["trunk_tilt"] == 1e-6
    assert float(tilt[1, 0]) <= 1e-6
    mask = ConstraintChecker(default_constraint_to_tol).get_full_mask(cost_dict)[Motion.type]["trunk_tilt"]
    assert not bool(mask[0, 0]) and bool(mask[1, 0])
