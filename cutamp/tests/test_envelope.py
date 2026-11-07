"""Route A's envelope (design 7.10, call 22): the pinned file and the cut of the real legs' limits to it (each joint
range at either end of its soft range on its own, never widening one; route A's caps both ankle pitches at home)."""

import dataclasses
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from cutamp.robots.envelope import Envelope, load_envelope
from cutamp.robots.t1 import REAL_LEGS
from cutamp.tests.conftest import needs_cuda

ENVELOPES = Path(__file__).resolve().parents[1] / "robots" / "assets" / "envelopes"
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
        joint_ranges={"Waist": (-0.911, 0.911), "Left_Ankle_Pitch": (-0.809, -0.2), "Right_Ankle_Pitch": (-0.809, -0.2)},
        speed_caps=None,
        acceleration_caps=None,
        trunk_speed_caps=None,
        payload_kg={"range": [0.0, 0.5], "tail_max": 1.0},
        stance_range={"xy_m": 2.0, "yaw_rad": math.pi},
        hold_s={"start_blend": 0.5, "grasp": 0.5, "end": 1.0},
    )
    assert REAL_LEGS.envelope == envelope and REAL_LEGS.envelope.name == "route_a_start"
    # The ankle pitches' upper end is home (the user, 2026-11-05: no shin pitches back past the home stance).
    for ankle in ("Left_Ankle_Pitch", "Right_Ankle_Pitch"):
        assert envelope.joint_ranges[ankle][1] == pytest.approx(REAL_LEGS.home[REAL_LEGS.joint_names.index(ankle)],
                                                                abs=1e-7)


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


@needs_cuda
def test_envelope_cuts_never_widen():
    from cutamp.robots.t1 import t1_curobo_cfg
    real = _limits(REAL_LEGS)
    # The base keeps the legs' reach, as the robot YAML gives it; Waist keeps its soft limit, which lies inside route
    # A's +-0.911.
    import yaml
    extra = yaml.safe_load(REAL_LEGS.cfg_path.read_text())["robot_cfg"]["kinematics"]["extra_links"].values()
    base = {link["joint_name"]: tuple(link["joint_limits"]) for link in extra if "joint_limits" in link}
    assert sorted(base) == sorted(n for n in REAL_LEGS.joint_names if n.startswith("base_j_"))
    for joint, limits in base.items():
        np.testing.assert_allclose(real[joint], limits, atol=1e-6, err_msg=joint)
    np.testing.assert_allclose(real["base_j_z"], BASE_Z, atol=1e-6)
    np.testing.assert_allclose(real["Waist"], (-0.9108, 0.9108), atol=1e-6)
    # Both ankle pitches keep their soft lower limit and stop at home, where the robot stands: home is on the bound.
    for ankle in ("Left_Ankle_Pitch", "Right_Ankle_Pitch"):
        home = REAL_LEGS.home[REAL_LEGS.joint_names.index(ankle)]
        np.testing.assert_allclose(real[ankle], (-0.809, -0.2), atol=1e-6, err_msg=ankle)
        assert real[ankle][0] < home <= real[ankle][1]
    for waist, want in (((-0.5, 0.5), (-0.5, 0.5)), ((-2.0, 2.0), (-0.9108, 0.9108))):
        got = _limits(_with_envelope(joint_ranges={**REAL_LEGS.envelope.joint_ranges, "Waist": waist}))
        np.testing.assert_allclose(got["Waist"], want, atol=1e-6, err_msg=str(waist))
        assert {n: v for n, v in got.items() if n != "Waist"} == {n: v for n, v in real.items() if n != "Waist"}
    with pytest.raises(ValueError, match="Waist"):
        t1_curobo_cfg(_with_envelope(joint_ranges={**REAL_LEGS.envelope.joint_ranges, "Waist": (2.0, 3.0)}))


@needs_cuda
def test_envelope_cut_moves_only_its_joint():
    # The cut zeroes position_limit_clip at the joint's index in cspace.joint_names, and cuRobo applies the clip
    # without reindexing it. A knee, far from Waist in the cspace order and with the other knee's clip, cut 0.1 rad
    # inside its soft limit at both ends moves cuRobo's limits on that knee alone.
    real = _limits(REAL_LEGS)
    lo, hi = real["Left_Knee_Pitch"]
    got = _limits(_with_envelope(joint_ranges={**REAL_LEGS.envelope.joint_ranges, "Left_Knee_Pitch": (lo + 0.1, hi - 0.1)}))
    np.testing.assert_allclose(got["Left_Knee_Pitch"], (lo + 0.1, hi - 0.1), atol=1e-6)
    assert {n: v for n, v in got.items() if n != "Left_Knee_Pitch"} == {
        n: v for n, v in real.items() if n != "Left_Knee_Pitch"}


@needs_cuda
def test_envelope_cut_is_one_sided_and_leaves_the_planner_urdf():
    # Each end is cut on its own: an off-centre Waist range keeps both its ends, where a clip off both ends would
    # plan in +-0.5. The cut goes into a copy of the URDF, which urdf_path then names; t1_planner.urdf is untouched,
    # and an envelope that cuts nothing plans on it.
    from cutamp.robots.t1 import T1_29DOF_DIR, t1_curobo_cfg
    planner_urdf = T1_29DOF_DIR / "t1_planner.urdf"
    before = hashlib.sha1(planner_urdf.read_bytes()).hexdigest()
    real = _limits(REAL_LEGS)
    got = _limits(_with_envelope(joint_ranges={**REAL_LEGS.envelope.joint_ranges, "Waist": (-0.5, 0.58)}))
    np.testing.assert_allclose(got["Waist"], (-0.5, 0.58), atol=1e-6)
    assert {n: v for n, v in got.items() if n != "Waist"} == {n: v for n, v in real.items() if n != "Waist"}
    assert t1_curobo_cfg(REAL_LEGS)["robot_cfg"]["kinematics"]["urdf_path"] != str(planner_urdf)
    uncut = _with_envelope(joint_ranges={"Waist": (-2.0, 2.0)})
    assert t1_curobo_cfg(uncut)["robot_cfg"]["kinematics"]["urdf_path"] == str(planner_urdf)
    np.testing.assert_allclose(_limits(uncut)["Left_Ankle_Pitch"], (-0.809, 0.289), atol=1e-6)
    assert hashlib.sha1(planner_urdf.read_bytes()).hexdigest() == before
