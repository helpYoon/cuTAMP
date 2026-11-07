"""The pinned copy of bfm_wm's T1, its MuJoCo fixture and the real-leg planner model derived from it (design 7.2,
call 18).

``write_mujoco_fixture.py`` runs once on bfm_wm's own MuJoCo and stores its FK, masses, inertias and soft limits
on bfm_wm's t1.xml in ``mujoco_fixture.npz``, and the planner model is checked against it. The model is built here
straight from ``t1_29dof.yml``."""

import hashlib
import importlib.util
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from cutamp.tests.conftest import needs_cuda

T1_29DOF = Path(__file__).resolve().parents[1] / "robots" / "assets" / "t1_29dof"
REPO = T1_29DOF.parents[3]
BFM_WM = Path("/home/yoonwoo/bfm_wm")
OUTPUTS = ("t1_planner.urdf", "t1_29dof.yml", "t1_29dof_spheres.yml")
HEAD = ("AAHead_yaw", "Head_pitch")
WXYZ_TO_XYZW = [1, 2, 3, 0]


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _provenance() -> dict:
    return yaml.safe_load((T1_29DOF / "PROVENANCE.yml").read_text())


def _fixture():
    return np.load(T1_29DOF / "mujoco_fixture.npz")


def _robot_yaml() -> dict:
    return yaml.safe_load((T1_29DOF / "t1_29dof.yml").read_text())


def _build_module():
    """``build_planner_model.py``, imported for its rename maps and gripper names."""
    spec = importlib.util.spec_from_file_location("build_planner_model", T1_29DOF / "build_planner_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kin(tool_frames=None):
    """A CoM-enabled cuRobo Kinematics straight from ``t1_29dof.yml``, its paths resolved against its folder."""
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import DeviceCfg
    cfg = _robot_yaml()
    kinematics = cfg["robot_cfg"]["kinematics"]
    for key in ("urdf_path", "collision_spheres", "asset_root_path"):
        kinematics[key] = str(T1_29DOF / kinematics[key])
    return Kinematics(KinematicsCfg.from_robot_yaml_file(cfg, tool_frames=tool_frames, device_cfg=DeviceCfg()),
                      compute_com=True)


def _joint_state(kin, fx):
    """The fixture's confs in ``kin``'s joint order, ``[n, 1, dof]``: the base joints at 0 put Trunk at the world
    origin, where MuJoCo's free joint at the identity puts it. The head, which the planner locks at 0, is 0 there."""
    from curobo.types import JointState
    names, fx_names, confs = list(kin.joint_names), [str(j) for j in fx["joint_names"]], fx["confs"]
    assert not confs[:, [fx_names.index(h) for h in HEAD]].any()
    q = np.zeros((len(confs), len(names)))
    for i, name in enumerate(names):
        if name in fx_names:
            q[:, i] = confs[:, fx_names.index(name)]
        else:
            assert name.startswith("base_j_"), name
    return JointState.from_position(kin.device_cfg.to_device(q).view(len(confs), 1, -1), joint_names=names)


def _canon(element):
    """An XML element as nested tuples (tag, attributes, text, children), whitespace aside."""
    return element.tag, sorted(element.attrib.items()), (element.text or "").strip(), [_canon(c) for c in element]


def _body_joints(fx) -> list:
    """bfm_wm's 29 joints (T1_JOINT_NAMES, the npz's order too) less the locked head: the planner's 27."""
    return [str(j) for j in fx["joint_names"] if str(j) not in HEAD]


def test_copy_matches_provenance():
    prov = _provenance()
    for rel, entry in prov["copied"].items():
        assert _sha1(T1_29DOF / rel) == entry["sha1"], rel
    meshes = {f"meshes/{p.name}" for p in (T1_29DOF / "meshes").iterdir()}
    assert len(meshes) == 31 and meshes == {rel for rel in prov["copied"] if rel.startswith("meshes/")}
    assert prov["copied"]["urdf/T1_29dof.urdf"]["sha1"].startswith("89ff2a5d")
    assert prov["not_copied"]["t1.xml"]["sha1"].startswith("589d9994")
    assert prov["not_copied"]["t1_newton.mjb"]["sha1"].startswith("4f2c6664")


@pytest.mark.skipif(not BFM_WM.is_dir(), reason="needs bfm_wm's checkout")
def test_copy_and_fixture_match_bfm_wm():
    prov = _provenance()
    entries = [e for section in ("copied", "not_copied") for e in prov[section].values()
               if e["from"].startswith("bfm_wm")]
    assert len(entries) == 35  # the URDF, 31 meshes, the npz, t1.xml and the mjb
    for entry in entries:
        assert _sha1(BFM_WM / entry["path"]) == entry["sha1"], entry["path"]
    assert str(_fixture()["t1_xml_sha1"]) == prov["not_copied"]["t1.xml"]["sha1"]


@needs_cuda  # the builder probes cuRobo's joint order on CUDA
def test_planner_model_regenerates_byte_identical(tmp_path):
    run = subprocess.run([sys.executable, str(T1_29DOF / "build_planner_model.py"), "--out", str(tmp_path)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    for name in OUTPUTS:
        data = (tmp_path / name).read_bytes()
        assert data == (T1_29DOF / name).read_bytes(), name
        assert str(tmp_path).encode() not in data and str(REPO).encode() not in data, name  # no absolute path


def test_base_limits_are_the_legs_reach():
    # The six base joints exist only because cuRobo needs a fixed root, so their limits are the legs' reach over the
    # soft limits (build_planner_model._base_limits), never a bound of their own: x, y and z 1 cm past it, pitch and
    # yaw the whole turn, roll 0.1 rad short of YXZ's singularity, the Trunk on its side. The top admits the
    # straight-knee stand (0.687 m). Then planted soles: confs that bend both legs alike in pitch alone, Waist
    # anywhere, keep both soles on their home poses, so every Trunk pose they reach lies inside the box, bows past
    # horizontal included. yourdfpy's FK finds the base that stands the left sole on its home pose.
    import math
    import yourdfpy
    extra = _robot_yaml()["robot_cfg"]["kinematics"]["extra_links"].values()
    box = {link["joint_name"]: tuple(link["joint_limits"]) for link in extra if "joint_limits" in link}
    cap = math.pi / 2 - 0.1
    assert box == {"base_j_x": (-0.517, 0.57), "base_j_y": (-0.31, 0.31), "base_j_z": (-0.046, 0.772),
                   "base_j_pitch": (-math.pi, math.pi), "base_j_roll": (-cap, cap), "base_j_yaw": (-math.pi, math.pi)}
    assert box["base_j_z"][1] > 0.687
    home = dict(zip(_robot_yaml()["cutamp"]["joint_order"], _robot_yaml()["cutamp"]["home"]))
    assert all(lo <= home[joint] <= hi for joint, (lo, hi) in box.items())

    u = yourdfpy.URDF.load(str(T1_29DOF / "t1_planner.urdf"), load_meshes=False, build_scene_graph=True,
                           build_collision_scene_graph=False, load_collision_meshes=False)
    limits = {j.get("name"): j.find("limit") for j in ET.parse(T1_29DOF / "t1_planner.urdf").getroot().findall("joint")}

    def soft(name):
        lo, hi = float(limits[name].get("lower")), float(limits[name].get("upper"))
        return (lo + hi) / 2 - 0.9 * (hi - lo) / 2, (lo + hi) / 2 + 0.9 * (hi - lo) / 2

    sole = np.eye(4)
    sole[:3, 3] = (0.0101, 0.0, -0.0433)

    def trunk_on_sole(q):
        u.update_cfg({j: q.get(j, 0.0) for j in u.actuated_joint_names})
        return np.linalg.inv(u.get_transform("left_foot_link", "Trunk") @ sole)

    base_home = np.eye(4)
    base_home[:3, 3] = (home["base_j_x"], home["base_j_y"], home["base_j_z"])  # home's base: a pure translation
    left_sole_home = base_home @ np.linalg.inv(trunk_on_sole(home))
    rng = np.random.default_rng(0)
    pitched = ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch")
    ranges = [soft(f"Left_{joint}") for joint in pitched] + [soft("Waist")]
    samples = np.concatenate([np.stack(np.meshgrid(*[np.linspace(lo, hi, 5) for lo, hi in ranges]), -1).reshape(-1, 4),
                              rng.uniform(*np.array(ranges).T, size=(400, 4))])
    bowed = 0
    for hip, knee, ankle, waist in samples:
        q = {"Waist": waist, **{f"{side}_{joint}": v for side in ("Left", "Right")
                                for joint, v in zip(pitched, (hip, knee, ankle))}}
        trunk = left_sole_home @ trunk_on_sole(q)
        pitch, roll, yaw = Rotation.from_matrix(trunk[:3, :3]).as_euler("YXZ")
        base = dict(zip(("base_j_x", "base_j_y", "base_j_z"), trunk[:3, 3]), base_j_pitch=pitch, base_j_roll=roll,
                    base_j_yaw=yaw)
        assert all(box[joint][0] <= v <= box[joint][1] for joint, v in base.items()), (q, base)
        bowed += abs(pitch) > math.pi / 2
    assert bowed > 0, bowed  # deep bows, past horizontal, stay inside the box


def test_planner_urdf_adds_only_world_and_gripper():
    build = _build_module()
    vendor = ET.parse(T1_29DOF / "urdf" / "T1_29dof.urdf").getroot()
    planner = ET.parse(T1_29DOF / "t1_planner.urdf").getroot()
    for link in vendor.findall("link"):  # the rename map, applied to the vendor file
        link.set("name", build.BODY_MAP.get(link.get("name"), link.get("name")))
    for joint in vendor.findall("joint"):
        joint.set("name", build.JOINT_MAP.get(joint.get("name"), joint.get("name")))
        for tag in ("parent", "child"):
            joint.find(tag).set("link", build.BODY_MAP.get(joint.find(tag).get("link"), joint.find(tag).get("link")))
    v_links, v_joints = ({e.get("name"): e for e in vendor.findall(tag)} for tag in ("link", "joint"))
    p_links, p_joints = ({e.get("name"): e for e in planner.findall(tag)} for tag in ("link", "joint"))
    gripper_links = set(build.GRIPPER_LINKS)
    gripper_joints = {n for n, j in p_joints.items() if j.find("child").get("link") in gripper_links}
    assert set(p_links) == set(v_links) | {"world"} | gripper_links
    assert set(p_joints) == set(v_joints) | {"world_joint"} | gripper_joints and len(gripper_joints) == 10
    for name, link in v_links.items():  # origins, axes, limits, inertials, visuals and collisions
        assert _canon(p_links[name]) == _canon(link), name
    for name, joint in v_joints.items():
        assert _canon(p_joints[name]) == _canon(joint), name
    world = p_joints["world_joint"]
    assert world.get("type") == "fixed" and world.find("parent").get("link") == "world"
    assert world.find("child").get("link") == "Trunk"
    origin = world.find("origin")
    assert [float(x) for x in (*origin.get("xyz").split(), *origin.get("rpy").split())] == [0.0] * 6
    assert all(p_links[n].find("inertial") is None for n in gripper_links)  # the gripper weighs nothing
    meshes = [m.get("filename") for m in planner.iter("mesh")]
    assert meshes and all(not Path(f).is_absolute() and (T1_29DOF / f).is_file() for f in meshes), meshes


@needs_cuda
def test_fk_matches_mujoco():
    fx = _fixture()
    bodies = [str(b) for b in fx["body_names"]]
    kin = _kin(tool_frames=bodies)
    assert list(kin.tool_frames) == bodies
    poses = kin.compute_kinematics(_joint_state(kin, fx)).tool_poses
    position = poses.position[:, 0].double().cpu().numpy()
    assert position.shape == fx["body_pos"].shape == (22, 30, 3)
    assert np.abs(position - fx["body_pos"]).max() < 1e-4
    ours = Rotation.from_quat(poses.quaternion[:, 0].double().cpu().numpy().reshape(-1, 4)[:, WXYZ_TO_XYZW])
    theirs = Rotation.from_quat(fx["body_quat"].reshape(-1, 4)[:, WXYZ_TO_XYZW])
    assert (ours.inv() * theirs).magnitude().max() < 1e-4


@needs_cuda
def test_masses_and_com_match_mujoco():
    fx = _fixture()
    bodies = [str(b) for b in fx["body_names"]]
    kin = _kin()
    kp = kin.kinematics_config
    mass = {name: float(kp.link_masses_com[i, 3]) for name, i in kp.link_name_to_idx_map.items()}
    for body, ref in zip(bodies, fx["body_mass"]):
        assert abs(mass[body] - ref) < 1e-6, body
    assert all(m == 0.0 for name, m in mass.items() if name not in bodies)  # the gripper and the extra links
    assert abs(sum(mass.values()) - 34.6325) < 1e-3
    com = kin.compute_kinematics(_joint_state(kin, fx)).robot_com[..., :3].reshape(-1, 3).double().cpu().numpy()
    assert np.abs(com - fx["com"]).max() < 1e-3


@needs_cuda
def test_inertias_match_mujoco():
    fx = _fixture()
    kp = _kin().kinematics_config
    for body, ref in zip(fx["body_names"], fx["body_inertia"]):
        ixx, iyy, izz, ixy, ixz, iyz = kp.link_inertias[kp.link_name_to_idx_map[str(body)], :6].tolist()
        got = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])  # R I R^T about the CoM, link frame
        assert np.abs(got - ref).max() < 1e-6, body


@needs_cuda
def test_limits_are_bfm_wm_soft_limits():
    fx = _fixture()
    fx_names = [str(j) for j in fx["joint_names"]]
    kin = _kin()
    names = list(kin.joint_names)
    cfg = _robot_yaml()
    # cuRobo adds position_limit_clip after putting the cspace in its own joint order, and never reorders the clip
    # itself (CSpaceParams.inplace_reindex), so the YAML must list the cspace in that order already.
    assert names == cfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"] == cfg["cutamp"]["joint_order"]
    lo, hi = kin.get_joint_limits().position.double().cpu().numpy()
    for joint in _body_joints(fx):
        i, k = names.index(joint), fx_names.index(joint)
        assert abs(lo[i] - fx["soft_lower"][k]) < 1e-6 and abs(hi[i] - fx["soft_upper"][k]) < 1e-6, joint
    for link in cfg["robot_cfg"]["kinematics"]["extra_links"].values():  # the base keeps its ranges (clip 0)
        if "joint_limits" in link:
            i = names.index(link["joint_name"])
            assert np.abs([lo[i], hi[i]] - np.array(link["joint_limits"])).max() < 1e-6, link["joint_name"]
    expected = {"Waist": (-0.9108, 0.9108), "Left_Knee_Pitch": (0.1073, 2.0378), "Right_Knee_Pitch": (0.1073, 2.0378)}
    for joint, (want_lo, want_hi) in expected.items():
        assert abs(lo[names.index(joint)] - want_lo) < 1e-3 and abs(hi[names.index(joint)] - want_hi) < 1e-3, joint
    for joint in ("Left_Ankle_Pitch", "Right_Ankle_Pitch"):
        assert abs(hi[names.index(joint)] - 0.289) < 1e-3, joint


@needs_cuda
def test_efforts_are_the_motor_peaks():
    fx = _fixture()
    fx_names = [str(j) for j in fx["joint_names"]]
    tau_max = np.load(T1_29DOF / "t1_actuators.npz")["tau_max"]
    kin = _kin()
    names = list(kin.joint_names)
    effort = kin.get_joint_limits().effort.double().cpu().numpy()
    table = {"Hip_Pitch": 98.8, "Hip_Roll": 68.0, "Hip_Yaw": 68.0, "Knee_Pitch": 130.5, "Ankle_Pitch": 73.1,
             "Ankle_Roll": 17.2, "Waist": 68.0, "Shoulder_Pitch": 38.3, "Shoulder_Roll": 38.3, "Elbow_Pitch": 38.3,
             "Elbow_Yaw": 38.3, "Wrist_Yaw": 38.3, "Wrist_Pitch": 7.0, "Hand_Roll": 7.0}  # design 7.4a, N m
    for joint in _body_joints(fx):
        i = names.index(joint)
        assert effort[0, i] == -effort[1, i], joint
        assert abs(effort[1, i] - tau_max[fx_names.index(joint)]) < 1e-4, joint
        assert abs(effort[1, i] - table[joint.removeprefix("Left_").removeprefix("Right_")]) < 1e-4, joint


@needs_cuda
def test_ankle_pitch_speed_cap():
    fx = _fixture()
    fx_names = [str(j) for j in fx["joint_names"]]
    v_max = np.load(T1_29DOF / "t1_actuators.npz")["v_max"]
    urdf_velocity = {j.get("name"): float(j.find("limit").get("velocity"))
                     for j in ET.parse(T1_29DOF / "t1_planner.urdf").getroot().findall("joint")
                     if j.find("limit") is not None}
    kin = _kin()
    names = list(kin.joint_names)
    velocity = kin.get_joint_limits().velocity.double().cpu().numpy()
    for joint in _body_joints(fx):
        i = names.index(joint)
        assert velocity[0, i] == -velocity[1, i], joint
        if joint.endswith("Ankle_Pitch"):  # cuRobo applies velocity_scale twice (build_planner_model.py)
            assert abs(velocity[1, i] - 7.29) < 1e-4 and abs(velocity[1, i] - v_max[fx_names.index(joint)]) < 1e-4, (
                joint, float(velocity[1, i]))
        else:
            assert abs(velocity[1, i] - urdf_velocity[joint]) < 1e-5, joint
