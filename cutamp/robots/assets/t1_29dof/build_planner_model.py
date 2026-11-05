"""Derive cuTAMP's real-leg planner model from the pinned copy of bfm_wm's T1 (design 7.2, call 18).

Writes three files into ``--out`` (default: this folder). They depend only on the pinned copy and on the simplified
model's gripper, spheres and robot config in ``../t1_description``, and are written byte for byte the same each run:

- ``t1_planner.urdf``: ``urdf/T1_29dof.urdf`` with bfm_wm's CamelCase names, a ``world`` link fixed to Trunk and the
  simplified model's gripper, weightless, at today's mount (call 8). Every vendor element is unchanged, joint limits
  included.
- ``t1_29dof_spheres.yml``: the simplified model's spheres without its legs, Trunk's moved into the vendor frame, plus
  4 per thigh and 4 per shank.
- ``t1_29dof.yml``: the cuRobo robot config, with paths relative to its folder, its six base joints' limits the legs'
  reach (``_base_limits``), and a ``cutamp:`` block (joint order, home, sole box, sole y, home mid-sole) that cuRobo
  never reads.

Run with the tamp env's python: ``SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0 python build_planner_model.py [--out DIR]``.
"""

import argparse
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
import yourdfpy
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
T1_DESC = HERE.parent / "t1_description"
VENDOR_URDF = HERE / "urdf" / "T1_29dof.urdf"
ACTUATORS = HERE / "t1_actuators.npz"
PLANNER_URDF, ROBOT_YML, SPHERES_YML = "t1_planner.urdf", "t1_29dof.yml", "t1_29dof_spheres.yml"
GRIPPER_MESH_DIR = "../t1_description"  # the gripper's meshes, seen from this folder

# bfm_wm utils/build_t1_mjcf.py:25-38, copied: vendor names to bfm_wm's, which equal t1.xml's and T1_JOINT_NAMES.
_ARM = {"shoulder_pitch": "Shoulder_Pitch", "shoulder_roll": "Shoulder_Roll", "elbow_pitch": "Elbow_Pitch",
        "elbow_yaw": "Elbow_Yaw", "wrist_pitch": "Wrist_Pitch", "wrist_yaw": "Wrist_Yaw", "wrist_roll": "Hand_Roll"}
_ARM_LINK = dict(zip(_ARM, ("1", "2", "3", "4", "5", "6", None)))
_LEG = {"hip_pitch": ("Hip_Pitch", "Hip_Pitch"), "hip_roll": ("Hip_Roll", "Hip_Roll"),
        "hip_yaw": ("Hip_Yaw", "Hip_Yaw"), "knee_pitch": ("Knee_Pitch", "Shank"),
        "ankle_pitch": ("Ankle_Pitch", "Ankle_Cross"), "ankle_roll": ("Ankle_Roll", None)}
JOINT_MAP = {"aahead_yaw_joint": "AAHead_yaw", "aahead_pitch_joint": "Head_pitch", "waist_yaw_joint": "Waist"}
BODY_MAP = {"trunk": "Trunk", "aahead_yaw_link": "H1", "aahead_pitch_link": "H2", "waist_yaw_link": "Waist"}
for _side, _S in (("left", "Left"), ("right", "Right")):
    for _k, _v in _ARM.items():
        JOINT_MAP[f"{_side}_{_k}_joint"] = f"{_S}_{_v}"
        BODY_MAP[f"{_side}_{_k}_link"] = f"A{_S[0]}{_ARM_LINK[_k]}" if _ARM_LINK[_k] else f"{_side}_hand_link"
    for _k, (_j, _b) in _LEG.items():
        JOINT_MAP[f"{_side}_{_k}_joint"] = f"{_S}_{_j}"
        BODY_MAP[f"{_side}_{_k}_link"] = f"{_b}_{_S}" if _b else f"{_side}_foot_link"

# bfm_wm robot/t1.py:49, copied: the MJCF's joint order, and the order of t1_actuators.npz's arrays.
T1_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
SIDES = (("left", "Left"), ("right", "Right"))
HEAD = ("AAHead_yaw", "Head_pitch")  # locked at 0; H2 stays in the tree as a collision link, so its mass counts
FINGERS = tuple(f"{s}_Link{n}" for s, _ in SIDES for n in ("1", "11", "2", "22"))  # locked open
GRIPPER_LINKS = tuple(f"{s}_{n}" for s, _ in SIDES for n in ("base_link", "Link1", "Link2", "Link11", "Link22"))
BODY_JOINTS = [j for j in T1_JOINT_NAMES if j not in HEAD]

# The floating base: six massless single-axis joints from world to Trunk, x, y, z, then YXZ Euler: pitch about the
# world's y, roll, then yaw about the Trunk's z (floor_wrench.base_wrench assumes this order). YXZ is singular at
# |roll| = 90 deg, the Trunk lying on its side, which planted soles cannot reach. Stage 2's ZYX was singular at a
# 90 deg bow, which floor picks reach. The joints exist only because cuRobo needs a fixed root, so their limits are
# the legs' reach (_base_limits), never a bound of their own.
BASE = (("x", "X_PRISM"), ("y", "Y_PRISM"), ("z", "Z_PRISM"), ("pitch", "Y_ROT"), ("roll", "X_ROT"), ("yaw", "Z_ROT"))
BASE_JOINTS = [f"base_j_{n}" for n, _ in BASE]
# _base_limits samples the Trunk's pose over each leg: Trunk to the sole, Waist included, since it turns the pelvis both
# hips hang from. LEG_GRID values per joint span its soft range, ends included; the REFINE_STARTS best samples of each
# x, y and z extreme start a bounded L-BFGS-B. x, y and z then get BASE_MARGIN on each side, rounded out to the mm.
# Roll stops ROLL_MARGIN short of the singularity at |roll| = pi/2, where base_wrench divides by cos(roll) and the
# pitch and yaw axes line up: there 1/cos(roll) is 10.0.
LEG_CHAIN = ("Waist", "Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")
LEG_GRID = 7
REFINE_STARTS = 8
BASE_MARGIN = 0.01  # m
ROLL_MARGIN = 0.1  # rad

SOFT_LIMIT_FACTOR = 0.9  # bfm_wm robot/t1.py:118, T1_CFG.soft_joint_pos_limit_factor (call 17)
ANKLE_PITCH = ("Left_Ankle_Pitch", "Right_Ankle_Pitch")  # capped at the motor's top speed, npz v_max (7.4a)
# Trajopt's internal timing. Trajopt times each segment against these and its success test checks the torques at
# that timing, so they shape each path and its time profile. The balance gate keeps that profile and sets only each
# segment's uniform time scale, from the robot's speed limits, efforts and the ZMP (motion_solver._pace), so their
# shape reaches every exported motion. They are not robot facts: the acceleration and jerk caps are
# the simplified model's (t1_planar_base.yml), and the base's velocity limit is cuRobo's default for extra links
# (link_params.py:32), written out. With non-binding caps in their place (1e4, 1e6, 1e3) both seed-0 runs of P2's
# check set failed every candidate's first trajopt on its torque check, so trajopt keeps these.
TRAJOPT_MAX_ACCELERATION = 15.0  # rad/s^2 (m/s^2 on the base's x, y, z)
TRAJOPT_MAX_JERK = 100.0  # rad/s^3
BASE_VELOCITY_LIMIT = 2.0  # rad/s, m/s: each of the six virtual base joints

# Design 7.2's sole frame and sole box, which _check_soles finds at the foot collision box's bottom centre and its x
# and y, to the 0.1 mm the design gives them in.
SOLE_OFFSET = (0.0101, 0.0, -0.0433)
SOLE_BOX = (0.2249, 0.100)
SOLE_FRAMES = tuple(f"{s}_sole_link" for s, _ in SIDES)
TOOL_FRAMES = ["left_base_link", "right_base_link", *SOLE_FRAMES, "stance"]

OLD_LEG_LINKS = ("mobile_base_link", "shank_link", "thigh_link")  # the simplified model's legs
TRUNK_SPHERE_SHIFT_X = 0.0625  # t1.urdf's Trunk frame to the vendor's (design 7.2)
LEG_SPHERE_RADIUS = {"Hip_Yaw": 0.055, "Shank": 0.045}  # stage 2 (REPORT.md section 5)

HEADER = "# Generated by build_planner_model.py from the pinned copy of bfm_wm's T1 (PROVENANCE.yml). Do not edit.\n"
ROBOT_HEADER = HEADER + (
    "# cuRobo robot config of the real-leg planner model (design 7.2); paths are relative to this folder. The cspace\n"
    "# lists cuRobo's own joint order for these tool frames, because cuRobo adds position_limit_clip "
    "(the soft limits,\n"
    "# call 17) after reordering the cspace and never reorders the clip itself: tool frames that change "
    "that order put\n"
    "# the soft limits on the wrong joints. cuRobo applies velocity_scale twice, so the Ankle_Pitch scale "
    "is the square\n"
    "# root of the cap over the URDF velocity: read velocity limits from cuRobo, not from this file. "
    "max_acceleration,\n"
    "# max_jerk and the base joints' velocity limits are trajopt's internal timing: they shape each path and its "
    "time\n"
    "# profile, and the balance gate sets only each segment's uniform time scale, from speed, effort and the ZMP.\n"
    "# cuRobo never reads the cutamp: block.\n")


def _floats(text: str) -> list:
    return [float(x) for x in text.split()]


def _build_urdf() -> tuple[ET.Element, dict]:
    """The planner URDF's root, and the body joints' elements by name."""
    root = ET.parse(VENDOR_URDF).getroot()
    for link in root.findall("link"):
        link.set("name", BODY_MAP.get(link.get("name"), link.get("name")))
    joints = {}
    for joint in root.findall("joint"):
        joint.set("name", JOINT_MAP.get(joint.get("name"), joint.get("name")))
        for tag in ("parent", "child"):
            element = joint.find(tag)
            element.set("link", BODY_MAP.get(element.get("link"), element.get("link")))
        if joint.get("name") in T1_JOINT_NAMES:
            joints[joint.get("name")] = joint
    assert sorted(joints) == sorted(T1_JOINT_NAMES), sorted(set(joints) ^ set(T1_JOINT_NAMES))
    # cuRobo cannot float the file's root (parser_base.py:50), so the base joints hang Trunk from a world link.
    world_joint = ET.Element("joint", name="world_joint", type="fixed")
    ET.SubElement(world_joint, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(world_joint, "parent", link="world")
    ET.SubElement(world_joint, "child", link="Trunk")
    root.insert(0, ET.Element("link", name="world"))
    root.insert(1, world_joint)
    simplified = ET.parse(T1_DESC / "t1_simplified.urdf").getroot()
    for link in simplified.findall("link"):
        if link.get("name") in GRIPPER_LINKS:
            for inertial in link.findall("inertial"):
                link.remove(inertial)
            for mesh in link.iter("mesh"):
                mesh.set("filename", f"{GRIPPER_MESH_DIR}/{mesh.get('filename')}")
            root.append(link)
    for joint in simplified.findall("joint"):
        if joint.find("child").get("link") in GRIPPER_LINKS:
            root.append(joint)
    return root, joints


def _check_soles(root: ET.Element) -> None:
    """Assert that each foot's collision box puts its bottom centre at SOLE_OFFSET and its x and y at SOLE_BOX."""
    for side, _ in SIDES:
        (link,) = [e for e in root.findall("link") if e.get("name") == f"{side}_foot_link"]
        (collision,) = link.findall("collision")
        size = np.array(_floats(collision.find("geometry/box").get("size")))
        assert _floats(collision.find("origin").get("rpy")) == [0.0, 0.0, 0.0]
        bottom = np.array(_floats(collision.find("origin").get("xyz"))) - [0.0, 0.0, size[2] / 2]
        assert np.round(bottom, 4).tolist() == list(SOLE_OFFSET), (side, bottom)
        assert np.round(size[:2], 4).tolist() == list(SOLE_BOX), (side, size)


def _build_spheres(joints: dict) -> dict:
    """The simplified model's spheres without its legs, Trunk's moved into the vendor frame, plus 4 on each Hip_Yaw
    link (the hip-pitch centre to the knee) and 4 on each Shank (the knee to the ankle)."""
    spheres = yaml.safe_load((T1_DESC / "t1_spheres.yml").read_text())["collision_spheres"]
    for key in OLD_LEG_LINKS:
        spheres.pop(key)
    for sphere in spheres["Trunk"]:
        sphere["center"][0] += TRUNK_SPHERE_SHIFT_X
    origin = {name: np.array(_floats(joint.find("origin").get("xyz"))) for name, joint in joints.items()}
    for _, side in SIDES:
        hip = -(origin[f"{side}_Hip_Roll"] + origin[f"{side}_Hip_Yaw"])  # the hip-pitch centre in Hip_Yaw, roll 0
        knee, ankle = origin[f"{side}_Knee_Pitch"], origin[f"{side}_Ankle_Pitch"]
        for key, a, b in ((f"Hip_Yaw_{side}", hip, knee), (f"Shank_{side}", np.zeros(3), ankle)):
            radius = LEG_SPHERE_RADIUS[key.removesuffix(f"_{side}")]
            spheres[key] = [{"center": [float(c) + 0.0 for c in a + t * (b - a)], "radius": radius}  # + 0.0: no -0.0
                            for t in np.linspace(0.0, 1.0, 4)]
    return spheres


def _home(urdf: Path, default_q: dict) -> tuple[dict, list, float]:
    """Home by joint name: the body at default_q, the base at (x0, y0, z0, 0, 0, 0) with the midpoint of the two
    ankle-pitch axes (the Ankle_Cross origins) over the world origin and both sole bottoms at z = 0. Also the home
    mid-sole [x, y, 0] and half the distance between the sole centres."""
    u = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True, build_collision_scene_graph=False,
                           load_collision_meshes=False)
    u.update_cfg({j: default_q.get(j, 0.0) for j in u.actuated_joint_names})
    ankles = np.stack([u.get_transform(f"Ankle_Cross_{side}", "Trunk")[:3, 3] for _, side in SIDES])
    feet = [u.get_transform(f"{side}_foot_link", "Trunk") for side, _ in SIDES]
    soles = np.stack([(foot @ [*SOLE_OFFSET, 1.0])[:3] for foot in feet])
    assert abs(soles[0, 2] - soles[1, 2]) < 1e-12, soles  # both soles level
    assert max(np.abs(foot[:3, :3] - np.eye(3)).max() for foot in feet) < 1e-12  # and flat
    x0, y0 = 0.0 - ankles[:, 0].mean(), 0.0 - ankles[:, 1].mean()
    z0 = 0.0 - soles[:, 2].mean()
    mid_sole = soles.mean(axis=0) + [x0, y0, z0]
    assert abs(mid_sole[2]) < 1e-12, mid_sole
    home = dict(zip(BASE_JOINTS, (float(x0), float(y0), float(z0), 0.0, 0.0, 0.0)))
    home.update({j: default_q[j] for j in BODY_JOINTS})
    sole_y = float(np.linalg.norm(soles[0] - soles[1]) / 2)
    return home, [float(mid_sole[0]) + 0.0, float(mid_sole[1]) + 0.0, 0.0], sole_y


def _soft_limits(joint: ET.Element) -> tuple[float, float]:
    """bfm_wm's soft limits of ``joint`` (call 17): its URDF range's midpoint ± SOFT_LIMIT_FACTOR × half the range."""
    limit = joint.find("limit")
    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
    mid, half = (lower + upper) / 2, (upper - lower) / 2
    return mid - SOFT_LIMIT_FACTOR * half, mid + SOFT_LIMIT_FACTOR * half


def _rotations(axis: list, angle: np.ndarray) -> np.ndarray:
    """``[N, 3, 3]``: rotations by ``angle`` ``[N]`` about the unit ``axis`` (Rodrigues)."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def _trunk_on_sole(joints: dict, leg: tuple, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Trunk's position ``[N, 3]`` and rotation ``[N, 3, 3]`` in the frame of the sole at the end of ``leg``'s
    joints, at their angles ``q`` ``[N, len(leg)]``."""
    rot, pos = np.broadcast_to(np.eye(3), (len(q), 3, 3)), np.zeros((len(q), 3))
    for name, angle in zip(leg, q.T):
        origin = joints[name].find("origin")
        roll, pitch, yaw = _floats(origin.get("rpy", "0 0 0"))
        fixed = _rotations([0.0, 0.0, 1.0], np.array([yaw])) @ _rotations([0.0, 1.0, 0.0], np.array([pitch])) @ (
            _rotations([1.0, 0.0, 0.0], np.array([roll])))  # URDF rpy: about the fixed x, then y, then z
        pos = pos + rot @ np.array(_floats(origin.get("xyz")))
        rot = rot @ fixed @ _rotations(_floats(joints[name].find("axis").get("xyz")), angle)
    pos = pos + rot @ np.array(SOLE_OFFSET)
    rot_t = rot.transpose(0, 2, 1)
    return -(rot_t @ pos[..., None])[..., 0], rot_t


def _leg_reach(joints: dict, home: dict, side: str, Side: str) -> dict:
    """``{coordinate: (min, max)}``: the extremes of the Trunk's base coordinates (x, y, z and YXZ's pitch, roll and
    yaw) over the ``side`` leg's soft limits, with its sole on its home pose. x, y and z are refined; the angles are
    the grid's."""
    leg = ("Waist", *(f"{Side}_{joint}" for joint in LEG_CHAIN[1:]))
    chain = [joints[name] for name in leg]
    assert chain[0].find("parent").get("link") == "Trunk" and chain[-1].find("child").get("link") == f"{side}_foot_link"
    assert all(a.find("child").get("link") == b.find("parent").get("link") for a, b in zip(chain, chain[1:]))
    lower, upper = np.array([_soft_limits(joint) for joint in chain]).T
    grid = np.stack(np.meshgrid(*np.linspace(lower, upper, LEG_GRID, axis=1), indexing="ij"), -1).reshape(-1, len(leg))
    trunk, rot = _trunk_on_sole(joints, leg, np.array([[home[name] for name in leg]]))
    assert np.abs(rot[0] - np.eye(3)).max() < 1e-12  # the sole stands flat under the unrotated home base
    sole = np.array([home[f"base_j_{n}"] for n in "xyz"]) - trunk[0]  # the sole's home position in the world

    def reach(q):
        p, r = _trunk_on_sole(joints, leg, np.atleast_2d(q))
        return p + sole, r

    pos, rot = reach(grid)
    extremes = {}
    for k, n in enumerate("xyz"):
        ends = []
        for sign in (-1.0, 1.0):
            starts = grid[np.argsort(-sign * pos[:, k])[:REFINE_STARTS]]
            best = min(minimize(lambda q: -sign * reach(q)[0][0, k], start, method="L-BFGS-B",
                                bounds=list(zip(lower, upper))).fun for start in starts)
            ends.append(-sign * float(best))
        extremes[n] = tuple(ends)
    pitch, yaw = np.arctan2(rot[:, 0, 2], rot[:, 2, 2]), np.arctan2(rot[:, 1, 0], rot[:, 1, 1])
    roll = np.arctan2(-rot[:, 1, 2], np.hypot(rot[:, 0, 2], rot[:, 2, 2]))
    extremes.update({n: (float(a.min()), float(a.max())) for n, a in (("yaw", yaw), ("pitch", pitch), ("roll", roll))})
    return extremes


def _base_limits(joints: dict, home: dict) -> dict:
    """Each base joint's limits, by name, from the legs' reach (``_leg_reach``). Each leg's reach, its sole on its home
    pose, holds every Trunk pose that both planted soles allow, so the overlap of the two never binds before the legs
    do: x, y and z take it, BASE_MARGIN wider.

    One leg alone turns the Trunk every way, so pitch and yaw end at the chart's edges, and roll ROLL_MARGIN short of
    the singularity: a lean onto the Trunk's side, which both planted soles keep it far from."""
    reach = [_leg_reach(joints, home, side, Side) for side, Side in SIDES]
    lo = {n: max(r[n][0] for r in reach) for n in reach[0]}
    hi = {n: min(r[n][1] for r in reach) for n in reach[0]}
    limits = {n: (math.floor((lo[n] - BASE_MARGIN) * 1000) / 1000, math.ceil((hi[n] + BASE_MARGIN) * 1000) / 1000)
              for n in "xyz"}
    cap = math.pi / 2 - ROLL_MARGIN
    assert lo["roll"] < -cap and hi["roll"] > cap, (lo["roll"], hi["roll"])
    for n in ("pitch", "yaw"):
        assert lo[n] < 0.01 - math.pi and hi[n] > math.pi - 0.01, (n, lo[n], hi[n])
    limits.update(pitch=(-math.pi, math.pi), roll=(-cap, cap), yaw=(-math.pi, math.pi))
    return limits


def _extra_links(mid_sole: list, base_limits: dict) -> dict:
    """The YAML's extra links, every one massless: the base chain, the sole frames, the attach links and stance."""
    identity = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    extra, parent = {}, "world"
    for i, (n, joint_type) in enumerate(BASE):
        link = {"parent_link_name": parent}
        if i == len(BASE) - 1:
            link["child_link_name"] = "Trunk"
        link.update(link_name=f"base_link_{n}", joint_name=f"base_j_{n}", joint_type=joint_type,
                    fixed_transform=list(identity), joint_limits=list(base_limits[n]),
                    joint_velocity_limits=[-BASE_VELOCITY_LIMIT, BASE_VELOCITY_LIMIT], link_mass=0.0)
        extra[f"base_link_{n}"] = link
        parent = f"base_link_{n}"
    for (side, _), sole in zip(SIDES, SOLE_FRAMES):
        extra[sole] = {"parent_link_name": f"{side}_foot_link", "link_name": sole, "joint_name": f"{sole}_joint",
                       "joint_type": "FIXED", "fixed_transform": [*SOLE_OFFSET, 1.0, 0.0, 0.0, 0.0], "link_mass": 0.0}
    for side, _ in SIDES:
        extra[f"attached_object_{side}"] = {
            "parent_link_name": f"{side}_base_link", "link_name": f"attached_object_{side}",
            "joint_name": f"attach_joint_{side}", "joint_type": "FIXED", "fixed_transform": list(identity),
            "link_mass": 0.0}
    extra["stance"] = {"parent_link_name": "world", "link_name": "stance", "joint_name": "stance_joint",
                       "joint_type": "FIXED", "fixed_transform": [*mid_sole, 1.0, 0.0, 0.0, 0.0], "link_mass": 0.0}
    return extra


def _self_collision_ignore(today: dict) -> dict:
    """The base chain; the simplified model's pairs without its legs; each leg chain, and Trunk and Waist against the
    hips and thighs. Left leg against right leg, and legs against arms, stay checked (design 7.2)."""
    chain = [f"base_link_{n}" for n, _ in BASE] + ["Trunk"]
    ignore = {link: chain[max(i - 1, 0):i] + chain[i + 1:i + 2] for i, link in enumerate(chain[:-1])}
    ignore.update({k: [x for x in v if x not in OLD_LEG_LINKS]
                   for k, v in today["self_collision_ignore"].items() if k not in OLD_LEG_LINKS})
    for side, Side in SIDES:
        hips = [f"Hip_Pitch_{Side}", f"Hip_Roll_{Side}", f"Hip_Yaw_{Side}"]
        for body in ("Waist", "Trunk"):
            ignore[body] = ignore[body] + hips
        ignore[f"Hip_Yaw_{Side}"] = ["Waist", "Trunk", f"Shank_{Side}"]
        ignore[f"Shank_{Side}"] = [f"Hip_Yaw_{Side}", f"Ankle_Cross_{Side}", f"{side}_foot_link"]
    return ignore


def _kinematics(mid_sole: list, base_limits: dict) -> dict:
    """The robot config's kinematics, except its cspace."""
    today = yaml.safe_load((T1_DESC / "t1_planar_base.yml").read_text())["robot_cfg"]["kinematics"]
    legs = [f"{key}_{Side}" for _, Side in SIDES for key in LEG_SPHERE_RADIUS]
    return {
        "format_version": 2.0,
        "asset_root_path": "",  # cuRobo then takes the URDF's folder, this one, which holds meshes/
        "urdf_path": PLANNER_URDF,
        "base_link": "world",
        "tool_frames": list(TOOL_FRAMES),
        "grasp_contact_link_names": today["grasp_contact_link_names"],
        "collision_link_names": [k for k in today["collision_link_names"] if k not in OLD_LEG_LINKS] + legs,
        "collision_spheres": SPHERES_YML,
        "collision_sphere_buffer": {},
        "extra_collision_spheres": today["extra_collision_spheres"],
        "extra_links": _extra_links(mid_sole, base_limits),
        "lock_joints": {j: 0.0 for j in (*HEAD, *FINGERS)},
        "use_global_cumul": True,
        "mesh_link_names": None,
        "self_collision_buffer": {},
        "self_collision_ignore": _self_collision_ignore(today),
    }


def _curobo_joint_order(kinematics: dict, out: Path) -> list:
    """cuRobo's joint order for ``kinematics``: KinematicsLoader._build_chain's walk from the first tool frame
    (kinematics_loader.py:215-260), which locking joints only shortens (lines 820-826). The loader locks joints only
    on CUDA, so this builds the model on the CPU without lock_joints or cspace, and drops the locked joints."""
    import torch
    from curobo.kinematics import KinematicsCfg
    from curobo.types import DeviceCfg
    probe = copy.deepcopy(kinematics)
    locked = probe.pop("lock_joints")
    for key in ("urdf_path", "collision_spheres", "asset_root_path"):
        probe[key] = str(out / probe[key])
    cfg = KinematicsCfg.from_robot_yaml_file({"robot_cfg": {"kinematics": probe}},
                                             device_cfg=DeviceCfg(device=torch.device("cpu")))
    return [j for j in cfg.kinematics_config.joint_names if j not in locked]


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data) -> bool:  # write a list used twice (the joint order) out in full both times
        return True


def _dump(data: dict) -> str:
    """YAML in the dict's own key order; PyYAML writes each float as its repr (plus ".0" in an exponent)."""
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, default_flow_style=None, width=120)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE, help="output folder (default: this script's)")
    out = parser.parse_args(argv).out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    root, joints = _build_urdf()
    _check_soles(root)
    ET.indent(root, space="  ")
    (out / PLANNER_URDF).write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n")
    (out / SPHERES_YML).write_text(HEADER + _dump({"collision_spheres": _build_spheres(joints)}))

    actuators = np.load(ACTUATORS)
    default_q = dict(zip(T1_JOINT_NAMES, actuators["default_q"].astype(np.float64).tolist()))
    home, mid_sole, sole_y = _home(out / PLANNER_URDF, default_q)
    kinematics = _kinematics(mid_sole, _base_limits(joints, home))
    order = _curobo_joint_order(kinematics, out)
    assert order[:6] == BASE_JOINTS and sorted(order) == sorted(BASE_JOINTS + BODY_JOINTS), order

    clip = dict.fromkeys(BASE_JOINTS, 0.0)  # the soft limits: midpoint ± SOFT_LIMIT_FACTOR × half-range
    for j in BODY_JOINTS:
        limit = joints[j].find("limit")
        half = (float(limit.get("upper")) - float(limit.get("lower"))) / 2
        clip[j] = half - SOFT_LIMIT_FACTOR * half
    # cuRobo multiplies the velocity limits by velocity_scale twice: in KinematicsLoader._update_joint_limits
    # (kinematics_loader.py:1121-1122) and again in KinematicsParams.__post_init__ (kinematics_params.py:162-163,
    # CSpaceParams.scale_joint_limits). So the scale that caps ankle pitch at v_max is sqrt(v_max / URDF velocity);
    # test_ankle_pitch_speed_cap checks cuRobo's resulting limit, and fails if the fork ever applies the scale once.
    v_max = dict(zip(T1_JOINT_NAMES, actuators["v_max"].astype(np.float64).tolist()))
    velocity_scale = {j: math.sqrt(v_max[j] / float(joints[j].find("limit").get("velocity"))) if j in ANKLE_PITCH
                      else 1.0 for j in order}
    kinematics["cspace"] = {
        "joint_names": order,
        "default_joint_position": [home[j] for j in order],
        "cspace_distance_weight": [1.0] * len(order),
        "null_space_weight": [1.0] * len(order),
        "acceleration_scale": [1.0] * len(order),
        "velocity_scale": [velocity_scale[j] for j in order],
        "position_limit_clip": [clip[j] for j in order],
        "max_acceleration": TRAJOPT_MAX_ACCELERATION,
        "max_jerk": TRAJOPT_MAX_JERK,
    }
    config = {
        "robot_cfg": {"kinematics": kinematics},
        "cutamp": {"joint_order": order, "home": [home[j] for j in order], "sole_box": list(SOLE_BOX),
                   "sole_y": sole_y, "home_mid_sole": mid_sole},
    }
    (out / ROBOT_YML).write_text(ROBOT_HEADER + _dump(config))


if __name__ == "__main__":
    main()
