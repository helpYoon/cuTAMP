"""The balance model's mass distribution: t1_simplified's link inertials, against yourdfpy.

yourdfpy reads each inertial as the URDF defines it (CoM = the inertial origin's translation in
the link frame), so it is the reference the planner's CoM has to match."""

from pathlib import Path

import numpy as np
import pytest
import yourdfpy
from scipy.spatial.transform import Rotation

from cutamp.robots.t1 import SIMPLIFIED as SIMPLIFIED_MODEL
from cutamp.robots.t1 import JOINT_NAMES_FULL, T1_ASSETS_DIR, get_t1_bodies, t1_curobo_cfg, t1_home
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda

SIMPLIFIED = T1_ASSETS_DIR / "t1_simplified.urdf"
HOME = dict(zip(JOINT_NAMES_FULL, t1_home))
BASE_JOINTS = ("base_j_x", "base_j_y", "base_j_yaw")  # the planar base, from cuRobo's YAML, not the URDF


def _links_parser(tmp_path):
    """cuRobo's parser on a URDF of fixed links: CoM at (0.10, 0.02, -0.03) unrotated and rotated,
    an inertial with no <origin>, one with mass 0, one with an all-zero tensor, and a link with no
    inertial. The tensor is not isotropic, so turning it shows."""
    from curobo.robot_parser import UrdfRobotParser
    inertia = '<inertia ixx="1e-3" ixy="1e-4" ixz="-2e-4" iyy="2e-3" iyz="3e-4" izz="3e-3"/>'
    inertials = {
        "unrot": f'<origin xyz="0.10 0.02 -0.03" rpy="0 0 0"/><mass value="1.0"/>{inertia}',
        "rot": f'<origin xyz="0.10 0.02 -0.03" rpy="0.3 -0.5 1.0"/><mass value="2.0"/>{inertia}',
        "no_origin": f'<mass value="0.5"/>{inertia}',
        "zero": f'<mass value="0.0"/>{inertia}',
        "zero_tensor": '<origin xyz="0 0 0" rpy="0 0 0"/><mass value="2.0"/>'
                       '<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>',
    }
    links = "".join(f'<link name="{n}"><inertial>{i}</inertial></link>' for n, i in inertials.items())
    joints = "".join(f'<joint name="j_{n}" type="fixed"><parent link="base"/><child link="{n}"/></joint>'
                     for n in [*inertials, "bare"])
    path = tmp_path / "links.urdf"
    path.write_text(f'<robot name="links"><link name="base"/><link name="bare"/>{links}{joints}</robot>')
    return UrdfRobotParser(str(path))


def _yourdfpy_com(cfg: dict, urdf: Path) -> np.ndarray:
    """Whole-robot CoM in the URDF root frame, which is ``mobile_base_link`` with the planar base at
    0. Joints missing from ``cfg`` (the planner's locked head and grippers) are 0."""
    u = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True,
                           build_collision_scene_graph=False, load_collision_meshes=False)
    u.update_cfg({j: cfg.get(j, 0.0) for j in u.actuated_joint_names})
    s, m = np.zeros(3), 0.0
    for name, link in u.link_map.items():
        if link.inertial is None or not link.inertial.mass:
            continue
        T = u.get_transform(name)
        s += link.inertial.mass * (T[:3, :3] @ link.inertial.origin[:3, 3] + T[:3, 3])
        m += link.inertial.mass
    return s / m


def test_parsed_link_com_is_the_inertial_xyz(tmp_path):
    # The CoM is the inertial origin's translation; the rpy only orients the inertia tensor.
    p = _links_parser(tmp_path)
    for name in ("unrot", "rot"):
        np.testing.assert_allclose(p.get_link_parameters(name).link_com, [0.10, 0.02, -0.03], atol=1e-7)
    np.testing.assert_allclose(p.get_link_parameters("no_origin").link_com, [0, 0, 0], atol=1e-7)


def test_massless_links_weigh_nothing(tmp_path):
    p = _links_parser(tmp_path)
    assert p.get_link_parameters("zero").link_mass == 0.0
    assert p.get_link_parameters("bare").link_mass == 0.0


def _six(m):  # cuRobo's layout
    return [m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]]


def test_parsed_link_inertia_is_the_urdf_tensor_in_the_link_frame(tmp_path):
    p = _links_parser(tmp_path)
    I = np.array([[1e-3, 1e-4, -2e-4], [1e-4, 2e-3, 3e-4], [-2e-4, 3e-4, 3e-3]])
    R = Rotation.from_euler("xyz", [0.3, -0.5, 1.0]).as_matrix()  # URDF rpy is Rz·Ry·Rx
    np.testing.assert_allclose(p.get_link_parameters("unrot").link_inertia, _six(I), atol=1e-12)
    np.testing.assert_allclose(p.get_link_parameters("rot").link_inertia, _six(R @ I @ R.T), atol=1e-12)
    np.testing.assert_allclose(p.get_link_parameters("zero_tensor").link_inertia,
                               [2e-4, 2e-4, 2e-4, 0, 0, 0], atol=1e-12)
    np.testing.assert_array_equal(p.get_link_parameters("bare").link_inertia, 0.0)


@needs_cuda
def test_curobo_link_inertias_match_the_urdf():
    # The ZMP gate reads inertias from the URDF itself (get_t1_bodies); cuRobo's copy must agree.
    # The YAML's extra links are not the URDF's: they keep LinkParams' default 1e-4 at any mass.
    b = get_t1_bodies(SIMPLIFIED_MODEL)
    kp = b.kinematics.config.kinematics_config
    extra = set(t1_curobo_cfg(SIMPLIFIED_MODEL)["robot_cfg"]["kinematics"]["extra_links"])
    for k, name in enumerate(b.kinematics.tool_frames):  # the massive links, in b's order
        if name in extra:
            continue
        got = kp.link_inertias[kp.link_name_to_idx_map[name], :6].cpu().numpy()
        np.testing.assert_allclose(got, _six(b.inertia[k].cpu().numpy()), rtol=1e-6, atol=1e-9)


@needs_cuda
def test_whole_robot_com_matches_yourdfpy():
    # The planner's CoM (cuRobo FK, base at 0) at home, the zero conf, two bows and 20 random confs.
    # The YAML's five 0.01 kg base links, absent from the URDF, move it by under 0.2 mm.
    import torch

    from cutamp.com_polygon_cost import _fk_com_and_base
    world = make_blocks_t1_world()
    names = list(world.kinematics.joint_names)
    home = world.q_init.view(1, -1)
    bows = home.repeat(2, 1)
    bows[:, names.index("Torso_Pitch")] = bows.new_tensor([-0.6, -1.2])
    lo, hi = world.kinematics.get_joint_limits().position
    rand = lo + torch.rand(20, len(names), generator=torch.Generator().manual_seed(0)).to(lo) * (hi - lo)
    rand[:, [names.index(j) for j in BASE_JOINTS]] = 0.0
    q = torch.cat([home, torch.zeros_like(home), bows, rand])
    com = _fk_com_and_base(world, q)[0].cpu().numpy()
    ref = np.stack([_yourdfpy_com(dict(zip(names, row)), SIMPLIFIED) for row in q.tolist()])
    assert np.abs(com - ref).max() < 1e-3


def test_simplified_trunk_inertial_matches_t1_urdf():
    # fd5d64c shifted it by the Trunk mesh's -0.0625 a second time; t1.urdf's value is the one
    # that matches the mesh (Trunk.STL volume centroid at x -0.0073 in the Trunk frame).
    def trunk_inertial(urdf):
        return yourdfpy.URDF.load(str(urdf), load_meshes=False).link_map["Trunk"].inertial.origin

    np.testing.assert_allclose(trunk_inertial(SIMPLIFIED), trunk_inertial(T1_ASSETS_DIR / "t1.urdf"), atol=1e-12)


GRIPPER = {f"{s}_{g}" for s in ("left", "right") for g in ("base_link", "Link1", "Link11", "Link2", "Link22")}
ARM = ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw", "Wrist_Pitch", "Wrist_Yaw", "Hand_Roll")


def _load(urdf: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True,
                              build_collision_scene_graph=False, load_collision_meshes=False)


def _com_of(u: yourdfpy.URDF, cfg: dict, skip=()) -> np.ndarray:
    """``u``'s CoM in its root frame at ``cfg``, leaving out the links in ``skip``."""
    u.update_cfg({j: cfg.get(j, 0.0) for j in u.actuated_joint_names})
    s, m = np.zeros(3), 0.0
    for name, link in u.link_map.items():
        if name in skip or link.inertial is None or not link.inertial.mass:
            continue
        T = u.get_transform(name)
        s += link.inertial.mass * (T[:3, :3] @ link.inertial.origin[:3, 3] + T[:3, 3])
        m += link.inertial.mass
    return s / m


def test_simplified_com_matches_t1_urdf():
    # The planner's model is the real T1 in double support, legs together: its x, y CoM is t1.urdf's,
    # in the feet's frame, for any leg, torso and arm posture. The gripper links t1.urdf lacks are
    # left out of the simplified side.
    simp, real = _load(SIMPLIFIED), _load(T1_ASSETS_DIR / "t1.urdf")
    rng = np.random.default_rng(0)
    lim = {"ankle_pitch": (-0.6, 0.3), "knee_pitch": (0.0, 1.8), "Torso_Pitch": (-1.6, 0.3), "Waist_Yaw": (-1.0, 1.0)}
    err = []
    for _ in range(400):
        arms = {f"{s}_{j}": rng.uniform(-1.5, 1.5) for s in ("Left", "Right") for j in ARM}
        leg = {k: rng.uniform(*v) for k, v in lim.items()}
        r_cfg = {**arms, "Waist": -leg["Waist_Yaw"]}
        for s in ("Left", "Right"):
            r_cfg.update({f"{s}_Hip_Pitch": leg["Torso_Pitch"], f"{s}_Knee_Pitch": leg["knee_pitch"],
                          f"{s}_Ankle_Pitch": leg["ankle_pitch"]})
        c = _com_of(real, r_cfg)
        L, R = real.get_transform("left_foot_link"), real.get_transform("right_foot_link")
        c_feet = L[:3, :3].T @ (c - (L[:3, 3] + R[:3, 3]) / 2)
        err.append(c_feet[:2] - _com_of(simp, {**arms, **leg}, skip=GRIPPER)[:2])
    assert np.abs(err).max() < 5e-4


def test_simplified_base_is_the_sole_plane():
    # mobile_base_link's origin is on the floor: the ankle sits as high above it as t1.urdf's ankle
    # sits above the foot box's bottom. The trajectory ZMP uses the base-frame z as the CoM height.
    simp, real = _load(SIMPLIFIED), _load(T1_ASSETS_DIR / "t1.urdf")
    simp.update_cfg({j: 0.0 for j in simp.actuated_joint_names})
    real.update_cfg({j: 0.0 for j in real.actuated_joint_names})
    box = real.link_map["left_foot_link"].collisions[0]
    sole = (real.get_transform("left_foot_link") @ box.origin)[2, 3] - box.geometry.box.size[2] / 2
    ankle_real = real.get_transform("Ankle_Cross_Left")[2, 3] - sole
    assert simp.get_transform("shank_link")[2, 3] == pytest.approx(ankle_real, abs=1e-3)


@pytest.mark.parametrize("torso_pitch, want_x", [(0.0, 0.00952), (-0.6, 0.0800), (-1.2, 0.1262), (-1.6, 0.1350)])
def test_reference_com_x(torso_pitch, want_x):
    # Measured with yourdfpy on the real-mass model. The vendor model (legs together, no gripper)
    # gives +0.80 cm at home and 7.94 / 12.67 / 13.62 cm for these bows; the grippers' 0.29 kg
    # makes up the difference.
    x = _yourdfpy_com({**HOME, "Torso_Pitch": torso_pitch}, SIMPLIFIED)[0]
    assert x == pytest.approx(want_x, abs=5e-4)
