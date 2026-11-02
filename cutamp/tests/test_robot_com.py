"""The balance model's mass distribution: t1_simplified's link inertials, against yourdfpy.

yourdfpy reads each inertial as the URDF defines it (CoM = the inertial origin's translation in
the link frame), so it is the reference the planner's CoM has to match."""

from pathlib import Path

import numpy as np
import pytest
import yourdfpy

from cutamp.robots.t1 import JOINT_NAMES_FULL, T1_ASSETS_DIR, t1_home
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda

SIMPLIFIED = T1_ASSETS_DIR / "t1_simplified.urdf"
HOME = dict(zip(JOINT_NAMES_FULL, t1_home))
BASE_JOINTS = ("base_j_x", "base_j_y", "base_j_yaw")  # the planar base, from cuRobo's YAML, not the URDF


def _links_parser(tmp_path):
    """cuRobo's parser on a URDF of fixed links: CoM at (0.10, 0.02, -0.03) unrotated and rotated,
    an inertial with no <origin>, one with mass 0, and a link with no inertial."""
    from curobo.robot_parser import UrdfRobotParser
    inertia = '<inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>'
    inertials = {
        "unrot": f'<origin xyz="0.10 0.02 -0.03" rpy="0 0 0"/><mass value="1.0"/>{inertia}',
        "rot": f'<origin xyz="0.10 0.02 -0.03" rpy="0.3 -0.5 1.0"/><mass value="2.0"/>{inertia}',
        "no_origin": f'<mass value="0.5"/>{inertia}',
        "zero": f'<mass value="0.0"/>{inertia}',
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


@pytest.mark.parametrize("torso_pitch, want_x", [(0.0, 0.00524), (-0.6, 0.0580), (-1.2, 0.0928), (-1.6, 0.0996)])
def test_reference_com_x(torso_pitch, want_x):
    # Measured with yourdfpy on the corrected model. The vendor model (legs together) gives
    # +0.80 cm at home and 7.94 / 12.67 / 13.62 cm for these bows: the 20 kg base box keeps the
    # simplified model's CoM lower and nearer the ankle (design spec 7.1).
    x = _yourdfpy_com({**HOME, "Torso_Pitch": torso_pitch}, SIMPLIFIED)[0]
    assert x == pytest.approx(want_x, abs=5e-4)
