"""The balance model's mass distribution: cuRobo's URDF parser (each link's CoM, mass and inertia, on a test URDF)
and the real legs' body inertias, which the multibody ZMP reads (get_t1_bodies), against cuRobo's own."""

import numpy as np
from scipy.spatial.transform import Rotation

from cutamp.robots.t1 import REAL_LEGS, get_t1_bodies, t1_curobo_cfg
from cutamp.tests.conftest import needs_cuda


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
    b = get_t1_bodies(REAL_LEGS)
    kp = b.kinematics.config.kinematics_config
    extra = set(t1_curobo_cfg(REAL_LEGS)["robot_cfg"]["kinematics"]["extra_links"])
    for k, name in enumerate(b.kinematics.tool_frames):  # the massive links, in b's order
        if name in extra:
            continue
        got = kp.link_inertias[kp.link_name_to_idx_map[name], :6].cpu().numpy()
        np.testing.assert_allclose(got, _six(b.inertia[k].cpu().numpy()), rtol=1e-6, atol=1e-9)
