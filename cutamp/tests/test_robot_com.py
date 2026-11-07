"""The balance model's mass distribution: cuTAMP's massless-link patch on cuRobo's URDF parser (on a test URDF), and
the real legs' body inertias, which the multibody ZMP reads (get_t1_bodies), against cuRobo's own."""

import numpy as np

from cutamp.robots.t1 import REAL_LEGS, get_t1_bodies, t1_curobo_cfg
from cutamp.tests.conftest import needs_cuda


def test_massless_links_weigh_nothing(tmp_path):
    # cuRobo would weigh a link with no inertial at 0.01 kg; _curobo_internals' patch gives it no mass or inertia.
    from curobo.robot_parser import UrdfRobotParser
    path = tmp_path / "links.urdf"
    path.write_text('<robot name="links"><link name="base"/><link name="bare"/>'
                    '<joint name="j_bare" type="fixed"><parent link="base"/><child link="bare"/></joint></robot>')
    bare = UrdfRobotParser(str(path)).get_link_parameters("bare")
    assert bare.link_mass == 0.0
    np.testing.assert_array_equal(bare.link_inertia, 0.0)


def _six(m):  # cuRobo's layout
    return [m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]]


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
