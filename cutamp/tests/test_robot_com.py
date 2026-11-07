"""The balance model's mass distribution: cuTAMP's massless-link patch on cuRobo's URDF parser (on a test URDF)."""

import numpy as np


def test_massless_links_weigh_nothing(tmp_path):
    # cuRobo would weigh a link with no inertial at 0.01 kg; _curobo_internals' patch gives it no mass or inertia.
    from curobo.robot_parser import UrdfRobotParser
    path = tmp_path / "links.urdf"
    path.write_text('<robot name="links"><link name="base"/><link name="bare"/>'
                    '<joint name="j_bare" type="fixed"><parent link="base"/><child link="bare"/></joint></robot>')
    bare = UrdfRobotParser(str(path)).get_link_parameters("bare")
    assert bare.link_mass == 0.0
    np.testing.assert_array_equal(bare.link_inertia, 0.0)

