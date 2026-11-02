"""The balance model's mass distribution: t1_simplified's link inertials, against yourdfpy.

yourdfpy reads each inertial as the URDF defines it (CoM = the inertial origin's translation in
the link frame), so it is the reference the planner's CoM has to match."""

from pathlib import Path

import numpy as np
import pytest
import yourdfpy

from cutamp.robots.t1 import JOINT_NAMES_FULL, T1_ASSETS_DIR, t1_home

SIMPLIFIED = T1_ASSETS_DIR / "t1_simplified.urdf"
HOME = dict(zip(JOINT_NAMES_FULL, t1_home))


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
