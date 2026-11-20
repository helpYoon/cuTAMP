"""The stationary wrist visual must leave the T1 gripper's moving fingers free."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
from yourdfpy import URDF


ASSETS = Path(__file__).resolve().parents[1] / "robots" / "assets" / "t1_29dof"


@pytest.mark.parametrize("side,sign,stock", [("left", 1, "AL7"), ("right", -1, "AR7")])
def test_only_articulated_jaws_occupy_the_finger_region(side, sign, stock):
    path = ASSETS / "t1_planner.urdf"
    urdf = ET.parse(path)
    link = f"{side}_hand_link"
    def visual(name):
        filename = urdf.find(f".//link[@name='{name}']/visual/geometry/mesh").get("filename")
        return trimesh.load_mesh(ASSETS / filename)

    mount = visual(link)
    original = trimesh.load_mesh(ASSETS / "meshes" / f"{stock}.STL")
    # Beyond 115 mm from the wrist only the articulated gripper should be visible.
    assert (sign * mount.vertices[:, 1]).max() < 0.115
    # Keep the proximal wrist housing and the raised camera.
    np.testing.assert_allclose((sign * mount.vertices[:, 1]).min(), (sign * original.vertices[:, 1]).min(), atol=1e-6)
    np.testing.assert_allclose(mount.bounds[1, 2], original.bounds[1, 2], atol=1e-6)

    robot = URDF.load(str(path), load_meshes=False)
    closed = dict(zip((f"{side}_{name}" for name in ("Link1", "Link11", "Link2", "Link22")), (1., -1., -1., 1.)))
    opening = []
    for state in ({k: 0. for k in closed}, closed):
        robot.update_cfg(state)
        jaw_centers = []
        for tip in (f"{side}_Link11", f"{side}_Link22"):
            jaw = visual(tip)
            jaw.apply_transform(robot.get_transform(tip, link))
            assert (sign * jaw.vertices[:, 1]).min() > 0.14
            jaw_centers.append(jaw.centroid[0])
        opening.append(abs(jaw_centers[0] - jaw_centers[1]))
    assert opening[0] - opening[1] > 0.07
