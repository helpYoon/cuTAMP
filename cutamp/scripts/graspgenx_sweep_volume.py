"""The T1 gripper's GraspGenX swept-volume numbers, computed from its meshes.

GraspGenX conditions on two boxes (gripper open and half-open) in its gripper frame G plus a fingertip depth. G has
its origin at ``*_base_link``'s origin with X_G = base +y (closing), Y_G = base +z, Z_G = base +x (approach). Each
box is the volume between the two pads' inner faces: X spans the gap, Y and Z the pads' common extent.

Usage:
    python -m cutamp.scripts.graspgenx_sweep_volume
"""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import brentq

DESC = Path(__file__).resolve().parents[1] / "robots" / "assets" / "t1_description"
OUT = DESC / "graspgenx" / "sweep_volume.json"
BASE_FROM_G = [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]


def _pad_vertices(root: ET.Element, theta: float):
    """Link11 and Link22 mesh vertices in the base frame, fingers posed at (theta, -theta, -theta, theta)."""
    joints = {j.get("name"): j for j in root.findall("joint")}

    def origin(j):
        o = j.find("origin")
        T = trimesh.transformations.euler_matrix(*[float(v) for v in o.get("rpy").split()], axes="sxyz")
        T[:3, 3] = [float(v) for v in o.get("xyz").split()]
        return T

    def rz(a):
        return trimesh.transformations.rotation_matrix(a, [0, 0, 1])

    T11 = origin(joints["left_Link1"]) @ rz(theta) @ origin(joints["left_Link11"]) @ rz(-theta)
    T22 = origin(joints["left_Link2"]) @ rz(-theta) @ origin(joints["left_Link22"]) @ rz(theta)
    a = trimesh.load(DESC / "meshes" / "left_Link11.STL").vertices
    b = trimesh.load(DESC / "meshes" / "left_Link22.STL").vertices
    return trimesh.transform_points(a, T11), trimesh.transform_points(b, T22)


def _box(root: ET.Element, theta: float):
    A, B = _pad_vertices(root, theta)
    y_lo, y_hi = A[:, 1].max(), B[:, 1].min()  # inner faces
    pads = np.concatenate([A, B])
    z_lo, z_hi = max(A[:, 2].min(), B[:, 2].min()), min(A[:, 2].max(), B[:, 2].max())  # common extent
    x_lo, x_hi = pads[:, 0].min(), pads[:, 0].max()
    # G axes: X_G = base y, Y_G = base z, Z_G = base x
    extents = [y_hi - y_lo, z_hi - z_lo, x_hi - x_lo]
    offset = [(y_hi + y_lo) / 2, (z_hi + z_lo) / 2, (x_hi + x_lo) / 2]
    return extents, offset


def compute() -> dict:
    urdf = DESC / "gripper.urdf"
    root = ET.parse(urdf).getroot()
    extents_open, offset_open = _box(root, 0.0)
    gap = lambda t: _box(root, t)[0][0]
    theta_mid = round(brentq(lambda t: gap(t) - extents_open[0] / 2, 0.0, 1.0), 4)
    extents_mid, offset_mid = _box(root, theta_mid)
    nums = {
        "extents_open": extents_open, "offset_open": offset_open,
        "extents_mid": extents_mid, "offset_mid": offset_mid,
        "fingertip_depth": offset_open[2],
    }
    nums = {k: [float(x) for x in v] if isinstance(v, list) else float(v) for k, v in nums.items()}
    sha = hashlib.sha256(json.dumps(nums, sort_keys=True).encode()).hexdigest()
    return {
        **nums, "base_from_G": BASE_FROM_G, "theta_mid": theta_mid, "sha256": sha,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(compute(), indent=2) + "\n")
    print(f"wrote {OUT}")
