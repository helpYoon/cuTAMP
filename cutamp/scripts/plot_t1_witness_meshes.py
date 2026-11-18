"""Render and independently verify the URDF mesh poses of two saved T1 no-step witnesses."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from yourdfpy import URDF

from cutamp.reachability import canonical_stance
from cutamp.robots.t1 import REAL_LEGS, TOOL_FRAME_FOR_ARM, load_sweep_volume, trunk_pose_from_q


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _view_basis(azimuth, elevation):
    az, el = np.deg2rad([azimuth, elevation])
    eye = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = _unit(np.cross([0, 0, 1], eye))
    up = np.cross(eye, right)
    return np.vstack([right, up, eye]).T


def _transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def _sole_world(urdf, root, cfg, side):
    extra = cfg["extra_links"][side + "_sole_link"]
    x, y, z, qw, qx, qy, qz = extra["fixed_transform"]
    foot_from_sole = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])
    foot_from_sole[:3, 3] = [x, y, z]
    return root @ urdf.get_transform(extra["parent_link_name"]) @ foot_from_sole


def _pose(urdf, report, row, cfg, contact_offset):
    home = np.array(report["home_soles"])
    _, frame = canonical_stance(home)
    c, s = np.cos(frame[2]), np.sin(frame[2])
    world_stance = np.array([[c, -s, 0, frame[0]], [s, c, 0, frame[1]], [0, 0, 1, 0], [0, 0, 0, 1]])
    q = row["q"]
    positions = dict(cfg["lock_joints"], **dict(zip(REAL_LEGS.joint_names, q)))
    urdf.update_cfg([positions[name] for name in urdf.actuated_joint_names])
    # The visual URDF attaches Trunk rigidly to world. The six floating-base
    # joints from the planner YAML supply its pose in the witness's stance frame.
    root = world_stance @ trunk_pose_from_q(REAL_LEGS, q).numpy()
    hand_frame = TOOL_FRAME_FOR_ARM[report["arm"]]
    point = (root @ urdf.get_transform(hand_frame) @ contact_offset)[:3]
    hand_error = float(np.linalg.norm(point - row["point_world"]))
    pos_errors, angle_errors = [], []
    for side, (x, y, heading) in zip(("left", "right"), home):
        foot = _sole_world(urdf, root, cfg, side)
        rotation = trimesh.transformations.rotation_matrix(heading, [0, 0, 1])[:3, :3]
        pos_errors.append(np.linalg.norm(foot[:3, 3] - [x, y, 0]))
        cosine = (np.trace(rotation.T @ foot[:3, :3]) - 1) / 2
        angle_errors.append(np.arccos(np.clip(cosine, -1, 1)))
    errors = {
        "hand_error_m": hand_error,
        "max_sole_error_m": float(max(pos_errors)),
        "max_sole_error_rad": float(max(angle_errors)),
    }
    if hand_error > 1e-4 or max(pos_errors) > 0.0003 + 1e-6 or max(angle_errors) > 0.001 + 1e-6:
        raise ValueError(f"Visual URDF pose disagrees with the recorded FK witness: {errors}")
    lines = {}
    lines["spine"] = [(root @ urdf.get_transform(link))[:3, 3].tolist() for link in ("Waist", "Trunk", "H2")]
    for side, arm_prefix in (("left", "AL"), ("right", "AR")):
        palm = root @ urdf.get_transform(TOOL_FRAME_FOR_ARM[side])
        arm_links = [f"{arm_prefix}{i}" for i in (1, 3, 5)]
        arm = [(root @ urdf.get_transform(link))[:3, 3].tolist() for link in arm_links]
        lines[f"{side}_arm"] = [*arm, palm[:3, 3].tolist(), (palm @ contact_offset)[:3].tolist()]
        leg_links = ["Waist", f"Hip_Pitch_{side.title()}", f"Shank_{side.title()}"]
        leg = [(root @ urdf.get_transform(link))[:3, 3].tolist() for link in leg_links]
        lines[f"{side}_leg"] = [*leg, _sole_world(urdf, root, cfg, side)[:3, 3].tolist()]
    errors["landmarks_world"] = lines
    return root, point, errors


def _draw_meshes(ax, urdf, root, view, arm, box, accent):
    triangles, colors = [], []
    geometries = []
    for node in urdf.scene.graph.nodes_geometry:
        local, key = urdf.scene.graph[node]
        name = node.lower()
        active = name.startswith(("al", "left_") if arm == "left" else ("ar", "right_"))
        active = active and "foot" not in name
        dark = any(part in name for part in ("ankle", "waist", "link11", "link22", "logo", "h1.", "hip_yaw"))
        color = accent if active else ([0.29, 0.36, 0.40] if dark else [0.71, 0.76, 0.78])
        geometries.append((urdf.scene.geometry[key], root @ local, np.array(color)))
    geometries.append((box, np.eye(4), np.array([0.91, 0.20, 0.44])))
    light = _unit(np.array([0.9, 0.35, 1]))
    for mesh, pose, rgb in geometries:
        vertices = _transform(mesh.vertices, pose)
        normals = mesh.face_normals @ pose[:3, :3].T
        # Discard rear-facing triangles and draw the visible triangles in one
        # far-to-near order across all links and the pick box.
        front = normals @ view[:, 2] > 0
        camera_xyz = vertices @ view
        triangles.append(camera_xyz[mesh.faces[front]])
        intensity = (
            0.56 + 0.30 * np.maximum(normals[front] @ light, 0) + 0.16 * np.maximum(normals[front] @ view[:, 2], 0)
        )
        colors.append(np.minimum(1, intensity[:, None] * rgb))
    triangles = np.concatenate(triangles)
    order = np.argsort(triangles[:, :, 2].mean(axis=1))
    collection = PolyCollection(
        triangles[order, :, :2],
        facecolors=np.concatenate(colors)[order],
        edgecolors="none",
        antialiased=False,
        zorder=2,
    )
    ax.add_collection(collection)
    return triangles[:, :, :2].reshape(-1, 2)


def plot_witness_meshes(report, output_path):
    """Save a side-by-side metric view and return checks on the hand and both complete sole transforms."""
    if isinstance(report, (str, Path)):
        report = json.loads(Path(report).read_text())
    rows = [next(r for r in report[mode] if r["order"] == "none") for mode in ("standing", "crouching")]
    if any(row["q"] is None for row in rows):
        raise ValueError("The report must contain both fixed-foot FK witnesses")
    # cuRobo may have normalized its cached config in place after map generation.
    cfg = yaml.safe_load(REAL_LEGS.cfg_path.read_text())["robot_cfg"]["kinematics"]
    urdf = URDF.load(
        str(REAL_LEGS.asset_root / "t1_planner.urdf"),
        filename_handler=lambda fname: str(REAL_LEGS.asset_root / fname),
        load_collision_meshes=False,
    )
    gripper = load_sweep_volume()
    offset = np.array(gripper["base_from_G"]) @ [*gripper["offset_mid"], 1]
    center, size = np.array(report["target_center"]), np.array(report["target_size"])
    box = trimesh.creation.box(extents=size)
    box.apply_translation(center)
    view = _view_basis(65 if report["arm"] == "left" else -65, 11)
    fore = 0.5 * (np.array(report["home_soles"])[0, :2] + np.array(report["home_soles"])[1, :2])
    fig = Figure(figsize=(9, 5.35), dpi=165, facecolor="#fafbf8")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.84, bottom=0.12, wspace=0.10)
    fig.text(0.05, 0.947, "Fixed-foot T1 poses for the low pick", color="#20333b", weight="bold", fontsize=17)
    fig.text(
        0.05,
        0.90,
        "Colored mesh: acting arm   •   Ring: hand contact center   •   Pink box: pick region",
        color="#455a63",
        fontsize=9,
    )
    extents, checks = [], {}
    for ax, row, mode, rgb in zip(axes, rows, ("Standing", "Crouching"), ([0.23, 0.60, 0.73], [0.90, 0.54, 0.30])):
        root, hand, checks[mode.lower()] = _pose(urdf, report, row, cfg, offset)
        xx, yy = fore
        for delta in np.arange(-0.3, 0.51, 0.1):
            for line in (
                [[xx + delta, yy - 0.3, 0], [xx + delta, yy + 0.4, 0]],
                [[xx - 0.3, yy + delta, 0], [xx + 0.5, yy + delta, 0]],
            ):
                pts = np.array(line) @ view
                ax.plot(pts[:, 0], pts[:, 1], color="#d0d9d8", lw=0.55, zorder=0)
        extents.append(_draw_meshes(ax, urdf, root, view, report["arm"], box, rgb))
        hand2d = hand @ view
        ax.scatter(*hand2d[:2], s=24, facecolors="none", edgecolors="#1c2730", linewidths=1.25, zorder=3)
        delta = row["point_distance_m"]
        metric = "inside pick region" if delta == 0 else f"{100 * delta:.1f} cm from pick region"
        ax.set_title(f"{mode}   ·   {metric}", fontsize=10.5, color="#20333b", fontweight="bold", pad=7)
        ax.set(aspect="equal", facecolor="#fafbf8")
        ax.set_axis_off()
    bounds = np.concatenate(extents)
    xmin, zmin = bounds.min(axis=0)
    xmax, zmax = bounds.max(axis=0)
    # Preserve camera, world-floor reference and scale across both panels.
    for ax in axes:
        ax.set(xlim=(xmin - 0.11, xmax + 0.10), ylim=(min(-0.07, zmin - 0.015), zmax + 0.035))
        x = xmax + 0.05
        ax.plot([x, x], [0.025, 0.225], color="#5c727a", lw=1.1)
        for y in [0.025, 0.225]:
            ax.plot([x - 0.012, x + 0.012], [y, y], color="#5c727a", lw=1.1)
        ax.text(x + 0.013, 0.125, "20 cm", rotation=90, va="center", fontsize=7, color="#536a72")
    fig.text(
        0.5,
        0.055,
        "Both soles fixed to the same floor poses     |     Orthographic view     |     10 cm floor grid",
        ha="center",
        color="#52666c",
        fontsize=9,
    )
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    fig.clear()
    return checks


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("report", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    print(json.dumps(plot_witness_meshes(args.report, args.output), indent=2))
