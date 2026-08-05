"""
Interactive YAML Collision Sphere Editor for Robot URDFs.

This tool allows you to:
1. Load existing collision spheres from cuRobo YAML configuration
2. Manually add/edit/delete collision spheres for specific links
3. Optionally auto-fit spheres from link collision geometry (F key)
4. Export back to YAML for cuRobo motion planning

Usage:
    python -m cutamp.scripts.robot_sphere_editor \
        --urdf path/to/robot.urdf \
        --yaml path/to/robot.yml \
        --links mobile_base_link shank_link thigh_link

Key Controls:
    L           - Cycle through links
    Tab         - Cycle through spheres in current link
    Arrow keys  - Move selected sphere (X/Y)
    PgUp/PgDn   - Move selected sphere (Z)
    +/-         - Adjust radius
    A           - Add sphere at link origin (manual placement)
    D           - Delete selected sphere
    F           - Auto-fit spheres from link collision geometry
    S           - Save to YAML
    R           - Reset view
    Q           - Quit
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from yourdfpy import URDF

from cutamp.scripts._sphere_editor_common import (
    SphereEditor,
    fit_spheres_to_mesh_file,
    get_collision_origin,
)

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


# =============================================================================
# Analytical Sphere Fitting for Primitives
# =============================================================================


def fit_spheres_to_box(
    size: Tuple[float, float, float],
    origin: np.ndarray,
    n_spheres: int = 8,
) -> np.ndarray:
    """Fit spheres to a box primitive."""
    sx, sy, sz = size
    half = np.array([sx / 2, sy / 2, sz / 2])
    radius = min(size) / 4

    corners = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                corners.append([dx * half[0], dy * half[1], dz * half[2]])
    corners = np.array(corners)

    faces = np.array([
        [half[0], 0, 0], [-half[0], 0, 0],
        [0, half[1], 0], [0, -half[1], 0],
        [0, 0, half[2]], [0, 0, -half[2]],
    ])

    all_pts = np.vstack([corners, faces])
    n_pts = min(n_spheres, len(all_pts))
    pts = all_pts[:n_pts]

    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
    pts_transformed = (origin @ pts_hom.T).T[:, :3]

    radii = np.full((len(pts_transformed), 1), radius)
    spheres = np.hstack([pts_transformed, radii])

    return spheres.astype(np.float32)


def fit_spheres_to_cylinder(
    radius: float,
    length: float,
    origin: np.ndarray,
    n_spheres: int = 5,
) -> np.ndarray:
    """Fit spheres to a cylinder primitive."""
    sphere_radius = radius
    half_len = length / 2

    if n_spheres == 1:
        z_positions = [0.0]
    else:
        z_positions = np.linspace(-half_len + sphere_radius, half_len - sphere_radius, n_spheres)

    pts = np.array([[0, 0, z] for z in z_positions])

    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
    pts_transformed = (origin @ pts_hom.T).T[:, :3]

    radii = np.full((len(pts_transformed), 1), sphere_radius)
    spheres = np.hstack([pts_transformed, radii])

    return spheres.astype(np.float32)


def fit_spheres_to_link(
    link,
    urdf_dir: Path,
    n_spheres: int = 10,
) -> np.ndarray:
    """Fit spheres to a URDF link based on its collision geometry type."""
    if not link.collisions:
        return np.zeros((0, 4), dtype=np.float32)

    collision = link.collisions[0]
    geom = collision.geometry
    origin = get_collision_origin(collision)

    if geom.box is not None:
        size = geom.box.size
        _log.info(f"  {link.name}: box {size}")
        return fit_spheres_to_box(size, origin, n_spheres)

    elif geom.cylinder is not None:
        radius = geom.cylinder.radius
        length = geom.cylinder.length
        _log.info(f"  {link.name}: cylinder r={radius}, L={length}")
        return fit_spheres_to_cylinder(radius, length, origin, n_spheres)

    elif geom.sphere is not None:
        r = geom.sphere.radius
        _log.info(f"  {link.name}: sphere r={r}")
        center = origin[:3, 3]
        return np.array([[center[0], center[1], center[2], r]], dtype=np.float32)

    elif geom.mesh is not None:
        mesh_filename = geom.mesh.filename
        mesh_path = urdf_dir / mesh_filename
        _log.info(f"  {link.name}: mesh {mesh_path.name}")
        return fit_spheres_to_mesh_file(mesh_path, origin, n_spheres)

    else:
        _log.warning(f"  {link.name}: unknown geometry type")
        return np.zeros((0, 4), dtype=np.float32)


# =============================================================================
# YAML I/O Functions
# =============================================================================


def load_yaml_spheres(yaml_path: Path) -> Dict[str, Any]:
    """Load the full YAML configuration."""
    if not yaml_path.exists():
        _log.warning(f"YAML file not found: {yaml_path}")
        return {}

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return data or {}


def yaml_spheres_to_numpy(yaml_data: Dict[str, Any], link_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Convert YAML collision_spheres to numpy arrays.

    Args:
        yaml_data: Loaded YAML data
        link_names: If provided, only load these links. If None, load all links.

    Returns:
        Dict mapping link names to (n, 4) numpy arrays
    """
    collision_spheres = yaml_data.get('collision_spheres', {})
    link_spheres = {}

    # Determine which links to load
    links_to_load = link_names if link_names else list(collision_spheres.keys())

    for link_name in links_to_load:
        spheres_list = collision_spheres.get(link_name, [])
        if spheres_list:
            spheres = []
            for s in spheres_list:
                center = s.get('center', [0, 0, 0])
                radius = s.get('radius', 0.01)
                spheres.append([*center, radius])
            link_spheres[link_name] = np.array(spheres, dtype=np.float32)
        else:
            link_spheres[link_name] = np.zeros((0, 4), dtype=np.float32)

    return link_spheres


def numpy_to_yaml_spheres(link_spheres: Dict[str, np.ndarray]) -> Dict[str, List[Dict]]:
    """Convert numpy arrays back to YAML format."""
    result = {}

    for link_name, spheres in link_spheres.items():
        if len(spheres) == 0:
            result[link_name] = []
        else:
            result[link_name] = [
                {
                    'center': [float(s[0]), float(s[1]), float(s[2])],
                    'radius': float(s[3]),
                }
                for s in spheres
            ]

    return result


def save_yaml_spheres(
    yaml_path: Path,
    yaml_data: Dict[str, Any],
    link_spheres: Dict[str, np.ndarray],
) -> None:
    """Save updated spheres back to YAML file."""
    # Update collision_spheres section
    if 'collision_spheres' not in yaml_data:
        yaml_data['collision_spheres'] = {}

    updated_spheres = numpy_to_yaml_spheres(link_spheres)
    yaml_data['collision_spheres'].update(updated_spheres)

    # Write with nice formatting
    with open(yaml_path, 'w') as f:
        f.write("##\n## Collision spheres configuration\n##\n")
        yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False, allow_unicode=True)

    _log.info(f"Saved collision spheres to {yaml_path}")


# =============================================================================
# Open3D Visualization
# =============================================================================


class YamlSphereEditor(SphereEditor):
    """Interactive sphere editor for YAML collision spheres."""

    window_title = "YAML Sphere Editor"

    def __init__(
        self,
        urdf: URDF,
        urdf_dir: Path,
        yaml_path: Path,
        yaml_data: Dict[str, Any],
        link_spheres: Dict[str, np.ndarray],
        target_links: List[str],
    ):
        # Editor state - only target links are editable
        super().__init__(urdf, urdf_dir, link_spheres, target_links)
        self.yaml_path = yaml_path
        self.yaml_data = yaml_data
        self.target_links = target_links

    def _update_info(self):
        """Print current state info."""
        link = self._get_current_link()
        if link is None:
            print("\rNo links selected")
            return

        spheres = self._get_current_spheres()
        n_spheres = len(spheres) if spheres is not None else 0
        sphere_idx = self.state.current_sphere_idx

        if n_spheres > 0 and sphere_idx < n_spheres:
            s = spheres[sphere_idx]
            print(f"\rLink: {link} ({self.state.current_link_idx + 1}/{len(self.state.link_names)}) | "
                  f"Sphere: {sphere_idx + 1}/{n_spheres} | "
                  f"Center: [{s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}] | "
                  f"Radius: {s[3]:.4f}   ", end="", flush=True)
        else:
            print(f"\rLink: {link} ({self.state.current_link_idx + 1}/{len(self.state.link_names)}) | "
                  f"No spheres (press A to add)   ", end="", flush=True)

    def _on_key_a(self, vis):
        """Add a sphere at the link origin."""
        link = self._get_current_link()
        if link is None:
            return False

        # Add sphere at origin with default radius
        new_sphere = np.array([[0, 0, 0, 0.05]], dtype=np.float32)

        if link in self.state.link_spheres and len(self.state.link_spheres[link]) > 0:
            self.state.link_spheres[link] = np.vstack([self.state.link_spheres[link], new_sphere])
        else:
            self.state.link_spheres[link] = new_sphere

        self.state.current_sphere_idx = len(self.state.link_spheres[link]) - 1
        self.state.modified = True
        self._rebuild_sphere_meshes()
        self._update_info()
        print(f"\nAdded sphere to {link}")
        return False

    def _on_key_f(self, vis):
        """Re-fit spheres for current link."""
        link = self._get_current_link()
        if link is None:
            return False

        urdf_link = self.urdf.link_map.get(link)
        if urdf_link is None:
            print(f"\nLink {link} not found in URDF")
            return False

        print(f"\nRe-fitting spheres for {link}...")
        new_spheres = fit_spheres_to_link(urdf_link, self.urdf_dir, n_spheres=5)

        if len(new_spheres) > 0:
            self.state.link_spheres[link] = new_spheres
            self.state.current_sphere_idx = 0
            self.state.modified = True
            self._rebuild_sphere_meshes()
            self._update_info()
            print(f"Fitted {len(new_spheres)} spheres")
        else:
            print("No spheres generated - link may not have collision geometry")

        return False

    def _on_key_s(self, vis):
        """Save to YAML file."""
        print("\n\nSaving to YAML...")
        save_yaml_spheres(self.yaml_path, self.yaml_data, self.state.link_spheres)
        self.state.modified = False
        print("Save complete!\n")
        self._update_info()
        return False

    def _print_intro(self):
        print("\n" + "=" * 60)
        print("YAML Collision Sphere Editor")
        print("=" * 60)
        print(f"\nYAML file: {self.yaml_path}")
        print(f"Target links: {', '.join(self.target_links)}")
        print("\nControls:")
        print("  L / Shift+L  - Cycle through links")
        print("  Tab          - Cycle through spheres")
        print("  Arrow keys   - Move sphere (X/Y)")
        print("  PgUp/PgDn    - Move sphere (Z)")
        print("  +/-          - Adjust radius")
        print("  A            - Add sphere")
        print("  D            - Delete sphere")
        print("  F            - Re-fit current link")
        print("  S            - Save to YAML")
        print("  R            - Reset view")
        print("  Q            - Quit")
        print("=" * 60 + "\n")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Interactive YAML Collision Sphere Editor")
    parser.add_argument(
        "--urdf",
        type=str,
        default="cutamp/robots/assets/t1_description/t1_simplified.urdf",
        help="Path to URDF file",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default="cutamp/robots/assets/t1_description/t1_spheres.yml",
        help="Path to YAML configuration file with collision_spheres",
    )
    parser.add_argument(
        "--links",
        type=str,
        nargs="+",
        default=None,
        help="Specific links to edit (default: all links with spheres in YAML)",
    )
    parser.add_argument(
        "--add-links",
        type=str,
        nargs="+",
        default=["mobile_base_link", "shank_link", "thigh_link"],
        help="Additional links to add for editing (even if empty in YAML)",
    )

    args = parser.parse_args()

    # Resolve paths
    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        urdf_path = Path.cwd() / urdf_path

    yaml_path = Path(args.yaml)
    if not yaml_path.is_absolute():
        yaml_path = Path.cwd() / yaml_path

    if not urdf_path.exists():
        print(f"Error: URDF not found: {urdf_path}")
        return 1

    urdf_dir = urdf_path.parent

    # Load URDF
    print(f"Loading URDF: {urdf_path}")
    urdf = URDF.load(str(urdf_path))

    # Load YAML
    print(f"Loading YAML: {yaml_path}")
    yaml_data = load_yaml_spheres(yaml_path)

    # Load spheres from YAML
    yaml_spheres = yaml_spheres_to_numpy(yaml_data)

    # Determine which links to edit
    if args.links:
        # User specified specific links
        target_link_names = args.links
    else:
        # Default: all links that have spheres in YAML + additional links
        target_link_names = list(yaml_spheres.keys())

        # Add extra links (like lifting column) even if they're not in YAML yet
        if args.add_links:
            for link_name in args.add_links:
                if link_name not in target_link_names:
                    target_link_names.append(link_name)

    # Validate links exist in URDF
    valid_links = []
    for link_name in target_link_names:
        if link_name in urdf.link_map:
            valid_links.append(link_name)
        else:
            print(f"Warning: Link '{link_name}' not found in URDF, skipping")

    if not valid_links:
        print("Error: No valid links to edit")
        return 1

    # Get spheres for all valid links (editable)
    link_spheres = {}
    for link_name in valid_links:
        if link_name in yaml_spheres:
            link_spheres[link_name] = yaml_spheres[link_name]
        else:
            link_spheres[link_name] = np.zeros((0, 4), dtype=np.float32)

    # Summary
    total_spheres = sum(len(s) for s in link_spheres.values())
    links_with_spheres = [k for k, v in link_spheres.items() if len(v) > 0]
    links_empty = [k for k, v in link_spheres.items() if len(v) == 0]

    print(f"\nEditable links: {len(valid_links)}")
    print(f"Total spheres: {total_spheres}")

    if links_with_spheres:
        print(f"\nLinks with spheres ({len(links_with_spheres)}):")
        for link_name in links_with_spheres:
            print(f"  {link_name}: {len(link_spheres[link_name])} spheres")

    if links_empty:
        print(f"\nLinks without spheres ({len(links_empty)}):")
        for link_name in links_empty:
            print(f"  {link_name}: 0 spheres")

    print("\nPress 'A' to add spheres, 'F' to auto-fit from geometry")

    # Run editor (all valid links are now editable)
    editor = YamlSphereEditor(
        urdf=urdf,
        urdf_dir=urdf_dir,
        yaml_path=yaml_path,
        yaml_data=yaml_data,
        link_spheres=link_spheres,
        target_links=valid_links,
    )
    editor.run()

    return 0


if __name__ == "__main__":
    exit(main())
