"""
Interactive YAML Collision Sphere Editor for Robot URDFs.

This tool allows you to:
1. Load existing collision spheres from cuRobo YAML configuration
2. Manually add/edit/delete collision spheres for specific links
3. Optionally auto-fit spheres from link collision geometry (F key)
4. Export back to YAML for cuRobo motion planning

Usage (the defaults edit the real legs' upper-body spheres against t1_planner.urdf):
    python -m cutamp.scripts.robot_sphere_editor --links Trunk AL3

Rerun cutamp/robots/assets/t1_29dof/build_planner_model.py after saving: it writes t1_29dof_spheres.yml from the
edited upper_body_spheres.yml.

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
    fit_spheres_to_link,
    load_yaml_spheres,
    save_yaml_spheres,
    yaml_spheres_to_numpy,
)

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


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

T1_29DOF = Path(__file__).resolve().parents[1] / "robots" / "assets" / "t1_29dof"
DEFAULT_URDF = T1_29DOF / "t1_planner.urdf"
DEFAULT_YAML = T1_29DOF / "upper_body_spheres.yml"


def default_paths() -> list:
    """The files the defaults read and write."""
    return [DEFAULT_URDF, DEFAULT_YAML]


def main():
    parser = argparse.ArgumentParser(description="Interactive YAML Collision Sphere Editor")
    parser.add_argument(
        "--urdf",
        type=str,
        default=str(DEFAULT_URDF),
        help="Path to URDF file",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=str(DEFAULT_YAML),
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
        default=[],
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

        # Add extra links even if they're not in YAML yet
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
