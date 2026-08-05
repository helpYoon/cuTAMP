"""
Interactive Gripper Sphere Editor for Robot URDFs.

This tool allows you to:
1. Auto-fit collision spheres to gripper links
2. Interactively adjust sphere positions and radii
3. Export to .pt tensor for gripper spheres (used by particle initialization)

Usage:
    python -m cutamp.scripts.gripper_sphere_editor --urdf path/to/robot.urdf

Key Controls:
    L           - Cycle through links
    Tab         - Cycle through spheres in current link
    Arrow keys  - Move selected sphere (X/Y)
    PgUp/PgDn   - Move selected sphere (Z)
    +/-         - Adjust radius
    A           - Add sphere at origin of current link
    D           - Delete selected sphere
    M           - Reassign sphere to a different link (then L to pick, Enter to confirm)
    F           - Re-fit spheres for current link
    S           - Save .pt file
    R           - Reset view
    Q           - Quit
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
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
# Link Sphere Fitting
# =============================================================================

def fit_spheres_to_link(
    link,
    urdf_dir: Path,
    n_spheres: int = 10,
) -> np.ndarray:
    """
    Fit spheres to a URDF link based on its collision geometry type.

    Args:
        link: yourdfpy Link object
        urdf_dir: Directory containing the URDF (for resolving mesh paths)
        n_spheres: Target number of spheres

    Returns:
        (n, 4) array of sphere centers and radii in link frame
    """
    if not link.collisions:
        return np.zeros((0, 4), dtype=np.float32)

    collision = link.collisions[0]
    geom = collision.geometry
    origin = get_collision_origin(collision)

    if geom.mesh is not None:
        mesh_filename = geom.mesh.filename
        mesh_path = urdf_dir / mesh_filename
        _log.info(f"  {link.name}: mesh {mesh_path.name}")
        return fit_spheres_to_mesh_file(mesh_path, origin, n_spheres)

    else:
        _log.warning(f"  {link.name}: unknown geometry type")
        return np.zeros((0, 4), dtype=np.float32)


# =============================================================================
# Export Functions
# =============================================================================

def load_spheres_from_yaml(
    yaml_path: Path,
    gripper_links: List[str],
) -> Dict[str, np.ndarray]:
    """
    Load gripper spheres from YAML file with correct per-link assignments.

    Args:
        yaml_path: Path to the YAML file (e.g., t1_spheres.yml)
        gripper_links: List of gripper link names to load

    Returns:
        Dict mapping link names to (n, 4) sphere arrays in link-local frames
    """
    if not yaml_path.exists():
        _log.info(f"No YAML file found at {yaml_path}")
        return {}

    _log.info(f"Loading spheres from YAML: {yaml_path.name}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    if 'collision_spheres' not in data:
        _log.warning("No collision_spheres key in YAML file")
        return {}

    result = {}
    for link_name in gripper_links:
        spheres_data = data['collision_spheres'].get(link_name, [])
        if not spheres_data:
            continue

        spheres = []
        for s in spheres_data:
            center = s.get('center', [0, 0, 0])
            radius = s.get('radius', 0.01)
            spheres.append([center[0], center[1], center[2], radius])

        if spheres:
            result[link_name] = np.array(spheres, dtype=np.float32)
            _log.info(f"  {link_name}: {len(spheres)} spheres")

    return result


def load_gripper_pt(
    urdf: URDF,
    pt_path: Path,
    gripper_links: List[str],
    ee_link: str,
) -> Dict[str, np.ndarray]:
    """
    Load gripper spheres from .pt file and distribute to link frames.

    Since spheres are stored in ee_link frame, we need to:
    1. Load all spheres
    2. Distribute them to the closest gripper link
    3. Transform back to each link's local frame

    Args:
        urdf: Parsed URDF object
        pt_path: Path to .pt file
        gripper_links: List of gripper link names
        ee_link: End-effector link name (frame spheres are stored in)

    Returns:
        Dict mapping link names to (n, 4) sphere arrays in link-local frames
    """
    if not pt_path.exists():
        _log.info(f"No existing .pt file found at {pt_path}")
        return {}

    _log.info(f"Loading existing spheres from {pt_path}")

    # Load spheres (in ee_link frame)
    spheres_ee = torch.load(pt_path, map_location="cpu").numpy()

    # Filter out invalid spheres
    valid_mask = spheres_ee[:, 3] > 0
    spheres_ee = spheres_ee[valid_mask]

    if len(spheres_ee) == 0:
        _log.warning("No valid spheres found in .pt file")
        return {}

    _log.info(f"Loaded {len(spheres_ee)} spheres from {pt_path.name}")

    # Get FK transforms at zero config
    urdf.update_cfg(np.zeros(urdf.num_actuated_joints))

    # Get transforms for ee_link and all gripper links
    try:
        T_world_ee = urdf.get_transform(ee_link)
    except Exception as e:
        _log.error(f"Could not get transform for ee_link {ee_link}: {e}")
        return {}

    link_transforms = {}  # T_ee_link for each link
    for link_name in gripper_links:
        try:
            T_world_link = urdf.get_transform(link_name)
            T_ee_link = np.linalg.inv(T_world_ee) @ T_world_link
            link_transforms[link_name] = T_ee_link
        except Exception as e:
            _log.warning(f"Could not get transform for link {link_name}: {e}")

    if not link_transforms:
        _log.error("No valid link transforms found")
        return {}

    # Distribute spheres to closest link (by origin distance)
    link_origins_ee = {}  # Link origins in ee frame
    for link_name, T_ee_link in link_transforms.items():
        link_origins_ee[link_name] = T_ee_link[:3, 3]

    link_spheres: Dict[str, List[np.ndarray]] = {name: [] for name in gripper_links}

    for sphere in spheres_ee:
        center_ee = sphere[:3]
        radius = sphere[3]

        # Find closest link
        min_dist = float('inf')
        closest_link = gripper_links[0]
        for link_name, origin in link_origins_ee.items():
            dist = np.linalg.norm(center_ee - origin)
            if dist < min_dist:
                min_dist = dist
                closest_link = link_name

        # Transform sphere center back to link-local frame
        T_ee_link = link_transforms[closest_link]
        T_link_ee = np.linalg.inv(T_ee_link)
        center_local = (T_link_ee[:3, :3] @ center_ee) + T_link_ee[:3, 3]

        link_spheres[closest_link].append(np.array([*center_local, radius], dtype=np.float32))

    # Convert to numpy arrays
    result = {}
    for link_name, spheres_list in link_spheres.items():
        if spheres_list:
            result[link_name] = np.array(spheres_list, dtype=np.float32)
            _log.info(f"  {link_name}: {len(spheres_list)} spheres")

    return result


def export_gripper_pt(
    urdf: URDF,
    link_spheres: Dict[str, np.ndarray],
    gripper_links: List[str],
    ee_link: str,
    output_path: Path,
) -> None:
    """
    Export gripper spheres as .pt tensor in ee_link frame.

    Args:
        urdf: Parsed URDF object
        link_spheres: Dict mapping link names to (n, 4) sphere arrays
        gripper_links: List of gripper link names to include
        ee_link: End-effector link name (frame for spheres)
        output_path: Path to save .pt file
    """
    all_spheres = []

    # Get FK to compute transforms
    # Set robot to zero configuration
    urdf.update_cfg(np.zeros(urdf.num_actuated_joints))

    for link_name in gripper_links:
        if link_name not in link_spheres:
            _log.warning(f"Gripper link {link_name} not in link_spheres, skipping")
            continue

        spheres = link_spheres[link_name]
        if len(spheres) == 0:
            continue

        # Get transforms using the scene graph
        try:
            # Get world-to-link transforms
            T_world_ee = urdf.get_transform(ee_link)
            T_world_link = urdf.get_transform(link_name)

            # Compute ee-to-link transform
            T_ee_link = np.linalg.inv(T_world_ee) @ T_world_link

            # Transform sphere centers to ee frame
            centers = spheres[:, :3]
            radii = spheres[:, 3:4]

            # Apply rotation and translation
            centers_in_ee = (T_ee_link[:3, :3] @ centers.T).T + T_ee_link[:3, 3]

            all_spheres.append(np.hstack([centers_in_ee, radii]))
            _log.info(f"  Added {len(spheres)} spheres from {link_name}")

        except Exception as e:
            _log.error(f"Error transforming spheres for {link_name}: {e}")

    if not all_spheres:
        _log.warning("No gripper spheres to export")
        return

    result = torch.tensor(np.vstack(all_spheres), dtype=torch.float32)
    torch.save(result, output_path)
    _log.info(f"Saved {len(result)} gripper spheres to {output_path}")


class _FlowList(list):
    """A list that will be serialized in flow style (inline) by YAML."""
    pass


def _flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(_FlowList, _flow_list_representer)


def export_spheres_to_yaml(
    link_spheres: Dict[str, np.ndarray],
    yaml_path: Path,
    gripper_links: List[str],
) -> None:
    """
    Export/update gripper spheres in the YAML collision spheres file.

    This updates only the specified gripper links in the YAML file,
    preserving all other links (arm, torso, etc.).

    Args:
        link_spheres: Dict mapping link names to (n, 4) sphere arrays
        yaml_path: Path to the YAML file (e.g., t1_spheres.yml)
        gripper_links: List of gripper link names to update
    """
    # Load existing YAML if it exists
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Ensure collision_spheres key exists
    if 'collision_spheres' not in data:
        data['collision_spheres'] = {}

    # Update only the gripper links
    for link_name in gripper_links:
        spheres = link_spheres.get(link_name)
        if spheres is None or len(spheres) == 0:
            # Set to empty list if no spheres
            data['collision_spheres'][link_name] = []
        else:
            # Convert to YAML format with flow-style centers
            sphere_list = []
            for sphere in spheres:
                sphere_list.append({
                    'center': _FlowList([float(sphere[0]), float(sphere[1]), float(sphere[2])]),
                    'radius': float(sphere[3])
                })
            data['collision_spheres'][link_name] = sphere_list

    # Write back to YAML
    with open(yaml_path, 'w') as f:
        # Write header comment
        f.write("##\n## Collision spheres configuration\n##\n")
        # Dump the data
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Count total spheres updated
    total = sum(len(link_spheres.get(link, [])) for link in gripper_links)
    _log.info(f"Updated {len(gripper_links)} gripper links ({total} spheres) in {yaml_path.name}")


# =============================================================================
# Open3D Visualization
# =============================================================================


class GripperSphereEditor(SphereEditor):
    """Interactive sphere editor using Open3D."""

    window_title = "Gripper Sphere Editor"
    radius_step = 0.002

    def __init__(
        self,
        urdf: URDF,
        urdf_dir: Path,
        link_spheres: Dict[str, np.ndarray],
        gripper_configs: List[Dict],
        spheres_yaml_path: Optional[Path] = None,
    ):
        super().__init__(urdf, urdf_dir, link_spheres, sorted(link_spheres.keys()))
        self.gripper_configs = gripper_configs
        self.spheres_yaml_path = spheres_yaml_path

    def _update_info(self):
        """Print current state info."""
        link = self._get_current_link()
        if link is None:
            print("\rNo links with spheres")
            return

        spheres = self._get_current_spheres()
        n_spheres = len(spheres) if spheres is not None else 0
        sphere_idx = self.state.current_sphere_idx

        # Show reassign mode info
        if self.state.reassign_mode:
            target_link = self.state.link_names[self.state.reassign_target_idx]
            print(f"\r[REASSIGN MODE] Move sphere to: {target_link} ({self.state.reassign_target_idx + 1}/{len(self.state.link_names)}) | "
                  f"Press L to cycle, Enter to confirm, Esc to cancel   ", end="", flush=True)
            return

        if n_spheres > 0 and sphere_idx < n_spheres:
            s = spheres[sphere_idx]
            print(f"\rLink: {link} ({self.state.current_link_idx + 1}/{len(self.state.link_names)}) | "
                  f"Sphere: {sphere_idx + 1}/{n_spheres} | "
                  f"Center: [{s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}] | "
                  f"Radius: {s[3]:.3f}   ", end="", flush=True)
        else:
            print(f"\rLink: {link} ({self.state.current_link_idx + 1}/{len(self.state.link_names)}) | "
                  f"No spheres   ", end="", flush=True)

    def _on_key_l(self, vis):
        """Cycle to next link (or target link in reassign mode)."""
        if not self.state.link_names:
            return False

        if self.state.reassign_mode:
            # In reassign mode, cycle through target links
            self.state.reassign_target_idx = (self.state.reassign_target_idx + 1) % len(self.state.link_names)
            self._update_info()
        else:
            # Normal mode: cycle through links
            self.state.current_link_idx = (self.state.current_link_idx + 1) % len(self.state.link_names)
            self.state.current_sphere_idx = 0
            self._rebuild_sphere_meshes()
            self._update_info()
        return False

    def _on_key_shift_l(self, vis):
        """Cycle to previous link (or target link in reassign mode)."""
        if not self.state.link_names:
            return False

        if self.state.reassign_mode:
            # In reassign mode, cycle through target links
            self.state.reassign_target_idx = (self.state.reassign_target_idx - 1) % len(self.state.link_names)
            self._update_info()
        else:
            # Normal mode: cycle through links
            self.state.current_link_idx = (self.state.current_link_idx - 1) % len(self.state.link_names)
            self.state.current_sphere_idx = 0
            self._rebuild_sphere_meshes()
            self._update_info()
        return False

    def _on_key_a(self, vis):
        """Add a sphere at the link origin."""
        link = self._get_current_link()
        if link is None:
            return False

        # Add sphere at origin with default radius
        new_sphere = np.array([[0, 0, 0, 0.02]], dtype=np.float32)

        if link in self.state.link_spheres:
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
            return False

        print(f"\nRe-fitting spheres for {link}...")
        new_spheres = fit_spheres_to_link(urdf_link, self.urdf_dir, n_spheres=10)

        if len(new_spheres) > 0:
            self.state.link_spheres[link] = new_spheres
            self.state.current_sphere_idx = 0
            self.state.modified = True
            self._rebuild_sphere_meshes()
            self._update_info()

        return False

    def _on_key_s(self, vis):
        """Save .pt files and YAML for all grippers."""
        print("\n\nSaving...")

        # Export .pt for each gripper
        for cfg in self.gripper_configs:
            print(f"\nExporting {cfg['name']} gripper to .pt...")
            export_gripper_pt(
                self.urdf,
                self.state.link_spheres,
                cfg["links"],
                cfg["ee_link"],
                cfg["output_path"],
            )

        # Export to YAML if path is set
        if self.spheres_yaml_path is not None:
            print(f"\nUpdating YAML: {self.spheres_yaml_path.name}...")
            all_gripper_links = []
            for cfg in self.gripper_configs:
                all_gripper_links.extend(cfg["links"])
            export_spheres_to_yaml(
                self.state.link_spheres,
                self.spheres_yaml_path,
                all_gripper_links,
            )

        self.state.modified = False
        print("\nSave complete!\n")
        self._update_info()
        return False

    def _on_key_m(self, vis):
        """Enter reassign mode to move sphere to a different link."""
        link = self._get_current_link()
        if link is None:
            return False

        spheres = self._get_current_spheres()
        if spheres is None or len(spheres) == 0:
            print("\nNo sphere selected to reassign")
            return False

        # Enter reassign mode
        self.state.reassign_mode = True
        self.state.reassign_target_idx = self.state.current_link_idx
        print("\n[REASSIGN MODE] Use L/Shift+L to select target link, Enter to confirm, Esc to cancel")
        self._update_info()
        return False

    def _on_key_enter(self, vis):
        """Confirm reassignment (in reassign mode)."""
        if not self.state.reassign_mode:
            return False

        source_link = self._get_current_link()
        target_link = self.state.link_names[self.state.reassign_target_idx]

        if source_link == target_link:
            print("\n  Cancelled: source and target are the same")
            self.state.reassign_mode = False
            self._update_info()
            return False

        # Get the sphere to move
        spheres = self.state.link_spheres.get(source_link)
        if spheres is None or len(spheres) == 0:
            self.state.reassign_mode = False
            self._update_info()
            return False

        idx = self.state.current_sphere_idx
        if idx >= len(spheres):
            self.state.reassign_mode = False
            self._update_info()
            return False

        sphere_to_move = spheres[idx].copy()

        # Transform sphere from source link frame to target link frame
        self.urdf.update_cfg(np.zeros(self.urdf.num_actuated_joints))
        try:
            T_world_source = self.urdf.get_transform(source_link)
            T_world_target = self.urdf.get_transform(target_link)
            T_target_source = np.linalg.inv(T_world_target) @ T_world_source

            # Transform center
            center_source = sphere_to_move[:3]
            center_target = T_target_source[:3, :3] @ center_source + T_target_source[:3, 3]
            sphere_to_move[:3] = center_target
        except Exception as e:
            _log.warning(f"Could not transform sphere: {e}")

        # Remove from source link
        self.state.link_spheres[source_link] = np.delete(spheres, idx, axis=0)

        # Add to target link
        if target_link in self.state.link_spheres and len(self.state.link_spheres[target_link]) > 0:
            self.state.link_spheres[target_link] = np.vstack([
                self.state.link_spheres[target_link],
                sphere_to_move.reshape(1, 4)
            ])
        else:
            self.state.link_spheres[target_link] = sphere_to_move.reshape(1, 4)

        # Update state
        self.state.reassign_mode = False
        self.state.current_link_idx = self.state.reassign_target_idx
        self.state.current_sphere_idx = len(self.state.link_spheres[target_link]) - 1
        self.state.modified = True

        print(f"\n  Moved sphere from {source_link} to {target_link}")
        self._rebuild_sphere_meshes()
        self._update_info()
        return False

    def _on_key_escape(self, vis):
        """Cancel reassign mode."""
        if self.state.reassign_mode:
            self.state.reassign_mode = False
            print("\n  Reassign cancelled")
            self._update_info()
        return False

    def _register_extra_key_callbacks(self):
        self.vis.register_key_callback(ord('M'), self._on_key_m)
        self.vis.register_key_callback(257, self._on_key_enter)   # Enter (GLFW_KEY_ENTER)
        self.vis.register_key_callback(256, self._on_key_escape)  # Escape (GLFW_KEY_ESCAPE)

    def _print_intro(self):
        print("\n" + "=" * 60)
        print("Gripper Sphere Editor - Interactive Sphere Editing")
        print("=" * 60)
        print("\nControls:")
        print("  L / Shift+L  - Cycle through links")
        print("  Tab          - Cycle through spheres")
        print("  Arrow keys   - Move sphere (X/Y)")
        print("  PgUp/PgDn    - Move sphere (Z)")
        print("  +/-          - Adjust radius")
        print("  A            - Add sphere")
        print("  D            - Delete sphere")
        print("  M            - Reassign sphere to different link")
        print("  F            - Re-fit current link")
        print("  S            - Save .pt file")
        print("  R            - Reset view")
        print("  Q            - Quit")
        print("=" * 60 + "\n")


# =============================================================================
# Main
# =============================================================================


# Default gripper configurations for T1 robot
T1_GRIPPERS = {
    "left": {
        "ee_link": "left_base_link",
        "links": ["left_base_link", "left_Link1", "left_Link11", "left_Link2", "left_Link22"],
        "output_suffix": "left_gripper_spheres.pt",
    },
    "right": {
        "ee_link": "right_base_link",
        "links": ["right_base_link", "right_Link1", "right_Link11", "right_Link2", "right_Link22"],
        "output_suffix": "right_gripper_spheres.pt",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Interactive Gripper Sphere Editor")
    parser.add_argument(
        "--urdf",
        type=str,
        default="cutamp/robots/assets/t1_description/t1_simplified.urdf",
        help="Path to URDF file",
    )
    parser.add_argument(
        "--gripper",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Which gripper(s) to edit and export (default: both)",
    )
    parser.add_argument(
        "--n-spheres",
        type=int,
        default=10,
        help="Target number of spheres per link",
    )
    parser.add_argument(
        "--exclude-links",
        type=str,
        nargs="*",
        default=[],
        help="Links to exclude from sphere fitting",
    )
    parser.add_argument(
        "--load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load existing spheres from YAML/.pt (use --no-load to always auto-fit)",
    )

    args = parser.parse_args()

    # Resolve paths
    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        urdf_path = Path.cwd() / urdf_path

    if not urdf_path.exists():
        print(f"Error: URDF not found: {urdf_path}")
        return 1

    urdf_dir = urdf_path.parent

    # Determine which grippers to process
    if args.gripper == "both":
        grippers_to_process = ["left", "right"]
    else:
        grippers_to_process = [args.gripper]

    # Collect all gripper links for visualization
    all_gripper_links = []
    gripper_configs = []
    for gripper_name in grippers_to_process:
        cfg = T1_GRIPPERS[gripper_name]
        all_gripper_links.extend(cfg["links"])
        output_path = urdf_dir / cfg["output_suffix"]
        gripper_configs.append({
            "name": gripper_name,
            "ee_link": cfg["ee_link"],
            "links": cfg["links"],
            "output_path": output_path,
        })

    # Load URDF
    print(f"Loading URDF: {urdf_path}")
    urdf = URDF.load(str(urdf_path))

    # Set up YAML path for spheres
    spheres_yaml_path = urdf_dir / "t1_spheres.yml"

    # Try loading existing spheres
    link_spheres = {}
    loaded_source = None

    if args.load:
        # First, try loading from YAML (has correct per-link assignments)
        if spheres_yaml_path.exists():
            print(f"\nLoading from YAML: {spheres_yaml_path.name}")
            yaml_spheres = load_spheres_from_yaml(spheres_yaml_path, all_gripper_links)
            if yaml_spheres:
                link_spheres.update(yaml_spheres)
                loaded_source = "YAML"

        # If YAML didn't have all links, try loading from .pt as fallback
        missing_links = [l for l in all_gripper_links if l not in link_spheres]
        if missing_links:
            print(f"\nChecking .pt files for missing links: {missing_links}")
            for cfg in gripper_configs:
                if cfg["output_path"].exists():
                    # Only load links that aren't already loaded from YAML
                    links_to_load = [l for l in cfg["links"] if l in missing_links]
                    if links_to_load:
                        print(f"  Loading from: {cfg['output_path'].name}")
                        loaded = load_gripper_pt(
                            urdf,
                            cfg["output_path"],
                            links_to_load,
                            cfg["ee_link"],
                        )
                        if loaded:
                            link_spheres.update(loaded)
                            if loaded_source:
                                loaded_source = "YAML + .pt"
                            else:
                                loaded_source = ".pt"

    if loaded_source:
        print(f"\nLoaded spheres from {loaded_source}")
    else:
        print(f"\nNo existing spheres found, auto-fitting...")

    # Auto-fit spheres for links that don't have loaded spheres
    links_to_fit = [
        link_name for link_name in all_gripper_links
        if link_name not in link_spheres and link_name not in args.exclude_links
    ]

    if links_to_fit:
        print(f"\nAuto-fitting spheres ({args.n_spheres} per link) for {len(links_to_fit)} links...")
        for link_name in links_to_fit:
            link = urdf.link_map.get(link_name)
            if link is None:
                _log.warning(f"Link {link_name} not found in URDF")
                continue
            spheres = fit_spheres_to_link(link, urdf_dir, args.n_spheres)
            if len(spheres) > 0:
                link_spheres[link_name] = spheres
                _log.info(f"  {link_name}: {len(spheres)} spheres")

    print(f"\nTotal: {len(link_spheres)} gripper links with spheres")
    total_spheres = sum(len(s) for s in link_spheres.values())
    print(f"Total spheres: {total_spheres}")

    print(f"\nGrippers to export: {', '.join(grippers_to_process)}")
    for cfg in gripper_configs:
        print(f"  {cfg['name']}: {cfg['ee_link']} -> {cfg['output_path'].name}")
    print(f"YAML spheres file: {spheres_yaml_path.name}")

    # Run editor
    editor = GripperSphereEditor(
        urdf=urdf,
        urdf_dir=urdf_dir,
        link_spheres=link_spheres,
        gripper_configs=gripper_configs,
        spheres_yaml_path=spheres_yaml_path,
    )
    editor.run()

    return 0


if __name__ == "__main__":
    exit(main())
