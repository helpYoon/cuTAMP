"""
Interactive YAML Collision Sphere Editor for Robot URDFs.

This tool allows you to:
1. Load existing collision spheres from cuRobo YAML configuration
2. Manually add/edit/delete collision spheres for specific links
3. Optionally auto-fit spheres from link collision geometry (F key)
4. Export back to YAML for cuRobo motion planning

Usage:
    python -m cutamp.scripts.yaml_sphere_editor \
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
import yaml
from yourdfpy import URDF

# cuRobo imports for mesh sphere fitting
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh

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


def fit_spheres_to_mesh_file(
    mesh_path: Path,
    origin: np.ndarray,
    n_spheres: int = 10,
    surface_sphere_radius: float = 0.01,
) -> np.ndarray:
    """Fit spheres to a mesh file using cuRobo's sphere fitting."""
    if not mesh_path.exists():
        _log.warning(f"Mesh file not found: {mesh_path}")
        return np.zeros((0, 4), dtype=np.float32)
    
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        pts, radii = fit_spheres_to_mesh(
            mesh,
            n_spheres=n_spheres,
            surface_sphere_radius=surface_sphere_radius,
            fit_type=SphereFitType.VOXEL,
            voxelize_method="subdivide",
        )
        
        if pts is None or len(pts) == 0:
            _log.warning(f"No spheres fit for mesh: {mesh_path}")
            return np.zeros((0, 4), dtype=np.float32)
        
        pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
        pts_transformed = (origin @ pts_hom.T).T[:, :3]
        
        radii = np.array(radii).reshape(-1, 1)
        spheres = np.hstack([pts_transformed, radii])
        
        return spheres.astype(np.float32)
    
    except Exception as e:
        _log.error(f"Error fitting spheres to mesh {mesh_path}: {e}")
        return np.zeros((0, 4), dtype=np.float32)


def get_collision_origin(collision) -> np.ndarray:
    """Get 4x4 transform from collision origin, defaulting to identity."""
    if collision.origin is not None:
        return collision.origin
    return np.eye(4)


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


@dataclass
class EditorState:
    """State for the interactive editor."""
    link_spheres: Dict[str, np.ndarray] = field(default_factory=dict)
    link_names: List[str] = field(default_factory=list)
    current_link_idx: int = 0
    current_sphere_idx: int = 0
    move_step: float = 0.005
    radius_step: float = 0.005
    modified: bool = False


def create_sphere_mesh(center: np.ndarray, radius: float, color: List[float]) -> o3d.geometry.TriangleMesh:
    """Create an Open3D sphere mesh."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def create_robot_mesh(urdf: URDF, urdf_dir: Path) -> o3d.geometry.TriangleMesh:
    """Create a combined mesh of the robot for visualization."""
    combined = o3d.geometry.TriangleMesh()
    
    urdf.update_cfg(np.zeros(urdf.num_actuated_joints))
    
    for link_name, link in urdf.link_map.items():
        if not link.visuals:
            continue
        
        for visual in link.visuals:
            geom = visual.geometry
            origin = visual.origin if visual.origin is not None else np.eye(4)
            
            try:
                T_world_link = urdf.get_transform(link_name)
                T_world_geom = T_world_link @ origin
                
                mesh = None
                
                if geom.box is not None:
                    mesh = o3d.geometry.TriangleMesh.create_box(*geom.box.size)
                    mesh.translate(-np.array(geom.box.size) / 2)
                
                elif geom.cylinder is not None:
                    # Open3D creates cylinders centered at origin (z: -height/2 to +height/2)
                    # URDF also centers cylinders at origin - no extra translation needed
                    mesh = o3d.geometry.TriangleMesh.create_cylinder(
                        radius=geom.cylinder.radius,
                        height=geom.cylinder.length,
                    )
                
                elif geom.sphere is not None:
                    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=geom.sphere.radius)
                
                elif geom.mesh is not None:
                    mesh_path = urdf_dir / geom.mesh.filename
                    if mesh_path.exists():
                        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                
                if mesh is not None:
                    mesh.transform(T_world_geom)
                    mesh.paint_uniform_color([0.7, 0.7, 0.7])
                    combined += mesh
            
            except Exception as e:
                _log.debug(f"Could not process visual for {link_name}: {e}")
    
    combined.compute_vertex_normals()
    return combined


class YamlSphereEditor:
    """Interactive sphere editor for YAML collision spheres."""
    
    def __init__(
        self,
        urdf: URDF,
        urdf_dir: Path,
        yaml_path: Path,
        yaml_data: Dict[str, Any],
        link_spheres: Dict[str, np.ndarray],
        target_links: List[str],
    ):
        self.urdf = urdf
        self.urdf_dir = urdf_dir
        self.yaml_path = yaml_path
        self.yaml_data = yaml_data
        self.target_links = target_links
        
        # Editor state - only target links are editable
        self.state = EditorState(
            link_spheres=link_spheres,
            link_names=target_links,
        )
        
        # Visualization
        self.vis = None
        self.robot_mesh = None
        self.sphere_meshes: Dict[str, List[o3d.geometry.TriangleMesh]] = {}
        
        # Colors
        self.color_normal = [0.2, 0.6, 0.9]  # Blue - editable spheres
        self.color_selected_link = [0.9, 0.6, 0.2]  # Orange - current link
        self.color_selected_sphere = [0.9, 0.2, 0.2]  # Red - selected sphere
    
    def _get_current_link(self) -> Optional[str]:
        if not self.state.link_names:
            return None
        return self.state.link_names[self.state.current_link_idx]
    
    def _get_current_spheres(self) -> Optional[np.ndarray]:
        link = self._get_current_link()
        if link is None:
            return None
        return self.state.link_spheres.get(link)
    
    def _rebuild_sphere_meshes(self):
        """Rebuild all sphere meshes based on current state."""
        # Remove old editable sphere meshes
        for link_name, meshes in self.sphere_meshes.items():
            for mesh in meshes:
                self.vis.remove_geometry(mesh, reset_bounding_box=False)
        self.sphere_meshes.clear()
        
        current_link = self._get_current_link()
        self.urdf.update_cfg(np.zeros(self.urdf.num_actuated_joints))
        
        # Draw editable spheres (target links)
        for link_name, spheres in self.state.link_spheres.items():
            meshes = []
            
            try:
                T_world_link = self.urdf.get_transform(link_name)
            except:
                T_world_link = np.eye(4)
            
            for i, sphere in enumerate(spheres):
                center_local = sphere[:3]
                radius = sphere[3]
                
                center_world = (T_world_link[:3, :3] @ center_local) + T_world_link[:3, 3]
                
                if link_name == current_link:
                    if i == self.state.current_sphere_idx:
                        color = self.color_selected_sphere
                    else:
                        color = self.color_selected_link
                else:
                    color = self.color_normal
                
                mesh = create_sphere_mesh(center_world, radius, color)
                meshes.append(mesh)
                self.vis.add_geometry(mesh, reset_bounding_box=False)
            
            self.sphere_meshes[link_name] = meshes
    
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
    
    def _on_key_l(self, vis):
        """Cycle to next link."""
        if self.state.link_names:
            self.state.current_link_idx = (self.state.current_link_idx + 1) % len(self.state.link_names)
            self.state.current_sphere_idx = 0
            self._rebuild_sphere_meshes()
            self._update_info()
        return False
    
    def _on_key_shift_l(self, vis):
        """Cycle to previous link."""
        if self.state.link_names:
            self.state.current_link_idx = (self.state.current_link_idx - 1) % len(self.state.link_names)
            self.state.current_sphere_idx = 0
            self._rebuild_sphere_meshes()
            self._update_info()
        return False
    
    def _on_key_tab(self, vis):
        """Cycle to next sphere."""
        spheres = self._get_current_spheres()
        if spheres is not None and len(spheres) > 0:
            self.state.current_sphere_idx = (self.state.current_sphere_idx + 1) % len(spheres)
            self._rebuild_sphere_meshes()
            self._update_info()
        return False
    
    def _move_sphere(self, dx: float, dy: float, dz: float):
        """Move the selected sphere."""
        link = self._get_current_link()
        if link is None:
            return
        
        spheres = self.state.link_spheres.get(link)
        if spheres is None or len(spheres) == 0:
            return
        
        idx = self.state.current_sphere_idx
        if idx >= len(spheres):
            return
        
        spheres[idx, 0] += dx
        spheres[idx, 1] += dy
        spheres[idx, 2] += dz
        self.state.modified = True
        self._rebuild_sphere_meshes()
        self._update_info()
    
    def _on_key_left(self, vis):
        self._move_sphere(-self.state.move_step, 0, 0)
        return False
    
    def _on_key_right(self, vis):
        self._move_sphere(self.state.move_step, 0, 0)
        return False
    
    def _on_key_up(self, vis):
        self._move_sphere(0, self.state.move_step, 0)
        return False
    
    def _on_key_down(self, vis):
        self._move_sphere(0, -self.state.move_step, 0)
        return False
    
    def _on_key_pageup(self, vis):
        self._move_sphere(0, 0, self.state.move_step)
        return False
    
    def _on_key_pagedown(self, vis):
        self._move_sphere(0, 0, -self.state.move_step)
        return False
    
    def _change_radius(self, delta: float):
        """Change the radius of the selected sphere."""
        link = self._get_current_link()
        if link is None:
            return
        
        spheres = self.state.link_spheres.get(link)
        if spheres is None or len(spheres) == 0:
            return
        
        idx = self.state.current_sphere_idx
        if idx >= len(spheres):
            return
        
        new_radius = max(0.001, spheres[idx, 3] + delta)
        spheres[idx, 3] = new_radius
        self.state.modified = True
        self._rebuild_sphere_meshes()
        self._update_info()
    
    def _on_key_plus(self, vis):
        self._change_radius(self.state.radius_step)
        return False
    
    def _on_key_minus(self, vis):
        self._change_radius(-self.state.radius_step)
        return False
    
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
    
    def _on_key_d(self, vis):
        """Delete the selected sphere."""
        link = self._get_current_link()
        if link is None:
            return False
        
        spheres = self.state.link_spheres.get(link)
        if spheres is None or len(spheres) == 0:
            return False
        
        idx = self.state.current_sphere_idx
        if idx >= len(spheres):
            return False
        
        self.state.link_spheres[link] = np.delete(spheres, idx, axis=0)
        self.state.current_sphere_idx = max(0, idx - 1)
        self.state.modified = True
        self._rebuild_sphere_meshes()
        self._update_info()
        print(f"\nDeleted sphere from {link}")
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
    
    def _on_key_q(self, vis):
        """Quit the editor."""
        if self.state.modified:
            print("\n\nWarning: Unsaved changes! Press 'S' to save or 'Q' again to quit.")
            self.state.modified = False
            return False
        
        print("\n\nQuitting...")
        vis.close()
        return False
    
    def _on_key_r(self, vis):
        """Reset view."""
        vis.reset_view_point(True)
        return False
    
    def run(self):
        """Run the interactive editor."""
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
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("YAML Sphere Editor", width=1280, height=720)
        
        # Register key callbacks
        self.vis.register_key_callback(ord('L'), self._on_key_l)
        self.vis.register_key_callback(ord('L') + 256, self._on_key_shift_l)
        self.vis.register_key_callback(258, self._on_key_tab)  # Tab
        self.vis.register_key_callback(263, self._on_key_left)
        self.vis.register_key_callback(262, self._on_key_right)
        self.vis.register_key_callback(265, self._on_key_up)
        self.vis.register_key_callback(264, self._on_key_down)
        self.vis.register_key_callback(266, self._on_key_pageup)
        self.vis.register_key_callback(267, self._on_key_pagedown)
        self.vis.register_key_callback(ord('='), self._on_key_plus)
        self.vis.register_key_callback(ord('+'), self._on_key_plus)
        self.vis.register_key_callback(ord('-'), self._on_key_minus)
        self.vis.register_key_callback(ord('A'), self._on_key_a)
        self.vis.register_key_callback(ord('D'), self._on_key_d)
        self.vis.register_key_callback(ord('F'), self._on_key_f)
        self.vis.register_key_callback(ord('S'), self._on_key_s)
        self.vis.register_key_callback(ord('Q'), self._on_key_q)
        self.vis.register_key_callback(ord('R'), self._on_key_r)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(coord_frame)
        
        # Add robot mesh
        print("Loading robot mesh...")
        self.robot_mesh = create_robot_mesh(self.urdf, self.urdf_dir)
        self.vis.add_geometry(self.robot_mesh)
        
        # Add editable sphere meshes (target links)
        self._rebuild_sphere_meshes()
        self._update_info()
        
        # Set render options
        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.array([0.1, 0.1, 0.1])
        render_opt.point_size = 5.0
        
        # Run
        self.vis.run()
        self.vis.destroy_window()


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
