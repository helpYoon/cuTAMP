"""Shared sphere-fitting helpers and base editor class for the interactive sphere editors."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d
import trimesh
from yourdfpy import URDF

# cuRobo imports for mesh sphere fitting
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh

_log = logging.getLogger(__name__)


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
    # For reassign mode
    reassign_mode: bool = False
    reassign_target_idx: int = 0


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


class SphereEditor:
    """Base class for the interactive Open3D sphere editors."""

    window_title = "Sphere Editor"
    radius_step = 0.005

    def __init__(
        self,
        urdf: URDF,
        urdf_dir: Path,
        link_spheres: Dict[str, np.ndarray],
        link_names: List[str],
    ):
        self.urdf = urdf
        self.urdf_dir = urdf_dir

        # Editor state
        self.state = EditorState(
            link_spheres=link_spheres,
            link_names=link_names,
            radius_step=self.radius_step,
        )

        # Visualization
        self.vis = None
        self.robot_mesh = None
        self.sphere_meshes: Dict[str, List[o3d.geometry.TriangleMesh]] = {}

        # Colors
        self.color_normal = [0.2, 0.6, 0.9]  # Blue
        self.color_selected_link = [0.9, 0.6, 0.2]  # Orange
        self.color_selected_sphere = [0.9, 0.2, 0.2]  # Red

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
        # Remove old sphere meshes
        for link_name, meshes in self.sphere_meshes.items():
            for mesh in meshes:
                self.vis.remove_geometry(mesh, reset_bounding_box=False)
        self.sphere_meshes.clear()

        current_link = self._get_current_link()
        self.urdf.update_cfg(np.zeros(self.urdf.num_actuated_joints))

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

    def _register_extra_key_callbacks(self):
        pass

    def run(self):
        """Run the interactive editor."""
        self._print_intro()

        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(self.window_title, width=1280, height=720)

        # Register key callbacks
        self.vis.register_key_callback(ord('L'), self._on_key_l)
        self.vis.register_key_callback(ord('L') + 256, self._on_key_shift_l)  # Shift+L
        self.vis.register_key_callback(258, self._on_key_tab)  # Tab (GLFW_KEY_TAB)
        self.vis.register_key_callback(263, self._on_key_left)   # Left arrow
        self.vis.register_key_callback(262, self._on_key_right)  # Right arrow
        self.vis.register_key_callback(265, self._on_key_up)     # Up arrow
        self.vis.register_key_callback(264, self._on_key_down)   # Down arrow
        self.vis.register_key_callback(266, self._on_key_pageup)    # Page Up
        self.vis.register_key_callback(267, self._on_key_pagedown)  # Page Down
        self.vis.register_key_callback(ord('='), self._on_key_plus)
        self.vis.register_key_callback(ord('+'), self._on_key_plus)
        self.vis.register_key_callback(ord('-'), self._on_key_minus)
        self.vis.register_key_callback(ord('A'), self._on_key_a)
        self.vis.register_key_callback(ord('D'), self._on_key_d)
        self.vis.register_key_callback(ord('F'), self._on_key_f)
        self.vis.register_key_callback(ord('S'), self._on_key_s)
        self.vis.register_key_callback(ord('Q'), self._on_key_q)
        self.vis.register_key_callback(ord('R'), self._on_key_r)
        self._register_extra_key_callbacks()

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(coord_frame)

        # Add robot mesh
        print("Loading robot mesh...")
        self.robot_mesh = create_robot_mesh(self.urdf, self.urdf_dir)
        self.vis.add_geometry(self.robot_mesh)

        # Add sphere meshes
        self._rebuild_sphere_meshes()
        self._update_info()

        # Set render options
        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.array([0.1, 0.1, 0.1])
        render_opt.point_size = 5.0

        # Run
        self.vis.run()
        self.vis.destroy_window()
