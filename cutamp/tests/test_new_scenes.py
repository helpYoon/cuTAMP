"""The new scenes (design 7.7): shelf_t1, cylinder_t1 and cup_t1, the procedural cup, and the fit of their held
objects."""
import numpy as np
import pytest
import yaml

from cutamp.envs.utils import FLOOR, TABLE_THICKNESS, load_env, raise_surface_height, resolve_env_path
from cutamp.tests.conftest import needs_cuda

#: Each new scene and the ``--surface_height`` run_cases.sh runs it at.
SCENES = {"shelf_t1": 0.0, "cylinder_t1": 0.5, "cup_t1": 0.5}


def _scene(name: str):
    return raise_surface_height(load_env(resolve_env_path(name)), SCENES[name])


def _xy_bounds(obj) -> np.ndarray:
    """``[2, 2]``: the object's world-frame xy box, from its posed trimesh."""
    return obj.get_trimesh_mesh(transform_with_pose=True).bounds[:, :2]


def test_cup_is_watertight():
    cup = next(o for o in _scene("cup_t1").movables if o.name == "cup").get_trimesh_mesh()  # as the planner loads it
    assert cup.is_watertight
    # Its outer diameter; 1 um for the OBJ's 8-digit coordinates, which leave ring vertices up to 3 nm out.
    assert 2 * np.hypot(cup.vertices[:, 0], cup.vertices[:, 1]).max() <= 0.05 + 1e-6


@pytest.mark.parametrize("name", SCENES)
def test_scenes_load(name):
    with open(resolve_env_path(name)) as f:  # the file's own quaternions: load_env would normalize any
        for objs in yaml.safe_load(f)["geometries"].values():
            for obj, entry in objs.items():
                assert abs(np.linalg.norm(entry["pose"][3:]) - 1) < 1e-9, obj
    env = _scene(name)
    floor = next(o for o in env.statics if o.name == "floor")
    assert (list(floor.dims), list(floor.pose)) == (FLOOR["dims"], FLOOR["pose"])
    raised = env.movables + env.type_to_objects["Surface"]
    for obj in raised:  # within standing reach
        assert 0.35 <= obj.pose[0] <= 0.55 and abs(obj.pose[1]) <= 0.3, obj.name
    if SCENES[name]:  # the table top under every raised object, a mesh included
        table = next(o for o in env.statics if o.name == "table")
        assert abs(table.pose[2] + TABLE_THICKNESS / 2 - SCENES[name]) < 1e-9
        lo, hi = _xy_bounds(table)
        for obj in raised:
            box = _xy_bounds(obj)
            assert (lo <= box[0] + 1e-9).all() and (box[1] <= hi + 1e-9).all(), obj.name


_THIN_WALL = ("cuRobo's VOXEL fit keeps spheres inscribed in the mesh, and the cup's 4 mm wall holds none above 2 mm: at "
              "6 slots it falls back to 4 surface spheres of 2 mm, and no count from 6 to 128 passes (2026-11-08)")


@needs_cuda
@pytest.mark.parametrize("name, obj_name", [
    ("shelf_t1", "box"),
    ("cylinder_t1", "cylinder"),
    pytest.param("cup_t1", "cup", marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=_THIN_WALL)),
])
def test_held_fit(name, obj_name):
    """The spheres ``motion_solver._attach_object`` fits to the held object at its pick: cuRobo's VOXEL fit of its
    trimesh posed at its start (``AttachmentManager``'s), with the 2 mm surface radius of ``attach_from_scene``'s
    default and as many spheres as the attach link has slots (build_planner_model.py's ``HELD_OBJECT_SPHERES``, in
    t1_29dof.yml). None may be a 2 mm surface fallback, and they reach within 2 cm of the object's bottom and top."""
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh

    from cutamp.robots.t1 import REAL_LEGS

    slots = yaml.safe_load(REAL_LEGS.cfg_path.read_text())["robot_cfg"]["kinematics"]["extra_collision_spheres"]
    assert slots["attached_object_left"] == slots["attached_object_right"]
    obj = next(o for o in _scene(name).movables if o.name == obj_name)
    mesh = obj.get_trimesh_mesh(transform_with_pose=True)
    fit = fit_spheres_to_mesh(mesh, num_spheres=slots["attached_object_left"], surface_radius=0.002,
                              fit_type=SphereFitType.VOXEL)
    z, r = fit.centers[:, 2].cpu().numpy(), fit.radii.cpu().numpy()
    assert len(r) and (r > 0.002 + 1e-6).all(), f"radii {np.round(r * 1000, 2)} mm"  # a fallback's is 2 mm
    bottom, top = mesh.bounds[:, 2]
    assert (z - r).min() - bottom <= 0.02 and top - (z + r).max() <= 0.02, \
        f"{(z - r).min() - bottom:.4f} m from the bottom, {top - (z + r).max():.4f} m from the top"
