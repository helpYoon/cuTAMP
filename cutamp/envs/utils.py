# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from typing import Dict, List, ClassVar, Set, Tuple

import numpy as np
import yaml
from curobo.scene import Cuboid, Obstacle, Cylinder, Mesh

from cutamp.t1_domain import all_t1_fluents
from cutamp.task_planning.base_structs import State


class TAMPEnvironment:
    _known_types: ClassVar[Set[str]] = {"Movable", "Surface"}

    def __init__(
        self,
        name: str,
        movables: List[Obstacle],
        statics: List[Obstacle],
        type_to_objects: Dict[str, List[Obstacle]],
        goal_state: State,
    ):
        self.name = name
        self.movables = movables
        self.statics = statics
        self.type_to_objects = type_to_objects
        self.goal_state = goal_state

        # No object (identified by name) should be in both movables and statics
        movable_names = {obj.name for obj in movables}
        static_names = {obj.name for obj in statics}
        intersection = movable_names.intersection(static_names)
        if intersection:
            raise ValueError(f"Objects cannot be both movable and static: {intersection}")

        # Make sure types are all known
        for obj_type in type_to_objects:
            if obj_type not in self._known_types:
                raise ValueError(f"Unknown object type: {obj_type}")


# raise_surface_height's table top: its thickness.
TABLE_THICKNESS = 0.03

_key_to_factory = {
    "cuboid": Cuboid,
    "cylinder": Cylinder,
    "mesh": Mesh,
}


def load_env(env_path: str) -> TAMPEnvironment:
    """Load the environment from a YAML file and parse into movable and static objects."""
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    with open(env_path, "r") as f:
        env_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Make sure expected keys are present
    for key in ["name", "geometries", "types"]:
        if key not in env_dict:
            raise ValueError(f"Key {key} not found in environment file")

    # Load the geometries
    name_to_obj = {}
    for key, obj_dicts in env_dict["geometries"].items():
        if key not in _key_to_factory:
            raise ValueError(f"Unknown shape type: {key}")
        cls = _key_to_factory[key]
        for name, obj_dict in obj_dicts.items():
            if name in name_to_obj:
                raise ValueError(f"Cannot have duplicate object names: {name}")
            if "pose" in obj_dict:  # wxyz; keep the direction, fix the length
                q = np.array(obj_dict["pose"][3:7], dtype=float)
                obj_dict["pose"] = [float(v) for v in obj_dict["pose"][:3]] + (q / np.linalg.norm(q)).tolist()
            if cls is Mesh and not os.path.isabs(obj_dict["file_path"]):
                obj_dict["file_path"] = os.path.join(os.path.dirname(os.path.abspath(env_path)), obj_dict["file_path"])
            obj = cls(name=name, **obj_dict)
            name_to_obj[name] = obj

    # Parse the types of the objects. If not specified then just a static object
    type_to_objects = {}
    for obj_type, objects in env_dict["types"].items():
        for obj in objects:
            if obj not in name_to_obj:
                raise ValueError(f"Object {obj} not found in geometries")
        type_to_objects[obj_type] = [name_to_obj[obj] for obj in objects]

    # Determine movables and statics. There is a Movable type which we use to determine movables
    movables, statics = [], []
    for obj in name_to_obj.values():
        if obj in type_to_objects["Movable"]:
            movables.append(obj)
        else:
            statics.append(obj)

    # Parse goal states
    name_to_fluent = {fluent.name: fluent for fluent in all_t1_fluents}
    goal_state = set()
    for atom_dict in env_dict["goal"]:
        if not (isinstance(atom_dict, dict) and len(atom_dict) == 1):
            raise ValueError(f"Goal atom should be a dict of length 1, not {atom_dict}")

        for fluent_name, values in atom_dict.items():
            if fluent_name not in name_to_fluent:
                raise ValueError(f"Unknown fluent: {fluent_name}")
            fluent = name_to_fluent[fluent_name]
            if len(values) != len(fluent.parameters):
                raise ValueError(f"Expected {len(fluent.parameters)} values for {fluent_name}, got {values}")
            goal_atom = fluent.ground(*values)
            goal_state.add(goal_atom)
    goal_state = frozenset(goal_state)

    env = TAMPEnvironment(
        name=env_dict["name"],
        movables=movables,
        statics=statics,
        type_to_objects=type_to_objects,
        goal_state=goal_state,
    )
    return env


def raise_surface_height(env: TAMPEnvironment, height: float) -> TAMPEnvironment:
    """Elevate the manipulation surface by ``height`` metres (mutates in place).

    Shifts the z of every Movable and every Surface object up by ``height`` and adds a static ``table`` top,
    ``TABLE_THICKNESS`` thick with its top at ``height``, flush with their xy box (a 5 cm overhang left blocks_t1 s0/s2
    at h 0.5 no plannable candidate); the floor stays put. ``height == 0`` leaves the env unchanged (the ground
    variant). Used by run_cutamp's ``--surface_height``.

    A nonzero ``height`` raises ``ValueError``, before anything moves, where the env has any other static, which
    would stay behind (shelf_t1's boards), or already has an object named ``table``.
    """
    if height == 0.0:
        return env
    targets = list(env.movables) + list(env.type_to_objects.get("Surface", []))
    if any(obj.name == "table" for obj in env.movables + env.statics):
        raise ValueError(f"{env.name} already has an object named 'table', the name of the table top that "
                         "raise_surface_height adds")
    lifted = {obj.name for obj in targets}
    left = [obj.name for obj in env.statics if obj.name != "floor" and obj.name not in lifted]
    if left:
        raise ValueError(f"{env.name}'s statics {', '.join(left)} would stay put while its movables and Surfaces rise "
                         f"{height} m: at a nonzero surface height only the floor and the Surfaces may be static")
    seen: Set[int] = set()
    for obj in targets:
        if id(obj) in seen:  # an object could be both movable and a surface
            continue
        seen.add(id(obj))
        obj.pose[2] += height
    env.statics.append(_table_top(list({id(o): o for o in targets}.values()), height))
    return env


def _xy_half_extents(obj: Obstacle) -> Tuple[float, float]:
    """Half extents of the object's axis-aligned xy box, from its (unit, wxyz) pose."""
    w, z = obj.pose[3], obj.pose[6]
    if isinstance(obj, Cuboid):
        yaw = 2 * np.arctan2(z, w)  # the scenes only yaw their objects
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        return (c * obj.dims[0] + s * obj.dims[1]) / 2, (s * obj.dims[0] + c * obj.dims[1]) / 2
    if isinstance(obj, Cylinder):
        return obj.radius, obj.radius
    if isinstance(obj, Mesh):  # e.g. cup_t1's cup: its posed vertices' farthest reach from its xy position
        x, y = np.abs(obj.get_trimesh_mesh(transform_with_pose=True).vertices[:, :2] - obj.pose[:2]).max(axis=0)
        return float(x), float(y)
    raise NotImplementedError(f"No table footprint for {type(obj)} {obj.name}")


def _table_top(objs: List[Obstacle], height: float) -> Cuboid:
    lo = np.min([[o.pose[0] - _xy_half_extents(o)[0], o.pose[1] - _xy_half_extents(o)[1]] for o in objs], axis=0)
    hi = np.max([[o.pose[0] + _xy_half_extents(o)[0], o.pose[1] + _xy_half_extents(o)[1]] for o in objs], axis=0)
    cx, cy = (lo + hi) / 2
    return Cuboid(
        name="table",
        dims=[float(hi[0] - lo[0]), float(hi[1] - lo[1]), TABLE_THICKNESS],
        pose=[float(cx), float(cy), height - TABLE_THICKNESS / 2, 1, 0, 0, 0],
        color=[160, 160, 160],
    )


def _get_object_dict(obj: Obstacle) -> Tuple[str, dict]:
    """Get the object as a dictionary, so we can serialize."""
    if obj.scale is not None:
        raise NotImplementedError(f"{obj.name} should not have a scale")
    if obj.texture is not None:
        raise NotImplementedError(f"{obj.name} should not have a texture")
    if isinstance(obj, Cuboid):
        return "cuboid", {
            "dims": obj.dims,
            "pose": obj.pose,
            "color": obj.color,
        }
    elif isinstance(obj, Cylinder):
        return "cylinder", {
            "radius": obj.radius,
            "height": obj.height,
            "pose": obj.pose,
            "color": obj.color,
        }
    elif isinstance(obj, Mesh):
        if obj.file_path is None:  # meshes come from files and are pre-scaled
            raise NotImplementedError(f"{obj.name} is a mesh without a file_path")
        return "mesh", {"file_path": obj.file_path, "pose": obj.pose, "color": obj.color}
    raise NotImplementedError(f"Unsupported object type: {type(obj)}")


def get_env_dict(env: TAMPEnvironment) -> dict:
    """Get the environment as a dictionary, so we can serialize."""
    geometries = {}
    for obj in env.movables + env.statics:
        geo_type, obj_dict = _get_object_dict(obj)
        if geo_type not in geometries:
            geometries[geo_type] = {}
        geometries[geo_type][obj.name] = obj_dict

    env_dict = {
        "name": env.name,
        "geometries": geometries,
        "types": {obj_type: [obj.name for obj in objs] for obj_type, objs in env.type_to_objects.items()},
        "goal": [{goal_atom.fluent.name: goal_atom.values} for goal_atom in env.goal_state],
    }
    return env_dict


def get_env_dir() -> str:
    """Get the directory containing the environment files."""
    return os.path.join(os.path.dirname(__file__), "assets")


def resolve_env_path(arg: str) -> str:
    """``--env``: an existing file as given, else the shipped scene ``arg`` names, with or without ``.yml``."""
    return arg if os.path.isfile(arg) else os.path.join(get_env_dir(), arg.removesuffix(".yml") + ".yml")
