"""Scenes: unit quaternions, mesh paths and I/O, suffix-matched tolerances, --env, floor and table tops."""
import os

import numpy as np
import yaml

from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import (
    FLOOR,
    TABLE_MARGIN,
    TABLE_THICKNESS,
    get_env_dict,
    get_env_dir,
    load_env,
    raise_surface_height,
    resolve_env_path,
)
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.task_planning.constraints import StablePlacement

BLOCKS = os.path.join(get_env_dir(), "blocks_t1.yml")


def test_quaternions_are_unit_after_load():
    env = load_env(BLOCKS)
    block2 = next(o for o in env.movables if o.name == "block2")
    q = np.array(block2.pose[3:])
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    raw = np.array([0.03, 0.0, 0.0, 0.99])
    assert np.allclose(q, raw / np.linalg.norm(raw), atol=1e-12)


def _mesh_env(folder, file_path, color=(10, 20, 30)):
    env_dict = {
        "name": "mesh_env",
        "geometries": {
            "cuboid": {"box": {"dims": [0.05, 0.05, 0.05], "pose": [0.4, 0, 0.031, 1, 0, 0, 0]}},
            "mesh": {"cup": {"file_path": file_path, "pose": [0.5, 0.1, 0.0, 1, 0, 0, 0], "color": list(color)}},
        },
        "types": {"Movable": ["box"], "Surface": ["cup"]},
        "goal": [{"On": ["box", "cup"]}],
    }
    path = folder / "mesh_env.yml"
    path.write_text(yaml.safe_dump(env_dict))
    return str(path)


def test_mesh_path_resolves_from_any_cwd(tmp_path, monkeypatch):
    (tmp_path / "cup.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    path = _mesh_env(tmp_path, "cup.obj")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    env = load_env(path)
    mesh = next(o for o in env.statics if o.name == "cup")
    assert os.path.isabs(mesh.file_path)
    assert os.path.exists(mesh.file_path)


def test_mesh_env_round_trip(tmp_path):
    (tmp_path / "cup.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    env = load_env(_mesh_env(tmp_path, "cup.obj"))
    before = next(o for o in env.statics if o.name == "cup")
    out = tmp_path / "round.yml"
    out.write_text(yaml.safe_dump(yaml.safe_load(yaml.safe_dump(get_env_dict(env)))))
    after = next(o for o in load_env(str(out)).statics if o.name == "cup")
    assert after.file_path == before.file_path
    assert list(after.pose) == list(before.pose)
    assert list(after.color) == list(before.color)


def test_suffix_tolerances():
    checker = ConstraintChecker(default_constraint_to_tol.copy())
    reducer = CostReducer(default_constraint_to_mult)
    for surface in ("shelf", "goal"):
        assert checker._get_tol(StablePlacement.type, f"{surface}_in_xy") == 1e-3
        assert checker._get_tol(StablePlacement.type, f"{surface}_support") == 1e-2
        assert reducer._get_multiplier(StablePlacement.type, f"{surface}_support") == 2.0


def test_resolve_env_path(tmp_path):
    assert resolve_env_path("blocks_t1") == BLOCKS
    assert resolve_env_path(BLOCKS) == BLOCKS
    other = tmp_path / "x.yml"
    other.write_text("")
    assert resolve_env_path(str(other)) == str(other)


def test_floor_is_in_the_env_file():
    env = load_env(BLOCKS)
    floor = next(o for o in env.statics if o.name == "floor")
    assert list(floor.dims) == FLOOR["dims"]
    assert list(floor.pose) == FLOOR["pose"]


def test_raise_adds_a_table_top():
    env = load_env(BLOCKS)
    raise_surface_height(env, 0.5)
    table = next(o for o in env.statics if o.name == "table")
    floor = next(o for o in env.statics if o.name == "floor")
    assert floor.pose[2] == FLOOR["pose"][2]
    assert abs(table.pose[2] + table.dims[2] / 2 - 0.5) < 1e-9
    assert abs(table.dims[2] - TABLE_THICKNESS) < 1e-12
    raised = list(env.movables) + list(env.type_to_objects["Surface"])
    for obj in raised:
        # the footprint covers the object's rotated xy box, grown by TABLE_MARGIN
        w, z = obj.pose[3], obj.pose[6]
        yaw = 2 * np.arctan2(z, w)
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        hx = (c * obj.dims[0] + s * obj.dims[1]) / 2
        hy = (s * obj.dims[0] + c * obj.dims[1]) / 2
        assert table.pose[0] - table.dims[0] / 2 <= obj.pose[0] - hx - TABLE_MARGIN + 1e-9
        assert table.pose[0] + table.dims[0] / 2 >= obj.pose[0] + hx + TABLE_MARGIN - 1e-9
        assert table.pose[1] - table.dims[1] / 2 <= obj.pose[1] - hy - TABLE_MARGIN + 1e-9
        assert table.pose[1] + table.dims[1] / 2 >= obj.pose[1] + hy + TABLE_MARGIN - 1e-9
    assert "table" not in [o.name for o in env.type_to_objects.get("Surface", [])]


def test_no_table_at_height_zero():
    env = load_env(BLOCKS)
    raise_surface_height(env, 0.0)
    assert "table" not in [o.name for o in env.statics]
