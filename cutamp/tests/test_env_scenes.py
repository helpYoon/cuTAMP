"""Scenes: unit quaternions, mesh paths and I/O, suffix-matched tolerances, --env, floor and table tops."""
import os

import numpy as np
import pytest
import yaml

from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import (
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


@pytest.mark.parametrize("flags, expected", [([], 641), (["--seed", "0"], 0)])
def test_cli_uses_scene_seed_unless_overridden(monkeypatch, flags, expected):
    import sys
    import torch
    from cutamp.scripts import run_cutamp as cli

    seeds = []
    monkeypatch.setattr(sys, "argv", ["run_cutamp", "--env", "two_steps_floor_transfer", *flags])
    monkeypatch.setattr(torch, "manual_seed", seeds.append)
    monkeypatch.setattr(cli, "cutamp_demo", lambda *args, **kwargs: None)
    cli.entrypoint()
    assert seeds == [expected]


def test_quaternions_are_unit_after_load():
    env = load_env(BLOCKS)
    block2 = next(o for o in env.movables if o.name == "block2")
    q = np.array(block2.pose[3:])
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    raw = np.array([0.03, 0.0, 0.0, 0.99])
    assert np.allclose(q, raw / np.linalg.norm(raw), atol=1e-12)


def test_mesh_path_resolves_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh = next(o for o in load_env(resolve_env_path("cup_t1")).movables if o.name == "cup")
    assert os.path.isabs(mesh.file_path)
    assert os.path.exists(mesh.file_path)


def test_mesh_env_round_trip(tmp_path):
    env = load_env(resolve_env_path("cup_t1"))
    before = next(o for o in env.movables if o.name == "cup")
    out = tmp_path / "round.yml"
    out.write_text(yaml.safe_dump(get_env_dict(env)))
    after = next(o for o in load_env(str(out)).movables if o.name == "cup")
    assert after.file_path == before.file_path
    assert list(after.pose) == list(before.pose)
    assert list(after.color) == list(before.color)


@pytest.mark.parametrize("name, start, seed", [
    ("blocks_t1", "home", None), ("two_steps_floor_transfer", "synthetic_standing", 641),
])
def test_scene_start_and_demo_seed_round_trip(tmp_path, name, start, seed):
    env = load_env(resolve_env_path(name))
    assert (env.robot_start, env.demo_seed) == (start, seed)
    out = tmp_path / "replay.yml"
    out.write_text(yaml.safe_dump(get_env_dict(env)))
    replay = load_env(str(out))
    assert (replay.robot_start, replay.demo_seed) == (start, seed)


def test_unknown_robot_start_rejected(tmp_path):
    scene = get_env_dict(load_env(BLOCKS))
    scene["robot_start"] = "standng"
    out = tmp_path / "invalid.yml"
    out.write_text(yaml.safe_dump(scene))
    with pytest.raises(ValueError, match="robot_start"):
        load_env(str(out))


@pytest.mark.parametrize("seed", [True, 1.5, -(2**63) - 1, 2**64])
def test_invalid_demo_seed_rejected(tmp_path, seed):
    scene = get_env_dict(load_env(BLOCKS))
    scene["demo_seed"] = seed
    out = tmp_path / "invalid.yml"
    out.write_text(yaml.safe_dump(scene))
    with pytest.raises(ValueError, match="demo_seed"):
        load_env(str(out))


def test_suffix_tolerances():
    checker = ConstraintChecker(default_constraint_to_tol.copy())
    reducer = CostReducer(default_constraint_to_mult)
    for surface in ("shelf", "goal"):
        assert checker._get_tol(StablePlacement.type, f"{surface}_in_xy") == 1e-3
        assert checker._get_tol(StablePlacement.type, f"{surface}_support") == 1e-2
        assert reducer._get_multiplier(StablePlacement.type, f"{surface}_support") == 2.0


def test_resolve_env_path(tmp_path):
    other = tmp_path / "x.yml"
    other.write_text("")
    assert resolve_env_path("blocks_t1") == resolve_env_path("blocks_t1.yml") == resolve_env_path(BLOCKS) == BLOCKS
    assert resolve_env_path(str(other)) == str(other)
    for arg in (str(tmp_path), "shelf_t2", str(tmp_path / "shelf_t1.yml")):  # a folder, an unknown name, no file
        with pytest.raises(FileNotFoundError, match="Environment file not found"):
            load_env(resolve_env_path(arg))


def test_floor_is_in_the_env_file():
    env = load_env(BLOCKS)
    floor = next(o for o in env.statics if o.name == "floor")
    assert floor.pose[2] + floor.dims[2] / 2 == 0  # its top at z = 0


def test_raise_adds_a_table_top():
    env = load_env(BLOCKS)
    raise_surface_height(env, 0.5)
    table = next(o for o in env.statics if o.name == "table")
    floor = next(o for o in env.statics if o.name == "floor")
    assert floor.pose[2] == -0.05
    assert abs(table.pose[2] + table.dims[2] / 2 - 0.5) < 1e-9
    assert abs(table.dims[2] - TABLE_THICKNESS) < 1e-12
    lo, hi = table.get_trimesh_mesh(transform_with_pose=True).bounds[:, :2]
    for obj in list(env.movables) + list(env.type_to_objects["Surface"]):  # the footprint covers each rotated xy box
        box = obj.get_trimesh_mesh(transform_with_pose=True).bounds[:, :2]
        assert (lo <= box[0] + 1e-9).all() and (box[1] <= hi + 1e-9).all(), obj.name
    assert "table" not in [o.name for o in env.type_to_objects.get("Surface", [])]


def test_no_table_at_height_zero():
    env = load_env(BLOCKS)
    raise_surface_height(env, 0.0)
    assert "table" not in [o.name for o in env.statics]


def _poses(env) -> dict:
    return {o.name: list(o.pose) for o in env.movables + env.statics}


def test_raise_refuses_statics_it_would_leave_behind():
    # At a nonzero height only the floor and the Surfaces may be static: shelf_t1's boards would stay below the box.
    env = load_env(resolve_env_path("shelf_t1"))
    before = _poses(env)
    with pytest.raises(ValueError, match="shelf_t1's statics lower_board, upper_board would stay put"):
        raise_surface_height(env, 0.5)
    assert _poses(env) == before  # nothing moved, and no table top


def test_raise_refuses_an_object_named_table():
    env = load_env(BLOCKS)
    next(o for o in env.statics if o.name == "goal").name = "table"  # a Surface: the height check passes it
    before = _poses(env)
    with pytest.raises(ValueError, match="blocks_t1 already has an object named 'table'"):
        raise_surface_height(env, 0.5)
    assert _poses(env) == before
