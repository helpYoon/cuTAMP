"""GraspGenX's grasps, made in-process when the world loads (``cutamp.grasps.grasp_sets``): one set per object shape,
bit-identical in every process, and outside the run's random state and numeric settings."""

import importlib.util
import os
import platform
import random
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import pytest
import torch
from curobo.scene import Cuboid

import cutamp.grasps as g
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.tests.conftest import assert_bitwise, needs_cuda, needs_graspgenx
from cutamp.utils.common import grasp_7dof_to_mat4x4

DEVICE = torch.device("cuda", 0)
README = "README.md, Installation, the GraspGenX paragraph"


def _blocks():
    """blocks_t1's movables: block2 and block3, two 5 cm blocks at different poses."""
    return load_env(os.path.join(get_env_dir(), "blocks_t1.yml")).movables


def _graspgenx_modules() -> set:
    """The graspgenx modules this process has imported."""
    return {m for m, mod in list(sys.modules.items()) if m.split(".")[0] == "graspgenx" and mod is not None}


def _trap_graspgenx(monkeypatch) -> set:
    """Any import of graspgenx now fails loudly. Returns the graspgenx modules imported so far, which the test must
    leave as they are."""
    monkeypatch.setitem(sys.modules, "graspgenx", None)
    return _graspgenx_modules()


def _graspgenx_installed(monkeypatch, installed: bool) -> None:
    """``importlib.util.find_spec`` reports graspgenx as installed or not; every other name it looks up as before."""
    find_spec, spec = importlib.util.find_spec, ModuleSpec("graspgenx", None) if installed else None
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, package=None: spec if name == "graspgenx" else find_spec(name, package))


def _readme_folders(monkeypatch, tmp_path, made=("model", "grippers")) -> dict:
    """``_README_FOLDERS`` at ``~/model`` and ``~/grippers``, with HOME at ``tmp_path``; makes the folders named in
    ``made`` (the model with both configs) and empties both variables. Returns each variable's absolute folder."""
    names = {"GRASPGENX_CHECKPOINT_DIR": "model", "GRASPGENX_GRIPPER_CFG_DIR": "grippers"}
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(g, "_README_FOLDERS", {var: f"~/{name}" for var, name in names.items()})
    if "model" in made:
        for side in ("gen", "dis"):
            (tmp_path / "model" / "release" / side).mkdir(parents=True)
            (tmp_path / "model" / "release" / side / "config.yaml").touch()
    if "grippers" in made:
        (tmp_path / "grippers").mkdir()
    for var in names:  # _checked_model_dir writes os.environ itself: monkeypatch restores what it set
        monkeypatch.setenv(var, "")
    return {var: str(tmp_path / name) for var, name in names.items()}


def _rng_states():
    """Python's, numpy's, torch's CPU and every CUDA generator's state, comparable with ``==``."""
    kind, keys, *rest = np.random.get_state()
    return (random.getstate(), (kind, keys.tobytes(), *rest), torch.get_rng_state().numpy().tobytes(),
            [s.numpy().tobytes() for s in torch.cuda.get_rng_state_all()])


def _flags():
    """The two TF32 flags and torch's deterministic mode, with its warn-only flag."""
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


def test_mesh_hash_is_shared_by_shape():
    blocks = {b.name: b for b in _blocks()}
    assert g.mesh_sha256(blocks["block2"]) == g.mesh_sha256(blocks["block3"])  # same shape, different poses
    big = Cuboid(name="big", pose=list(blocks["block2"].pose), dims=[0.06, 0.06, 0.06])
    assert g.mesh_sha256(big) != g.mesh_sha256(blocks["block2"])


def test_missing_graspgenx_names_the_platform(monkeypatch, tmp_path):
    # GraspGenX installs on x86_64 only: elsewhere setup stops on this, before the folder check (which would fail
    # here too) and before anything imports graspgenx.
    blocks = _blocks()
    monkeypatch.setenv("GRASPGENX_CHECKPOINT_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("GRASPGENX_GRIPPER_CFG_DIR", str(tmp_path))
    before = _trap_graspgenx(monkeypatch)
    _graspgenx_installed(monkeypatch, False)
    with pytest.raises(RuntimeError) as err:
        g.grasp_sets(blocks, "cpu")
    message = str(err.value)
    assert "GraspGenX is not installed, and planning needs it" in message
    assert f"x86_64 only, and this machine is {platform.machine()}" in message and README in message
    assert _graspgenx_modules() == before


def test_missing_model_names_the_readme(monkeypatch, tmp_path):
    blocks = _blocks()
    # A warm cache: the check runs before the lookup, so it raises even where every shape's grasps exist.
    monkeypatch.setattr(g, "_SETS", {g.mesh_sha256(b): g.GraspSet(torch.zeros(0, 7), torch.zeros(0)) for b in blocks})
    monkeypatch.setenv("GRASPGENX_CHECKPOINT_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("GRASPGENX_GRIPPER_CFG_DIR", str(tmp_path))
    before = _trap_graspgenx(monkeypatch)
    _graspgenx_installed(monkeypatch, True)  # past the install check, on any machine
    with pytest.raises(RuntimeError) as err:
        g.grasp_sets(blocks, "cpu")
    assert "model folders are missing" in str(err.value) and README in str(err.value)
    assert _graspgenx_modules() == before


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_setup_check_takes_the_readme_folders(monkeypatch, tmp_path, value):
    # Unset or empty, each variable gets its README folder, made absolute: load_model_cfg doesn't expand "~".
    folders = _readme_folders(monkeypatch, tmp_path)
    if value is None:
        for var in folders:
            monkeypatch.delenv(var)
    before = _trap_graspgenx(monkeypatch)
    ckpt = g._checked_model_dir()
    assert os.path.isabs(ckpt) and ckpt == folders["GRASPGENX_CHECKPOINT_DIR"]
    assert {var: os.environ[var] for var in folders} == folders
    assert _graspgenx_modules() == before


def test_setup_check_names_the_readme_without_a_readme_folder(monkeypatch, tmp_path):
    _readme_folders(monkeypatch, tmp_path, made={"model"})
    before = _trap_graspgenx(monkeypatch)
    with pytest.raises(RuntimeError, match=README):
        g._checked_model_dir()
    assert _graspgenx_modules() == before


@needs_cuda
@needs_graspgenx
def test_block_grasps(monkeypatch):
    blocks = _blocks()
    g.grasp_sets(blocks, DEVICE)  # imports graspgenx, if this process hasn't yet
    planner = sys.modules["graspgenx.samplers.planner"]
    run, tags = planner.run_planner_on_object, []

    def spy(*args, **kwargs):
        out = run(*args, **kwargs)
        tags.append(out[2])  # "diff" or "obb" per grasp
        return out

    monkeypatch.setattr(planner, "run_planner_on_object", spy)
    monkeypatch.setattr(g, "_SETS", {})
    sets = g.grasp_sets(blocks, DEVICE)
    assert sets["block2"] is sets["block3"] and len(tags) == 1  # one shape, one generation
    assert (len(tags[0]), tags[0].count("diff"), tags[0].count("obb")) == (4100, 500, 3600)
    grasps, scores = sets["block2"]
    assert grasps.shape == (4100, 7) and scores.shape == (4100,)
    assert grasps.dtype == scores.dtype == torch.float32 and grasps.device == scores.device == DEVICE
    assert ((grasps[:, 3:].norm(dim=-1) - 1).abs() <= 1e-6).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    approach_up = grasp_7dof_to_mat4x4(grasps)[:, 2, 2]  # Z_G . z_obj
    assert (approach_up.abs() < 0.1).any()


@needs_cuda
@needs_graspgenx
def test_generation_repeats_and_keeps_the_run_rng(monkeypatch):
    blocks, made = _blocks(), []
    # Cold, then warm, with both TF32 flags on as cuRobo's import leaves them; then warm with them off.
    for cold, tf32 in ((True, True), (False, True), (False, False)):
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", tf32)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", tf32)
        monkeypatch.setattr(g, "_SETS", {})
        if cold:
            g._load_sampler.cache_clear()  # the model is built again, drawing from torch's CPU generator
        rng, flags = _rng_states(), _flags()
        made.append(g.grasp_sets(blocks, DEVICE)["block2"])
        assert _rng_states() == rng, f"cold={cold} tf32={tf32}: a generator's state changed"
        assert _flags() == flags, f"cold={cold} tf32={tf32}: {_flags()} != {flags}"
    assert_bitwise(made[1], made[0], "warm vs cold", raw=True)
    assert_bitwise(made[2], made[0], "TF32 off vs on", raw=True)


# A fresh process: cwd is this tree's root, so ``-c`` imports this tree's cutamp. Its root logger is untouched.
_CHILD = """
import logging, os, sys
import torch
import cutamp
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.grasps import grasp_sets
assert os.path.realpath(cutamp.__file__).startswith(os.getcwd() + os.sep), cutamp.__file__
blocks = load_env(os.path.join(get_env_dir(), "blocks_t1.yml")).movables
root = logging.getLogger()
before = root.level, list(root.handlers)
grasps, scores = grasp_sets(blocks, torch.device("cuda", 0))["block2"]
assert (root.level, list(root.handlers)) == before, (root.level, root.handlers)
torch.save({"grasps": grasps.cpu(), "scores": scores.cpu()}, sys.argv[1])
"""


@needs_cuda
@needs_graspgenx
def test_fresh_process_matches(tmp_path):
    out = tmp_path / "grasps.pt"
    subprocess.run([sys.executable, "-B", "-c", _CHILD, str(out)], cwd=Path(__file__).resolve().parents[2],
                   check=True, timeout=300)
    grasps, scores = g.grasp_sets(_blocks(), DEVICE)["block2"]
    assert_bitwise({"grasps": grasps.cpu(), "scores": scores.cpu()}, torch.load(out), "this process vs a fresh one",
                   raw=True)
