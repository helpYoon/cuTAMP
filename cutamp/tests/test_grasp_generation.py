"""GraspGenX's grasps, made in-process when the world loads (``cutamp.grasps.grasp_sets``): one set per object shape,
bit-identical in every process, and outside the run's random state and numeric settings."""

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from curobo.scene import Cuboid

import cutamp.grasps as g
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.tests.conftest import assert_bitwise, needs_cuda
from cutamp.utils.common import grasp_7dof_to_mat4x4

DEVICE = torch.device("cuda", 0)


def _blocks():
    """blocks_t1's movables: block2 and block3, two 5 cm blocks at different poses."""
    return load_env(os.path.join(get_env_dir(), "blocks_t1.yml")).movables


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


def test_missing_model_names_the_readme(monkeypatch, tmp_path):
    blocks = _blocks()
    # A warm cache: the check runs before the lookup, so it raises even where every shape's grasps exist.
    monkeypatch.setattr(g, "_SETS", {g.mesh_sha256(b): g.GraspSet(torch.zeros(0, 7), torch.zeros(0)) for b in blocks})
    monkeypatch.setenv("GRASPGENX_CHECKPOINT_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("GRASPGENX_GRIPPER_CFG_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "graspgenx", None)  # any import of graspgenx now fails loudly
    with pytest.raises(RuntimeError) as err:
        g.grasp_sets(blocks, "cpu")
    assert "README.md, Installation, the GraspGenX paragraph" in str(err.value)


@needs_cuda
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
def test_generation_repeats_and_keeps_the_run_rng(monkeypatch):
    blocks, made = _blocks(), []
    # Cold, then warm, with both TF32 flags on as cuRobo's import leaves them; then warm with them off.
    for cold, tf32 in ((True, True), (False, True), (False, False)):
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", tf32)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", tf32)
        monkeypatch.setattr(g, "_SETS", {})
        if cold:
            monkeypatch.setattr(g, "_SAMPLER", None)  # the model is built again, drawing from torch's CPU generator
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
def test_fresh_process_matches(tmp_path):
    out = tmp_path / "grasps.pt"
    subprocess.run([sys.executable, "-B", "-c", _CHILD, str(out)], cwd=Path(__file__).resolve().parents[2],
                   check=True, timeout=300)
    grasps, scores = g.grasp_sets(_blocks(), DEVICE)["block2"]
    assert_bitwise({"grasps": grasps.cpu(), "scores": scores.cpu()}, torch.load(out), "this process vs a fresh one",
                   raw=True)
