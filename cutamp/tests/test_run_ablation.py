"""Runner plan/manifest logic (no GPU: executor stubbed)."""
import json
import subprocess
from pathlib import Path

from cutamp.scripts import run_ablation
from cutamp.scripts.run_ablation import CELLS, build_run_plan, run_study


def test_cells_match_spec():
    assert set(CELLS) == {"full", "no_com", "no_reik", "standard"}
    assert CELLS["full"] == []
    assert "--no_com_all" in CELLS["no_com"]
    for cell in ("no_reik", "standard"):
        assert "--no_coupled_reik" in CELLS[cell]
        assert "--force_adam" in CELLS[cell]
        assert "--soft_in_adam" in CELLS[cell] and "always" in CELLS[cell]
    assert "--no_com_all" in CELLS["standard"]


def test_build_run_plan_grid_and_ids():
    plan = build_run_plan(heights=[0.0, 0.5], seeds=[0, 1])
    assert len(plan) == 4 * 2 * 2
    ids = [r["experiment_id"] for r in plan]
    assert len(set(ids)) == len(ids)  # unique
    r = plan[0]
    assert {"cell", "height", "seed", "experiment_id", "args"} <= set(r)
    assert f"--seed" in r["args"] and "--surface_height" in r["args"]


def test_run_study_is_resumable(tmp_path):
    calls = []
    def fake_exec(run):  # stub executor: pretend every run succeeds
        calls.append(run["experiment_id"])
        return 0
    plan = build_run_plan(heights=[0.0], seeds=[0])
    run_study(plan, out_dir=tmp_path, execute=fake_exec)
    assert len(calls) == 4
    manifest = json.load(open(tmp_path / "manifest.json"))
    assert all(r["status"] == "done" for r in manifest)
    # Second invocation: nothing re-runs.
    calls.clear()
    run_study(plan, out_dir=tmp_path, execute=fake_exec)
    assert calls == []


def test_run_study_retries_failed_runs(tmp_path):
    calls = []

    def failing_exec(run):
        calls.append(run["experiment_id"])
        # Fail only the second run (no_com cell)
        return 0 if run["cell"] != "no_com" else 42

    plan = build_run_plan(heights=[0.0], seeds=[0])
    run_study(plan, out_dir=tmp_path, execute=failing_exec)
    assert len(calls) == 4
    manifest = json.load(open(tmp_path / "manifest.json"))

    # Three done, one failed
    done_runs = [r for r in manifest if r["status"] == "done"]
    failed_runs = [r for r in manifest if r["status"].startswith("failed")]
    assert len(done_runs) == 3
    assert len(failed_runs) == 1
    assert failed_runs[0]["experiment_id"] == "ablation_no_com_h000_s0"
    assert failed_runs[0]["status"] == "failed(42)"

    # Now retry with a succeeding executor
    calls.clear()

    def succeeding_exec(run):
        calls.append(run["experiment_id"])
        return 0

    run_study(plan, out_dir=tmp_path, execute=succeeding_exec)

    # Only the failed run should be retried
    assert calls == ["ablation_no_com_h000_s0"]

    manifest = json.load(open(tmp_path / "manifest.json"))
    assert all(r["status"] == "done" for r in manifest)
    assert len(manifest) == 4


def test_experiment_root_defaults_under_out_dir(tmp_path):
    """Each study is self-contained under its own --out dir by default,
    so a --smoke run against a fresh --out can never collide with a real
    study's experiment dirs (see run_study docstring)."""
    plan = build_run_plan(heights=[0.0], seeds=[0])

    def fake_exec(run):
        return 0

    manifest = run_study(plan, out_dir=tmp_path, execute=fake_exec)
    expected_root = tmp_path / "experiments"
    assert manifest  # sanity: plan produced runs
    for r in manifest:
        assert r["experiment_path"] == str(expected_root / r["experiment_id"])


def test_experiment_root_isolated_across_studies(tmp_path):
    """Two studies (e.g. a real run and a later --smoke pilot) under
    different --out dirs never share an experiment_root."""
    plan = build_run_plan(heights=[0.0], seeds=[0])

    def fake_exec(run):
        return 0

    out_a, out_b = tmp_path / "study_a", tmp_path / "study_b"
    manifest_a = run_study(plan, out_dir=out_a, execute=fake_exec)
    manifest_b = run_study(plan, out_dir=out_b, execute=fake_exec)
    roots_a = {Path(r["experiment_path"]).parent for r in manifest_a}
    roots_b = {Path(r["experiment_path"]).parent for r in manifest_b}
    assert roots_a == {out_a / "experiments"}
    assert roots_b == {out_b / "experiments"}
    assert roots_a.isdisjoint(roots_b)


def test_subprocess_exec_forwards_experiment_root(monkeypatch, tmp_path):
    """_subprocess_exec must pass --experiment_root through to
    run_cutamp (the CLI flag already exists, run_cutamp.py:240)."""
    captured = {}

    def fake_call(cmd, env=None):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    run = build_run_plan(heights=[0.0], seeds=[0])[0]
    experiment_root = tmp_path / "experiments"
    code = run_ablation._subprocess_exec(run, experiment_root)
    assert code == 0
    cmd = captured["cmd"]
    assert "--experiment_root" in cmd
    idx = cmd.index("--experiment_root")
    assert cmd[idx + 1] == str(experiment_root)


def test_run_study_default_execute_forwards_experiment_root(monkeypatch, tmp_path):
    """End-to-end (executor still stubbed at subprocess.call): the
    default execute path threads out_dir/experiments into the actual
    run_cutamp command line."""
    captured_cmds = []

    def fake_call(cmd, env=None):
        captured_cmds.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    plan = build_run_plan(heights=[0.0], seeds=[0])
    manifest = run_study(plan, out_dir=tmp_path)
    expected_root = str(tmp_path / "experiments")
    assert len(captured_cmds) == len(plan)
    for cmd in captured_cmds:
        idx = cmd.index("--experiment_root")
        assert cmd[idx + 1] == expected_root
    for r in manifest:
        assert r["experiment_path"].startswith(expected_root)
