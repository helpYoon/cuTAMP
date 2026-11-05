"""Runner plan/manifest logic (no GPU: executor stubbed)."""
import json
import subprocess
from pathlib import Path

from cutamp.scripts import run_ablation
from cutamp.scripts.run_ablation import CELLS, GATE_CELLS, build_gate_plan, build_run_plan, run_study


def test_cells_run_their_refresh(monkeypatch):
    # Each run's argv through run_cutamp's own CLI defaults (balance-first and graphs on): only
    # full and no_com may run the balance-first refresh, and default_reik keeps the default one.
    import sys

    import cutamp.scripts.run_cutamp as rc
    from cutamp.config import uses_cuda_graphs
    seen = []
    monkeypatch.setattr(rc, "cutamp_demo", lambda env, config, **kw: seen.append(config))
    want = {"full": (True, True), "default_reik": (False, False), "no_com": (True, True)}
    for run in build_run_plan(heights=[0.0], seeds=[0]):
        if run["cell"] in want:
            monkeypatch.setattr(sys, "argv", ["run_cutamp", *run["args"]])
            rc.entrypoint()
            c = seen[-1]
            assert c.coupled_reik and (c.reik_balance_first, uses_cuda_graphs(c)) == want[run["cell"]], run["cell"]


def test_cells_match_spec():
    assert set(CELLS) == {"full", "default_reik", "no_com", "no_reik", "standard"}
    for cell in ("full", "no_com"):
        assert "--reik_balance_first" in CELLS[cell] and "--cuda_graphs" in CELLS[cell]
    assert "--no_com_all" in CELLS["no_com"]
    for cell in ("no_reik", "standard"):
        assert "--no_coupled_reik" in CELLS[cell]
        assert "--force_adam" in CELLS[cell]
        assert "--soft_in_adam" in CELLS[cell] and "always" in CELLS[cell]
    assert "--no_com_all" in CELLS["standard"]


def test_build_run_plan_grid_and_ids():
    plan = build_run_plan(heights=[0.0, 0.5], seeds=[0, 1])
    assert len(plan) == len(CELLS) * 2 * 2
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
    assert len(calls) == len(CELLS)
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
    assert len(calls) == len(CELLS)
    manifest = json.load(open(tmp_path / "manifest.json"))

    # All but no_com done, no_com failed
    done_runs = [r for r in manifest if r["status"] == "done"]
    failed_runs = [r for r in manifest if r["status"].startswith("failed")]
    assert len(done_runs) == len(CELLS) - 1
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
    assert len(manifest) == len(CELLS)


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


def test_gate_plan_interleaves(tmp_path):
    # Design section 8's speed gate: 4 heights x 10 seeds x n (256, 512) x both models, in that loop order, so the
    # two models alternate through one session; --seed and PYTHONHASHSEED make every run replayable.
    plan = build_gate_plan()
    assert len(plan) == 160
    assert [(r["height"], r["seed"], r["n"], r["cell"]) for r in plan[:4]] == [
        (0.0, 0, 256, "simplified"), (0.0, 0, 256, "real_legs"),
        (0.0, 0, 512, "simplified"), (0.0, 0, 512, "real_legs"),
    ]
    assert len({r["experiment_id"] for r in plan}) == 160
    for r in plan:
        assert r["env"] == {"PYTHONHASHSEED": str(r["seed"])}
        assert r["experiment_id"] == f"gate_{r['cell']}_n{r['n']}_h{int(r['height'] * 100):03d}_s{r['seed']}"
        assert r["args"] == ["-n", str(r["n"]), "--disable_visualizer", "--motion_plan", "--save_plan", "auto",
                             "--surface_height", str(r["height"]), "--seed", str(r["seed"]),
                             "--experiment_id", r["experiment_id"], *GATE_CELLS[r["cell"]]]
    assert [(r["height"], r["seed"]) for r in build_gate_plan(heights=[0.5], seeds=[3])] == [(0.5, 3)] * 4
    # Each manifest row gains n, for the per-n gate.
    manifest = run_study(plan[:4], out_dir=tmp_path, execute=lambda run: 0)
    assert [(row["cell"], row["n"]) for row in manifest] == [
        ("simplified", 256), ("real_legs", 256), ("simplified", 512), ("real_legs", 512),
    ]


def test_gate_runs_reach_run_cutamp(monkeypatch, tmp_path):
    # Each gate run's argv through run_cutamp's own CLI: its model, n and the section's soft costs (the real legs add
    # their own default, minimize_body_movement), a motion plan saved per run; the subprocess sets its hash seed.
    import sys

    import cutamp.scripts.run_cutamp as rc
    seen = []
    monkeypatch.setattr(rc, "cutamp_demo", lambda env, config, **kw: seen.append((config, kw)))
    soft = {"simplified": ["joint_limit_margin"], "real_legs": ["joint_limit_margin", "minimize_body_movement"]}
    # Seed 0: --seed sets torch's global seed, and test_cells_run_their_refresh already leaves it at 0.
    for run in build_gate_plan(heights=[0.5], seeds=[0]):
        monkeypatch.setattr(sys, "argv", ["run_cutamp", *run["args"]])
        rc.entrypoint()
        config, kw = seen[-1]
        assert (config.robot, config.num_particles, config.curobo_plan, config.enable_visualizer) == (
            run["cell"], run["n"], True, False)
        assert config.soft_cost == soft[run["cell"]]
        assert kw == {"experiment_id": run["experiment_id"], "save_plan": "auto"}

    captured = {}

    def fake_call(cmd, env=None):
        captured.update(cmd=cmd, env=env)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    assert run_ablation._subprocess_exec(build_gate_plan(heights=[0.5], seeds=[3])[1], tmp_path) == 0
    assert captured["env"]["PYTHONHASHSEED"] == "3" and captured["env"]["PYTHONPATH"] == ""
    assert captured["cmd"][-2:] == ["--experiment_root", str(tmp_path)]
