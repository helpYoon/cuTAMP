"""Runner plan/manifest logic (no GPU: executor stubbed)."""
import json

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
