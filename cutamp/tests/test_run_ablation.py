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
