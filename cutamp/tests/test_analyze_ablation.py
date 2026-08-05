"""Analysis pipeline on synthetic experiment dirs (no GPU)."""
import json
from pathlib import Path

from cutamp.scripts.analyze_ablation import aggregate, load_study


def _fake_run(root: Path, cell, height, seed, sat=100, sat_com=90):
    exp = root / f"ablation_{cell}_h{int(height*100):03d}_s{seed}"
    (exp / "optimization").mkdir(parents=True)
    opt = {
        "num_satisfying_final": sat,
        "num_satisfying_com_feasible": sat_com,
        "best_soft_cost_com_feasible": 20.0 + seed,
        "best_com_margin": 0.05,
        "elapsed": [0.1, 30.0],
        "num_satisfying": [1, sat],
    }
    (exp / "optimization" / "opt_0001.json").write_text(json.dumps(opt))
    (exp / "overall_metrics.json").write_text(json.dumps({
        "traj_com_violation_frac": 0.0 if cell in ("full", "no_reik") else 0.2,
        "traj_com_frames": 500,
    }))
    return {"cell": cell, "height": height, "seed": seed,
            "experiment_id": exp.name, "status": "done",
            "experiment_path": str(exp)}


def test_load_and_aggregate(tmp_path):
    manifest = [
        _fake_run(tmp_path, c, 0.0, s)
        for c in ("full", "standard") for s in (0, 1)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    records = load_study(tmp_path / "manifest.json")
    assert len(records) == 4
    assert records[0]["num_satisfying_com_feasible"] == 90
    agg = aggregate(records)
    mean, std = agg[("full", 0.0)]["best_soft_cost_com_feasible"]
    assert mean == 20.5 and round(std, 2) == 0.71  # seeds 0,1 -> 20,21


def test_main_writes_outputs(tmp_path, monkeypatch):
    manifest = [
        _fake_run(tmp_path, c, h, s)
        for c in ("full", "no_com", "no_reik", "standard")
        for h in (0.0, 0.5) for s in (0, 1)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    import sys
    from cutamp.scripts import analyze_ablation
    monkeypatch.setattr(sys, "argv", [
        "analyze_ablation", "--manifest", str(tmp_path / "manifest.json"),
        "--out", str(tmp_path / "out"),
    ])
    analyze_ablation.main()
    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "results.md").exists()
    figs = list((tmp_path / "out" / "figures").glob("*.png"))
    assert len(figs) >= 3
