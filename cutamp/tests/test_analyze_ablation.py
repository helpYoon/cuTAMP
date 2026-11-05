"""Analysis pipeline on synthetic experiment dirs (no GPU, except the real-leg report's test, which runs cuRobo)."""
import json
import statistics
from pathlib import Path

import pytest

from cutamp.scripts.analyze_ablation import aggregate, aggregate_deltas, compute_deltas, gate_summary, load_study
from cutamp.tests.conftest import needs_cuda


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


def test_missing_overall_metrics_fields(tmp_path):
    """Test that load_study handles absence of traj_com_violation_frac and motion_plan_success."""
    exp = tmp_path / "ablation_test_h000_s0"
    (exp / "optimization").mkdir(parents=True)
    opt = {
        "num_satisfying_final": 100,
        "num_satisfying_com_feasible": 90,
        "best_soft_cost_com_feasible": 20.0,
        "best_com_margin": 0.05,
        "elapsed": [0.1, 30.0],
    }
    (exp / "optimization" / "opt_0001.json").write_text(json.dumps(opt))
    # overall_metrics.json lacks both traj_com_violation_frac and motion_plan_success
    (exp / "overall_metrics.json").write_text(json.dumps({"traj_com_frames": 500}))
    manifest = [{
        "cell": "test",
        "height": 0.0,
        "seed": 0,
        "experiment_id": exp.name,
        "status": "done",
        "experiment_path": str(exp),
    }]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    records = load_study(tmp_path / "manifest.json")
    assert len(records) == 1
    assert records[0]["traj_com_violation_frac"] is None
    assert records[0]["motion_plan_success"] is None
    # aggregate should exclude None values
    agg = aggregate(records)
    assert ("test", 0.0) in agg
    assert "traj_com_violation_frac" not in agg[("test", 0.0)]
    assert "motion_plan_success" not in agg[("test", 0.0)]


def test_skip_non_done_status(tmp_path):
    """Test that load_study skips runs with non-done status."""
    # Create a "done" run
    done_run = _fake_run(tmp_path, "full", 0.0, 0)
    # Create a "failed" run (experiment dir doesn't need to exist)
    failed_run = {
        "cell": "full",
        "height": 0.0,
        "seed": 1,
        "experiment_id": "ablation_full_h000_s1",
        "status": "failed(1)",
        "experiment_path": str(tmp_path / "ablation_full_h000_s1"),
    }
    manifest = [done_run, failed_run]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    records = load_study(tmp_path / "manifest.json")
    # Only the "done" run should be in records
    assert len(records) == 1
    assert records[0]["seed"] == 0


def test_main_writes_outputs(tmp_path, monkeypatch):
    manifest = [
        _fake_run(tmp_path, c, h, s)
        for c in ("full", "default_reik", "no_com", "no_reik", "standard")
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
    assert (tmp_path / "out" / "deltas.csv").exists()
    assert (tmp_path / "out" / "results.md").exists()
    md = (tmp_path / "out" / "results.md").read_text()
    assert "Per-seed deltas vs. `full`" in md
    figs = list((tmp_path / "out" / "figures").glob("*.png"))
    assert len(figs) >= 3


def test_motion_plan_success_false_counts_in_aggregate(tmp_path):
    """A curobo_plan run that never reached motion planning now writes
    motion_plan_success=False (not an absent key, see algorithm.py's
    overall_metrics init); the analyzer must count it in the per-cell
    mean instead of silently dropping it from the denominator. Also
    covers found_solution, which is written to overall_metrics.json in
    every run but wasn't previously consumed by the analyzer."""
    exp = tmp_path / "ablation_full_h000_s0"
    (exp / "optimization").mkdir(parents=True)
    (exp / "optimization" / "opt_0001.json").write_text(json.dumps({
        "num_satisfying_final": 0, "elapsed": [0.1], "num_steps": 30,
    }))
    (exp / "overall_metrics.json").write_text(json.dumps({
        "motion_plan_success": False, "found_solution": False,
    }))
    manifest = [{"cell": "full", "height": 0.0, "seed": 0,
                 "experiment_id": exp.name, "status": "done",
                 "experiment_path": str(exp)}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    records = load_study(tmp_path / "manifest.json")
    assert records[0]["motion_plan_success"] == 0.0
    assert records[0]["found_solution"] == 0.0
    agg = aggregate(records)
    assert agg[("full", 0.0)]["motion_plan_success"] == (0.0, 0.0)
    assert agg[("full", 0.0)]["found_solution"] == (0.0, 0.0)


def test_loop_s_sums_across_opt_files(tmp_path):
    """Multi-skeleton run: two abandoned plan skeletons before the one
    that finally succeeds, each writing its own opt_*.json. loop_s must
    be the run's total optimization wall-clock (sum of each file's final
    `elapsed`), not just the last skeleton's — previously it silently
    under-reported loop_s (and thus ms/step) on any multi-skeleton run."""
    exp = tmp_path / "ablation_full_h000_s0"
    (exp / "optimization").mkdir(parents=True)
    for i, (elapsed_final, sat) in enumerate([(5.0, 0), (7.0, 0), (3.0, 100)], start=1):
        opt = {
            "num_satisfying_final": sat,
            "elapsed": [0.0, elapsed_final],
            "num_steps": 30,
        }
        (exp / "optimization" / f"opt_{i:04d}.json").write_text(json.dumps(opt))
    manifest = [{"cell": "full", "height": 0.0, "seed": 0,
                 "experiment_id": exp.name, "status": "done",
                 "experiment_path": str(exp)}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    records = load_study(tmp_path / "manifest.json")
    assert len(records) == 1
    assert records[0]["loop_s"] == 15.0  # 5 + 7 + 3
    assert records[0]["num_satisfying_final"] == 100  # from the LAST file only
    # Steps EXECUTED (len(elapsed)=2 per file), not the num_steps=30 budget:
    # early stopping makes the budget a systematic overcount in real runs.
    assert records[0]["ms_per_step"] == 1000.0 * 15.0 / 6


def test_skeleton_started_after_the_budget_adds_no_step(tmp_path):
    # With --max_duration, a skeleton started after the budget writes elapsed=[]: it ran no step, so it must
    # not count as one in ms/step (known-bugs N6).
    exp = tmp_path / "ablation_full_h000_s0"
    (exp / "optimization").mkdir(parents=True)
    for i, elapsed in enumerate([[0.0, 1.0, 3.0], []], start=1):
        (exp / "optimization" / f"opt_{i:04d}.json").write_text(json.dumps({"elapsed": elapsed}))
    manifest = [{"cell": "full", "height": 0.0, "seed": 0,
                 "experiment_id": exp.name, "status": "done",
                 "experiment_path": str(exp)}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    record = load_study(tmp_path / "manifest.json")[0]
    assert record["loop_s"] == 3.0
    assert record["ms_per_step"] == 1000.0 * 3.0 / 3


def test_compute_deltas_paired_by_height_and_seed():
    records = [
        {"cell": "full", "height": 0.0, "seed": 0, "num_satisfying_final": 100, "loop_s": 10.0},
        {"cell": "full", "height": 0.0, "seed": 1, "num_satisfying_final": 80, "loop_s": 12.0},
        {"cell": "standard", "height": 0.0, "seed": 0, "num_satisfying_final": 60, "loop_s": 20.0},
        {"cell": "standard", "height": 0.0, "seed": 1, "num_satisfying_final": 90, "loop_s": 8.0},
        # seed 2 has no `full` counterpart at this height: must be dropped, not paired.
        {"cell": "standard", "height": 0.0, "seed": 2, "num_satisfying_final": 50, "loop_s": 5.0},
    ]
    rows = compute_deltas(records)
    by_metric = {(r["seed"], r["metric"]): r["delta"] for r in rows if r["cell"] == "standard"}
    assert by_metric[(0, "num_satisfying_final")] == -40
    assert by_metric[(1, "num_satisfying_final")] == 10
    assert by_metric[(0, "loop_s")] == 10.0
    assert by_metric[(1, "loop_s")] == -4.0
    assert (2, "num_satisfying_final") not in by_metric
    assert all(r["cell"] != "full" for r in rows)  # full is never a delta target

    agg = aggregate_deltas(rows)
    mean, std = agg[("standard", 0.0)]["num_satisfying_final"]
    assert mean == -15.0 and round(std, 3) == round(statistics.stdev([-40, 10]), 3)


def _patch(run, overall=None, refreshes=None):
    exp = Path(run["experiment_path"])
    if overall is not None:
        m = json.loads((exp / "overall_metrics.json").read_text())
        (exp / "overall_metrics.json").write_text(json.dumps({**m, **overall}))
    if refreshes is not None:
        o = json.loads((exp / "optimization" / "opt_0001.json").read_text())
        o["reik_refresh"] = [{"fallback": f} for f in refreshes]
        (exp / "optimization" / "opt_0001.json").write_text(json.dumps(o))


def test_stable_plan_wilson_and_curobo_calls(tmp_path, monkeypatch):
    import math
    import sys
    from cutamp.scripts import analyze_ablation
    from cutamp.scripts.analyze_ablation import stable_rates, wilson

    runs = {(c, s): _fake_run(tmp_path, c, 0.0, s) for c in ("full", "default_reik") for s in (0, 1, 2)}
    _patch(runs["full", 0], {"motion_plan_success": True}, [False, False])  # viol 0.0: balanced
    _patch(runs["full", 1], {"motion_plan_success": False})                 # no plan
    # full s2: no motion_plan_success key -> unknown, dropped from n
    _patch(runs["default_reik", 0], {"motion_plan_success": True}, [True, False, True])  # viol 0.2
    _patch(runs["default_reik", 1], {"motion_plan_success": True, "traj_com_violation_frac": None})  # audit failed
    (tmp_path / "manifest.json").write_text(json.dumps(list(runs.values())))

    recs = {(r["cell"], r["seed"]): r for r in load_study(tmp_path / "manifest.json")}
    assert [recs["full", s]["stable_plan"] for s in (0, 1, 2)] == [1.0, 0.0, None]
    assert [recs["default_reik", s]["stable_plan"] for s in (0, 1, 2)] == [0.0, None, None]
    assert recs["full", 0]["reik_curobo_calls"] == 0
    assert recs["default_reik", 0]["reik_curobo_calls"] == 2
    assert recs["default_reik", 2]["reik_curobo_calls"] is None  # no refresh records (no re-IK)

    rates = stable_rates(recs.values())
    assert rates[("full", 0.0)] == rates[("full", "all")] == (1, 2)
    assert rates[("default_reik", 0.0)] == (0, 1)

    lo, hi = wilson(5, 10)
    assert math.isclose(lo, 0.2366, abs_tol=1e-4) and math.isclose(hi, 0.7634, abs_tol=1e-4)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and math.isclose(hi, 0.2775, abs_tol=1e-4)

    monkeypatch.setattr(sys, "argv", ["analyze_ablation", "--manifest", str(tmp_path / "manifest.json"),
                                      "--out", str(tmp_path / "out")])
    analyze_ablation.main()
    md = (tmp_path / "out" / "results.md").read_text()
    assert "## Balanced-plan rate" in md
    assert "| Full (CoM + balance-first re-IK) | all | 1/2 | 0.50 |" in md


def test_margin_is_the_planned_particles(tmp_path):
    # After a motion-plan retry, best_com_margin describes rank 0, which wasn't planned. The table takes the
    # planned particle's planned_com_margin, and best_com_margin for runs that predate it (known-bugs N3).
    runs = [_fake_run(tmp_path, "full", 0.0, s) for s in (0, 1)]
    _patch(runs[0], {"planned_com_margin": 0.03})
    (tmp_path / "manifest.json").write_text(json.dumps(runs))
    assert [r["best_com_margin"] for r in load_study(tmp_path / "manifest.json")] == [0.03, 0.05]


def test_restored_init_reaches_the_table(tmp_path):
    # Without re-IK, Adam empties the satisfying set and _BestSnapshot puts the step-0 IK init back
    # (known-bugs B13). adam_gain and restored must carry that into the table; runs that predate
    # the fields show a dash.
    from cutamp.scripts.analyze_ablation import _write_md

    runs = [_fake_run(tmp_path, "no_reik", 0.0, s) for s in (0, 1)] + [_fake_run(tmp_path, "full", 0.0, 0)]
    for run in runs[:2]:
        f = Path(run["experiment_path"]) / "optimization" / "opt_0001.json"
        f.write_text(json.dumps({**json.loads(f.read_text()), "adam_gain": 0, "restored": True}))
    (tmp_path / "manifest.json").write_text(json.dumps(runs))

    recs = load_study(tmp_path / "manifest.json")
    assert [(r["adam_gain"], r["restored"]) for r in recs] == [(0, 1.0), (0, 1.0), (None, None)]
    agg = aggregate(recs)
    assert agg[("no_reik", 0.0)]["restored"] == (1.0, 0.0)
    _write_md(agg, {}, tmp_path / "results.md")
    rows = {line.split(" | ")[0]: line for line in (tmp_path / "results.md").read_text().splitlines()}
    assert rows["| CoM only (no re-IK)"].endswith("| 0 ± 0 | 1.00 ± 0.00 |")
    assert rows["| Full (CoM + balance-first re-IK)"].endswith("| — | — |")


def _timers(**totals):
    """A timer_metrics.json body (TorchTimer.get_summaries) with one call per timer."""
    return {name: {"total": t, "mean": t, "median": t, "std": 0.0, "count": 1} for name, t in totals.items()}


TOP_TIMERS = {"load_tamp_world": 0.41, "warmup_ik_solver": 1.15, "get_plan_generator": 0.003,
              "sample_initial_plans": 2.71, "sort_plans": 0.0002, "start_optimization": 8.06}


def test_planning_s_sums_the_top_timers(tmp_path):
    # Design section 8: planning time is the sum of the top-level timers. The stage times nest inside them (seed IK in
    # sample_initial_plans; trajopt and the final repair in start_optimization), and first_solution overlaps.
    nested = {"initialize_particles": 3.1, "curobo_planning": 1.7, "final_repair": 0.45, "first_solution": 5.0,
              "optimization_step": 2.6, "curobo_motion_gen_warmup": 0.3, "setup_optimizer": 0.2}
    runs = [_fake_run(tmp_path, "real_legs", 0.5, s) for s in (0, 1, 2)]
    for run, timers in zip(runs, (_timers(**TOP_TIMERS, **nested),
                                  _timers(**{k: v for k, v in TOP_TIMERS.items() if k != "sort_plans"}))):
        (Path(run["experiment_path"]) / "timer_metrics.json").write_text(json.dumps(timers))
    runs[0]["n"] = 256
    (tmp_path / "manifest.json").write_text(json.dumps(runs))
    full, partial, none = load_study(tmp_path / "manifest.json")
    assert full["planning_s"] == pytest.approx(sum(TOP_TIMERS.values()), abs=1e-12)
    assert (full["seed_ik_s"], full["trajopt_s"], full["final_repair_s"], full["load_tamp_world_s"]) == (
        3.1, 1.7, 0.45, 0.41)
    assert full["n"] == 256 and full["ms_per_step"] == 1000.0 * 30.0 / 2
    # A sum missing a top timer would understate the run: it reports none.
    assert partial["planning_s"] is None and partial["load_tamp_world_s"] == 0.41 and partial["seed_ik_s"] is None
    assert none["n"] is None
    assert all(none[k] is None for k in ("planning_s", "seed_ik_s", "trajopt_s", "final_repair_s", "load_tamp_world_s"))


def test_torque_failures_count_every_torque_form(tmp_path):
    # A failed candidate's message names torque in trajopt's reason (": torque") and in each of the balance gate's
    # torque raises: no time scale up to the cap keeps every effort (_torque_scale), or the frames it returns pass one
    # (_check_torques). Trajopt's ": other" and the gate's velocity and ZMP raises don't count.
    from types import SimpleNamespace

    import torch

    from cutamp.motion_solver import RETIME_MAX, _check_torques, _check_velocity, _plan_failed, _torque_scale
    from cutamp.robots.t1 import REAL_LEGS

    n = len(REAL_LEGS.joint_names)

    class Torques:  # world.segment_torques's interface: [T, dof] torques, and an effort and a speed limit per joint
        def __init__(self, tau):
            self.tau, self.effort, self.velocity = tau, torch.ones(n), torch.ones(n)

        def __call__(self, js):
            return torch.full((getattr(js, "position", torch.zeros(3, n)).shape[0], n), self.tau)

    def world(tau):
        return SimpleNamespace(model=REAL_LEGS, segment_torques=Torques(tau))

    rest = torch.zeros(3, n)
    js = SimpleNamespace(position=rest, velocity=rest + 2 * RETIME_MAX, acceleration=rest,
                         joint_names=list(REAL_LEGS.joint_names))  # twice the speed limit at the slowdown cap
    raised = []
    for check, args in ((_check_torques, (world(2.0), None)), (_torque_scale, (world(2.0), js, 0.5, RETIME_MAX)),
                        (_check_velocity, (world(0.5), js, RETIME_MAX))):
        with pytest.raises(RuntimeError) as e:
            check(*args)
        raised.append(str(e.value))
    result = SimpleNamespace(js_solution=object(), get_interpolated_plan=lambda: None)  # a trajopt result
    messages = [str(_plan_failed(world(2.0), result, "Retract plan failed for op")), *raised,
                str(_plan_failed(world(0.5), result, "Retract plan failed for op")),
                f"trajectory ZMP stays outside the support region at the {RETIME_MAX}x slowdown cap"]
    assert messages[0].endswith(": torque") and messages[4].endswith(": other")
    assert f"at the {RETIME_MAX}x slowdown cap" in messages[2] and "velocity passes" in messages[3]
    runs = [_fake_run(tmp_path, "real_legs", 0.5, s) for s in (0, 1)]
    _patch(runs[0], {"motion_plan_failures": messages})
    (tmp_path / "manifest.json").write_text(json.dumps(runs))
    assert [r["torque_failures"] for r in load_study(tmp_path / "manifest.json")] == [3, None]


def _gate_records(ratio=1.4, n=256, balanced=40, slow_tail=1.0):
    """40 runs per model at ``n``: simplified planning times 10 s + k / 4, the real legs ``ratio`` times as long, with
    their top five ``slow_tail`` times longer again, and ``balanced`` of their 40 plans balanced."""
    records = []
    for k in range(40):
        for cell, scale in (("simplified", 1.0), ("real_legs", ratio * (slow_tail if k >= 35 else 1.0))):
            real = cell == "real_legs"
            records.append({
                "cell": cell, "n": n, "height": 0.0, "seed": k, "found_solution": 1.0, "motion_plan_success": 1.0,
                "stable_plan": float(not real or k < balanced), "planning_s": (10.0 + 0.25 * k) * scale,
                "seed_ik_s": 4.0 * scale, "ms_per_step": 8.75 * scale, "trajopt_s": 2.0 * scale,
                "final_repair_s": 0.45, "load_tamp_world_s": 0.4, "torque_failures": int(real and k < 3),
            })
    return records


def test_gate_summary_flags():
    # Design section 8: decided at n = 256, the real legs pass with 40/40 on all three counts, a median planning-time
    # ratio of at most 1.5 and a p90 ratio of at most 2.0 (numpy's linear percentile).
    import numpy as np

    s = gate_summary(_gate_records(1.4) + _gate_records(1.6, n=512))
    assert s["decided_at"] == 256 and s["pass"] == {"success": True, "median": True, "p90": True, "pass": True}
    at = s["by_n"][256]
    simplified, real = at["cells"]["simplified"], at["cells"]["real_legs"]
    times = [10.0 + 0.25 * k for k in range(40)]
    assert simplified["planning_s"] == pytest.approx({"median": float(np.percentile(times, 50)),
                                                      "p90": float(np.percentile(times, 90))})
    assert simplified["planning_s"]["p90"] == pytest.approx(18.775)  # 35.1 of 39 places in: 18.75 + 0.1 * 0.25
    assert at["ratio"]["median"] == pytest.approx(1.4) and at["ratio"]["p90"] == pytest.approx(1.4)
    counts = ("runs", "timed", "found_solution", "motion_plan_success", "stable_plan", "torque_failures")
    assert {k: real[k] for k in counts} == {"runs": 40, "timed": 40, "found_solution": 40, "motion_plan_success": 40,
                                            "stable_plan": 40, "torque_failures": 3}
    assert real["stage_medians"] == pytest.approx({"seed_ik_s": 5.6, "ms_per_step": 12.25, "trajopt_s": 2.8,
                                                   "final_repair_s": 0.45, "load_tamp_world_s": 0.4})
    assert simplified["torque_failures"] == 0
    # n = 512 is reported, but it does not decide.
    assert s["by_n"][512]["flags"]["median"] is False and s["by_n"][512]["ratio"]["median"] == pytest.approx(1.6)

    assert gate_summary(_gate_records(1.6))["pass"] == {"success": True, "median": False, "p90": True, "pass": False}
    assert gate_summary(_gate_records(1.4, balanced=39))["pass"] == {
        "success": False, "median": True, "p90": True, "pass": False}
    slow = gate_summary(_gate_records(1.4, slow_tail=2.0))["by_n"][256]
    assert slow["ratio"]["median"] == pytest.approx(1.4) and slow["ratio"]["p90"] == pytest.approx(2.8)
    assert slow["flags"] == {"success": True, "median": True, "p90": False, "pass": False}
    # A run that never finished drops out of load_study, so a cell short of its 40 runs cannot pass, the baseline's
    # included: a missing simplified run would move the ratios as much.
    for cell in ("real_legs", "simplified"):
        short = [r for r in _gate_records(1.4) if not (r["cell"] == cell and r["seed"] == 0)]
        assert gate_summary(short)["pass"]["success"] is False, cell
        # A run missing a top-level timer has no planning time: it drops out of the median and p90, so it fails too.
        untimed = [{**r, "planning_s": None} if r["cell"] == cell and r["seed"] == 0 else r for r in _gate_records(1.4)]
        summary = gate_summary(untimed)
        assert summary["pass"]["success"] is False, cell
        assert (summary["by_n"][256]["cells"][cell]["runs"], summary["by_n"][256]["cells"][cell]["timed"]) == (40, 39)
    assert gate_summary(_gate_records(1.4, n=512))["pass"] is None


def test_gate_md_tells_a_short_study_from_a_failed_one(tmp_path):
    # A study short of the gate's 40 runs per model says so beside its FAIL; one with all 40 runs that fails does not.
    from cutamp.scripts.analyze_ablation import _write_gate_md
    _write_gate_md(gate_summary(_gate_records(1.6)), tmp_path / "failed.md")
    failed = (tmp_path / "failed.md").read_text()
    assert "**FAIL**." in failed and "not 40 runs" not in failed
    short = [r for r in _gate_records(1.4) if not (r["cell"] == "real_legs" and r["seed"] < 3)]
    _write_gate_md(gate_summary(short), tmp_path / "short.md")
    assert "**FAIL** (not 40 runs: real_legs 37)." in (tmp_path / "short.md").read_text()


def test_main_gate_writes_gate_outputs(tmp_path, monkeypatch):
    import sys

    from cutamp.scripts import analyze_ablation

    runs = []
    for cell in ("simplified", "real_legs"):
        run = {**_fake_run(tmp_path, cell, 0.5, 0), "n": 256}
        (Path(run["experiment_path"]) / "timer_metrics.json").write_text(json.dumps(_timers(**TOP_TIMERS)))
        runs.append(run)
    (tmp_path / "manifest.json").write_text(json.dumps(runs))
    monkeypatch.setattr(sys, "argv", ["analyze_ablation", "--manifest", str(tmp_path / "manifest.json"),
                                      "--out", str(tmp_path / "out"), "--gate"])
    analyze_ablation.main()
    summary = json.loads((tmp_path / "out" / "gate.json").read_text())
    assert summary["by_n"]["256"]["ratio"]["median"] == pytest.approx(1.0)
    assert summary["pass"]["success"] is False  # one run per model, not the gate's 40
    md = (tmp_path / "out" / "gate.md").read_text()
    assert "## n = 256" in md and "| real_legs | 1 | 1 |" in md  # runs, and runs with a planning time
    assert "**FAIL** (not 40 runs: simplified 1, real_legs 1)." in md
    assert "| failed past an effort |" in md and "Failed past an effort counts the motion-plan candidates" in md


@needs_cuda
def test_real_legs_report_on_a_still_home_plan(tmp_path):
    # real_legs_report on a raw plan of two still segments at home: cuRobo's torques with the floor's push give
    # design 7.4a's feet-on-the-floor statics (knees -4.57, ankle pitches +3.92 N m), the soles sit on the stance,
    # the Trunk's tilt is home's, and the margins come from overall_metrics. A schema-3 plan beside it is not a
    # real-leg one.
    import pickle

    import numpy as np
    import torch

    from cutamp.robots.t1 import REAL_LEGS, trunk_pose_from_q
    from cutamp.scripts import real_legs_report

    names = list(REAL_LEGS.joint_names)
    still = np.tile(np.asarray(REAL_LEGS.home, dtype=np.float32), (3, 1))
    zero = np.zeros_like(still)
    segment = {"arm": None, "dt": 0.025, "position": still, "velocity": zero, "acceleration": zero,
               "planner_timing": None}
    study = tmp_path / "study" / "experiments"
    for exp_id, plan in (("gate_real_legs_n256_h050_s0", {"schema_version": "stage3-raw", "model": "real_legs",
                                                           "joint_names": names, "segments": [segment, segment]}),
                         ("gate_simplified_n256_h050_s0", {"schema_version": 3})):
        (study / exp_id).mkdir(parents=True)
        with open(study / exp_id / "motion_plan.pkl", "wb") as f:
            pickle.dump(plan, f)
    (study / "gate_real_legs_n256_h050_s0" / "overall_metrics.json").write_text(json.dumps(
        {"traj_com_min_margin": 0.03, "traj_zmp_min_margin": 0.02, "retime_max": 1.0}))

    prefix = tmp_path / "out" / "report"
    real_legs_report.main(["--study", str(tmp_path / "study"), "--out", str(prefix)])
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert [r["run"] for r in report["runs"]] == ["gate_real_legs_n256_h050_s0"]
    run, joints = report["runs"][0], {j["joint"]: j for j in report["joints"]}
    assert len(joints) == 27 and not any(j.startswith("base_j_") for j in joints)  # the actuated joints
    for side in ("Left", "Right"):
        assert abs(joints[f"{side}_Knee_Pitch"]["peak_torque"] - 4.57) <= 0.3
        assert abs(joints[f"{side}_Ankle_Pitch"]["peak_torque"] - 3.92) <= 0.3
        assert joints[f"{side}_Knee_Pitch"]["effort"] == pytest.approx(130.5)
        assert joints[f"{side}_Knee_Pitch"]["knee_speed"] == pytest.approx(6.71)
        assert joints[f"{side}_Ankle_Pitch"]["velocity_limit"] == pytest.approx(7.29)  # cuRobo's, not 9.57
    assert joints["Waist"]["knee_speed"] == pytest.approx(1.88)
    assert joints["Waist"]["velocity_limit"] == pytest.approx(14.66)
    assert all(j["peak_speed"] == 0.0 for j in joints.values())
    home_tilt = float(torch.acos(trunk_pose_from_q(REAL_LEGS, REAL_LEGS.home)[2, 2]))
    assert run["max_tilt"] == pytest.approx(home_tilt, abs=1e-4)
    assert run["max_sole_drift_m"] < 1e-5 and run["max_sole_drift_rad"] < 1e-3
    assert (run["segments"], run["slowed"], run["com_min_margin"], run["zmp_min_margin"], run["retime_max"]) == (
        2, 0, 0.03, 0.02, 1.0)
    assert run["unslowed_over_effort"] == []
    md = prefix.with_suffix(".md").read_text()
    assert "| Left_Knee_Pitch |" in md and "| gate_real_legs_n256_h050_s0 |" in md


@needs_cuda
def test_real_legs_report_counts_sped_up_and_slowed_segments(tmp_path):
    # On the real legs the gate can speed a segment up (s < 1) as well as slow it: each retimed segment keeps the
    # planner's timing, so its s is its frames over the planner's. The report counts both, takes the gate's time
    # scales and binding checks (by check) from overall_metrics, and writes them in its tables.
    import pickle

    import numpy as np

    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.scripts import real_legs_report

    names = list(REAL_LEGS.joint_names)

    def still(n):
        position = np.tile(np.asarray(REAL_LEGS.home, dtype=np.float32), (n, 1))
        return {"position": position, "velocity": np.zeros_like(position), "acceleration": np.zeros_like(position)}

    segments = [{"arm": "left", "dt": 0.025, **still(3), "planner_timing": still(5)},   # sped up, s = 0.5
                {"arm": "left", "dt": 0.025, **still(5), "planner_timing": still(3)},   # slowed, s = 2
                {"arm": "right", "dt": 0.025, **still(3), "planner_timing": None}]       # s = 1
    run = tmp_path / "study" / "gate_real_legs_n256_h000_s0"
    run.mkdir(parents=True)
    with open(run / "motion_plan.pkl", "wb") as f:
        pickle.dump({"schema_version": "stage3-raw", "model": "real_legs", "joint_names": names,
                     "segments": segments}, f)
    (run / "overall_metrics.json").write_text(json.dumps({
        "traj_com_min_margin": 0.03, "traj_zmp_min_margin": 0.02, "retime_min": 0.5, "retime_max": 2.0,
        "retime_binding": {"velocity Waist": 1, "zmp": 1, "torque Left_Knee_Pitch": 1}}))

    prefix = tmp_path / "out" / "report"
    real_legs_report.main(["--study", str(tmp_path / "study"), "--out", str(prefix)])
    report = json.loads(prefix.with_suffix(".json").read_text())
    (r,) = report["runs"]
    assert (r["segments"], r["slowed"], r["sped_up"], r["retime_min"], r["retime_max"]) == (3, 1, 1, 0.5, 2.0)
    assert r["binding"] == {"torque": 1, "velocity": 1, "zmp": 1}
    assert (report["summary"]["retime_min"], report["summary"]["retime_max"]) == (0.5, 2.0)
    assert r["unslowed_over_effort"] == []  # only the segment at the planner's timing is read for it
    md = prefix.with_suffix(".md").read_text()
    assert "| gate_real_legs_n256_h000_s0 | 3 | 1 | 1 | 0.50 | 2.00 | torque 1, velocity 1, zmp 1 |" in md
    assert "time scales from 0.50x to 2.00x" in md
