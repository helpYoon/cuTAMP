"""Smoke test for --coupled_reik path. Catches the A1-class regression where
the coupled-reIK feature TypeErrors on first refresh."""
import pytest

from cutamp.tests.conftest import needs_cuda


@needs_cuda
@pytest.mark.parametrize("reik_balance_first", [False, True])
def test_coupled_reik_runs_without_error(tmp_path, monkeypatch, reik_balance_first):
    """Run cutamp_demo with --coupled_reik for a few Adam steps. Assert no
    exception, and that each mode refreshes through its own path: balance-first
    through ``_refresh_balanced``, default through the per-dep ``gn_project``."""
    # Lazy import inside the test so pytest collection doesn't pull in torch/cuda.
    from cutamp.config import TAMPConfiguration
    import os

    from cutamp import optimize_plan
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.scripts.run_cutamp import cutamp_demo

    calls = {"_refresh_balanced": 0, "gn_project": 0}

    def count_calls(name):
        original = getattr(optimize_plan, name)

        def counted(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(optimize_plan, name, counted)

    for name in calls:
        count_calls(name)

    config = TAMPConfiguration(
        num_particles=8,
        num_opt_steps=5,
        coupled_reik=True,
        reik_interval=2,
        reik_balance_first=reik_balance_first,
        optimize_soft_costs=True,
        soft_cost=["place_close_to_base"],
        enable_visualizer=False,
        curobo_plan=False,  # skip motion plan to keep test fast
        experiment_root=str(tmp_path),
    )
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    # cutamp_demo returns without raising if --coupled_reik path is healthy.
    cutamp_demo(env, config)

    import glob, json

    opt_files = sorted(glob.glob(str(tmp_path / "**" / "optimization" / "opt_*.json"), recursive=True))
    assert opt_files, "no optimization metrics written"
    opt_metrics = json.load(open(opt_files[-1]))

    # Ablation yardstick: measure-only metrics recorded in EVERY arm.
    assert "num_satisfying_com_feasible" in opt_metrics
    assert "best_com_margin" in opt_metrics
    assert "best_soft_cost_com_feasible" in opt_metrics
    assert opt_metrics["num_satisfying_com_feasible"] <= opt_metrics["num_satisfying_final"]

    # Balance-first records reach opt_metrics only through the post-loop flush().
    records = opt_metrics.get("reik_refresh")
    assert records, "no reik_refresh records"
    if reik_balance_first:
        assert calls["_refresh_balanced"] >= 1 and calls["gn_project"] == 0, calls
        # No cuRobo during Adam: every GN miss stays a miss.
        assert all(r["fallback"] is False and r["n_ik_fail"] == r["gn_fail"] for r in records)
    else:
        assert calls["_refresh_balanced"] == 0 and calls["gn_project"] >= 1, calls


@needs_cuda
def test_early_stop_closes_step_timer(tmp_path, monkeypatch):
    """Early stop must stop the optimization_step timer: the next skeleton
    (no satisfying particles, or every motion-plan candidate failed) reuses
    the timer, and an open one crashed it with "Timer already started"."""
    import os

    from cutamp.config import TAMPConfiguration
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.optimize_plan import ParticleOptimizer
    from cutamp.scripts.run_cutamp import cutamp_demo

    calls = []
    original = ParticleOptimizer.__call__

    def wrapped(self, plan_info, timer, visualizer):
        out = original(self, plan_info, timer, visualizer)
        calls.append((out[1], timer.has_timer("optimization_step"),
                      timer.get_summaries()["optimization_step"]["count"]))
        return out

    monkeypatch.setattr(ParticleOptimizer, "__call__", wrapped)
    config = TAMPConfiguration(
        num_particles=8,
        num_opt_steps=50,
        early_stop_patience=1,
        enable_visualizer=False,
        curobo_plan=False,
        experiment_root=str(tmp_path),
    )
    cutamp_demo(load_env(os.path.join(get_env_dir(), "blocks_t1.yml")), config)

    metrics, timer_open, step_count = calls[0]
    assert metrics.get("early_stopped_at") is not None, "run never early-stopped"
    assert not timer_open
    assert step_count == len(metrics["elapsed"])  # the stopping step is timed too
