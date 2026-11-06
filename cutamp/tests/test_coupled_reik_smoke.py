"""Smoke test for the coupled re-IK path. Catches the A1-class regression where
the coupled-reIK feature TypeErrors on first refresh."""
import pytest

from cutamp.tests.conftest import needs_cuda


@needs_cuda
@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_coupled_reik_runs_without_error(tmp_path, monkeypatch, cuda_graphs):
    """Run cutamp_demo with coupled re-IK for a few Adam steps: no exception, and the balance-first
    refresh runs on the run's world."""
    # Lazy import inside the test so pytest collection doesn't pull in torch/cuda.
    from cutamp.config import TAMPConfiguration
    import os

    from cutamp import optimize_plan
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.scripts.run_cutamp import cutamp_demo

    seen = []  # per refresh call: (function, its world)

    def spy(name, world_arg):
        original = getattr(optimize_plan, name)

        def counted(*args, **kwargs):
            seen.append((name, args[world_arg]))
            return original(*args, **kwargs)

        monkeypatch.setattr(optimize_plan, name, counted)

    spy("_refresh_balanced", 2)  # (particles, ik_deps, world, gate, step)

    config = TAMPConfiguration(
        num_particles=8,
        num_opt_steps=5,
        reik_interval=2,
        cuda_graphs=cuda_graphs,
        soft_cost=["place_close_to_base"],
        enable_visualizer=False,
        curobo_plan=False,  # skip motion plan to keep test fast
        experiment_root=str(tmp_path),
    )
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    # cutamp_demo returns without raising if the coupled re-IK path is healthy.
    cutamp_demo(env, config)

    import glob, json

    opt_files = sorted(glob.glob(str(tmp_path / "**" / "optimization" / "opt_*.json"), recursive=True))
    assert opt_files, "no optimization metrics written"
    opt_metrics = json.load(open(opt_files[-1]))

    # Measure-only metric, recorded in every run.
    assert "best_com_margin" in opt_metrics
    # What Adam added over step 0 (the IK init), and whether the reported particles are a restored
    # snapshot. Gain 0 with a restore means the run reports its IK init (known-bugs B13).
    assert opt_metrics["adam_gain"] == opt_metrics["best_num_satisfying"] - opt_metrics["num_satisfying"][0] >= 0
    assert isinstance(opt_metrics["restored"], bool)

    # Without --motion_plan there is no planner thread, so G_step is captured at step 0, and
    # G_refresh right after refresh #1 (reik_interval=2, so step 1).
    if cuda_graphs:
        assert opt_metrics["capture_step"] == 0
        assert opt_metrics["refresh_capture_step"] == 1
    else:
        assert "capture_step" not in opt_metrics

    # Balance-first records reach opt_metrics only through the post-loop flush().
    records = opt_metrics.get("reik_refresh")
    assert records, "no reik_refresh records"
    assert seen, "no refresh reached the spies"
    world = seen[0][1]
    assert all(w is world for _, w in seen)
    # Drift needs a previous target: every record but the first refresh's has it.
    first = min(r["step"] for r in records)
    assert all(("pos_drift_mean" in r) == (r["step"] != first) for r in records)


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
