"""Smoke test for the coupled re-IK path. Catches the A1-class regression where
the coupled-reIK feature TypeErrors on first refresh."""
import pytest

from cutamp.tests.conftest import needs_cuda, needs_graspgenx


@needs_cuda
@needs_graspgenx
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
