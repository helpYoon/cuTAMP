"""Smoke test for --coupled_reik path. Catches the A1-class regression where
the coupled-reIK feature TypeErrors on first refresh."""
from cutamp.tests.conftest import needs_cuda


@needs_cuda
def test_coupled_reik_runs_without_error(tmp_path):
    """Run cutamp_demo with --coupled_reik for a few Adam steps. Assert no exception."""
    # Lazy import inside the test so pytest collection doesn't pull in torch/cuda.
    from cutamp.config import TAMPConfiguration
    import os

    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.scripts.run_cutamp import cutamp_demo

    config = TAMPConfiguration(
        num_particles=8,
        num_opt_steps=5,
        coupled_reik=True,
        reik_interval=2,
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
