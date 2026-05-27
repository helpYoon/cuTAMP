"""Smoke test for --coupled_reik path. Catches the A1-class regression where
the coupled-reIK feature TypeErrors on first refresh."""
import os
import pytest


@pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)
def test_coupled_reik_runs_without_error(tmp_path):
    """Run cutamp_demo with --coupled_reik for a few Adam steps. Assert no exception."""
    # Lazy import inside the test so pytest collection doesn't pull in torch/cuda.
    from cutamp.config import TAMPConfiguration
    from cutamp.scripts.run_cutamp import cutamp_demo, load_demo_env

    config = TAMPConfiguration(
        num_particles=8,
        num_opt_steps=5,
        coupled_reik=True,
        reik_interval=2,
        optimize_soft_costs=True,
        soft_cost=["com_polygon"],
        enable_visualizer=False,
        curobo_plan=False,  # skip motion plan to keep test fast
        experiment_root=str(tmp_path),
    )
    env = load_demo_env("blocks_t1")
    # cutamp_demo returns without raising if --coupled_reik path is healthy.
    cutamp_demo(env, config)
