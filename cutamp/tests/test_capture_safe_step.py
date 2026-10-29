"""The in-loop Adam step's capture-safe prerequisites: what the eager step
skips, and what it may sync on."""

from cutamp.config import TAMPConfiguration
from cutamp.tests.conftest import SpyVis, clone_particles, make_plan_info, needs_cuda, run_optimizer


@needs_cuda
def test_noop_visualizer_skips_vis_block(blocks_t1_plan):
    world, skeleton, base = blocks_t1_plan
    config = TAMPConfiguration(num_particles=64, num_opt_steps=12, early_stop_patience=0,
                               reik_balance_first=True, enable_visualizer=False)
    _, timer = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config)
    assert "visualize_opt_rollout" not in timer.get_summaries()  # P13: the no-op visualizer skips the block
    vis = SpyVis(config)
    run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config, visualizer=vis)
    assert vis.opt_steps == [0, 10, 11]
