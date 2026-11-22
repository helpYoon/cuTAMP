"""``--cuda_graphs`` end to end through ``cutamp_demo`` (which builds its own world, so no
real_world), on blocks_t1's standing plan, with the vis block, every spied replay checked bitwise.
Slow: runs only with CUTAMP_SLOW=1."""

import os

import pytest

from cutamp.config import TAMPConfiguration
from cutamp.tests import conftest
from cutamp.tests.conftest import SpyVis, install_refresh_spy, install_step_replay_spy, needs_cuda, needs_graspgenx

needs_slow = pytest.mark.skipif(os.environ.get("CUTAMP_SLOW") != "1", reason="slow live test; set CUTAMP_SLOW=1")


def _run_demo(tmp_path, monkeypatch, n) -> list:
    """Each ParticleOptimizer call's metrics."""
    import cutamp.algorithm
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.optimize_plan import ParticleOptimizer
    from cutamp.scripts.run_cutamp import cutamp_demo

    monkeypatch.setattr(cutamp.algorithm, "Visualizer", lambda config, q_init: SpyVis(config))  # the vis block runs
    conftest.standing_skeleton(monkeypatch, *conftest.BLOCKS_T1_ACTIONS)
    calls, optimize = [], ParticleOptimizer.__call__

    def spy(self, *args, **kwargs):
        result = optimize(self, *args, **kwargs)
        calls.append(result[1])
        return result

    monkeypatch.setattr(ParticleOptimizer, "__call__", spy)
    # A fixed window of Adam steps for the replay comparisons, whenever the objective plateaus.
    config = TAMPConfiguration(num_particles=n, cuda_graphs=True, num_opt_steps=26, early_stop_patience=0,
                               soft_cost=["joint_limit_margin", "minimize_body_movement"],
                               enable_visualizer=False, curobo_plan=True, experiment_root=str(tmp_path))
    cutamp_demo(load_env(os.path.join(get_env_dir(), "blocks_t1.yml")), config)
    return calls


@needs_cuda
@needs_graspgenx
@needs_slow
@pytest.mark.parametrize("n", [64, 512])
def test_real_legs_step_and_refresh_replay_bitwise(tmp_path, monkeypatch, n):
    # The real legs' shapes, with the 12 sole rows and the closing step inside G_refresh and the retract confs'
    # motion FK and self-collision cost inside G_step: every spied replay matches its eager twin (the spies raise), at
    # the test batch and at 512 particles.
    from cutamp.cost_function import CostFunction

    retract_calls, retract_values = [], CostFunction._retract_motion_values

    def count(self, achieved_stances=None):
        retract_calls.append(1)
        return retract_values(self, achieved_stances)

    monkeypatch.setattr(CostFunction, "_retract_motion_values", count)
    replays, eager = install_step_replay_spy(monkeypatch), install_refresh_spy(monkeypatch)
    calls = _run_demo(tmp_path, monkeypatch, n)
    assert any(c["capture_step"] is not None and c["refresh_capture_step"] is not None for c in calls)
    assert max(map(len, replays)) >= 20 and len(eager) >= 1
    assert retract_calls  # the skeleton has retracts, so the replayed G_step holds their motion cost
