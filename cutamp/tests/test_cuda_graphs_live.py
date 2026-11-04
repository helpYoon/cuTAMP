"""``--cuda_graphs`` end to end through ``cutamp_demo`` (which builds its own world, so no
blocks_t1_plan), with the real planner thread and the vis block, every spied replay checked bitwise.
Slow: runs only with CUTAMP_SLOW=1."""

import json
import os

import pytest

from cutamp.config import TAMPConfiguration
from cutamp.tests.conftest import SpyVis, install_refresh_spy, install_step_replay_spy, needs_cuda

needs_slow = pytest.mark.skipif(os.environ.get("CUTAMP_SLOW") != "1", reason="slow live test; set CUTAMP_SLOW=1")


def _run_demo(tmp_path, monkeypatch, n, robot="simplified") -> list:
    import cutamp.algorithm
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.scripts.run_cutamp import cutamp_demo

    monkeypatch.setattr(cutamp.algorithm, "Visualizer", lambda config, q_init: SpyVis(config))  # the vis block runs
    # The real legs keep their default minimize_body_movement, as the speed gate's real-leg cell does, and plan no
    # motion: get_t1_motion_planner has no real-leg planner yet.
    real = robot == "real_legs"
    config = TAMPConfiguration(robot=robot, num_particles=n, reik_balance_first=True, cuda_graphs=True,
                               soft_cost=["joint_limit_margin"] + (["minimize_body_movement"] if real else []),
                               enable_visualizer=False, curobo_plan=not real, experiment_root=str(tmp_path))
    cutamp_demo(load_env(os.path.join(get_env_dir(), "blocks_t1.yml")), config)
    return [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*/optimization/opt_*.json"))]


@needs_cuda
@needs_slow
@pytest.mark.parametrize("n, step_spy", [(64, True), (256, True), (512, False)])
def test_live_lockstep_through_cutamp_demo(tmp_path, monkeypatch, n, step_spy):
    # Each spied replay must match its eager counterpart (the spies raise). curobo_plan=True runs
    # the real planner-build thread, so call 1 captures only once the build has ended.
    replays = install_step_replay_spy(monkeypatch) if step_spy else None
    eager = install_refresh_spy(monkeypatch)
    calls = _run_demo(tmp_path, monkeypatch, n)
    assert any(c["capture_step"] is not None and c["refresh_capture_step"] is not None for c in calls)
    if step_spy:
        assert max(map(len, replays)) >= 20
    assert len(eager) >= 1


@needs_cuda
@needs_slow
def test_real_legs_step_and_refresh_replay_bitwise(tmp_path, monkeypatch):
    # The real legs' larger shapes, with the 12 sole rows inside G_refresh and the retract confs' motion FK and
    # self-collision cost inside G_step: every spied replay matches its eager twin (the spies raise).
    from cutamp.cost_function import CostFunction

    retract_calls, retract_values = [], CostFunction._retract_motion_values

    def count(self):
        retract_calls.append(1)
        return retract_values(self)

    monkeypatch.setattr(CostFunction, "_retract_motion_values", count)
    replays, eager = install_step_replay_spy(monkeypatch), install_refresh_spy(monkeypatch)
    _run_demo(tmp_path, monkeypatch, 64, robot="real_legs")
    assert max(map(len, replays)) >= 20 and len(eager) >= 1
    assert retract_calls  # the skeleton has retracts, so the replayed G_step holds their motion cost


@needs_cuda
@needs_slow
def test_live_capture_waits_for_the_planner_thread(tmp_path, monkeypatch):
    # The build waits until call 1 has run a step, so the real build thread is alive at call 1's
    # first capture checks and planner_busy must defer the capture. An event, not a sleep: the
    # build's lead over call 1 moves with plan sampling.
    import threading

    import cutamp.algorithm
    import cutamp.optimize_plan as optimize_plan

    stepped, builds = threading.Event(), []
    build, update = cutamp.algorithm.build_motion_planner, optimize_plan._BestSnapshot.update

    def update_spy(self, *args, **kwargs):  # once per Adam step, after that step's capture check
        stepped.set()
        return update(self, *args, **kwargs)

    def slow_build(world, config):
        assert stepped.wait(timeout=120), "no optimize call ran a step"
        builds.append(threading.current_thread().name)
        return build(world, config)

    monkeypatch.setattr(optimize_plan._BestSnapshot, "update", update_spy)
    monkeypatch.setattr(cutamp.algorithm, "build_motion_planner", slow_build)
    replays, eager = install_step_replay_spy(monkeypatch), install_refresh_spy(monkeypatch)
    call1 = _run_demo(tmp_path, monkeypatch, 64)[0]
    (overall,) = [json.loads(p.read_text()) for p in tmp_path.glob("*/overall_metrics.json")]
    assert builds == ["motion-planner-warmup"]
    assert call1["capture_step"] > 0 and call1["refresh_capture_step"] >= call1["capture_step"]
    # The spies raise on any mismatch, so every compared replay matched.
    assert replays and eager
    assert "found_solution" in overall                  # the run finished
