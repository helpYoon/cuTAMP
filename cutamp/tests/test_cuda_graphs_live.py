"""``--cuda_graphs`` end to end (design 6.2): ``cutamp_demo`` with the real motion-planner
build thread and the vis block on, every spied G_step and G_refresh replay checked against
its eager counterpart bitwise. Slow, so it runs only with CUTAMP_SLOW=1. No module fixture:
``cutamp_demo`` builds its own world."""

import json
import os

import pytest

from cutamp.tests.conftest import SpyVis, install_refresh_spy, install_step_replay_spy, needs_cuda, needs_slow


@needs_cuda
@needs_slow
@pytest.mark.parametrize("n, step_spy", [(64, True), (256, True), (512, False)])
def test_live_lockstep_through_cutamp_demo(tmp_path, monkeypatch, n, step_spy):
    # Ruling 4: at n=64 and n=256, one G_step graph matches its eager twin for at least 20
    # consecutive replays; at every n, each G_refresh replay matches an eager refresh. The
    # spies raise on any mismatch. curobo_plan=True runs the real planner-build thread, so
    # planner_busy is live: call 1 captures only once the build has ended.
    import cutamp.algorithm
    from cutamp.config import TAMPConfiguration
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.scripts.run_cutamp import cutamp_demo

    monkeypatch.setattr(cutamp.algorithm, "Visualizer", lambda config, q_init: SpyVis(config))  # the vis block runs
    replays = install_step_replay_spy(monkeypatch) if step_spy else None
    eager = install_refresh_spy(monkeypatch)
    config = TAMPConfiguration(num_particles=n, reik_balance_first=True, cuda_graphs=True,
                               soft_cost=["joint_limit_margin"], enable_visualizer=False, curobo_plan=True,
                               experiment_root=str(tmp_path))
    cutamp_demo(load_env(os.path.join(get_env_dir(), "blocks_t1.yml")), config)
    calls = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*/optimization/opt_*.json"))]
    print(f"live lockstep n={n}: G_step replays compared per graph "
          f"{[len(r) for r in replays] if step_spy else 'not spied'}, G_refresh replays compared {len(eager)}")
    for i, c in enumerate(calls, 1):
        print(f"  call {i}: " + ", ".join(
            f"{k}={c[k]}" for k in ("capture_step", "capture_s", "refresh_capture_step", "refresh_capture_s")))
    assert any(c["capture_step"] is not None and c["refresh_capture_step"] is not None for c in calls)
    if step_spy:
        assert max(map(len, replays)) >= 20 and all(r == list(range(len(r))) for r in replays)
    assert len(eager) >= 1


@needs_cuda
@needs_slow
def test_live_capture_waits_for_the_planner_thread(tmp_path, monkeypatch):
    # Final review M2: the planner build sleeps until call 1 has run step 0, so the real build
    # thread is alive at call 1's first capture checks, and planner_busy (its is_alive,
    # algorithm.py) must defer the capture past step 0. Without planner_busy=planner_busy at
    # algorithm.py's ParticleOptimizer, call 1 captures at step 0. An event, not a fixed sleep:
    # at n=64 the build ends ~2.6 s before call 1 begins, by a margin that moves with plan sampling.
    import threading

    import cutamp.algorithm
    import cutamp.optimize_plan as optimize_plan
    from cutamp.config import TAMPConfiguration
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.scripts.run_cutamp import cutamp_demo

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
    monkeypatch.setattr(cutamp.algorithm, "Visualizer", lambda config, q_init: SpyVis(config))
    replays, eager = install_step_replay_spy(monkeypatch), install_refresh_spy(monkeypatch)
    config = TAMPConfiguration(num_particles=64, reik_balance_first=True, cuda_graphs=True,
                               soft_cost=["joint_limit_margin"], enable_visualizer=False, curobo_plan=True,
                               experiment_root=str(tmp_path))
    cutamp_demo(load_env(os.path.join(get_env_dir(), "blocks_t1.yml")), config)
    call1 = json.loads(sorted(tmp_path.glob("*/optimization/opt_*.json"))[0].read_text())
    (overall,) = [json.loads(p.read_text()) for p in tmp_path.glob("*/overall_metrics.json")]
    print(f"live deferral: call 1 capture_step={call1['capture_step']}, refresh_capture_step="
          f"{call1['refresh_capture_step']}; G_step replays compared per graph {[len(r) for r in replays]}, "
          f"G_refresh replays compared {len(eager)}; motion_plan_success={overall.get('motion_plan_success')}")
    assert builds == ["motion-planner-warmup"]
    assert call1["capture_step"] > 0 and call1["refresh_capture_step"] >= call1["capture_step"]
    # The spies raise on any mismatch, so every compared replay matched.
    assert replays and all(r == list(range(len(r))) for r in replays) and eager
    assert "found_solution" in overall                  # the run finished
