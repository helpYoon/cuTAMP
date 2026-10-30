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
