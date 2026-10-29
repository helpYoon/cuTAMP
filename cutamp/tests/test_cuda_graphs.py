"""``--cuda_graphs`` (Mode H): the balance-first Adam step and re-IK refresh, captured
as G_step and G_refresh and replayed, each replay against its eager counterpart bitwise."""

import gc
import itertools
import subprocess
import sys
import weakref
from collections import Counter
from pathlib import Path

import pytest
import torch

from cutamp.config import SUPPORTED_SOFT_COSTS, TAMPConfiguration
from cutamp.tests.conftest import (
    SpyVis, clone_particles, install_refresh_spy, install_step_replay_spy, make_optimizer, make_plan_info, needs_cuda,
    repair_cuda_rng, run_optimizer,
)
from cutamp.tests.test_step_lockstep import _site_ptrs


def test_cuda_graphs_validation():
    from cutamp.config import TAMPConfiguration, validate_tamp_config
    assert TAMPConfiguration().cuda_graphs is False
    with pytest.raises(ValueError, match="cuda_graphs needs reik_balance_first"):
        validate_tamp_config(TAMPConfiguration(cuda_graphs=True))
    with pytest.raises(ValueError, match="align_yaw"):
        validate_tamp_config(TAMPConfiguration(cuda_graphs=True, reik_balance_first=True, soft_cost=["align_yaw"]))
    validate_tamp_config(TAMPConfiguration(cuda_graphs=True, reik_balance_first=True))
    validate_tamp_config(TAMPConfiguration(soft_cost=["align_yaw"]))  # eager mode keeps align_yaw


def test_cli_passes_cuda_graphs(monkeypatch):
    import cutamp.scripts.run_cutamp as rc
    seen = []
    monkeypatch.setattr(rc, "cutamp_demo", lambda env, config, **kw: seen.append(config))
    for argv, want in ((["--reik_balance_first", "--cuda_graphs"], True), (["--reik_balance_first"], False)):
        monkeypatch.setattr(sys, "argv", ["run_cutamp", *argv])
        rc.entrypoint()
        assert seen[-1].cuda_graphs is want


# The failing capture's process: cwd is this tree's root, so ``-c`` imports this tree's cutamp.
_FAILED_CAPTURE_CHILD = """
import os
import pytest
import torch
import cutamp
from cutamp.cuda_graphs import capture
from cutamp.tests.conftest import repair_cuda_rng
assert os.path.realpath(cutamp.__file__).startswith(os.getcwd() + os.sep), cutamp.__file__
with pytest.raises(RuntimeError, match="--cuda_graphs"):
    capture("bad", body=lambda: torch.ones(1, device="cuda").sum().item(), warmup=lambda: None, ptrs=tuple)
assert torch.cuda.current_stream() == torch.cuda.default_stream()
repair_cuda_rng()
assert torch.rand(4, device="cuda").isfinite().all()
"""


@needs_cuda
def test_failed_capture_restores_default_stream():
    # A failed capture leaves its process's allocator in capture mode until the process exits
    # (cuda_graphs.capture), so it runs in a child. No flock there: the child runs inside this
    # process's GPU lock, and a second flock on the same file would deadlock.
    child = subprocess.run([sys.executable, "-c", _FAILED_CAPTURE_CHILD], cwd=Path(__file__).resolve().parents[2],
                           capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stderr


@needs_cuda
def test_guard_trips_on_resize():
    from curobo.types import JointState
    from cutamp.cuda_graphs import capture, graph_ptrs
    from cutamp.robots.t1 import get_t1_kinematics
    kin = get_t1_kinematics()
    q = (0.1 * torch.randn(4, len(kin.joint_names), device="cuda")).requires_grad_()

    def body():  # square(): cuRobo's FK backward needs a contiguous grad, and sum()'s is expanded
        kin.compute_kinematics(JointState.from_position(q)).robot_spheres.square().sum().backward()

    def warmup():
        body()
        q.grad = None

    for breaker in ("resize", "drop grad"):
        q.grad = None
        g = capture("toy", body, warmup, ptrs=lambda: graph_ptrs([q, q.grad], [kin]))
        g.replay()
        if breaker == "resize":
            kin.update_batch_size(8, 1)
        else:
            q.grad = None
        with pytest.raises(AssertionError, match="toy"):
            g.replay()
        g.release()


def _graph_config(**kw):
    return TAMPConfiguration(**{**dict(num_particles=64, early_stop_patience=0, reik_balance_first=True,
                                       cuda_graphs=True, enable_visualizer=False), **kw})


@needs_cuda
@pytest.mark.parametrize("sc", [None, *(s for s in SUPPORTED_SOFT_COSTS if s != "align_yaw")])
def test_step_graph_captures_for_soft_cost(blocks_t1_plan, monkeypatch, sc):  # Review Focus 4
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=2, soft_cost=None if sc is None else [sc])
    replays = install_step_replay_spy(monkeypatch)
    m, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config,
                         visualizer=SpyVis(config))
    assert m["capture_step"] == 0 and replays == [[0, 1]]  # replay 1 runs on inputs Adam has moved


@needs_cuda
def test_step_graph_lockstep(blocks_t1_plan, monkeypatch):            # Review Focus 1
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=25)
    replays, vis = install_step_replay_spy(monkeypatch), SpyVis(config)
    m, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config, visualizer=vis)
    assert m["capture_step"] == 0 and m["capture_s"] > 0
    assert replays == [list(range(25))]             # one graph; step 0 too: the capture itself runs nothing
    assert vis.opt_steps == [0, 10, 20, 24]
    assert vis.losses == [m["loss"][s] for s in vis.opt_steps]


@needs_cuda
@pytest.mark.parametrize("reik_interval, num_opt_steps, capture_step", [(5, 50, 4), (1, 12, 0)])
def test_refresh_graph_lockstep(blocks_t1_plan, monkeypatch, reik_interval, num_opt_steps, capture_step):
    from cutamp.optimize_plan import _QueuedRefresh, _ReikDriftGate
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=num_opt_steps, reik_interval=reik_interval)
    eager = install_refresh_spy(monkeypatch)
    m, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config)
    assert m["refresh_capture_step"] == capture_step and m["refresh_capture_s"] > 0
    steps = list(range(capture_step + reik_interval, num_opt_steps, reik_interval))
    assert [s for s, _ in eager] == steps            # 9..49 = refreshes 2-10; 1..11 (Review Focus 5)
    # flush() turns the replays' slots into the records eager packs give, but for solve_s
    q_names = tuple(r["conf"] for r in m["reik_refresh"] if r["step"] == capture_step)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); e1.record()
    ref = _ReikDriftGate(len(q_names), 0.0, 0.0)
    ref._queued = [_QueuedRefresh(s, q_names, 64, p, e0, e1) for s, p in eager]
    ref.flush()
    strip = lambda rs: [{k: v for k, v in r.items() if k != "solve_s"} for r in rs]
    assert strip(ref.stats) == strip([r for r in m["reik_refresh"] if r["step"] in set(steps)])


@needs_cuda
def test_deferred_capture_after_planner_ends(blocks_t1_plan, monkeypatch, record_property):
    import cutamp.optimize_plan as optimize_plan

    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=20)
    replays = install_step_replay_spy(monkeypatch)
    eager = install_refresh_spy(monkeypatch)
    busy = itertools.count()
    m, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config,
                         visualizer=SpyVis(config), planner_busy=lambda: next(busy) < 7)
    assert m["capture_step"] == 7 and replays == [list(range(13))]
    # Design 4.6: refresh #1 (step 4) runs eagerly with no G_step yet, so G_refresh waits for refresh #2.
    assert m["refresh_capture_step"] == 9 and [s for s, _ in eager] == [14, 19]
    assert next(busy) == 8                           # consulted once per step, until the capture
    # Ruling 6 / design open question 1: report, don't assert, whether the deferred capture
    # also succeeds without the penalties-cache clear. The clear stays either way.
    monkeypatch.setattr(optimize_plan, "_clear_step_autograd",
                        lambda cost_fn, params: [setattr(p, "grad", None) for p in params])
    busy2 = itertools.count()
    try:
        m2, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config,
                              visualizer=SpyVis(config), planner_busy=lambda: next(busy2) < 7)
        outcome = f"captured at step {m2['capture_step']}"
    except RuntimeError as e:
        # A failed capture leaves this process's allocator in capture mode for every later test: a finding, not a pass.
        outcome = f"capture failed: {e.__cause__!r}"
        repair_cuda_rng()
    print("deferred capture without the penalties-cache clear:", outcome)
    record_property("capture_without_penalties_cache_clear", outcome)


@needs_cuda
def test_second_optimize_call_recaptures(blocks_t1_plan, monkeypatch):  # Review Focus 2
    from cutamp.cuda_graphs import CapturedGraph
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=10)
    plan_info, mem, sites, released = make_plan_info(world, skeleton, clone_particles(base), config), [], [], []
    orig = CapturedGraph.release

    def release(self):                               # what each release must free; no strong refs kept
        outs = [weakref.ref(t) for t in (self.outputs or ()) if isinstance(t, torch.Tensor)]
        released.append((self.name, self.cuda_graph.pool(), outs))
        orig(self)

    monkeypatch.setattr(CapturedGraph, "release", release)
    opt = make_optimizer(config)
    for call in range(2):
        m, _ = run_optimizer(plan_info, config, optimizer=opt)
        assert m["capture_step"] == 0 and m["capture_s"] > 0
        assert m["refresh_capture_step"] == 4
        assert Counter(n for n, _, _ in released) == {"G_step": call + 1, "G_refresh": call + 1}  # once per call each
        assert all(v.grad is None for v in plan_info["particles"].values())
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        assert all(r() is None for _, _, outs in released for r in outs)  # no static output outlives its call
        pools = {p for _, p, _ in released}
        assert not [s for s in torch.cuda.memory_snapshot() if s["segment_pool_id"] in pools]  # nor a pool block
        mem.append(torch.cuda.memory_allocated())
        sites.append(_site_ptrs(world))
    # The FK sites outlive the call, sized by the batch they last ran. Same plan, so the same
    # buffers after both calls: they count the same in both readings and can't hide a leak.
    assert sites[1] == sites[0]
    assert mem[1] == mem[0]                          # and nothing grows from call to call


@needs_cuda
def test_uncaptured_and_short_calls(blocks_t1_plan, monkeypatch):     # Review Focus 3
    from cutamp.cuda_graphs import CapturedGraph
    world, skeleton, base = blocks_t1_plan
    released, orig = [], CapturedGraph.release
    monkeypatch.setattr(CapturedGraph, "release", lambda self: released.append(self.name) or orig(self))
    keys = ("capture_s", "capture_step", "refresh_capture_s", "refresh_capture_step")

    def run(planner_busy=None, **kw):
        config = _graph_config(**kw)
        info = make_plan_info(world, skeleton, clone_particles(base), config)
        return (*run_optimizer(info, config, planner_busy=planner_busy), info["particles"])
    m, _, _ = run(planner_busy=lambda: True, num_opt_steps=8)          # planner busy all loop
    assert [m[k] for k in keys] == [None] * 4 and released == []
    m, _, _ = run(num_opt_steps=3)                                     # ends before refresh #1
    assert m["capture_step"] == 0 and released == ["G_step"]
    assert m["refresh_capture_step"] is None
    m, timer, parts = run(num_opt_steps=50, early_stop_patience=1)     # early stop
    assert m.get("early_stopped_at") is not None and not timer.has_timer("optimization_step")
    assert len(released) == 1 + sum(m[k] is not None for k in ("capture_step", "refresh_capture_step"))
    assert all(v.grad is None for v in parts.values())
