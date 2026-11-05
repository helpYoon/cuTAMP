"""``--cuda_graphs``: the balance-first Adam step and re-IK refresh, captured
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
from torch.profiler import ProfilerActivity, profile

from cutamp.config import SUPPORTED_SOFT_COSTS, TAMPConfiguration, uses_cuda_graphs, validate_tamp_config
from cutamp.tests.conftest import (
    SpyVis, clone_particles, cupti_sync_events, install_refresh_spy, install_step_replay_spy, make_optimizer,
    make_plan_info, needs_cuda, run_optimizer,
)
from cutamp.tests.test_step_lockstep import _site_ptrs
from cutamp.utils.timer import TorchTimer


def test_uses_cuda_graphs():
    # On by default; runs the graphs can't capture stay eager and still validate.
    assert uses_cuda_graphs(TAMPConfiguration())
    # The two ablation arms name the simplified model, the only one that runs them (design call 15).
    for kw in ({"cuda_graphs": False}, {"reik_balance_first": False, "robot": "simplified"},
               {"coupled_reik": False, "robot": "simplified"}, {"soft_cost": ["align_yaw"]}):
        validate_tamp_config(TAMPConfiguration(**kw))
        assert not uses_cuda_graphs(TAMPConfiguration(**kw)), kw


def test_cli_defaults_on(monkeypatch):
    # --reik_balance_first and --cuda_graphs restate the defaults; each --no_ flag turns its own off. The pose-only
    # refresh is an ablation arm, so its run names the simplified model (design call 15).
    import cutamp.scripts.run_cutamp as rc
    seen = []
    monkeypatch.setattr(rc, "cutamp_demo", lambda env, config, **kw: seen.append(config))
    for argv, balance_first, graphs in (([], True, True), (["--reik_balance_first", "--cuda_graphs"], True, True),
                                        (["--robot", "simplified", "--no_reik_balance_first"], False, True),
                                        (["--no_cuda_graphs"], True, False)):
        monkeypatch.setattr(sys, "argv", ["run_cutamp", *argv])
        rc.entrypoint()
        assert (seen[-1].reik_balance_first, seen[-1].cuda_graphs) == (balance_first, graphs), argv


# The failing capture's process: cwd is this tree's root, so ``-c`` imports this tree's cutamp.
_FAILED_CAPTURE_CHILD = """
import os
import pytest
import torch
import cutamp
from cutamp.cuda_graphs import CapturedGraph
assert os.path.realpath(cutamp.__file__).startswith(os.getcwd() + os.sep), cutamp.__file__
with pytest.raises(RuntimeError, match="--no_cuda_graphs"):
    CapturedGraph("bad", body=lambda: torch.ones(1, device="cuda").sum().item(), warmup=lambda: None, ptrs=tuple)
assert torch.cuda.current_stream() == torch.cuda.default_stream()
"""


@needs_cuda
def test_failed_capture_restores_default_stream():
    # A failed capture leaves its process's allocator in capture mode until the process exits
    # (cuda_graphs.CapturedGraph), so it runs in a child. No flock there: the child runs inside this
    # process's GPU lock, and a second flock on the same file would deadlock.
    child = subprocess.run([sys.executable, "-c", _FAILED_CAPTURE_CHILD], cwd=Path(__file__).resolve().parents[2],
                           capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stderr


@needs_cuda
def test_guard_trips_on_resize():
    from curobo.types import JointState
    from cutamp.cuda_graphs import CapturedGraph, graph_ptrs
    from cutamp.robots.t1 import SIMPLIFIED, get_t1_kinematics
    kin = get_t1_kinematics(SIMPLIFIED)
    q = (0.1 * torch.randn(4, len(kin.joint_names), device="cuda")).requires_grad_()

    def body():  # square(): cuRobo's FK backward needs a contiguous grad, and sum()'s is expanded
        kin.compute_kinematics(JointState.from_position(q)).robot_spheres.square().sum().backward()

    def warmup():
        body()
        q.grad = None

    for breaker in ("resize", "drop grad"):
        q.grad = None
        g = CapturedGraph("toy", body, warmup, ptrs=lambda: graph_ptrs([q, q.grad], [kin]))
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
@pytest.mark.parametrize("sc", [pytest.param(None, id="no_soft_cost"),
                                *(s for s in SUPPORTED_SOFT_COSTS if s != "align_yaw")])
def test_step_graph_captures_for_soft_cost(blocks_t1_plan, monkeypatch, sc):
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=2, soft_cost=[] if sc is None else [sc])
    replays = install_step_replay_spy(monkeypatch)
    m, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config,
                         visualizer=SpyVis(config))
    assert m["capture_step"] == 0 and replays == [[0, 1]]  # replay 1 runs on inputs Adam has moved


@needs_cuda
def test_step_graph_lockstep(blocks_t1_plan, monkeypatch):
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=25)
    replays, vis = install_step_replay_spy(monkeypatch), SpyVis(config)
    m, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config, visualizer=vis)
    assert m["capture_step"] == 0 and m["capture_s"] > 0
    assert replays == [list(range(25))]             # one graph; step 0 too: the capture itself runs nothing
    assert vis.opt_steps == [0, 10, 20, 24]
    assert vis.losses == [m["loss"][s] for s in vis.opt_steps]


@needs_cuda
def test_refresh_graph_lockstep(blocks_t1_plan, monkeypatch):
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=12, reik_interval=1)
    eager = install_refresh_spy(monkeypatch)
    m, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config)
    assert m["refresh_capture_step"] == 0 and m["refresh_capture_s"] > 0
    assert eager == list(range(1, 12))


@needs_cuda
def test_deferred_capture_after_planner_ends(blocks_t1_plan, monkeypatch):
    # SpyVis on: the loop's vis = None must drop an eager step's graph before a later capture.
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=20)
    replays = install_step_replay_spy(monkeypatch)
    busy = itertools.count()
    m, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config,
                         visualizer=SpyVis(config), planner_busy=lambda: next(busy) < 7)
    assert m["capture_step"] == 7 and replays == [list(range(13))]
    assert next(busy) == 8                           # consulted once per step, until the capture


# On both models; None is the default model and shares the module's world. The leak check is the bytes the tensors
# request, exact on both models. On the real legs the final check repairs with cuRobo's CUDA-graphed IK, whose line
# search (line_search_strategy.py, jit_get_x_set) reuses cached blocks of one size class, about 1.2 MB each, held on
# the C++ side. The caching allocator counts each request at the size of the block it lands in, so there
# memory_allocated moves between calls (by 16 to 80 KiB, seen) with nothing new held; it is exact only on the
# simplified model. (test_cuda_graphs_whole_call compares the calls' results bitwise, not their memory.)
@needs_cuda
@pytest.mark.parametrize("blocks_t1_plan", ["simplified", pytest.param(None, id="default")], indirect=True)
def test_second_optimize_call_recaptures(blocks_t1_plan, monkeypatch):
    from cutamp.cuda_graphs import CapturedGraph
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=10, robot=world.model.name)
    plan_info, mem, sites, released = make_plan_info(world, skeleton, clone_particles(base), config), [], [], []
    orig, req = CapturedGraph.release, []            # req: the bytes the tensors request, after each call

    def release(self):                               # what each release must free; no strong refs kept
        outs = [weakref.ref(t) for t in (self.outputs or ()) if isinstance(t, torch.Tensor)]
        released.append((self.name, self.cuda_graph.pool(), outs))
        orig(self)

    monkeypatch.setattr(CapturedGraph, "release", release)
    opt = make_optimizer(config)
    for call in range(2):
        m, _ = run_optimizer(plan_info, config, optimizer=opt)
        assert Counter(n for n, _, _ in released) == {"G_step": call + 1, "G_refresh": call + 1}  # once per call each
        assert all(v.grad is None for v in plan_info["particles"].values())
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        assert all(r() is None for _, _, outs in released for r in outs)  # no static output outlives its call
        pools = {p for _, p, _ in released}
        assert not [s for s in torch.cuda.memory_snapshot() if s["segment_pool_id"] in pools]  # nor a pool block
        mem.append(torch.cuda.memory_allocated())
        req.append(torch.cuda.memory_stats()["requested_bytes.all.current"])
        sites.append(_site_ptrs(world))
    # The FK sites outlive the call, sized by the batch they last ran. Same plan, so the same
    # buffers after both calls: they count the same in both readings and can't hide a leak.
    assert sites[1] == sites[0]
    assert req[1] == req[0], (req, mem)              # nothing a tensor holds grows from call to call
    if world.model.name == "simplified":
        assert mem[1] == mem[0]                      # nor the allocator's blocks (see above for the real legs)


@needs_cuda
def test_uncaptured_and_short_calls(blocks_t1_plan, monkeypatch):
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


class _ProfilingTimer(TorchTimer):
    """Profiles the optimization steps in ``steps``; ``names[step]`` is that step's event names."""

    def __init__(self, steps):
        super().__init__()
        self.steps, self.names, self._started, self._prof = set(steps), {}, 0, None

    def start(self, metric: str) -> None:
        if metric == "optimization_step":
            if self._started in self.steps:
                self._prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], **cupti_sync_events())
                self._prof.__enter__()
            self._started += 1
        super().start(metric)

    def stop(self, metric: str) -> float:
        duration = super().stop(metric)
        if metric == "optimization_step":
            self.close()
        return duration

    def close(self) -> None:
        """Exit the open profile, if any: a step that raises never reaches ``stop``."""
        if self._prof is not None:
            prof, self._prof = self._prof, None
            prof.__exit__(None, None, None)
            self.names[self._started - 1] = [e.name for e in prof.events()]


@needs_cuda
@pytest.mark.parametrize("arm", ["S1", "G"])
def test_loop_sync_audit(blocks_t1_plan, arm):
    # Each profiled step has the loop's 3 device syncs, the profiler's own and packed.tolist()'s copy
    # and stream sync; a refresh step adds 2 timing events. A G step is 1 graph launch (2 with the
    # refresh) plus the eager Adam step (the captured body alone is ~1,000 kernels, hence < 100);
    # inside the graphs, the global-mode capture is the audit, and every capturable soft cost is on.
    world, skeleton, base = blocks_t1_plan
    config = _graph_config(num_opt_steps=50, cuda_graphs=arm == "G",
                           soft_cost=[s for s in SUPPORTED_SOFT_COSTS if s != "align_yaw"])
    timer = _ProfilingTimer(range(40, 50))
    try:
        m, _ = run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config, timer=timer)
    finally:
        timer.close()
    assert sorted(timer.names) == list(range(40, 50))
    # The step where num_satisfying first turns positive adds the first-solution sync.
    first = next((s for s, k in enumerate(m["num_satisfying"]) if k > 0), None)
    sync_events = bool(cupti_sync_events())
    for step, names in sorted(timer.names.items()):
        if step == first:
            continue
        kind = "refresh" if step in (44, 49) else "plain"
        got = {
            "Synchronize": Counter(n for n in names if "Synchronize" in n),
            "DtoH": sum("DtoH" in n for n in names),
            "HtoD": sum("HtoD" in n for n in names),
            "EventRecord": sum("EventRecord" in n for n in names),
            "cudaGraphLaunch": names.count("cudaGraphLaunch"),
        }
        want = {
            "Synchronize": {"cudaDeviceSynchronize": 4, "cudaStreamSynchronize": 1},
            "DtoH": 1,
            "HtoD": 0,
            "EventRecord": 2 if kind == "refresh" else 0,
            "cudaGraphLaunch": 0 if arm == "S1" else 1 + (kind == "refresh"),
        }
        if sync_events:  # CUPTI's sync records also catch driver-API waits
            got["sync events"] = Counter(n for n in names if n.endswith(" Sync"))
            want["sync events"] = {"Context Sync": 4, "Stream Sync": 1}
        launches = sum(n.startswith(("cudaLaunch", "cuLaunch")) for n in names)
        assert got == want, f"{arm} step {step} ({kind})"
        if arm == "G":
            assert launches < 100, f"G step {step} ({kind}): {launches} kernel launches, not the eager Adam step's"
