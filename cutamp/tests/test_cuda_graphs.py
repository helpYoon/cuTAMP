"""``--cuda_graphs``: the Adam step, captured as G_step and replayed, each replay against its eager counterpart
bitwise, on a plan without steps (:func:`~cutamp.tests.conftest.zero_step_plan`)."""

import gc
import subprocess
import sys
import weakref
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from cutamp.tests.conftest import (
    SpyVis, cupti_sync_events, install_step_replay_spy, make_optimizer, needs_cuda, plan_info, run_optimizer,
    zero_step_plan,
)
from cutamp.tests.test_step_lockstep import _site_ptrs
from cutamp.utils.timer import TorchTimer


def test_cli_defaults_on(monkeypatch):
    # --cuda_graphs restates the default; --no_cuda_graphs turns the graphs off.
    import cutamp.scripts.run_cutamp as rc
    seen = []
    monkeypatch.setattr(rc, "cutamp_demo", lambda env, config, **kw: seen.append(config))
    for argv, graphs in (([], True), (["--cuda_graphs"], True), (["--no_cuda_graphs"], False)):
        monkeypatch.setattr(sys, "argv", ["run_cutamp", *argv])
        rc.entrypoint()
        assert seen[-1].cuda_graphs == graphs, argv


def test_cli_reaches_cutamp_demo(monkeypatch):
    # The flags a motion-planning run passes: n, the motion plan, no visualizer, an experiment id and an automatically
    # named saved plan all reach cutamp_demo.
    import cutamp.scripts.run_cutamp as rc
    seen = []
    monkeypatch.setattr(rc, "cutamp_demo", lambda env, config, **kw: seen.append((config, kw)))
    monkeypatch.setattr(sys, "argv", ["run_cutamp", "-n", "64", "--disable_visualizer", "--motion_plan",
                                      "--save_plan", "auto", "--experiment_id", "cli_run"])
    rc.entrypoint()
    config, kw = seen[-1]
    assert (config.num_particles, config.curobo_plan, config.enable_visualizer) == (64, True, False)
    assert kw == {"experiment_id": "cli_run", "save_plan": "auto"}


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
    from cutamp.robots.t1 import REAL_LEGS, get_t1_kinematics
    kin = get_t1_kinematics(REAL_LEGS)
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


def _graph_plan(world, **kw):
    """blocks_t1's plan without steps on ``world``, at 64 particles, recorded, with no early stop."""
    return zero_step_plan(world, **{**dict(early_stop_patience=0, cuda_graphs=True, enable_visualizer=False), **kw})


@needs_cuda
def test_step_graph_lockstep(real_world, monkeypatch):
    plan = _graph_plan(real_world, num_opt_steps=25)
    config = plan.config
    replays, vis = install_step_replay_spy(monkeypatch), SpyVis(config)
    m, _ = run_optimizer(plan_info(plan), config, visualizer=vis)
    assert m["capture_step"] == 0 and m["capture_s"] > 0
    assert replays == [list(range(25))]             # one graph; step 0 too: the capture itself runs nothing
    assert vis.opt_steps == [0, 10, 20, 24]
    assert vis.losses == [m["loss"][s] for s in vis.opt_steps]


# The leak check is the bytes the tensors request, which is exact. memory_allocated is not compared: the caching
# allocator counts each request at the size of the block it lands in, so memory_allocated can move between calls with
# nothing new held. (test_cuda_graphs_whole_call compares the calls' results bitwise, not their memory.)
@needs_cuda
def test_second_optimize_call_recaptures(real_world, monkeypatch):
    from cutamp.cuda_graphs import CapturedGraph
    world = real_world
    plan = _graph_plan(world, num_opt_steps=10)
    config, info, mem, sites, released = plan.config, plan_info(plan), [], [], []
    orig, req = CapturedGraph.release, []            # req: the bytes the tensors request, after each call

    def release(self):                               # what each release must free; no strong refs kept
        outs = [weakref.ref(t) for t in (self.outputs or ()) if isinstance(t, torch.Tensor)]
        released.append((self.name, self.cuda_graph.pool(), outs))
        orig(self)

    monkeypatch.setattr(CapturedGraph, "release", release)
    opt = make_optimizer(config)
    for call in range(2):
        m, _ = run_optimizer(info, config, optimizer=opt)
        assert Counter(n for n, _, _ in released) == {"G_step": call + 1}  # once per call
        assert all(v.grad is None for v in info["particles"].values())
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


@needs_cuda
def test_short_and_early_stopped_calls(real_world, monkeypatch):
    from cutamp.cuda_graphs import CapturedGraph
    released, orig = [], CapturedGraph.release
    monkeypatch.setattr(CapturedGraph, "release", lambda self: released.append(self.name) or orig(self))

    def run(**kw):
        plan = _graph_plan(real_world, **kw)
        return (*run_optimizer(plan_info(plan), plan.config), plan.particles)
    m, _, _ = run(num_opt_steps=3)                                     # ends before refresh #1
    assert m["capture_step"] == 0 and released == ["G_step"]
    m, timer, parts = run(num_opt_steps=50, early_stop_patience=1)     # early stop
    assert m.get("early_stopped_at") is not None and "optimization_step" not in timer._start_time
    assert timer.get_summaries()["optimization_step"]["count"] == len(m["elapsed"])  # the stopping step is timed too
    assert released == ["G_step"] * 2
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
def test_loop_sync_audit(real_world):
    # Each profiled step without a refresh has the loop's 3 device syncs, the profiler's own and packed.tolist()'s
    # copy and stream sync. A G step is 1 graph launch plus the eager Adam step and the plan's own step, so it launches
    # fewer kernels than the same step run eagerly (S1), which also launches the captured body's; inside the graph,
    # the global-mode capture is the audit. A refresh step (44, 49) also runs the plan's eager refresh, which this
    # audit leaves out.
    sync_events = bool(cupti_sync_events())
    launches = {}
    for arm in ("S1", "G"):
        plan = _graph_plan(real_world, num_opt_steps=50, cuda_graphs=arm == "G")
        timer = _ProfilingTimer(range(40, 50))
        try:
            m, _ = run_optimizer(plan_info(plan), plan.config, timer=timer)
        finally:
            timer.close()
        assert sorted(timer.names) == list(range(40, 50))
        # The step where num_satisfying first turns positive adds the first-solution sync.
        first = next((s for s, k in enumerate(m["num_satisfying"]) if k > 0), None)
        for step, names in sorted(timer.names.items()):
            if step == first or step in (44, 49):
                continue
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
                "EventRecord": 0,
                "cudaGraphLaunch": 0 if arm == "S1" else 1,
            }
            if sync_events:  # CUPTI's sync records also catch driver-API waits
                got["sync events"] = Counter(n for n in names if n.endswith(" Sync"))
                want["sync events"] = {"Context Sync": 4, "Stream Sync": 1}
            assert got == want, f"{arm} step {step}"
            launches[arm, step] = sum(n.startswith(("cudaLaunch", "cuLaunch")) for n in names)
    for (arm, step), count in launches.items():
        if arm == "G" and ("S1", step) in launches:
            eager = launches["S1", step]
            assert count < eager, f"G step {step}: {count} kernel launches, the eager step's {eager}"
