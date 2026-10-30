"""``--cuda_graphs`` against eager, one whole optimize call at a time: nothing eager runs in a graph
call, and the recorded calls must match as raw bytes."""

import itertools
import random
import sys

import numpy as np
import torch

from cutamp.tests.conftest import (
    assert_bitwise, bitwise_diffs, clone_particles, make_optimizer, make_plan_info, needs_cuda,
)
from cutamp.tests.test_cuda_graphs import _graph_config

_TIMING = ("elapsed", "opt_start_to_first_sol")
_CAPTURE = ("capture_s", "capture_step", "refresh_capture_s", "refresh_capture_step")  # G-only keys


def _install_recorder(monkeypatch) -> dict:
    """Record one optimize call into ``box["rec"]``, the same in both modes and with no eager body
    (a replay bypasses ``_StepBody.__call__``, so the per-step hook is ``_BestSnapshot.update``): each
    step's packed scalars, particles and grads, the final refresh's particles, and its ``_ik_for_pose``
    results (balance-first calls it only there)."""
    import cutamp.optimize_plan as op
    box = {"rec": None}
    update, refresh, ik = op._BestSnapshot.update, op.ParticleOptimizer._refresh_ik_deps, op._ik_for_pose

    def update_spy(self, step, num_satisfying, min_loss, particles):
        loop = sys._getframe(1).f_locals  # ParticleOptimizer.__call__'s frame
        box["rec"]["steps"].append({
            "packed": loop["packed"].clone(), "num_satisfying": num_satisfying, "min_cost": min_loss,
            "particles": clone_particles(particles),
            "grads": [None if p.grad is None else p.grad.clone() for p in loop["body"].params],
        })
        return update(self, step, num_satisfying, min_loss, particles)

    def refresh_spy(self, particles, ik_deps, world, gate=None, step=None):
        if gate is None:
            box["rec"]["pre_final"] = clone_particles(particles)
        refresh(self, particles, ik_deps, world, gate, step)
        if gate is None:
            box["rec"]["post_final"] = clone_particles(particles)

    def ik_spy(*args, **kwargs):
        res = ik(*args, **kwargs)
        box["rec"]["ik"].append((res.solution.clone(), res.success.clone()))
        return res

    monkeypatch.setattr(op._BestSnapshot, "update", update_spy)
    monkeypatch.setattr(op.ParticleOptimizer, "_refresh_ik_deps", refresh_spy)
    monkeypatch.setattr(op, "_ik_for_pose", ik_spy)
    return box


@needs_cuda
def test_graph_calls_give_the_eager_result(blocks_t1_plan, monkeypatch):
    from cutamp.utils.timer import TorchTimer
    from cutamp.utils.visualizer import Visualizer
    world, skeleton, base = blocks_t1_plan
    box = _install_recorder(monkeypatch)
    rng = torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate()

    def call(cuda_graphs, planner_busy=None) -> dict:
        """One optimize call on a fresh copy of the init, from the same RNG states."""
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])
        np.random.set_state(rng[2])
        random.setstate(rng[3])
        world.ik_solver.reset_seed()  # the IK seed sampler persists across solves
        config = _graph_config(num_opt_steps=100, cuda_graphs=cuda_graphs)
        box["rec"] = rec = {"steps": [], "pre_final": None, "post_final": None, "ik": []}
        info = make_plan_info(world, skeleton, clone_particles(base), config)
        optimizer = make_optimizer(config, planner_busy=planner_busy)
        found, m, exceeded = optimizer(info, TorchTimer(), Visualizer(config, None))
        rec.update(final=clone_particles(info["particles"]), found=found, exceeded=exceeded,
                   metrics={k: v for k, v in m.items() if k not in _TIMING + _CAPTURE})
        rec["metrics"]["reik_refresh"] = [{k: v for k, v in r.items() if k != "solve_s"} for r in m["reik_refresh"]]
        rec["capture"] = tuple(m.get(k, "absent") for k in ("capture_step", "refresh_capture_step"))
        return rec

    e1, g1, e2, g2 = call(False), call(True), call(False), call(True)
    busy = itertools.count()
    gd = call(True, planner_busy=lambda: next(busy) < 7)    # as test_deferred_capture_after_planner_ends
    assert [c["capture"] for c in (e1, g1, e2, g2, gd)] == [("absent",) * 2, (0, 4), ("absent",) * 2, (0, 4), (7, 9)]
    assert next(busy) == 8 and all(len(c["steps"]) == 100 for c in (e1, g1, e2, g2, gd))
    assert bitwise_diffs(e1["steps"][0]["particles"], e1["steps"][-1]["particles"])  # the recording isn't vacuous
    for name, got, want in (("G1 vs E1", g1, e1),       # the claim
                            ("E2 vs E1", e2, e1),       # the graph call leaves no state behind
                            ("G2 vs G1", g2, g1),       # a recaptured call gives the same result
                            ("Gd vs E1", gd, e1)):      # so does a capture deferred to mid-call
        assert_bitwise(*({k: v for k, v in c.items() if k != "capture"} for c in (got, want)), name, raw=True)
