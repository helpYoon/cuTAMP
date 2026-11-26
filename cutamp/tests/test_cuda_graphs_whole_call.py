"""``--cuda_graphs`` against eager, one whole optimize call at a time on a plan without steps: nothing eager runs in
a graph call. Adam moves the postures, whose gradients come from cuRobo's kinematics backward, which sums with
shared-memory atomicAdd (kinematics_backward_kernel.cuh:180-215): two calls part at rounding from the first update
and Adam carries it on, so the calls match at float32 tolerance on their first judged step, recorded before any
update, and in their shape after it. Each later step against its eager twin on the same state is
test_cuda_graphs.test_step_graph_lockstep's."""

import random
import sys

import numpy as np
import torch

from cutamp.tests.conftest import assert_bitwise, bitwise_diffs, clone_particles, make_optimizer, needs_cuda, plan_info
from cutamp.tests.test_cuda_graphs import _graph_plan

_TIMING = ("elapsed", "opt_start_to_first_sol")
_CAPTURE = ("capture_s", "capture_step")  # G-only keys


def _install_recorder(monkeypatch) -> dict:
    """Record one optimize call into ``box["rec"]``, the same in both modes and with no eager body
    (a replay bypasses ``_StepBody.__call__``, so the hook is ``_BestSnapshot.update``, on each judged step): each
    judged step's packed scalars, particles and grads."""
    import cutamp.optimize_plan as op
    box = {"rec": None}
    update = op._BestSnapshot.update

    def update_spy(self, step, num_satisfying, min_loss):
        loop = sys._getframe(1).f_locals  # ParticleOptimizer.__call__'s frame
        box["rec"]["steps"].append({
            "packed": loop["packed"].clone(), "num_satisfying": num_satisfying, "min_cost": min_loss,
            "particles": clone_particles(loop["particles"]),
            "grads": [None if p.grad is None else p.grad.clone() for p in loop["body"].params],
        })
        return update(self, step, num_satisfying, min_loss)

    monkeypatch.setattr(op._BestSnapshot, "update", update_spy)
    return box


@needs_cuda
def test_graph_calls_give_the_eager_result(real_world, monkeypatch):
    from cutamp.utils.timer import TorchTimer
    from cutamp.utils.visualizer import Visualizer
    box = _install_recorder(monkeypatch)
    rng = torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate()

    def call(cuda_graphs) -> dict:
        """One optimize call on a fresh plan, from the same RNG states."""
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])
        np.random.set_state(rng[2])
        random.setstate(rng[3])
        plan = _graph_plan(real_world, num_opt_steps=100, cuda_graphs=cuda_graphs)
        config, info = plan.config, plan_info(plan)
        box["rec"] = rec = {"steps": []}
        found, m, exceeded = make_optimizer(config)(info, TorchTimer(), Visualizer(config, None))
        rec.update(final=clone_particles(info["particles"]), found=found, exceeded=exceeded,
                   metrics={k: v for k, v in m.items() if k not in _TIMING + _CAPTURE + ("steps",)})
        rec["metrics"]["steps"] = {k: v for k, v in m["steps"].items() if k != "refresh_s"}  # its refreshes' times
        rec["capture"] = m.get("capture_step", "absent")
        return rec

    e1, g1, e2, g2 = call(False), call(True), call(False), call(True)
    assert [c["capture"] for c in (e1, g1, e2, g2)] == ["absent", 0, "absent", 0]
    assert all(len(c["steps"]) == 20 for c in (e1, g1, e2, g2))   # every judged step: one in each 5
    assert bitwise_diffs(e1["steps"][0]["particles"], e1["steps"][-1]["particles"])  # the recording isn't vacuous
    for name, got, want in (("G1 vs E1", g1, e1),       # the claim
                            ("E2 vs E1", e2, e1),       # the graph call leaves no state behind
                            ("G2 vs G1", g2, g1)):      # a recaptured call gives the same result
        assert_bitwise(got["steps"][0], want["steps"][0], f"{name}, step 0", close=True)
        assert (list(got["metrics"]), got["found"], got["exceeded"]) == (
            list(want["metrics"]), want["found"], want["exceeded"]), name
