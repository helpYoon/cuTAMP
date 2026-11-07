"""Unit tests for _ReikDriftGate (re-IK refresh instrumentation).

Exercises the stats records, not the IK solver. Needs CUDA events.
"""
import torch

from cutamp.optimize_plan import _ReikDriftGate
from cutamp.tests.conftest import needs_cuda


@needs_cuda
def test_log_batch_flush_records():
    # Refresh stats stay on the device until flush(), which turns them into
    # one record per dep: its miss counts and event time.
    D, B = 2, 4
    names = ["left_q1", "right_q2"]
    pose_ok = torch.tensor([1, 1, 0, 1, 0, 0, 1, 1], device="cuda").bool()
    com_ok = torch.tensor([1, 0, 1, 1, 1, 0, 1, 1], device="cuda").bool()
    gate = _ReikDriftGate(pos_tol=0.005, rot_tol=0.05)
    events = []
    # Two GPU waits, ~0.4 ms and ~35 ms, so solve_s differs and only a tens-of-ms host stall could flip
    # the two refreshes. Warm the spin kernel and drain first.
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    for step, cycles in ((7, 1_000_000), (12, 100_000_000)):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(cycles)
        end.record()
        events += [(start, end)] * D
        gate.log_batch(step, names, B, pose_ok & com_ok, pose_ok, com_ok, start, end)
    gate.flush()

    assert [(r["step"], r["conf"], r["batch"]) for r in gate.stats] == [(s, n, B) for s in (7, 12) for n in names]
    for rec, counts, (start, end) in zip(gate.stats, [(2, 1, 1), (2, 2, 1)] * 2, events):
        assert (rec["gn_fail"], rec["gn_fail_pose"], rec["gn_fail_com"]) == counts
        # The refresh's event time, split evenly over its deps.
        assert rec["solve_s"] == round(start.elapsed_time(end) / 1e3 / D, 4)
    assert 0 < gate.stats[0]["solve_s"] < gate.stats[2]["solve_s"]

    types = {"step": int, "conf": str, "batch": int, "solve_s": float, "gn_fail": int, "gn_fail_pose": int,
             "gn_fail_com": int}
    assert {k: type(v) for k, v in gate.stats[0].items()} == types
    gate.flush()  # the queue is empty now: nothing is appended twice
    assert len(gate.stats) == 4
