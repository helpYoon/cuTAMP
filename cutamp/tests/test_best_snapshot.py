"""Unit tests for _BestSnapshot (best-seen particle tracking + early stop).

CPU-only: exercises the bookkeeping, not the optimizer.
"""
import torch

from cutamp.optimize_plan import _BestSnapshot


def _particles(val: float) -> dict:
    return {"q": torch.full((4, 3), val), "pose": torch.full((4, 2), val)}


def test_snapshot_tracks_best_and_restores():
    snap = _BestSnapshot()
    p = _particles(1.0)
    snap.update(0, 10, 5.0, p)
    # Mutate in place past the snapshot (worse count follows).
    for v in p.values():
        v.fill_(2.0)
    snap.update(1, 7, 5.0, p)
    assert snap.best_sat == 10 and snap.best_step == 0
    # Restore rewrites the live tensors with the step-0 state.
    assert snap.restore_if_better(p, current_num_satisfying=7)
    assert torch.all(p["q"] == 1.0) and torch.all(p["pose"] == 1.0)


def test_restore_keeps_current_state_on_tie():
    snap = _BestSnapshot()
    p = _particles(1.0)
    snap.update(0, 10, 5.0, p)
    for v in p.values():
        v.fill_(2.0)
    # Tie: current state has had more soft-cost descent; keep it.
    assert not snap.restore_if_better(p, current_num_satisfying=10)
    assert torch.all(p["q"] == 2.0)


def test_patience_ignores_noise_but_resets_on_meaningful_progress():
    snap = _BestSnapshot()
    p = _particles(0.0)
    snap.update(0, 100, 5.0, p)
    # +1 count records are noise: they snapshot but must NOT reset patience.
    snap.update(10, 101, 5.0, p)
    assert snap.best_sat == 101
    assert snap.last_improve_step == 0
    assert snap.should_stop(step=51, patience=50)
    # A >= MIN_DELTA jump resets the clock.
    snap.update(60, 100 + _BestSnapshot.MIN_DELTA, 5.0, p)
    assert snap.last_improve_step == 60
    assert not snap.should_stop(step=100, patience=50)
    # Best-particle loss improvement also resets it (soft-cost descent proxy).
    snap.update(120, 90, 5.0 - 2 * _BestSnapshot.MIN_LOSS_DELTA, p)
    assert snap.last_improve_step == 120


def test_patience_zero_disables_early_stop():
    snap = _BestSnapshot()
    snap.update(0, 10, 5.0, _particles(0.0))
    assert not snap.should_stop(step=10_000, patience=0)


def test_patience_extends_only_for_better_feasible_center():
    from cutamp.optimize_plan import _FeasibleArchive

    p = _particles(0.)
    snap = _BestSnapshot()
    feasible = torch.tensor([True, False, False, False])
    center = torch.tensor([.09, .03, .02, .01])
    effort = torch.zeros(4)
    archive = _FeasibleArchive(p, center, effort)
    for step, distance in [(0, .09), (1, .08), (2, .07), (3, .06)]:
        center[0] = distance
        archive.observe(p, feasible, center, effort)
        snap.update(step, 1, 10., p)
        assert not snap.should_stop(step, 2, archive)
    # Count, weighted loss and feasible centering now plateau. Moving an infeasible
    # row toward the centroid cannot extend the patience window again.
    center[1] = 0.
    archive.observe(p, feasible, center, effort)
    assert snap.should_stop(6, 2, archive)
    assert not snap.should_stop(6, 0, archive)


def test_for_a_stepped_plan_only_a_better_plan_restarts_the_countdown():
    from cutamp.optimize_plan import _FeasibleArchive

    p = _particles(0.)
    snap = _BestSnapshot()
    feasible = torch.tensor([False, False, False, False])
    center = torch.tensor([.09, .03, .02, .01])
    effort = torch.zeros(4)
    archive = _FeasibleArchive(p, center, effort)

    def stop(step, patience=2):
        return snap.should_stop(step, patience, archive, by_rank=True)

    archive.observe(p, feasible, center, effort)       # no valid plan yet: the count and loss clock decides
    snap.update(0, 0, 10., p)
    assert not stop(2) and stop(3)
    feasible[0] = True                                  # the first valid plan starts the clock at its own step
    archive.observe(p, feasible, center, effort)
    snap.update(4, 1, 10., p)
    assert not stop(4)
    snap.update(5, 1 + _BestSnapshot.MIN_DELTA, 10. - 2 * _BestSnapshot.MIN_LOSS_DELTA, p)
    assert snap.last_improve_step == 5                  # a count jump and a loss drop no longer restart it
    assert not stop(6) and stop(7)
    center[0] = .05                                     # a better valid plan does, from its step
    archive.observe(p, feasible, center, effort)
    assert not stop(8) and not stop(10) and stop(11)
    center[1] = 0.                                      # an invalid row getting better does not
    archive.observe(p, feasible, center, effort)
    assert stop(12) and not stop(12, patience=0)


def test_for_a_stepped_plan_a_lower_effort_alone_does_not_restart_the_countdown():
    from cutamp.optimize_plan import _FeasibleArchive

    p = _particles(0.)
    snap = _BestSnapshot()
    feasible = torch.tensor([True, False, False, False])
    rank, effort = torch.tensor([3., 0., 0., 0.]), torch.tensor([.9, 0., 0., 0.])
    archive = _FeasibleArchive(p, rank, effort)

    def stop(step):
        return snap.should_stop(step, 2, archive, by_rank=True)

    for step, lower in enumerate((.9, .8, .7, .6)):          # the valid plan's joint effort alone keeps falling
        effort[0] = lower
        archive.observe(p, feasible, rank, effort)
        snap.update(step, 1, 10., p)
        assert stop(step) == (step == 3)
    assert archive.effort[0] == effort[0]                       # the record took each lower effort


def test_the_record_keeps_the_better_rank_then_the_lower_effort():
    from cutamp.cost_reduction import CostReducer
    from cutamp.optimize_plan import _FeasibleArchive, _quality
    p = _particles(0.)
    stepped = {"soft": {"type": "cost", "values": {}, "rank": torch.tensor([5., 5., 5., 5.]),
                        "effort": torch.tensor([.3, .3, .3, .3]), "max_center_distance_m": torch.zeros(4)}}
    primary, secondary = _quality(stepped, CostReducer({}))
    assert primary is stepped["soft"]["rank"] and secondary is stepped["soft"]["effort"]
    standing = {"soft": {"type": "cost", "values": {"x": torch.ones(4)}, "max_center_distance_m": torch.arange(4.)}}
    primary, secondary = _quality(standing, CostReducer({}))
    assert primary is standing["soft"]["max_center_distance_m"] and torch.equal(secondary, torch.ones(4))
    assert _quality({}, CostReducer({})) == (None, None)
    archive = _FeasibleArchive(p, *_quality(stepped, CostReducer({})))
    valid = torch.tensor([True, True, True, False])
    for name in p:
        p[name] += 1.0
    inf = float("inf")
    archive.observe(p, valid, torch.tensor([4., 5., 6., 0.]), torch.tensor([.9, .2, .1, .0]))
    assert archive.center.tolist() == [4., 5., 6., inf]            # the record starts every particle at infinity
    archive.observe(p, valid, torch.tensor([4., 5., 7., 0.]), torch.tensor([.5, .1, .0, .0]))
    assert archive.center.tolist() == [4., 5., 6., inf]            # a worse rank never replaces a better one
    torch.testing.assert_close(archive.effort, torch.tensor([.5, .1, .1, inf]))   # at the same rank, the lower effort


def test_preserve_best_feasible_center_for_each_particle():
    from cutamp.optimize_plan import _FeasibleArchive

    p = {"q": torch.tensor([[1.], [2.], [3.]])}
    archive = _FeasibleArchive(p, torch.zeros(3), torch.zeros(3))
    archive.observe(p, torch.tensor([True, True, False]), torch.tensor([.1, .3, .0]), torch.tensor([8., 1., 0.]))
    p["q"] += 10
    archive.observe(p, torch.tensor([True, True, True]), torch.tensor([.2, .3, .6]), torch.tensor([0., .9, 10.]))
    p["q"] += 10
    # A failed constraint rules out even perfect centering. Recheck the last iterate before restoring.
    restored = archive.restore(p, torch.tensor([False, True, True]), torch.tensor([0., .4, .5]), torch.zeros(3))
    assert restored
    assert p["q"].flatten().tolist() == [1., 12., 23.]


def test_for_a_stepped_plan_the_loop_ends_when_a_valid_plan_reaches_rank_zero():
    from cutamp.optimize_plan import _FeasibleArchive

    p = _particles(0.)
    snap = _BestSnapshot()
    rank, effort = torch.tensor([3., 0., 5., 5.]), torch.tensor([.5, .5, .5, .5])
    feasible = torch.tensor([True, False, False, False])
    archive = _FeasibleArchive(p, rank, effort)
    archive.observe(p, feasible, rank, effort)
    snap.update(0, 1, 10., p)
    assert not snap.should_stop(0, 50, archive, by_rank=True)   # a valid plan above rank 0, and a row at 0 not valid
    rank[0] = 1.
    archive.observe(p, feasible, rank, effort)
    assert not snap.should_stop(5, 50, archive, by_rank=True)
    rank[0] = 0.                                                # no rank is lower: nothing could restart the clock
    archive.observe(p, feasible, rank, effort)
    assert snap.should_stop(10, 50, archive, by_rank=True)
    assert not snap.should_stop(10, 0, archive, by_rank=True)   # patience 0 still switches the rule off
