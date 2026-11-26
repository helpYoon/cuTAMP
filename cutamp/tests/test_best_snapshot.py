"""Unit tests for _BestSnapshot (the Adam loop's early stop) and _FeasibleArchive (each particle's best valid state).

CPU-only: exercises the bookkeeping, not the optimizer.
"""
import torch

from cutamp.optimize_plan import _BestSnapshot, _FeasibleArchive


def _particles(val: float) -> dict:
    return {"q": torch.full((4, 3), val), "pose": torch.full((4, 2), val)}


def _no_valid_plan() -> _FeasibleArchive:
    """A record that has seen no valid particle."""
    return _FeasibleArchive(_particles(0.), *torch.zeros(3, 4))


def test_patience_ignores_noise_but_resets_on_meaningful_progress():
    snap, archive = _BestSnapshot(), _no_valid_plan()
    snap.update(0, 100, 5.0)
    # +1 count records are noise: they must NOT reset patience.
    snap.update(10, 101, 5.0)
    assert snap.best_sat == 101
    assert snap.last_improve_step == 0
    assert snap.should_stop(51, 50, archive)
    # A >= MIN_DELTA jump resets the clock.
    snap.update(60, 100 + _BestSnapshot.MIN_DELTA, 5.0)
    assert snap.last_improve_step == 60
    assert not snap.should_stop(100, 50, archive)
    # Best-particle loss improvement also resets it.
    snap.update(120, 90, 5.0 - 2 * _BestSnapshot.MIN_LOSS_DELTA)
    assert snap.last_improve_step == 120


def test_patience_zero_disables_early_stop():
    snap = _BestSnapshot()
    snap.update(0, 10, 5.0)
    assert not snap.should_stop(10_000, 0, _no_valid_plan())


def test_for_a_plan_only_a_better_plan_restarts_the_countdown():
    p = _particles(0.)
    snap = _BestSnapshot()
    feasible = torch.tensor([False, False, False, False])
    score = torch.tensor([9., 3., 2., 1.])
    keys = score, torch.full((4,), 5.), torch.zeros(4)  # the score, a rank above 0 and the joint effort
    archive = _FeasibleArchive(p, *keys)

    def stop(step, patience=2):
        return snap.should_stop(step, patience, archive)

    archive.observe(p, feasible, *keys)                 # no valid plan yet: the count and loss clock decides
    snap.update(0, 0, 10.)
    assert not stop(2) and stop(3)
    feasible[0] = True                                  # the first valid plan starts the clock at its own step
    archive.observe(p, feasible, *keys)
    snap.update(4, 1, 10.)
    assert not stop(4)
    snap.update(5, 1 + _BestSnapshot.MIN_DELTA, 10. - 2 * _BestSnapshot.MIN_LOSS_DELTA)
    assert snap.last_improve_step == 5                  # a count jump and a loss drop no longer restart it
    assert not stop(6) and stop(7)
    score[0] = 5.                                       # a valid plan of a lower score does, from its step
    archive.observe(p, feasible, *keys)
    assert not stop(8) and not stop(10) and stop(11)
    score[1] = 0.                                       # an invalid row getting better does not
    archive.observe(p, feasible, *keys)
    assert stop(12) and not stop(12, patience=0)


def test_for_a_plan_a_lower_effort_alone_does_not_restart_the_countdown():
    p = _particles(0.)
    snap = _BestSnapshot()
    feasible = torch.tensor([True, False, False, False])
    score, effort = torch.tensor([3., 0., 0., 0.]), torch.tensor([.9, 0., 0., 0.])
    archive = _FeasibleArchive(p, score, score, effort)

    def stop(step):
        return snap.should_stop(step, 2, archive)

    for step, lower in enumerate((.9, .8, .7, .6)):          # the valid plan's joint effort alone keeps falling
        effort[0] = lower
        archive.observe(p, feasible, score, score, effort)
        snap.update(step, 1, 10.)
        assert stop(step) == (step == 3)
    assert archive.effort[0] == effort[0]                       # the record took each lower effort


def test_the_record_keeps_the_lower_score_then_the_lower_rank_then_the_lower_effort():
    from cutamp.optimize_plan import _quality
    p = _particles(0.)
    cost = {"soft": {"type": "cost", "values": {}, "score": torch.tensor([5., 5., 5., 5.]),
                     "rank": torch.zeros(4), "effort": torch.tensor([.3, .3, .3, .3]),
                     "max_center_distance_m": torch.zeros(4)}}
    keys = ("score", "rank", "effort")
    assert all(a is cost["soft"][k] for a, k in zip(_quality(cost), keys, strict=True))
    archive = _FeasibleArchive(p, *_quality(cost))
    valid = torch.tensor([True, True, True, False])
    for name in p:
        p[name] += 1.0
    inf, rank = float("inf"), torch.tensor([1., 1., 1., 0.])
    archive.observe(p, valid, torch.tensor([4., 5., 6., 0.]), rank, torch.tensor([.9, .2, .1, .0]))
    assert archive.score.tolist() == [4., 5., 6., inf]             # the record starts every particle at infinity
    archive.observe(p, valid, torch.tensor([4., 5., 7., 0.]), rank, torch.tensor([.5, .1, .0, .0]))
    assert archive.score.tolist() == [4., 5., 6., inf]             # a higher score never replaces a lower one
    torch.testing.assert_close(archive.effort, torch.tensor([.5, .1, .1, inf]))   # all else the same, the lower effort
    archive.observe(p, valid, torch.tensor([4., 5., 6., 0.]), torch.tensor([2., 0., 1., 0.]), torch.ones(4))
    torch.testing.assert_close(archive.effort, torch.tensor([.5, 1., .1, inf]))   # at one score, a lower rank first
    assert archive.best() == [4., 1., .5]                         # the best plan on record: lowest score, then rank


def test_preserve_best_feasible_center_for_each_particle():
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


def test_for_a_plan_the_loop_ends_when_the_best_plan_is_centred():
    # Particle 0 is centred but folded (rank 0, score 40); particle 1 is upright and 3 units off centre (score 23).
    p = _particles(0.)
    snap = _BestSnapshot()
    score, rank, effort = torch.tensor([40., 23., 9., 9.]), torch.tensor([0., 3., 0., 0.]), torch.full((4,), .5)
    feasible = torch.tensor([True, True, False, False])
    archive = _FeasibleArchive(p, score, rank, effort)
    archive.observe(p, feasible, score, rank, effort)
    snap.update(0, 2, 10.)
    assert not snap.should_stop(0, 50, archive)   # a valid plan has rank 0, but the best plan is not it
    score[1], rank[1] = 21., 1.
    archive.observe(p, feasible, score, rank, effort)
    assert not snap.should_stop(5, 50, archive)
    score[1], rank[1] = 20., 0.                    # the best plan has no unit of side lean or off centre left
    archive.observe(p, feasible, score, rank, effort)
    assert snap.should_stop(10, 50, archive)
    assert not snap.should_stop(10, 0, archive)   # patience 0 still switches the rule off
