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
