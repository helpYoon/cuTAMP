"""Unit tests for cuRobo workarounds in _curobo_internals.py."""
from types import SimpleNamespace
import pytest
import torch


def _make_plan_result(positions, success=False, cspace_error=None,
                     joint_names=None):
    """Build a minimal duck-typed result object that cspace_plan_succeeded
    will accept."""
    plan = SimpleNamespace(
        position=torch.tensor(positions),
        joint_names=joint_names,
    )
    return SimpleNamespace(
        success=torch.tensor([success]),
        cspace_error=cspace_error,
        interpolated_trajectory=plan,
        js_solution=None,
    )


def test_cspace_plan_succeeded_inspects_all_batches():
    """For a [B, T, dof] plan (sim returns these), the salvage path must
    check every batch element's last timestep, not just the last batch.
    Regression for the > 2 vs >= 2 squeeze-loop bug."""
    from cutamp._curobo_internals import cspace_plan_succeeded

    # Two batches, T=2 timesteps, dof=3. Batch 0 lands EXACTLY on goal;
    # batch 1 is far off. The salvage path should report success because
    # at least one batch met tolerance — UNLESS the squeeze loop drops
    # batch 0 by picking only the last batch.
    positions = [
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],   # batch 0 ends at goal
        [[0.0, 0.0, 0.0], [9.9, 9.9, 9.9]],   # batch 1 ends far away
    ]
    target = SimpleNamespace(
        position=torch.tensor([0.5, 0.5, 0.5]),
        joint_names=None,
    )
    result = _make_plan_result(positions, success=False, cspace_error=None)
    assert cspace_plan_succeeded(result, target, tol=1e-3) is True, (
        "cspace_plan_succeeded missed batch 0's success because the "
        "squeeze loop dropped earlier batches."
    )
