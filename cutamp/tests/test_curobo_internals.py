"""Unit tests for cuRobo workarounds in _curobo_internals.py."""
from types import SimpleNamespace

import torch


def test_cspace_plan_succeeded_trusts_curobo_success():
    """A plan cuRobo rejected stays rejected even though its trajectory ends
    exactly on the goal (cuRobo pins the last waypoint there). That is the
    shape of every seed-1 T1 failure: Left_Ankle_Pitch past its upper limit
    mid-path, endpoint exact, ``cspace_error`` never set."""
    from cutamp._curobo_internals import cspace_plan_succeeded

    ends_on_goal = SimpleNamespace(position=torch.zeros(1, 1, 4, 18), joint_names=None)

    def result(success):
        return SimpleNamespace(success=torch.tensor([[success]]), cspace_error=None,
                               interpolated_trajectory=ends_on_goal, js_solution=ends_on_goal)

    assert cspace_plan_succeeded(result(False)) is False
    assert cspace_plan_succeeded(result(True)) is True
    assert cspace_plan_succeeded(None) is False
