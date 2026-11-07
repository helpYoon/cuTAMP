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


def test_patches_are_installed_and_the_guard_names_a_changed_function(monkeypatch):
    """``import cutamp`` installed the patches: a massless URDF link weighs nothing (test_robot_com.py checks the
    values), and one ``wp.Stream`` serves every launch on a CUDA stream. The source guard rejects a cuRobo whose
    pinned functions differ from v0.9.0's, naming the function."""
    import pytest
    import warp as wp
    from curobo._src.robot.parser.parser_urdf import UrdfRobotParser

    import cutamp._curobo_internals as ci

    assert getattr(UrdfRobotParser.get_link_parameters, "_cutamp_patch", False)
    if torch.cuda.is_available():
        wp.init()  # cuRobo initializes warp before its first launch
        stream = torch.cuda.current_stream()
        assert wp.stream_from_torch(stream) is wp.stream_from_torch(stream)
    path = "curobo._src.util.warp:get_warp_device_stream"
    monkeypatch.setitem(ci._PINNED_SOURCES, path, "0" * 12)
    with pytest.raises(RuntimeError, match="get_warp_device_stream"):
        ci._check_curobo_sources()
