"""Unit tests for cuRobo workarounds in _curobo_internals.py."""
import torch


def test_patches_are_installed_and_the_guard_names_a_changed_function(monkeypatch):
    """``import cutamp`` installed the patches: a URDF link with no inertial weighs nothing (test_robot_com.py checks
    the values), and one ``wp.Stream`` serves every launch on a CUDA stream. The source guard rejects a cuRobo whose
    pinned functions differ from v0.9.0's, naming the function."""
    import pytest
    import warp as wp
    from curobo._src.robot.parser.parser_urdf import UrdfRobotParser

    import cutamp._curobo_internals as ci

    assert hasattr(UrdfRobotParser.get_link_parameters, "__wrapped__")
    if torch.cuda.is_available():
        wp.init()  # cuRobo initializes warp before its first launch
        stream = torch.cuda.current_stream()
        assert wp.stream_from_torch(stream) is wp.stream_from_torch(stream)
    path = "curobo._src.util.warp:get_warp_device_stream"
    monkeypatch.setitem(ci._PINNED_SOURCES, path, "0" * 12)
    with pytest.raises(RuntimeError, match="get_warp_device_stream"):
        ci._check_curobo_sources()
