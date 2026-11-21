"""Geometric tests for the grasp-free, stance-conditioned hand map."""

import numpy as np
import pytest

from cutamp.reachability import ReachMap, _box_gaps, apply_command, canonical_stance, endpoint_orders, stance_key

HOME = np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])


def placed(soles, dx, dy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    moved = soles.copy()
    moved[:, :2] = soles[:, :2] @ np.array([[c, s], [-s, c]]) + [dx, dy]
    moved[:, 2] += yaw
    return moved


def test_step_geometry_and_frames():
    destination = apply_command(HOME, "L", (0.33, 0.48, 0))
    local, frame = canonical_stance(destination)
    np.testing.assert_allclose(frame, [0.165, 0.14, 0])
    assert stance_key(local) == stance_key(placed(destination, -2, 3, 3.9))
    turned = apply_command(HOME, "L", (0, 0.28, np.deg2rad(50)))
    turned = apply_command(turned, "R", (0, -0.28, np.deg2rad(20)))
    np.testing.assert_allclose(np.rad2deg(turned[:, 2]), [50, 70])
    assert np.rad2deg(canonical_stance(turned)[1][2]) == pytest.approx(60)
    assert apply_command(HOME, "R", (0, 0.28, 0)) is None
    assert apply_command(HOME, "L", (0, 0.20, 0)) is None  # travel below the trained minimum
    orders = endpoint_orders(HOME)
    assert list(orders) == ["none", "L", "R", "LR", "RL"]
    assert all(len(choices) for choices in orders.values())
    for sole_pair in orders["R"]:
        np.testing.assert_array_equal(sole_pair[0], HOME[0])


def layer(posture, points):
    return {
        "arm": "left",
        "posture": posture,
        "points": np.array(points),
        "q": np.arange(len(points) * 2).reshape(-1, 2),
        "margin": np.full(len(points), 0.065),
    }


def test_map_roundtrip_keeps_one_witness_per_cell(tmp_path):
    atlas = ReachMap(0.02, metadata={"q_names": ["a", "b"]})
    atlas.add_stance(
        HOME,
        [
            layer("standing", [[0.091, 0.011, 0.301]]),
            layer("crouch_40", [[0.091, 0.011, 0.201], [0.092, 0.012, 0.202]]),
        ],
    )
    path = tmp_path / "map.npz"
    atlas.save(path)
    saved = ReachMap.load(path)
    assert saved.metadata == atlas.metadata
    standing, crouch = saved.layers(HOME, "left")
    assert (standing["posture"], crouch["posture"]) == ("standing", "crouch_40")
    assert len(crouch["ijk"]) == 1


def test_rotated_cell_to_box_distance():
    # The cell [0, .1]^2 is centered .25 m short of the target on x. The closest
    # vertex of the rotated target is on the y=.05 line, at x=.3-.03*sqrt(2).
    (gap,) = _box_gaps(np.array([[-0.25, 0.0, 0.0]]), np.array([0.03, 0.03, 0.05]), 0.05, np.pi / 4)
    assert gap == pytest.approx(0.30 - 0.03 * np.sqrt(2) - 0.1)


def test_intersecting_boxes_with_no_vertex_inside_have_no_gap():
    (gap,) = _box_gaps(np.zeros((1, 3)), np.array([0.125, 0.006, 0.04]), 0.05, 0.6)
    assert gap == 0


def test_failed_map_save_leaves_previous_map_loadable(tmp_path):
    atlas = ReachMap(0.02)
    path = tmp_path / "map.npz"
    atlas.save(path)
    atlas.metadata["unserializable"] = object()
    with pytest.raises(TypeError):
        atlas.save(path)
    assert ReachMap.load(path).metadata == {}
