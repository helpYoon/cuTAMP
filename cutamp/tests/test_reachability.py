"""Geometric tests for the grasp-free, stance-conditioned hand map."""

import numpy as np
import pytest

from cutamp.reachability import ReachMap, apply_command, canonical_stance, endpoint_orders, rank_orders, stance_key

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


def test_map_roundtrip_and_crouch_witness(tmp_path):
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
    assert len(saved.layers(HOME, "left")[1]["ijk"]) == 1
    orders = {"none": [HOME], "L": endpoint_orders(HOME)["L"]}
    kwargs = {"orders": orders}
    result = rank_orders(saved, HOME, [0.09, 0.01, 0.21], [0.012, 0.012, 0.012], **kwargs)
    assert result[0]["order"] == "none"
    assert result[0]["distance_m"] == 0
    assert result[0]["posture"] == "crouch_40"
    assert result[1]["distance_m"] is None and result[1]["covered_count"] == 0
    assert result[1]["uncovered_count"] == result[1]["endpoint_count"]
    standing = rank_orders(saved, HOME, [0.09, 0.01, 0.21], [0.012, 0.012, 0.012], posture="standing", **kwargs)
    assert standing[0]["distance_m"] == pytest.approx(0.30 - 0.216)


def test_rotated_cell_to_box_distance_and_rigid_invariance():
    # In local coordinates this cell is [0, .1]^2. The closest vertex of the
    # rotated target is on the y=.05 line, at x=.3-.03*sqrt(2).
    atlas = ReachMap(0.1)
    atlas.add_stance(HOME, [layer("standing", [[0.05, 0.05, 0.05]])])
    target = [0.30, 0.05, 0.05]
    dims = [0.06, 0.06, 0.10]
    yaw = np.pi / 4
    expected = 0.30 - 0.03 * np.sqrt(2) - 0.1
    (result,) = rank_orders(atlas, HOME, target, dims, target_yaw=yaw, orders={"none": [HOME]})
    assert result["distance_m"] == pytest.approx(expected)
    moved = placed(HOME, 0.6, -0.8, 2.9)
    xy = placed(np.array([[target[0], target[1], 0]]), 0.6, -0.8, 2.9)[0, :2]
    (shifted,) = rank_orders(atlas, moved, [*xy, target[2]], dims, target_yaw=yaw + 2.9, orders={"none": [moved]})
    assert shifted["distance_m"] == pytest.approx(expected)


def test_intersecting_boxes_with_no_vertex_inside_still_hit():
    atlas = ReachMap(0.1)
    atlas.add_stance(HOME, [layer("standing", [[0.05, 0.05, 0.05]])])
    (result,) = rank_orders(
        atlas, HOME, [0.05, 0.05, 0.05], [0.25, 0.012, 0.08], target_yaw=0.6, orders={"none": [HOME]}
    )
    assert result["distance_m"] == 0


def test_touching_voxels_keep_zero_score_after_a_rigid_transform():
    atlas = ReachMap(0.02)
    atlas.add_stance(HOME, [layer("standing", [[0.095, 0.005, 0.215]])])
    # Target x min = cell x max; touching closed volumes have distance zero.
    target_center = np.array([[0.115, 0.01, 0.0]])
    for angle in (0.0, 0.37, 2.09):
        home = placed(HOME, 1.5, -0.29, angle)
        xy = placed(target_center, 1.5, -0.29, angle)[0, :2]
        (hit,) = rank_orders(atlas, home, [*xy, 0.21], [0.03, 0.02, 0.02], target_yaw=angle, orders={"none": [home]})
        assert hit["distance_m"] == 0 and hit["status"] == "hit"


def test_failed_map_save_leaves_previous_map_loadable(tmp_path):
    atlas = ReachMap(0.02)
    path = tmp_path / "map.npz"
    atlas.save(path)
    atlas.metadata["unserializable"] = object()
    with pytest.raises(TypeError):
        atlas.save(path)
    assert ReachMap.load(path).metadata == {}


def test_roundoff_in_equal_gaps_preserves_stable_order():
    atlas = ReachMap(0.1)
    atlas.add_stance(HOME, [layer("standing", [[0.05, 0.05, 0.05]])])
    orders = {"LR": [HOME], "RL": [placed(HOME, 1e-15, 0, 0)]}
    rows = rank_orders(atlas, HOME, [0.20, 0.05, 0.05], [0.02, 0.02, 0.02], orders=orders)
    assert [r["order"] for r in rows] == ["LR", "RL"]
