"""CPU search over exact foot orders and a transported nominal hand workspace."""


import numpy as np
import pytest

from cutamp.landing_accuracy import CHECKPOINT_SHA256, LANDING_ACCURACY, SOURCE_SHA256
from cutamp.reachability import ORDERS, REACH_MAP, ReachMap
from cutamp.step_search import StepSearch, TargetRegion

HOME = ((0.0, 0.10, 0.0), (0.0, -0.10, 0.0))
FWD = (0.12, 0.20, 0.0)


def test_stances_with_identical_frames_reuse_the_same_hand_map_query(monkeypatch):
    from cutamp import step_search

    calls = []
    box_gaps = step_search._box_gaps
    def measured(*args):
        calls.append(1)
        return box_gaps(*args)
    monkeypatch.setattr(step_search, "_box_gaps", measured)
    search = StepSearch(atlas(), commands=())
    region = target(0.005)
    (home,) = search.edges(HOME, region)
    initial_count = len(calls)
    widened = ((0.0, 0.12, 0.0), (0.0, -0.12, 0.0))
    (wide,) = search.edges(widened, region)
    assert home.soles != wide.soles and home.witness == wide.witness
    assert len(calls) == initial_count > 0
    search.edges(widened, target(0.065))
    assert len(calls) > initial_count


def atlas(points=((0.005, 0.005, 0.505),), arm="left"):
    m = ReachMap(0.01, {"home_soles": HOME, "sole_box_m": [0.20, 0.08]})
    m.add_stance(
        HOME,
        [
            {
                "arm": arm,
                "posture": "crouch",
                "points": points,
                "q": [[42.0]] * len(points),
                "margin": [0.06] * len(points),
            }
        ],
    )
    return m


def target(x, **kwargs):
    return TargetRegion("left", (x, 0.005, 0.505), (0.004, 0.004, 0.004), **kwargs)


def stage(foot):
    return {
        "foot": foot,
        "issue_legal": True,
        "handoff": {
            "issued_xy_error_m": [0, 0],
            "issued_yaw_error_deg": [0, 0],
            "nominal_xy_error_m": [0, 0],
            "nominal_yaw_error_deg": [0, 0],
        },
    }


def profile(order, trials):
    provenance = {
        "model_only": True,
        "control_hz": 50,
        "warmup_ticks": 50,
        "hold_ticks": 75,
        "inter_step_hold_ticks": 0,
        "xy_threshold_m": 0.02,
        "yaw_threshold_deg": 1.0,
        "source_sha256": SOURCE_SHA256.copy(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    commands = [FWD if foot == "L" else (FWD[0], -FWD[1], -FWD[2]) for foot in order]
    return {
        "home_soles": HOME,
        "provenance": provenance,
        "schedules": [{"order": order, "commands": commands, "trials": trials}],
    }


def passing(*orders):
    """A lag-8 profile in which each single-step order passed one trial of FWD."""
    records = profile(orders[0], [{"lag": 8, "stages": [stage(orders[0])]}])
    for order in orders[1:]:
        records["schedules"] += profile(order, [{"lag": 8, "stages": [stage(order)]}])["schedules"]
    return records


def test_carry_incoming_feet_and_rank_complete_paths_against_enumeration():
    search = StepSearch(atlas(), accuracy=passing("L", "R"), lag=8, commands=(FWD,))
    goals = (target(0.065), target(0.065))
    initial = search.edges(HOME, goals[0])
    assert [e.order for e in initial] == list(ORDERS)
    assert [e.cost for e in initial] == [(1, 0), (0, 1), (0, 1), (1, 2), (1, 2)]
    second = search.edges(initial[1].soles, goals[1])
    assert second[0].cost == (0, 0)
    assert second[0].posture == "crouch" and second[0].witness == (42.0,)
    exhaustive = []
    for pick in initial:
        for place in search.edges(pick.soles, goals[1]):
            exhaustive.append(tuple(a + b for a, b in zip(pick.cost, place.cost)))
    plans = list(search.schedules(HOME, goals))
    assert [p.cost for p in plans] == sorted(exhaustive)
    assert [e.order for e in plans[0].edges] == ["L", "none"]
    assert (
        any(p.edges[0].order == "L" and p.edges[1].order == "LR" for p in plans) is False
    )  # identical first landing is illegal
    another = StepSearch(atlas(), commands=(FWD, (0.24, 0.20, 0)))
    first = another.edges(HOME, goals[0])
    a = next(e for e in first if e.order == "L")
    assert any(e.order == "L" for e in another.edges(a.soles, goals[1]))  # foot flags reset


def test_missing_map_or_region_is_finite_and_floor_collision_excludes_endpoint():
    search = StepSearch(atlas(), commands=(FWD,))
    unknown = TargetRegion("right", None, None)
    assert next(search.schedules(HOME, [unknown])).cost == (1, 0)
    assert search.edges(HOME, TargetRegion("right", (0.0, 0.0, 0.5), (0.01, 0.01, 0.01)))[0].distance_m is None
    floor = TargetRegion("left", (0.0, 0.1, 0.025), (0.05, 0.05, 0.05), solid_pick=True)
    assert "none" not in [e.order for e in search.edges(HOME, floor)]
    huge = TargetRegion("left", (0.0, 0.0, 0.025), (5.0, 5.0, 0.05), solid_pick=True)
    assert list(search.schedules(HOME, [huge])) == []
    (goal,) = search.schedules(HOME, [])
    assert goal.cost == (0, 0) and goal.edges == ()


def test_only_objects_that_intersect_the_sole_height_band_block_a_landing():
    search = StepSearch(atlas(), commands=())
    for bottom, expected in ((0.006, ("L",)), (0.05, ()), (-0.08, ())):
        cube = TargetRegion("left", (0.0, 0.1, bottom + 0.025), (0.05,) * 3, solid_pick=True)
        assert search.blocked_feet(HOME, cube) == expected


def test_floor_pick_uses_oriented_sole_and_box_interiors():
    search = StepSearch(atlas(), commands=())
    # A thin box crosses the thin left sole at right angles without containing any
    # of its vertices. Translating by .11 m leaves only an edge in contact.
    for shift, expected in ((0, ("L",)), (0.11, ())):
        for angle in (0, 0.73):
            c, s = np.cos(angle), np.sin(angle)
            turn = np.array([[c, s], [-s, c]])
            feet = tuple((*(np.array(sole[:2]) @ turn + (0.3, 0.4)), sole[2] + angle) for sole in HOME)
            xy = np.array([shift, 0.1]) @ turn + (0.3, 0.4)
            cube = TargetRegion("left", (*xy, 0.025), (0.12, 0.02, 0.05), yaw=angle + np.pi / 2, solid_pick=True)
            assert search.blocked_feet(feet, cube) == expected


def test_matching_signed_commands_only_and_transferred_prior():
    records = profile("L", [{"lag": 8, "stages": [stage("L")], "all_handoffs_pass": True}])
    search = StepSearch(atlas(), accuracy=records, lag=8, commands=(FWD,))
    query = target(0.065)
    edges = search.edges(HOME, query)
    l, r = edges[1:3]
    assert l.cost == (0, 1) and l.accuracy == "profiled" and (l.passes, l.trials) == (1, 1)
    assert r.distance_m == 0 and r.cost == (1, 1) and r.accuracy == "unknown"
    shifted = tuple((x + 0.4, y, a) for x, y, a in HOME)
    moved = search.edges(shifted, target(0.465))[1]
    assert moved.cost == l.cost and moved.accuracy == "transferred"
    assert search.edges(HOME, target(0.065, loaded=True))[1].cost == (1, 1)
    assert StepSearch(atlas(), commands=(FWD,), lag=8).edges(HOME, query)[1].cost == (1, 1)


def test_screened_search_prefers_loaded_reach_over_a_known_miss():
    search = StepSearch(atlas(), commands=(FWD,), lag=2)
    # From the starting feet the grasp-independent place query misses by 6 cm.
    # A step overlaps the reach map but has no controller prior with this payload.
    region = target(0.065, loaded=True)
    reference = list(search.schedules(HOME, [region]))
    screened = list(search.schedules(HOME, [region], prefer_reach=True))
    assert reference[0].edges[0].order == "none"
    assert screened[0].edges[0].distance_m == 0 and screened[0].edges[0].accuracy == "unknown"
    assert {plan.edges[0] for plan in reference} == {plan.edges[0] for plan in screened}


def test_accuracy_requires_each_issued_pair_legal_and_final_original_pair():
    trials = [{"lag": 8, "stages": [stage("L"), stage("R")], "all_handoffs_pass": True} for _ in range(6)]
    trials[0]["stages"][0]["handoff"]["issued_xy_error_m"][1] = 0.020  # planted foot, strict boundary
    trials[1]["stages"][1]["handoff"]["nominal_yaw_error_deg"][0] = 1.0  # aggregate drift
    trials[2]["stages"][1]["issue_legal"] = False
    trials[3]["stages"][0]["handoff"]["nominal_xy_error_m"][0] = 0.022  # intermediate nominal only
    del trials[4]["stages"][1]["handoff"]["issued_xy_error_m"]
    trials[5].pop("stages")  # an old/reduced unversioned flag is insufficient
    seen = StepSearch(atlas(), accuracy=profile("LR", trials), lag=8, commands=(FWD,))
    edge = next(e for e in seen.edges(HOME, target(0.065)) if e.order == "LR")
    assert (edge.passes, edge.trials) == (1, 4)


@pytest.mark.parametrize("field", ["source_sha256", "checkpoint_sha256", "hold_ticks"])
def test_different_model_or_handoff_profile_does_not_supply_accuracy(field):
    records = profile("L", [{"lag": 8, "all_handoffs_pass": True, "stages": [stage("L")]}])
    records["provenance"][field] = {} if field == "source_sha256" else "other"
    with pytest.raises(ValueError, match="incompatible"):
        StepSearch(atlas(), accuracy=records, lag=8, commands=(FWD,))


def test_yawed_voxels_keep_pose_and_do_not_fill_sampling_gaps():
    search = StepSearch(atlas(points=((0.005, 0.005, 0.505), (0.155, 0.005, 0.505))), commands=())
    a = 0.731
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    feet = tuple((*((R @ np.array(s[:2])) + [0.43, 0.2]), a) for s in HOME)
    for x in (0.011, 0.08):
        query = TargetRegion("left", (x, 0.005, 0.505), (0.004, 0.006, 0.01), yaw=0.73)
        xy = R @ np.array(query.center[:2]) + [0.43, 0.2]
        moved = TargetRegion("left", (*xy, 0.505), query.size, yaw=0.73 + a)
        assert search.edges(feet, moved)[0].distance_m == pytest.approx(
            search.edges(HOME, query)[0].distance_m, abs=1e-12
        )
    assert search.edges(HOME, target(0.08))[0].cost == (1, 0)


def test_saved_low_pick_matches_geometric_and_lag8_contributor_counts():
    search = StepSearch.load(
        REACH_MAP, accuracy_path=LANDING_ACCURACY, lag=8
    )
    pick = TargetRegion("left", (0.48, 0.18, 0.025), (0.05, 0.05, 0.05), solid_pick=True)
    place = TargetRegion("left", (0.30, -0.40, 0.65), (0.05, 0.05, 0.05), loaded=True)
    edges = search.edges(search.home, pick)
    assert [sum(e.order == o and e.distance_m == 0 for e in edges) for o in ORDERS] == [0, 4, 4, 25, 22]
    assert [sum(e.order == o and e.cost[0] == 0 for e in edges) for o in ORDERS] == [0, 1, 0, 0, 0]
    assert next(e for e in edges if e.cost == (0, 1)).commands == ((0.24, 0.21, 0.0),)
    best = next(search.schedules(search.home, (pick, place)))
    assert [e.order for e in best.edges] == ["L", "none"] and best.cost == (1, 1)
    assert best.edges[0].commands == ((0.12, 0.21, 0.0),)
    unloaded = TargetRegion("left", place.center, place.size)
    alternative = next(search.schedules(search.home, (pick, unloaded)))
    assert [e.order for e in alternative.edges] == ["L", "none"] and alternative.cost == (1, 1)


def test_scene_pick_and_place_geometry_use_each_pose_once():
    from types import SimpleNamespace

    from scipy.spatial.transform import Rotation

    quat = Rotation.from_euler("z", 90.0, degrees=True).as_quat()[[3, 0, 1, 2]].tolist()
    obj = SimpleNamespace(
        pose=[1.0, 2.0, 0.2, *quat],
        get_trimesh_mesh=lambda: SimpleNamespace(bounds=np.array([[0.1, 0.1, 0.0], [0.3, 0.3, 0.4]])),
    )
    surface = SimpleNamespace(dims=[0.8, 0.6, 0.1], pose=[0.2, -0.4, 0.7, *quat])
    p = TargetRegion.pick("left", obj)
    np.testing.assert_allclose(p.center, [0.8, 2.2, 0.4])
    assert p.yaw == pytest.approx(np.pi / 2) and p.size == pytest.approx([0.2, 0.2, 0.4])
    assert p.coarse and not p.solid_pick
    p = TargetRegion.place("right", obj, surface)
    assert p.center == pytest.approx([0.2, -0.4, 0.95]) and p.size == pytest.approx([0.8, 0.6, 0.4])
    assert p.yaw == pytest.approx(np.pi / 2) and p.loaded
    assert TargetRegion.place("left", obj, None).center is None
    cube = SimpleNamespace(dims=(0.1, 0.1, 0.1), pose=(0.0, 0.1, 0.05, 1, 0, 0, 0))
    assert TargetRegion.pick("left", cube).solid_pick
    tilt = Rotation.from_euler("x", 20, degrees=True).as_quat()[[3, 0, 1, 2]].tolist()
    assert TargetRegion.place("left", obj, SimpleNamespace(dims=(1, 1, 0.02), pose=[0, 0, 0.5, *tilt])).center is None


def test_fixed_order_schedule_keeps_endpoint_alternatives_and_validates_length():
    s = StepSearch(atlas(), accuracy=passing("L"), lag=8, commands=(FWD, (0.24, 0.20, 0)))
    targets = (target(0.065), target(0.065))
    paths = list(s.schedules(HOME, targets, orders=("L", "none")))
    assert len(paths) == 2 and [p.cost for p in paths] == [(0, 1), (2, 1)]
    with pytest.raises(ValueError):
        list(s.schedules(HOME, targets, orders=("LL", "none")))
    with pytest.raises(ValueError):
        list(s.schedules(HOME, targets, orders=("L",)))


def test_nominal_build_samples_one_stance_for_both_arms_and_crouch(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from cutamp.scripts.t1_reachability import build_map

    sampled = []

    class Sampler:
        home_soles = HOME

        def __init__(self, **kwargs):
            self.metadata = {"sole_box_m": [0.20, 0.08]}

        def map_stance(self, soles):
            sampled.append(soles.copy())
            return [
                {"arm": arm, "posture": posture, "points": [[0.1, 0.2, 0.3]], "q": [[0.1]], "margin": [0.06]}
                for arm in ("left", "right")
                for posture in ("standing", "crouch")
            ]

    monkeypatch.setitem(sys.modules, "cutamp.reachability_fk", SimpleNamespace(T1MapSampler=Sampler))
    path = tmp_path / "nominal.npz"
    build_map(path, 10, 7, 0.015, "cuda", nominal_only=True)
    assert len(sampled) == 1
    saved = ReachMap.load(path)
    assert [r["posture"] for r in saved.layers(HOME, "right")] == ["standing", "crouch"]
    assert saved.metadata["nominal_only"] is True
