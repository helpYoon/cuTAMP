"""Exercise the public planner with synthetic refinement and a real symbolic domain."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import cutamp.algorithm as alg
from cutamp import t1_domain
from cutamp.task_planning import landing_options
from cutamp.config import TAMPConfiguration
from cutamp.step_planner import StepInputs, StepResult, StepRow, furthest_endpoint
from cutamp.tests.conftest import first_schedules


def test_predictor_cache_is_scoped_to_deployment_stream_and_worker(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from cutamp import landing_accuracy, landing_predictor

    bundle = tmp_path / "landing.pt"
    bundle.write_bytes(b"deployment")
    monkeypatch.setattr(landing_accuracy, "LANDING_BUNDLE", bundle)
    config = TAMPConfiguration()
    stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(landing_predictor, "LandingPredictor", lambda path, **kw: SimpleNamespace(
        device=stream.device, metadata={"deployment_sha256": "deployment"}, **kw
    ))

    model = alg._load_predictor(config)
    assert alg._load_predictor(config) is model and model.use_cuda_graph
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(alg._load_predictor, config).result() is not model
    stream.cuda_stream = 1
    other = alg._load_predictor(config)
    assert other is not model
    bundle.write_bytes(b"changed deployment")
    assert alg._load_predictor(config) is not other
    eager = replace(config, cuda_graphs=False)
    assert alg._load_predictor(eager) is not alg._load_predictor(eager)
    bundle.unlink()
    with pytest.raises(FileNotFoundError):
        alg._load_predictor(config)


def test_endpoint_prefix_belongs_to_one_row():
    assert furthest_endpoint(torch.tensor([[True, False], [False, True]])) == 1
    assert furthest_endpoint(torch.tensor([[False, False]])) == 0
    assert furthest_endpoint(torch.ones((0, 2), dtype=torch.bool)) is None


@pytest.fixture
def harness(monkeypatch, tmp_path):
    from cutamp import landing_accuracy
    from cutamp.task_planning.landing_options import LandingOptions
    from cutamp.tests.test_landing_options import atlas

    goal = frozenset([t1_domain.On.ground("block", "table")])
    world = SimpleNamespace(
        initial_state=t1_domain.get_initial_state(["block"], ["table"]),
        goal_state=goal,
        grasps={},
        device="cpu",
        get_object=lambda name: SimpleNamespace(name=name, pose=(0.3, 0, 0.55, 1, 0, 0, 0), dims=(0.1, 0.1, 0.1)),
        stance_soles=torch.eye(4).repeat(2, 1, 1),
        arm_home_ee_world={"left": torch.tensor([0., .2, .5]), "right": torch.tensor([0., -.2, .5])},
    )
    world.stance_soles[:, 1, 3] = torch.tensor([0.1, -0.1])
    seen = []
    timer = SimpleNamespace(
        time=lambda *a, **kw: nullcontext(), start=lambda key: None, stop=lambda key: 0.0, get_summaries=dict
    )
    monkeypatch.setattr(
        alg, "setup_cutamp", lambda *a, **kw: (SimpleNamespace(log_dict=lambda *a: None), None, timer, world)
    )
    monkeypatch.setattr(alg, "grasp_sets", lambda objs, dev, **kw: seen.append("generate") or {"block": object()})
    monkeypatch.setattr(alg, "start_sampler_load", lambda: seen.append("load"))
    monkeypatch.setattr(alg, "wait_for_sampler_load", lambda: seen.append("wait"))

    class Search(LandingOptions):
        def edges(self, *args, **kwargs):
            seen.append("rank")
            return super().edges(*args, **kwargs)

    # The left hand reaches the block and its place from the home stance.
    reach = atlas(points=((.3, .005, .555), (.3, .005, .655)))
    monkeypatch.setattr(landing_options, "load_landing_options",
                        lambda cfg, **kw: (Search(reach, commands=((0.12, 0.21, 0),)), {}))
    monkeypatch.setattr(landing_accuracy, "LANDING_BUNDLE", tmp_path / "no-checkpoint")
    cfg = {
        "num_initial_plans": 1,
        "enable_visualizer": False,
        "curobo_plan": False,  # exercise the keyframe search unless a test requests motion
        "experiment_root": str(tmp_path),
    }

    def run(**options):
        return alg.run_cutamp(
            None,
            TAMPConfiguration(**(cfg | options)),
            SimpleNamespace(cost_config={}),
            SimpleNamespace(constraint_config={}),
            experiment_id="test",
        )

    return run, seen, world


def _no_refinement(monkeypatch):
    """Record the search's attempts without optimizing any of them."""
    def unresolved(*args, **kwargs):
        return StepResult([], None, "unresolved")

    monkeypatch.setattr(alg, "_load_predictor", lambda config: object())
    monkeypatch.setattr(alg, "_refine_step_schedule", unresolved)


@pytest.mark.parametrize("options,expected", [({}, None), ({"max_loop_dur": 2.}, 2.)])
def test_search_deadline_scope_and_world_setup(harness, monkeypatch, options, expected):
    run, seen, _ = harness
    clock = [alg.monotonic()]
    monkeypatch.setattr(alg, "monotonic", lambda: clock[0])
    setup = alg.setup_cutamp

    def slow_setup(*args, **kwargs):
        clock[0] += 9.
        return setup(*args, **kwargs)

    monkeypatch.setattr(alg, "setup_cutamp", slow_setup)
    _no_refinement(monkeypatch)
    first_schedules(monkeypatch, 1)
    _, _, metrics = run(**options)
    assert metrics["search_budget_s"] == expected
    if expected is not None:
        assert metrics["time_limit_reached"] and metrics["status"] == "unresolved"
        assert metrics["num_optimized_plans"] == 0 and not seen and not metrics["step_search"]
    else:
        assert not metrics["time_limit_reached"] and len(metrics["step_attempts"]) == 1


def test_search_deadline_passed_in_the_model_load_stops_the_plan(harness, monkeypatch):
    run, _, _ = harness
    clock = [alg.monotonic()]
    monkeypatch.setattr(alg, "monotonic", lambda: clock[0])

    def expensive_model_load(config):
        clock[0] += 10.
        return object()

    monkeypatch.setattr(alg, "_load_predictor", expensive_model_load)
    monkeypatch.setattr(alg, "_refine_step_schedule", lambda *a, **kw: pytest.fail("started after deadline"))
    monkeypatch.setattr(alg, "play_curobo_motion", lambda *a: None)
    motion, count, metrics = run(curobo_plan=True, max_loop_dur=9.5)
    assert metrics["search_budget_s"] == 9.5
    assert motion is None and count == 0
    assert metrics["time_limit_reached"] and not metrics["found_solution"]
    assert metrics["step_attempts"][-1]["status"] == "budget_exhausted"


@pytest.mark.parametrize("explicit", [None, 12.])
def test_full_motion_returns_on_success_with_time_remaining(harness, monkeypatch, explicit):
    run, _, _ = harness
    clock = [alg.monotonic()]
    monkeypatch.setattr(alg, "monotonic", lambda: clock[0])
    row = StepRow(1, {}, 0., {"com_center_distances_m": [.02, .03], "secondary_effort": 0., "score": 0.,
                              "rank": 0., "joint_effort": 0.}, ["complete"])

    def refine(*args, **kwargs):
        clock[0] += 5.2
        return StepResult([row], None, "model_feasible")

    monkeypatch.setattr(alg, "_load_predictor", lambda config: object())
    monkeypatch.setattr(alg, "_refine_step_schedule", refine)
    monkeypatch.setattr(alg, "play_curobo_motion", lambda *a: None)
    motion, _, metrics = run(curobo_plan=True, max_loop_dur=explicit)
    assert motion == ["complete"]
    assert len(metrics["step_attempts"]) == 1
    assert not metrics["time_limit_reached"]


@pytest.mark.parametrize("side,near_arm", [(-.15, "right"), (.15, "left")])
@pytest.mark.parametrize("near_can_reach", [True, False])
def test_plan_search_charges_arm_choice_before_planning_the_pick(harness, monkeypatch, side, near_arm, near_can_reach):
    from cutamp.reachability import ReachMap
    from cutamp.task_planning.landing_options import LandingOptions

    run, seen, world = harness
    homes = {"left": torch.tensor([0., .2, .5]), "right": torch.tensor([0., -.2, .5])}
    world.arm_home_ee_world = homes
    far_arm = "left" if near_arm == "right" else "right"
    pick = (.8, side, .5)
    pad = (.85, side, .45)
    world.get_object = lambda name: SimpleNamespace(
        name=name, dims=(.1, .1, .1), pose=(*(pick if name == "block" else pad), 1., 0., 0., 0.)
    )
    soles = ((0., .1, 0.), (0., -.1, 0.))
    atlas = ReachMap(.01, {"home_soles": soles})
    layers = []
    for arm in homes:
        dx = -1.2 if arm == near_arm and not near_can_reach else 0.
        layers.append({
            "arm": arm, "posture": "stand", "q": [[0.], [0.]], "margin": [.06, .06],
            "points": [(pick[0] + dx, side, .5), (pad[0] + dx, side, .55)],
        })
    atlas.add_stance(soles, layers)
    search = LandingOptions(atlas, commands=((.12, .2, 0),))
    queried = []
    edges = search.edges

    def record_edges(incoming, target):
        queried.append(target.arm)
        return edges(incoming, target)

    monkeypatch.setattr(search, "edges", record_edges)
    monkeypatch.setattr(landing_options, "load_landing_options", lambda *args, **kwargs: (search, {}))
    # Always declare the expected winner last, so declaration order alone cannot pass.
    winner = near_arm if near_can_reach else far_arm
    ops = sorted(t1_domain.all_t1_operators, key=lambda op: op.metadata.arm == winner)
    monkeypatch.setattr(alg, "all_t1_operators", ops)
    _no_refinement(monkeypatch)
    first_schedules(monkeypatch, 1)
    _, _, metrics = run(num_initial_plans=2)

    attempt, = metrics["step_attempts"]
    assert any(op.startswith(f"{winner.capitalize()}Pick(block,") for op in attempt["skeleton"])
    assert attempt["orders"] == ["none", "none"]
    approach = (torch.tensor(pick) - homes[winner]).norm().item()
    assert attempt["arm_approach_m"] == pytest.approx(approach)
    assert attempt["search_priority"] == pytest.approx([approach, 0, 0])
    # Opening the near-arm branch requires only its approach cost. If both tasks
    # can be reached with that hand and fixed feet, A* can return it immediately.
    assert queried[0] == near_arm
    assert set(queried) == ({near_arm} if near_can_reach else {near_arm, far_arm})
    assert seen.count("generate") == 1  # only for the one attempted skeleton


def test_arm_approach_accumulates_only_picks_in_their_incoming_stances(harness, monkeypatch):
    from math import cos, dist, sin
    from cutamp.reachability import ReachMap, canonical_stance
    from cutamp.task_planning.landing_options import LandingOptions

    run, _, world = harness
    points = {"a": (.88, .12, .50), "b": (.88, -.12, .50), "pad": (.88, 0., .48)}
    world.initial_state = t1_domain.get_initial_state(["a", "b"], ["pad"])
    world.goal_state = frozenset(t1_domain.On.ground(obj, "pad") for obj in ("a", "b"))
    world.get_object = lambda name: SimpleNamespace(name=name, dims=(.02,) * 3, pose=(*points[name], 1., 0., 0., 0.))
    world.arm_home_ee_world = {arm: torch.tensor([0., y, .50]) for arm, y in (("left", .2), ("right", -.2))}
    soles = ((0., .1, 0.), (0., -.1, 0.))
    atlas = ReachMap(.01, {"home_soles": soles})
    samples = [(.76, y, .50) for y in (.12, -.12, 0.)]
    atlas.add_stance(soles, [
        dict(arm=arm, posture="stand", points=samples, q=[[0.]] * 3, margin=[.06] * 3)
        for arm in world.arm_home_ee_world
    ])
    search = LandingOptions(atlas, commands=((.24, .20, 0.),))
    monkeypatch.setattr(landing_options, "load_landing_options", lambda *args, **kwargs: (search, {}))
    _no_refinement(monkeypatch)
    first_schedules(monkeypatch, 3)

    _, _, metrics = run(num_initial_plans=2)
    checked_shift = False
    for attempt in metrics["step_attempts"]:
        actions = [op for op in attempt["skeleton"] if "Pick(" in op or "Place(" in op]
        charged = []
        for op, gap in zip(actions, attempt["gaps"], strict=True):
            if "Pick(" in op:
                arm, args = op.split("Pick(")
                target = points[args.split(",")[0]]
                _, (x, y, yaw) = canonical_stance(gap["incoming_soles"])
                hx, hy, hz = world.arm_home_ee_world[arm.lower()].tolist()
                hand = (x + hx * cos(yaw) - hy * sin(yaw), y + hx * sin(yaw) + hy * cos(yaw), hz)
                charged.append(dist(hand, target))
                checked_shift |= len(charged) == 2 and dist((x, y), (0., 0.)) > .01
        assert len(charged) == 2
        assert attempt["arm_approach_m"] == pytest.approx(sum(charged))
    assert checked_shift


def test_pose_still_ranks_both_hands_when_object_reach_bounds_are_unknown(harness, monkeypatch):
    from cutamp.task_planning.landing_options import LandingOptions
    from cutamp.tests.test_landing_options import atlas

    run, _, world = harness
    pick = (.5, -.15, .5)
    world.get_object = lambda name: SimpleNamespace(name=name, pose=(*pick, 1., 0., 0., 0.))
    homes = {"left": torch.tensor([0., .2, .5]), "right": torch.tensor([0., -.2, .5])}
    world.arm_home_ee_world = homes
    search = LandingOptions(atlas(), commands=())
    monkeypatch.setattr(landing_options, "load_landing_options", lambda *a, **kw: (search, {}))
    _no_refinement(monkeypatch)
    first_schedules(monkeypatch, 2)
    _, _, metrics = run(num_initial_plans=2)

    right, left = metrics["step_attempts"]
    for attempt, arm in ((right, "right"), (left, "left")):
        assert any(op.startswith(f"{arm.capitalize()}Pick(block,") for op in attempt["skeleton"])
        assert attempt["missing_reach_evidence"] and attempt["gaps"][0]["distance_m"] is None
        distance = (torch.tensor(pick) - homes[arm]).norm().item()
        assert attempt["arm_approach_m"] == pytest.approx(distance)
        assert attempt["search_priority"] == pytest.approx([distance, 2, 0])


@pytest.mark.parametrize("orders", [("L", "none"), ("none", "none")])  # a plan without steps needs the model too
def test_missing_predictor_never_marks_model_feasible_or_draws_grasps(harness, monkeypatch, orders):
    from cutamp.tests.conftest import only_schedule

    run, seen, _ = harness
    only_schedule(monkeypatch, orders, "left")
    _, n, m = run()
    assert n == 0 and not m["found_solution"] and m["status"] == "unavailable"
    assert m["step_attempts"][0]["status"] == "unavailable"
    assert "bundle" in m["step_attempts"][0]["reason"].lower()
    assert "generate" not in seen


def test_every_legal_order_pair_once_then_the_search_runs_out(harness, monkeypatch):
    from cutamp.task_planning.landing_options import TargetRegion

    run, seen, world = harness
    empty = StepResult([], None, "initialization_failed")
    monkeypatch.setattr(alg, "_load_predictor", lambda config: object())
    monkeypatch.setattr(alg, "_refine_step_schedule", lambda *a, **k: empty)
    _, n, metrics = run()
    attempts = metrics["step_attempts"]
    assert n == 0 and metrics["status"] == "unresolved"
    assert attempts[0]["orders"] == ["none", "none"]
    assert seen.index("rank") < seen.index("generate") and seen.count("generate") == 1
    assert seen.index("generate") < seen.index("wait")
    search, _ = landing_options.load_landing_options(None)
    block = world.get_object("block")
    pick, place = TargetRegion.pick("left", block), TargetRegion.place("left", block, world.get_object("table"))
    home = ((0., .1, 0.), (0., -.1, 0.))
    legal = {(a.order, b.order) for a in search.edges(home, pick) for b in search.edges(a.soles, place)}
    assert len({tuple(a["orders"]) for a in attempts}) == len(attempts) == len(legal) < 25
    assert {tuple(a["orders"]) for a in attempts} == legal


def test_retry_failed_skeletons_then_return_best_connected_row(harness, monkeypatch):
    run, _, _ = harness
    calls = []
    monkeypatch.setattr(alg, "_load_predictor", lambda *a: object())
    monkeypatch.setattr(alg, "play_curobo_motion", lambda *a: None)

    def refine(p, *args, **kwargs):
        if ("R", "none") in calls:
            pytest.fail("refined another skeleton after success")
        calls.append(p.orders)
        if not p.step_count:  # the first plan takes no step and fails
            return StepResult([], torch.tensor([[False, True]]), "unresolved")
        rows = [
            StepRow(7, {}, 0.2, {"com_center_distances_m": [.001, .05], "secondary_effort": .1,
                                 "score": 10., "rank": 10., "joint_effort": .1}),
            StepRow(8, {"q": torch.tensor([8])}, 0.6, {"com_center_distances_m": [.04, .04], "secondary_effort": .6,
                                                       "score": 8., "rank": 8., "joint_effort": .6}),
            StepRow(9, {}, 0.1, {"com_center_distances_m": [.001, .001], "secondary_effort": 0.,
                                 "score": 0., "rank": 0., "joint_effort": 0.}),
        ]
        if p.orders == ("R", "none"):
            rows[0].motion_plan = ["less centered"]
            rows[1].motion_plan = ["motion"]
        return StepResult(rows, torch.ones(len(rows), 2, dtype=torch.bool), "model_feasible")

    monkeypatch.setattr(alg, "_refine_step_schedule", refine)
    motion, count, m = run(curobo_plan=True)
    assert motion == ["motion"] and m["motion_plan_success"] and m["selected_particle"]["q"] == [8]
    assert count == 2 and m["row_index"] == 8 and m["best_soft_cost"] == 0.6 and m["max_center_distance_m"] == .04
    assert calls[0] == ("none", "none") and calls[-1] == ("R", "none") and len(calls) > 2
    assert [(a["status"], a["accepted_rows"]) for a in m["step_attempts"]] == [
        ("unresolved", 0), *[("model_feasible", 0)] * (len(calls) - 2), ("model_feasible", 2),
    ]
    assert not m["time_limit_reached"]


@pytest.mark.parametrize("steps", [0, 1])  # the winning plan takes no step (the first plan), or one (the second)
@pytest.mark.parametrize("weight,winning_id", [(1.0, 2), (0.1, 2), (0.0, 1)])
def test_motion_feasibility_then_rank_then_joint_effort_in_successful_skeleton(
    harness, monkeypatch, weight, winning_id, steps
):
    run, _, _ = harness
    viewer, displayed = object(), []
    setup = alg.setup_cutamp
    def with_viewer(*args, **kwargs):
        logger, _, timer, world = setup(*args, **kwargs)
        return logger, viewer, timer, world
    monkeypatch.setattr(alg, "setup_cutamp", with_viewer)
    monkeypatch.setattr(alg, "play_curobo_motion", lambda world, motion, vis: displayed.append((motion, vis)))
    monkeypatch.setattr(alg, "_load_predictor", lambda *a: object())

    def refine(proposal, *args, **kwargs):
        assert kwargs["visualizer"] is viewer  # the Adam loop's; the route keeps it from the motion stage when centring
        if proposal.step_count != steps:
            return StepResult([], None, "unresolved")
        candidates = [
            (1, [.0, .03], .01, .0, 4102., ["better centred, leans a unit more"]),
            (2, [.049, .049], 8., 1., 9., ["selected"]),
            (3, [.049, .005], 6., 2., 9., ["higher effort"]),
            (4, [.001, .001], .001, .001, 0., None),
        ]
        rows = [
            StepRow(i, {}, soft, {"gaps": [{"center_distance_m": x} for x in ds], "secondary_effort": effort,
                                  "score": rank, "rank": rank, "joint_effort": effort}, motion)
            for i, ds, soft, effort, rank, motion in candidates
        ]
        return StepResult(rows, None, "model_feasible")

    monkeypatch.setattr(alg, "_refine_step_schedule", refine)
    motion, _, metrics = run(curobo_plan=True, com_center_weight=weight)
    assert metrics["row_index"] == winning_id
    assert len(metrics["step_attempts"]) == steps + 1 and metrics["total_steps"] == steps
    if winning_id == 2:
        assert motion == ["selected"]
        assert displayed == [(motion, viewer)]
        assert metrics["max_center_distance_m"] == .049
        assert metrics["secondary_effort"] == 1.
    else:
        assert motion == ["better centred, leans a unit more"]


def test_secondary_effort_excludes_only_centering_even_with_custom_weights():
    from cutamp.cost_reduction import CostReducer

    reducer = CostReducer({"soft": {"com_center_distance_squared": 17., "posture": 2.}, "length": {"travel": 3.}})
    costs = {
        "soft": {"type": "cost", "values": {
            "com_center_distance_squared": torch.tensor([[2., 0.], [0., 3.]]),
            "posture": torch.tensor([[1., 1.], [1., 2.]]),
        }},
        "length": {"type": "cost", "values": {"travel": torch.tensor([1., 2.])}},
        "collision": {"type": "constraint", "values": {"violation": torch.tensor([10., 20.])}},
    }
    assert reducer.secondary_effort(costs).tolist() == [7., 12.]
    assert reducer.soft_costs(costs).tolist() == [41., 63.]


def test_logs_only_the_final_accepted_rows_model_evidence(harness, monkeypatch, tmp_path):
    run, _, _ = harness
    evidence_calls = []
    costs = {"L": 2.0, "R": 1.0}
    setup = alg.setup_cutamp

    def recording_setup(*args, **kwargs):
        logger, viz, timer, world = setup(*args, **kwargs)
        logger.log_dict = lambda name, data: tmp_path / (name + ".json")
        return logger, viz, timer, world

    monkeypatch.setattr(alg, "setup_cutamp", recording_setup)
    monkeypatch.setattr(alg, "_load_predictor", lambda *a: object())

    def refine(proposal, *args, **kwargs):
        if proposal.orders[1] != "none" or proposal.orders[0] not in costs:
            return StepResult([], None, "unresolved")
        side = proposal.orders[0]
        row = StepRow(2, {}, costs[side], {"gaps": [{"center_distance_m": .04}], "secondary_effort": costs[side],
                                           "score": 8., "rank": 8., "joint_effort": costs[side]},
                      evidence=lambda: evidence_calls.append(side) or {"side": side})
        return StepResult([row], None, "model_feasible")

    monkeypatch.setattr(alg, "_refine_step_schedule", refine)
    _, _, result = run()
    assert result["gap_schedule"] == ["R", "none"]
    assert evidence_calls == ["R"]
    assert result["model_evidence_path"] == str(tmp_path / "model_evidence.json")


def test_default_centroid_preference_and_zero_weight_ablation():
    from dataclasses import replace

    from cutamp.config import validate_tamp_config
    from cutamp.scripts.utils import default_constraint_to_mult

    cfg = TAMPConfiguration()
    assert cfg.com_center_weight == 1
    assert default_constraint_to_mult["soft"]["com_center_distance_squared"] == 1
    validate_tamp_config(replace(cfg, com_center_weight=0))
    for invalid in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="com_center_weight"):
            validate_tamp_config(replace(cfg, com_center_weight=invalid))


def test_particle_and_motion_defaults():
    assert TAMPConfiguration().num_particles == 128
    assert TAMPConfiguration().curobo_plan
    assert TAMPConfiguration(num_particles=8).num_particles == 8


@pytest.mark.parametrize("limit,expire,expected", [(1, False, [5]), (5, False, [5, 2]), (5, True, [5])])
def test_stepped_motion_honors_global_retry_order_and_deadline(monkeypatch, limit, expire, expected):
    from cutamp import step_loop

    # Row 5 is the best of three rows sharing a grasp, row 2 has a distinct grasp.
    rows = [
        StepRow(i, {"grasp1_idx": torch.tensor(grasp), "q1": torch.zeros(33)}, soft, {"best_com_margin": 0.06})
        for i, grasp, soft in [(0, 0, 4.0), (1, 0, 0.3), (2, 1, 1.0), (5, 0, 0.1)]
    ]
    pools = {"grasp1": (torch.zeros(2, 7), torch.ones(2))}
    calls = []
    monkeypatch.setattr(step_loop, "monotonic", lambda: 1.0 if expire and calls else 0.0)
    monkeypatch.setattr(step_loop, "worst_posture", lambda world, confs: (torch.zeros(4),) * 3)
    op = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="pick")), values=("q1",))

    def attempt(row):
        calls.append(row.index)
        if row.index == 2:
            return ["validated path"], {"motion_validation": "accepted"}
        return None, {"motion_validation": "rejected", "reason": "connection failed"}

    cfg = TAMPConfiguration(motion_plan_max_retries=limit, com_center_weight=0)
    report = step_loop._validate_motion_rows(rows, pools, [op], None, cfg, attempt, deadline=0.5)
    assert calls == expected
    assert report["motion_plan_attempts"] == len(calls)
    assert report["motion_plan_failures"] == ["connection failed"]
    winners = [r for r in rows if r.motion_plan is not None]
    assert [r.index for r in winners] == ([2] if 2 in expected else [])
    if winners:
        assert winners[0].metrics["motion_plan_rank"] == 1 and winners[0].soft_cost == 1.0


def test_stepped_motion_with_no_grasp_columns_uses_particle_cost(monkeypatch):
    from cutamp import step_loop

    op = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="place")), values=("q1",))
    rows = [StepRow(i, {"q1": torch.zeros(33)}, cost, {"best_com_margin": 0.08}) for i, cost in enumerate((2., 1.))]
    monkeypatch.setattr(step_loop, "worst_posture", lambda w, qs: (torch.zeros(2),) * 3)
    report = step_loop._validate_motion_rows(
        rows, {}, [op], None, TAMPConfiguration(com_center_weight=0), lambda r: ([r.index], {"motion_validation": "accepted"}),
        deadline=None,
    )
    assert report["motion_plan_attempts"] == 1 and rows[1].motion_plan == [1]


def test_rank_assets_are_reused_and_regeneration_invalidates_cache(tmp_path, monkeypatch):
    import os
    from dataclasses import replace

    from cutamp import landing_accuracy, reachability
    from cutamp.reachability import ReachMap

    path = tmp_path / "nominal.npz"
    monkeypatch.setattr(reachability, "REACH_MAP", path)
    monkeypatch.setattr(landing_accuracy, "LANDING_ACCURACY", tmp_path / "missing.json")
    cfg = TAMPConfiguration()
    with pytest.raises(FileNotFoundError, match="Reach map"):
        landing_options.load_landing_options(cfg)
    path.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="Reach map"):
        landing_options.load_landing_options(cfg)
    home = [[0, 0.1, 0], [0, -0.1, 0]]
    ReachMap(0.01, {"home_soles": home}).save(path)
    first, info = landing_options.load_landing_options(cfg)
    assert info["accuracy_status"] == "unavailable"
    assert landing_options.load_landing_options(cfg)[0] is first
    ReachMap(0.01, {"home_soles": home}).save(tmp_path / "regenerated.npz")
    os.replace(tmp_path / "regenerated.npz", path)
    second, _ = landing_options.load_landing_options(cfg)
    assert second is not first
    assert landing_options.load_landing_options(cfg)[0] is second
    assert landing_options.load_landing_options(replace(cfg, step_lag=5))[0] is not second


def test_zero_step_schedule_comes_first(harness, monkeypatch):
    run, _, _ = harness

    def refine(*args, **kwargs):
        return StepResult(
            [StepRow(0, {"q": torch.zeros(2)}, 0.1,
                     {"min_com_margin_m": 0.06, "com_center_distances_m": [.02, .03], "secondary_effort": .09,
                      "score": 0., "rank": 0., "joint_effort": .09})],
            torch.ones(1, 2, dtype=torch.bool),
            "model_feasible",
        )

    monkeypatch.setattr(alg, "_load_predictor", lambda config: object())
    monkeypatch.setattr(alg, "_refine_step_schedule", refine)
    first_schedules(monkeypatch, 1)
    _, n, m = run()
    assert n == 1 and m["total_steps"] == 0 and m["status"] == "model_feasible"


from cutamp.tests.conftest import needs_cuda, needs_graspgenx


@needs_cuda
@needs_graspgenx
@pytest.mark.parametrize("with_motion", [False, True])
def test_real_pick_only_checks_bound_configuration_and_returns_its_row(tmp_path, monkeypatch, with_motion):
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs.utils import load_env, resolve_env_path
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.t1_domain import all_t1_fluents
    from cutamp.tests.conftest import only_schedule

    env = load_env(resolve_env_path("blocks_t1"))
    env.goal_state = frozenset([next(f for f in all_t1_fluents if f.name == "LeftHolding").ground("block3")])
    block = next(obj for obj in env.movables if obj.name == "block3")
    block.pose[:3] = [0.38, 0.23, 0.73]
    cfg = TAMPConfiguration(
        curobo_plan=with_motion,
        num_initial_plans=1,
        num_particles=16,
        num_opt_steps=20,
        enable_visualizer=False,
        cuda_graphs=False,
        experiment_root=str(tmp_path),
    )
    only_schedule(monkeypatch, ("none",), "left")
    first_schedules(monkeypatch, 1)
    torch.manual_seed(1)
    motion, n, metrics = alg.run_cutamp(
        env,
        cfg,
        CostReducer(default_constraint_to_mult),
        ConstraintChecker(default_constraint_to_tol),
        experiment_id="standing-pick",
    )
    # A plan without steps runs the stepped route on the feet the robot stands on; its motion predicts no landing.
    assert n > 0 and metrics["total_steps"] == 0
    if with_motion:
        assert motion is not None and metrics["status"] == "model_predicted_motion"
        assert metrics["max_predicted_landing_xy_m"] is None and metrics["max_predicted_landing_yaw_deg"] is None
    else:
        assert motion is None and metrics["status"] == "model_feasible"
    assert metrics["min_com_margin_m"] == min(metrics["com_margins_m"]) >= cfg.com_comfort_margin_m
    assert metrics["selected_particle"]["left_q1"] != metrics["selected_particle"]["left_q0"]
    assert metrics["row_index"] < cfg.num_particles and metrics["best_soft_cost"] >= 0


def test_unmatched_dynamics_never_receive_reference_tracking_credit():
    from cutamp.landing_predictor import LandingState
    from cutamp.task_planning.landing_options import TargetRegion

    cfg = TAMPConfiguration(step_lag=8)
    d = torch.zeros(1, 38)
    d[:, 0] = 0.25 / 0.45
    d[:, -5:] = 1.0
    assert StepInputs(LandingState((), d, "synthetic_home_50_hold")).matches_accuracy_context(8)
    d[0, 1] = 0.25  # encoder offset differs from the unloaded home profile
    assert not StepInputs(LandingState((), d, "synthetic_home_50_hold")).matches_accuracy_context(8)
    search, meta = landing_options.load_landing_options(cfg, applicable=False)
    assert meta["accuracy_status"] == "dynamics outside the profiled condition"
    options = search.edges(search.home, TargetRegion("left", None, None))
    assert all(e.accuracy == "unknown" for e in options if e.commands)
