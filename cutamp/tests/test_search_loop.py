"""Exercise the public planner with synthetic refinement and a real symbolic domain."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import cutamp.algorithm as alg
from cutamp import t1_domain
from cutamp.config import TAMPConfiguration
from cutamp.step_planner import StepInputs, StepResult, StepRow, furthest_endpoint


def test_screened_predictor_cache_is_scoped_to_deployment_stream_and_worker(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from cutamp import landing_predictor

    bundle = tmp_path / "landing.pt"
    bundle.write_bytes(b"deployment")
    config = TAMPConfiguration(landing_bundle=str(bundle))
    stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(landing_predictor, "LandingPredictor", lambda path, **kw: SimpleNamespace(**kw))

    model = alg._load_predictor(config)
    assert alg._load_predictor(config) is model and model.use_cuda_graph
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(alg._load_predictor, config).result() is not model
    stream.cuda_stream = 1
    other = alg._load_predictor(config)
    assert other is not model
    bundle.write_bytes(b"changed deployment")
    assert alg._load_predictor(config) is not other
    reference = replace(config, step_refine_iterations=2)
    assert alg._load_predictor(reference) is not alg._load_predictor(reference)
    bundle.unlink()
    with pytest.raises(FileNotFoundError):
        alg._load_predictor(config)


def test_standing_feedback_locates_scene_and_placement_failures_at_their_actions():
    from cutamp.step_planner import standing_endpoint_flags
    from cutamp.task_planning import task_plan_generator

    init = t1_domain.get_initial_state(["block"], ["table"])
    goal = frozenset([t1_domain.On.ground("block", "table")])
    plan = next(task_plan_generator(init, goal, t1_domain.all_t1_operators))
    ops = [op for op in plan if op.operator.metadata.is_actionable]
    placement = ops[1].arg("placement")
    mask = torch.ones(2, 2, dtype=torch.bool)
    masks = {
        "ComComfort": {"margin_shortfall_squared": mask},
        "KinematicConstraint": {"sole_pos_err": mask.repeat_interleave(2, dim=1)},
        "Motion": {"joint_limit": mask, "self_collision": mask},
        "Collision": {"robot_to_world": torch.tensor([[True, True], [False, True]])},
        "StablePlacement": {"table_support": torch.tensor([[False], [True]]), "table_in_xy": mask[:, :1]},
    }
    rollout = {"ts_to_pose_ts": {0: 0, 1: 1}}
    cost = SimpleNamespace(retract_info=[], surface_to_placements={"table": [("block", placement)]})
    ep = standing_endpoint_flags(masks, rollout, plan, cost)
    assert ep.tolist() == [[True, False], [False, True]]
    assert furthest_endpoint(ep) == 1


def test_endpoint_prefix_belongs_to_one_row():
    assert furthest_endpoint(torch.tensor([[True, False], [False, True]])) == 1
    assert furthest_endpoint(torch.tensor([[False, False]])) == 0
    assert furthest_endpoint(torch.ones((0, 2), dtype=torch.bool)) is None


@pytest.fixture
def harness(monkeypatch, tmp_path):
    from cutamp.step_search import StepSearch

    goal = frozenset([t1_domain.On.ground("block", "table")])
    world = SimpleNamespace(
        initial_state=t1_domain.get_initial_state(["block"], ["table"]),
        goal_state=goal,
        grasps={},
        device="cpu",
        get_object=lambda name: SimpleNamespace(name=name, pose=(0.3, 0, 0.55, 1, 0, 0, 0), dims=(0.1, 0.1, 0.1)),
        stance_soles=torch.eye(4).repeat(2, 1, 1),
    )
    world.stance_soles[:, 1, 3] = torch.tensor([0.1, -0.1])
    seen = []
    timer = SimpleNamespace(
        time=lambda *a, **kw: nullcontext(), start=lambda key: None, stop=lambda key: 0.0, get_summaries=dict
    )
    monkeypatch.setattr(
        alg, "setup_cutamp", lambda *a, **kw: (SimpleNamespace(log_dict=lambda *a: None), None, timer, world)
    )
    monkeypatch.setattr(alg, "make_arm_affinity_priority_fn", lambda w: None)
    monkeypatch.setattr(alg, "ParticleInitializer", lambda *a, **kw: object())
    monkeypatch.setattr(alg, "ParticleOptimizer", lambda *a, **kw: object())
    monkeypatch.setattr(alg, "grasp_sets", lambda objs, dev, **kw: seen.append("generate") or {"block": object()})
    monkeypatch.setattr(alg, "start_sampler_load", lambda: seen.append("load"))
    monkeypatch.setattr(alg, "wait_for_sampler_load", lambda: seen.append("wait"))

    class Search(StepSearch):
        def schedules(self, *args, **kwargs):
            seen.append("rank")
            yield from super().schedules(*args, **kwargs)

    monkeypatch.setattr(alg, "_load_step_search", lambda cfg, **kw: (Search(None, commands=((0.12, 0.21, 0),)), {}))
    cfg = {
        "num_initial_plans": 1,
        "enable_visualizer": False,
        "step_mode": "model",
        "curobo_plan": False,  # exercise the keyframe search unless a test requests motion
        "experiment_root": str(tmp_path),
        "landing_bundle": str(tmp_path / "no-checkpoint"),
    }

    def run(**options):
        inputs = options.pop("inputs", None)
        return alg.run_cutamp(
            None,
            TAMPConfiguration(**(cfg | options)),
            SimpleNamespace(cost_config={}),
            SimpleNamespace(constraint_config={}),
            experiment_id="test",
            step_inputs=inputs,
        )

    return run, seen, world


def test_base_symbolic_expansion_obeys_the_same_search_deadline(monkeypatch):
    import cutamp.task_planning.search as symbolic

    now = iter((0.0, 2.0))
    monkeypatch.setattr(symbolic, "monotonic", lambda: next(now))
    initial = t1_domain.get_initial_state(["block"], ["table"])
    goal = frozenset([t1_domain.On.ground("block", "table")])
    plans = symbolic.breadth_first_search(initial, goal, t1_domain.all_t1_operators, deadline=1.0)
    assert list(plans) == []


def test_standing_motion_respects_an_expired_requested_search_budget():
    motion, _, tries, rank, _ = alg._motion_plan_with_retries(
        {"plan_skeleton": []}, [{}], None, None, None, None, ik_tol=None, deadline=0
    )
    assert motion is None and rank is None and tries == 0


@pytest.mark.parametrize("options,expected", [
    ({}, 8.), ({"max_loop_dur": 2.}, 2.),
    ({"step_mode": "rank"}, None), ({"step_mode": "geometric"}, None),
    ({"step_derivative_method": "autodiff"}, None), ({"step_proposal_count": 0}, None),
    ({"step_proposal_count": 64, "step_refine_iterations": 2}, None),
])
def test_search_deadline_scope_and_world_setup(harness, monkeypatch, options, expected):
    run, seen, _ = harness
    clock = [alg.monotonic()]
    monkeypatch.setattr(alg, "monotonic", lambda: clock[0])
    setup = alg.setup_cutamp

    def slow_setup(*args, **kwargs):
        clock[0] += 9.
        return setup(*args, **kwargs)

    monkeypatch.setattr(alg, "setup_cutamp", slow_setup)
    monkeypatch.setattr(alg, "_refine_standing_schedule", lambda *a, **kw: StepResult([], None, "unresolved"))
    _, _, metrics = run(max_step_schedules=1, **options)
    assert metrics["search_budget_s"] == expected
    if expected is not None:
        assert metrics["time_limit_reached"] and metrics["status"] == "unresolved"
        assert metrics["num_optimized_plans"] == 0 and not seen and not metrics["step_search"]
    else:
        assert not metrics["time_limit_reached"] and len(metrics["step_attempts"]) == 1


def test_search_deadline_returns_incumbent_without_starting_next_refinement(harness, monkeypatch):
    run, _, _ = harness
    clock = [alg.monotonic()]
    monkeypatch.setattr(alg, "monotonic", lambda: clock[0])

    def expensive_model_load(config):
        clock[0] += 10.
        return object()

    row = StepRow(3, {}, 0., {"com_center_distances_m": [.04, .03], "secondary_effort": 0.}, ["accepted"])
    monkeypatch.setattr(alg, "_refine_standing_schedule", lambda *a, **kw: StepResult([row], None, "standing_feasible"))
    monkeypatch.setattr(alg, "_load_predictor", expensive_model_load)
    monkeypatch.setattr(alg, "_refine_step_schedule", lambda *a, **kw: pytest.fail("started after deadline"))
    monkeypatch.setattr(alg, "play_curobo_motion", lambda *a: None)
    motion, count, metrics = run(curobo_plan=True)
    assert metrics["search_budget_s"] == 9.5
    assert motion == ["accepted"] and count == 1 and metrics["row_index"] == 3
    assert metrics["time_limit_reached"] and metrics["status"] == "motion_feasible"
    assert metrics["step_attempts"][-1]["status"] == "budget_exhausted"


@pytest.mark.parametrize("explicit, expected_attempts", [(None, 1), (12., 2)])
def test_full_motion_improvement_deadline_starts_only_after_acceptance(harness, monkeypatch, explicit, expected_attempts):
    run, _, _ = harness
    clock = [alg.monotonic()]
    monkeypatch.setattr(alg, "monotonic", lambda: clock[0])
    row = StepRow(1, {}, 0., {"com_center_distances_m": [.02, .03], "secondary_effort": 0.}, ["complete"])

    def standing(*args, **kwargs):
        clock[0] += 5.2  # cold start exceeds the short target before a valid motion exists
        return StepResult([row], None, "standing_feasible")

    monkeypatch.setattr(alg, "_refine_standing_schedule", standing)
    monkeypatch.setattr(alg, "_load_predictor", lambda config: object())
    monkeypatch.setattr(alg, "_refine_step_schedule", lambda *a, **kw: StepResult([], None, "unresolved"))
    monkeypatch.setattr(alg, "play_curobo_motion", lambda *a: None)
    motion, _, metrics = run(curobo_plan=True, max_loop_dur=explicit, max_step_schedules=2)
    assert motion == ["complete"]
    assert len(metrics["step_attempts"]) == expected_attempts


def test_ranking_only_enters_at_run_cutamp_and_never_generates_grasps(harness, monkeypatch):
    run, seen, _ = harness
    monkeypatch.setattr(alg, "ParticleInitializer", lambda *a: pytest.fail("initialization during ranking"))
    _, n, metrics = run(step_mode="rank", max_step_schedules=3)
    assert n == 0 and not metrics["found_solution"] and metrics["status"] == "ranked"
    assert len(metrics["step_attempts"]) == 3
    assert seen and set(seen) == {"rank"}


def test_missing_predictor_never_marks_model_feasible_or_draws_grasps(harness, monkeypatch):
    run, seen, _ = harness
    _, n, m = run(inputs=StepInputs(orders=("L", "none")))
    assert n == 0 and not m["found_solution"] and m["step_attempts"][0]["status"] == "unavailable"
    assert "bundle" in m["step_attempts"][0]["reason"].lower()
    assert "generate" not in seen


def test_full_fallback_all_25_order_pairs_only_once_in_budget(harness, monkeypatch):
    run, seen, _ = harness
    empty = StepResult([], None, "initialization_failed")
    monkeypatch.setattr(alg, "_refine_standing_schedule", lambda *a, **k: empty)
    monkeypatch.setattr(alg, "_refine_step_schedule", lambda *a, **k: empty)
    _, n, metrics = run(step_mode="geometric", max_step_schedules=25)
    attempts = metrics["step_attempts"]
    assert n == 0 and len({tuple(a["orders"]) for a in attempts}) == len(attempts) == 25
    assert attempts[0]["orders"] == ["none", "none"]
    assert seen.index("rank") < seen.index("generate") and seen.count("generate") == 1
    assert seen.index("generate") < seen.index("wait")


def test_accept_at_requested_level_and_keep_row_cost_after_retry(harness, monkeypatch):
    run, _, _ = harness
    calls = []
    monkeypatch.setattr(alg, "_load_predictor", lambda *a: object())
    monkeypatch.setattr(
        alg, "_refine_standing_schedule", lambda *a, **k: StepResult([], torch.tensor([[False, True]]), "unresolved")
    )

    def refine(p, *args, **kwargs):
        calls.append(p.orders)
        rows = [
            StepRow(7, {"q": torch.tensor([7])}, 0.2, {"min_com_margin_m": 0.052}),
            StepRow(8, {"q": torch.tensor([8])}, 0.6, {"min_com_margin_m": 0.056}),
        ]
        if p.orders == ("R", "none"):
            rows[1].motion_plan = ["motion"]
        return StepResult(rows, torch.ones(2, 2, dtype=torch.bool), "model_feasible")

    monkeypatch.setattr(alg, "_refine_step_schedule", refine)
    motion, _, m = run(curobo_plan=True, max_step_schedules=25, com_center_weight=0)
    assert motion == ["motion"] and m["motion_plan_success"] and m["selected_particle"]["q"] == [8]
    assert m["row_index"] == 8 and m["best_soft_cost"] == 0.6 and m["min_com_margin_m"] == 0.056
    assert calls[:2] == [("L", "none"), ("R", "none")]  # failed pick prioritizes that gap
    assert any(a["status"] == "model_feasible" and a["accepted_rows"] == 0 for a in m["step_attempts"])
    assert all(sum(len(o) for o in p if o != "none") <= 1 for p in calls)


@pytest.mark.parametrize("weight,winning_id", [(None, 2), (0.1, 2), (0.0, 0)])
def test_motion_feasibility_then_worst_center_then_count_then_effort(harness, monkeypatch, weight, winning_id):
    run, _, _ = harness
    viewer, displayed = object(), []
    setup = alg.setup_cutamp
    def with_viewer(*args, **kwargs):
        logger, _, timer, world = setup(*args, **kwargs)
        return logger, viewer, timer, world
    monkeypatch.setattr(alg, "setup_cutamp", with_viewer)
    monkeypatch.setattr(alg, "play_curobo_motion", lambda world, motion, vis: displayed.append((motion, vis)))
    monkeypatch.setattr(alg, "_load_predictor", lambda *a: object())
    standing = StepRow(0, {}, 0.01, {"com_center_distances_m": [0., .05], "secondary_effort": 0.}, ["standing"])
    monkeypatch.setattr(alg, "_refine_standing_schedule", lambda *a, **kw: StepResult([standing], None, "standing_feasible"))

    def refine(proposal, *args, **kwargs):
        if weight != 0:
            assert kwargs["visualizer"] is None
        if proposal.orders == ("L", "none"):
            candidates = [
                (1, [.0, .0495], .01, .0, ["worse at place"]),
                (2, [.049, .049], 8., 1., ["selected"]),
                (3, [.049, .005], 6., 2., ["higher effort"]),
                (4, [.001, .001], .001, .001, None),
            ]
        elif proposal.orders == ("LR", "none"):
            candidates = [(5, [.049, .0001], .005, 0., ["extra step"])]
        else:
            return StepResult([], None, "unresolved")
        rows = [
            StepRow(i, {}, soft, {"gaps": [{"center_distance_m": x} for x in ds], "secondary_effort": effort}, motion)
            for i, ds, soft, effort, motion in candidates
        ]
        return StepResult(rows, None, "model_feasible")

    monkeypatch.setattr(alg, "_refine_step_schedule", refine)
    motion, _, metrics = run(curobo_plan=True, max_step_schedules=25, com_center_weight=weight)
    assert metrics["row_index"] == winning_id
    if winning_id == 2:
        assert motion == ["selected"]
        assert displayed == [(motion, viewer)]
        assert metrics["max_center_distance_m"] == .049
        assert metrics["total_steps"] == 1 and metrics["secondary_effort"] == 1.
        assert any(a["orders"] == ["LR", "none"] for a in metrics["step_attempts"])
    else:
        assert motion == ["standing"] and len(metrics["step_attempts"]) == 1


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
    monkeypatch.setattr(alg, "_refine_standing_schedule", lambda *a, **k: StepResult([], None, "unresolved"))

    def refine(proposal, *args, **kwargs):
        if proposal.orders[1] != "none" or proposal.orders[0] not in costs:
            return StepResult([], None, "unresolved")
        side = proposal.orders[0]
        row = StepRow(2, {}, costs[side], {"gaps": [{"center_distance_m": .04}], "secondary_effort": costs[side]},
                      evidence=lambda: evidence_calls.append(side) or {"side": side})
        return StepResult([row], None, "model_feasible")

    monkeypatch.setattr(alg, "_refine_step_schedule", refine)
    _, _, result = run(max_step_schedules=25)
    assert result["gap_schedule"] == ["R", "none"]
    assert evidence_calls == ["R"]
    assert result["model_evidence_path"] == str(tmp_path / "model_evidence.json")


def test_default_centroid_preference_and_zero_weight_ablation():
    from dataclasses import replace

    from cutamp.config import validate_tamp_config
    from cutamp.scripts.utils import default_constraint_to_mult

    cfg = TAMPConfiguration()
    assert cfg.for_refinement(stepped=False).com_center_weight == 1
    assert default_constraint_to_mult["soft"]["com_center_distance_squared"] == 1
    validate_tamp_config(replace(cfg, com_center_weight=0))
    for invalid in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="com_center_weight"):
            validate_tamp_config(replace(cfg, com_center_weight=invalid))


def test_fast_refinement_defaults_follow_the_selected_schedule_and_preserve_overrides():
    from dataclasses import replace

    from cutamp.config import validate_tamp_config

    model = TAMPConfiguration(step_mode="model")

    def settings(cfg, stepped):
        resolved = cfg.for_refinement(stepped=stepped)
        return resolved.step_refine_iterations, resolved.step_trust_scale, resolved.com_center_weight

    assert settings(model, True) == (0, 2.0, 4.0)
    assert settings(replace(model, step_proposal_count=0), True) == (2, 2.0, 4.0)
    assert settings(model, False) == (4, 1.0, 1.0)
    assert settings(replace(model, step_mode="geometric"), True) == (4, 1.0, 1.0)
    assert settings(replace(model, step_derivative_method="autodiff"), True) == (4, 1.0, 1.0)
    for weight in (0.0, 1.0, 3.0):
        custom = replace(model, step_refine_iterations=0, step_trust_scale=1.5, com_center_weight=weight)
        assert settings(custom, True) == settings(custom, False) == (0, 1.5, weight)
    for scale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="step_trust_scale"):
            validate_tamp_config(replace(model, step_trust_scale=scale))
    with pytest.raises(ValueError, match="step_derivative_method"):
        validate_tamp_config(replace(model, step_derivative_method="unknown"))


@pytest.mark.parametrize("limit,expire,expected", [(1, False, [5]), (5, False, [5, 2]), (5, True, [5])])
def test_stepped_motion_honors_global_retry_order_and_deadline(monkeypatch, limit, expire, expected):
    from cutamp import step_particles

    # Row 5 is the best of three rows sharing a grasp, row 2 has a distinct grasp.
    rows = [
        StepRow(i, {"grasp1_idx": torch.tensor(grasp), "q1": torch.zeros(33)}, soft, {"best_com_margin": 0.06})
        for i, grasp, soft in [(0, 0, 4.0), (1, 0, 0.3), (2, 1, 1.0), (5, 0, 0.1)]
    ]
    pools = {"grasp1": (torch.zeros(2, 7), torch.ones(2))}
    calls = []
    monkeypatch.setattr(step_particles, "monotonic", lambda: 1.0 if expire and calls else 0.0)
    monkeypatch.setattr(step_particles, "worst_posture", lambda world, confs: (torch.zeros(4),) * 3)
    op = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="pick")), values=("q1",))

    def attempt(row):
        calls.append(row.index)
        if row.index == 2:
            return ["validated path"], {"motion_validation": "accepted"}
        return None, {"motion_validation": "rejected", "reason": "connection failed"}

    cfg = TAMPConfiguration(motion_plan_max_retries=limit, com_center_weight=0)
    report = step_particles._validate_motion_rows(rows, pools, [op], None, cfg, attempt, deadline=0.5)
    assert calls == expected
    assert report["motion_plan_attempts"] == len(calls)
    assert report["motion_plan_failures"] == ["connection failed"]
    winners = [r for r in rows if r.motion_plan is not None]
    assert [r.index for r in winners] == ([2] if 2 in expected else [])
    if winners:
        assert winners[0].metrics["motion_plan_rank"] == 1 and winners[0].soft_cost == 1.0


def test_stepped_motion_with_no_grasp_columns_uses_particle_cost(monkeypatch):
    from cutamp import step_particles

    op = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="place")), values=("q1",))
    rows = [StepRow(i, {"q1": torch.zeros(33)}, cost, {"best_com_margin": 0.08}) for i, cost in enumerate((2., 1.))]
    monkeypatch.setattr(step_particles, "worst_posture", lambda w, qs: (torch.zeros(2),) * 3)
    report = step_particles._validate_motion_rows(
        rows, {}, [op], None, TAMPConfiguration(com_center_weight=0), lambda r: ([r.index], {"motion_validation": "accepted"}),
        deadline=None,
    )
    assert report["motion_plan_attempts"] == 1 and rows[1].motion_plan == [1]


def test_validator_uses_the_globally_ranked_particles_own_trace(monkeypatch):
    from cutamp import motion_solver, step_particles

    cfg = TAMPConfiguration(
        step_mode="model", num_particles=4, step_particle_batch_size=2, motion_plan_max_retries=1, curobo_plan=True,
    )
    pools = {"grasp1": (torch.eye(7)[:2], torch.tensor([0.9, 0.5]))}
    op = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="pick")), values=("q1",))
    proposal = SimpleNamespace(skeleton=[op], seed=None)
    world = SimpleNamespace(q_init=torch.zeros(33))
    monkeypatch.setattr(step_particles, "ParticleInitializer", lambda *a: SimpleNamespace(_pick_pools=lambda p: pools))
    monkeypatch.setattr(step_particles, "worst_posture", lambda w, qs: (torch.zeros(len(qs[0])),) * 3)
    monkeypatch.setattr(step_particles.IKTol, "from_checker", lambda c: None)
    chunks, attempted = [], []

    def feasible_chunk(proposal, world, config, reducer, checker, inputs, predictor, pools, seeds, first, stop,
                       motion_inputs, deadline):
        chunks.append(first)
        rows = []
        for i in range(first, stop):
            if i != 2:
                motion_inputs[i] = (predictor, {0: torch.tensor([[i]])})
            particle = {"grasp1_idx": torch.tensor(i % 2), "grasp1": pools["grasp1"][0][i % 2],
                        "q1": torch.full((33,), float(i))}
            distances = ((.001, .09), (.08, .08), (.01, .01), (.07, .07))[i]
            effort = (0.1, .2, .01, 4.)[i]
            metrics = {"best_com_margin": .06, "secondary_effort": effort,
                       "gaps": [{"center_distance_m": d} for d in distances]}
            rows.append(StepRow(i, particle, effort + 4 * sum(d ** 2 for d in distances), metrics))
        return StepResult(rows, torch.ones(len(rows), 1, dtype=torch.bool), "model_feasible")

    def validate(skeleton, particle, traces, adapter, world, config, validator, **kwargs):
        assert validator is independent
        i = int(particle["q1"][0])
        assert traces[0].item() == i
        attempted.append(i)
        return ["path"], {"motion_validation": "accepted"}

    independent = object()
    monkeypatch.setattr(step_particles, "_refine_batch", feasible_chunk)
    monkeypatch.setattr(motion_solver, "plan_validated_step_motion", validate)
    result = step_particles.refine_step_particles(
        proposal, world, cfg, None, None, inputs=StepInputs(transition_validator=independent), predictor=object(),
        timer=None, visualizer=None, deadline=None,
    )
    assert chunks == [0, 2] and attempted == [3]
    assert [row.index for row in result.rows if row.motion_plan] == [3]
    assert result.diagnostics["motion_plan_attempts"] == 1 and result.diagnostics["motion_plan_failures"] == []


def test_invalid_fixed_command_sequence_is_rejected():
    with pytest.raises(ValueError, match="one seed"):
        StepInputs(orders=("LR",), commands=(((0.1, 0.2, 0.0),),))
    with pytest.raises(ValueError, match="finite"):
        StepInputs(orders=("R",), commands=(((0.1, float("nan"), 0.0),),))


def test_rank_assets_are_reused_and_regeneration_invalidates_cache(tmp_path):
    from dataclasses import replace

    from cutamp.reachability import ReachMap

    path = tmp_path / "nominal.npz"
    cfg = TAMPConfiguration(
        step_mode="rank", step_reach_map=str(path), step_accuracy_table=str(tmp_path / "missing.json")
    )
    first, info = alg._load_step_search(cfg)
    assert info["map_status"] == "unavailable"
    assert alg._load_step_search(cfg)[0] is first
    home = [[0, 0.1, 0], [0, -0.1, 0]]
    ReachMap(0.01, {"home_soles": home}).save(path)
    second, info = alg._load_step_search(cfg)
    assert second is not first and "map_status" not in info
    assert alg._load_step_search(cfg)[0] is second
    assert alg._load_step_search(replace(cfg, step_lag=5))[0] is not second


def test_missed_ranked_zero_step_recovers_in_count_order(harness, monkeypatch):
    run, _, _ = harness

    def standing(*args, **kwargs):
        return StepResult(
            [StepRow(0, {"q": torch.zeros(2)}, 0.1,
                     {"min_com_margin_m": 0.06, "com_center_distances_m": [.02, .03], "secondary_effort": .09})],
            torch.ones(1, 2, dtype=torch.bool),
            "standing_feasible",
        )

    monkeypatch.setattr(alg, "_refine_standing_schedule", standing)
    _, n, m = run(max_step_schedules=1)
    assert n == 1 and m["total_steps"] == 0 and m["status"] == "standing_feasible"
    assert m["schedule_limit_reached"]


from cutamp.tests.conftest import needs_cuda, needs_graspgenx


@needs_cuda
@needs_graspgenx
def test_real_pick_only_checks_bound_configuration_and_returns_its_row(tmp_path):
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs.utils import load_env, resolve_env_path
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.t1_domain import all_t1_fluents

    env = load_env(resolve_env_path("blocks_t1"))
    env.goal_state = frozenset([next(f for f in all_t1_fluents if f.name == "LeftHolding").ground("block3")])
    block = next(obj for obj in env.movables if obj.name == "block3")
    block.pose[:3] = [0.38, 0.23, 0.73]
    cfg = TAMPConfiguration(
        step_mode="model", curobo_plan=False,
        num_initial_plans=1,
        num_particles=16,
        num_opt_steps=20,
        enable_visualizer=False,
        cuda_graphs=False,
        experiment_root=str(tmp_path),
    )
    torch.manual_seed(1)
    motion, n, metrics = alg.run_cutamp(
        env,
        cfg,
        CostReducer(default_constraint_to_mult),
        ConstraintChecker(default_constraint_to_tol),
        experiment_id="standing-pick",
        step_inputs=StepInputs(orders=("none",), arms=("left",)),
    )
    assert motion is None and n > 0 and metrics["status"] == "standing_feasible"
    assert metrics["com_margins_m"] == [metrics["min_com_margin_m"]] and metrics["min_com_margin_m"] >= 0.05
    assert metrics["selected_particle"]["left_q1"] != metrics["selected_particle"]["left_q0"]
    assert metrics["row_index"] < cfg.num_particles and metrics["best_soft_cost"] >= 0


def test_unmatched_dynamics_never_receive_reference_tracking_credit():
    from cutamp.step_search import TargetRegion

    cfg = TAMPConfiguration(step_mode="model", step_lag=8)
    d = torch.zeros(1, 38)
    d[:, 0] = 0.25 / 0.45
    d[:, -5:] = 1.0
    assert StepInputs(dynamics=d).matches_accuracy_context(8)
    d[0, 1] = 0.25  # encoder offset differs from the unloaded home profile
    assert not StepInputs(dynamics=d).matches_accuracy_context(8)
    search, meta = alg._load_step_search(cfg, applicable=False)
    assert meta["accuracy_status"] == "dynamics outside the profiled condition"
    home = search.home or ((0, 0.1, 0), (0, -0.1, 0))
    options = search.edges(home, TargetRegion("left", None, None))
    assert all(e.accuracy == "unknown" for e in options if e.commands)


def test_count_fallback_uses_heuristic_to_break_equal_length_ties():
    from cutamp.reachability import ReachMap
    from cutamp.step_planner import ScheduleSearch
    from cutamp.step_search import StepSearch, TargetRegion
    from cutamp.task_planning import task_plan_generator

    home = ((0.0, 0.10, 0.0), (0.0, -0.10, 0.0))
    atlas = ReachMap(0.01, {"home_soles": home})
    atlas.add_stance(
        home, [{"arm": "left", "posture": "squat", "points": [(0.005, 0.005, 0.505)], "q": [[1.0]], "margin": [0.06]}]
    )
    search = StepSearch(atlas, commands=((0.12, 0.25, 0.0),))
    initial = t1_domain.get_initial_state(["block"], ["table"])
    goal = frozenset([next(f for f in t1_domain.all_t1_fluents if f.name == "LeftHolding").ground("block")])
    plan = next(task_plan_generator(initial, goal, t1_domain.all_t1_operators))
    world = SimpleNamespace(initial_state=initial, goal_state=goal)
    # Both exact single steps have the same forward motion. The R stance moves the map towards negative y.
    r = TargetRegion("left", (0.065, -0.02, 0.505), (0.005, 0.005, 0.005))
    stream = ScheduleSearch(search, plan, [r], world, home, strategy="count")
    assert stream.next().orders == ("none",)
    assert stream.next().orders == ("R",)
