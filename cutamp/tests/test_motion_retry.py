"""Tests for the next-best-particle motion-plan retry
(algorithm._motion_plan_with_retries, motion_solver._reset_attachment_state)."""
import contextlib
import math
from types import SimpleNamespace

import pytest
import torch

import cutamp.algorithm as alg
import cutamp.com_polygon_cost as cpc
import cutamp.motion_solver as ms
from cutamp.config import TAMPConfiguration
from cutamp.particle_initialization import PickChoice
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


@pytest.mark.parametrize("centering_weight", [0.0, 1.0])
def test_retry_skips_candidates_that_repeat_a_failed_plan(monkeypatch, centering_weight):
    # Ops read: 0 -> left_q0, left_q1; 1 -> grasp1, left_q1; 2 -> pose1, left_q3.
    skeleton = [SimpleNamespace(values=v) for v in (
        ("left_q0", "traj1", "left_q1"),
        ("block3", "grasp1", "left_q1"),
        ("block3", "grasp1", "pose1", "goal", "left_q3"),
    )]

    def cand(q1, q3):
        return {"left_q0": torch.zeros(2), "grasp1": torch.zeros(7), "pose1": torch.zeros(4),
                "left_q1": torch.tensor([q1, 0.0]), "left_q3": torch.tensor([q3, 0.0])}

    planned, tol = [], object()  # tol: the run's IK tolerances, which the retry passes through

    def fake_solve_curobo(plan_info, particle, *a, ik_tol, **k):
        assert ik_tol is tol
        q1, q3 = float(particle["left_q1"][0]), float(particle["left_q3"][0])
        planned.append((q1, q3))
        if q1 == 0 or q3 == 0:  # pick fails on q1 == 0, else place fails on q3 == 0
            e = RuntimeError("pick failed" if q1 == 0 else "place failed")
            e.op_idx = 1 if q1 == 0 else 2
            raise e
        return "PLAN", {}

    monkeypatch.setattr(alg, "solve_curobo", fake_solve_curobo)
    candidates = [cand(0, 0), cand(0, 1), cand(1, 0), cand(1, 0), cand(1, 1)]
    plan, _, attempts, rank, failures = alg._motion_plan_with_retries(
        {"plan_skeleton": skeleton}, candidates, None, TAMPConfiguration(com_center_weight=centering_weight),
        None, None, ik_tol=tol,
    )
    # (0, 1) repeats (0, 0) through the failing pick; the second (1, 0) repeats
    # the first through the failing place; (1, 1) differs at the place, so it runs.
    assert planned == [(0, 0), (1, 0), (1, 1)]
    # rank names the planned particle, which attempts can't once candidates are skipped (known-bugs N3)
    assert (plan, attempts, rank) == ("PLAN", 3, 4)
    # Each planned candidate that failed keeps its message, so a report can count what each was lost to.
    assert failures == ["pick failed", "place failed"]


def _run_skeletons(monkeypatch, results, *, initial_witness=False, skeleton_budget=30, centers=None, viewer=None) -> dict:
    """run_cutamp's overall_metrics, on the CPU, over one skeleton per entry of ``results``, in order: ``None`` where
    no particle satisfies, else ``(planned, attempts)``: its motion plan tries ``attempts`` candidates, each failing
    but the last if ``planned``. Everything else run_cutamp calls is a stand-in."""

    timer = SimpleNamespace(time=lambda *a, **k: contextlib.nullcontext(), start=lambda m: None,
                            stop=lambda m: 0.0, get_summaries=dict)  # TorchTimer syncs CUDA

    def sample(plan_gen, world, config, timer, idx, *args):
        if idx == len(results):
            raise StopIteration
        choice = PickChoice(f"block{idx}", "grasp1", torch.zeros(1, 7), torch.ones(1), torch.ones(1, dtype=torch.bool),
                            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.long))
        return {"idx": idx, "plan_skeleton": [], "heuristic": float(idx), "grasp_choice": [choice]}, initial_witness

    def optimize(plan_info, timer, visualizer):
        satisfying = results[plan_info["idx"]] is not None
        metrics = {"best_cost": 0.0, "best_soft_cost": 0.0, "num_satisfying_final": int(satisfying)}
        if centers is not None and satisfying:
            distances = torch.as_tensor(centers[plan_info["idx"]], dtype=torch.float64)
            plan_info["row_quality"] = {"max_center_distance_m": distances, "secondary_effort": distances * 0,
                                        "best_cost": distances * 0, "best_soft_cost": distances * 0}
            metrics.update(max_center_distance_m=float(distances.min()), secondary_effort=0.)
        return satisfying, metrics, False

    def plan(plan_info, *args, **kwargs):
        planned, attempts = results[plan_info["idx"]]
        failures = [f"skeleton {plan_info['idx']}, candidate {i}" for i in range(attempts - planned)]
        if viewer is not None:
            assert args[4] is None, "intermediate candidates must leave the winner's timeline available"
        return (f"plan-{plan_info['idx']}", {}, attempts, attempts - 1, failures) if planned else (None, None, attempts, None, failures)

    logger, world = SimpleNamespace(log_dict=lambda name, d: None), SimpleNamespace(initial_state=None, goal_state=None)
    monkeypatch.setattr(alg, "setup_cutamp", lambda *args, **kwargs: (logger, viewer, timer, world))
    monkeypatch.setattr(alg, "ParticleInitializer", lambda *args: SimpleNamespace(ik_tol=None))
    monkeypatch.setattr(alg, "build_motion_planner", lambda world: None)
    monkeypatch.setattr(alg, "start_sampler_load", lambda: None)  # no model in a stand-in run
    monkeypatch.setattr(alg, "make_arm_affinity_priority_fn", lambda world: None)
    monkeypatch.setattr(alg, "task_plan_generator", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(alg, "sample_plan_skeleton", sample)
    monkeypatch.setattr(alg, "ParticleOptimizer", lambda *args, **kwargs: optimize)
    def candidates(info, *args, return_indices=False, **kwargs):
        count = len(centers[info["idx"]]) if centers is not None else 0
        return ([{}] * count, list(range(count))) if return_indices else [{}] * count
    monkeypatch.setattr(alg, "get_top_satisfying_particles", candidates)
    monkeypatch.setattr(alg, "_motion_plan_with_retries", plan)
    monkeypatch.setattr(alg, "_planned_metrics", lambda world, plan, candidates, rank: {"motion_plan_rank": rank})
    config = TAMPConfiguration(step_mode="off", curobo_plan=True, enable_visualizer=False, num_initial_plans=skeleton_budget,
                               com_center_weight=1.0 if centers is not None else 0.0)
    return alg.run_cutamp(None, config, SimpleNamespace(cost_config={}), SimpleNamespace(constraint_config={}),
                          experiment_id="fake")[2]


@pytest.mark.parametrize("budget, solved", [(1, False), (2, True)])
def test_failed_motion_resumes_skeletons_after_early_kinematic_hit(monkeypatch, budget, solved):
    metrics = _run_skeletons(monkeypatch, [(False, 2), (True, 1)], initial_witness=True, skeleton_budget=budget)
    assert metrics["found_solution"] is solved and metrics["motion_plan_success"] is solved
    assert metrics["num_initial_plans"] == metrics["num_optimized_plans"] == budget


def test_only_selected_motion_appears_on_curobo_timeline(monkeypatch):
    viewer, displayed = object(), []
    monkeypatch.setattr(alg, "play_curobo_motion", lambda world, motion, vis: displayed.append((motion, vis)), raising=False)
    metrics = _run_skeletons(monkeypatch, [(True, 1), (True, 1), (False, 1)],
                             centers=[[.01], [.09], [.02]], viewer=viewer)
    assert metrics["num_optimized_plans"] == 3
    assert displayed == [("plan-0", viewer)]


def test_motion_playback_starts_at_original_scene_and_replays_held_object():
    class Recording(alg.Visualizer):
        def __init__(self):
            self.events = []

        def set_time_seconds(self, timeline, val):
            self.events.append((timeline, val))

        def set_joint_positions(self, q):
            self.events.append(("start", q.tolist()))

        def log_mat4x4(self, name, pose):
            self.events.append((name, float(pose[0, 3])))

        def log_joint_trajectory_with_mat4x4s(self, traj, mat4x4s, timeline, start_time, dt):
            self.events.append((timeline, start_time, dt, traj.tolist(), mat4x4s["world/part"][:, 0, 3].tolist()))

    start_pose = torch.eye(4)
    start_pose[0, 3] = .5
    object_path = start_pose.repeat(2, 1, 1)
    object_path[-1, 0, 3] = .75
    world = SimpleNamespace(kinematics=SimpleNamespace(joint_names=["b", "a"]),
                            movables=[SimpleNamespace(name="part")], get_object_pose=lambda part: start_pose)
    plan = SimpleNamespace(position=torch.tensor([[1., 2.], [3., 4.]]), joint_names=["a", "b"])
    entries = [{"plan": plan, "dt": .02, "held_objs": {"left": ("part", object_path)}}]
    viewer = Recording()
    ms.play_curobo_motion(world, entries, viewer)
    assert viewer.events == [("curobo", 0.), ("start", [2., 1.]), ("world/part", .5),
                             ("curobo", 0., .02, [[2., 1.], [4., 3.]], [.5, .75])]


def test_rerun_segment_returns_native_end_for_the_next_segment(monkeypatch):
    from cutamp.utils import visualizer as viz

    viewer = object.__new__(viz.RerunVisualizer)
    viewer.robot = SimpleNamespace(get_rr_columns=lambda trajectory, **kw: {})
    spans = []

    def receive(path, *, indexes, columns):
        assert path == "world/part" and indexes[0].timeline_name() == "curobo"
        spans.append(indexes[0].as_arrow_array().cast("int64").to_pylist())

    monkeypatch.setattr(viz.rr, "send_columns", receive)
    start = 1.0
    for count in (3, 2, 1):
        transforms = torch.eye(4).repeat(count, 1, 1)
        start = viewer.log_joint_trajectory_with_mat4x4s(
            torch.zeros(count, 33), {"world/part": transforms}, "curobo", start, 0.125, head=torch.zeros(count, 2)
        )
    assert spans == [
        [1_000_000_000, 1_125_000_000, 1_250_000_000],
        [1_250_000_000, 1_375_000_000],
        [1_375_000_000],
    ]
    assert start == 1.375


@pytest.mark.parametrize("results, attempts", [
    ([(True, 2)], 2),  # one skeleton: its own count
    ([(False, 3), (True, 2)], 5),  # the first skeleton's candidates all fail: both skeletons' count
    ([None, (False, 1), (True, 1)], 2),  # a skeleton with no satisfying particle plans nothing
    ([None], None),  # nothing planned: no count
])
def test_motion_plan_attempts_add_up_over_skeletons(monkeypatch, results, attempts):
    # As the failures do: every planned candidate either fails or gives the plan.
    metrics = _run_skeletons(monkeypatch, results)
    assert metrics.get("motion_plan_attempts") == attempts
    assert (attempts or 0) == len(metrics["motion_plan_failures"]) + metrics["motion_plan_success"]


def test_retry_gives_cuRobos_full_budget_only_to_candidates_that_ran_out_of_attempts(monkeypatch):
    # The balance gate rejects candidate 0's plan, which no budget changes; candidate 1's pick needs more than 2
    # plan_cspace attempts (out_of_attempts, as _plan_arm_to_conf sets it). The cheap pass fails both; the full pass
    # skips 0 and plans 1.
    skeleton = [SimpleNamespace(values=("left_q0", "traj1", "left_q1"))]
    calls = []

    def fake_solve_curobo(plan_info, particle, *a, max_attempts, **k):
        q1 = float(particle["left_q1"][0])
        calls.append((q1, max_attempts))
        if q1 == 0 or max_attempts < 20:
            e = RuntimeError("trajectory ZMP leaves the support region" if q1 == 0 else "pick failed")
            e.op_idx = 0
            if q1 == 0:
                e.check = "zmp"
            else:
                e.out_of_attempts = True
            raise e
        return "PLAN", {}

    monkeypatch.setattr(alg, "solve_curobo", fake_solve_curobo)
    candidates = [{"left_q0": torch.zeros(2), "left_q1": torch.tensor([q1, 0.0])} for q1 in (0.0, 1.0)]
    plan, _, attempts, rank, failures = alg._motion_plan_with_retries(
        {"plan_skeleton": skeleton}, candidates, None, None, None, None, ik_tol=None,
    )
    assert calls == [(0.0, 2), (1.0, 2), (1.0, 20)]
    assert (plan, attempts, rank) == ("PLAN", 3, 1)
    assert failures == ["trajectory ZMP leaves the support region", "pick failed"]


@pytest.mark.parametrize("weight, expected_calls, winner", [
    (None, [(0, 2), (0, 20)], 0),
    (0.0, [(0, 2), (1, 2)], 1),
])
def test_centered_candidate_gets_full_budget_before_trying_a_worse_posture(monkeypatch, weight, expected_calls, winner):
    skeleton = [SimpleNamespace(values=("left_q0", "left_q1"))]
    candidates = [{"left_q0": torch.zeros(1), "left_q1": torch.tensor([i])} for i in (0, 1)]
    calls = []

    def solve(info, particle, *args, max_attempts, **kwargs):
        index = particle["left_q1"].item()
        calls.append((index, max_attempts))
        if index == 0 and max_attempts == 2:
            err = RuntimeError("pick approach needs more attempts")
            err.out_of_attempts = True
            raise err
        return ["motion"], {}

    monkeypatch.setattr(alg, "solve_curobo", solve)
    _, _, _, rank, _ = alg._motion_plan_with_retries(
        {"plan_skeleton": skeleton}, candidates, None, TAMPConfiguration(com_center_weight=weight),
        None, None, ik_tol=None,
    )
    assert calls == expected_calls and rank == winner


def test_com_margin_is_the_min_over_the_particles_confs(monkeypatch):
    # planned_com_margin: the planned particle's margin, computed as optimize_plan computes best_com_margin for the
    # lowest-soft-cost satisfying particle, which need not be the planned one (known-bugs N3). Grasps and placements
    # aren't confs.
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: q[:, 0])
    particle = {"left_q0": torch.tensor([0.3, 0.0]), "left_q1": torch.tensor([0.1, 0.0]),
                "right_q3": torch.tensor([0.2, 0.0]), "grasp1": torch.tensor([-1.0, 0.0, 0.0, 0.0])}
    assert cpc.particle_com_margin(None, particle) == pytest.approx(0.1)


def test_planned_metrics_describe_the_planned_particle(monkeypatch):
    # After a retry the plan comes from candidates[rank], not rank 0 (known-bugs N3). Tilt and travel are the worst
    # over its pick and place confs, without the retract; planned_com_margin stays the min over all its confs.
    from cutamp.tests.test_grasp_choice import _fake_world, _op
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: q[:, 5])
    skeleton = [_op("pick", "block", "grasp1", "left_q1"), _op("place", "block", "grasp1", "pose1", "table", "left_q2"),
                _op("retract", "block", "grasp1", "left_q2", "traj", "left_q3")]
    conf = lambda z, pitch, margin: torch.tensor([0.0, 0.0, z, pitch, 0.0, margin])
    candidates = [{name: conf(0.7, 0.0, 0.3) for name in ("left_q1", "left_q2", "left_q3")},
                  {"left_q1": conf(0.5, 0.2, 0.1), "left_q2": conf(0.6, 0.1, 0.2), "left_q3": conf(0.0, 1.0, 0.4)}]
    assert alg._planned_metrics(_fake_world(), skeleton, candidates, 1) == {
        "motion_plan_rank": 1, "planned_com_margin": pytest.approx(0.1),
        "planned_trunk_tilt_deg": pytest.approx(math.degrees(0.2)), "planned_trunk_travel": pytest.approx(0.2)}


def test_metrics_report_the_last_optimized_skeletons_grasp_choice(monkeypatch):
    # The first skeleton's tries all fail and the second plans: its choice is the one reported. A run that never
    # plans reports the last skeleton it optimized (G12).
    objects = lambda results: [s["object"] for s in _run_skeletons(monkeypatch, results)["grasp_choice"]]
    assert objects([(False, 3), (True, 2)]) == ["block1"]
    assert objects([None]) == ["block0"]


@needs_cuda
def test_reset_puts_placed_blocks_back():
    # A Place moves the block's collision obstacle to the placement. If that
    # particle then fails, the retry on the next one must collide against the
    # block at its start pose, not the failed particle's placement.
    from curobo.types import Pose

    world = make_blocks_t1_world()
    planner = ms.build_motion_planner(world, use_cuda_graph=False)
    cuboids = planner.scene_collision_checker.data.cuboids

    def offset(obj):  # start-pose center in the obstacle's frame; 0 = obstacle at start pose
        inv = cuboids.inv_pose[0, cuboids.get_idx(obj.name, env_idx=0), :7]
        start = world.get_object_pose(obj)[None, :3, 3].contiguous()
        return float(Pose(inv[None, :3].clone(), inv[None, 3:].clone()).transform_points(start).norm())

    for obj in world.movables:
        placed = world.get_object_pose(obj).clone()
        placed[0, 3] += 0.2
        # What a Place does.
        planner.scene_collision_checker.update_obstacle_pose(name=obj.name, w_obj_pose=Pose.from_matrix(placed))
        assert offset(obj) == pytest.approx(0.2, abs=1e-3)

    ms._reset_attachment_state(planner, world)  # first thing the retry's solve_curobo does
    for obj in world.movables:
        assert offset(obj) < 1e-4, obj.name


def test_centered_fixed_stance_motion_compares_successful_rows_across_skeletons(monkeypatch):
    # Skeleton 0's best posture fails motion, row 1 passes. Skeleton 1 then improves the successful plan.
    # Skeleton 2 has the best posture but fails motion, so it cannot replace the incumbent.
    result = _run_skeletons(monkeypatch, [(True, 2), (True, 1), (False, 1)], initial_witness=True,
                            centers=[[.01, .18], [.11], [.001]])
    assert result["num_optimized_plans"] == 3
    assert result["motion_plan_success"]
    assert result["max_center_distance_m"] == pytest.approx(.11)
    assert result["motion_plan_rank"] == 0


def test_fixed_stance_motion_skips_dominated_skeletons(monkeypatch):
    # The second skeleton already loses at its most centered kinematic candidate.
    result = _run_skeletons(monkeypatch, [(True, 1), (True, 1)], initial_witness=True,
                            centers=[[.06], [.07]])
    assert result["num_optimized_plans"] == 2
    assert result["motion_plan_attempts"] == 1
    assert result["max_center_distance_m"] == pytest.approx(.06)
