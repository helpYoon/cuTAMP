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


def test_retry_skips_candidates_that_repeat_a_failed_plan(monkeypatch):
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
            e = RuntimeError("pick failed: torque" if q1 == 0 else "place failed: other")
            e.op_idx = 1 if q1 == 0 else 2
            raise e
        return "PLAN", {}

    monkeypatch.setattr(alg, "solve_curobo", fake_solve_curobo)
    candidates = [cand(0, 0), cand(0, 1), cand(1, 0), cand(1, 0), cand(1, 1)]
    plan, _, attempts, rank, failures = alg._motion_plan_with_retries(
        {"plan_skeleton": skeleton}, candidates, None, None, None, None, ik_tol=tol,
    )
    # (0, 1) repeats (0, 0) through the failing pick; the second (1, 0) repeats
    # the first through the failing place; (1, 1) differs at the place, so it runs.
    assert planned == [(0, 0), (1, 0), (1, 1)]
    # rank names the planned particle, which attempts can't once candidates are skipped (known-bugs N3)
    assert (plan, attempts, rank) == ("PLAN", 3, 4)
    # Each planned candidate that failed keeps its message, so a report can count those lost to torque (7.4a).
    assert failures == ["pick failed: torque", "place failed: other"]


def _run_skeletons(monkeypatch, results) -> dict:
    """run_cutamp's overall_metrics, on the CPU, over one skeleton per entry of ``results``, in order: ``None`` where
    no particle satisfies, else ``(planned, attempts)``: its motion plan tries ``attempts`` candidates, each failing
    but the last if ``planned``. Everything else run_cutamp calls is a stand-in."""

    class Timer:  # TorchTimer syncs CUDA
        def time(self, *args, **kwargs):
            return contextlib.nullcontext()

        def start(self, metric):
            pass

        def stop(self, metric):
            return 0.0

        def get_summaries(self):
            return {}

    def sample(plan_gen, world, config, timer, idx, *args):
        if idx == len(results):
            raise StopIteration
        choice = PickChoice(f"block{idx}", "grasp1", torch.zeros(1, 7), torch.ones(1), torch.ones(1, dtype=torch.bool),
                            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.long))
        return {"idx": idx, "plan_skeleton": [], "heuristic": float(idx), "grasp_choice": [choice]}, False

    def optimize(plan_info, timer, visualizer):
        satisfying = results[plan_info["idx"]] is not None
        return satisfying, {"best_cost": 0.0, "best_soft_cost": 0.0, "num_satisfying_final": int(satisfying)}, False

    def plan(plan_info, *args, **kwargs):
        planned, attempts = results[plan_info["idx"]]
        failures = [f"skeleton {plan_info['idx']}, candidate {i}" for i in range(attempts - planned)]
        return ("PLAN", {}, attempts, attempts - 1, failures) if planned else (None, None, attempts, None, failures)

    logger, world = SimpleNamespace(log_dict=lambda name, d: None), SimpleNamespace(initial_state=None, goal_state=None)
    monkeypatch.setattr(alg, "setup_cutamp", lambda *args, **kwargs: (logger, None, Timer(), world))
    monkeypatch.setattr(alg, "ParticleInitializer", lambda *args: SimpleNamespace(ik_tol=None))
    monkeypatch.setattr(alg, "build_motion_planner", lambda world: None)
    monkeypatch.setattr(alg, "make_arm_affinity_priority_fn", lambda world: None)
    monkeypatch.setattr(alg, "task_plan_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(alg, "sample_plan_skeleton", sample)
    monkeypatch.setattr(alg, "ParticleOptimizer", lambda *args, **kwargs: optimize)
    monkeypatch.setattr(alg, "get_top_satisfying_particles", lambda *args, **kwargs: [])
    monkeypatch.setattr(alg, "_motion_plan_with_retries", plan)
    monkeypatch.setattr(alg, "_planned_metrics", lambda *args: {})
    config = TAMPConfiguration(curobo_plan=True, enable_visualizer=False)
    return alg.run_cutamp(None, config, SimpleNamespace(cost_config={}), SimpleNamespace(constraint_config={}),
                          experiment_id="fake")[2]


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
            e = RuntimeError("trajectory ZMP leaves the support region" if q1 == 0 else "pick failed: other")
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
    assert failures == ["trajectory ZMP leaves the support region", "pick failed: other"]


def test_com_margin_is_the_min_over_the_particles_confs(monkeypatch):
    # planned_com_margin: the planned particle's margin, computed as optimize_plan computes best_com_margin
    # for rank 0, which a retry didn't plan (known-bugs N3). Grasps and placements aren't confs.
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: q[:, 0])
    particle = {"left_q0": torch.tensor([0.3, 0.0]), "left_q1": torch.tensor([0.1, 0.0]),
                "right_q3": torch.tensor([0.2, 0.0]), "grasp1": torch.tensor([-1.0, 0.0, 0.0, 0.0])}
    assert cpc.particle_com_margin(None, particle) == pytest.approx(0.1)


def test_planned_metrics_describe_the_planned_particle(monkeypatch):
    # After a retry the plan comes from candidates[rank], not rank 0 (known-bugs N3). Tilt and travel are the worst
    # over its pick and place confs, without the retract; planned_com_margin stays the min over all its confs.
    from cutamp.particle_initialization import _BASE_JOINTS
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: q[:, 5])
    world = SimpleNamespace(model=SimpleNamespace(joint_names=_BASE_JOINTS, home=(0.0, 0.0, 0.7, 0.0, 0.0, 0.0)))
    op = lambda kind, *v: SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type=kind)), values=v)
    skeleton = [op("pick", "block", "grasp1", "left_q1"), op("place", "block", "grasp1", "pose1", "table", "left_q2"),
                op("retract", "block", "grasp1", "left_q2", "traj", "left_q3")]
    conf = lambda z, pitch, margin: torch.tensor([0.0, 0.0, z, pitch, 0.0, margin])
    candidates = [{name: conf(0.7, 0.0, 0.3) for name in ("left_q1", "left_q2", "left_q3")},
                  {"left_q1": conf(0.5, 0.2, 0.1), "left_q2": conf(0.6, 0.1, 0.2), "left_q3": conf(0.0, 1.0, 0.4)}]
    assert alg._planned_metrics(world, skeleton, candidates, 1) == {
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
