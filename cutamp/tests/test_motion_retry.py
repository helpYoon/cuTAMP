"""Tests for the next-best-particle motion-plan retry
(algorithm._motion_plan_with_retries, motion_solver._reset_attachment_state)."""
from types import SimpleNamespace

import pytest
import torch

import cutamp.algorithm as alg
import cutamp.com_polygon_cost as cpc
import cutamp.motion_solver as ms
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


def test_com_margin_is_the_min_over_the_particles_confs(monkeypatch):
    # planned_com_margin: the planned particle's margin, computed as optimize_plan computes best_com_margin
    # for rank 0, which a retry didn't plan (known-bugs N3). Grasps and placements aren't confs.
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: q[:, 0])
    particle = {"left_q0": torch.tensor([0.3, 0.0]), "left_q1": torch.tensor([0.1, 0.0]),
                "right_q3": torch.tensor([0.2, 0.0]), "grasp1": torch.tensor([-1.0, 0.0, 0.0, 0.0])}
    assert cpc.particle_com_margin(None, particle) == pytest.approx(0.1)


def test_planned_metrics_describe_the_planned_particle(monkeypatch):
    # After a retry the plan comes from candidates[rank], not rank 0 (known-bugs N3).
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: q[:, 0])
    candidates = [{"left_q0": torch.tensor([0.3, 0.0])}, {"left_q0": torch.tensor([0.1, 0.0])}]
    assert alg._planned_metrics(None, candidates, 1) == {
        "motion_plan_rank": 1, "planned_com_margin": pytest.approx(0.1)}


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
