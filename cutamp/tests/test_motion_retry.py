"""Tests for the next-best-particle motion-plan retry
(algorithm._motion_plan_with_retries, motion_solver._reset_attachment_state)."""
from types import SimpleNamespace

import pytest
import torch

import cutamp.algorithm as alg
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
        return {"left_q0": torch.zeros(2), "grasp1": torch.zeros(4), "pose1": torch.zeros(4),
                "left_q1": torch.tensor([q1, 0.0]), "left_q3": torch.tensor([q3, 0.0])}

    planned = []

    def fake_solve_curobo(plan_info, particle, *a, **k):
        q1, q3 = float(particle["left_q1"][0]), float(particle["left_q3"][0])
        planned.append((q1, q3))
        if q1 == 0 or q3 == 0:  # pick fails on q1 == 0, else place fails on q3 == 0
            e = RuntimeError("plan failed")
            e.op_idx = 1 if q1 == 0 else 2
            raise e
        return "PLAN", {}

    monkeypatch.setattr(alg, "solve_curobo", fake_solve_curobo)
    candidates = [cand(0, 0), cand(0, 1), cand(1, 0), cand(1, 0), cand(1, 1)]
    plan, _, skip, attempts = alg._motion_plan_with_retries(
        {"plan_skeleton": skeleton}, candidates, None, None, None, None,
    )
    # (0, 1) repeats (0, 0) through the failing pick; the second (1, 0) repeats
    # the first through the failing place; (1, 1) differs at the place, so it runs.
    assert planned == [(0, 0), (1, 0), (1, 1)]
    assert (plan, skip, attempts) == ("PLAN", False, 3)


@needs_cuda
def test_reset_puts_placed_blocks_back():
    # A Place moves the block's collision obstacle to the placement. If that
    # particle then fails, the retry on the next one must collide against the
    # block at its start pose, not the failed particle's placement.
    from curobo.types import Pose

    world = make_blocks_t1_world()
    planner = world.get_motion_planner(use_cuda_graph=False)
    cuboids = planner.scene_collision_checker.data.cuboids

    def offset(obj):  # start-pose center in the obstacle's frame; 0 = obstacle at start pose
        inv = cuboids.inv_pose[0, cuboids.get_idx(obj.name, env_idx=0), :7]
        start = world.get_object_pose(obj)[None, :3, 3].contiguous()
        return float(Pose(inv[None, :3].clone(), inv[None, 3:].clone()).transform_points(start).norm())

    for obj in world.movables:
        placed = world.get_object_pose(obj).clone()
        placed[0, 3] += 0.2
        ms._update_world_obstacle_pose(planner, obj.name, placed)  # what a Place does
        assert offset(obj) == pytest.approx(0.2, abs=1e-3)

    ms._reset_attachment_state(planner, world)  # first thing the retry's solve_curobo does
    for obj in world.movables:
        assert offset(obj) < 1e-4, obj.name
