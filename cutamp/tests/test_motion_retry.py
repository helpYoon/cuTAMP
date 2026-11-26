"""Tests for the motion stage's playback, demo report and retry reset (motion_solver._reset_attachment_state)."""
from types import SimpleNamespace

import pytest
import torch

import cutamp.algorithm as alg
import cutamp.motion_solver as ms
from cutamp.config import TAMPConfiguration
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def test_demo_reports_why_no_motion_can_be_displayed(tmp_path, caplog, monkeypatch):
    from cutamp.scripts import run_cutamp as cli

    metrics = {
        "time_limit_reached": True, "effective_search_budget_s": 9.5,
        "motion_plan_failures": ["Controller history cannot be issued from the actual arriving configuration",
                                 "T1 step refinement exceeded the planner deadline"],
    }
    monkeypatch.setattr(cli, "run_cutamp", lambda *args, **kwargs: (None, 0, metrics))
    config = TAMPConfiguration(curobo_plan=True, experiment_root=str(tmp_path))
    cli.cutamp_demo(None, config, experiment_id="floor-transfer")
    assert "No complete motion" in caplog.text
    assert "9.5 s" in caplog.text
    assert metrics["motion_plan_failures"][0] in caplog.text
    assert str(tmp_path / "floor-transfer" / "overall_metrics.json") in caplog.text


def test_demo_names_successful_rerun_recording_and_playback_timeline(caplog, monkeypatch):
    from cutamp.scripts import run_cutamp as cli

    monkeypatch.setattr(cli, "run_cutamp", lambda *args, **kwargs: ([object()] * 12, 1, {}))
    caplog.set_level("INFO", logger=cli.__name__)
    cli.cutamp_demo(SimpleNamespace(name="two_steps_floor_transfer"), TAMPConfiguration(),
                    experiment_id="floor-transfer")
    assert "12 segments" in caplog.text and "two_steps_floor_transfer" in caplog.text
    assert "floor-transfer" in caplog.text and "curobo" in caplog.text


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
