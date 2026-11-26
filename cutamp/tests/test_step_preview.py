"""Regressions for the stepped motion the learned model predicts."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cutamp.tests.conftest import needs_cuda


def test_a_landing_far_from_its_command_is_still_issued(monkeypatch):
    from cutamp import step_preview
    monkeypatch.setattr(step_preview, "landing_constraint_rows",
                        lambda *a: ({"row": torch.zeros(1)}, torch.ones(1, dtype=torch.bool), None))
    predictor = SimpleNamespace(model=SimpleNamespace(feet=lambda tensors: (torch.zeros(1, 2, 3),)))
    start = SimpleNamespace(tensors=(None,) * 6)
    step_preview._check_issued_landings(predictor, start, "LR", None,
                                        SimpleNamespace(tracking_ok=torch.zeros(1, dtype=torch.bool)))


@needs_cuda
def test_public_stepped_preview_connects_pick_and_place_with_checked_model_landings(tmp_path, monkeypatch):
    """A fixed grasp and foot order isolate motion compilation. The Adam loop optimizes the foot commands, postures
    and placement on this GPU."""
    from curobo.scene import Cuboid

    import cutamp.motion_solver as ms
    from cutamp.algorithm import run_cutamp
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs import TAMPEnvironment
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.tests.conftest import first_schedules, only_schedule
    from cutamp.tests.paired_grasp_scene import paired_setup
    from cutamp.utils.plan_processor import raw_motion_plan

    monkeypatch.setattr(ms, "build_motion_planner", lambda *a, **kw: pytest.fail("the direct paths are feasible"))

    fixture = json.loads((Path(__file__).parent / "fixtures/t1_four_step_transfer.json").read_text())
    old, q, inputs = paired_setup("far", monkeypatch)
    part, floor, pad = (Cuboid(**obj) for obj in fixture["scene"])
    env = TAMPEnvironment(
        "fresh_foot_commands", [part], [floor, pad], {"Movable": [part], "Surface": [pad]}, old.goal_state
    )
    only_schedule(monkeypatch, ("LR", "LR"), "left")
    first_schedules(monkeypatch, 1)
    torch.manual_seed(641)  # the placement is drawn
    cfg = TAMPConfiguration(
        max_loop_dur=60,
        enable_visualizer=False,
        experiment_root=str(tmp_path),
    )
    from cutamp.landing_predictor import LandingPredictor
    from cutamp.step_preview import PredictedSteppedMotion
    from cutamp.step_validation import StepStateAdapter

    in_motion, rollouts, real = [False], [], LandingPredictor.rollout
    holds, real_hold = [], StepStateAdapter.initialize
    real_segments = PredictedSteppedMotion.step_segments

    def segments(self, *a, **k):
        in_motion[0] = True
        try:
            return real_segments(self, *a, **k)
        finally:
            in_motion[0] = False

    def counted(self, *a, **k):
        rollouts.append(in_motion[0])
        return real(self, *a, **k)

    def held(self, *a, **k):
        holds.append(in_motion[0])
        return real_hold(self, *a, **k)

    monkeypatch.setattr(PredictedSteppedMotion, "step_segments", segments)
    monkeypatch.setattr(LandingPredictor, "rollout", counted)
    monkeypatch.setattr(StepStateAdapter, "initialize", held)
    path, _, metrics = run_cutamp(
        env,
        cfg,
        CostReducer(default_constraint_to_mult.copy()),
        ConstraintChecker(default_constraint_to_tol.copy()),
        q_init=q,
        experiment_id="preview",
        step_inputs=inputs,
    )
    assert metrics["status"] == "model_predicted_motion", metrics["motion_plan_failures"]
    assert metrics["motion_validation"] == "model_predicted" and metrics["independent_motion"] == "untested"
    assert math.isfinite(metrics["max_predicted_landing_xy_m"]) and math.isfinite(metrics["max_predicted_landing_yaw_deg"])
    assert len(metrics["connected_motion_gaps"]) == 2
    raw = raw_motion_plan(path, REAL_LEGS)
    assert raw["motion_validation"] == "model_predicted"
    segs = raw["segments"]
    assert [s["gripper"] for s in segs if s["gripper"]] == ["close", "open"]
    assert [s["phase"] for s in segs if s["phase"].startswith("model_step")] == ["model_step_L", "model_step_R"] * 2
    assert segs[-1]["phase"] == "back_out"
    for s in segs:
        assert s["head"].shape == (len(s["position"]), 2)
        if s["phase"].startswith("model_"):
            assert s["step"]["source"] == "frozen_policy_world_model"
    # The world model ran in the loop, never in the motion stage.
    assert rollouts and not any(rollouts) and not any(holds)


def test_a_stored_step_must_start_where_the_arm_motion_ended():
    from cutamp import step_preview
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.step_validation import StepStateAdapter

    adapter = StepStateAdapter(SimpleNamespace(metadata={"joint_names": ["AAHead_yaw", "Head_pitch",
                                                                         *REAL_LEGS.joint_names[6:]]}))
    q = torch.tensor(REAL_LEGS.home).double()[None]
    preview = object.__new__(step_preview.PredictedSteppedMotion)
    preview.adapter, preview.deadline, preview.by_op = adapter, None, {0: (0, 0)}
    preview.inputs = {0: (SimpleNamespace(tensors=adapter.physical(q)), None, None, None)}  # a stored start, no hold
    arrived = q.clone()
    arrived[0, 0] += 0.006  # 6 mm from the stored start, past the 5 mm join
    with pytest.raises(ValueError, match="Stored step does not start where the arm motion ended"):
        preview.step_segments(0, SimpleNamespace(position=arrived, velocity=None), {}, {})


def test_fallback_uses_current_scene_and_attachments_at_each_solve(monkeypatch):
    from cutamp import step_preview

    preview = object.__new__(step_preview.PredictedSteppedMotion)
    preview.stance, preview.world = object(), object()
    preview.physical_cfg = preview._fallback = None
    source = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    target = torch.zeros_like(source)
    manager = lambda data: SimpleNamespace(kinematics_params=SimpleNamespace(link_spheres=data))
    preview.attachment_manager = manager(source)
    enabled, positions = {}, {}
    scene = SimpleNamespace(
        enable_obstacle=lambda name, enable: enabled.update({name: enable}),
        update_obstacle_pose=lambda name, w_obj_pose: positions.update({name: w_obj_pose.position.clone()}),
    )
    full = SimpleNamespace(scene_collision_checker=scene, attachment_manager=manager(target))
    built = []

    def build(world, *, initial_soles, robot_cfg):
        built.append((world, initial_soles, robot_cfg))
        return full

    monkeypatch.setattr(step_preview.ms, "build_motion_planner", build)
    pose = torch.eye(4)
    assert preview.fallback_planner({"box": pose}, {"left": ("box", None)}) is full
    assert not enabled["box"] and torch.equal(target, source)
    source.fill_(-100)
    pose[0, 3] = 0.7
    assert preview.fallback_planner({"box": pose}, {}) is full
    assert enabled["box"] and torch.equal(target, source)
    torch.testing.assert_close(positions["box"].flatten(), pose[:3, 3])
    assert built == [(preview.world, preview.stance, None)]
