"""Regressions for distinguishing an execution prediction from an independent transition."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cutamp.landing_accuracy import LANDING_BUNDLE
from cutamp.landing_predictor import landing_errors
from cutamp.step_preview import _check_issued_landings
from cutamp.tests.conftest import needs_cuda


def test_composed_stance_miss_is_rejected_even_if_the_two_individual_steps_track():
    pair = torch.zeros(1, 2, 3, dtype=torch.float64)
    displacement = pair.clone()
    displacement[..., 0] = 0.011
    assert landing_errors(displacement, pair)[2].all()
    tracking = landing_errors(2 * displacement, pair)[2]
    assert not tracking.all()
    with pytest.raises(ValueError, match="composed stance"):
        _check_issued_landings(None, None, "LR", None, SimpleNamespace(tracking_ok=tracking))


@needs_cuda
def test_public_stepped_preview_connects_pick_and_place_with_checked_model_landings(tmp_path, monkeypatch):
    """Fixed bindings isolate motion compilation. Learned step Adam selects each actual foot command on this GPU."""
    from dataclasses import replace

    from curobo.scene import Cuboid

    import cutamp.motion_solver as ms
    from cutamp.algorithm import run_cutamp
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs import TAMPEnvironment
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.tests.conftest import only_schedule
    from cutamp.tests.paired_grasp_scene import paired_setup
    from cutamp.utils.plan_processor import raw_motion_plan

    monkeypatch.setattr(ms, "build_motion_planner", lambda *a, **kw: pytest.fail("the direct paths are feasible"))

    fixture = json.loads((Path(__file__).parent / "fixtures/t1_four_step_transfer.json").read_text())
    old, q, inputs = paired_setup("far", place=True)
    part, floor, pad = (Cuboid(**obj) for obj in fixture["scene"])
    env = TAMPEnvironment(
        "fresh_foot_commands", [part], [floor, pad], {"Movable": [part], "Surface": [pad]}, old.goal_state
    )
    only_schedule(monkeypatch, ("LR", "LR"), "left")
    inputs = replace(
        inputs,
        commands=None,
        posture_seeds=tuple(g["posture_seed"] for g in fixture["gaps"]),
        placements={"pose1": fixture["placement"]},
    )
    cfg = TAMPConfiguration(
        step_mode="model",
        num_particles=16,
        max_step_schedules=1,
        max_loop_dur=60,
        landing_bundle=str(LANDING_BUNDLE),
        enable_visualizer=False,
        experiment_root=str(tmp_path),
    )
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
    assert metrics["max_predicted_landing_xy_m"] < 0.020 and metrics["max_predicted_landing_yaw_deg"] < 1.0
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
