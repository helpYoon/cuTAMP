"""Tests for COM-anchored pick/place terminals (plan_cspace to the conf)."""
import os

import pytest
import torch

import cutamp.motion_solver as ms
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def test_plan_arm_to_conf_targets_the_conf(monkeypatch):
    # _plan_arm_to_conf must plan with plan_cspace to a target built FROM the
    # conf (not a Cartesian pose), and return the cuRobo result on success.
    sentinel_target = object()
    monkeypatch.setattr(ms, "_planner_js_from_full",
                        lambda planner, conf, names: sentinel_target)
    monkeypatch.setattr(ms, "_cspace_plan_succeeded", lambda result, target: True)
    calls = {}

    class FakePlanner:
        def plan_cspace(self, target_js, last_js, max_attempts, enable_graph_attempt):
            calls["target_js"] = target_js
            calls["last_js"] = last_js
            calls["max_attempts"] = max_attempts
            return "FAKE_RESULT"

    res = ms._plan_arm_to_conf(
        FakePlanner(), torch.zeros(21), last_js="LAST", full_joint_names=["a"],
        disable_obstacle=None, retries=1, ground_op="OP",
    )
    assert res == "FAKE_RESULT"
    assert calls["target_js"] is sentinel_target  # planned to the conf, not a pose
    assert calls["last_js"] == "LAST"
    assert calls["max_attempts"] == ms.CSPACE_MAX_ATTEMPTS


def test_plan_arm_to_conf_raises_on_failure(monkeypatch):
    monkeypatch.setattr(ms, "_planner_js_from_full", lambda *a: object())
    monkeypatch.setattr(ms, "_cspace_plan_succeeded", lambda *a: False)

    class FakePlanner:
        def __init__(self):
            self.n = 0
        def plan_cspace(self, *a, **k):
            self.n += 1
            return "R"

    fp = FakePlanner()
    with pytest.raises(RuntimeError, match="cspace-anchored"):
        ms._plan_arm_to_conf(fp, torch.zeros(21), "LAST", ["a"],
                             disable_obstacle=None, retries=3, ground_op="OP")
    assert fp.n == 3  # retried `retries` times before raising


def _generate_plan_segments(seed: int):
    """Run the full cuTAMP pipeline once for blocks_t1 and return
    ``(world, segments)``: a TAMPWorld for COM checks plus the processed
    motion-plan segments (schema v3).

    ``run_cutamp(env, config, cost_reducer, constraint_checker, q_init=None,
    experiment_id=None)`` takes an env (not a prebuilt world), builds the
    world itself via ``setup_cutamp``, and returns a
    ``(curobo_plan, num_satisfying)`` tuple; the raw plan is then processed
    to schema v3 with ``process_motion_plan``. The TAMPWorld used for the
    COM checks is built separately from the same env — its kinematics
    depend only on the robot/env, so it matches the planner's.
    """
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.config import TAMPConfiguration
    from cutamp.algorithm import run_cutamp
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.scripts.utils import (
        default_constraint_to_mult,
        default_constraint_to_tol,
    )
    from cutamp.utils.plan_processor import process_motion_plan

    torch.manual_seed(seed)
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    config = TAMPConfiguration(
        num_particles=64,
        num_opt_steps=50,
        curobo_plan=True,
        enable_com_polygon=True,
        # Pin the soft-cost trio OFF: these tests assert segment schema +
        # arrival COM-in-hull, which the CoM-aware IK + hard gate guarantee
        # regardless; running the default coupled-reik optimization here only
        # doubles runtime (trio coverage lives in test_coupled_reik_smoke).
        optimize_soft_costs=False,
        coupled_reik=False,
        enable_visualizer=False,
        enable_experiment_logging=False,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    curobo_plan, _num_satisfying = run_cutamp(
        env, config, cost_reducer, constraint_checker,
        experiment_id="test_place_arrival",
    )
    if curobo_plan is None:
        raise RuntimeError(
            "run_cutamp produced no motion plan (curobo_plan is None) for "
            f"seed={seed}; cannot check place arrival COM."
        )
    processed = process_motion_plan(curobo_plan)

    # Separate TAMPWorld for COM checks (kinematics depend only on env/robot).
    world = make_blocks_t1_world()
    return world, processed["segments"]


def _frame_q(q_home, idx, P, t):
    """Full-dof configuration for frame ``t`` of a segment's position dict."""
    from cutamp.robots.t1 import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

    q = q_home.clone()
    q[idx["Torso_Pitch"]] = float(P["trunk_pitch"][t])
    q[idx["Waist_Yaw"]] = float(P["trunk_yaw"][t])
    q[idx["ankle_pitch"]] = float(P["ankle_pitch"][t])
    q[idx["knee_pitch"]] = float(P["knee_pitch"][t])
    for j, n in enumerate(LEFT_ARM_JOINT_NAMES):
        q[idx[n]] = float(P["left_arm"][t][j])
    for j, n in enumerate(RIGHT_ARM_JOINT_NAMES):
        q[idx[n]] = float(P["right_arm"][t][j])
    return q


@needs_cuda
@pytest.mark.parametrize("seed", [0, 1])
def test_all_segment_arrivals_com_in_hull(seed):
    # A full plan generates (no RuntimeError from solve_curobo), produces
    # segments, and EVERY segment's ARRIVAL (last frame) — pick, place,
    # retract — is COM-in-hull, since each is a cspace-anchored conf or
    # retract-home. (Pre-fix, place arrivals leaked out.) Seed 1 exercises
    # the pick branch anchoring added in Task 3.
    from cutamp.robots.t1 import t1_home
    from cutamp.com_polygon_cost import compute_com_polygon_penalties, COM_TOL

    world, segments = _generate_plan_segments(seed=seed)
    assert len(segments) > 0, "plan produced no segments"
    names = list(world.kinematics.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    q_home = torch.as_tensor(t1_home, dtype=torch.float32,
                             device=world.kinematics.device_cfg.device)
    worst = 0.0
    for si, seg in enumerate(segments):
        P = seg["position"]; T = seg["T"]
        q = _frame_q(q_home, idx, P, T - 1).unsqueeze(0)
        pen = float(compute_com_polygon_penalties(world, {"q": q})["q"][0])
        worst = max(worst, pen)
        assert pen <= COM_TOL, (
            f"segment {si} arrival COM penalty {pen:.6f} > tol {COM_TOL}"
        )
    print(f"all {len(segments)} arrivals in hull; worst penalty {worst:.6f} <= {COM_TOL}")
