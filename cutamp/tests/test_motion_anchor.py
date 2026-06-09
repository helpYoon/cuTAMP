"""Tests for COM-anchored pick/place terminals (plan_cspace to the conf)."""
import pytest
import torch

import cutamp.motion_solver as ms


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


import os

needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _generate_plan_segments(seed: int):
    """Run the full cuTAMP pipeline once for blocks_t1 and return the processed
    motion-plan segments (schema v3) + a TAMPWorld for COM checks.

    NOTE on the real run_cutamp API (the plan's `result["curobo_plan"]` guess
    was wrong): the actual signature is

        run_cutamp(env, config, cost_reducer, constraint_checker,
                   q_init=None, experiment_id=None) -> (curobo_plan, num_satisfying)

    i.e. it takes an `env` (not a prebuilt `world`) plus a CostReducer and a
    ConstraintChecker, builds the world itself via `setup_cutamp`, and returns
    a (raw curobo_plan list, num_satisfying) TUPLE — not a dict. We replicate
    the exact construction done by `cutamp/scripts/run_cutamp.py:cutamp_demo`
    (default_constraint_to_mult / default_constraint_to_tol), then process the
    raw plan to schema v3 with `process_motion_plan`. The TAMPWorld used for
    the COM checks is built separately from the same env/config — its
    kinematics depend only on the robot/env, so it matches the planner's.
    """
    import torch
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from cutamp.robots.t1 import t1_home
    from curobo.types import DeviceCfg
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
    dc = DeviceCfg()
    robot = load_robot_container("t1", dc)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=dc.device)
    world = TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=q_init,
                      enable_com_polygon=True)
    return world, processed["segments"]


@needs_cuda
def test_place_arrival_is_com_in_hull():
    import torch
    from cutamp.robots.t1 import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES, t1_home
    from cutamp.com_polygon_cost import compute_com_polygon_penalties, COM_TOL

    world, segments = _generate_plan_segments(seed=0)
    names = list(world.kinematics.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    q_home = torch.as_tensor(t1_home, dtype=torch.float32,
                             device=world.kinematics.device_cfg.device)

    def frame_q(P, t):
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

    # Every segment's ARRIVAL (last frame) must be COM-in-hull, since each is a
    # cspace-anchored conf or retract-home. (Pre-fix, place arrivals leaked out.)
    assert segments, "no trajectory segments were produced"
    worst = 0.0
    for si, seg in enumerate(segments):
        P = seg["position"]; T = seg["T"]
        q = frame_q(P, T - 1).unsqueeze(0)
        pen = float(compute_com_polygon_penalties(world, {"q": q})["q"][0])
        worst = max(worst, pen)
        assert pen <= COM_TOL, (
            f"segment {si} arrival COM penalty {pen:.6f} > tol {COM_TOL}"
        )
    print(f"worst arrival penalty {worst:.6f} <= {COM_TOL}")
