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
    monkeypatch.setattr(ms, "_cspace_plan_succeeded", lambda result: True)
    calls = {}

    class FakePlanner:
        def plan_cspace(self, target_js, last_js):
            calls["target_js"] = target_js
            calls["last_js"] = last_js
            return "FAKE_RESULT"

    res = ms._plan_arm_to_conf(
        FakePlanner(), torch.zeros(21), last_js="LAST", full_joint_names=["a"],
        disable_obstacle=None, ground_op="OP",
    )
    assert res == "FAKE_RESULT"
    assert calls["target_js"] is sentinel_target  # planned to the conf, not a pose
    assert calls["last_js"] == "LAST"


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
                             disable_obstacle=None, ground_op="OP")
    assert fp.n == 1  # no repeat: the same start and goal plan the same way


def test_audit_counts_each_junction_frame_once(monkeypatch):
    # Each segment is planned from the previous one's last frame, so its frame 0 repeats that frame. The audit
    # counted every junction twice: 7 extra frames in an 8-segment blocks_t1 plan (known-bugs N3).
    from types import SimpleNamespace
    from curobo.types import JointState
    import cutamp.com_polygon_cost as cpc

    monkeypatch.setattr(cpc, "compute_support_mask", lambda world, q: q[:, 0] < 1.0)  # a >= 1: outside
    names = ["a", "b"]

    def segment(*a):
        return {"plan": JointState.from_position(torch.tensor([[x, 0.0] for x in a]), joint_names=names)}

    plans = [segment(0.0, 0.0, 1.0), segment(1.0, 0.0, 0.0), segment(0.0, 0.0, 0.0)]  # outside only at a junction
    world = SimpleNamespace(kinematics=SimpleNamespace(joint_names=names))
    assert ms._audit_traj_com(world, None, plans) == {"traj_com_violation_frac": 1 / 7, "traj_com_frames": 7}


def _generate_plan_segments(seed: int, enable_com_polygon: bool = True):
    """Run the full cuTAMP pipeline once for blocks_t1 and return
    ``(world, segments, overall_metrics)``: a TAMPWorld for COM checks, the
    processed motion-plan segments (schema v3), and the ``overall_metrics``
    dict (includes the measure-only trajectory CoM audit).

    ``run_cutamp(env, config, cost_reducer, constraint_checker, q_init=None,
    experiment_id=None)`` takes an env (not a prebuilt world), builds the
    world itself via ``setup_cutamp``, and returns a
    ``(curobo_plan, num_satisfying, overall_metrics)`` tuple; the raw plan is
    then processed to schema v3 with ``process_motion_plan``. The TAMPWorld
    used for the COM checks is built separately from the same env — its
    kinematics depend only on the robot/env, so it matches the planner's.
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
        enable_com_polygon=enable_com_polygon,
        # Pin coupled re-IK OFF, which keeps the soft costs out of Adam: these
        # tests assert segment schema + arrival COM-in-hull, which the CoM-aware
        # IK + hard gate guarantee regardless; running the default coupled-reik
        # optimization here only doubles runtime (coverage of that lives in
        # test_coupled_reik_smoke).
        coupled_reik=False,
        enable_visualizer=False,
        enable_experiment_logging=False,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    curobo_plan, _num_satisfying, overall_metrics = run_cutamp(
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
    return world, processed["segments"], overall_metrics


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
    # retract-home. (Pre-fix, place arrivals leaked out.) Both seeds exercise
    # the pick and the place anchoring: each plan calls _plan_arm_to_conf once
    # per LeftPick, RightPick, LeftPlace and RightPlace (measured 2026-11-01).
    from cutamp.robots.t1 import t1_home
    from cutamp.com_polygon_cost import compute_com_polygon_penalties, COM_TOL

    world, segments, overall = _generate_plan_segments(seed=seed)
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

    # Trajectory audit is recorded (measure-only) whenever a plan is produced.
    assert overall["motion_plan_success"] is True
    assert 0.0 <= overall["traj_com_violation_frac"] <= 1.0
    assert overall["traj_com_frames"] > 0
    # The balance gate: no frame's CoM or ZMP leaves the support rectangle.
    assert overall["traj_com_violation_frac"] == 0.0
    assert overall["traj_com_min_margin"] >= 0.0 and overall["traj_zmp_min_margin"] >= 0.0
    assert overall["retime_max"] >= 1.0 and overall["plan_duration_s"] > 0.0


@needs_cuda
def test_no_balance_arm_is_not_gated(monkeypatch):
    # The no-CoM ablation arms keep their plans as planned: the gate never runs, and no gate metric is recorded.
    import cutamp.motion_solver as ms

    def boom(*a, **k):
        raise AssertionError("balance gate ran with enable_com_polygon=False")

    monkeypatch.setattr(ms, "_balance_gate", boom)
    _, segments, overall = _generate_plan_segments(seed=0, enable_com_polygon=False)
    assert len(segments) > 0 and "retime_max" not in overall and overall["plan_duration_s"] > 0.0
