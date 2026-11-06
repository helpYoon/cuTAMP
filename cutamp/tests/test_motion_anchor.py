"""COM-anchored pick/place terminals (plan_cspace to the conf), the trajectory audit, and one real-legs blocks_t1
plan's balance metrics against its exported frames."""
import os
import pickle

import numpy as np
import pytest
import torch

import cutamp.motion_solver as ms
from cutamp.tests.conftest import needs_cuda


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
        FakePlanner(), torch.zeros(33), last_js="LAST", full_joint_names=["a"],
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
        ms._plan_arm_to_conf(fp, torch.zeros(33), "LAST", ["a"],
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


@needs_cuda
def test_real_legs_plan_audit(tmp_path, monkeypatch):
    # One real-legs plan as the slow end-to-end test runs it (blocks_t1, seed 0, 64 particles, the surface at 0.5 m,
    # the motion plan saved raw): the metrics the gate records hold for the exported segments, and the CoM and the
    # multibody ZMP re-measured on those frames stay inside the gate's rectangle (compute_support_mask's). On the real
    # legs the gate sets each segment's pace, faster or slower than the planner's (measured 2026-11-05: 0.28x to
    # 2.47x over 8 segments), so retime_max may be under 1.
    import cutamp.algorithm as alg
    import cutamp.scripts.run_cutamp as rc
    from curobo.types import JointState
    from cutamp.com_polygon_cost import com_margin_from_base, com_zmp_in_support, compute_support_mask
    from cutamp.config import TAMPConfiguration
    from cutamp.envs.utils import get_env_dir, load_env, raise_surface_height

    seen = {}
    run, solve = rc.run_cutamp, alg.solve_curobo

    def run_spy(*args, **kwargs):
        seen["run"] = run(*args, **kwargs)
        return seen["run"]

    def solve_spy(plan_info, particle, world, *args, **kwargs):
        seen["world"] = world
        return solve(plan_info, particle, world, *args, **kwargs)

    monkeypatch.setattr(rc, "run_cutamp", run_spy)
    monkeypatch.setattr(alg, "solve_curobo", solve_spy)
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    raise_surface_height(env, 0.5)
    torch.manual_seed(0)  # run_cutamp's --seed 0
    config = TAMPConfiguration(robot="real_legs", num_particles=64, curobo_plan=True, enable_visualizer=False,
                               experiment_root=str(tmp_path))
    rc.cutamp_demo(env, config, experiment_id="audit", save_plan="auto")

    _, _, metrics = seen["run"]
    assert metrics["found_solution"] and metrics["motion_plan_success"], metrics.get("motion_plan_failures")
    assert metrics["traj_com_min_margin"] >= 0 and metrics["traj_zmp_min_margin"] >= 0
    with open(tmp_path / "audit" / "motion_plan.pkl", "rb") as f:
        segments = pickle.load(f)["segments"]
    assert metrics["plan_duration_s"] == pytest.approx(sum((len(s["position"]) - 1) * s["dt"] for s in segments))
    scales = []
    for k, seg in enumerate(segments):
        timing = seg["planner_timing"]
        n = len(seg["position"])
        scales.append((n - 1) / ((n if timing is None else len(timing["position"])) - 1))
        if timing is not None:  # the same path at the planner's timing: from the same start to the same arrival
            # The planner-timing spline misses a joint that moves to 0 by a rounding error (measured 3e-27).
            np.testing.assert_allclose(timing["position"][[0, -1]], seg["position"][[0, -1]], rtol=0, atol=1e-6,
                                       err_msg=f"segment {k}")
    assert metrics["retime_max"] == pytest.approx(max(scales))
    # Measured again on the frames actually exported, so a gated segment that never reaches the export fails here.
    world = seen["world"]
    for k, seg in enumerate(segments):
        js = JointState(position=torch.as_tensor(seg["position"], device=world.device),
                        velocity=torch.as_tensor(seg["velocity"], device=world.device),
                        acceleration=torch.as_tensor(seg["acceleration"], device=world.device),
                        joint_names=list(world.model.joint_names))
        assert bool(compute_support_mask(world, js.position).all()), f"segment {k}: an exported CoM is outside"
        _, zmp = com_zmp_in_support(world, ms._taylor_positions(world, None, js), ms.ZMP_STEP)
        margin = com_margin_from_base(zmp, world.model.support.gate_half).min().item()
        assert margin >= 0, f"segment {k}: an exported ZMP is {-100 * margin:.2f} cm outside the gate's rectangle"
