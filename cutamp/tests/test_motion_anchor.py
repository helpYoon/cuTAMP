"""COM-anchored pick/place terminals (plan_cspace to the conf), and blocks_t1 end to end on the real legs through
cutamp_demo: its retract values, planted soles, torques and speeds, saved raw plan, and balance metrics against its
exported frames."""
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import cutamp.motion_solver as ms
from cutamp.tests.conftest import needs_cuda


_NAMES = [f"j{i}" for i in range(33)]


def test_cspace_plan_succeeded_trusts_curobo_success():
    """A plan cuRobo rejected stays rejected even though its trajectory ends exactly on the goal (cuRobo pins the last
    waypoint there): Left_Ankle_Pitch past its limit mid-path, endpoint exact, ``cspace_error`` never set."""
    ends_on_goal = SimpleNamespace(position=torch.zeros(1, 1, 4, 18), joint_names=None)

    def result(success):
        return SimpleNamespace(success=torch.tensor([[success]]), cspace_error=None,
                               interpolated_trajectory=ends_on_goal, js_solution=ends_on_goal)

    assert ms._cspace_plan_succeeded(result(False)) is False
    assert ms._cspace_plan_succeeded(result(True)) is True
    assert ms._cspace_plan_succeeded(None) is False


def test_plan_arm_to_conf_targets_the_conf(monkeypatch):
    # _plan_arm_to_conf must plan with plan_cspace to a target built FROM the
    # conf (not a Cartesian pose), and return the cuRobo result on success.
    monkeypatch.setattr(ms, "_cspace_plan_succeeded", lambda result: True)
    calls = {}

    class FakePlanner:
        kinematics = SimpleNamespace(joint_names=_NAMES)

        def plan_cspace(self, target_js, last_js):
            calls["target_js"] = target_js
            calls["last_js"] = last_js
            return "FAKE_RESULT"

    conf = torch.arange(33.0)
    res = ms._plan_arm_to_conf(FakePlanner(), conf, last_js="LAST", disable_obstacle=None, ground_op="OP")
    assert res == "FAKE_RESULT"
    # planned to the conf, at rest, in the planner's joint order, not to a pose
    target = calls["target_js"]
    assert torch.equal(target.position, conf[None]) and not target.velocity.any()
    assert target.joint_names == _NAMES
    assert calls["last_js"] == "LAST"


def test_plan_arm_to_conf_raises_on_failure(monkeypatch):
    monkeypatch.setattr(ms, "_cspace_plan_succeeded", lambda *a: False)

    class FakePlanner:
        kinematics = SimpleNamespace(joint_names=_NAMES)

        def __init__(self):
            self.n = 0
        def plan_cspace(self, *a, **k):
            self.n += 1
            return "R"

    fp = FakePlanner()
    with pytest.raises(RuntimeError, match="cspace-anchored"):
        ms._plan_arm_to_conf(fp, torch.zeros(33), "LAST", disable_obstacle=None, ground_op="OP")
    assert fp.n == 1  # no repeat: the same start and goal plan the same way


def _sole_errors(world, q: torch.Tensor):
    """``([T, 2] m, [T, 2] rad)``: each sole's pose error at the confs ``q`` ``[T, dof]`` against the stance."""
    from cutamp.com_polygon_cost import _fk_state
    from cutamp.costs import curobo_pose_error
    ks = _fk_state(world, q)
    soles = torch.stack([ks.tool_poses.get_link_pose(f, make_contiguous=True).get_matrix()
                         for f in world.model.sole_frames], dim=1)
    return curobo_pose_error(soles, world.stance_soles.expand_as(soles))


@needs_cuda
def test_real_legs_plan_audit(tmp_path, monkeypatch):
    # One real-legs plan end to end (blocks_t1, seed 0, 64 particles, the surface at 0.5 m, the motion plan saved
    # raw): setup_cutamp builds the world from REAL_LEGS, load_t1_container and q_init = REAL_LEGS.home, and the
    # spies only record. The metrics the gate records hold for the exported segments, and the CoM and the multibody
    # ZMP re-measured on those frames stay inside the gate's rectangle (support.gate_half). On the real legs the
    # gate sets each segment's pace, faster or slower than the planner's (measured 2026-11-05: 0.28x to 2.47x over 8
    # segments), so retime_max may be under 1.
    import cutamp.algorithm as alg
    import cutamp.scripts.run_cutamp as rc
    from curobo.types import JointState
    from cutamp.com_polygon_cost import com_margin_from_base, com_zmp_in_support, soles_planted
    from cutamp.config import TAMPConfiguration
    from cutamp.envs.utils import get_env_dir, load_env, raise_surface_height
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.task_planning.constraints import Motion
    from cutamp.tests.test_real_legs_motion import _actuated

    seen = {"retract": []}
    run, solve, top = rc.run_cutamp, alg.solve_curobo, alg.get_top_satisfying_particles

    def run_spy(*args, **kwargs):
        seen["run"] = run(*args, **kwargs)
        return seen["run"]

    def solve_spy(plan_info, particle, world, *args, **kwargs):
        seen["world"] = world
        return solve(plan_info, particle, world, *args, **kwargs)

    def top_spy(plan_info, config, constraint_checker, cost_reducer, n):
        # The satisfying particles' retract values, which the Motion loss and the mask include.
        particles = plan_info["particles"]
        with torch.no_grad():
            cost_dict = plan_info["cost_fn"](plan_info["rollout_fn"](particles), particles)
        mask = constraint_checker.get_mask(cost_dict, verbose=False)
        values = cost_dict[Motion.type]["values"]
        seen["retract"].append({k: values[k][mask].cpu() for k in ("retract_self_collision", "retract_joint_limit")})
        return top(plan_info, config, constraint_checker, cost_reducer, n)

    monkeypatch.setattr(rc, "run_cutamp", run_spy)
    monkeypatch.setattr(alg, "solve_curobo", solve_spy)
    monkeypatch.setattr(alg, "get_top_satisfying_particles", top_spy)
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    raise_surface_height(env, 0.5)
    torch.manual_seed(0)  # run_cutamp's --seed 0
    config = TAMPConfiguration(num_particles=64, curobo_plan=True, enable_visualizer=False,
                               experiment_root=str(tmp_path))
    rc.cutamp_demo(env, config, experiment_id="audit", save_plan="auto")

    curobo_plan, _, metrics = seen["run"]
    assert metrics["found_solution"] and metrics["motion_plan_success"], metrics.get("motion_plan_failures")
    assert metrics["traj_com_min_margin"] >= 0 and metrics["traj_zmp_min_margin"] >= 0
    assert seen["retract"] and all(not bool(v.any()) for d in seen["retract"] for v in d.values())
    world = seen["world"]
    torques = world.segment_torques
    for k, entry in enumerate(curobo_plan):
        pos, rot = _sole_errors(world, ms._full_positions(world, entry["plan"]))
        assert bool(soles_planted(pos, rot).all()), (k, float(pos.max()), float(rot.max()))
        excess = (torques(entry["plan"]).abs() - torques.effort).amax(0)
        assert float(excess.max()) <= 0.0, (k, world.model.joint_names[int(excess.argmax())], float(excess.max()))
        speed = (ms._model_frames(world, entry["plan"]).velocity.abs() / torques.velocity)[:, _actuated(world)]
        assert float(speed.max()) <= 1.0, (k, float(speed.max()))  # the pace the gate set keeps every speed limit

    # The --save_plan auto pickle (decision 9): the raw plan, each segment as planned, in the model's 33 joints, and
    # each segment starting where the one before it ends.
    with open(tmp_path / "audit" / "motion_plan.pkl", "rb") as f:
        raw = pickle.load(f)
    names = list(REAL_LEGS.joint_names)
    assert len(names) == 33
    assert (raw["schema_version"], raw["model"], raw["joint_names"]) == ("stage3-raw", "real_legs", names)
    segments = raw["segments"]
    assert len(segments) == len(curobo_plan)
    for k, (seg, entry) in enumerate(zip(segments, curobo_plan)):
        assert set(seg) == {"arm", "dt", "position", "velocity", "acceleration", "planner_timing"}, k
        assert (seg["arm"], seg["dt"]) == (entry["arm"], entry["dt"]), k
        assert (seg["planner_timing"] is None) == (entry.get("planner_timing") is None), k
        for timing in [seg] + ([seg["planner_timing"]] if seg["planner_timing"] is not None else []):
            n = len(timing["position"])
            for key in ("position", "velocity", "acceleration"):
                assert timing[key].dtype == np.float32 and timing[key].shape == (n, 33), (k, key)
        np.testing.assert_array_equal(seg["position"], ms._full_positions(world, entry["plan"]).cpu().numpy())
        if k:
            np.testing.assert_allclose(seg["position"][0], segments[k - 1]["position"][-1], rtol=0, atol=1e-5,
                                       err_msg=f"segment {k} starts away from segment {k - 1}'s end")

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
    for k, seg in enumerate(segments):
        js = JointState(position=torch.as_tensor(seg["position"], device=world.device),
                        velocity=torch.as_tensor(seg["velocity"], device=world.device),
                        acceleration=torch.as_tensor(seg["acceleration"], device=world.device),
                        joint_names=list(world.model.joint_names))
        com, zmp = com_zmp_in_support(world, js)
        half = world.model.support.gate_half
        margin = com_margin_from_base(com, half).min().item()
        assert margin >= 0, f"segment {k}: an exported CoM is {-100 * margin:.2f} cm outside the gate's rectangle"
        margin = com_margin_from_base(zmp, half).min().item()
        assert margin >= 0, f"segment {k}: an exported ZMP is {-100 * margin:.2f} cm outside the gate's rectangle"
