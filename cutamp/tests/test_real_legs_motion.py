"""Motion planning on the real legs with planted feet and cuRobo's torque check (design 7.4a, 7.5): the floor's push on
every trajopt transition, SegmentTorques against trajopt's own torques, the balance gate's sole, tilt and torque checks,
the failed-trajopt reason, and blocks_t1 end to end through cutamp_demo, with its saved raw plan.

The slow end-to-end test comes first: cutamp_demo builds its own world, and the module's ``real_world`` and ``planner``
must not exist beside it (two worlds OOM a 24 GiB card). Module fixtures are built at their first use, so pytest's
file order keeps them apart."""

import dataclasses
import math
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cutamp.robots.t1 import REAL_LEGS
from cutamp.tests.conftest import needs_cuda

needs_slow = pytest.mark.skipif(os.environ.get("CUTAMP_SLOW") != "1", reason="slow live test; set CUTAMP_SLOW=1")

DT = 0.025  # the hand-built segments' frame spacing, s
SHOULDERS = ("Left_Shoulder_Pitch", "Right_Shoulder_Pitch")


def _ramp(world, moves, T):
    """The rest-to-rest min-jerk move ``moves`` ({joint: delta} from home) over T s every DT s, the other joints at
    home: a JointState in ``world.model.joint_names`` order."""
    from curobo.types import JointState
    from cutamp.tests.test_balance_gate import _min_jerk
    names = list(world.model.joint_names)
    n = int(round(T / DT)) + 1
    _, p, v, a = (torch.as_tensor(x, dtype=torch.float32, device=world.device) for x in _min_jerk(T, n))
    pos = torch.tensor(world.model.home, device=world.device).repeat(n, 1)
    vel, acc = torch.zeros_like(pos), torch.zeros_like(pos)
    for name, delta in moves.items():
        j = names.index(name)
        pos[:, j] += delta * p
        vel[:, j], acc[:, j] = delta * v, delta * a
    return JointState(position=pos, velocity=vel, acceleration=acc, joint_names=names)


def _still(world, q: torch.Tensor, T: float):
    """The conf ``q`` held at rest for T s, every DT s: a JointState in ``world.model.joint_names`` order."""
    from curobo.types import JointState
    pos = q.to(world.device, torch.float32).repeat(int(round(T / DT)) + 1, 1)
    return JointState(position=pos, velocity=torch.zeros_like(pos), acceleration=torch.zeros_like(pos),
                      joint_names=list(world.model.joint_names))


def _shoulder_peak(world, js) -> float:
    """The larger shoulder pitch's peak |torque| over ``js``'s frames, by ``world.segment_torques``."""
    columns = [world.model.joint_names.index(j) for j in SHOULDERS]
    return float(world.segment_torques(js).abs()[:, columns].max())


def _swing(world):
    """Both shoulder pitches from home's 0.2 rad to -2.0 rad in 0.5 s, the soles fixed: fast enough that the gate
    slows it for its ZMP (by 2.6x), and within every effort, slowed or not. In 0.3 s its ZMP would need 4.6x, past
    the gate's 4x cap, and each ankle pitch would carry 117 N m against its 73.1."""
    return _ramp(world, {j: -2.2 for j in SHOULDERS}, 0.5)


def _sole_errors(world, q: torch.Tensor):
    """``([T, 2] m, [T, 2] rad)``: each sole's pose error at the confs ``q`` ``[T, dof]`` against the stance."""
    from cutamp.com_polygon_cost import _fk_state
    from cutamp.costs import curobo_pose_error
    ks = _fk_state(world, q)
    soles = torch.stack([ks.tool_poses.get_link_pose(f, make_contiguous=True).get_matrix()
                         for f in world.model.sole_frames], dim=1)
    return curobo_pose_error(soles, world.stance_soles.expand_as(soles))


def _transitions(planner):
    """Every trajopt transition: each rollout's cost (``transition_model``) and success-test
    (``metrics_transition_model``) ones, and the interpolated re-check's."""
    ts = planner.trajopt_solver
    rollouts = list(ts.get_all_rollout_instances()) + list(ts.additional_metrics_rollouts.values())
    return [tm for r in rollouts for tm in (r.transition_model, r.metrics_transition_model)]


def _transition_torques(tm, js) -> torch.Tensor:
    """``[T, dof]``: the joint torques the trajopt transition ``tm`` computes (its compute_augmented_state) at the
    frames of ``js``. The frames are cycled to fill ``tm``'s current batch shape, one chunk at a time: RNEA and the
    floor's push are per frame, and a call at another shape would reallocate buffers that cuRobo's CUDA graphs
    captured."""
    from curobo.types import JointState
    js = js.reorder(list(tm.joint_names))
    b, h = tm.robot_model._batch, tm.robot_model._horizon
    assert b * h <= tm.robot_dynamics._tau_buffer.shape[0]  # so Dynamics does not grow its buffers either
    q, v, a = (x.reshape(-1, x.shape[-1]) for x in (js.position, js.velocity, js.acceleration))
    out = []
    for start in range(0, q.shape[0], b * h):
        idx = (start + torch.arange(b * h, device=q.device)) % q.shape[0]
        chunk = JointState(position=q[idx].view(b, h, -1), velocity=v[idx].view(b, h, -1),
                           acceleration=a[idx].view(b, h, -1), jerk=torch.zeros_like(q[idx]).view(b, h, -1),
                           joint_names=list(tm.joint_names))
        tau = tm.compute_augmented_state(chunk).joint_torque.reshape(b * h, -1)
        out.append(tau[: q.shape[0] - start].clone())
    return torch.cat(out)


@needs_cuda
@needs_slow
def test_real_legs_blocks_t1_end_to_end(tmp_path, monkeypatch):
    # The only test of the --robot real_legs path (ruling R18): cutamp_demo -> setup_cutamp builds the world from
    # get_model, load_t1_container and q_init = REAL_LEGS.home. The spies only record.
    import cutamp.algorithm as alg
    import cutamp.scripts.run_cutamp as rc
    from cutamp.com_polygon_cost import soles_planted
    from cutamp.config import TAMPConfiguration
    from cutamp.envs.utils import get_env_dir, load_env, raise_surface_height
    from cutamp.motion_solver import _full_positions
    from cutamp.task_planning.constraints import Motion

    seen = {"retract": []}
    run, solve, top = rc.run_cutamp, alg.solve_curobo, alg.get_top_satisfying_particles

    def run_spy(*args, **kwargs):
        seen["run"] = run(*args, **kwargs)
        return seen["run"]

    def solve_spy(plan_info, particle, world, *args, **kwargs):
        seen["world"] = world
        return solve(plan_info, particle, world, *args, **kwargs)

    def top_spy(plan_info, config, constraint_checker, cost_reducer, n):
        # The satisfying particles' retract values (Task 4 put them in the Motion loss and the mask).
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
    config = TAMPConfiguration(robot="real_legs", num_particles=64, curobo_plan=True, enable_visualizer=False,
                               experiment_root=str(tmp_path))
    rc.cutamp_demo(env, config, experiment_id="e2e", save_plan="auto")

    curobo_plan, _, metrics = seen["run"]
    assert metrics["found_solution"] and metrics["motion_plan_success"], metrics.get("motion_plan_failures")
    assert metrics["traj_com_violation_frac"] == 0
    assert seen["retract"] and all(not bool(v.any()) for d in seen["retract"] for v in d.values())
    world = seen["world"]
    torques = world.segment_torques
    for k, entry in enumerate(curobo_plan):
        pos, rot = _sole_errors(world, _full_positions(world, None, entry["plan"]))
        assert bool(soles_planted(pos, rot).all()), (k, float(pos.max()), float(rot.max()))
        excess = (torques(entry["plan"]).abs() - torques.effort).amax(0)
        assert float(excess.max()) <= 0.0, (k, world.model.joint_names[int(excess.argmax())], float(excess.max()))

    # The --save_plan auto pickle (decision 9): the raw plan, each segment as planned, in the model's 33 joints, and
    # each segment starting where the one before it ends.
    with open(tmp_path / "e2e" / "motion_plan.pkl", "rb") as f:
        raw = pickle.load(f)
    names = list(REAL_LEGS.joint_names)
    assert len(names) == 33
    assert (raw["schema_version"], raw["model"], raw["joint_names"]) == ("stage3-raw", "real_legs", names)
    assert len(raw["segments"]) == len(curobo_plan)
    for k, (seg, entry) in enumerate(zip(raw["segments"], curobo_plan)):
        assert set(seg) == {"arm", "dt", "position", "velocity", "acceleration", "planner_timing"}, k
        assert (seg["arm"], seg["dt"]) == (entry["arm"], entry["dt"]), k
        assert (seg["planner_timing"] is None) == (entry.get("planner_timing") is None), k
        for timing in [seg] + ([seg["planner_timing"]] if seg["planner_timing"] is not None else []):
            n = len(timing["position"])
            for key in ("position", "velocity", "acceleration"):
                assert timing[key].dtype == np.float32 and timing[key].shape == (n, 33), (k, key)
        np.testing.assert_array_equal(seg["position"], _full_positions(world, None, entry["plan"]).cpu().numpy())
        if k:
            np.testing.assert_allclose(seg["position"][0], raw["segments"][k - 1]["position"][-1], rtol=0, atol=1e-5,
                                       err_msg=f"segment {k} starts away from segment {k - 1}'s end")


@pytest.fixture(scope="module")
def planner(real_world):
    """``real_world``'s motion planner as solve_curobo builds it, built once."""
    from cutamp.config import TAMPConfiguration
    from cutamp.motion_solver import build_motion_planner
    with build_motion_planner(real_world, TAMPConfiguration(robot="real_legs")) as mp:
        yield mp


@needs_cuda
def test_static_home_torques(real_world):
    # 7.4a's feet-on-the-floor statics at home (section 11's scratch run): each knee -4.57 N m and each ankle pitch
    # +3.92 N m, and the floor leaves the six base joints nothing.
    names = list(real_world.model.joint_names)
    torques = real_world.segment_torques
    tau = torques(_ramp(real_world, {}, 2 * DT))  # 3 frames of home at rest
    assert tau.shape == (3, len(names)) and torques.effort.shape == (len(names),)
    for side in ("Left", "Right"):
        knee, ankle = tau[:, names.index(f"{side}_Knee_Pitch")], tau[:, names.index(f"{side}_Ankle_Pitch")]
        assert float((knee + 4.57).abs().max()) <= 0.3, knee
        assert float((ankle - 3.92).abs().max()) <= 0.3, ankle
    assert float(tau[:, real_world.model.base].abs().max()) <= 0.05


@needs_cuda
def test_trajopt_caps_ankle_pitch_at_its_top_speed(planner):
    # The fork applies cspace.velocity_scale twice (ruling R14): t1_29dof.yml carries sqrt(7.29 / 12.57) on both
    # Ankle_Pitch joints, so every trajopt transition must read 7.29 rad/s, the motor's top speed (design 7.4a).
    for tm in _transitions(planner):
        names = list(tm.joint_names)
        for side in ("Left", "Right"):
            v = tm.joint_limits.velocity[:, names.index(f"{side}_Ankle_Pitch")]
            torch.testing.assert_close(v.abs(), torch.full_like(v, 7.29), atol=1e-4, rtol=0)


@needs_cuda
def test_planner_hooks_every_trajopt_transition(real_world, planner):
    # Stage 2's first install missed the metrics transitions: the success test then saw legs in the air. Every
    # trajopt transition carries the floor's push, and at rest at home it leaves the base joints nothing.
    transitions = _transitions(planner)
    assert len(transitions) >= 4
    assert all(tm._external_wrench_fn is not None for tm in transitions)
    assert all(r.transition_model._external_wrench_fn is None for r in planner.ik_solver.get_all_rollout_instances())
    # The trajopt CoM cost finds its support frame among the planner's tool poses (ruling R23).
    assert "stance" in planner.trajopt_solver.tool_frames
    assert all("stance" in tm.robot_model.tool_frames for tm in transitions)
    names = list(planner.trajopt_solver.joint_names)
    base = [names.index(j) for j in real_world.model.joint_names[real_world.model.base]]
    rest = _ramp(real_world, {}, 2 * DT)
    loads = [float(_transition_torques(tm, rest)[:, base].abs().max()) for tm in transitions]
    assert max(loads) <= 0.05, loads


@needs_cuda
def test_planner_registers_the_tilt_cost_only_under_a_bound(real_world, planner):
    # Route A's envelope sets no tilt bound (ruling R28), so its planner has no TrunkTiltCost; under an envelope with
    # one, trajopt bounds the tilt (weight 1e5, the bound's cosine, on the base's pitch and roll columns).
    from cutamp._curobo_internals import iter_rollouts
    from cutamp.joint_limit_cost import JointLimitMarginCost, TrunkTiltCost
    from cutamp.robots.t1 import get_t1_motion_planner
    from cutamp.utils.common import get_world_cfg

    def extra_costs(mp, kind):
        return [c for r in iter_rollouts(mp) for mgr in (r.cost_manager, r.metrics_cost_manager)
                for c in getattr(mgr, "_extra_costs", {}).values() if isinstance(c, kind)]

    assert REAL_LEGS.envelope.trunk_tilt_max is None
    assert not extra_costs(planner, TrunkTiltCost) and not extra_costs(planner, JointLimitMarginCost)
    bounded = dataclasses.replace(REAL_LEGS, envelope=REAL_LEGS.envelope._replace(trunk_tilt_max=1.0))
    with get_t1_motion_planner(bounded, get_world_cfg(real_world.env, include_movables=True),
                               use_cuda_graph=False, device_cfg=real_world.device_cfg) as mp:
        costs = extra_costs(mp, TrunkTiltCost)
        names = list(mp.kinematics.joint_names)
        assert costs and not extra_costs(mp, JointLimitMarginCost)
        columns = (names.index("base_j_pitch"), names.index("base_j_roll"))
        for c in costs:
            assert (c.config.pitch_index, c.config.roll_index) == columns
            assert c.config.cos_max == math.cos(1.0) and float(c._weight.max()) == 1e5


def _plan_reach(world, real_plan, planner):
    """A planned home -> reach segment: plan_cspace's result for the first pick conf of ``real_plan`` that plans."""
    from cutamp.motion_solver import _plan_arm_to_conf, _planner_js_from_full
    skeleton, particles = real_plan
    op = next(op for op in skeleton if op.operator.metadata.action_type == "pick")
    obj, conf = op.values[0], op.values[-1]
    names = list(world.model.joint_names)
    home = torch.tensor(world.model.home, device=world.device)
    start = _planner_js_from_full(planner, home, names)
    for q in particles[conf][(particles[conf] != home).any(-1)][:8]:  # IK-solved reaches; a few may not plan
        try:
            return _plan_arm_to_conf(planner, q, start, names, disable_obstacle=obj, ground_op=op)
        except RuntimeError:
            continue
    raise AssertionError("no reach planned from home")


@needs_cuda
def test_segment_torques_equal_trajopt_s(real_world, real_plan, planner):
    # The gate re-runs cuRobo's own two passes: on a planned home -> reach segment, SegmentTorques equals the torque
    # every trajopt transition computes, the cost's and the success test's.
    from cutamp.motion_solver import _interp_plan
    names = list(real_world.model.joint_names)
    plan = planner.kinematics.get_active_js(_interp_plan(_plan_reach(real_world, real_plan, planner)))
    tau = real_world.segment_torques(plan)
    assert tau.shape == (plan.position.shape[-2], len(names))
    assert float(tau[:, real_world.model.body].abs().max()) > 1.0  # the legs carry the robot: not a vacuous match
    for tm in _transitions(planner):
        torch.testing.assert_close(_transition_torques(tm, plan), tau, atol=1e-3, rtol=0)


@needs_cuda
def test_planned_torques_equal_segment_torques(real_world, real_plan, planner):
    # The torques trajopt computed while it planned, read where the plan left them, equal SegmentTorques on the same
    # frames: the metrics rollout's (the trajopt horizon, computed inside its replayed CUDA graph) and the
    # interpolated rollout's last ones (its success test). A graph captured before the floor's push was hooked would
    # replay without the push, while the eager calls above would still see it.
    from curobo.types import JointState
    _plan_reach(real_world, real_plan, planner)
    ts = planner.trajopt_solver
    names = list(real_world.model.joint_names)
    interpolated = ts.additional_metrics_rollouts["interpolated_rollout"]
    executors = {"metrics": ts.metrics_rollout._compute_metrics_from_action_executor,
                 "interpolated": interpolated._compute_metrics_from_state_executor}
    for label, executor in executors.items():
        assert executor._graph is not None, label  # captured, so the plan's last call replayed it
        state = executor._graph_output[0].state
        js, columns = state.joint_state, list(state.joint_state.joint_names)
        n = len(columns)
        frames = JointState(position=js.position.reshape(-1, n), velocity=js.velocity.reshape(-1, n),
                            acceleration=js.acceleration.reshape(-1, n), joint_names=columns)
        tau = real_world.segment_torques(frames)
        planned = state.joint_torque.reshape(-1, n)[:, [columns.index(j) for j in names]]
        assert float(tau[:, real_world.model.body].abs().max()) > 1.0, label  # the legs carry the robot
        torch.testing.assert_close(planned, tau, atol=1e-3, rtol=0, msg=lambda m: f"{label}: {m}")


@needs_cuda
def test_gate_fails_a_slowed_segment_past_a_torque_limit(real_world, planner, monkeypatch):
    # Trajopt bounds the torque at the planner's timing only; the gate's slowed, resampled segment runs the two
    # passes again (design 7.4a), and fails past an effort. The slowed swing's shoulders carry less than the planned
    # one's, so an effort between the two peaks passes only if the gate checks the frames it returns.
    from cutamp.motion_solver import _balance_gate
    swing = _swing(real_world)
    out, stats, fast = _balance_gate(real_world, planner, swing, DT)
    assert stats["retime"] > 1.0 and fast is not None  # slowed, and passes at the real efforts
    planned_peak, slowed_peak = _shoulder_peak(real_world, swing), _shoulder_peak(real_world, out)
    assert planned_peak > slowed_peak
    torques = real_world.segment_torques
    columns = [real_world.model.joint_names.index(j) for j in SHOULDERS]
    for shoulder_effort, passes in ((0.5 * (planned_peak + slowed_peak), True), (0.1, False)):
        effort = torques.effort.clone()
        effort[columns] = shoulder_effort
        monkeypatch.setattr(torques, "effort", effort)
        if passes:
            _balance_gate(real_world, planner, swing, DT)
        else:
            with pytest.raises(RuntimeError, match="torque"):
                _balance_gate(real_world, planner, swing, DT)


@needs_cuda
def test_gate_checks_the_soles_and_tilt_of_the_frames_it_returns(real_world, planner, monkeypatch):
    # A slowed segment's frames are spline resamples between the planned ones, and they are what is saved and
    # executed: the gate checks their soles and tilt as well as the planned frames'.
    import cutamp.motion_solver as ms
    checked, check = [], ms._check_planted

    def spy(world, q):
        checked.append(q.clone())
        return check(world, q)

    monkeypatch.setattr(ms, "_check_planted", spy)
    swing = _swing(real_world)
    out, stats, _ = ms._balance_gate(real_world, planner, swing, DT)
    assert stats["retime"] > 1.0
    assert [q.shape[0] for q in checked] == [swing.position.shape[-2], out.position.shape[-2]]
    torch.testing.assert_close(checked[1], ms._full_positions(real_world, planner, out), rtol=0, atol=0)


@needs_cuda
def test_gate_rejects_sole_drift(real_world, planner, monkeypatch):
    import cutamp.com_polygon_cost as cpc
    from cutamp.motion_solver import _audit_traj_com, _balance_gate
    # Every CoM and ZMP margin the real-leg gate and audit take is against the gate's rectangle about the stance
    # (ruling R23).
    halves, margin = [], cpc.com_margin_from_base

    def spy(xy, half):
        halves.append(tuple(half))
        return margin(xy, half)

    monkeypatch.setattr(cpc, "com_margin_from_base", spy)
    still = _ramp(real_world, {}, 1.0)
    out, stats, fast = _balance_gate(real_world, planner, still, DT)
    assert out is still and stats["retime"] == 1.0 and fast is None
    audit = _audit_traj_com(real_world, planner, [{"plan": still}])
    assert audit == {"traj_com_violation_frac": 0.0, "traj_com_frames": still.position.shape[0]}
    assert halves and set(halves) == {REAL_LEGS.support.gate_half}
    # The whole robot slides 1 cm with the legs fixed: the soles leave the stance.
    with pytest.raises(RuntimeError, match="sole"):
        _balance_gate(real_world, planner, _ramp(real_world, {"base_j_x": 0.01}, 1.0), DT)


def _bowed(world, real_plan, planner):
    """A still segment at a bowed conf the gate passes under route A's envelope, and the conf's tilt, acos(cos p cos
    r) of the base's pitch p and roll r (YXZ): an IK-solved pick conf whose base both pitches and rolls."""
    from cutamp.motion_solver import _balance_gate
    skeleton, particles = real_plan
    names = list(world.model.joint_names)
    pitch, roll = names.index("base_j_pitch"), names.index("base_j_roll")
    for op in skeleton:
        if op.operator.metadata.action_type != "pick":
            continue
        q = particles[op.values[-1]].double()
        tilt = torch.acos(torch.cos(q[:, pitch]) * torch.cos(q[:, roll]))
        for k in torch.nonzero((tilt > 0.2) & (q[:, roll].abs() > 0.05)).flatten().tolist():
            still = _still(world, q[k], 0.5)
            try:
                _balance_gate(world, planner, still, DT)
            except RuntimeError:  # its soles or CoM fail on their own
                continue
            return still, float(tilt[k])
    raise AssertionError("no bowed pick conf passes the gate")


@needs_cuda
def test_gate_bounds_a_bowed_confs_tilt(real_world, real_plan, planner, monkeypatch):
    # Under an envelope with a tilt bound the gate checks every frame's Trunk tilt: a bound 1e-3 rad under a bowed
    # conf's acos(cos p cos r) fails it, and one 1e-3 rad over passes it.
    from cutamp.motion_solver import _balance_gate
    still, tilt = _bowed(real_world, real_plan, planner)
    for bound, passes in ((tilt + 1e-3, True), (tilt - 1e-3, False)):
        model = dataclasses.replace(REAL_LEGS, envelope=REAL_LEGS.envelope._replace(trunk_tilt_max=bound))
        monkeypatch.setattr(real_world, "robot_container", dataclasses.replace(real_world.robot_container, model=model))
        if passes:
            _balance_gate(real_world, planner, still, DT)
        else:
            with pytest.raises(RuntimeError, match="tilt"):
                _balance_gate(real_world, planner, still, DT)


@needs_cuda
def test_gate_tilt_check_fails_a_nan_tilt(real_world, monkeypatch):
    # A NaN Trunk pose fails the tilt check (the soles stay finite here, so the tilt check is the one that runs).
    import cutamp.com_polygon_cost as cpc
    from cutamp.motion_solver import _check_planted
    fk = cpc._fk_state
    monkeypatch.setattr(cpc, "_fk_state", lambda world, q, kin=None: fk(world, q if kin is None else q * math.nan, kin))
    bounded = dataclasses.replace(REAL_LEGS, envelope=REAL_LEGS.envelope._replace(trunk_tilt_max=1.0))
    monkeypatch.setattr(real_world, "robot_container", dataclasses.replace(real_world.robot_container, model=bounded))
    with pytest.raises(RuntimeError, match="tilt"):
        _check_planted(real_world, _ramp(real_world, {}, 2 * DT).position)


def _result(js_solution, interpolated=None):
    """A trajopt result's two trajectories: the horizon's (``js_solution``) and the interpolated one."""
    return SimpleNamespace(js_solution=js_solution, get_interpolated_plan=lambda: interpolated)


@needs_cuda
def test_failure_reason_names_torque(real_world, monkeypatch):
    # "torque" when either trajectory of a failed trajopt passes an effort: the horizon's, or the interpolated one
    # between its frames that trajopt's success test re-checks.
    from cutamp.motion_solver import _failure_reason
    swing, still = _swing(real_world), _ramp(real_world, {}, 2 * DT)
    assert _failure_reason(real_world, None) == "other"
    assert _failure_reason(real_world, _result(None)) == "other"
    assert _failure_reason(real_world, _result(still, still)) == "other"
    assert _failure_reason(real_world, _result(swing, swing)) == "other"  # within the real efforts
    torques = real_world.segment_torques
    columns = [real_world.model.joint_names.index(j) for j in SHOULDERS]
    still_peak, swing_peak = _shoulder_peak(real_world, still), _shoulder_peak(real_world, swing)
    assert still_peak < swing_peak
    effort = torques.effort.clone()
    effort[columns] = 0.5 * (still_peak + swing_peak)
    monkeypatch.setattr(torques, "effort", effort)
    assert _failure_reason(real_world, _result(swing, still)) == "torque"
    assert _failure_reason(real_world, _result(still, swing)) == "torque"
    assert _failure_reason(real_world, _result(None, swing)) == "torque"
    assert _failure_reason(real_world, _result(still, still)) == "other"


def test_torque_check_fails_a_nan_torque():
    # A NaN torque fails the gate's torque check, and names its joint; torques within every effort pass.
    from cutamp.motion_solver import _check_torques
    n = len(REAL_LEGS.joint_names)

    class Torques:  # world.segment_torques's interface: [T, dof] torques and an effort per joint
        effort = torch.ones(n)

        def __init__(self, tau):
            self.tau = tau

        def __call__(self, js):
            return self.tau

    tau = torch.zeros(3, n)
    _check_torques(SimpleNamespace(model=REAL_LEGS, segment_torques=Torques(tau)), None)
    tau[1, 7] = math.nan
    with pytest.raises(RuntimeError, match=f"torque passes {REAL_LEGS.joint_names[7]}'s"):
        _check_torques(SimpleNamespace(model=REAL_LEGS, segment_torques=Torques(tau)), None)


def test_soles_planted_at_the_checker_tolerances():
    # One rule for a planted sole, the constraint checker's: an error at the tolerance passes, past it or NaN fails.
    # The kinematic constraint and the balance gate share the tolerances.
    from cutamp.com_polygon_cost import SOLE_POS_TOL, SOLE_ROT_TOL, soles_planted
    from cutamp.scripts.utils import default_constraint_to_tol
    from cutamp.task_planning.constraints import KinematicConstraint
    tol = default_constraint_to_tol[KinematicConstraint.type]
    assert (tol["sole_pos_err"], tol["sole_rot_err"]) == (SOLE_POS_TOL, SOLE_ROT_TOL) == (0.005, 0.05)
    pos = torch.tensor([SOLE_POS_TOL, 0.0, 1.01 * SOLE_POS_TOL, 0.0, math.nan, 0.0])
    rot = torch.tensor([SOLE_ROT_TOL, SOLE_ROT_TOL, 0.0, 1.01 * SOLE_ROT_TOL, 0.0, math.nan])
    assert soles_planted(pos, rot).tolist() == [True, True, False, False, False, False]
