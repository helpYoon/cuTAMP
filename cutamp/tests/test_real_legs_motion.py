"""Motion planning on the real legs with planted feet and cuRobo's torque check (design 7.4a, 7.5): the floor's push on
every trajopt transition, SegmentTorques against trajopt's own torques, the balance gate's multibody ZMP against a
float64 reference, its sole, tilt, CoM and torque checks, the failed-trajopt reason, and blocks_t1 end to end through
cutamp_demo, with its saved raw plan.

The slow end-to-end test comes first: cutamp_demo builds its own world, and the module's ``real_world`` and ``planner``
must not exist beside it (two worlds OOM a 24 GiB card). Module fixtures are built at their first use, so pytest's
file order keeps them apart."""

import dataclasses
import functools
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


def _ramp(world, moves, T, start=None):
    """The rest-to-rest min-jerk move ``moves`` ({joint: delta} from its start) over T s every DT s, each joint
    starting at home or at its ``start`` value ({joint: position}): a JointState in ``world.model.joint_names``
    order."""
    from curobo.types import JointState
    from cutamp.tests.test_balance_gate import _min_jerk
    names = list(world.model.joint_names)
    n = int(round(T / DT)) + 1
    _, p, v, a = (torch.as_tensor(x, dtype=torch.float32, device=world.device) for x in _min_jerk(T, n))
    pos = torch.tensor(world.model.home, device=world.device).repeat(n, 1)
    for name, value in (start or {}).items():
        pos[:, names.index(name)] = value
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
    # cutamp_demo end to end: setup_cutamp builds the world from REAL_LEGS, load_t1_container and
    # q_init = REAL_LEGS.home. The spies only record.
    import cutamp.algorithm as alg
    import cutamp.scripts.run_cutamp as rc
    from cutamp.com_polygon_cost import soles_planted
    from cutamp.config import TAMPConfiguration
    from cutamp.envs.utils import get_env_dir, load_env, raise_surface_height
    from cutamp.motion_solver import _full_positions, _model_frames
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
        speed = (_model_frames(world, None, entry["plan"]).velocity.abs() / torques.velocity)[:, _actuated(world)]
        assert float(speed.max()) <= 1.0, (k, float(speed.max()))  # the pace the gate set keeps every speed limit

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
    from cutamp.motion_solver import build_motion_planner
    with build_motion_planner(real_world) as mp:
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
    # The fork applies cspace.velocity_scale twice: t1_29dof.yml carries sqrt(7.29 / 12.57) on both
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
    # The trajopt CoM cost finds its support frame among the planner's tool poses.
    assert "stance" in planner.trajopt_solver.tool_frames
    assert all("stance" in tm.robot_model.tool_frames for tm in transitions)
    names = list(planner.trajopt_solver.joint_names)
    base = [names.index(j) for j in real_world.model.joint_names[real_world.model.base]]
    rest = _ramp(real_world, {}, 2 * DT)
    loads = [float(_transition_torques(tm, rest)[:, base].abs().max()) for tm in transitions]
    assert max(loads) <= 0.05, loads


@needs_cuda
def test_planner_registers_the_tilt_cost_only_under_a_bound(real_world, planner):
    # Route A's envelope sets no tilt bound, so its planner has no TrunkTiltCost; under an envelope with
    # one, trajopt bounds the tilt (weight 1e5, the bound's cosine, on the base's pitch and roll columns).
    from cutamp._curobo_internals import iter_rollouts
    from cutamp.joint_limit_cost import TrunkTiltCost
    from cutamp.robots.t1 import get_t1_motion_planner
    from cutamp.utils.common import get_world_cfg

    def extra_costs(mp, kind):
        return [c for r in iter_rollouts(mp) for mgr in (r.cost_manager, r.metrics_cost_manager)
                for c in getattr(mgr, "_extra_costs", {}).values() if isinstance(c, kind)]

    assert REAL_LEGS.envelope.trunk_tilt_max is None
    assert not extra_costs(planner, TrunkTiltCost)
    bounded = dataclasses.replace(REAL_LEGS, envelope=REAL_LEGS.envelope._replace(trunk_tilt_max=1.0))
    with get_t1_motion_planner(bounded, get_world_cfg(real_world.env, include_movables=True),
                               use_cuda_graph=False, device_cfg=real_world.device_cfg) as mp:
        costs = extra_costs(mp, TrunkTiltCost)
        names = list(mp.kinematics.joint_names)
        assert costs
        columns = (names.index("base_j_pitch"), names.index("base_j_roll"))
        for c in costs:
            assert (c.config.pitch_index, c.config.roll_index) == columns
            assert c.config.cos_max == math.cos(1.0) and float(c._weight.max()) == 1e5


def _plan_reach(world, real_plan, planner):
    """A planned home -> reach segment: plan_cspace's result for the first pick conf of ``real_plan`` that plans."""
    from cutamp.motion_solver import _plan_arm_to_conf, _planner_js_from_full
    from cutamp.tests.test_real_legs_gn import _accepted
    skeleton, particles = real_plan
    op = next(op for op in skeleton if op.operator.metadata.action_type == "pick")
    obj, conf = op.values[0], op.values[-1]
    names = list(world.model.joint_names)
    home = torch.tensor(world.model.home, device=world.device)
    start = _planner_js_from_full(planner, home, names)
    for q in particles[conf][_accepted(world, real_plan, conf)][:8]:  # GN IK's accepted reaches; a few may not plan
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
    # Trajopt bounds the torque at its own timing only; the gate runs the two passes (design 7.4a) at every time scale
    # it tries and on the frames it returns, and fails past an effort. The slowed swing's shoulders carry less than
    # the planned one's: an effort between the two peaks passes only if the gate checks the torque at the scale it
    # picks, not at the planner's; one under the shoulders' static load fails at every scale.
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
    # Every CoM and ZMP margin the real-leg gate and audit take is against the gate's rectangle about the stance.
    halves, margin = [], cpc.com_margin_from_base

    def spy(xy, half):
        halves.append(tuple(half))
        return margin(xy, half)

    monkeypatch.setattr(cpc, "com_margin_from_base", spy)
    still = _ramp(real_world, {}, 1.0)
    out, stats, fast = _balance_gate(real_world, planner, still, DT)
    # Nothing moves, so no check binds: the gate keeps one step, the shortest segment _retime makes.
    assert out.position.shape[0] == 2 and stats["retime"] == 1 / 40 and stats["binding"] == "none"
    assert fast is not None and fast.position.shape == still.position.shape
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


def _actuated(world) -> list:
    """The model's actuated joints' columns: every joint but the base's six virtual ones."""
    base = set(range(len(world.model.joint_names))[world.model.base])
    return [j for j in range(len(world.model.joint_names)) if j not in base]


def _peak_shares(world, js):
    """``(speed, torque)`` ``[actuated]``: each actuated joint's peak |velocity| over cuRobo's limit and peak |torque|
    (``world.segment_torques``) over its effort, over the frames of ``js`` (the model's joint order)."""
    torques, act = world.segment_torques, _actuated(world)
    speed = js.velocity.reshape(-1, js.velocity.shape[-1])[:, act].abs().amax(0) / torques.velocity[act]
    return speed, torques(js).abs().amax(0)[act] / torques.effort[act]


def _elbow_sweep(world):
    """The left elbow pitch from -1.87 to +1.87 rad, min-jerk over 2 s (81 frames): its torques allow 0.13x and its ZMP
    0.17x, its peak speed only 0.1993x of cuRobo's 17.59 rad/s."""
    return _ramp(world, {"Left_Elbow_Pitch": 3.74}, 2.0, start={"Left_Elbow_Pitch": -1.87})


@needs_cuda
def test_uniform_time_scaling_matches_the_resampled_frames(real_world, real_plan, planner):
    # Under a uniform time scale s a frame's velocities scale by 1/s and its accelerations by 1/s^2, so the gate tries
    # each s on the planner's own frames (_at_scales). On a planned reach at s = 0.5 and 2.0, cuRobo's torques with the
    # floor's push on those frames are tau_0 + (tau_1 - tau_0) / s^2 to 1e-3 N m (float32), tau_0 at rest and tau_1
    # at the plan's own timing (what _torque_scale solves exactly), and they match the torques on _retime(plan, s)'s
    # frames at the same instants to 0.02 N m: the gap between _retime's spline derivatives and trajopt's, measured
    # at 0.003 N m.
    from curobo.types import JointState
    from cutamp.motion_solver import _at_scales, _interp_plan, _model_frames, _retime
    plan = _model_frames(real_world, planner, _interp_plan(_plan_reach(real_world, real_plan, planner)))
    n = plan.position.shape[0] - (plan.position.shape[0] - 1) % 2  # odd, so s = 0.5 lands on every other frame
    q, v, a = plan.position[:n], plan.velocity[:n], plan.acceleration[:n]
    names = plan.joint_names
    plan = JointState(position=q, velocity=v, acceleration=a, joint_names=names)
    torques = real_world.segment_torques
    tau0 = torques(JointState(position=q, velocity=torch.zeros_like(v), acceleration=torch.zeros_like(a),
                              joint_names=names))
    tau1 = torques(plan)
    assert float((tau1 - tau0)[:, real_world.model.body].abs().max()) > 0.05  # the reach's dynamics show
    for s in (0.5, 2.0):
        scaled = torques(_at_scales(plan, torch.tensor([s])))
        torch.testing.assert_close(scaled, tau0 + (tau1 - tau0) / s**2, atol=1e-3, rtol=0)
        out = _retime(plan, s, DT)
        assert out.position.shape[0] == round(s * (n - 1)) + 1  # no rounding up: _retime's frames are at s itself
        k = torch.arange(0, n, 2 if s < 1 else 1, device=q.device)  # the planner's frames at _retime's instants
        idx = (k * s).round().long()
        torch.testing.assert_close(out.position[idx], q[k], atol=1e-6, rtol=0)
        resampled = torques(JointState(position=out.position[idx], velocity=out.velocity[idx],
                                       acceleration=out.acceleration[idx], joint_names=names))
        torch.testing.assert_close(scaled[k], resampled, atol=0.02, rtol=0)


@needs_cuda
def test_gate_paces_a_velocity_bound_segment_at_its_limit(real_world, planner):
    # Nothing but its speed limits the elbow sweep: the gate speeds it up 5x (16 steps of DT for its 0.1993 x 80 =
    # 15.95), its fastest actuated joint ends at its limit to the search tolerance, and the binding check names it.
    from cutamp.motion_solver import RETIME_TOL, _balance_gate
    names = real_world.model.joint_names
    out, stats, fast = _balance_gate(real_world, planner, _elbow_sweep(real_world), DT)
    assert stats["binding"] == "velocity Left_Elbow_Pitch"
    assert stats["retime"] == 0.2 and out.position.shape[0] == 17
    assert fast is not None and fast.position.shape[0] == 81  # the planner's timing, kept: s != 1
    speed, effort = _peak_shares(real_world, out)
    assert names[_actuated(real_world)[int(speed.argmax())]] == "Left_Elbow_Pitch"
    assert 1 / (1 + RETIME_TOL) <= float(speed.max()) <= 1.0
    assert float(effort.max()) <= 1.0 and stats["zmp_min_margin"] >= 0


@needs_cuda
def test_gate_keeps_a_segments_ends(real_world, planner):
    # The gate resamples a segment at its own pace (_retime's spline), and the segment still starts and ends on the
    # planner's frames bit for bit: the next segment starts from its last frame, the anchored conf. The spline's own
    # end value misses a joint that moves to 0 (here the left elbow, -1 -> 0 rad) by a rounding error, and the next
    # trajopt would start there.
    from cutamp.motion_solver import _balance_gate
    plan = _ramp(real_world, {"Left_Elbow_Pitch": 1.0}, 2.0, start={"Left_Elbow_Pitch": -1.0})
    assert float(plan.position[-1, real_world.model.joint_names.index("Left_Elbow_Pitch")]) == 0.0
    out, stats, _ = _balance_gate(real_world, planner, plan, DT)
    assert stats["retime"] != 1.0
    assert torch.equal(out.position[[0, -1]], plan.position[[0, -1]])


@needs_cuda
def test_gate_reports_the_torque_or_the_zmp_that_binds(real_world, planner):
    # Two min-jerk moves of both shoulder pitches over 2 s. Swinging opposite ways (0.7 -> -0.7 and -0.7 -> 0.7 rad),
    # the arms cancel each other's ZMP shift, so a shoulder's torque binds first (0.112x, the ZMP 0.104x); both
    # forward to -2.0 rad, the ZMP binds (0.64x, the torque 0.20x). Each ends at its bound: the joint near its
    # effort, and the ZMP inside but within the search tolerance of leaving.
    from cutamp.motion_solver import RETIME_TOL, _balance_gate, _model_frames, _zmp_margins, _zmp_scale
    names = real_world.model.joint_names
    opposite = _ramp(real_world, {"Left_Shoulder_Pitch": -1.4, "Right_Shoulder_Pitch": 1.4}, 2.0,
                     start={"Left_Shoulder_Pitch": 0.7, "Right_Shoulder_Pitch": -0.7})
    out, stats, _ = _balance_gate(real_world, planner, opposite, DT)
    check, joint = stats["binding"].split()
    assert check == "torque" and joint in SHOULDERS and stats["retime"] < 0.2
    _, effort = _peak_shares(real_world, out)
    assert 0.9 <= float(effort[_actuated(real_world).index(names.index(joint))]) <= 1.0
    forward = _ramp(real_world, {j: -2.2 for j in SHOULDERS}, 2.0)
    out, stats, _ = _balance_gate(real_world, planner, forward, DT)
    assert stats["binding"] == "zmp" and 0.5 < stats["retime"] < 1.0 and 0.0 <= stats["zmp_min_margin"] < 0.01
    # The search's bound: the ZMP passes at its scale and fails RETIME_TOL under it.
    js = _model_frames(real_world, planner, forward)
    s, binds = _zmp_scale(real_world, planner, js, 0.3, 4.0)
    margins = _zmp_margins(real_world, planner, js, torch.tensor([s, s / (1 + RETIME_TOL)]))
    assert binds and float(margins[0]) >= 0 > float(margins[1])


@needs_cuda
def test_gate_still_slows_a_segment_that_needs_it(real_world, planner):
    # The 0.5 s shoulder swing's ZMP needs a 2.6x slowdown: the gate slows it, keeps the planner's timing, and every
    # frame it returns passes: the ZMP inside, every torque within its effort, every speed within its limit.
    from cutamp.motion_solver import _balance_gate
    swing = _swing(real_world)
    out, stats, fast = _balance_gate(real_world, planner, swing, DT)
    n = swing.position.shape[0]
    assert stats["binding"] == "zmp" and stats["retime"] > 2.0 and stats["zmp_min_margin"] >= 0
    assert out.position.shape[0] == round(stats["retime"] * (n - 1)) + 1
    assert fast is not None and fast.position.shape == swing.position.shape
    speed, effort = _peak_shares(real_world, out)
    assert float(speed.max()) <= 1.0 and float(effort.max()) <= 1.0


@needs_cuda
def test_gate_raises_naming_the_check_no_scale_passes(real_world, planner, monkeypatch):
    # With no time scale up to RETIME_MAX passing, the gate raises and names the check: in 0.3 s the shoulder swing's
    # ZMP needs 4.6x; an elbow effort under the elbow's static load fails at every scale; a speed limit 1/100 of the
    # elbow sweep's peak needs 100x.
    from cutamp.motion_solver import RETIME_MAX, _balance_gate
    with pytest.raises(RuntimeError, match=f"ZMP stays outside the support region at the {RETIME_MAX}x") as e:
        _balance_gate(real_world, planner, _ramp(real_world, {j: -2.2 for j in SHOULDERS}, 0.3), DT)
    assert e.value.check == "zmp"
    torques, sweep = real_world.segment_torques, _elbow_sweep(real_world)
    j = real_world.model.joint_names.index("Left_Elbow_Pitch")
    peak = float(sweep.velocity[:, j].abs().max())
    for attr, value, message in (("effort", 1e-3, "torque passes Left_Elbow_Pitch's"),
                                 ("velocity", peak / 100, "velocity passes Left_Elbow_Pitch's")):
        limits = getattr(torques, attr).clone()
        limits[j] = value
        with monkeypatch.context() as m:
            m.setattr(torques, attr, limits)
            with pytest.raises(RuntimeError, match=f"{message} .* at the {RETIME_MAX}x slowdown cap") as e:
                _balance_gate(real_world, planner, sweep, DT)
        assert e.value.check == f"{message.split()[0]} Left_Elbow_Pitch"


@needs_cuda
def test_gate_slows_a_resampled_segment_that_fails_its_recheck(real_world, planner, monkeypatch):
    # The frames _retime returns are spline samples, so a speed, torque or ZMP check can fail on them where it passed
    # on the planner's frames: the gate slows that segment by RETIME_RECHECK_STEP until its re-check passes (the
    # elbow sweep: 16 steps, then 17 for 1.05 x 15.95). The soles, the tilt and the CoM follow the path, which no time
    # scale changes, so their re-check failure raises at once.
    import cutamp.motion_solver as ms
    calls, recheck = [], ms._recheck

    def failing_once(check):
        def spy(world, planner, js, planted=True):
            calls.append(js.position.shape[-2])
            if len(calls) == 1:
                raise ms._gate_error(check, f"{check} fails")
            return recheck(world, planner, js, planted)
        return spy

    monkeypatch.setattr(ms, "_recheck", failing_once("zmp"))
    out, stats, _ = ms._balance_gate(real_world, planner, _elbow_sweep(real_world), DT)
    assert calls == [17, 18] and stats["binding"] == "zmp" and stats["retime"] == 17 / 80
    calls.clear()
    monkeypatch.setattr(ms, "_recheck", failing_once("soles"))
    with pytest.raises(RuntimeError, match="soles fails"):
        ms._balance_gate(real_world, planner, _elbow_sweep(real_world), DT)
    assert calls == [17]


def _on_the_stance(world, moves, T, eps=0.01):
    """The rest-to-rest min-jerk move ``moves`` ({joint: delta} from home) over T s every DT s, the base re-solved at
    every frame so that both soles stay on the stance (particle_initialization._stand_on_the_stance, which puts the
    left sole there: the legs must move alike for the right one to follow), its velocity and acceleration by central
    differences over eps s."""
    from curobo.types import JointState
    from cutamp.particle_initialization import _stand_on_the_stance
    names = list(world.model.joint_names)
    home = torch.tensor(world.model.home, device=world.device, dtype=torch.float64)
    lower, upper = world.kinematics.get_joint_limits().position

    def ramp(t):
        u = (t / T).clamp(0.0, 1.0)
        p, v, a = (10 * u**3 - 15 * u**4 + 6 * u**5, (30 * u**2 - 60 * u**3 + 30 * u**4) / T,
                   (60 * u - 180 * u**2 + 120 * u**3) / T**2)
        q = home.repeat(len(t), 1)
        vel, acc = torch.zeros_like(q), torch.zeros_like(q)
        for name, delta in moves.items():
            j = names.index(name)
            q[:, j] += delta * p
            vel[:, j], acc[:, j] = delta * v, delta * a
        return q, vel, acc

    n = int(round(T / DT)) + 1
    t = torch.linspace(0.0, T, n, dtype=torch.float64, device=world.device)
    q, v, a = ramp(t)
    stood = _stand_on_the_stance(world, torch.cat([ramp(t - eps)[0], q, ramp(t + eps)[0]]).float(), lower, upper)
    before, q, after = stood.double().split(n)
    b = world.model.base
    v[:, b], a[:, b] = (after - before)[:, b] / (2 * eps), (after - 2 * q + before)[:, b] / eps**2
    return JointState(position=q.float(), velocity=v.float(), acceleration=a.float(), joint_names=names)


def _squat(world, theta=0.4, T=0.5, eps=0.01):
    """Both legs bending alike from home, min-jerk over T s every DT s (each hip and ankle pitch by -theta, each knee
    by 2 theta), on the stance (``_on_the_stance``): the Trunk drops 7.5 cm."""
    bends = (("Hip_Pitch", -theta), ("Knee_Pitch", 2 * theta), ("Ankle_Pitch", -theta))
    return _on_the_stance(world, {f"{side}_{joint}": delta for side in ("Left", "Right") for joint, delta in bends},
                          T, eps)


def _lean(world, pitch, T=2.0):
    """Both ankle pitches tipping the robot forward by ``pitch`` rad from home, min-jerk over T s, on the stance
    (``_on_the_stance``): the whole robot leans about its ankles, the CoM moving forward about 0.53 m per rad from
    home's 1.2 cm ahead of the mid-sole."""
    return _on_the_stance(world, {f"{side}_Ankle_Pitch": -pitch for side in ("Left", "Right")}, T)


@needs_cuda
def test_gate_finds_no_zmp_where_the_floor_would_have_to_pull(real_world, planner):
    # The floor can only push. Sped up past about 0.38x, the squat's CoM starts falling faster than g: there
    # sum m (g + z'') is negative, the soles would leave the floor, and there is no ZMP. com_zmp_in_support gives
    # those frames +inf, so the search's margins are -inf at those scales. The torque bound (0.28x) lies below that
    # crossing, and the gate still times the squat above it, where the ZMP binds (0.65x). Frames resampled at 0.3x
    # fail their re-check on "zmp", which says why.
    from cutamp.com_polygon_cost import _fk_com_and_base, com_zmp_in_support
    from cutamp.motion_solver import (RETIME_MAX, ZMP_STEP, _at_scales, _balance_gate, _check_planted, _check_velocity,
                                      _recheck, _retime, _taylor_positions, _torque_scale, _zmp_margins)
    world, squat = real_world, _squat(real_world)
    _check_planted(world, squat.position)

    def loads(js, scales):
        """``[K, T]``: each frame's sum m (g + z'') over M g under each scale, from the CoM's height (cuRobo's CoM)
        at the Taylor positions com_zmp_in_support takes."""
        q3 = _taylor_positions(world, planner, _at_scales(js, scales))
        z = torch.stack([_fk_com_and_base(world, q)[0][:, 2].clone() for q in q3])  # cuRobo reuses its FK buffers
        return (1 + (z[2] - 2 * z[1] + z[0]) / ZMP_STEP**2 / 9.81).reshape(len(scales), -1)

    fast = torch.tensor([0.1, 0.2, 0.3])
    load = loads(squat, fast)
    assert float(load.amin(-1).max()) < -0.5  # at each of these scales some frame would need the floor to pull
    _, zmp = com_zmp_in_support(world, _taylor_positions(world, planner, _at_scales(squat, fast)), ZMP_STEP)
    zmp = zmp.reshape(len(fast), -1, 2)
    assert bool(torch.isposinf(zmp[load < -0.05]).all()) and bool(torch.isfinite(zmp[load > 0.05]).all())
    assert bool((_zmp_margins(world, planner, squat, fast) == -math.inf).all())
    grid = torch.linspace(0.3, 0.5, 201)
    crossing = float(grid[(loads(squat, grid).amin(-1) > 0).cpu()][0])  # the fastest at which the floor always pushes
    s_t, joint = _torque_scale(world, squat, _check_velocity(world, squat, RETIME_MAX)[0], RETIME_MAX)
    assert joint is not None and s_t < crossing  # where the ZMP's search starts
    out, stats, _ = _balance_gate(world, planner, squat, DT)
    assert stats["binding"] == "zmp" and stats["retime"] > crossing and stats["zmp_min_margin"] >= 0
    assert float(loads(squat, torch.tensor([stats["retime"]])).min()) > 0 and float(loads(out, torch.ones(1)).min()) > 0
    with pytest.raises(RuntimeError, match="floor would have to pull on the retimed trajectory's soles") as e:
        _recheck(world, planner, _retime(squat, 0.3, DT))
    assert e.value.check == "zmp"


@needs_cuda
def test_gate_reports_the_smaller_margin_of_its_frames_and_the_planners(real_world, planner):
    # A sped-up segment returns fewer frames than the planner's, which the search ran on, and they can miss its
    # tightest instant: the opposite shoulder swings, sped up to 0.11x by a shoulder's torque, return 10 frames, whose
    # ZMP margin is 2.00 cm, against 1.62 cm on the planner's 81 at the same scale. The stats report the smaller, and
    # the CoM's margin likewise.
    from cutamp.com_polygon_cost import com_margin_from_base, com_zmp_in_support
    from cutamp.motion_solver import ZMP_STEP, _balance_gate, _model_frames, _recheck, _taylor_positions, _zmp_margins
    world = real_world
    opposite = _ramp(world, {"Left_Shoulder_Pitch": -1.4, "Right_Shoulder_Pitch": 1.4}, 2.0,
                     start={"Left_Shoulder_Pitch": 0.7, "Right_Shoulder_Pitch": -0.7})
    out, stats, _ = _balance_gate(world, planner, opposite, DT)
    s, n_out = stats["retime"], out.position.shape[0]
    assert s < 0.2 and n_out == round(s * (opposite.position.shape[0] - 1)) + 1
    com_out, zmp_out = _recheck(world, planner, out)
    js = _model_frames(world, planner, opposite)
    zmp_planner = float(_zmp_margins(world, planner, js, torch.tensor([s]))[0])
    assert 0 <= zmp_planner < zmp_out - 0.002
    assert stats["zmp_min_margin"] == zmp_planner
    com, _ = com_zmp_in_support(world, _taylor_positions(world, planner, js), ZMP_STEP)
    com_planner = com_margin_from_base(com, world.model.support.gate_half).min().item()
    assert stats["com_min_margin"] == min(com_out, com_planner)


@functools.lru_cache(maxsize=1)
def _planner_urdf():
    """t1_planner.urdf as yourdfpy parses it, float64: the Trunk at its root, each link's inertial and each joint as the
    URDF defines them. No scene graph: yourdfpy 0.0.58 builds it through a NumPy conversion deprecated since 1.25 (one
    DeprecationWarning per revolute joint, an error in a later NumPy), so _reference_zmp chains the joints itself."""
    import yourdfpy
    from cutamp.robots.t1 import T1_29DOF_DIR
    return yourdfpy.URDF.load(str(T1_29DOF_DIR / "t1_planner.urdf"), load_meshes=False, build_scene_graph=False,
                              build_collision_scene_graph=False, load_collision_meshes=False)


def _reference_zmp(q_of_t, t, h=1e-4, g=9.81):
    """``(zmp_xy, com_xy)`` at time ``t`` of the trajectory ``q_of_t`` (float64 confs in ``REAL_LEGS.joint_names``
    order), about the home mid-sole on flat ground, from first principles: float64 FK on t1_planner.urdf's joints as
    yourdfpy parses them (each child at its parent times the joint's origin and its rotation about its axis, yourdfpy's
    own product), the Trunk placed by trunk_pose_from_q and the joints the planner locks (head, fingers) at 0; every
    link's CoM acceleration by central differences of the true trajectory, its spin L = R I R^T w (w from R' R^T)
    differentiated the same way; sum m (g + z'') x - m z x'' - L'_y over the links (y: + L'_x), over sum m (g + z''),
    with z the height above the soles' plane (world z = 0). The mid-sole is the soles' midpoint at home: each foot
    link's pose times t1_29dof.yml's fixed sole offset."""
    import yaml
    from trimesh.transformations import rotation_matrix

    from cutamp.robots.t1 import T1_29DOF_DIR, trunk_pose_from_q
    u = _planner_urdf()
    joint_to = {j.child: j for j in u.robot.joints}
    assert {j.type for j in u.robot.joints} == {"fixed", "revolute"} and not any(j.mimic for j in u.robot.joints)
    (root,) = {link.name for link in u.robot.links} - set(joint_to)  # "world", whose fixed joint carries the Trunk
    links = [(n, link.inertial) for n, link in u.link_map.items() if link.inertial is not None and link.inertial.mass]
    m = np.array([i.mass for _, i in links])
    com = np.stack([i.origin[:3, 3] for _, i in links])                                     # in the link frame
    inertia = np.stack([i.origin[:3, :3] @ i.inertia @ i.origin[:3, :3].T for _, i in links])  # about it, too

    def poses(q, frames):
        value = dict(zip(REAL_LEGS.joint_names, q.tolist()))
        X = {root: trunk_pose_from_q(REAL_LEGS, q).numpy()}  # the root-Trunk joint is the identity

        def pose(link):
            if link not in X:
                j = joint_to[link]
                motion = rotation_matrix(value.get(j.name, 0.0), j.axis) if j.type == "revolute" else np.eye(4)
                X[link] = pose(j.parent) @ (np.eye(4) if j.origin is None else j.origin) @ motion
            return X[link]

        return np.stack([pose(f) for f in frames])

    extra = yaml.safe_load((T1_29DOF_DIR / "t1_29dof.yml").read_text())["robot_cfg"]["kinematics"]["extra_links"]
    feet = poses(np.array(REAL_LEGS.home), ("left_foot_link", "right_foot_link"))
    soles = [F[:3, :3] @ extra[f"{side}_sole_link"]["fixed_transform"][:3] + F[:3, 3]
             for F, side in zip(feet, ("left", "right"))]
    mid_sole = np.mean(soles, 0)[:2]

    def state(tau):
        X = poses(q_of_t(t + tau), [n for n, _ in links])
        return np.einsum("lij,lj->li", X[:, :3, :3], com) + X[:, :3, 3], X[:, :3, :3]

    st = {k: state(k * h) for k in (-2, -1, 0, 1, 2)}
    x = st[0][0]
    a = (st[1][0] - 2 * x + st[-1][0]) / h**2

    def spin(k):
        W = (st[k + 1][1] - st[k - 1][1]) / (2 * h) @ st[k][1].transpose(0, 2, 1)
        w = 0.5 * np.stack([W[:, 2, 1] - W[:, 1, 2], W[:, 0, 2] - W[:, 2, 0], W[:, 1, 0] - W[:, 0, 1]], -1)
        return np.einsum("lij,ljk,lmk,lm->li", st[k][1], inertia, st[k][1], w).sum(0)

    dL = (spin(1) - spin(-1)) / (2 * h)
    f = m * (g + a[:, 2])
    zmp = np.array([(f * x[:, 0] - m * x[:, 2] * a[:, 0]).sum() - dL[1],
                    (f * x[:, 1] - m * x[:, 2] * a[:, 1]).sum() + dL[0]]) / f.sum()
    return zmp - mid_sole, (m[:, None] * x).sum(0)[:2] / m.sum() - mid_sole


@needs_cuda
def test_real_legs_zmp_matches_multibody_reference(real_world, planner):
    # Home to a leaning conf in 0.6 s, min-jerk in every joint, the base's six with them: the hips bow 0.5 rad, the
    # knees bend 0.3 and the ankles 0.15, the Waist turns 0.4, the left shoulder swings 1 rad forward and the right
    # elbow bends 0.8, the base re-solved so that the end conf stands on the stance (its Trunk pitched 0.35 rad and
    # yawed -0.4). Every link moves and spins. The gate's CoM and multibody ZMP, from the plan's own velocities and
    # accelerations as the gate takes them, match the float64 first-principles ones at every frame, at rest and at the
    # acceleration peaks, where the ZMP runs 9 cm from the CoM. Measured 2026-11-05: the CoM within 1.3e-8 m and the
    # ZMP within 2.7e-5 m of the reference, against tolerances of 2e-4 and 3e-4 m.
    from cutamp.com_polygon_cost import com_zmp_in_support
    from cutamp.motion_solver import ZMP_STEP, _model_frames, _taylor_positions
    from cutamp.particle_initialization import _stand_on_the_stance
    world, names, T = real_world, list(real_world.model.joint_names), 0.6
    home = torch.tensor(world.model.home, dtype=torch.float64)
    bent = home.clone()
    for joint, delta in (("Left_Hip_Pitch", -0.5), ("Right_Hip_Pitch", -0.5), ("Left_Knee_Pitch", 0.3),
                         ("Right_Knee_Pitch", 0.3), ("Left_Ankle_Pitch", -0.15), ("Right_Ankle_Pitch", -0.15),
                         ("Waist", 0.4), ("Left_Shoulder_Pitch", -1.0), ("Right_Elbow_Pitch", -0.8)):
        bent[names.index(joint)] += delta
    lo, hi = world.kinematics.get_joint_limits().position
    end = _stand_on_the_stance(world, bent.view(1, -1).to(world.device, torch.float32), lo, hi)[0].double().cpu()
    assert float(end[names.index("base_j_pitch")]) == pytest.approx(0.35, abs=1e-4)  # it stands: no fallback to home
    delta = end - home
    plan = _ramp(world, {j: float(d) for j, d in zip(names, delta) if d != 0}, T)
    com, zmp = com_zmp_in_support(world, _taylor_positions(world, planner, _model_frames(world, planner, plan)),
                                  ZMP_STEP)
    assert float((zmp - com).abs().max()) > 0.05  # the motion moves the ZMP: not a static match

    def q_of_t(t):
        u = min(max(t / T, 0.0), 1.0)
        return (home + (10 * u**3 - 15 * u**4 + 6 * u**5) * delta).numpy()

    for k in range(plan.position.shape[0]):
        ref_zmp, ref_com = _reference_zmp(q_of_t, k * DT)
        np.testing.assert_allclose(com[k].cpu().numpy(), ref_com, atol=2e-4, err_msg=f"frame {k}")
        np.testing.assert_allclose(zmp[k].cpu().numpy(), ref_zmp, atol=3e-4, err_msg=f"frame {k}")


@needs_cuda
@pytest.mark.parametrize("site", ["recheck", "planner"])
def test_real_legs_gate_rejects_com_outside_support(real_world, planner, site):
    # The CoM follows the path, which no time scale changes: the gate fails a segment whose CoM leaves its rectangle,
    # with check "com", on the planner's own frames and on the frames it returns (_recheck). Leaning 0.25 rad
    # over 2 s, the CoM ends 14.4 cm ahead of the mid-sole, 3.2 cm past the toes (the soles' front edge is at
    # 11.245 cm, the gate's at 9.245); leaning 0.12 rad it ends at 7.6 cm, 1.6 cm inside the gate's edge, and passes.
    import cutamp.motion_solver as ms
    from cutamp.com_polygon_cost import COM_SAFETY_MARGIN, com_zmp_in_support
    world = real_world
    gate = {"recheck": lambda js: ms._recheck(world, planner, js)[0],
            "planner": lambda js: ms._balance_gate(world, planner, js, DT)[1]["com_min_margin"]}[site]
    raised = {"recheck": "^retimed trajectory CoM leaves", "planner": "^trajectory CoM leaves"}[site]
    out = _lean(world, 0.25)
    com, _ = com_zmp_in_support(world, ms._taylor_positions(world, planner, out), ms.ZMP_STEP)
    assert float(com[-1, 0]) > world.model.support.gate_half[0] + COM_SAFETY_MARGIN  # past the soles' front edge
    with pytest.raises(RuntimeError, match=raised) as e:
        gate(out)
    assert e.value.check == "com"
    assert gate(_lean(world, 0.12)) >= 0.0


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


def test_plan_failed_names_torque_in_every_torque_form():
    # A failed candidate's message names torque in trajopt's reason (": torque") and in each of the balance gate's
    # torque raises: no time scale up to the cap keeps every effort (_torque_scale), or the frames it returns pass one
    # (_check_torques). Trajopt's ": other" and the gate's velocity and ZMP raises don't. The last assert pins which of
    # the six messages name torque.
    from cutamp.motion_solver import RETIME_MAX, _check_torques, _check_velocity, _plan_failed, _torque_scale

    n = len(REAL_LEGS.joint_names)

    class Torques:  # world.segment_torques's interface: [T, dof] torques, and an effort and a speed limit per joint
        def __init__(self, tau):
            self.tau, self.effort, self.velocity = tau, torch.ones(n), torch.ones(n)

        def __call__(self, js):
            return torch.full((getattr(js, "position", torch.zeros(3, n)).shape[0], n), self.tau)

    def world(tau):
        return SimpleNamespace(model=REAL_LEGS, segment_torques=Torques(tau))

    rest = torch.zeros(3, n)
    js = SimpleNamespace(position=rest, velocity=rest + 2 * RETIME_MAX, acceleration=rest,
                         joint_names=list(REAL_LEGS.joint_names))  # twice the speed limit at the slowdown cap
    raised = []
    for check, args in ((_check_torques, (world(2.0), None)), (_torque_scale, (world(2.0), js, 0.5, RETIME_MAX)),
                        (_check_velocity, (world(0.5), js, RETIME_MAX))):
        with pytest.raises(RuntimeError) as e:
            check(*args)
        raised.append(str(e.value))
    result = SimpleNamespace(js_solution=object(), get_interpolated_plan=lambda: None)  # a trajopt result
    messages = [str(_plan_failed(world(2.0), result, "Retract plan failed for op")), *raised,
                str(_plan_failed(world(0.5), result, "Retract plan failed for op")),
                f"trajectory ZMP stays outside the support region at the {RETIME_MAX}x slowdown cap"]
    assert messages[0].endswith(": torque") and messages[4].endswith(": other")
    assert f"at the {RETIME_MAX}x slowdown cap" in messages[2] and "velocity passes" in messages[3]
    assert [("torque" in m) for m in messages] == [True, True, True, False, False, False]


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
