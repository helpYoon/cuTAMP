"""State conversion and checked handoffs for the T1 trained landing controller."""

from pathlib import Path

import pytest
import torch

from cutamp.robots.t1 import REAL_LEGS
from cutamp.step_validation import StepStateAdapter
from cutamp.tests.conftest import needs_cuda


def test_named_state_and_noncommuting_angular_velocity():
    from types import SimpleNamespace

    meta = {"joint_names": ["AAHead_yaw", "Head_pitch", *reversed(REAL_LEGS.joint_names[6:])]}
    adapter = StepStateAdapter(SimpleNamespace(metadata=meta))
    q = torch.tensor(REAL_LEGS.home).double().repeat(3, 1)
    q[1:, :6] += torch.tensor([0.03, -0.04, 0.01, 0.19, -0.33, 0.52])
    q[1, 10] += 0.1
    q[2, 16] -= 0.12
    v = torch.linspace(-0.06, 0.09, 33).double().expand_as(q)
    physical = adapter.physical(q, v)
    restored, restored_v = adapter.planner(physical)
    torch.testing.assert_close(restored, q)
    torch.testing.assert_close(restored_v, v)
    assert physical[4].shape == (3, 29) and torch.count_nonzero(physical[4][..., :2]) == 0
    eps = 1e-6
    rot = physical[1]
    drdt = (adapter.physical(q + eps * v)[1] - adapter.physical(q - eps * v)[1]) / (2 * eps)
    skew = drdt @ rot.transpose(-1, -2)
    w_fd = skew[:, [2, 0, 1], [1, 2, 0]]
    torch.testing.assert_close(w_fd, physical[3])


@needs_cuda
def test_source_sole_frames_and_observed_handoffs():
    from curobo.types import DeviceCfg, JointState

    from cutamp.landing_predictor import LandingPredictor
    from cutamp.robots.t1 import get_t1_kinematics

    bundle = Path(__file__).resolve().parents[2] / "data/t1-reachability/landing-accuracy/landing-seed3.pt"
    pred = LandingPredictor(bundle)
    adapter = StepStateAdapter(pred)
    q = torch.tensor(REAL_LEGS.home, device="cuda").repeat(3, 1)
    q[1:, :6] += q.new_tensor([0.07, 0.04, -0.02, 0.11, 0.09, 0.19])
    q[1, 6] += 0.23
    q[2, 21] -= 0.10
    fk = get_t1_kinematics(REAL_LEGS, DeviceCfg()).compute_kinematics(JointState.from_position(q))
    T = torch.stack(
        [
            fk.tool_poses.get_link_pose(f, make_contiguous=True).get_matrix().reshape(-1, 4, 4)
            for f in REAL_LEGS.sole_frames
        ],
        dim=1,
    )
    sole = torch.stack((T[..., 0, 3], T[..., 1, 3], torch.atan2(T[..., 1, 0], T[..., 0, 0])), -1)
    torch.testing.assert_close(pred.model.feet(adapter.physical(q))[0], sole, atol=0.00006, rtol=0)
    ctx = q.new_zeros(3, 38)
    state = pred.standing_start(ctx)
    start_q, start_qd = adapter.planner(state.tensors[:6])
    same = adapter.initialize(start_q, ctx, start_qd, observed=state)
    assert same.state is state and same.compatible.all()
    wrong = start_q.clone()
    wrong[1, 0] += 0.1
    wrong[2, 9] += 0.2
    assert adapter.initialize(wrong, ctx, start_qd, observed=state).compatible.tolist() == [True, False, False]
    drifting = start_qd.clone()
    drifting[0, 0] += 0.25
    assert not adapter.initialize(start_q, ctx, drifting, observed=state).compatible[0]


@pytest.fixture(scope="module")
def rollout_scene():
    from curobo.types import DeviceCfg

    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.landing_predictor import LandingPredictor
    from cutamp.robots import load_t1_container
    from cutamp.tamp_world import TAMPWorld

    root = Path(__file__).resolve().parents[2]
    predictor = LandingPredictor(root / "data/t1-reachability/landing-accuracy/landing-seed3.pt")
    dc = DeviceCfg()
    q = dc.to_device(REAL_LEGS.home)
    world = TAMPWorld(
        load_env(str(Path(get_env_dir()) / "blocks_t1.yml")), dc, load_t1_container(REAL_LEGS, dc), q, grasps={}
    )
    adapter = StepStateAdapter(predictor)
    context = q.new_zeros(1, 38)
    context[:, 0] = 0.25 / 0.45
    context[:, -5:] = -1
    supplied = predictor.standing_start(context)
    command = q.new_tensor([[[0.12, 0.21, 0]]])
    return predictor, adapter, world, context, supplied, command


@needs_cuda
def test_checked_hold_prepares_actual_home_without_instantaneous_pose_reset(rollout_scene):
    _, adapter, world, context, _, _ = rollout_scene
    initial = adapter.initialize(world.q_init[None], context, world=world)
    assert initial.compatible.all(), initial.diagnostics
    assert initial.preparation.steps[0].valid_ticks.sum().item() == 51
    trace = initial.preparation.steps[0].trajectory
    q, qd = adapter.planner(trace)
    torch.testing.assert_close(q[:, 0], world.q_init[None])
    torch.testing.assert_close(qd[:, 0], torch.zeros_like(qd[:, 0]))
    t, v = adapter.planner(initial.state.tensors[:6])
    torch.testing.assert_close(q[:, -1], t)
    torch.testing.assert_close(qd[:, -1], v)
    assert not torch.allclose(t, world.q_init[None], atol=1.0e-4)


@needs_cuda
def test_swing_geometry_clearance_and_control_derivative(rollout_scene):
    from cutamp.step_validation import sweep_constraint_rows, validate_step_geometry

    p, adapter, world, _, start, command = rollout_scene
    command = command.detach().clone().requires_grad_()
    result = p.rollout(start, "L", command)
    assert validate_step_geometry(world, adapter, result)["valid"].all()
    # The previously distant movable now intersects the leading edge of the swing foot.
    pose = torch.eye(4, device=command.device)
    pose[:3, 3] = command.new_tensor([0.273, 0.094, 0.024])
    fixed = {"block2": pose}
    checked = validate_step_geometry(world, adapter, result, object_poses=fixed)
    assert not checked["valid"].any() and checked["penetration"]["left_sole_link_block2"] > 0
    rows = sweep_constraint_rows(world, adapter, result, fixed)
    (grad,) = torch.autograd.grad(rows["swept_geometry"].sum(), command)
    assert torch.isfinite(grad).all() and grad[0, 0, 0].abs() > 0.01
    # A carried block grazes a foreign arm/body; the exemption applies to its holding hand only.
    body_at_trunk = pose.clone()
    body_at_trunk[:3, 3] = 0
    held = {"left": ("block2", body_at_trunk)}
    loaded = validate_step_geometry(world, adapter, result, held=held)
    assert loaded["load_status"] == "load_unmodeled"
    assert loaded["steps"][0]["held_objects"]["left"].shape[-2:] == (4, 4)


@needs_cuda
def test_independent_timeline_must_match_issued_command_and_velocity(rollout_scene):
    from dataclasses import replace

    from cutamp.step_validation import TransitionEvidence, TransitionRequest, TransitionUnavailable, verify_transition

    p, adapter, world, _, start, cmd = rollout_scene
    with torch.no_grad():
        out = p.rollout(start, "L", cmd)
    s = out.steps[0]
    q0, v0 = adapter.planner(start.tensors[:6])
    q, v = adapter.planner(s.trajectory)
    q, v = torch.cat((q0, q[0])), torch.cat((v0, v[0]))
    req = TransitionRequest("L", cmd[0, 0], start, s, q0[0], v0[0], world, {}, {})
    # A controlled validator double exercises the joining contract. This is not independent physical evidence.
    ev = TransitionEvidence("controlled_trace", q, v, torch.zeros_like(v), 0.02, True, True, False, s.terminal)
    js, soles, record = verify_transition(req, ev, adapter, out.intended_final)
    assert record["source"] == ev.source and record["foot"] == "L"
    torch.testing.assert_close(js.velocity[-1], v[-1])
    torch.testing.assert_close(soles, s.achieved[0], atol=0.0001, rtol=0.0)
    with pytest.raises(TransitionUnavailable):
        verify_transition(req, None, adapter)
    with pytest.raises(ValueError, match="carried-object"):
        verify_transition(replace(req, held={"right": ("block2", None)}), ev, adapter)
    with pytest.raises(ValueError, match="every STEP"):
        verify_transition(req, replace(ev, dt=0.05), adapter)
    with pytest.raises(ValueError, match="initial|arriving"):
        verify_transition(replace(req, initial_velocity=v0[0] + 0.5), ev, adapter)
    with pytest.raises(ValueError, match="precision"):
        verify_transition(replace(req, command=cmd[0, 0] + 0.2), ev, adapter)


@needs_cuda
def test_step_record_and_supported_motion_handoff_in_solver(rollout_scene, monkeypatch):
    """A controlled validator and cuRobo segment isolate joining/serialization without a simulator."""
    from dataclasses import replace
    from types import SimpleNamespace

    from curobo.types import JointState

    from cutamp import motion_solver as ms
    from cutamp.config import TAMPConfiguration
    from cutamp.step_validation import TransitionEvidence
    from cutamp.t1_domain import LeftRetractFree, LeftStep
    from cutamp.utils.plan_processor import raw_motion_plan
    from cutamp.utils.timer import TorchTimer
    from cutamp.utils.visualizer import Visualizer

    pred, adapter, world, _, start, command = rollout_scene
    q0, v0 = adapter.planner(start.tensors[:6])
    with torch.no_grad():
        out = pred.rollout(start, "L", command)
    state = out.steps[0]
    q, v = adapter.planner(state.trajectory)
    q, v = torch.cat((q0, q[0])), torch.cat((v0, v[0]))
    acc = torch.zeros_like(v)
    committed = {}
    # This controlled independent source parks the two extra physical head joints
    # during HOLD. The supported cuRobo model locks them at zero.
    terminal = [x.clone() for x in state.terminal.tensors]
    head = [adapter.names.index(n) for n in ("AAHead_yaw", "Head_pitch")]
    terminal[4][:, head] = terminal[5][:, head] = 0
    terminal[7][:, -1, head] = -pred.model.home_joints.reshape(-1)[head]
    terminal[7][:, -1, [len(adapter.names) + i for i in head]] = 0
    measured = replace(state.terminal, tensors=tuple(terminal), history_source="controlled_head_handoff")

    def evidence(req):
        return TransitionEvidence("controlled_trace", q, v, acc, 0.02, True, True, False, measured)

    step = LeftStep.ground({param.name: param.name for param in LeftStep.parameters})
    retract = LeftRetractFree.ground({param.name: param.name for param in LeftRetractFree.parameters})
    planner = SimpleNamespace(
        kinematics=world.kinematics,
        set_support_stance=lambda s: torch.testing.assert_close(s, state.achieved[0], atol=0.0001, rtol=0),
    )
    monkeypatch.setattr(ms, "_reset_attachment_state", lambda *_: None)

    def plan_segment(planner, conf, last_js, **kw):
        torch.testing.assert_close(last_js.position[0], q[-1])
        torch.testing.assert_close(last_js.velocity[0], v[-1])
        committed["joined"] = True
        return JointState(
            q[-1:].repeat(2, 1), v[-1:].repeat(2, 1), acc[-1:].repeat(2, 1), joint_names=list(world.model.joint_names)
        )

    monkeypatch.setattr(ms, "_plan_arm_to_conf", plan_segment)
    monkeypatch.setattr(ms, "_interp_plan", lambda js: js)
    monkeypatch.setattr(ms, "_plan_dt", lambda _: 0.02)
    cfg = TAMPConfiguration()
    vis = Visualizer(cfg, None)
    args = ([step, retract], {"left_q0": q0[0], "q_retract": q[-1]}, {0: (start, command, out)}, adapter, world, cfg)
    unavailable, status = ms.plan_validated_step_motion(*args, None, timer=TorchTimer(), visualizer=vis, ik_tol=None)
    assert unavailable is None and status["motion_validation"] == "unavailable"
    head_missing, status = ms.plan_validated_step_motion(
        *args,
        lambda req: replace(evidence(req), terminal=None),
        timer=TorchTimer(),
        visualizer=vis,
        ik_tol=None,
        get_planner=lambda: planner,
    )
    assert head_missing is None and "measured terminal head" in status["reason"]
    plan, met = ms.plan_validated_step_motion(
        *args, evidence, timer=TorchTimer(), visualizer=vis, ik_tol=None, get_planner=lambda: planner
    )
    assert committed and plan is not None, met
    assert [entry["phase"] for entry in plan] == ["step", "retract"]
    raw = raw_motion_plan(plan, world.model)["segments"]
    assert raw[0]["step"]["source"] == "controlled_trace"
    assert raw[0]["step"]["foot"] == "L"
    assert raw[0]["velocity"][-1].tolist() == raw[1]["velocity"][0].tolist()
    from cutamp.landing_predictor import LandingState

    replay = raw[0]["step"]["model_prediction"]
    incoming = LandingState(
        tuple(q0.new_tensor(x)[None] for x in replay["initial"].values()),
        q0.new_tensor(replay["dynamics_normalized"])[None],
        replay["history_source"],
    )
    with torch.no_grad():
        result = pred.rollout(incoming, replay["foot"], q0.new_tensor(replay["command"])[None, None])
    torch.testing.assert_close(result.steps[0].achieved, state.achieved)
    assert len(replay["actions"]) == replay["step_ticks"] + replay["hold_ticks"]
    assert raw[0]["step"]["measured_terminal"]["tensors"][-1] == state.terminal.tensors[-1][0].tolist()


@needs_cuda
def test_independent_hold_preparation_requires_live_controller_history(rollout_scene):
    from dataclasses import replace

    from cutamp.step_validation import TransitionEvidence, TransitionRequest, TransitionUnavailable, verify_transition

    _, adapter, world, context, _, _ = rollout_scene
    real = adapter.initialize(world.q_init[None], context, world=world)
    hold = real.preparation
    cmd = world.q_init.new_empty(0)
    q, qd = adapter.planner(hold.steps[0].trajectory)
    req = TransitionRequest("HOLD", cmd, hold.initial, hold.steps[0], q[0, 0], qd[0, 0], world, {}, {})
    evidence = TransitionEvidence("controlled_hold", q[0], qd[0], qd[0] * 0, 0.02, True, True, False, real.state)
    js, _, info = verify_transition(req, evidence, adapter)
    assert info["phase"] == "step_prepare" and info["clock"] == 50
    torch.testing.assert_close(js.position[0], world.q_init)
    torch.testing.assert_close(js.velocity[-1], qd[0, -1])
    with pytest.raises(TransitionUnavailable, match="histories"):
        verify_transition(req, replace(evidence, terminal=None), adapter)
    stale = list(real.state.tensors)
    stale[7] = stale[7].clone()
    stale[7][:, -1, 9] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        verify_transition(req, replace(evidence, terminal=replace(real.state, tensors=tuple(stale))), adapter)


@needs_cuda
def test_run_cutamp_motion_without_validator_keeps_feasible_model_row(tmp_path):
    from cutamp.tests.test_step_planner_integration import paired_case

    metrics = paired_case("near", tmp_path, motion=True)
    assert not metrics["found_solution"] and not metrics["motion_plan_success"]
    attempt = next(a for a in metrics["step_attempts"] if a["status"] == "model_feasible")
    row = attempt["particle_candidate"]
    assert attempt["accepted_rows"] == 0 and row["model_feasible"]
    assert row["motion_validation"] == "unavailable" and "validator" in row["motion_reason"]
    assert row["gaps"][0]["margin_m"] >= 0.05
    assert max(row["gaps"][0]["final_xy_error_m"]) < 0.02
    assert max(row["gaps"][0]["final_yaw_error_deg"]) < 1.0


@needs_cuda
def test_supported_motion_optimizer_moves_cost_with_realized_stance(rollout_scene):
    from types import SimpleNamespace

    from cutamp.com_polygon_cost import ComOverAchievedStanceCost, ComOverBasePolygonCostCfg

    _, _, world, _, _, _ = rollout_scene
    q = world.q_init
    s = q.new_tensor([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])
    cfg = ComOverBasePolygonCostCfg(
        weight=[1.0], device_cfg=world.device_cfg, half_extents=q.new_ones(2), base_link_name=world.model.support.link
    )
    cost = ComOverAchievedStanceCost(cfg, s, world.model.sole_box)
    state = SimpleNamespace(cuda_robot_model_state=SimpleNamespace(robot_com=q.new_zeros(1, 1, 4)))
    before = (cost.normals.data_ptr(), cost.bounds.data_ptr())
    assert float(cost.forward(state)) == 0
    shifted = s + q.new_tensor([0.30, 0.05, 0])
    cost.set_stance(shifted)
    assert float(cost.forward(state)) > 0.01
    state.cuda_robot_model_state.robot_com[0, 0, :2] = shifted[:, :2].mean(0)
    assert float(cost.forward(state)) == 0
    assert (cost.normals.data_ptr(), cost.bounds.data_ptr()) == before


@needs_cuda
def test_rebind_grasp_and_approach_to_independently_achieved_soles(rollout_scene):
    from cutamp.config import TAMPConfiguration
    from cutamp.motion_solver import _hand_poses
    from cutamp.particle_initialization import IKTol
    from cutamp.step_validation import SteppedMotion
    from cutamp.t1_domain import LeftPick, LeftPlace
    from cutamp.utils.common import mat4x4_to_grasp_7dof

    predictor, adapter, world, _, start, command = rollout_scene
    with torch.no_grad():
        step = predictor.rollout(start, "L", command).steps[0]
    motion = SteppedMotion([], {}, adapter, world, None)
    motion.stance = step.achieved[0]
    q = adapter.planner(step.terminal.tensors[:6])[0]
    tool = world.tool_from_ee[world.tool_frame_for_arm("left")]
    wrist = _hand_poses(world, q, "left")[0]
    objects = {"obj": torch.eye(4, device=q.device)}
    grasp = mat4x4_to_grasp_7dof(wrist @ torch.linalg.inv(tool))
    for op in (LeftPick, LeftPlace):
        action = op.ground({p.name: p.name for p in op.parameters})
        particle = {"q": q[0], "grasp": grasp.clone(), "placement": q.new_zeros(4)}
        pre, post, _ = motion.approach(
            0, action, particle, objects, TAMPConfiguration(), IKTol(0.005, 0.05, 0.005, 0.05)
        )
        for pose in (pre, particle["q"], post):
            soles = predictor.model.feet(adapter.physical(pose[None]))[0][0]
            torch.testing.assert_close(soles, motion.stance, atol=0.002, rtol=0)
        actual = _hand_poses(world, particle["q"][None], "left")[0]
        assert torch.linalg.vector_norm(actual[:3, 3] - wrist[:3, 3]) < 0.005
        torch.testing.assert_close(particle["grasp"], grasp)


@needs_cuda
def test_selected_model_evidence_replays_synthetic_preparation_from_its_true_first_frame(rollout_scene):
    from cutamp.landing_predictor import prediction_precision
    from cutamp.step_particles import _evidence

    predictor, adapter, world, context, _, command = rollout_scene
    q = world.q_init.repeat(2, 1)
    q[1, 0] += 0.03
    dynamics = context.expand(2, -1)
    with torch.no_grad(), prediction_precision():
        start = adapter.initialize(q, dynamics, world=world)
        cmds = command.expand(2, -1, -1)
        result = predictor.rollout(start.state, "L", cmds)
        record = _evidence(adapter, {1: (start.state, cmds, result, start.preparation)}, 1)()["1"][0]
        prep = record["preparation"]
        assert prep["history_source"] == "synthetic_hold_issue"
        assert prep["hold_ticks"] == 50 and len(prep["actions"]) == 50 and len(prep["sole_trace"]) == 51
        assert prep["trajectory"]["root_position"][0] == q[1, :3].tolist()
        assert prep["terminal"] == record["initial"]
        values = tuple(q.new_tensor(x)[None] for x in prep["initial"].values())
        for _ in range(prep["hold_ticks"]):
            values = predictor.model(values, q.new_zeros(1, 5), context)
        for v, target in zip(values, record["initial"].values()):
            torch.testing.assert_close(v[0], q.new_tensor(target), atol=1.0e-4, rtol=1.0e-4)
