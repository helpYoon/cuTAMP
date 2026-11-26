"""Motion planning on the real legs with planted feet and the torque checks (design 7.4a, 7.5): the floor's push on
every trajopt transition, SegmentTorques against trajopt's own torques, trajopt's success test leaving torque to the
balance gate, the gate's multibody ZMP against a float64 reference, and its sole, CoM and torque checks. blocks_t1 end
to end on the real legs is test_motion_anchor's."""

import functools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cutamp.robots.t1 import REAL_LEGS
from cutamp.tests.conftest import needs_cuda

DT = 0.025  # the hand-built segments' frame spacing, s


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
    # cuRobo applies cspace.velocity_scale twice: t1_29dof.yml carries sqrt(7.29 / 12.57) on both
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
    assert all("compute_augmented_state" in vars(tm) and tm.robot_model.compute_com for tm in transitions)
    assert all("compute_augmented_state" not in vars(tm) for r in planner.ik_solver.get_all_rollout_instances()
               for tm in (r.transition_model, r.metrics_transition_model))
    # The trajopt CoM cost finds its support frame among the planner's tool poses.
    assert "stance" in planner.trajopt_solver.tool_frames
    assert all("stance" in tm.robot_model.tool_frames for tm in transitions)
    names = list(planner.trajopt_solver.joint_names)
    base = [names.index(j) for j in real_world.model.joint_names[real_world.model.base]]
    rest = _ramp(real_world, {}, 2 * DT)
    loads = [float(_transition_torques(tm, rest)[:, base].abs().max()) for tm in transitions]
    assert max(loads) <= 0.05, loads


@needs_cuda
def test_trajopt_success_test_leaves_torque_to_the_gate(planner):
    # With timeopt off, trajopt times each segment at the fastest pace its speed, acceleration and jerk caps allow.
    # A success test bounding torque there dropped segments that fit slower: at h 0.2 (2026-11-09), every segment
    # dropped on torque fit at half speed. The gate slows a segment until every effort holds (_retime), so
    # the success test (the metrics rollouts: trajopt's horizon and the interpolated re-check) bounds every
    # joint's position, speed, acceleration and jerk but no torque, and trajopt's cost still penalizes torque.
    ts = planner.trajopt_solver
    tests = [ts.metrics_rollout, ts.additional_metrics_rollouts["interpolated_rollout"]]
    for r in tests:
        weight = r.metrics_constraint_manager.get_cost("cspace").config.weight  # position, vel, acc, jerk, torque
        assert float(weight[4]) == 0.0 and float(weight[:4].min()) > 0.0, r.rollout_instance_name
    for r in ts.optimizer_rollouts:
        assert float(r.cost_manager.get_cost("cspace").config.weight[4]) > 0.0


def _plan_reach(world, real_plan, planner):
    """A planned home -> reach segment: plan_cspace's result for the first pick conf of ``real_plan`` that plans."""
    from curobo.types import JointState
    from cutamp.motion_solver import _plan_arm_to_conf
    from cutamp.tests.test_real_legs_gn import _accepted
    skeleton, particles = real_plan
    op = next(op for op in skeleton if op.operator.metadata.action_type == "pick")
    obj, conf = op.values[0], op.values[-1]
    names = list(world.model.joint_names)
    home = torch.tensor(world.model.home, device=world.device)
    start = JointState.from_position(home[None], joint_names=names)
    for q in particles[conf][_accepted(world, real_plan, conf)][:8]:  # reaches the IK check accepts; some may not plan
        try:
            return _plan_arm_to_conf(planner, q, start, disable_obstacle=obj, ground_op=op)
        except RuntimeError:
            continue
    raise AssertionError("no reach planned from home")


@needs_cuda
def test_segment_torques_equal_trajopt_s(real_world, real_plan, planner):
    # The gate re-runs cuRobo's own two passes: on a planned home -> reach segment, SegmentTorques equals the torque
    # every trajopt transition computes, the cost's and the metrics'.
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
    # interpolated rollout's last ones (its re-check). A graph captured before the floor's push was hooked would
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


def _actuated(world) -> list:
    """The model's actuated joints' columns: every joint but the base's six virtual ones."""
    base = set(range(len(world.model.joint_names))[world.model.base])
    return [j for j in range(len(world.model.joint_names)) if j not in base]


def _on_the_stance(world, moves, T, eps=0.01):
    """The rest-to-rest min-jerk move ``moves`` ({joint: delta} from home) over T s every DT s, the base re-solved at
    every frame so that both soles stay on the stance (test_real_legs_init._stand_on_the_stance, which puts the
    left sole there: the legs must move alike for the right one to follow), its velocity and acceleration by central
    differences over eps s."""
    from curobo.types import JointState
    from cutamp.tests.test_real_legs_init import _stand_on_the_stance
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
def test_real_legs_zmp_matches_multibody_reference(real_world):
    # Home to a leaning conf in 0.6 s, min-jerk in every joint, the base's six with them: the hips bow 0.5 rad, the
    # knees bend 0.3 and the ankles 0.15, the Waist turns 0.4, the left shoulder swings 1 rad forward and the right
    # elbow bends 0.8, the base re-solved so that the end conf stands on the stance (its Trunk pitched 0.35 rad and
    # yawed -0.4). Every link moves and spins. The gate's CoM and multibody ZMP, from the plan's own velocities and
    # accelerations as the gate takes them, match the float64 first-principles ones at every frame, at rest and at the
    # acceleration peaks, where the ZMP runs 9 cm from the CoM. Measured 2026-11-07: the CoM within 1.0e-8 m and the
    # ZMP within 2.7e-5 m of the reference, against tolerances of 2e-4 and 3e-4 m.
    from cutamp.com_polygon_cost import com_zmp_in_support
    from cutamp.motion_solver import _model_frames
    from cutamp.tests.test_real_legs_init import _stand_on_the_stance
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
    com, zmp = com_zmp_in_support(world, _model_frames(world, plan))
    assert float((zmp - com).abs().max()) > 0.05  # the motion moves the ZMP: not a static match

    def q_of_t(t):
        u = min(max(t / T, 0.0), 1.0)
        return (home + (10 * u**3 - 15 * u**4 + 6 * u**5) * delta).numpy()

    for k in range(plan.position.shape[0]):
        ref_zmp, ref_com = _reference_zmp(q_of_t, k * DT)
        np.testing.assert_allclose(com[k].cpu().numpy(), ref_com, atol=2e-4, err_msg=f"frame {k}")
        np.testing.assert_allclose(zmp[k].cpu().numpy(), ref_zmp, atol=3e-4, err_msg=f"frame {k}")


def test_gate_names_torque_in_every_torque_form():
    # Only the gate fails a segment on torque (trajopt's success test leaves it to the gate), and a failed
    # candidate's message names torque in its torque raise (_check_torques). Its velocity and ZMP raises don't. The
    # last assert pins which of the three messages name torque.
    from cutamp.motion_solver import RETIME_MAX, _check_torques, _check_velocity

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
    for check, args in ((_check_torques, (world(2.0), None)), (_check_velocity, (world(0.5), js, RETIME_MAX))):
        with pytest.raises(RuntimeError) as e:
            check(*args)
        raised.append(str(e.value))
    messages = [*raised, f"trajectory ZMP stays outside the support region at the {RETIME_MAX}x slowdown cap"]
    assert "velocity passes" in messages[1]
    assert [("torque" in m) for m in messages] == [True, False, False]


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
