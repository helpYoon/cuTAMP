"""GN and the gates on the real legs (design 7.4). On the real legs:
- 12 sole rows in GN's level 1 hold both soles on the stance, beside the active CoM rows, and the base columns are
  free (``_row_layout`` freezes them only where ``model.lock_base``);
- level 1 holds the CoM in the real legs' keyframe rectangle (ruling R22);
- level 1 holds the Trunk tilt within an envelope's maximum: one row, active where the tilt is past the bound or a
  step would push it past, as a CoM axis is. Route A's envelope sets no tilt bound, so the tilt tests plan on the
  real legs under an envelope that bounds it at 1.0 rad (``_with_tilt_bound``);
- each call ends with a closing step, so the soles hold after every call, out-of-reach targets included, and a
  row that starts within one step clamp of a tilt bound is within it after the call;
- the merge gate (``gn_gate``), the final repair and the kinematic constraint all check the soles against the
  stance.
The GPU tests share conftest's module-scoped ``real_world`` and ``real_plan``: one world per module. Each takes its
confs from ``real_plan``, so no test solves the world's 64-row IK at another batch size (ruling R25)."""

import dataclasses

import pytest
import torch

from cutamp.robots.t1 import REAL_LEGS
from cutamp.tests.conftest import clone_particles, needs_cuda
from cutamp.tests.test_real_legs_init import POS_TOL, ROT_TOL, _home, _poses, _sole_errors

#: The 3 cm move of every hand target in the GN tests.
MOVE = (0.03, 0.0, 0.0)
#: A move no balanced conf standing on the stance can follow: 30 cm forward and 20 cm down.
FAR = (0.30, 0.0, -0.20)
#: The tilt tests' maximum Trunk tilt, rad, in an envelope that replaces route A's, which sets none.
TILT_MAX = 1.0
#: The real legs under that envelope.
TILTED = dataclasses.replace(REAL_LEGS, envelope=REAL_LEGS.envelope._replace(trunk_tilt_max=TILT_MAX))


def _with_tilt_bound(world, monkeypatch, tilt_max=TILT_MAX):
    """``world`` on TILTED until the test ends (``tilt_max`` None: left on REAL_LEGS). GN, the cost function and the
    final repair read the bound from ``world.model``."""
    if tilt_max is not None:
        assert tilt_max == TILT_MAX
        monkeypatch.setattr(world, "robot_container", dataclasses.replace(world.robot_container, model=TILTED))
    return world


def _in_reach(q):
    """The rows of ``q`` whose Trunk tilt starts at most ``GN_STEP_CLAMP`` (0.15 rad) past TILT_MAX: one call
    brings them inside it. On blocks_t1's first pick that is 36 of 64 rows on one init and 5 of 60 on the other,
    whose other rows the init's IK bows mostly 0.15-0.4 rad past it."""
    from cutamp.gn_refresh import GN_STEP_CLAMP
    return q[_tilt(q) <= TILT_MAX + GN_STEP_CLAMP]


def _tilt(q):
    """``[B]`` float64: the Trunk tilt from the vertical, ``arccos(cos(pitch) cos(roll))`` of the base."""
    names = list(REAL_LEGS.joint_names)
    q = q.double()
    c = q[:, names.index("base_j_pitch")].cos() * q[:, names.index("base_j_roll")].cos()
    return torch.arccos(c.clamp(-1.0, 1.0))


def _first_pick(world, plan):
    """``(arm, conf, q [k, dof])``: the skeleton's first pick, its conf and the rows of it that the init's IK solved
    (a row whose IK failed holds home). The first skeleton follows PYTHONHASHSEED: on blocks_t1 its first pick has
    64 solved rows, every CoM inside the keyframe rectangle, on one init, and 60, 9 of them outside, on the other."""
    skeleton, particles = plan
    op = next(op for op in skeleton if op.operator.metadata.action_type == "pick")
    conf = op.values[-1]
    q = particles[conf]
    return op.operator.metadata.arm, conf, q[(q != _home(world)).any(-1)]


def _moved_targets(world, q, arm, move=MOVE):
    """``(targets [B, 4, 4], before [B])``: each row's own hand pose at ``q`` moved by ``move``, and its hand error
    there."""
    targets = _poses(world, q, world.tool_frame_for_arm(arm))
    targets[:, :3, 3] += torch.tensor(move, device=q.device)
    return targets, _hand_error(world, q, targets, arm)


def _hand_error(world, q, targets, arm):
    from cutamp.costs import curobo_pose_error
    return curobo_pose_error(_poses(world, q, world.tool_frame_for_arm(arm)), targets)[0]


def _sole_rows(world, q, arm, goals):
    """``(r [B, 12], jac [B, 12, dof], col_mask [B, dof])``: :func:`_sole_residual_and_jacobian` against ``goals``
    ``[2, 4, 4]`` at ``q``, on ``world.kinematics_jac`` and ``arm``'s column mask. Cloned: the next FK call rewrites
    cuRobo's buffers."""
    from curobo.types import JointState
    from cutamp.gn_refresh import _row_layout, _sole_residual_and_jacobian
    kin = world.kinematics_jac
    state = kin.compute_kinematics(JointState.from_position(q.contiguous(), joint_names=list(kin.joint_names)))
    col_mask, _ = _row_layout(world.model, (arm,), q.shape[0], q.shape[1], q.device, q.dtype)
    sole_idx = tuple(list(kin.tool_frames).index(frame) for frame in world.model.sole_frames)
    r, jac = _sole_residual_and_jacobian(state, sole_idx, goals, col_mask)
    return r.clone(), jac.clone(), col_mask


@needs_cuda
def test_sole_rows_match_finite_differences(real_world, real_plan):
    # Rows 0-5 are the left sole's world [Δpos; rotvec], 6-11 the right's, the metric of GN's hand rows: the goal
    # minus FK, so 1 cm of base_j_x off the stance reads -1 cm in x on both soles.
    model = real_world.model
    arm, _, q = _first_pick(real_world, real_plan)
    home = _home(real_world).repeat(2, 1)
    home[1, list(model.joint_names).index("base_j_x")] += 0.01
    want = torch.zeros(2, 12, device=home.device)
    want[1, [0, 6]] = -0.01
    torch.testing.assert_close(_sole_rows(real_world, home, arm, real_world.stance_soles)[0], want, rtol=0, atol=1e-5)

    # At 8 IK-solved confs, the residual's central differences are minus the Jacobian on every column GN may move: the
    # base's among them, and only the idle arm's frozen. Each conf aims at its own soles, so the differences see the
    # Jacobian alone: against the stance, a residual of e rad moves the rotation rows' derivative by up to e/2 (the log
    # map's first-order term; 6.2e-3 measured at 15 mrad), and at h = 1e-4 float32 FK noise reaches 1.4e-3 (h = 1e-3:
    # 9e-5).
    idle = model.right_arm if arm == "left" else model.left_arm
    h = 1e-3
    assert len(q) >= 8
    for qi in q[:8]:
        own = torch.stack([_poses(real_world, qi[None], frame)[0] for frame in model.sole_frames])
        _, jac, col_mask = _sole_rows(real_world, qi[None], arm, own)
        free = col_mask[0].nonzero().flatten()
        assert free.tolist() == [k for k in range(len(qi)) if not idle.start <= k < idle.stop]
        assert not bool(jac[..., idle].any())
        dq = h * torch.eye(len(qi), device=qi.device)[free]                       # one row per free column
        fd = -(_sole_rows(real_world, qi + dq, arm, own)[0] - _sole_rows(real_world, qi - dq, arm, own)[0]) / (2 * h)
        err = (jac[0][:, free] - fd.T).abs().amax(0)                              # [free]
        assert float(err.max()) <= 1e-3, model.joint_names[int(free[err.argmax()])]


def _assert_soles_held(world, q):
    pos, rot = _sole_errors(world, q)
    assert float(pos.max()) <= 1e-3 and float(rot.max()) <= 0.01, (float(pos.max()), float(rot.max()))


@needs_cuda
def test_gn_holds_the_soles(real_world, real_plan):
    # Every solved pick row, its hand target moved 3 cm: after one call GN has reached toward it with the legs and
    # the base, holding both soles on the stance and the CoM in the keyframe rectangle (level 1), also on the rows
    # whose CoM starts outside, which need the largest moves.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project_balanced
    arm, _, q0 = _first_pick(real_world, real_plan)
    targets, before = _moved_targets(real_world, q0, arm)
    q = gn_project_balanced(real_world, q0, targets, arm)
    _assert_soles_held(real_world, q)
    assert bool(compute_com_polygon_mask(real_world, q).all())
    after = _hand_error(real_world, q, targets, arm)
    assert float(after.median()) < float(before.median()), (float(before.median()), float(after.median()))


@needs_cuda
def test_gn_brings_the_trunk_tilt_inside_the_bound(real_world, real_plan, monkeypatch):
    # On the floor scene the init's IK bows most first-pick rows past 1.0 rad. Under a 1.0 rad bound, one call, the
    # hand targets moved 3 cm, brings every row that starts within one step clamp of it (``_in_reach``) inside it,
    # with both soles on the stance and the CoM in the keyframe rectangle on every row. Rows further past take more
    # calls, and one bowed past horizontal can stay past: the base's ZYX angles hold that bow as roll near pi, where
    # the tilt row's roll derivative vanishes, with pitch at its cap.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project_balanced
    arm, _, q0 = _first_pick(real_world, real_plan)
    _with_tilt_bound(real_world, monkeypatch)
    q0 = _in_reach(q0)
    assert int((_tilt(q0) > TILT_MAX).sum()) >= 3
    targets, _ = _moved_targets(real_world, q0, arm)
    q = gn_project_balanced(real_world, q0, targets, arm)
    assert float(_tilt(q).max()) <= TILT_MAX + 1e-6, float(_tilt(q).max())
    _assert_soles_held(real_world, q)
    assert bool(compute_com_polygon_mask(real_world, q).all())


@needs_cuda
def test_tilt_row_leaves_rows_inside_the_bound_alone(real_world, real_plan, monkeypatch):
    # Home rows reaching 3 cm, and the first pick's rows each aiming at its own hand pose, so a step only closes their
    # soles and, past the bound, their tilt. The rows that start inside the bound with their CoM in the rectangle
    # come out of a call bitwise as they do with the tilt row's Jacobian zeroed; the rows that start past the bound
    # do not, and the zeroed call leaves them past it.
    import cutamp.gn_refresh as gn_refresh
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    arm, _, q_pick = _first_pick(real_world, real_plan)
    _with_tilt_bound(real_world, monkeypatch)
    home = _home(real_world).repeat(8, 1)
    q0 = torch.cat([home, q_pick])
    targets = torch.cat([_moved_targets(real_world, home, arm)[0],
                         _poses(real_world, q_pick, real_world.tool_frame_for_arm(arm))])
    q = gn_refresh.gn_project_balanced(real_world, q0, targets, arm)
    row = gn_refresh._tilt_row

    def zeroed(q, pitch, roll, col_mask):
        c, jac = row(q, pitch, roll, col_mask)
        return c, torch.zeros_like(jac)

    monkeypatch.setattr(gn_refresh, "_tilt_row", zeroed)
    q_zeroed = gn_refresh.gn_project_balanced(real_world, q0, targets, arm)
    tilt0 = _tilt(q0)
    calm = (tilt0 <= TILT_MAX - 0.05) & compute_com_polygon_mask(real_world, q0)
    past = tilt0 > TILT_MAX
    assert bool(calm[:8].all()) and bool(past.any()), int(past.sum())
    assert not torch.equal(q[:8], home)  # not vacuous: the home rows reached
    assert torch.equal(q[calm], q_zeroed[calm])
    assert not bool((q[past] == q_zeroed[past]).all(-1).any())
    assert float(_tilt(q_zeroed[past]).max()) > TILT_MAX


@needs_cuda
@pytest.mark.parametrize("tilt_max", [None, TILT_MAX])
def test_gn_holds_the_soles_out_of_reach(real_world, real_plan, monkeypatch, tilt_max):
    # Hand targets no balanced conf on the stance can reach: four calls, each from the last result as successive
    # refreshes run, keep both soles on the stance after every call, and under a tilt bound the Trunk tilt of the rows
    # in reach of it (``_in_reach``) within it, and every hand stays over 2 cm off, so the feet did not slide to fake
    # the reach. Reaching for them drives joints into their limits on every call, where a clamped step or the drift
    # of a saturated last step would move the soles.
    from cutamp.gn_refresh import gn_project_balanced
    arm, _, q = _first_pick(real_world, real_plan)
    _with_tilt_bound(real_world, monkeypatch, tilt_max)
    if tilt_max is not None:
        q = _in_reach(q)
    targets, _ = _moved_targets(real_world, q, arm, FAR)
    for _ in range(4):
        q = gn_project_balanced(real_world, q, targets, arm)
        _assert_soles_held(real_world, q)
        if tilt_max is not None:
            assert float(_tilt(q).max()) <= tilt_max + 1e-6, float(_tilt(q).max())
        hand = _hand_error(real_world, q, targets, arm)
        assert float(hand.min()) > 0.02, float(hand.min())


@needs_cuda
def test_gn_holds_the_com_in_the_real_legs_rectangle(real_world, real_plan, monkeypatch):
    # Ruling R22: _balanced_step's half=None default is the simplified rectangle, so every real-leg call passes the
    # real legs' own, with the sole rows.
    import cutamp.gn_refresh as gn_refresh
    arm, _, q0 = _first_pick(real_world, real_plan)
    targets, _ = _moved_targets(real_world, q0, arm)
    calls, step = [], gn_refresh._balanced_step

    def spy(*args, **kwargs):
        calls.append((kwargs.get("half"), kwargs.get("jac_sole") is not None, kwargs.get("r_sole") is not None))
        return step(*args, **kwargs)

    monkeypatch.setattr(gn_refresh, "_balanced_step", spy)
    gn_refresh.gn_project_balanced(real_world, q0, targets, arm)
    assert calls == [(REAL_LEGS.support.keyframe_half, True, True)] * gn_refresh.GN_ITERS


@needs_cuda
def test_gn_without_sole_rows_slides_the_soles(real_world, real_plan, monkeypatch):
    # The same reach with the sole rows zeroed: the free base and legs carry the hand, and the soles slide off the
    # stance, so the rows bear the load.
    import cutamp.gn_refresh as gn_refresh
    arm, _, q0 = _first_pick(real_world, real_plan)
    targets, _ = _moved_targets(real_world, q0, arm)

    def zeros(state, sole_idx, goals, col_mask):
        return col_mask.new_zeros(col_mask.shape[0], 12), col_mask.new_zeros(col_mask.shape[0], 12, col_mask.shape[1])

    monkeypatch.setattr(gn_refresh, "_sole_residual_and_jacobian", zeros)
    pos, _ = _sole_errors(real_world, gn_refresh.gn_project_balanced(real_world, q0, targets, arm))
    assert float(pos.max()) > 0.005, float(pos.max())


@needs_cuda
def test_gate_rejects_drifted_soles(real_world):
    # Each row aims at its own hand pose, so its hand is exact: home passes, and home shifted 1 cm by base_j_x fails
    # pose_ok on its soles alone. Its CoM moved with it, so com_ok would pass it (design 7.4).
    from cutamp.gn_refresh import gn_gate
    names = list(real_world.model.joint_names)
    q = _home(real_world).repeat(2, 1)
    q[1, names.index("base_j_x")] += 0.01
    targets = _poses(real_world, q, real_world.tool_frame_for_arm("left"))
    pose_ok, com_ok = gn_gate(real_world, q, targets, "left", POS_TOL, ROT_TOL)
    assert pose_ok.tolist() == [True, False]
    assert com_ok.tolist() == [True, True]


@needs_cuda
def test_soles_on_stance_at_the_checker_tolerances(real_world):
    # The final repair's sole test, on world.kinematics and the checker's own sole tolerances: home stands on the
    # stance, and home shifted 1 cm by base_j_x does not.
    from cutamp.com_polygon_cost import _fk_state
    from cutamp.gn_refresh import soles_on_stance
    from cutamp.scripts.utils import default_constraint_to_tol
    from cutamp.task_planning.constraints import KinematicConstraint

    tol = default_constraint_to_tol[KinematicConstraint.type]
    q = _home(real_world).repeat(2, 1)
    q[1, list(real_world.model.joint_names).index("base_j_x")] += 0.01
    ok = soles_on_stance(real_world, _fk_state(real_world, q), tol["sole_pos_err"], tol["sole_rot_err"])
    assert ok.tolist() == [True, False]


def test_tilt_within_bound_on_the_motion_values_metric():
    # The final repair's tilt test, on the Motion value's metric and tolerance: under a 1.0 rad bound, upright and on
    # the bound (roll 1.0 rad) pass, 0.01 rad past it fails; route A's envelope, which sets no bound, and a model
    # without an envelope pass every conf.
    from types import SimpleNamespace
    from cutamp.gn_refresh import tilt_within_bound
    from cutamp.robots.t1 import SIMPLIFIED
    names = list(REAL_LEGS.joint_names)
    q = torch.tensor(REAL_LEGS.home).repeat(3, 1)
    q[1:, names.index("base_j_roll")] = torch.tensor([1.0, 1.01])
    assert tilt_within_bound(SimpleNamespace(model=TILTED), q, 1e-6).tolist() == [True, True, False]
    assert tilt_within_bound(SimpleNamespace(model=REAL_LEGS), q, 1e-6).all()
    assert tilt_within_bound(SimpleNamespace(model=SIMPLIFIED), q[:, :len(SIMPLIFIED.joint_names)], 1e-6).all()


@needs_cuda
@pytest.mark.parametrize("tilt_max", [None, TILT_MAX])
def test_final_repair_takes_no_conf_past_the_tilt_bound(real_world, real_plan, monkeypatch, tilt_max):
    # The final pre-check refresh re-solves the rows that miss the hand, the soles or the CoM on cuRobo's IK, which
    # does not see a tilt bound. Two first-pick rows miss their hand (a shoulder moved 0.3 rad), and the stubbed IK
    # returns their own confs, one of them tilted 1.03 rad: under a 1.0 rad bound the row within it takes its repair
    # and the tilted one keeps its conf; route A sets no bound, so both take their repairs.
    import cutamp.optimize_plan as op
    from cutamp.config import TAMPConfiguration
    from cutamp.tests.conftest import make_optimizer
    skeleton, particles = real_plan
    particles = clone_particles(particles)
    arm, conf, _ = _first_pick(real_world, real_plan)
    _with_tilt_bound(real_world, monkeypatch, tilt_max)
    names = list(REAL_LEGS.joint_names)
    q = particles[conf].detach()
    solved = (q != _home(real_world)).any(-1).nonzero().flatten()
    b = int(solved[_tilt(q[solved]).argmin()])  # its own conf, its repair, is within the bound
    a = int(solved[solved != b][0])
    assert float(_tilt(q[b:b + 1])) <= TILT_MAX - 0.01
    repaired = q.clone()
    repaired[a, names.index("base_j_pitch")], repaired[a, names.index("base_j_roll")] = 0.6, 0.9
    assert float(_tilt(repaired[a:a + 1])) > TILT_MAX + 0.01
    with torch.no_grad():
        particles[conf][[a, b], names.index("Left_Shoulder_Pitch" if arm == "left" else "Right_Shoulder_Pitch")] += 0.3
    moved = particles[conf].detach().clone()
    monkeypatch.setattr(op, "_ik_for_pose", lambda *args, **kwargs: None)
    monkeypatch.setattr(op, "_ik_solution_to_full_q", lambda result, world: repaired)
    monkeypatch.setattr(op, "_ik_success_mask", lambda result: torch.ones(len(repaired), dtype=torch.bool,
                                                                            device=repaired.device))
    deps = [dep for dep in op._build_ik_deps(skeleton, real_world) if dep.q_name == conf]
    make_optimizer(TAMPConfiguration(robot="real_legs", num_particles=64))._refresh_ik_deps(particles, deps, real_world)
    assert torch.equal(particles[conf][a], moved[a] if tilt_max is not None else repaired[a])
    assert torch.equal(particles[conf][b], repaired[b])


@needs_cuda
def test_kinematic_constraint_checks_the_soles(real_world, real_plan):
    # One particle's pick conf shifted 1 cm by base_j_x: its two soles' sole_pos_err read 1 cm, and on the sole values
    # ConstraintChecker drops that particle only.
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_function import CostFunction
    from cutamp.cost_reduction import CostReducer
    from cutamp.rollout import RolloutFunction
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.task_planning.constraints import KinematicConstraint

    kc = KinematicConstraint.type
    assert {n: default_constraint_to_tol[kc][n] for n in ("sole_pos_err", "sole_rot_err")} == {
        "sole_pos_err": POS_TOL, "sole_rot_err": ROT_TOL}
    reducer = CostReducer(default_constraint_to_mult.copy())
    assert [reducer._get_multiplier(kc, n) for n in ("sole_pos_err", "sole_rot_err")] == [1.0, 5.0]

    skeleton, particles = real_plan
    config = TAMPConfiguration(robot="real_legs", num_particles=64)
    rollout_fn, cost_fn = RolloutFunction(skeleton, real_world, config), CostFunction(skeleton, real_world, config)
    checker = ConstraintChecker(default_constraint_to_tol.copy())
    _, conf, _ = _first_pick(real_world, real_plan)
    t = rollout_fn.conf_params.index(conf)

    def sole_values(p):
        with torch.no_grad():
            rollout = rollout_fn(p)
            values = cost_fn(rollout, p)[kc]["values"]
        assert rollout["world_from_soles"].shape == (64, len(rollout_fn.conf_params), 2, 4, 4)
        soles = {n: values[n] for n in ("sole_pos_err", "sole_rot_err")}
        return soles, checker.get_mask({kc: {"type": "constraint", "values": soles}}, verbose=False)

    before, mask_before = sole_values(particles)
    assert before["sole_pos_err"].shape == before["sole_rot_err"].shape == (64, 2 * len(rollout_fn.conf_params))
    assert bool(mask_before.all())
    row = int(before["sole_pos_err"][:, 2 * t:2 * t + 2].amax(-1).argmin())  # its soles start within 1 mm
    assert float(before["sole_pos_err"][row, 2 * t:2 * t + 2].max()) < 1e-3

    shifted = clone_particles(particles)
    with torch.no_grad():
        shifted[conf][row, list(real_world.model.joint_names).index("base_j_x")] += 0.01
    after, mask_after = sole_values(shifted)
    err = after["sole_pos_err"][row, 2 * t:2 * t + 2]
    assert float((err - 0.01).abs().max()) < 1e-3, err
    expected = mask_before.clone()
    expected[row] = False
    assert torch.equal(mask_after, expected)
