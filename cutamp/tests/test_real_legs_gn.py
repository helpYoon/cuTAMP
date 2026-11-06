"""GN and the gates on the real legs (design 7.4). On the real legs:
- 12 sole rows in GN's level 1 hold both soles on the stance, beside the active CoM rows, and the base columns are
  free;
- level 1 holds the CoM in the real legs' keyframe rectangle, and level 3 pulls it toward the centre, so GN IK's
  confs don't sit on the rectangle's edge;
- level 1 holds the Trunk tilt within an envelope's maximum: one row, active where the tilt is past the bound or a
  step would push it past, as a CoM axis is. Route A's envelope sets no tilt bound, so the tilt tests plan on the
  real legs under an envelope that bounds it at 1.0 rad (``_with_tilt_bound``);
- each call ends with a closing step, so the soles hold after every call, out-of-reach targets included; under a
  tilt bound a row that starts within REACH of it is within it after one call, and stays within it after
  out-of-reach calls;
- the merge gate (``gn_gate``), the final repair and the kinematic constraint all check the soles against the
  stance;
- level 1's collision row is positive exactly where the scene's collision_fn or cuRobo's self-collision kernel is,
  and it clears contact holding the hand; GN IK accepts only confs that pass the checks on their own, and the
  movables it checks sit where the rollout puts them.
The GPU tests share conftest's module-scoped ``real_world`` and ``real_plan``: one world per module, and each test
takes its confs from ``real_plan``."""

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


#: How far past TILT_MAX, rad, a row may start for one GN call to bring it inside, with margin: on blocks_t1's first
#: pick one call brings in every row that starts up to 0.026 rad past it on one init (the next starts 0.51 past), and
#: every row, up to 0.21 rad past, on the other (GN IK's init, PYTHONHASHSEED 0 and 1, 2026-11-06).
REACH = 0.3
#: The most GN IK's accepted first-place rows' median |CoM| / half may be on either axis, with margin: at most 0.06
#: on both inits (PYTHONHASHSEED 0 and 1, 2026-11-06).
CENTRED = 0.25


def _in_reach(q):
    """The rows of ``q`` whose Trunk tilt starts at most REACH past TILT_MAX. On blocks_t1's first pick that is 52 of
    64 rows on one init and all 64 on the other."""
    return q[_tilt(q) <= TILT_MAX + REACH]


def _tilt(q):
    """``[B]`` float64: the Trunk tilt from the vertical, ``arccos(cos(pitch) cos(roll))`` of the base."""
    names = list(REAL_LEGS.joint_names)
    q = q.double()
    c = q[:, names.index("base_j_pitch")].cos() * q[:, names.index("base_j_roll")].cos()
    return torch.arccos(c.clamp(-1.0, 1.0))


def _com_frac(world, q):
    """``[B, 2]``: the CoM about the support's centre, in its frame, over the keyframe rectangle's half-extents (+-1 on
    its edges)."""
    from cutamp.com_polygon_cost import _fk_com_and_base, _project_com_to_base
    support = world.model.support
    com, base_T = _fk_com_and_base(world, q)
    return _project_com_to_base(com, base_T, support.center)[:, :2] / torch.tensor(support.keyframe_half,
                                                                                  device=q.device)


def _accepted(world, plan, conf):
    """``[B]`` bool: the rows of ``conf`` (a pick's or a place's) that GN IK's check accepts
    (:func:`~cutamp.particle_initialization.ik_check` on the default tolerances)."""
    from cutamp.optimize_plan import _build_ik_deps
    from cutamp.particle_initialization import IKTol, ik_check
    from cutamp.tests.conftest import default_checker
    skeleton, particles = plan
    dep = next(dep for dep in _build_ik_deps(skeleton, world) if dep.q_name == conf)
    return ik_check(world, particles[conf], dep.target_fn(particles), dep.arm, dep.movables_fn(particles),
                    IKTol.from_checker(default_checker()))[0]


def _first_pick(world, plan):
    """``(arm, conf, q [k, dof])``: the skeleton's first pick, its conf and the rows of it that the init's GN IK
    accepted (the others hold its closest balanced iterate). The first skeleton follows PYTHONHASHSEED: on blocks_t1
    its first pick accepts all 64 rows on both inits (PYTHONHASHSEED 0 and 1, 2026-11-06), every CoM inside the
    keyframe rectangle."""
    skeleton, particles = plan
    op = next(op for op in skeleton if op.operator.metadata.action_type == "pick")
    conf = op.values[-1]
    return op.operator.metadata.arm, conf, particles[conf][_accepted(world, plan, conf)]


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
    # On the floor scene the init's GN IK bows some first-pick rows past 1.0 rad (17 of 64 on one init, 5 of them in
    # reach; 15 of 64, all in reach, on the other). Under a 1.0 rad bound, one call, the hand targets moved 3 cm,
    # brings every row that starts within REACH of it (``_in_reach``) inside it, with both soles on the stance and the
    # CoM in the keyframe rectangle on every row. Rows further past take more calls.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project_balanced
    arm, _, q0 = _first_pick(real_world, real_plan)
    _with_tilt_bound(real_world, monkeypatch)
    q0 = _in_reach(q0)
    assert bool((_tilt(q0) > TILT_MAX).any())  # not vacuous
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
    # refreshes run, keep both soles on the stance after every call, and every hand stays over 2 cm off, so the feet
    # did not slide to fake the reach. Reaching for them drives joints into their limits on every call, where a
    # clamped step or the drift of a saturated last step would move the soles. Under a tilt bound the calls start from
    # the rows in reach of it (``_in_reach``) after one 3 cm call has brought them inside it, as in the test above,
    # and they stay inside it after every call too. A row still past the bound when it reaches out of reach could come
    # out of that call with its soles off by more than this test's 1 mm, though inside the gate's 5 mm (2 mm, a row
    # 0.27 rad past, on an earlier init); on today's inits none does (0.04 mm at most, 2026-11-06).
    from cutamp.gn_refresh import gn_project_balanced
    arm, _, q = _first_pick(real_world, real_plan)
    _with_tilt_bound(real_world, monkeypatch, tilt_max)
    if tilt_max is not None:
        q = _in_reach(q)
        q = gn_project_balanced(real_world, q, _moved_targets(real_world, q, arm)[0], arm)
        assert float(_tilt(q).max()) <= tilt_max + 1e-6, float(_tilt(q).max())
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
    # Every iteration passes _balanced_step the support's keyframe rectangle and the sole rows, by name.
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
def test_gn_ik_centres_the_com(real_world, real_plan, monkeypatch):
    # Level 3 pulls the CoM toward the keyframe rectangle's centre in the freedom the hand, the soles, balance and
    # collision leave. Without it GN IK leaves the init's accepted first-pick and first-place rows on the edge (a
    # median |CoM| / half of 0.95 in x and 0.96 in y on blocks_t1, both inits, 2026-11-06), the base slid sideways
    # onto one leg, where trajopt's move there from home needed more than the Left_Ankle_Roll's effort. With it the
    # medians drop on both axes, and the first place's are at most 0.06 on both inits. The first pick's are 0.00 on
    # one init and 0.23 (x) and 0.76 (y) on the other, where a further pull would lower the Trunk onto the right
    # Shank (level 3 stops short of a contact).
    import cutamp.gn_refresh as gn_refresh
    import cutamp.particle_initialization as pi
    from cutamp.optimize_plan import _build_ik_deps
    from cutamp.tests.conftest import default_checker
    skeleton, particles = real_plan
    deps = {dep.q_name: dep for dep in _build_ik_deps(skeleton, real_world)}
    for kind in ("pick", "place"):
        conf = next(op for op in skeleton if op.operator.metadata.action_type == kind).values[-1]
        dep = deps[conf]
        centred = _com_frac(real_world, particles[conf][_accepted(real_world, real_plan, conf)]).abs().median(0).values
        monkeypatch.setattr(gn_refresh, "_centring_step", lambda ja, jac, *_: torch.zeros_like(jac[:, 0]))
        torch.manual_seed(0)
        q, ok = pi.gn_ik(real_world, dep.target_fn(particles), dep.arm, dep.movables_fn(particles),
                         pi.IKTol.from_checker(default_checker()))
        monkeypatch.undo()
        parked = _com_frac(real_world, q[ok]).abs().median(0).values
        assert float(parked.min()) >= 0.8, (kind, parked.tolist())
        assert bool((centred < parked).all()), (kind, centred.tolist(), parked.tolist())
        if kind == "place":
            assert float(centred.max()) <= CENTRED, centred.tolist()


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
    # the bound (roll 1.0 rad) pass, 0.01 rad past it fails; route A's envelope, which sets no bound, passes every
    # conf.
    from types import SimpleNamespace
    from cutamp.gn_refresh import tilt_within_bound
    names = list(REAL_LEGS.joint_names)
    q = torch.tensor(REAL_LEGS.home).repeat(3, 1)
    q[1:, names.index("base_j_roll")] = torch.tensor([1.0, 1.01])
    assert tilt_within_bound(SimpleNamespace(model=TILTED), q, 1e-6).tolist() == [True, True, False]
    assert tilt_within_bound(SimpleNamespace(model=REAL_LEGS), q, 1e-6).all()


@needs_cuda
def test_final_repair_takes_only_accepted_repairs(real_world, real_plan, monkeypatch):
    # The final pre-check repair re-solves the rows the check rejects and takes GN IK's result only where the check
    # rejected the stored conf and accepts the repair. Two accepted first-pick rows miss their hand once a shoulder
    # moves 0.3 rad; the stubbed GN IK returns the init's confs, accepted on one of them and on a third row, still
    # accepted, whose shoulder it moves 0.1 rad: the first takes its repair, the other two keep their stored confs,
    # and so does every other row.
    import cutamp.optimize_plan as op
    from cutamp.config import TAMPConfiguration
    from cutamp.tests.conftest import make_optimizer
    skeleton, particles = real_plan
    particles = clone_particles(particles)
    arm, conf, _ = _first_pick(real_world, real_plan)
    a, b, c = _accepted(real_world, real_plan, conf).nonzero().flatten()[:3].tolist()
    q = particles[conf].detach().clone()
    j = list(REAL_LEGS.joint_names).index(f"{arm.title()}_Shoulder_Pitch")
    with torch.no_grad():
        particles[conf][[a, b], j] += 0.3
    moved = particles[conf].detach().clone()
    calls = []

    def gn_ik(world, targets, arms, movables, tol, q_start=None):
        calls.append(q_start.clone())
        ok = torch.zeros(len(q), dtype=torch.bool, device=q.device)
        ok[[b, c]] = True
        q_ret = q.clone()
        q_ret[c, j] += 0.1
        return q_ret, ok

    monkeypatch.setattr(op, "gn_ik", gn_ik)
    deps = [dep for dep in op._build_ik_deps(skeleton, real_world) if dep.q_name == conf]
    make_optimizer(TAMPConfiguration(num_particles=64))._refresh_ik_deps(particles, deps, real_world)
    assert len(calls) == 1 and torch.equal(calls[0], moved)  # GN IK starts from the stored confs
    want = moved.clone()
    want[b] = q[b]
    assert torch.equal(particles[conf], want)


@needs_cuda
def test_ik_check_rejects_a_conf_past_the_tilt_bound(real_world, real_plan, monkeypatch):
    # GN IK and the final repair accept a conf only inside the envelope's tilt bound, on the Motion value's metric:
    # the most tilted accepted first-pick row passes under a bound 0.01 rad above its tilt and fails under one 0.01 rad
    # below it, the conf unchanged.
    import dataclasses
    from cutamp.optimize_plan import _build_ik_deps
    from cutamp.particle_initialization import IKTol, ik_check
    from cutamp.tests.conftest import default_checker
    skeleton, particles = real_plan
    _, conf, q = _first_pick(real_world, real_plan)
    row = int(_tilt(q).argmax())
    tilt = float(_tilt(q[row:row + 1]))
    assert tilt > 0.02, tilt
    dep = next(dep for dep in _build_ik_deps(skeleton, real_world) if dep.q_name == conf)
    rows = _accepted(real_world, real_plan, conf).nonzero().flatten()[row:row + 1]
    target, movables = dep.target_fn(particles)[rows], dep.movables_fn(particles)[rows]
    tol = IKTol.from_checker(default_checker())

    def ok_under(tilt_max):
        model = dataclasses.replace(REAL_LEGS, envelope=REAL_LEGS.envelope._replace(trunk_tilt_max=tilt_max))
        monkeypatch.setattr(real_world, "robot_container", dataclasses.replace(real_world.robot_container, model=model))
        return bool(ik_check(real_world, q[row:row + 1], target, dep.arm, movables, tol)[0])

    assert ok_under(tilt + 0.01) and not ok_under(tilt - 0.01)


def _spheres(world, q):
    """``[B, S, 4]``: the robot's collision spheres at ``q``, cloned: the next FK call rewrites cuRobo's buffers."""
    from curobo.types import JointState
    return world.kinematics.compute_kinematics(JointState.from_position(q.contiguous())).robot_spheres.reshape(
        len(q), -1, 4).clone()


def _self_collision(world, q):
    """``[B]``: cuRobo's self-collision kernel at ``q``, built and called as the cost function's ``self_collision``."""
    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
    cost = SelfCollisionCost(SelfCollisionCostCfg(
        weight=world.device_cfg.to_device([1.0]), device_cfg=world.device_cfg,
        self_collision_kin_config=world.kinematics.get_self_collision_config()))
    cost.setup_batch_tensors(len(q), 1)
    return cost.forward(_spheres(world, q).unsqueeze(1).contiguous()).reshape(-1)


@needs_cuda
def test_collision_penetration_has_the_checkers_zero_set(real_world):
    # GN's collision row is the summed penetration with the scene, the movables and the robot's own sphere pairs, the
    # self term a torch sum over the kernel's pairs. On 3072 noisy stance-rooted confs of both arms it is positive
    # exactly where the scene's collision_fn or cuRobo's self-collision kernel is.
    from cutamp.gn_refresh import collision_penetration
    from cutamp.particle_initialization import _noisy_start
    torch.manual_seed(0)
    q = torch.cat([_noisy_start(real_world, 512, arm, s) for arm in ("left", "right") for s in (0.3, 0.6, 1.0)])
    spheres = _spheres(real_world, q)
    scene = real_world.collision_fn(spheres.unsqueeze(1).contiguous()).reshape(-1)
    hit = (scene > 0) | (_self_collision(real_world, q) > 0)
    assert 0.05 < float(hit.float().mean()) < 0.95, float(hit.float().mean())  # not vacuous
    assert torch.equal(collision_penetration(real_world, spheres) > 0, hit)


@needs_cuda
def test_collision_row_clears_contact_holding_the_hand(real_world, real_plan):
    # Noisy stance-rooted confs that stand balanced in self-contact, each aiming at its own hand pose, so only the
    # collision row has work to do: GN IK's clearing iterations take every row out of contact (cuRobo's kernel) with
    # the hand, the soles and the CoM held. Without the row they all stay in contact.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project_balanced
    from cutamp.particle_initialization import GN_IK_CLEAR_ITERS, _noisy_start
    arm = _first_pick(real_world, real_plan)[0]
    torch.manual_seed(0)
    q = _noisy_start(real_world, 1024, arm, 0.3)
    standing = (_sole_errors(real_world, q)[0].amax(-1) <= 1e-4) & compute_com_polygon_mask(real_world, q)
    q0 = q[standing & (_self_collision(real_world, q) > 0)][:64]
    assert len(q0) >= 32, len(q0)
    targets = _poses(real_world, q0, real_world.tool_frame_for_arm(arm))
    q = gn_project_balanced(real_world, q0, targets, arm, iters=GN_IK_CLEAR_ITERS)
    assert not bool((_self_collision(real_world, q) > 0).any())
    from cutamp.costs import curobo_pose_error
    pos, rot = curobo_pose_error(_poses(real_world, q, real_world.tool_frame_for_arm(arm)), targets)
    assert float(pos.max()) <= POS_TOL and float(rot.max()) <= ROT_TOL
    _assert_soles_held(real_world, q)
    assert bool(compute_com_polygon_mask(real_world, q).all())
    blind = gn_project_balanced(real_world, q0, targets, arm, iters=GN_IK_CLEAR_ITERS, collision=False)
    assert bool((_self_collision(real_world, blind) > 0).all())


@needs_cuda
def test_gn_ik_accepts_only_confs_that_pass_the_checks(real_world, real_plan):
    # The init's GN IK confs at the first pick and the first place: every row the IK accepts passes the checks on
    # their own, at the checker's tolerances: the hand, both soles, the CoM in the keyframe rectangle, cuRobo's
    # self-collision kernel, the scene, and the movables at the checker's zero activation distance. The IK accepts at
    # least 48 first-pick rows (64 on both inits, 2026-11-06) and some first-place rows (42 on both).
    from cutamp.com_polygon_cost import SOLE_POS_TOL, SOLE_ROT_TOL, compute_com_polygon_mask
    from cutamp.costs import curobo_pose_error, sphere_to_sphere_overlap
    from cutamp.optimize_plan import _build_ik_deps
    skeleton, particles = real_plan
    deps = {dep.q_name: dep for dep in _build_ik_deps(skeleton, real_world)}
    for kind, least in (("pick", 48), ("place", 1)):
        conf = next(op for op in skeleton if op.operator.metadata.action_type == kind).values[-1]
        dep, q = deps[conf], particles[conf]
        pos, rot = curobo_pose_error(_poses(real_world, q, real_world.tool_frame_for_arm(dep.arm)),
                                     dep.target_fn(particles))[:2]
        sole_pos, sole_rot = _sole_errors(real_world, q)
        spheres = _spheres(real_world, q).unsqueeze(1).contiguous()
        movables = sphere_to_sphere_overlap(spheres, dep.movables_fn(particles).unsqueeze(1), activation_distance=0.0)
        passes = ((pos <= POS_TOL) & (rot <= ROT_TOL)
                  & (sole_pos.amax(-1) <= SOLE_POS_TOL) & (sole_rot.amax(-1) <= SOLE_ROT_TOL)
                  & compute_com_polygon_mask(real_world, q) & (_self_collision(real_world, q) <= 0)
                  & (real_world.collision_fn(spheres).reshape(-1) <= 0) & (movables.reshape(-1) <= 0))
        accepted = _accepted(real_world, real_plan, conf)
        assert int(accepted.sum()) >= least, (kind, int(accepted.sum()))
        assert not bool((accepted & ~passes).any()), (kind, int((accepted & ~passes).sum()))


@needs_cuda
def test_movable_spheres_are_where_the_rollout_puts_the_objects(real_world, real_plan, monkeypatch):
    # At every pick and place conf, the movables GN checks (movable_spheres_fn) are the cost function's
    # robot_to_movables spheres at that conf: the rollout's object poses at its timestep. TF32 off: the loop's TF32
    # matmuls round the two transforms' shapes differently, up to 0.12 mm.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    from cutamp.config import TAMPConfiguration
    from cutamp.rollout import RolloutFunction, movable_spheres_fn
    from cutamp.utils.common import transform_spheres
    skeleton, particles = real_plan
    rollout = RolloutFunction(skeleton, real_world, TAMPConfiguration(num_particles=64))(particles)
    spheres = torch.cat([transform_spheres(real_world.get_collision_spheres(obj), rollout["obj_to_pose"][obj.name])
                         for obj in real_world.movables], dim=2)
    ops = [(idx, op) for idx, op in enumerate(skeleton) if not op.operator.metadata.is_motion]
    assert any(op.operator.metadata.action_type == "place" for _, op in ops)
    for ts, (idx, _) in enumerate(ops):
        torch.testing.assert_close(movable_spheres_fn(real_world, skeleton, idx)(particles),
                                   spheres[:, rollout["ts_to_pose_ts"][ts]], rtol=0, atol=1e-6)


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
    config = TAMPConfiguration(num_particles=64)
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
