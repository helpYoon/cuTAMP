from dataclasses import fields
from types import SimpleNamespace

import torch

from cutamp.com_polygon_cost import SOLE_POS_TOL, SOLE_ROT_TOL
from cutamp.conf_locking import get_conf_locked_dofs
from cutamp.config import TAMPConfiguration
from cutamp.cost_reduction import CostReducer
from cutamp.robots.t1 import REAL_LEGS
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tamp_world import FK_SITES


def test_a_stepped_plan_overrides_two_settings_only():
    base = TAMPConfiguration()
    stepped = base.for_steps()
    assert stepped.conf_lr == 7e-3 and stepped.soft_cost == []
    changed = {f.name for f in fields(base) if getattr(base, f.name) != getattr(stepped, f.name)}
    assert changed == {"conf_lr", "soft_cost"}


def test_a_stepped_retract_locks_only_its_idle_arm():
    op = SimpleNamespace(values=("q_start", "traj", "q_r"),
                         operator=SimpleNamespace(metadata=SimpleNamespace(arm="left", action_type="retract")))
    idle = list(range(REAL_LEGS.right_arm.start, REAL_LEGS.right_arm.stop))
    assert get_conf_locked_dofs([op], "q_r", REAL_LEGS, stepped=True) == idle
    assert set(get_conf_locked_dofs([op], "q_r", REAL_LEGS)) > set(idle)


def test_the_new_rows_have_settings():
    mult, tol = default_constraint_to_mult, default_constraint_to_tol
    assert tol["ComComfort"]["margin_shortfall"] == 0.0
    assert tol["KinematicConstraint"]["retract_sole_pos_err"] == SOLE_POS_TOL
    assert tol["KinematicConstraint"]["retract_sole_rot_err"] == SOLE_ROT_TOL
    assert tol["StepCommand"]["default"] == 0.0 and tol["StepTests"]["refresh_tests"] == 0.0
    assert "stay_near_start" not in mult["soft"]
    assert {"stepped_scene", "retract_soles"} <= set(FK_SITES)


def test_a_failed_refresh_test_adds_nothing_to_the_loss():
    cost = {"StepTests": {"type": "constraint", "values": {"refresh_tests": torch.ones(4)}}}
    reducer = CostReducer(default_constraint_to_mult.copy())
    assert (reducer(cost) == 0).all()


import tempfile
from time import monotonic

import pytest

from cutamp import algorithm as alg
from cutamp import held
from cutamp.algorithm import run_cutamp
from cutamp.constraint_checker import ConstraintChecker
from cutamp.particle_initialization import ParticleInitializer
from cutamp.step_loop import SteppedPlan
from cutamp.step_planner import StepResult
from cutamp.step_refresh import NoStartingPath, as_commanded, command_rows
from cutamp.tests.conftest import first_schedules, needs_cuda, only_schedule
from cutamp.tests.paired_grasp_scene import paired_setup


@pytest.fixture(scope="module")
def stepped_call():
    """What the planner hands the stepped route for the paired floor pick: one left step before the pick, none before
    the place."""
    got = {}

    def capture(proposal, world, config, reducer, checker, **kw):
        got.update(proposal=proposal, world=world, config=config, reducer=reducer, checker=checker, **kw)
        return StepResult([], None, "unresolved", "captured")

    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as out:
        env, q, inputs = paired_setup("near", mp)
        only_schedule(mp, ("L", "none"), "left")
        first_schedules(mp, 1)
        mp.setattr(alg, "_refine_step_schedule", capture)
        torch.manual_seed(641)
        cfg = TAMPConfiguration(enable_visualizer=False, curobo_plan=False, experiment_root=out, num_particles=16)
        run_cutamp(env, cfg, CostReducer(default_constraint_to_mult.copy()),
                   ConstraintChecker(default_constraint_to_tol.copy()), q_init=q, step_inputs=inputs,
                   experiment_id="capture")
    got["config"] = got["config"].for_steps()
    got["pools"] = ParticleInitializer(got["world"], got["config"], got["checker"])._pick_pools(got["proposal"].skeleton)
    return got


def make_plan(c, **kw):
    return SteppedPlan(c["proposal"], c["world"], c["config"], c["predictor"], c["inputs"], c["pools"], c["checker"],
                       reducer=c["reducer"], generator=torch.Generator(device="cuda").manual_seed(0), **kw)


@needs_cuda
def test_starting_values(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    assert plan.orders == ("L", "") and plan.command_names == {0: "step_commands_0"}
    u = p["step_commands_0"]
    assert u.shape == (16, 1, 3)
    proposal = stepped_call["proposal"]
    assert len(proposal.targets) == len(proposal.schedule.edges) == len(proposal.orders) and proposal.search is not None
    assert plan.stats["draw"]["reached"] and plan.stats["draw"]["seconds"] > 0
    after, _ = as_commanded(plan.orders, plan.incoming, [u, None])
    assert all((v == 0).all() for v in command_rows(plan.orders, plan.incoming, [u, None], after).values())
    (land,) = plan.landing_confs
    start, commands, prediction, _ = plan.last[0]
    torch.testing.assert_close(p[land], plan.adapter.planner(prediction.steps[-1].terminal.tensors[:6])[0])
    standin_after, final = plan.standin.stances(plan.incoming, [u, None])
    torch.testing.assert_close(standin_after[0][:, -1], prediction.steps[-1].achieved, rtol=0, atol=1e-5)
    torch.testing.assert_close(final[1], final[0])
    assert plan.tests_ok.shape == (16,) and plan.tests_ok.dtype == torch.bool
    assert all(torch.isfinite(v).all() for v in plan.start_postures.values()) and len(plan.start_postures) >= 3
    assert not any(k.endswith("_ik_reference") for k in p) and len(plan.references) == 2


@needs_cuda
def test_a_refresh_follows_the_commands(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    (land,) = plan.landing_confs
    kept, before = p[land], p[land].clone()
    with torch.no_grad():
        p["step_commands_0"][..., 0] += 0.005
    plan.refresh(p)
    assert p[land] is kept and not torch.equal(p[land], before)
    assert torch.equal(plan.standin.at[0], p["step_commands_0"])
    assert len(plan.stats["refresh_s"]) == len(plan.stats["standin_error_m"]) == 2  # the first refresh, then this one


@needs_cuda
def test_an_expired_budget_stops_the_start(stepped_call):
    with pytest.raises(TimeoutError):
        make_plan(stepped_call, deadline=monotonic() - 1.0)


@needs_cuda
def test_a_diverged_row_leaves_its_neighbours_alone(stepped_call):
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    a = make_plan(stepped_call)
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state(rng[1])
    b = make_plan(stepped_call)  # the same draws
    assert all(torch.equal(v, b.particles[k]) for k, v in a.particles.items())
    a.refresh(a.particles)
    state = b._first.state  # the first gap's cached hold, reused by every refresh
    state.tensors = tuple(t.clone() for t in state.tensors)  # the observed state's rows are expanded views
    state.tensors[0][3] = state.tensors[4][3] = float("nan")  # body position and joints of row 3
    b.refresh(b.particles)
    others = torch.arange(16, device="cuda") != 3
    assert not b.tests_ok[3] and torch.equal(b.tests_ok[others], a.tests_ok[others])
    assert all(torch.equal(v[others], a.particles[k][others]) for k, v in b.particles.items())
    # Row 3 stands on a pair that is not finite: its postures stay as they were.
    assert all(torch.equal(b.particles[k][3], start[3]) for k, start in b.start_postures.items())


from cutamp.com_polygon_cost import physical_support_margin
from cutamp.optimize_plan import ADAM_CONSIDER_TYPES


@needs_cuda
def test_the_cost_reads_the_standin(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    for name in ("step_commands_0", *plan.start_postures):
        p[name].requires_grad_()
    cost = plan(plan.rollout_fn(p), p)
    assert set(cost["ComComfort"]["values"]) == {"margin_shortfall"}
    assert {"StepCommand", "StepTests"} <= set(cost)
    assert not set(cost["ComPolygon"]["values"]) & plan.landing_confs
    assert cost["KinematicConstraint"]["values"]["retract_sole_pos_err"].shape[0] == 16
    assert torch.equal(cost["StepTests"]["values"]["refresh_tests"] > 0, ~plan.tests_ok)
    stepped_call["reducer"](cost, consider_types=ADAM_CONSIDER_TYPES).sum().backward()
    g = p["step_commands_0"].grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


@needs_cuda
def test_the_balance_cost_is_the_shortfall_itself(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    with torch.no_grad():
        rollout = plan.rollout_fn(p)
        cost = plan(rollout, p)
        table = plan.stance_table(p)
        stances = torch.stack([table[c] for c in rollout["conf_params"]], 1)
        margin = physical_support_margin(rollout["com_xy"], stances, stepped_call["world"].model.sole_box)
    expected = (stepped_call["config"].com_comfort_margin_m - margin).clamp_min(0)
    torch.testing.assert_close(cost["ComComfort"]["values"]["margin_shortfall"], expected)


@needs_cuda
def test_the_preferences_in_the_loss(stepped_call):
    from cutamp.com_polygon_cost import physical_support_centroid
    from cutamp.stance import UNIT_M, UNIT_RAD
    from cutamp.step_costs import joint_effort, plan_rank, side_lean
    plan = make_plan(stepped_call)
    p, world = plan.particles, plan.world
    with torch.no_grad():
        rollout = plan.rollout(p)
        cost = plan(rollout, p)
        soft = cost["soft"]
        table = plan.stance_table(p)
        names = rollout["conf_params"]
        q = torch.stack([p[name] for name in names], 1)
        stances = torch.stack([table[name] for name in names], 1)
        lean = side_lean(q, stances, world.model)
        distance = (rollout["com_xy"] - physical_support_centroid(stances, world.model.sole_box)).norm(dim=-1)
        effort = joint_effort(world, q)
    assert set(soft["values"]) == {"side_lean", "off_centre", "joint_effort"}
    torch.testing.assert_close(soft["values"]["side_lean"], (lean / UNIT_RAD).square().sum(1).to(soft["values"]["side_lean"]))
    torch.testing.assert_close(soft["values"]["off_centre"], (distance / UNIT_M).sum(1).to(soft["values"]["off_centre"]))
    torch.testing.assert_close(soft["values"]["joint_effort"], effort.sum(1).to(soft["values"]["joint_effort"]))
    torch.testing.assert_close(soft["rank"], plan_rank(lean.abs().asin().amax(1), distance.amax(1)).to(soft["rank"]))
    torch.testing.assert_close(soft["effort"], effort.amax(1).to(soft["effort"]))


@needs_cuda
def test_joint_effort_has_a_slope(stepped_call):
    from cutamp.step_costs import joint_effort
    plan = make_plan(stepped_call)
    name = plan.groups[0][1].arg("q")
    q = plan.particles[name].detach()[:2, None].clone().requires_grad_(True)      # [2, 1, dof]
    (grad,) = torch.autograd.grad(joint_effort(plan.world, q).sum(), q)
    knee = list(plan.world.model.joint_names).index("Left_Knee_Pitch")
    step = torch.zeros_like(q)
    step[..., knee] = 1e-3
    with torch.no_grad():
        measured = (joint_effort(plan.world, q + step) - joint_effort(plan.world, q - step)) / 2e-3
    torch.testing.assert_close(grad[..., knee], measured, rtol=1e-3, atol=1e-5)
    assert grad.isfinite().all()


@needs_cuda
def test_result_rows(stepped_call):
    from cutamp.step_costs import plan_rank
    plan = make_plan(stepped_call)
    res = plan.result(plan.particles, stepped_call["reducer"], stepped_call["checker"])
    assert res.endpoint_ok.shape == (16, 2) and res.status in ("model_feasible", "unresolved")
    assert set(plan.motion_inputs) == {row.index for row in res.rows}
    assert res.rows
    for row in res.rows:
        m = row.metrics
        assert len(m["com_center_distances_m"]) == len(m["com_margins_m"]) == 2
        assert row.max_center_distance_m == max(m["com_center_distances_m"])
        pick, place = m["gaps"]
        assert pick["order"] == "L" and place["order"] == "none" and pick["achieved"] == place["achieved"]
        assert len(pick["commands"]) == 1 and len(pick["final_xy_error_m"]) == 2
        assert sum(k.endswith("_ik_reference") for k in row.particle) == 2
        assert {"rank", "joint_effort", "side_lean_rad", "joint_efforts"} <= set(m)
        assert len(m["side_lean_rad"]) == len(m["joint_efforts"]) == 2
        assert m["joint_effort"] == max(m["joint_efforts"])
        lean, centre = max(abs(x) for x in m["side_lean_rad"]), max(m["com_center_distances_m"])
        assert m["rank"] == float(plan_rank(torch.tensor(lean), torch.tensor(centre)))


def test_valid_rows_go_to_arm_motion_by_rank_then_effort():
    from cutamp.step_loop import _validate_motion_rows
    from cutamp.step_planner import StepRow
    rows = [StepRow(i, {}, 0.0, {"rank": rank, "joint_effort": effort, "com_center_distances_m": [centre],
                                "secondary_effort": 0.0})
            for i, (rank, effort, centre) in enumerate([(4096.0, 0.1, 0.0), (0.0, 0.3, 0.004), (0.0, 0.2, 0.004)])]
    tried = []
    def attempt(row):
        tried.append(row.index)
        return None, {"motion_validation": "failed"}
    config = SimpleNamespace(com_center_weight=1.0, motion_plan_max_retries=3)
    _validate_motion_rows(rows, {}, [], None, config, attempt, deadline=None)
    assert tried == [2, 1, 0] and rows[1].order == (0.0, 0.3)


from cutamp.step_loop import refine_stepped_plan


@needs_cuda
def test_expired_budget_is_reported(stepped_call):
    c = stepped_call
    res = refine_stepped_plan(c["proposal"], c["world"], c["config"], c["reducer"], c["checker"], inputs=c["inputs"],
                              predictor=c["predictor"], timer=c["timer"], visualizer=c["visualizer"],
                              deadline=monotonic() - 1.0)
    assert res.status == "budget_exhausted" and res.rows == []


@needs_cuda
def test_a_missing_predictor_is_reported(stepped_call):
    c = stepped_call
    res = refine_stepped_plan(c["proposal"], c["world"], c["config"], c["reducer"], c["checker"], inputs=c["inputs"],
                              predictor=None, timer=c["timer"], visualizer=c["visualizer"], deadline=None)
    assert res.status == "unavailable"


@needs_cuda
def test_a_plan_without_a_starting_path_fails_its_initialization(stepped_call, monkeypatch):
    def refuse(*args, **kwargs):
        raise NoStartingPath("16 of 16 particles found no legal, feet-clear grid path in 1000 rounds")

    monkeypatch.setattr("cutamp.step_loop.draw_grid_paths", refuse)
    c = stepped_call
    res = refine_stepped_plan(c["proposal"], c["world"], c["config"], c["reducer"], c["checker"], inputs=c["inputs"],
                              predictor=c["predictor"], timer=c["timer"], visualizer=c["visualizer"], deadline=None)
    assert res.status == "initialization_failed" and res.rows == []
    assert res.reason == "16 of 16 particles found no legal, feet-clear grid path in 1000 rounds"


@needs_cuda
def test_an_unresolved_result_counts_the_particles_each_requirement_fails(stepped_call):
    plan = make_plan(stepped_call)
    # Grid starts can be feasible; a step 0.5 m forward, beyond the trained 0.33 m, makes every particle fail.
    with torch.no_grad():
        plan.particles["step_commands_0"][..., 0] = 0.5
    res = plan.result(plan.particles, stepped_call["reducer"], stepped_call["checker"])
    assert res.status == "unresolved"
    counts = res.diagnostics["failure_counts"]
    assert set(res.diagnostics["failed_constraints"]) <= set(counts) and all(0 < v <= 16 for v in counts.values())


@needs_cuda
def test_a_refresh_puts_drifted_postures_back_on_the_landing(stepped_call):
    """Adam slid each posture's base 3 cm (as measured in the loop); the refresh puts feet and hand back."""
    plan = make_plan(stepped_call)
    p = plan.particles
    names = list(plan.start_postures)
    kept = {k: p[k] for k in names}
    for k in names:
        p[k].requires_grad_()
    with torch.no_grad():
        for k in names:
            p[k][:, :2] += 0.03  # base x and y
    drifted = plan(plan.rollout_fn(p), p)["KinematicConstraint"]["values"]
    assert drifted["sole_pos_err"].median() > SOLE_POS_TOL
    plan.refresh(p)
    assert all(p[k] is kept[k] and p[k].requires_grad for k in names)
    values = plan(plan.rollout_fn(p), p)["KinematicConstraint"]["values"]
    assert values["sole_pos_err"].median() <= SOLE_POS_TOL
    assert values["retract_sole_pos_err"].median() <= SOLE_POS_TOL
    assert values["pos_err"].median() <= 0.005


@needs_cuda
def test_the_refresh_tests_do_not_compare_the_landing_with_the_command(stepped_call):
    from cutamp.step_loop import landing_constraint_rows
    plan = make_plan(stepped_call)
    start, commands, prediction, _ = plan.last[0]
    feet = plan.predictor.model.feet(start.tensors[:6])[0]
    rows, passed, _ = landing_constraint_rows(plan.predictor, feet, plan.orders[0], commands, prediction)
    assert not any(k.endswith(("precision", "deadline_xy", "deadline_yaw")) for k in rows)
    phases = prediction.steps[0].phases
    assert torch.equal(passed, phases.deadline_ok & phases.hold_ok & phases.support_ok)


@needs_cuda
def test_self_collision_slope_scales_with_its_weight_only_when_asked(stepped_call):
    from curobo.types import JointState
    from cutamp.cost_function import CostFunction
    from cutamp.gn_refresh import _self_pairs
    world = stepped_call["world"]
    plan = make_plan(stepped_call)
    standing = CostFunction(plan.arm_skeleton, world, stepped_call["config"])
    i, j, _ = _self_pairs(world.kinematics)
    spheres = world.kinematics.compute_kinematics(JointState.from_position(world.q_init[None])).robot_spheres
    spheres = spheres.reshape(1, 1, -1, 4).clone()
    spheres[0, 0, i[0], :3] = spheres[0, 0, j[0], :3]      # one checked pair, centre on centre

    def slope(cost_fn, scale):
        s = spheres.clone().requires_grad_(True)
        cost_fn.self_collision_cost_fn.setup_batch_tensors(1, 1)
        value = cost_fn.self_collision_cost_fn.forward(s)
        assert value.sum() > 0
        return torch.autograd.grad((scale * value).sum(), s)[0].abs().sum()

    assert float(slope(plan.cost_function, 1000.0)) == pytest.approx(1000.0 * float(slope(plan.cost_function, 1.0)), rel=1e-4)
    assert float(slope(standing, 1000.0)) == pytest.approx(float(slope(standing, 1.0)), rel=1e-6)


@needs_cuda
def test_a_refresh_puts_postures_back_and_solves_none(stepped_call, monkeypatch):
    plan = make_plan(stepped_call)
    p, calls = plan.particles, []
    monkeypatch.setattr(plan, "_solve_postures", lambda *a, **k: pytest.fail("a refresh solved a posture"))
    run_gap, put_back = plan._run_gap, plan.put_back
    monkeypatch.setattr(plan, "_run_gap", lambda j, *a: (calls.append(("gap", j)), run_gap(j, *a))[1])
    monkeypatch.setattr(plan, "put_back", lambda q, gaps=None: (calls.append(("back", tuple(gaps))), put_back(q, gaps))[1])
    with torch.no_grad():
        p["step_commands_0"][..., 0] += 0.005
        p[plan.groups[0][1].arg("q")][0, :2] += 0.30        # row 0's pick 30 cm off its feet: one correction is capped
    plan.refresh(p)
    assert calls == [("gap", 0), ("back", (0,)), ("back", (1,))]       # each gap's postures before the next gap
    capped = plan.stats["correction_capped"][-1]
    assert "resolve_s" not in plan.stats and type(capped) is int and 1 <= capped <= 16 * len(plan.gap_of)


@needs_cuda
def test_the_postures_follow_the_feet(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    with torch.no_grad():
        table, targets = plan.stance_table(p), plan.hand_targets(p)
        reach = torch.ones(16, dtype=torch.bool, device="cuda")
        # A posture its limits keep off its hand and feet is the checker's to reject, not this test's to measure.
        for name, j in plan.gap_of.items():
            op = plan.groups[j][1]
            hand = (targets[j], op.operator.metadata.arm) if name == op.arg("q") else ()
            reach &= held.held_rows(plan.world, p[name], table[name], *hand)[0].abs().amax(1) < 5e-3
        p["step_commands_0"][..., 1] += 0.005                           # every stance moves 5 mm
        moved = plan(plan.rollout(p), p)["KinematicConstraint"]["values"]
    assert reach.sum() >= 8
    for row in ("pos_err", "sole_pos_err", "retract_sole_pos_err"):
        assert moved[row][reach].amax() < 1e-3
    assert set(plan.held_jac) == set(plan.gap_of)


@needs_cuda
def test_every_posture_on_a_stance_hands_on_its_share(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    for name in ("step_commands_0", *plan.gap_of):
        p[name].requires_grad_()
    plan.rollout(p)
    u = p["step_commands_0"]
    shares = []
    for only in (None, *plan.gap_of):                                   # every posture, then one at a time
        u.grad = torch.zeros_like(u)
        for name in plan.gap_of:
            p[name].grad = torch.ones_like(p[name]) if only in (None, name) else torch.zeros_like(p[name])
        plan.hand_on(p)
        shares.append(u.grad.clone())
        if only is None:
            for name in plan.gap_of:
                assert (plan.held_jac[name] @ p[name].grad.double()[..., None]).abs().amax() < 1e-4
    torch.testing.assert_close(shares[0], sum(shares[1:]), atol=1e-5, rtol=1e-4)
    assert all(s.isfinite().all() and s.abs().sum() > 0 for s in shares[1:])   # the place has no step of its own


@needs_cuda
def test_the_placements_share_is_the_slope_along_the_held_surface(stepped_call):
    """The place posture's gradient reaches the placement through hand_on as the slope of the cost along the held
    surface, measured by putting the posture back on the hand target of a moved placement: its x, y and z move the
    hand's position rows, its yaw the rotation rows too."""
    plan = make_plan(stepped_call)
    p = plan.particles
    op = plan.groups[1][1]
    name, key, arm = op.arg("q"), op.arg("placement"), op.operator.metadata.arm
    for k in ("step_commands_0", key, *plan.gap_of):
        p[k].requires_grad_()
    plan.rollout(p)                                                     # every posture put back, its J kept
    w = torch.randn(p[name].shape[1], generator=torch.Generator().manual_seed(3)).cuda()
    for k in ("step_commands_0", key, *plan.gap_of):
        p[k].grad = w.expand_as(p[k]).clone() if k == name else torch.zeros_like(p[k])
    plan.hand_on(p)
    share = p[key].grad.clone()
    d = 1e-3
    with torch.no_grad():
        stance, q0 = plan.stance_table(p)[name], p[name].detach().clone()

        def put_back_at(c, by):
            moved = {**p, key: p[key].detach().clone()}
            moved[key][:, c] += by
            return held.put_back(plan.world, q0, stance, plan.hand_targets(moved)[1], arm)[0]

        for c in range(4):
            measured = ((put_back_at(c, d) - put_back_at(c, -d)) @ w) / (2 * d)
            error = (measured - share[:, c]).abs() / (share[:, c].abs() + 1e-3 * w.norm())
            assert error.median() < 0.05, (c, error)


@needs_cuda
def test_a_placement_stays_on_its_support(stepped_call):
    from torch.optim import Adam
    from cutamp.samplers import project_supported_placement
    plan = make_plan(stepped_call)
    p, world = plan.particles, plan.world
    op = plan.groups[1][1]
    key = op.arg("placement")
    for name in (key, "step_commands_0", *plan.gap_of):
        p[name].requires_grad_()
    plan.rollout(p)
    for name in (key, "step_commands_0", *plan.gap_of):
        p[name].grad = torch.zeros_like(p[name])
    p[key].grad[:, 2] = 1.0                                             # pushed down, off its support
    params = [p[name] for name in (key, "step_commands_0", *plan.gap_of)]
    plan.step(Adam(params, lr=0.05), params, p)
    on = project_supported_placement(p[key].detach(), world.get_collision_spheres(op.arg("obj")),
                                     world.get_aabb(op.arg("surface")))
    torch.testing.assert_close(p[key].detach(), on)


@needs_cuda
def test_the_hand_and_feet_rows_leave_the_loss(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    with torch.no_grad():
        cost = plan(plan.rollout(p), p)
        before = stepped_call["reducer"](cost)
        for row in cost["KinematicConstraint"]["values"].values():
            row += 1.0
    assert set(cost["KinematicConstraint"]["weights"]) == set(cost["KinematicConstraint"]["values"])
    torch.testing.assert_close(stepped_call["reducer"](cost), before)


@needs_cuda
def test_the_loss_weighs_each_requirement_per_particle(stepped_call):
    from cutamp.stance import UNIT_M
    plan = make_plan(stepped_call)
    p = plan.particles
    with torch.no_grad():
        cost = plan(plan.rollout(p), p)
    weights = cost["ComComfort"]["weights"]["margin_shortfall"]
    assert weights.shape == cost["ComComfort"]["values"]["margin_shortfall"].shape and (weights == 1 / UNIT_M).all()
    assert all((w == 0).all() for w in cost["KinematicConstraint"]["weights"].values())
    short = cost["ComComfort"]["values"]["margin_shortfall"] > 0
    plan.refresh(p)
    with torch.no_grad():
        again = plan(plan.rollout(p), p)["ComComfort"]["weights"]["margin_shortfall"]
    assert torch.equal(again > 1 / UNIT_M, short)


@needs_cuda
def test_the_cost_at_start_weights_stays_while_the_weights_double(stepped_call):
    """Two refreshes double every violated row's weight while the values stay: the total the countdown reads stays,
    the loss's total rises."""
    from cutamp.step_costs import GROWTH
    plan, reducer = make_plan(stepped_call), stepped_call["reducer"]
    p = plan.particles
    with torch.no_grad():
        rollout = plan.rollout(p)
        cost = plan(rollout, p)
        at_start, total = cost["soft"]["cost_at_start_weights"], reducer(cost)
        for _ in range(2):
            plan.weights.grow()
            for line in plan.checks.values():
                line["weight"] = torch.where(line["value"] > 0, line["weight"] * GROWTH, line["weight"])
        again = plan(rollout, p)
        grown = reducer(again)
    torch.testing.assert_close(at_start, total, equal_nan=True)     # every weight is at its start at the first call
    assert not at_start.requires_grad and at_start.shape == (16,)
    torch.testing.assert_close(again["soft"]["cost_at_start_weights"], at_start, rtol=0, atol=0, equal_nan=True)
    finite = total.isfinite()
    assert (grown[finite] >= total[finite]).all() and (grown[finite] > total[finite]).any()


@needs_cuda
def test_the_checks_are_a_straight_line_in_the_commands(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    line = plan.checks[0]
    assert line["value"].shape == line["weight"].shape == (16, len(line["names"]))
    assert line["slope"].shape == (16, len(line["names"]), 3) and torch.equal(line["at"], plan.standin.at[0])
    assert any(name.startswith("geometry_") for name in line["names"])
    assert not any(name.endswith(("travel", "bound", "precision", "deadline_xy")) for name in line["names"])
    u = p["step_commands_0"].requires_grad_()
    cost = plan(plan.rollout(p), p)
    torch.testing.assert_close(cost["StepChecks"]["values"]["lines"],
                               (line["weight"] * line["value"].clamp_min(0)).sum(1).to(u), atol=1e-6, rtol=1e-5)
    k = 0
    line["value"][:], line["slope"][:] = -1.0, 0.0                  # every other check met, with no slope
    line["value"][:, k], line["slope"][:, k] = 1.0, torch.tensor([2.0, 0.0, 0.0], device=u.device)
    cost = plan(plan.rollout(p), p)
    (grad,) = torch.autograd.grad(cost["StepChecks"]["values"]["lines"].sum(), u)
    torch.testing.assert_close(grad[:, 0, 0], 2.0 * line["weight"][:, k].to(grad))


@needs_cuda
def test_a_checks_weight_doubles_while_it_fails(stepped_call, monkeypatch):
    from cutamp import step_loop
    from cutamp.step_costs import GROWTH, WEIGHT_CAP
    plan = make_plan(stepped_call)
    weight = plan.checks[0]["weight"]
    assert (weight == 1.0).all()
    k, read = 0, step_loop.lines

    def failing(*args):                                             # check k fails on every particle, the rest pass
        value, slope = read(*args)
        value[:], value[:, k] = -1.0, 1.0
        return value, slope

    monkeypatch.setattr(step_loop, "lines", failing)
    weight[0, k], weight[1, k] = WEIGHT_CAP, 3.0
    before = weight.clone()
    plan.refresh(plan.particles)
    after = plan.checks[0]["weight"]
    assert after[0, k] == WEIGHT_CAP and after[1, k] == 3.0 * GROWTH and (after[2:, k] == GROWTH).all()
    others = torch.arange(after.shape[1], device=after.device) != k
    assert torch.equal(after[:, others], before[:, others])
    assert len(plan.stats["check_line_error"]) == 1


def test_clearances_are_read_on_the_copies_of_violating_particles_only():
    from cutamp.step_loop import _clearances
    n, blocks = 3, 4
    row = torch.arange(blocks * n, dtype=torch.float64)
    table = {"a": 5.0 + row, "b": 100.0 + row}                               # every row of the rollout its own values
    table["a"][:n] = torch.tensor([1e-6, 2e-6, 0.0], dtype=torch.float64)   # the verdict accepts 1e-6: only particle 1
    table["b"][:n] = 0.0
    verdict = torch.tensor([True, False, True] + [False] * (blocks - 1) * n)
    calls = []

    def check(rows):
        calls.append(rows.tolist())
        return {"valid": verdict[rows], "steps": [], "penetration": {k: v[rows] for k, v in table.items()},
                "load_status": "unloaded"}

    valid, pen = _clearances(check, n, blocks, "cpu")
    assert calls == [[0, 1, 2], [4, 7, 10]]                      # the own rows, then particle 1's three copies
    assert valid.tolist() == [True, False, True]
    assert pen["a"].reshape(blocks, n).tolist() == [[1e-6, 2e-6, 0.0], [1e-6, 9.0, 0.0], [1e-6, 12.0, 0.0],
                                                    [1e-6, 15.0, 0.0]]     # rows 4, 7 and 10 where they belong
    assert pen["b"].reshape(blocks, n).tolist() == [[0.0, 0.0, 0.0], [0.0, 104.0, 0.0], [0.0, 107.0, 0.0],
                                                    [0.0, 110.0, 0.0]]
    table["a"][1] = 0.0
    calls.clear()
    _clearances(check, n, blocks, "cpu")
    assert calls == [[0, 1, 2]]                                  # nothing violated: the own rows only


@needs_cuda
def test_a_refresh_checks_the_copies_of_violating_particles_only(stepped_call, monkeypatch):
    from cutamp import step_loop
    plan = make_plan(stepped_call)
    p, n = plan.particles, 16
    _, objects, _, held = step_loop._placements_and_grasps(plan.groups, plan.world, plan.pools, 0, n, bindings=p,
                                                           grasp_indices=p)
    mark = torch.arange(n, device=p["left_q0"].device) * 1e-4                   # each particle's scene is its own
    scene = {name: pose.clone() for name, pose in objects[0].items()}
    carried = {arm: (name, pose.clone()) for arm, (name, pose) in held[1].items()}   # as if the pick's object is held
    for pose in (*scene.values(), *(pose for _, pose in carried.values())):
        pose[:, 0, 3] += mark
    rollout, calls = {}, []
    refresh_gap, geometry = step_loop.refresh_gap, step_loop.validate_step_geometry

    def gap(*args, **kw):
        out = refresh_gap(*args, **kw)
        rollout["all"] = out.copies
        return out

    def check(world, adapter, prediction, *, object_poses, held):
        out = geometry(world, adapter, prediction, object_poses=object_poses, held=held)
        if not calls:                                           # particles 2 and 5 violate a clearance, the rest none
            for value in out["penetration"].values():
                value.zero_()
            next(iter(out["penetration"].values()))[[2, 5]] = 1.0
        calls.append((prediction.intended_final, object_poses, held))
        return out

    monkeypatch.setattr(step_loop, "refresh_gap", gap)
    monkeypatch.setattr(step_loop, "validate_step_geometry", check)
    plan._run_gap(0, p, None, plan.incoming, scene, carried)
    rows = torch.tensor([18, 21, 34, 37, 50, 53], device=mark.device)           # their copies, block by block
    assert [len(intended) for intended, _, _ in calls] == [n, len(rows)]
    for (intended, poses, hand), chosen in zip(calls, (torch.arange(n, device=mark.device), rows)):
        assert torch.equal(intended, rollout["all"].intended_final[chosen])
        assert all(torch.equal(poses[name], scene[name][chosen % n]) for name in scene)
        assert all(torch.equal(hand[arm][1], carried[arm][1][chosen % n]) for arm in carried)
    line = plan.checks[0]
    geometry_checks = [k for k, name in enumerate(line["names"]) if name.startswith("geometry_")]
    others = [i for i in range(n) if i not in (2, 5)]
    assert (line["slope"][others][:, geometry_checks] == 0).all()               # no violation, no slope


@needs_cuda
def test_a_particle_the_geometry_check_fails_fails_its_refresh(stepped_call, monkeypatch):
    from cutamp import step_loop
    plan = make_plan(stepped_call)
    p, n = plan.particles, 16
    _, objects, _, held = step_loop._placements_and_grasps(plan.groups, plan.world, plan.pools, 0, n, bindings=p,
                                                           grasp_indices=p)
    passed = plan._run_gap(0, p, None, plan.incoming, objects[0], held[0])[0]
    assert passed.any()
    row, real = int(passed.nonzero()[0]), step_loop.validate_step_geometry

    def failing(*args, **kw):
        out = real(*args, **kw)
        if len(out["valid"]) == n:                               # the own rows' verdict; a copies call has 3 k rows
            out["valid"][row] = False
        return out

    monkeypatch.setattr(step_loop, "validate_step_geometry", failing)
    again = plan._run_gap(0, p, None, plan.incoming, objects[0], held[0])[0]
    others = torch.arange(n, device=passed.device) != row
    assert not again[row] and torch.equal(again[others], passed[others])


@needs_cuda
def test_a_checked_row_reads_what_a_check_of_every_row_reads(stepped_call, monkeypatch):
    """Up to rounding: an overlap is summed over thousands of sphere pairs in float32, and the last bits of a batched
    product follow the size of the batch, so the two cannot agree to the bit."""
    from cutamp import step_loop
    plan = make_plan(stepped_call)
    p, n = plan.particles, 16
    _, objects, _, held = step_loop._placements_and_grasps(plan.groups, plan.world, plan.pools, 0, n, bindings=p,
                                                           grasp_indices=p)
    # As if the pick's object is held, pushed 15 cm along one of the six axes of the hand, so that it overlaps the
    # robot's body on some particles.
    axis = torch.cat((torch.eye(3), -torch.eye(3))).to(p["left_q0"])[torch.arange(n) % 6]
    carried = {}
    for arm, (name, pose) in held[1].items():
        carried[arm] = (name, pose.clone())
        carried[arm][1][:, :3, 3] += 0.15 * axis
    assert carried
    rollout, seen = {}, []
    refresh_gap, real = step_loop.refresh_gap, step_loop.validate_step_geometry

    def gap(*args, **kw):
        rollout["all"] = refresh_gap(*args, **kw)
        return rollout["all"]

    def recorded(world, adapter, prediction, *, object_poses, held):
        out = real(world, adapter, prediction, object_poses=object_poses, held=held)
        seen.append({name: value.clone() for name, value in out["penetration"].items()})
        if len(seen) == 1:                                       # particles 2 and 5 violate a clearance as well
            next(iter(out["penetration"].values()))[[2, 5]] = 1.0
        return out

    monkeypatch.setattr(step_loop, "refresh_gap", gap)
    monkeypatch.setattr(step_loop, "validate_step_geometry", recorded)
    plan._run_gap(0, p, None, plan.incoming, objects[0], carried)
    copies = rollout["all"].copies
    assert all(part.isfinite().all() for step in copies.steps for part in step.trajectory)   # no row was zeroed
    every = real(plan.world, plan.adapter, copies, object_poses={k: v.repeat(4, 1, 1) for k, v in objects[0].items()},
                 held={arm: (name, pose.repeat(4, 1, 1)) for arm, (name, pose) in carried.items()})["penetration"]
    own, theirs = seen
    violating = (torch.stack(tuple(own.values()), 1) > 1e-6).any(1)
    violating[[2, 5]] = True
    rows = (torch.arange(1, 4, device=violating.device)[:, None] * n + violating.nonzero().flatten()).flatten()
    assert any((value > 0).any() for name, value in every.items() if name.startswith("payload_robot"))
    for name in every:
        torch.testing.assert_close(own[name], every[name][:n], rtol=1e-5, atol=1e-7, msg=name)
        torch.testing.assert_close(theirs[name], every[name][rows], rtol=1e-5, atol=1e-7, msg=name)
