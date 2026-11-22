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
    assert mult["ComComfort"]["margin_shortfall"] == 10.0 and tol["ComComfort"]["margin_shortfall"] == 0.0
    assert mult["KinematicConstraint"]["retract_sole_rot_err"] == 5.0
    assert tol["KinematicConstraint"]["retract_sole_pos_err"] == SOLE_POS_TOL
    assert tol["KinematicConstraint"]["retract_sole_rot_err"] == SOLE_ROT_TOL
    assert tol["StepCommand"]["default"] == 0.0 and tol["StepTests"]["refresh_tests"] == 0.0
    assert mult["soft"]["stay_near_start"] == 1.0
    assert {"stepped_scene", "retract_soles"} <= set(FK_SITES)


def test_a_failed_refresh_test_adds_nothing_to_the_loss():
    cost = {"StepTests": {"type": "constraint", "values": {"refresh_tests": torch.ones(4)}}}
    reducer = CostReducer(default_constraint_to_mult.copy())
    assert (reducer(cost) == 0).all()


import tempfile
from time import monotonic

import pytest

from cutamp import algorithm as alg
from cutamp.algorithm import run_cutamp
from cutamp.constraint_checker import ConstraintChecker
from cutamp.particle_initialization import ParticleInitializer
from cutamp.step_loop import SteppedPlan
from cutamp.step_planner import StepResult
from cutamp.step_refresh import as_commanded, command_rows
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
                       generator=torch.Generator(device="cuda").manual_seed(0), **kw)


@needs_cuda
def test_starting_values(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    assert plan.orders == ("L", "") and plan.command_names == {0: "step_commands_0"}
    u = p["step_commands_0"]
    assert u.shape == (16, 1, 3)
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
    plan = make_plan(stepped_call)
    p = plan.particles
    plan.refresh(p)
    kept, ok = {k: v.clone() for k, v in p.items()}, plan.tests_ok.clone()
    state = plan._first.state  # the first gap's cached hold, reused by every refresh
    state.tensors = tuple(t.clone() for t in state.tensors)  # the observed state's rows are expanded views
    state.tensors[0][3] = state.tensors[4][3] = float("nan")  # body position and joints of row 3
    plan.refresh(p)
    others = torch.arange(16, device="cuda") != 3
    assert not plan.tests_ok[3] and torch.equal(plan.tests_ok[others], ok[others])
    assert all(torch.equal(v[others], kept[k][others]) for k, v in p.items())


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
    assert (cost["soft"]["values"]["stay_near_start"] == 0).all()
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
def test_staying_near_the_start_counts_every_joint(stepped_call):
    plan = make_plan(stepped_call)
    p = plan.particles
    name = next(iter(plan.start_postures))
    with torch.no_grad():
        p[name][:, 7] += 0.1
        cost = plan(plan.rollout_fn(p), p)
    torch.testing.assert_close(cost["soft"]["values"]["stay_near_start"], torch.full((16,), 0.01, device="cuda"), atol=1e-6, rtol=0)


@needs_cuda
def test_result_rows(stepped_call):
    plan = make_plan(stepped_call)
    res = plan.result(plan.particles, stepped_call["reducer"], stepped_call["checker"])
    assert res.endpoint_ok.shape == (16, 2) and res.status in ("model_feasible", "unresolved")
    assert set(plan.motion_inputs) == {row.index for row in res.rows}
    for row in res.rows:
        m = row.metrics
        assert len(m["com_center_distances_m"]) == len(m["com_margins_m"]) == 2
        assert row.max_center_distance_m == max(m["com_center_distances_m"])
        pick, place = m["gaps"]
        assert pick["order"] == "L" and place["order"] == "none" and pick["achieved"] == place["achieved"]
        assert len(pick["commands"]) == 1 and len(pick["final_xy_error_m"]) == 2
        assert sum(k.endswith("_ik_reference") for k in row.particle) == 2


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
def test_an_unresolved_result_counts_the_particles_each_requirement_fails(stepped_call):
    plan = make_plan(stepped_call)
    res = plan.result(plan.particles, stepped_call["reducer"], stepped_call["checker"])
    assert res.status == "unresolved"  # no particle is feasible at the start on this scene
    counts = res.diagnostics["failure_counts"]
    assert set(res.diagnostics["failed_constraints"]) <= set(counts) and all(0 < v <= 16 for v in counts.values())
