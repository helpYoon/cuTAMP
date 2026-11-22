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
