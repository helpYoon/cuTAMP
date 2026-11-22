"""Matched local-GPU acceptance through the public cuTAMP entry point."""

import pytest
import torch

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tests.conftest import first_schedules, needs_cuda, only_schedule
from cutamp.tests.paired_grasp_scene import paired_setup


def paired_case(label, output):
    cfg = TAMPConfiguration(
        enable_visualizer=False,
        curobo_plan=False,
        experiment_root=str(output),
    )
    torch.manual_seed(641)
    with pytest.MonkeyPatch.context() as mp:
        env, q, inp_args = paired_setup(label, mp)
        only_schedule(mp, ("L", "none"), "left")
        first_schedules(mp, 1)
        _, count, metrics = run_cutamp(
            env,
            cfg,
            CostReducer(default_constraint_to_mult.copy()),
            ConstraintChecker(default_constraint_to_tol.copy()),
            q_init=q,
            step_inputs=inp_args,
            experiment_id=label + "-place",
        )
    assert count > 0, metrics
    return metrics


@needs_cuda
def test_shared_grasp_and_achieved_stance_at_pick_and_place(tmp_path):
    for label in ("near", "far"):
        result = paired_case(label, tmp_path)
        pick, place = result["gaps"]
        assert result["model_feasible"] and pick["order"] == "L" and place["order"] == "none"
        assert pick["achieved"] == place["achieved"]
        assert min(result["com_margins_m"]) >= 0.05
        assert max(pick["final_xy_error_m"]) < 0.02 and max(pick["final_yaw_error_deg"]) < 1.0
        assert result["largest_arm_joint_move_rad"] < 1.0
