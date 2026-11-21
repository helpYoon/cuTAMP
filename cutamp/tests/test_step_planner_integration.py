"""Matched local-GPU acceptance through the public cuTAMP entry point."""

import torch

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.landing_accuracy import LANDING_BUNDLE
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tests.conftest import needs_cuda
from cutamp.tests.paired_grasp_scene import paired_setup


def paired_case(label, output, *, place=False, motion=False, validator=None, motion_validation="independent"):
    env, q, inp_args = paired_setup(label, place=place)
    inp_args.transition_validator = validator
    bundle = LANDING_BUNDLE
    cfg = TAMPConfiguration(
        step_mode="model",
        step_refine_iterations=5,  # a short Adam run stays inside the dense teacher's precision on this fixture
        num_particles=1,
        enable_visualizer=False,
        curobo_plan=motion,
        step_motion_validation=motion_validation,
        max_step_schedules=1,
        experiment_root=str(output),
        landing_bundle=str(bundle),
    )
    torch.manual_seed(641)
    path, count, metrics = run_cutamp(
        env,
        cfg,
        CostReducer(default_constraint_to_mult.copy()),
        ConstraintChecker(default_constraint_to_tol.copy()),
        q_init=q,
        step_inputs=inp_args,
        experiment_id=label + ("-place" if place else "-pick"),
    )
    if motion:
        assert path is None and count == 0, metrics
    else:
        assert count > 0, metrics
    return metrics


@needs_cuda
def test_shared_grasp_and_achieved_stance_at_pick_and_place(tmp_path):
    for label in ("near", "far"):
        result = paired_case(label, tmp_path, place=True)
        pick, place = result["gaps"]
        assert result["model_feasible"] and place["order"] == "none"
        assert pick["achieved"] == place["achieved"]
        assert min(pick["margin_m"], place["margin_m"]) >= 0.05
        assert max(pick["final_xy_error_m"]) < 0.02 and max(pick["final_yaw_error_deg"]) < 1.0


@needs_cuda
def test_public_validated_motion_receives_optimized_winning_trace(tmp_path):
    requests = []

    def inspect_request(request):
        requests.append((request.foot, request.command.detach().cpu().tolist(), request.predicted.clock.item()))
        return None  # This source has no independent execution evidence.

    metrics = paired_case("near", tmp_path, motion=True, validator=inspect_request)
    attempt = metrics["step_attempts"][0]
    row = attempt["particle_candidate"]
    assert attempt["status"] == "model_feasible" and not metrics["found_solution"]
    assert metrics["motion_plan_attempts"] == 1 and "no transition" in metrics["motion_plan_failures"][0]
    gap = row["gaps"][0]
    assert requests == [("L", gap["commands"][0], gap["steps"][0]["clock"])]
