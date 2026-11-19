"""Matched local-GPU acceptance through the public cuTAMP entry point."""

import torch

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tests.conftest import needs_cuda
from cutamp.tests.paired_grasp_scene import ROOT, paired_setup


def paired_case(label, output, *, place=False, motion=False):
    env, q, inp_args = paired_setup(label, place=place)
    bundle = ROOT / "data/t1-reachability/landing-accuracy/landing-seed3.pt"
    cfg = TAMPConfiguration(
        step_mode="model",
        num_particles=1,
        enable_visualizer=False,
        curobo_plan=motion,
        step_refine_iterations=4,
        max_step_schedules=1,
        max_loop_dur=180,
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
def test_changing_only_grasp_changes_optimized_landing_in_run_cutamp(tmp_path):
    pairs = [paired_case(label, tmp_path) for label in ("near", "far")]
    # Metrics refer to the actual row that met all checks.
    steps = [r["gaps"][0] for r in pairs]
    assert steps[0]["seed_commands"] == steps[1]["seed_commands"]
    assert steps[0]["initial_achieved"] == steps[1]["initial_achieved"]
    assert steps[0]["commands"] != steps[1]["commands"]
    assert torch.dist(torch.tensor(steps[0]["achieved"][0][:2]), torch.tensor(steps[1]["achieved"][0][:2])) > 0.004
    assert steps[0]["initial_margin_gradient"] != steps[1]["initial_margin_gradient"]
    assert steps[0]["initial_center_gradient"] != steps[1]["initial_center_gradient"]
    assert steps[1]["initial_margin_m"] < 0.05 <= steps[1]["margin_m"]
    for g in steps:
        assert g["center_distance_m"] < g["initial_center_distance_m"] - 0.001
        measured = torch.dist(torch.tensor(g["com_xy"]), torch.tensor(g["hull_centroid_xy"]))
        assert abs(measured.item() - g["center_distance_m"]) < 1e-6
        assert g["margin_m"] >= 0.05
        assert max(g["steps"][0]["xy_error_m"]) < 0.02
        assert max(g["steps"][0]["yaw_error_deg"]) < 1.0


@needs_cuda
def test_shared_grasp_and_achieved_stance_at_pick_and_place(tmp_path):
    for label in ("near", "far"):
        result = paired_case(label, tmp_path, place=True)
        pick, place = result["gaps"]
        assert result["model_feasible"] and place["order"] == "none"
        assert pick["achieved"] == place["achieved"]
        assert min(pick["margin_m"], place["margin_m"]) >= 0.05
        assert max(pick["final_xy_error_m"]) < 0.02 and max(pick["final_yaw_error_deg"]) < 1.0
