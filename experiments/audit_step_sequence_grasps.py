"""Compare the learned and teacher grasp-to-foot gradients at the same T1 issue state."""

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from cutamp.landing_accuracy import LANDING_BUNDLE
from cutamp.landing_predictor import LandingPredictor, nominal_landing_dynamics, prediction_precision
from cutamp.step_adam import optimize_particles
from cutamp.step_ik import solve_stance_ik
from cutamp.step_refinement import _evaluate, stance_center_residual
from cutamp.tests.test_grasp_stance_ik import paired_inputs
from experiments.step_sequence_adapter import load_student, predict_sequence


def time_landing_gradient(teacher, student, template, batch=64):
    """Same two-step commands and scalar terminal-foot objective, synchronized warm wall time."""
    issue = teacher.standing_start(nominal_landing_dynamics(template, batch, 2))
    feet = teacher.model.feet(issue.tensors[:6])[0]
    sides = torch.tensor([0, 1], device=template.device).expand(batch, -1)
    command = template.new_tensor([[0.12, 0.21, 0.0], [0.12, -0.21, 0.0]]).repeat(batch, 1, 1)

    def original(c):
        return teacher.rollout(issue, "LR", c).steps[-1].achieved

    def learned(c):
        return student(issue.tensors, issue.dynamics, feet, sides, c)[1][:, 1, -1]

    def timed(fn):
        c = command.detach().requires_grad_()
        torch.cuda.synchronize()
        start = perf_counter()
        feet_xy = fn(c)[..., :2]
        (g,) = torch.autograd.grad(feet_xy.square().sum(), c)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        if not g.isfinite().all():
            raise ValueError("Nonfinite foot gradient in latency measurement")
        return elapsed

    result = {"batch": batch, "order": "LR", "objective": "squared terminal sole XY", "units": "seconds"}
    for name, fn in (("teacher", original), ("student", learned)):
        timed(fn)
        result[name] = [timed(fn) for _ in range(5)]
    result["teacher_over_student_median"] = median(result["teacher"]) / median(result["student"])
    return result


@prediction_precision()
def audit(student_path, bundle, constrained=False):
    teacher = LandingPredictor(bundle)
    student, _ = load_student(student_path)
    wrist, seed = paired_inputs("cuda", torch.float32)
    start = teacher.standing_start(nominal_landing_dynamics(wrist, 2, 2))
    feet = teacher.model.feet(start.tensors[:6])[0]
    sides = torch.tensor([[0, -1]], device=wrist.device).expand(2, -1)
    initial = wrist.new_tensor([[[0.12, 0.21, 0.0]]]).repeat(2, 1, 1)

    def student_feet(command):
        padded = torch.cat((command, torch.zeros_like(command)), dim=1)
        _, soles = student(start.tensors, start.dynamics, feet, sides, padded)
        return soles[:, 0, -1]

    def centers(soles):
        ik = solve_stance_ik(soles, wrist, seed, "left", center_posture=True)
        return stance_center_residual(ik).square().sum(1)

    def true_metrics(command):
        with torch.no_grad():
            pred = teacher.rollout(start, "L", command)
            soles = pred.steps[0].achieved
            ik = solve_stance_ik(soles, wrist, seed, "left", center_posture=True)
            return {
                "swing_xy": soles[:, 0, :2].tolist(),
                "center_cm": (stance_center_residual(ik).norm(dim=1) * 100).tolist(),
                "max_xy_error_cm": (pred.final_xy_error.amax(1) * 100).tolist(),
                "max_yaw_error_deg": pred.final_yaw_error.amax(1).rad2deg().tolist(),
                "landing_pass": pred.tracking_ok.tolist(),
                "ik_pass": ik.ok.tolist(),
                "support_margin_m": ik.margin.tolist(),
            }

    if constrained:
        # Reuse the original 12-update paired-grasp experiment and its inequality rows.
        def evaluate(command, *, exact):
            prediction = (
                teacher.rollout(start, "L", command)
                if exact
                else predict_sequence(student, teacher, start, "L", command)
            )
            return _evaluate(
                teacher,
                start,
                feet,
                "L",
                command,
                "left",
                wrist,
                seed,
                0.05,
                None,
                None,
                None,
                prediction=prediction,
                solve_ik=partial(solve_stance_ik, center_posture=True),
            )

        def objective(params):
            result = evaluate(params["foot"], exact=False)
            return result.matrix().relu().square().sum(1), result.worst_center_squared()

        best, diagnostics = optimize_particles({"foot": initial}, objective, 12, 0.003)
        with torch.no_grad():
            verified = evaluate(best["foot"], exact=True)
        return {
            "grasp": ["near", "far"],
            "updates": 12,
            "objective": "feasibility then squared CoM distance",
            "initial_command": initial.tolist(),
            "optimized_command": best["foot"].tolist(),
            "selected_iteration": diagnostics["iteration"].tolist(),
            "exact_feasible": verified.ok.tolist(),
            "exact_initial": true_metrics(initial),
            "exact_final": true_metrics(best["foot"]),
        }

    q = initial.clone().requires_grad_()
    (exact_grad,) = torch.autograd.grad(centers(teacher.rollout(start, "L", q).steps[0].achieved).sum(), q)
    (student_grad,) = torch.autograd.grad(centers(student_feet(q)).sum(), q)
    assert torch.isfinite(student_grad).all() and torch.isfinite(exact_grad).all()
    # Keep this intervention small and test both proposals using original replay.
    opt = torch.optim.Adam([q], lr=0.003)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        centers(student_feet(q)).sum().backward()
        opt.step()
    return {
        "grasp": ["near", "far"],
        "objective": "squared CoM-to-centroid distance only",
        "exact_initial": true_metrics(initial),
        "exact_after_two_updates": true_metrics(q.detach()),
        "teacher_gradient": exact_grad[:, 0].tolist(),
        "student_gradient": student_grad[:, 0].tolist(),
        "cosine": torch.nn.functional.cosine_similarity(student_grad[:, 0], exact_grad[:, 0]).tolist(),
        "initial_command": initial[:, 0].tolist(),
        "optimized_command": q.detach().tolist(),
        "same_initial_command": bool(torch.equal(initial[0], initial[1])),
        "isolated_gradient_latency": time_landing_gradient(teacher, student, wrist),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=LANDING_BUNDLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--constrained", action="store_true", help="Run the 12-update constrained fixture instead")
    args = parser.parse_args()
    result = audit(args.student, args.bundle, args.constrained)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)
