"""Same-state, same-command intervention on the grasp with the frozen deployed tick."""
import argparse
import json
from functools import partial
from pathlib import Path

import torch

from cutamp.landing_accuracy import LANDING_BUNDLE
from cutamp.landing_predictor import LandingPredictor, nominal_landing_dynamics, prediction_precision
from cutamp.step_adam import optimize_particles
from cutamp.step_ik import solve_stance_ik
from cutamp.step_refinement import _evaluate
from cutamp.tests.test_grasp_stance_ik import paired_inputs


@prediction_precision()
def run():
    predictor = LandingPredictor(LANDING_BUNDLE)
    wrist, seed = paired_inputs("cuda", torch.float32)
    start = predictor.standing_start(nominal_landing_dynamics(wrist, 2, 2))
    feet = predictor.model.feet(start.tensors[:6])[0]
    initial = wrist.new_tensor([[[0.12, 0.21, 0.0]]]).repeat(2, 1, 1)

    def evaluate(u):
        pred = predictor.rollout(start, "L", u)
        return _evaluate(predictor, start, feet, "L", u, "left", wrist, seed, 0.05,
                         None, None, None, prediction=pred, solve_ik=partial(solve_stance_ik, center_posture=True))

    def objective(x):
        ev = evaluate(x["foot"])
        return ev.matrix().relu().square().sum(1), ev.worst_center_squared()

    def metrics(u):
        with torch.no_grad():
            ev = evaluate(u)
            return dict(command=u.tolist(), achieved=ev.prediction.steps[0].achieved.tolist(),
                        center_cm=(100 * ev.center.norm(dim=-1)).flatten().tolist(),
                        xy_cm=(100 * ev.prediction.steps[0].xy_error.amax(1)).tolist(),
                        yaw_deg=ev.prediction.steps[0].yaw_error.amax(1).rad2deg().tolist(),
                        valid=ev.ok.tolist())

    u = initial.clone().requires_grad_()
    grad, = torch.autograd.grad(evaluate(u).worst_center_squared().sum(), u)
    dx = torch.zeros_like(initial)
    dx[:, :, 0] = 0.0004
    with torch.no_grad():
        fd = (evaluate(initial + dx).worst_center_squared() -
              evaluate(initial - dx).worst_center_squared()) / 0.0008
    torch.testing.assert_close(fd, grad[:, 0, 0].double(), atol=0.0005, rtol=0.04)
    best, diagnostics = optimize_particles({"foot": initial}, objective, 12, 0.003)
    final = metrics(best["foot"])
    if not all(final["valid"]) or max(final["xy_cm"]) >= 2 or max(final["yaw_deg"]) >= 1:
        raise AssertionError(f"Failed model or stance checks: {final}")
    if torch.allclose(grad[0], grad[1]) or torch.allclose(best["foot"][0], best["foot"][1]):
        raise AssertionError("The grasp intervention must change the gradient and the selected command")
    return dict(grasp=["near", "far"], initial=metrics(initial), final=final,
                center_gradient=grad.tolist(), x_derivative_finite_difference=fd.tolist(),
                best_iteration=diagnostics["iteration"].tolist(),
                all_weights_frozen=all(not p.requires_grad for p in predictor.model.parameters()),
                checkpoint=predictor.metadata["checkpoint_sha256"], gpu=torch.cuda.get_device_name())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    result = run()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output, result["final"], flush=True)
