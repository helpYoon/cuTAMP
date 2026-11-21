"""A new HOLD must replay its own state and adjoints during the Adam pullback."""
import torch

from cutamp.landing_accuracy import LANDING_BUNDLE
from cutamp.landing_predictor import LandingPredictor, nominal_landing_dynamics, prediction_precision
from cutamp.robots.t1 import REAL_LEGS
from cutamp.step_validation import StepStateAdapter
from cutamp.tests.conftest import needs_cuda


@needs_cuda
def test_objective_terminal_history_and_overlapping_gradients():
    predictor = LandingPredictor(LANDING_BUNDLE, use_cuda_graph=True)
    adapter = StepStateAdapter(predictor)
    q = predictor.model.home_joints.new_tensor(REAL_LEGS.home).repeat(4, 1)
    ctx = nominal_landing_dynamics(q, len(q), 2)
    active = q.detach().clone().requires_grad_()
    perturbed = q.clone()
    perturbed[:, 9] += 0.002
    perturbed.requires_grad_()
    with prediction_precision():
        actual = [adapter.objective_state(pose, ctx, torch.zeros_like(pose)).tensors for pose in (active, perturbed)]
        expected = [adapter.initialize(pose, ctx).state.tensors for pose in (active, perturbed)]
        combined_actual = combined_expected = 0
        generator = torch.Generator(device=q.device).manual_seed(20261120)
        for actual_state, expected_state in zip(actual, expected):
            for a, e in zip(actual_state, expected_state):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
                w = torch.randn(e.shape, device=e.device, dtype=e.dtype, generator=generator)
                combined_actual = combined_actual + (a * w).sum()
                combined_expected = combined_expected + (e * w).sum()
        actual_grads = torch.autograd.grad(combined_actual, (active, perturbed))
        expected_grads = torch.autograd.grad(combined_expected, (active, perturbed))
    for a, e in zip(actual_grads, expected_grads):
        assert torch.isfinite(e).all() and e.norm() > 0
        torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-5)
