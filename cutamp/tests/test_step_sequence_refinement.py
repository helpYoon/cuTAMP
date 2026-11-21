from types import SimpleNamespace

import pytest
import torch

from cutamp import step_particles
from cutamp.step_ik import _IK_TOLERANCE


@pytest.mark.parametrize("optimizer", ["learned_adam", "adam"])
def test_adam_verification_refits_with_original_tolerance_and_no_warm_history(monkeypatch, optimizer):
    # Keep the actual Adam loop and replace only its recursive particle evaluations.
    outer = step_particles._refine_batch
    step = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="step")))
    pick = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="pick")))
    proposal = SimpleNamespace(skeleton=(step, pick))
    initial = torch.full((3, 1, 3), 0.1)
    monkeypatch.setattr(step_particles, "CostFunction", lambda *args: None)
    monkeypatch.setattr(step_particles, "_placements_and_grasps", lambda *a: ({}, {}, [], []))
    monkeypatch.setattr(step_particles, "_command_inputs", lambda *args: initial)
    histories = []
    verified = []

    def evaluate(*args, bindings, objective, ik_starts, **kwargs):
        x = bindings["foot0"]
        if objective:
            expected = 5e-7 if optimizer == "learned_adam" else 2e-9
            assert _IK_TOLERANCE.get() == expected
            histories.append(ik_starts)
            return x.new_zeros(len(x)), x.square().flatten(1).sum(1)
        assert ik_starts is None and _IK_TOLERANCE.get() == 2e-9
        assert not torch.is_grad_enabled()
        verified.append(x.clone())
        return SimpleNamespace(rows=[], diagnostics={})

    monkeypatch.setattr(step_particles, "_refine_batch", evaluate)
    cfg = SimpleNamespace(step_optimizer=optimizer, require_modeled_load=False, step_ik_objective_tol=5e-7,
                          step_ik_warm_start=True, step_adam_plateau_mm=None,
                          step_adam_trace_center=False, step_refine_iterations=2, lr=0.007)
    world = SimpleNamespace(q_init=initial)
    inputs = SimpleNamespace(placements=None)
    for _ in range(2):
        result = outer(proposal, world, cfg, None, None, inputs, None, None, None, 0, 3, None, None)
        assert result.diagnostics["adam"]["iterations"] == 2
    assert len(verified) == 2 and (verified[0] < initial).all()
    assert len(histories) == 6
    if optimizer == "learned_adam":
        assert isinstance(histories[0], dict)
        assert histories[0] is histories[1] is histories[2]
        assert histories[3] is histories[4] is histories[5]
        assert histories[0] is not histories[3]
    else:
        assert all(value is None for value in histories)


def test_unsupported_payload_is_rejected_before_adam(monkeypatch):
    step = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="step")))
    place = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="place")))
    proposal = SimpleNamespace(skeleton=(step, place))
    world = SimpleNamespace(q_init=torch.zeros(1, 33))
    monkeypatch.setattr(step_particles, "CostFunction", lambda *args: None)
    monkeypatch.setattr(step_particles, "_placements_and_grasps",
                        lambda *args: ({}, [], [], [{"right": ("cube", None)}]))
    cfg = SimpleNamespace(step_optimizer="learned_adam", require_modeled_load=True)
    result = step_particles._refine_batch(proposal, world, cfg, None, None, None,
                                          None, None, None, 0, 1, None, None)
    assert result.status == "unavailable" and "payload" in result.reason


def test_elapsed_deadline_aborts_objective_before_particle_evaluation():
    with pytest.raises(TimeoutError, match="planner deadline"):
        step_particles._refine_batch(None, None, None, None, None, None, None, None, None,
                                     0, 1, None, 0., objective=True)
