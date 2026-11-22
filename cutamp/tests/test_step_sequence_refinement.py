from types import SimpleNamespace

import pytest
import torch

from cutamp import step_particles
from cutamp.step_ik import _IK_TOLERANCE


def test_every_particle_starts_at_the_landing_commands_the_astar_chose():
    stepped = ((0.32, -0.21, 0.0), (0.0, 0.21, 0.0))  # the A*'s RL edge before a pick
    edges = (SimpleNamespace(commands=()), SimpleNamespace(commands=stepped))
    starts = step_particles._landing_starts(SimpleNamespace(schedule=SimpleNamespace(edges=edges)), 4, torch.zeros(33))
    assert starts[0].shape == (4, 0, 3)
    assert torch.equal(starts[1], torch.tensor(stepped).expand(4, -1, -1))


def test_adam_verification_refits_with_original_tolerance_and_no_warm_history(monkeypatch):
    # Keep the actual Adam loop and replace only its recursive particle evaluations.
    outer = step_particles._refine_batch
    step = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="step")))
    pick = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type="pick")))
    proposal = SimpleNamespace(skeleton=(step, pick))
    initial = torch.full((3, 1, 3), 0.1)
    monkeypatch.setattr(step_particles, "CostFunction", lambda *args: None)
    monkeypatch.setattr(step_particles, "_placements_and_grasps", lambda *a: ({}, {}, [], []))
    histories = []
    verified = []

    def evaluate(*args, bindings, objective, ik_starts, **kwargs):
        x = bindings["foot0"]
        if objective:
            assert _IK_TOLERANCE.get() == 5e-7
            histories.append(ik_starts)
            return x.new_zeros(len(x)), x.square().flatten(1).sum(1)
        assert ik_starts is None and _IK_TOLERANCE.get() == 2e-9
        assert not torch.is_grad_enabled()
        verified.append(x.clone())
        return SimpleNamespace(rows=[], diagnostics={})

    monkeypatch.setattr(step_particles, "_refine_batch", evaluate)
    cfg = SimpleNamespace(step_adam_plateau_mm=None, step_refine_iterations=2, lr=0.007)
    world = SimpleNamespace(q_init=initial)
    for _ in range(2):
        result = outer(proposal, world, cfg, None, None, None, None, None, [initial], 0, 3, None, None)
        assert result.diagnostics["adam"]["iterations"] == 2
    assert len(verified) == 2 and (verified[0] < initial).all()
    assert len(histories) == 6
    assert isinstance(histories[0], dict)
    assert histories[0] is histories[1] is histories[2]
    assert histories[3] is histories[4] is histories[5]
    assert histories[0] is not histories[3]


def test_elapsed_deadline_aborts_objective_before_particle_evaluation():
    with pytest.raises(TimeoutError, match="planner deadline"):
        step_particles._refine_batch(None, None, None, None, None, None, None, None, None,
                                     0, 1, None, 0., objective=True)
