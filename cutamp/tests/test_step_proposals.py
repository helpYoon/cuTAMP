"""Regression for preserving each particle's own incoming state in forward screening."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from cutamp import step_proposals as proposals
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.landing_predictor import LandingState


def test_forward_proposals_require_model_and_preserve_explicit_updates():
    config = TAMPConfiguration(step_mode="model")
    assert config.num_particles == 64
    assert config.for_refinement(stepped=True).step_proposal_count == 192
    assert config.for_refinement(stepped=True).step_place_proposal_count == 128
    assert replace(config, step_proposal_count=48).for_refinement(stepped=True).step_place_proposal_count == 48
    assert config.for_refinement(stepped=False).step_proposal_count == 0
    assert TAMPConfiguration().num_particles == 64
    assert TAMPConfiguration().curobo_plan
    assert TAMPConfiguration(step_mode="off").num_particles == 256
    assert TAMPConfiguration(step_mode="off").for_refinement(stepped=False).com_center_weight == 0
    assert TAMPConfiguration(step_mode="off", com_center_weight=2.0).for_refinement(stepped=False).com_center_weight == 2.0
    assert TAMPConfiguration(step_mode="model", num_particles=8).num_particles == 8
    assert (
        replace(config, step_derivative_method="finite_difference").for_refinement(stepped=True).step_proposal_count
        == 0
    )
    assert replace(config, step_refine_iterations=3).for_refinement(stepped=True).step_proposal_count == 0
    validate_tamp_config(config)
    assert config.for_refinement(stepped=True).step_refine_iterations == 0
    assert replace(config, step_refine_iterations=3).for_refinement(stepped=True).step_refine_iterations == 3
    for bad in [
        replace(config, step_mode="geometric", step_proposal_count=64),
        replace(config, step_proposal_count=-1),
    ]:
        with pytest.raises(ValueError):
            validate_tamp_config(bad)


@dataclass
class _Step:
    achieved: torch.Tensor


@dataclass
class _Prediction:
    owner: torch.Tensor
    steps: tuple[_Step, ...]


def test_shortlisting_checks_next_candidate_and_preserves_the_scene_per_particle(monkeypatch):
    context = torch.zeros(2, 38)
    context[:, 0] = torch.arange(2)
    state = LandingState((context[:, :1],), context, "observed")
    adapter = SimpleNamespace(predictor=SimpleNamespace(model=SimpleNamespace(feet=lambda t: (t[0],))))
    scene = torch.eye(4).repeat(2, 1, 1)
    scene[:, 0, 3] = torch.arange(2)
    touched = []

    def evaluate(predictor, incoming, _, order, commands, arm, wrist, seed, *args, **kwargs):
        owner = incoming.dynamics[:, 0].long()
        assert torch.equal(wrist[:, 0, 3].long(), owner)
        cost = torch.arange(len(commands)).to(commands)
        center = torch.stack([cost, torch.zeros_like(cost)], dim=-1)[:, None]
        feet = commands.expand(-1, 2, -1).clone()
        prediction = _Prediction(owner, (_Step(feet),))
        return SimpleNamespace(prediction=prediction, ok=owner >= 0, center=center, ik=SimpleNamespace(q=seed))

    def check(world, adapter, prediction, *, object_poses, held):
        owner = prediction.owner
        assert torch.equal(owner, object_poses["block"][:, 0, 3].long())
        assert torch.equal(owner, held["left"][1][:, 0, 3].long())
        # First block contains the rank-0 through rank-7 candidates for both particles.
        # The second particle requires further checking.
        ok = owner == 0 if not touched else torch.ones_like(owner, dtype=torch.bool)
        touched.append(len(owner))
        return {"valid": ok}

    monkeypatch.setattr(proposals, "_evaluate", evaluate)
    monkeypatch.setattr(proposals, "validate_step_geometry", check)
    result = proposals.rank_landing_commands(
        None,
        adapter,
        state,
        "L",
        torch.zeros(2, 1, 3),
        scene,
        context[:, :33],
        "left",
        count=21,
        max_choices=1,
        object_poses={"block": scene},
        held={"left": ("block", scene)},
    )
    assert touched == [16, 2]
    assert result.particle_indices.tolist() == [0, 1]
    assert result.centering_m.tolist() == [0, 17]
    assert result.prediction(1).owner.tolist() == [1]
    torch.testing.assert_close(result.prediction(1).steps[-1].achieved, result.soles[1:2])

def test_remaining_budget_is_assigned_only_to_valid_prefix_owners(monkeypatch):
    n = 8
    context = torch.zeros(n, 38)
    context[:, 0] = torch.arange(n)
    initial = LandingState((context[:, :1],), context, "observed")
    adapter = SimpleNamespace(predictor=SimpleNamespace(model=SimpleNamespace(feet=lambda x: (x[0],))))
    scene = torch.eye(4).repeat(n, 1, 1)
    scene[:, 0, 3] = torch.arange(n)

    def evaluate(predictor, incoming, _, order, commands, arm, wrist, seeds, *args, **kwargs):
        owner = incoming.dynamics[:, 0].long()
        assert owner.tolist() == [2, 6] * 10
        torch.testing.assert_close(wrist, scene[owner])
        pose = commands.expand(-1, 2, 3).clone()
        pred = _Prediction(owner, (_Step(pose),))
        return SimpleNamespace(prediction=pred, ok=owner >= 0, center=pose[:, :1, :2], ik=SimpleNamespace(q=seeds))

    def valid(world, adapter, pred, *, object_poses, held):
        torch.testing.assert_close(object_poses["part"], scene[pred.owner])
        torch.testing.assert_close(held["right"][1], scene[pred.owner])
        return {"valid": torch.ones_like(pred.owner, dtype=torch.bool)}

    monkeypatch.setattr(proposals, "_evaluate", evaluate)
    monkeypatch.setattr(proposals, "validate_step_geometry", valid)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[2, 6]] = True
    best = proposals.rank_landing_commands(None, adapter, initial, "R", torch.zeros(n, 1, 3), scene,
                                          context[:, :33], "left", object_poses={"part": scene},
                                          held={"right": ("part", scene)}, eligible_sources=mask, count=20, max_choices=1)
    assert best.particle_indices.tolist() == [2, 6]
    assert best.prediction(1).owner.item() == 6


@pytest.mark.parametrize("accepted_source", [2, -1])
def test_forward_batch_reuses_selected_or_nominal_prediction_for_every_source(monkeypatch, accepted_source):
    n = 5
    ctx = torch.zeros(n, 38)
    ctx[:, 0] = torch.arange(n)
    state = LandingState((ctx[:, :1],), ctx, "observed")
    adapter = SimpleNamespace(predictor=SimpleNamespace(model=SimpleNamespace(feet=lambda x: (x[0],))))
    wrist = torch.eye(4).expand(n, -1, -1)
    nominal = torch.zeros(n, 1, 3)
    nominal[:, 0, 0] = torch.arange(n)
    calls = []

    def evaluate(predictor, initial, incoming, order, commands, arm, wrist, seed, *args, **kwargs):
        owner = initial.dynamics[:, 0].long()
        calls.append(owner)
        feet = commands.expand(-1, 2, -1).clone()
        distance = torch.arange(len(owner), 0, -1).to(commands)
        center = torch.stack([distance, torch.zeros_like(distance)], -1)[:, None]
        return SimpleNamespace(prediction=_Prediction(owner, (_Step(feet),)), ok=(owner == accepted_source),
                               center=center, ik=SimpleNamespace(q=seed))

    monkeypatch.setattr(proposals, "_evaluate", evaluate)
    monkeypatch.setattr(proposals, "validate_step_geometry", lambda w, a, p, **kw: {"valid": p.owner == 2})
    result = proposals.rank_landing_commands(None, adapter, state, "L", nominal, wrist, ctx[:, :33], "left",
                                           count=9, max_choices=1, eligible_sources=ctx[:, 0].long() % 2 == 0,
                                           include_fallback=True)
    assert len(calls) == 1
    assert result.particle_indices.tolist() == ([2] if accepted_source == 2 else [])
    expected = nominal.clone()
    if accepted_source == 2:
        expected[2] = result.commands[0]
        assert not torch.equal(nominal[2], expected[2])
    assert result.batch_prediction.owner.tolist() == list(range(n))
    torch.testing.assert_close(result.batch_prediction.steps[0].achieved[:, 0], expected[:, 0])


def test_place_line_search_stops_before_next_ik_when_deadline_expires(monkeypatch):
    from cutamp import step_particles, step_refinement

    clock = [0.]
    monkeypatch.setattr(step_refinement, "monotonic", lambda: clock[0])
    eye = torch.eye(4)
    world = SimpleNamespace(tool_from_ee={"tool": eye}, tool_frame_for_arm=lambda arm: "tool",
                            get_collision_spheres=lambda obj: torch.tensor([[0., 0., 0., .01]]),
                            get_aabb=lambda surf: torch.tensor([[-1., -1., 0.], [1., 1., 1.]]))
    op = SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(arm="left")), arg=lambda v: v)
    invocations = []

    def solve(soles, wrist, seed, arm, **kwargs):
        invocations.append(len(wrist))
        clock[0] = 1.
        return SimpleNamespace(xy=wrist[:, :2, 3])

    monkeypatch.setattr(step_particles, "solve_stance_ik", solve)
    monkeypatch.setattr(step_particles, "stance_ik_rows", lambda ik, m: {"supported": ik.xy[:, 0] * 0.})
    monkeypatch.setattr(step_particles, "stance_center_residual", lambda ik: ik.xy)
    with pytest.raises(TimeoutError, match="planner deadline"):
        step_particles._place_update(world, op, torch.tensor([[.3, 0., 0., 0.]]), torch.zeros(1, 2, 3),
                                     torch.zeros(1, 33), torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]),
                                     .05, 4., deadline=1.)
    assert invocations == [1]
