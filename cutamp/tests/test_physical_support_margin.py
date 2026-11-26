"""Physical sole hull distances and comfort at a manipulation keyframe."""

import math
import sys

import pytest
import torch

from cutamp.com_polygon_cost import physical_support_margin
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.tests.conftest import needs_cuda


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def test_margin_is_inclusive_at_five_centimeters():
    soles = tensor([[-.1, .1, 0], [-.1, -.1, 0]]).expand(4, 2, 3)
    com = tensor([[-.049, 0], [-.05, 0], [-.051, 0], [.03, .17]])
    margin = physical_support_margin(com, soles, (.2, .1))
    torch.testing.assert_close(margin, tensor([.049, .05, .051, -.03]))
    assert (margin >= .05).tolist() == [False, True, True, False]


def test_rotated_staggered_stances_keep_their_own_diagonal_edge_across_batch_dimensions():
    # A pair of squares translated along x=y. The lower-right bridge runs along x-y=0.2.
    soles = tensor([[0., 0., 0.], [.2, .2, 0.]])
    com = tensor([.16, .04])
    margin = (.2 - (.16 - .04)) / math.sqrt(2)
    angle = .73
    c, s = math.cos(angle), math.sin(angle)
    rotation = tensor([[c, -s], [s, c]])
    translation = tensor([-.4, .7])
    moved_soles = torch.cat((soles[:, :2] @ rotation.T + translation, soles[:, 2:] + angle), dim=1)
    batch = torch.stack([soles, moved_soles]).reshape(1, 2, 2, 3)
    queries = torch.stack([com, com @ rotation.T + translation]).reshape(1, 2, 2)
    torch.testing.assert_close(physical_support_margin(queries, batch, (.2, .2)), tensor([[margin, margin]]))


def test_near_edge_shortfall_gradient_follows_com_and_separately_rotated_soles():
    soles = tensor([[.14, .10, .20], [-.03, -.10, -.31]]).requires_grad_(True)
    com = tensor([.14, -.008]).requires_grad_(True)
    com_direction = tensor([-.15, .2])
    sole_direction = tensor([[.3, -.4, .25], [-.5, .1, -.82]])

    def shortfall(c, feet):
        return (.05 - physical_support_margin(c, feet, (.22, .10))).clamp_min(0)

    value = shortfall(com, soles)
    assert value > 0
    dcom, dsoles = torch.autograd.grad(value, (com, soles))
    assert dsoles[:, :2].abs().sum() > 0 and dsoles[:, 2].abs().sum() > 0
    delta = 1e-6
    for dc, df in [(com_direction, torch.zeros_like(soles)), (torch.zeros_like(com), sole_direction)]:
        expected = (shortfall(com + delta * dc, soles + delta * df)
                    - shortfall(com - delta * dc, soles - delta * df)) / (2 * delta)
        torch.testing.assert_close((dcom * dc).sum() + (dsoles * df).sum(), expected, atol=1e-8, rtol=1e-7)


def test_coincident_soles_and_invalid_rows():
    soles = tensor([[[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., float("nan"), 0.]]])
    margin = physical_support_margin(tensor([[0., 0.], [0., 0.]]), soles, (.22, .10))
    assert margin[0].item() == pytest.approx(.05)
    assert not (margin[1] >= 0)


@pytest.mark.parametrize("threshold", [-.01, math.nan, math.inf, -math.inf])
def test_comfort_margin_must_be_finite_and_nonnegative(threshold):
    with pytest.raises(ValueError, match="com_comfort_margin_m"):
        validate_tamp_config(TAMPConfiguration(com_comfort_margin_m=threshold))


def test_cli_forwards_physical_margin_in_meters(monkeypatch):
    from cutamp.scripts import run_cutamp as cli
    assert TAMPConfiguration().com_comfort_margin_m == .05
    validate_tamp_config(TAMPConfiguration(com_comfort_margin_m=0.))
    seen = []
    monkeypatch.setattr(sys, "argv", ["cutamp-demo", "--com_comfort_margin_m", ".036"])
    monkeypatch.setattr(cli, "cutamp_demo", lambda env, config, **kw: seen.append(config))
    cli.entrypoint()
    assert seen[0].com_comfort_margin_m == .036


@pytest.mark.parametrize("component", ["cost", "rollout"])
def test_fixed_stance_evaluator_rejects_a_manually_inserted_step(component):
    from cutamp.cost_function import CostFunction
    from cutamp.rollout import RolloutFunction
    from cutamp.t1_domain import LeftStep

    step = LeftStep.ground({p.name: p.name for p in LeftStep.parameters})
    with pytest.raises(NotImplementedError, match="LeftStep.*achieved sole"):
        if component == "cost":
            CostFunction([step], None, TAMPConfiguration())
        else:
            RolloutFunction([step], None)


@needs_cuda
def test_default_cost_checks_the_same_particle_at_each_fixed_stance_wrist():
    from pathlib import Path

    from curobo.types import DeviceCfg

    from cutamp.cost_function import CostFunction
    from cutamp.cost_reduction import CostReducer
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.robots import load_t1_container
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.rollout import RolloutFunction
    from cutamp.scripts.utils import default_constraint_to_mult
    from cutamp.t1_domain import LeftPick, LeftPlace, RightPick
    from cutamp.tamp_world import TAMPWorld
    from cutamp.tests.conftest import default_checker, fixed_stances

    dc = DeviceCfg()
    home = dc.to_device(REAL_LEGS.home)
    env = load_env(str(Path(get_env_dir()) / "blocks_t1.yml"))
    world = TAMPWorld(env, dc, load_t1_container(REAL_LEGS, dc), home, grasps={})
    skeleton = [
        LeftPick.ground({"obj": "block3", "grasp": "g1", "q": "left_q1"}),
        LeftPlace.ground({"obj": "block3", "grasp": "g1", "placement": "p1", "surface": "goal", "q": "left_q2"}),
        RightPick.ground({"obj": "block2", "grasp": "g2", "q": "right_q1"}),
    ]
    particles = {"left_q0": home.repeat(4, 1), "right_q0": home.repeat(4, 1),
                 "p1": dc.to_device([.55, 0, .035, 0]).repeat(4, 1)}
    for name in ("g1", "g2"):
        particles[name] = dc.to_device([0, 0, 0, 1, 0, 0, 0]).repeat(4, 1)
    wrist_confs = ("left_q1", "left_q2", "right_q1")
    for i, name in enumerate(wrist_confs, start=1):
        q = home.repeat(4, 1)
        q[i, REAL_LEGS.joint_names.index("base_j_x")] += .16
        particles[name] = q.requires_grad_()

    rollout = RolloutFunction(skeleton, world)(particles)
    cost_fn = CostFunction(skeleton, world, TAMPConfiguration())
    costs = cost_fn(rollout, particles, achieved_stances=fixed_stances(world, particles))
    comfort = {"ComComfort": costs["ComComfort"]}
    checker = default_checker()
    mask = checker.get_full_mask(comfort)["ComComfort"]["margin_shortfall_squared"]
    assert mask.tolist() == [[True, True, True], [False, True, True],
                             [True, False, True], [True, True, False]]
    assert checker.get_mask(comfort).tolist() == [True, False, False, False]
    reduced = CostReducer(default_constraint_to_mult)(comfort)
    for i, grad in enumerate(torch.autograd.grad(reduced.sum(), [particles[n] for n in wrist_confs]), start=1):
        assert grad[i, REAL_LEGS.joint_names.index("base_j_x")] > 0
        assert torch.count_nonzero(grad[mask[:, i - 1]]) == 0
