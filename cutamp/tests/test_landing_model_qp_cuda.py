"""Fresh rollout acceptance and grasp-driven optimization of a T1 foot command."""

import os
from pathlib import Path

import pytest
import torch

from cutamp.tests.conftest import needs_cuda
from cutamp.tests.test_grasp_stance_ik import paired_inputs


@needs_cuda
def test_only_grasp_moves_the_optimized_model_command():
    from cutamp.landing_predictor import LandingPredictor
    from cutamp.step_refinement import refine_gap, stance_center_residual

    data = Path(__file__).resolve().parents[2] / "data/t1-reachability"
    bundle = Path(os.environ.get("CUTAMP_LANDING_BUNDLE", data / "landing-accuracy/landing-seed3.pt"))
    if not bundle.is_file() and "CUTAMP_LANDING_BUNDLE" not in os.environ:
        pytest.skip("Requires the local seed-3 CUDA bundle")
    p = LandingPredictor(bundle)
    dynamics = torch.zeros(2, 38, device=p.device)
    dynamics[:, 0] = 0.25 / 0.45
    dynamics[:, -5:] = -1.0
    home = p.standing_start(dynamics)
    u = torch.tensor([[[0.12, 0.21, 0.0]]], device=p.device).repeat(2, 1, 1)
    wrist, seed = paired_inputs(p.device, torch.float32)
    out = refine_gap(p, home, "L", u, "left", wrist, seed, max_iterations=4)
    assert out.ok.tolist() == [True, True], {k: v.tolist() for k, v in out.rows.items()}
    assert out.initial_margin[0] > 0.05 > out.initial_margin[1]
    assert (out.commands[0] - u[0]).norm() > 0.002
    assert (out.commands[1] - u[1]).norm() > 0.002
    assert (out.commands[1] - out.commands[0]).norm() > 0.001
    assert torch.all(out.center_distance < out.initial_center_distance - 0.001)
    assert (out.initial_center_gradient[0] - out.initial_center_gradient[1]).norm() > 0.003
    assert (out.achieved[1, 0, :2] - out.achieved[0, 0, :2]).norm() > 0.002
    assert (out.initial_gradient[0] - out.initial_gradient[1]).norm() > 0.01
    assert out.margin.min() >= 0.05 and out.prediction.tracking_ok.all()
    assert all(
        step.phases.deadline_ok.all() and step.phases.hold_ok.all() and step.phases.support_ok.all()
        for step in out.prediction.steps
    )

    # Full-reroll differences test the complete chain including the re-solved IK and
    # its posture contribution, with clocks unchanged across the perturbation.
    from cutamp.step_ik import solve_stance_ik

    delta = torch.zeros_like(u)
    delta[:, :, 0] = 0.0007
    with torch.no_grad():
        lo, hi = (p.rollout(home, "L", u + sign * delta) for sign in (-1, 1))
        assert torch.equal(lo.steps[0].clock, hi.steps[0].clock)
        lower = solve_stance_ik(lo.steps[0].achieved, wrist, seed, "left")
        upper = solve_stance_ik(hi.steps[0].achieved, wrist, seed, "left")
    fd = (upper.margin - lower.margin) / 0.0014
    torch.testing.assert_close(out.initial_gradient[:, 0, 0].double(), fd, atol=0.006, rtol=0.02)
    center_fd = (stance_center_residual(upper).square().sum(1) - stance_center_residual(lower).square().sum(1)) / 0.0014
    torch.testing.assert_close(out.initial_center_gradient[:, 0, 0].double(), center_fd, rtol=0.02, atol=0.0004)
    larger = u.clone()
    larger[:, :, 0] = 0.32
    with torch.no_grad():
        prediction = p.rollout(home, "L", larger)
        post = solve_stance_ik(prediction.steps[-1].achieved, wrist, seed, "left")
    assert not prediction.tracking_ok.any()
    assert (post.margin > out.initial_margin).any()


@needs_cuda
def test_constructing_world_preserves_ik_robot_config():
    from curobo.types import DeviceCfg

    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.robots import load_t1_container
    from cutamp.robots.t1 import REAL_LEGS, t1_curobo_cfg
    from cutamp.step_ik import _Kinematics
    from cutamp.tamp_world import TAMPWorld

    dc = DeviceCfg()
    env = load_env(str(Path(get_env_dir()) / "blocks_t1.yml"))
    for _ in range(2):
        world = TAMPWorld(env, dc, load_t1_container(REAL_LEGS, dc), dc.to_device(REAL_LEGS.home), grasps={})
        assert isinstance(t1_curobo_cfg(REAL_LEGS)["robot_cfg"]["kinematics"]["extra_links"]["base_link_pitch"], dict)
        k = _Kinematics(world.q_init.device)
        q = world.q_init.reshape(1, -1).double()
        _, _, com = k.fk(q, "left")
        assert torch.isfinite(com).all()


@needs_cuda
def test_zero_step_places_grasp_selects_the_earlier_landing():
    """Both picks have the same already-comfortable wrist, only the downstream wrist differs."""
    from cutamp.landing_predictor import LandingPredictor
    from cutamp.step_ik import solve_stance_ik
    from cutamp.step_refinement import refine_gap, stance_center_residual, stance_ik_rows

    path = Path(__file__).resolve().parents[2] / "data/t1-reachability/landing-accuracy/landing-seed3.pt"
    bundle = Path(os.environ.get("CUTAMP_LANDING_BUNDLE", path))
    if not bundle.exists() and "CUTAMP_LANDING_BUNDLE" not in os.environ:
        pytest.skip("Requires the seed-3 deployment")
    p = LandingPredictor(bundle)
    wrist, seed = paired_inputs(p.device, torch.float32)
    dynamics = seed.new_zeros(2, 38)
    dynamics[:, 0] = 0.25 / 0.45
    dynamics[:, -5:] = -1.0
    start = p.standing_start(dynamics)
    commands = seed.new_tensor([[[0.12, 0.21, 0.0]]]).expand(2, -1, -1)

    def place_rows(pred, ik):
        future = solve_stance_ik(ik.soles, wrist, seed, "left")
        return {"place_" + name: row for name, row in stance_ik_rows(future, 0.05).items()}, [
            stance_center_residual(future)
        ]

    pick = refine_gap(
        p, start, "L", commands, "left", wrist[:1].expand_as(wrist), seed, max_iterations=4, extra_rows=place_rows
    )
    assert pick.ok.tolist() == [True, True], {n: r.tolist() for n, r in pick.rows.items()}
    torch.testing.assert_close(pick.initial_gradient[0], pick.initial_gradient[1])
    assert (pick.commands - commands).norm(dim=-1).min() > 0.002
    assert (pick.initial_center_gradient[1] - pick.initial_center_gradient[0]).norm() > 0.003
    assert (pick.commands[1] - pick.commands[0]).norm() > 0.001
    after = solve_stance_ik(pick.achieved, wrist, seed, "left")
    assert (after.margin >= 0.05).all() and pick.prediction.tracking_ok.all()
    with torch.no_grad():
        delta = torch.zeros_like(commands)
        delta[:, 0, 0] = 0.0007

        def shared_stance_loss(u):
            achieved = p.rollout(start, "L", u).steps[0].achieved
            primary = solve_stance_ik(achieved, wrist[:1].expand_as(wrist), seed, "left")
            place = solve_stance_ik(achieved, wrist, seed, "left")
            return stance_center_residual(primary).square().sum(1) + stance_center_residual(place).square().sum(1)

        fd = (shared_stance_loss(commands + delta) - shared_stance_loss(commands - delta)) / 0.0014
    torch.testing.assert_close(pick.initial_center_gradient[:, 0, 0].double(), fd, rtol=0.02, atol=0.0008)


@needs_cuda
def test_captured_qp_updates_different_grasp_rows_like_eager():
    from cutamp.landing_qp import composite_step

    # These slope and margin values were measured from the real matched grasp fixture.
    device = torch.device("cuda:0")
    g = torch.tensor([[-0.008963], [0.004524]], device=device, dtype=torch.float64)
    jac = g.new_tensor([[[-0.490531, 0.230353, 0.000792]], [[-0.556350, 0.272953, 0.002734]]])
    offset = g.new_zeros(2, 5)
    metric = g.new_tensor([1.0, 1.0, 0.1124])
    J = torch.diag(metric).expand(2, 3, 3).contiguous()
    # Replay the augmented effort matrix for two coordinate residuals as well as
    # the three command offsets. A feasible row can improve a nonzero offset.
    xy_jac = g.new_tensor([[0.46, 0.08, 0.01], [-0.05, 0.38, -0.01]]).expand(2, -1, -1)
    J = torch.cat((J, xy_jac), 1)
    limit = g.new_tensor([0.018, 0.018, 0.0349]).expand(2, -1).contiguous()

    def update():
        return composite_step(g, jac, offset, J, limit, metric)[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        update()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = update()
    for demand, center_x in ((0.004524, 0.0), (0.0015, -0.02)):
        g[1, 0] = demand
        offset[:, 3] = center_x
        expected = update()
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=2e-7, atol=3e-8)
        assert captured[1, 0] > 0.001
        assert (captured[0].norm() < 1e-6).item() == (center_x == 0)


@needs_cuda
@pytest.mark.parametrize("order", ["LR", "RL"])
def test_first_command_moves_second_issue_and_final_grasp_margin(order):
    from cutamp.landing_predictor import LandingPredictor, prediction_precision
    from cutamp.step_ik import solve_stance_ik

    path = Path(__file__).resolve().parents[2] / "data/t1-reachability/landing-accuracy/landing-seed3.pt"
    bundle = Path(os.environ.get("CUTAMP_LANDING_BUNDLE", path))
    if not bundle.exists() and "CUTAMP_LANDING_BUNDLE" not in os.environ:
        pytest.skip("Requires the seed-3 deployment")
    p = LandingPredictor(bundle)
    wrists, seeds = paired_inputs(p.device, torch.float32)
    d = seeds.new_zeros(2, 38)
    d[:, 0] = 0.25 / 0.45
    d[:, -5:] = -1.0
    state = p.standing_start(d)
    c = d.new_tensor([[0.12, 0.21 if s == "L" else -0.21, 0.0] for s in order])[None].repeat(2, 1, 1).requires_grad_()

    def score(u):
        out = p.rollout(state, order, u)
        result = solve_stance_ik(out.steps[-1].achieved, wrists, seeds, "left")
        second_target_x = out.steps[-1].reference[:, "LR".index(order[-1]), 0]
        return result.margin, second_target_x, tuple(s.clock for s in out.steps)

    with prediction_precision():
        margin, second_x, clocks = score(c)
        (dm,) = torch.autograd.grad(margin.sum(), c, retain_graph=True)
        (dx,) = torch.autograd.grad(second_x.sum(), c)
        delta = torch.zeros_like(c)
        delta[:, 0, 0] = 0.0007
        with torch.no_grad():
            low, lx, lc = score(c - delta)
            high, hx, hc = score(c + delta)
    assert all(torch.equal(a, b) and torch.equal(b, e) for a, b, e in zip(lc, clocks, hc))
    for gradient, finite_difference in ((dm, (high - low) / 0.0014), (dx, (hx - lx) / 0.0014)):
        assert gradient[:, 0, 0].abs().min() > 0.05
        torch.testing.assert_close(gradient[:, 0, 0].double(), finite_difference.double(), atol=0.004, rtol=0.01)
