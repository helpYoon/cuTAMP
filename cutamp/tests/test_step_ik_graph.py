"""Captured bounded IK preserves the endpoint, its implicit gradient, and the active set."""

import torch

from cutamp.robots.t1 import REAL_LEGS
from cutamp.step_ik import _Kinematics, solve_stance_ik, stance_ik_graphs, stance_ik_tolerance
from cutamp.tests.conftest import needs_cuda
from cutamp.tests.test_grasp_stance_ik import paired_inputs


@needs_cuda
def test_captured_ik_gradients_match_eager_across_replays():
    wrist, seed = paired_inputs("cuda", torch.float64)
    wrist, seed = wrist.repeat(8, 1, 1), seed.repeat(8, 1)
    feet = seed.new_tensor([[[0.154, 0.094, 0.022], [0.042, -0.118, 0.029]]]).repeat(len(seed), 1, 1)

    def solve(enabled):
        results = []
        with stance_ik_graphs(enabled):
            for offset in (0.0, 0.001):
                inputs = [value.clone().requires_grad_() for value in (feet + offset, wrist, seed)]
                endpoint = solve_stance_ik(*inputs, "left", center_posture=True)
                assert endpoint.ok.all()
                results.append((inputs, endpoint))
        # The second replay must preserve the first implicit backward's saved state.
        return [(result.q, *torch.autograd.grad(result.margin.sum(), inputs)) for inputs, result in results]

    eager, graphed = solve(False), solve(True)
    for reference, actual in zip(eager, graphed, strict=True):
        for a, b in zip(reference, actual, strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


@needs_cuda
def test_captured_ik_compacts_and_scatter_preserves_bounds():
    kin = _Kinematics(torch.device("cuda"))
    seed = kin.lower.new_tensor(REAL_LEGS.home).repeat(32, 1)
    low, high = kin.lower.expand_as(seed).clone(), kin.upper.expand_as(seed).clone()
    low[-1, :3] = high[-1, :3] = seed[-1, :3]
    target = kin.fk(seed, "right", jacobian=False)[0]
    target[-2:, 24] += seed.new_tensor([0.04, 0.4])
    with stance_ik_graphs(False):
        eager = kin.solve(target, seed, "right", bounds=(low, high))
    with stance_ik_graphs(True):
        graphed = kin.solve(target, seed, "right", bounds=(low, high))
    assert {key[0] for key in kin._solve_graphs.cache} == {4, 32}
    assert graphed[2][-1, :3].all()
    for a, b in zip(eager, graphed, strict=True):
        torch.testing.assert_close(a, b, atol=4e-6, rtol=1e-6)


@needs_cuda
def test_centered_graph_reused_across_fixed_and_free_idle_arm():
    kin = _Kinematics(torch.device("cuda"))
    seed = kin.lower.new_tensor(REAL_LEGS.home).repeat(16, 1)
    idle = REAL_LEGS.left_arm
    lo, hi = kin.lower.expand_as(seed).clone(), kin.upper.expand_as(seed).clone()
    # Each row pins its idle shoulder at a different angle; the start must also be projected.
    lo[:, idle] = hi[:, idle] = seed[:, idle]
    lo[:, idle.start] += torch.linspace(0.01, 0.04, len(seed), device=seed.device)
    hi[:, idle.start] = lo[:, idle.start]
    initial = seed.clone()
    initial[:, idle.start] += 0.05
    pose, com = kin.fk(seed, "right", jacobian=False)
    target = torch.cat((pose, 0.1 * com[:, :2]), -1)

    def solve(graphs):
        with stance_ik_graphs(graphs):
            return kin.solve(target, seed, "right", bounds=(lo, hi), center_weight=0.1, iterations=2, initial=initial)

    for release_row in (False, True):
        if release_row:
            hi[-1, idle.start] += 0.15
        expected, captured = solve(False), solve(True)
        torch.testing.assert_close(captured[0], expected[0], atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(captured[3], expected[3], atol=2e-8, rtol=2e-8)
    assert len(kin._solve_graphs.cache) == 2


@needs_cuda
def test_graph_cache_preserves_each_ik_tolerance():
    kin = _Kinematics(torch.device("cuda"))
    seed = kin.lower.new_tensor(REAL_LEGS.home).repeat(16, 1)
    target = kin.fk(seed, "right", jacobian=False)[0]
    target[:, 24] += 0.04
    with stance_ik_graphs(True):
        tight = kin.solve(target, seed, "right")[0]
        tight_keys = set(kin._solve_graphs.cache)
        with stance_ik_tolerance(2e-7):
            kin.solve(target, seed, "right")
        assert set(kin._solve_graphs.cache) > tight_keys
        repeated = kin.solve(target, seed, "right")[0]
        torch.testing.assert_close(tight, repeated, rtol=0, atol=0)
