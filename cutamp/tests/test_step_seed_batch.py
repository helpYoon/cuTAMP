from types import SimpleNamespace

import pytest
import torch

from cutamp import step_particles


@pytest.mark.parametrize("mixed", [False, True])
def test_seed_transforms_batch_witnesses_and_preserve_particle_gradients(monkeypatch, mixed):
    orders = ["RL"] * 8 if not mixed else ["LR", None, "none", "RL", "L", "LR", "missing", "R"]
    gen = torch.Generator().manual_seed(37)
    n = len(orders)
    dtype = torch.float64
    incoming = torch.randn(n, 2, 3, generator=gen, dtype=dtype).requires_grad_()
    commands = torch.randn(n, 2, 3, generator=gen, dtype=dtype).requires_grad_()
    rest = torch.randn(n, 12, generator=gen, dtype=dtype).requires_grad_()
    weights = torch.randn(n, 12, generator=gen, dtype=dtype)
    seeds = []
    reference = []
    for i, order in enumerate(orders):
        witness = tuple(torch.randn(12, generator=gen, dtype=dtype).tolist()) if order != "missing" else None
        edge = SimpleNamespace(order=order, witness=witness)
        seeds.append(SimpleNamespace(schedule=SimpleNamespace(edges=[edge])) if order is not None else None)
        if order is None or witness is None:
            reference.append(rest[i])
        else:
            nominal = step_particles._nominal(incoming[i : i + 1], order, commands[i : i + 1])
            reference.append(step_particles._carry_witness(rest.new_tensor([witness]), nominal)[0])
    expected = torch.stack(reference)
    original = step_particles._carry_witness
    batch_sizes = []

    def capture(q, feet):
        batch_sizes.append(len(q))
        return original(q, feet)

    monkeypatch.setattr(step_particles, "_carry_witness", capture)
    actual = step_particles._inputs_for_gap(
        SimpleNamespace(posture_seeds=None), seeds, 0, None, incoming, commands, rest, 0, n
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    inputs = (incoming, commands, rest)
    expected_grads = torch.autograd.grad(expected, inputs, weights, allow_unused=True)
    actual_grads = torch.autograd.grad(actual, inputs, weights, allow_unused=True)
    for a, e in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-10)
    assert max(batch_sizes) > 1


def test_all_fallback_and_explicit_seed_batch_slice():
    rest = torch.randn(3, 12)
    incoming = torch.zeros(3, 2, 3)
    commands = torch.empty(3, 0, 3)
    args = ([None] * 3, 0, None, incoming, commands, rest, 2, 5)
    fallback = step_particles._inputs_for_gap(SimpleNamespace(posture_seeds=None), *args)
    torch.testing.assert_close(fallback, rest)
    supplied = torch.arange(60).view(5, 12).float()
    override = step_particles._inputs_for_gap(SimpleNamespace(posture_seeds=[supplied]), *args)
    torch.testing.assert_close(override, supplied[2:5])
