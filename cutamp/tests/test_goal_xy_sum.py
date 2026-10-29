"""goal_in_xy's per-object sum, ``cost_function._per_object_sum``: the fixed-order
replacement for cfa7a46's ``scatter_add_`` (cost_function.py:236-239), whose CUDA
atomics add in a varying order, so goal_in_xy was not bitwise run to run.

HEAD's form has no fixed bits to match, so the new op is gated by: determinism
over repeats (HEAD's form is the positive control), a float64-reference error
within HEAD's own spread and error, bitwise grads, sync-free capture, and
run-to-run bitwise in the real cost function, with and without grad, through
the call site it uses, where the particle grads are the ones cfa7a46's form gives."""

import os

import numpy as np
import pytest
import torch

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda

pytestmark = pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")

# (200, 56) has a block over 128 columns: the reduction's vectorized path (_per_object_sum's docstring).
_CASES = [(counts, b) for counts in ((50, 50), (37, 50, 13), (200, 56)) for b in (64, 256)]


def _block_index(counts, b):
    """HEAD's sphere -> object map (cost_function.py:127-135), expanded to ``b`` rows."""
    return torch.tensor([o for o, n in enumerate(counts) for _ in range(n)], device="cuda")[None].expand(b, -1)


def _head_sum(x, idx, k):
    """HEAD's form (cost_function.py:236-239): atomic adds in a varying order."""
    return torch.zeros(x.shape[0], k, dtype=x.dtype, device=x.device).scatter_add_(1, idx, x)


def _not_equal(got, want):
    """The entries of ``got`` that are not ``torch.equal`` to ``want``'s, with their max abs diff."""
    assert got.keys() == want.keys(), got.keys() ^ want.keys()
    return {k: (got[k].double() - want[k].double()).abs().max().item()
            for k in want if not torch.equal(got[k], want[k])}


def _grads(out):
    """The ``("grad", name)`` entries of a :func:`test_cost_function_is_run_to_run_bitwise` output."""
    return {k: v for k, v in out.items() if k[0] == "grad"}


@pytest.fixture
def _numpy_rng_restored():
    """Puts numpy's global RNG back as the test found it, so the tests after one
    that seeds it keep their own draws."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@needs_cuda
def test_per_object_sum_is_fixed_order():
    from cutamp.cost_function import _per_object_sum
    cases = []
    for counts, b in _CASES:  # 1-3 hold whatever HEAD's scatter_add_ does
        g = torch.Generator(device="cuda").manual_seed(b + len(counts))
        x = torch.rand(b, sum(counts), device="cuda", generator=g)
        gy = torch.rand(b, len(counts), device="cuda", generator=g)
        idx, k = _block_index(counts, b), len(counts)
        want = _per_object_sum(x, counts)
        # 1. Determinism over repeats.
        assert all(torch.equal(_per_object_sum(x, counts), want) for _ in range(200)), (counts, b)
        # 2. Bitwise grads: the gather scatter_add_'s backward gives, for any upstream grad.
        xr = x.clone().requires_grad_()
        got, = torch.autograd.grad(_per_object_sum(xr, counts), xr, gy)
        head_got, = torch.autograd.grad(_head_sum(xr, idx, k), xr, gy)
        assert torch.equal(got, gy.gather(1, idx)) and torch.equal(got, head_got), (counts, b)
        # 3. Sync-free, forward and backward (the capture rule).
        torch.cuda.set_sync_debug_mode("error")
        try:
            torch.autograd.grad(_per_object_sum(xr, counts), xr, gy)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        cases.append((counts, b, x, idx, k, want))
    if torch.are_deterministic_algorithms_enabled():
        pytest.skip("deterministic algorithms are on, so HEAD's scatter_add_ takes its deterministic kernel and has "
                    "no atomics to measure: skipped the positive control, the spread bound and the HEAD-error bound "
                    "(checks 1-3 ran and passed)")
    for counts, b, x, idx, k, want in cases:  # 4-5 measure HEAD's atomics
        # 4. The positive control: HEAD's form is not deterministic.
        heads = torch.stack([_head_sum(x, idx, k) for _ in range(200)])
        assert not all(torch.equal(h, heads[0]) for h in heads[1:]), (
            f"{counts}, B={b}: HEAD's form repeated bitwise 200 times, so check 1 has no power on this GPU")
        # 5. The right block sums, no less accurate than HEAD: the float64 error is within
        #    HEAD's spread over its 200 calls, and within HEAD's own error.
        ref = torch.zeros(b, k, dtype=torch.float64).scatter_add_(1, idx.cpu(), x.double().cpu())  # CPU: sequential
        err = (want.double().cpu() - ref).abs().max().item()
        spread = (heads.amax(0) - heads.amin(0)).max().item()
        head_err = (heads.double().cpu() - ref).abs().max().item()
        assert err <= head_err, f"{counts}, B={b}: B0 error {err:.3g} exceeds HEAD's {head_err:.3g}"
        assert err <= spread, (
            f"{counts}, B={b}: HEAD's spread {spread:.3g} is below B0's error {err:.3g}: "
            "HEAD's atomics vary too little on this GPU for the spread bound")


@needs_cuda
def test_per_object_sum_replays_like_eager():
    from cutamp.cost_function import _per_object_sum
    g, side = torch.Generator(device="cuda").manual_seed(0), torch.cuda.Stream()
    for counts in ((50, 50), (200, 56)):     # the second has a block over 128 columns
        n, k = sum(counts), len(counts)
        x = torch.rand(64, n, device="cuda", generator=g).requires_grad_()
        gy = torch.rand(64, k, device="cuda", generator=g)

        def body():
            y = _per_object_sum(x, counts)
            return y, torch.autograd.grad(y, x, gy)[0]

        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side, capture_error_mode="global"):
            y_static, gx_static = body()
        for r in range(10):                  # fresh inputs each time, as Adam moves them
            with torch.no_grad():
                x.copy_(torch.rand(64, n, device="cuda", generator=g))
                gy.copy_(torch.rand(64, k, device="cuda", generator=g))
            graph.replay()
            y, gx = body()
            assert torch.equal(y_static, y) and torch.equal(gx_static, gx), (counts, r)


@needs_cuda
def test_cost_function_is_run_to_run_bitwise(monkeypatch, _numpy_rng_restored):
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_function import CostFunction
    from cutamp.cost_reduction import CostReducer
    from cutamp.optimize_plan import _adam_consider_types
    from cutamp.particle_initialization import ParticleInitializer
    from cutamp.rollout import RolloutFunction
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.t1_domain import all_t1_operators
    from cutamp.task_planning import task_plan_generator

    np.random.seed(0)  # the world's block spheres are sampled from numpy's global RNG: a recreatable state
    world = make_blocks_t1_world(collision_activation_distance=0.0, ik_max_batch_size=64)
    skeleton = next(task_plan_generator(world.initial_state, world.goal_state, all_t1_operators))
    where = f"skeleton {' | '.join(map(str, skeleton))}; PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}"
    config = TAMPConfiguration(num_particles=64)
    torch.manual_seed(0)
    particles = ParticleInitializer(world, config)(skeleton)
    rollout_fn, cost_fn = RolloutFunction(skeleton, world, config), CostFunction(skeleton, world, config)
    reducer = CostReducer(default_constraint_to_mult.copy())
    checker = ConstraintChecker(default_constraint_to_tol.copy())

    # The init state, where sums have the most nonzero sphere terms. Asserted: 8 repeats bitwise
    # equal to `want`. With cfa7a46's atomics patched in at this seeded state, a probe's 400 calls
    # gave 28-39 distinct goal_in_xy results (PYTHONHASHSEED 0, 2) and 5-8 (PYTHONHASHSEED 1, the
    # other skeleton); in its 5-result run, 8 repeats would all agree with probability 0.25.
    def outputs():
        with torch.no_grad():
            cost_dict = cost_fn(rollout_fn(particles), particles)
            out = {(c, n): v.clone() for c, e in cost_dict.items() for n, v in e["values"].items()}
            out["costs"] = reducer(cost_dict, consider_types=_adam_consider_types(config)).clone()
            out["mask"] = checker.get_mask(cost_dict, verbose=False).clone()
        return out

    outputs()                                   # warm-up: TorchScript's profiling run
    want = outputs()
    assert ("StablePlacement", "goal_in_xy") in want
    for r in range(8):
        bad = _not_equal(outputs(), want)
        assert not bad, f"repeat {r}: not bitwise run to run: {bad} ({where})"   # RED at cfa7a46: goal_in_xy
    for surface, placements in cost_fn.surface_to_placements.items():  # the blocks the split assumes
        counts = cost_fn.surface_to_sphere_counts[surface]
        assert counts == tuple(world.get_collision_spheres(o).shape[0] for o, _ in placements)
        assert all(type(n) is int for n in counts)                  # Python ints: no host read per call
        assert cost_fn.surface_to_sphere_idx_map[surface].tolist() == [
            i for i, n in enumerate(counts) for _ in range(n)]     # the amin's map agrees with the blocks

    # The call site: one _per_object_sum call per surface, on that surface's blocks,
    # and its result is the value. An inline replacement there fails this.
    from cutamp.cost_function import _per_object_sum
    calls = []

    def recording(x, counts):
        result = _per_object_sum(x, counts)
        calls.append((x.shape, counts, result))
        return result

    with monkeypatch.context() as m:
        m.setattr("cutamp.cost_function._per_object_sum", recording)
        with torch.no_grad():
            values = cost_fn(rollout_fn(particles), particles)["StablePlacement"]["values"]
    assert len(calls) == len(cost_fn.surface_to_placements), [c[:2] for c in calls]
    for surface, (shape, counts, result) in zip(cost_fn.surface_to_placements, calls):
        assert counts is cost_fn.surface_to_sphere_counts[surface] and shape == (64, sum(counts)), (surface, shape)
        assert values[f"{surface}_in_xy"] is result, surface

    def grad_outputs():  # with grad, on fresh leaf clones: TorchScript specializes on requires_grad
        leaves = {k: v.detach().clone().requires_grad_(v.is_floating_point()) for k, v in particles.items()}
        cost_dict = cost_fn(rollout_fn(leaves), leaves)
        costs = reducer(cost_dict, consider_types=_adam_consider_types(config))
        out = {(c, n): v.detach().clone() for c, e in cost_dict.items() for n, v in e["values"].items()}
        out["costs"] = costs.detach().clone()
        out["mask"] = checker.get_mask(cost_dict, verbose=False).clone()
        costs.mean().backward()                 # the Adam loop's loss (optimize_plan.py:706-708, :777)
        out.update({("grad", k): v.grad.clone() for k, v in leaves.items() if v.grad is not None})
        return out

    grad_outputs()                              # warm-up, as for the no_grad repeats
    want = grad_outputs()
    assert _grads(want)
    for r in range(8):
        bad = _not_equal(grad_outputs(), want)
        assert not bad, f"grad-enabled repeat {r}: not bitwise run to run: {bad} ({where})"

    # In situ, B0 leaves the particle grads as cfa7a46's form gives them: its zeros + scatter_add_,
    # on each surface's own idx map. Only the grads are compared: that form's values vary.
    idx_for = {c: cost_fn.surface_to_sphere_idx_map[s][None].expand(64, -1)
               for s, c in cost_fn.surface_to_sphere_counts.items()}
    with monkeypatch.context() as m:
        m.setattr("cutamp.cost_function._per_object_sum", lambda x, c: _head_sum(x, idx_for[c], len(c)))
        head = grad_outputs()
    bad = _not_equal(_grads(head), _grads(want))
    assert not bad, f"particle grads under cfa7a46's scatter_add_ differ from B0's: {bad} ({where})"
