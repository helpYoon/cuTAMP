"""Lockstep of the in-loop Adam step against HEAD's math.

HEAD's math is cfa7a46 + B0 (ruling 11): goal_in_xy summed per object in a
fixed order, frozen here as :func:`_head_per_object_sum`. No oracle may vendor
``stable_placement_costs`` or restore ``scatter_add_``.

Every in-loop step runs three more times on clones of its particles: twice as
the code now runs it, and once under :func:`_head_math`. All three must match
bitwise at every step, in both modes: run to run, and against HEAD, the gate
for edits that change both arms.
"""

from contextlib import contextmanager
from typing import Dict, List, Tuple
from unittest import mock

import pytest
import torch

from cutamp.tests.conftest import (
    assert_bitwise, clone_particles, make_plan_info, needs_cuda, run_optimizer, step_outputs,
)


class _ResetFK:
    """``kin`` with fresh buffers before every FK call; everything else delegates."""

    def __init__(self, kin):
        self._kin = kin

    def compute_kinematics(self, js):
        self._kin.update_batch_size(js.position.shape[0], 1, reset_buffers=True)
        return self._kin.compute_kinematics(js)

    def __getattr__(self, name):
        return getattr(self._kin, name)


# A verbatim copy of cutamp.cost_function._per_object_sum as B0 (4cf62d9) committed it:
# HEAD's goal_in_xy sum, frozen, so a later edit to production's sum fails "vs HEAD".
def _head_per_object_sum(x: torch.Tensor, counts: Tuple[int, ...]) -> torch.Tensor:
    """``[B, len(counts)]``: the sums of ``x`` ``[B, sum(counts)]`` over consecutive
    column blocks of ``counts[i]`` columns, one fixed-order reduction per block.
    Bitwise run to run and under CUDA-graph replay, which ``scatter_add_`` is not:
    its CUDA atomics add in a varying order, so its rounding varies. The backward
    gives each column its block's upstream grad, the gather ``scatter_add_``'s
    backward gives. ``counts`` are Python ints, so there is no host read."""
    return torch.stack([block.sum(-1) for block in x.split(counts, dim=1)], dim=1)


@contextmanager
def _head_math(world):
    """HEAD's math: every FK site on the one shared ``world.kinematics``, and
    :func:`_head_per_object_sum` for goal_in_xy's sum. The reset before each FK
    call keeps two live calls off one set of buffers, the job HEAD's two resets
    did; which buffers a call uses doesn't change its values."""
    shared = _ResetFK(world.kinematics)
    with mock.patch.object(world, "site_kinematics", lambda name: shared), \
            mock.patch("cutamp.cost_function._per_object_sum", _head_per_object_sum):
        yield


def _head_step(particles, rollout_fn, cost_fn, cost_reducer, get_mask, consider_types, with_soft,
               params: List[str], masks: Dict[str, torch.Tensor]) -> dict:
    """One Adam step as HEAD's loop runs it on ``particles`` (optimize_plan.py
    :704-715 without the host read, then the backward and grad masks of
    :777-778), as :func:`step_outputs`. ``params`` names the optimized tensors,
    the only ones whose grads are masked and returned; ``masks`` maps names to
    grad masks."""
    from cutamp.conf_locking import apply_grad_masks

    rollout = rollout_fn(particles)
    cost_dict = cost_fn(rollout, particles)
    costs = cost_reducer(cost_dict, consider_types=consider_types)
    satisfying_mask = get_mask(cost_dict, verbose=False)
    loss = costs.mean()
    scalars = [satisfying_mask.sum().to(costs.dtype), loss, costs.min()]
    if with_soft:
        soft_costs = cost_reducer.soft_costs(cost_dict)
        scalars.append(soft_costs.masked_fill(~satisfying_mask, float("inf")).min())
    packed = torch.stack(scalars)
    loss.backward()
    apply_grad_masks([{"params": [particles[n] for n in params]}], {id(particles[n]): m for n, m in masks.items()})
    return step_outputs(packed, (rollout, cost_dict, costs, satisfying_mask), [particles[n] for n in params])


def _site_ptrs(world) -> dict:
    """Each built FK site's batch, horizon and buffer pointers."""
    return {
        name: (kin._batch, kin._horizon, tuple(0 if b is None else b.data_ptr() for b in kin._buffers.values()))
        for name, kin in world._site_kinematics.items()
    }


@needs_cuda
@pytest.mark.parametrize("reik_balance_first", [False, True])
def test_step_lockstep_vs_head(blocks_t1_plan, monkeypatch, reik_balance_first):
    # Each in-loop step, on clones: the dedicated never-reset FK sites run to run,
    # then vs HEAD's one shared instance, bitwise. The sites' buffers never move.
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.optimize_plan import _adam_consider_types
    from cutamp.rollout import RolloutFunction
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

    world, skeleton, base = blocks_t1_plan
    config = TAMPConfiguration(num_particles=64, num_opt_steps=50, early_stop_patience=0,
                               reik_balance_first=reik_balance_first, enable_visualizer=False)
    plan_info = make_plan_info(world, skeleton, clone_particles(base), config)
    step_args = (plan_info["cost_fn"], CostReducer(default_constraint_to_mult.copy()),
                 ConstraintChecker(default_constraint_to_tol.copy()).get_mask,
                 _adam_consider_types(config), config.soft_cost is not None)
    orig, ptrs = RolloutFunction.__call__, []

    def spy(self, p):
        if torch.is_grad_enabled():  # the in-loop forward: the init and final rollouts run under no_grad
            k, params = len(ptrs), [n for n, t in p.items() if t.requires_grad]
            new = _head_step(clone_particles(p), lambda x: orig(self, x), *step_args, params, {})
            again = _head_step(clone_particles(p), lambda x: orig(self, x), *step_args, params, {})
            with _head_math(world):
                ref = _head_step(clone_particles(p), lambda x: orig(self, x), *step_args, params, {})
            assert_bitwise(again, new, f"step {k}: run to run")
            assert_bitwise(new, ref, f"step {k}")
            ptrs.append(_site_ptrs(world))
        return orig(self, p)

    monkeypatch.setattr(RolloutFunction, "__call__", spy)
    run_optimizer(plan_info, config)
    assert len(ptrs) == 50
    for site in ("rollout", "retract", "com"):
        assert all(p[site] == ptrs[0][site] for p in ptrs), site
    if reik_balance_first:  # refresh #1, at step 4, builds the gate site
        gates = [p.get("gate") for p in ptrs[5:]]
        assert gates[0] is not None and gates == [gates[0]] * len(gates)
