"""Lockstep of the in-loop Adam step against HEAD's math.

HEAD's math is cfa7a46 + B0 (ruling 11): goal_in_xy summed per object in a
fixed order, frozen here as :func:`_head_per_object_sum`. No oracle may vendor
``stable_placement_costs`` or restore ``scatter_add_``.

Every in-loop step, one ``_StepBody`` call, first runs twice more on clones of
its particles: once under :func:`_head_math`, and once as the code now runs it
(its twin). The step must match both bitwise, at every step and in every case:
its twin run to run, and HEAD, the gate for edits that change both arms.
"""

import os
from contextlib import contextmanager
from typing import Dict, List, Tuple
from unittest import mock

import pytest
import torch

from cutamp.config import SUPPORTED_SOFT_COSTS
from cutamp.tests.conftest import (
    SpyVis, assert_bitwise, clone_particles, make_plan_info, needs_cuda, run_optimizer, step_outputs, twin_body,
)


class _ResetFK:
    """``kin`` with fresh buffers before every FK call, counted in ``calls``;
    everything else delegates."""

    def __init__(self, kin):
        self._kin = kin
        self.calls = 0

    def compute_kinematics(self, js):
        self.calls += 1
        self._kin.update_batch_size(js.position.shape[0], 1, reset_buffers=True)
        return self._kin.compute_kinematics(js)

    def __getattr__(self, name):
        return getattr(self._kin, name)


# A verbatim copy of cutamp.cost_function._per_object_sum as B0 (4cf62d9) committed it:
# HEAD's goal_in_xy sum, frozen. The lockstep checks production's sum against it on the
# run's own inputs, which are sparse; test_frozen_sum_matches_production checks it on
# dense inputs at the run's layouts and two more, where a reordered sum changes bits.
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
    did; which buffers a call uses doesn't change its values. Yields the shared
    :class:`_ResetFK`."""
    shared = _ResetFK(world.kinematics)
    with mock.patch.object(world, "site_kinematics", lambda name: shared), \
            mock.patch("cutamp.cost_function._per_object_sum", _head_per_object_sum):
        yield shared


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


# The shared kinematics_config tensors the FK kernels take raw pointers to, besides each
# site's own buffers: the forward's 9, then the 5 more the backward takes (curobo's
# _src/curobolib/cuda_ops/kinematics.py:156-237, :325-355).
_FK_CONFIG_TENSORS = (
    "fixed_transforms", "link_spheres", "link_masses_com", "joint_map_type", "joint_map", "link_map",
    "tool_frame_map", "link_sphere_idx_map", "joint_offset_map",
    "link_chain_data", "link_chain_offsets", "joint_links_data", "joint_links_offsets", "joint_affects_endeffector",
)


def _site_ptrs(world) -> dict:
    """Each built FK site's batch, horizon and buffer pointers, and under
    ``"config"`` the pointers of the :data:`_FK_CONFIG_TENSORS` all sites share."""
    def ptr(t):
        return 0 if t is None else t.data_ptr()

    cfg = world.kinematics.config.kinematics_config
    ptrs = {name: (kin._batch, kin._horizon, tuple(map(ptr, kin._buffers.values())))
            for name, kin in world._site_kinematics.items()}
    ptrs["config"] = tuple(ptr(getattr(cfg, n)) for n in _FK_CONFIG_TENSORS)
    return ptrs


def _install_lockstep(monkeypatch, world) -> List[tuple]:
    """Wrap ``_StepBody.__call__`` so every in-loop step checks itself bitwise,
    on clones of its particles: against its twin (:func:`twin_body`), run to
    run, and against :func:`_head_step` under :func:`_head_math`, given the
    body's own functions. Each step also checks that the oracle reached every
    FK call. Returns the list each step appends ``(names, got, ptrs)`` to: the
    optimized params' particle names, the step's :func:`step_outputs`, and
    :func:`_site_ptrs` after it."""
    import cutamp.cost_function as cost_function
    import cutamp.optimize_plan as optimize_plan
    from curobo.kinematics import Kinematics

    orig, orig_fk = optimize_plan._StepBody.__call__, Kinematics.compute_kinematics
    fk_calls, steps = [], []

    def counted_fk(kin, *args, **kwargs):  # every FK call, however the caller got its instance
        fk_calls.append(kin)
        return orig_fk(kin, *args, **kwargs)

    def site_fk(fn):
        """``fn()``, and how many FK calls it made on a site."""
        fk_calls.clear()
        out = fn()
        return out, sum(any(kin is site for site in world._site_kinematics.values()) for kin in fk_calls)

    def lockstep(self):
        k = len(steps)
        name_of = {id(t): n for n, t in self.particles.items()}
        names = [name_of[id(p)] for p in self.params]
        masks = {name_of[i]: m for i, m in self.grad_masks.items() if i in name_of}
        with _head_math(world) as shared:
            assert cost_function._per_object_sum is _head_per_object_sum, f"step {k}"
            ref, ref_sites = site_fk(lambda: _head_step(
                clone_particles(self.particles), self.rollout_fn, self.cost_fn, self.cost_reducer, self.get_mask,
                self.consider_types, self.with_soft, names, masks))
        # The twin runs before the step: its backward overwrites the sites' grad_out, which the
        # step's .grad may hold until optimizer.step().
        twin = twin_body(self)
        again, twin_sites = site_fk(lambda: step_outputs(*orig(twin), twin.params))
        (packed, vis), n_sites = site_fk(lambda: orig(self))
        got = step_outputs(packed, vis, self.params)
        assert ref_sites == 0 and shared.calls == twin_sites == n_sites > 0, (  # the oracle reached every FK call
            f"step {k}: the step made {n_sites} site FK calls and its twin {twin_sites}; "
            f"ref {shared.calls} reset ones, {ref_sites} site ones")
        assert_bitwise(got, again, f"step {k}: run to run")
        assert_bitwise(got, ref, f"step {k}")
        steps.append((names, got, _site_ptrs(world)))
        return packed, vis

    monkeypatch.setattr(Kinematics, "compute_kinematics", counted_fk)
    monkeypatch.setattr(optimize_plan._StepBody, "__call__", lockstep)
    return steps


@needs_cuda
@pytest.mark.parametrize("reik_balance_first,coupled_reik", [(False, True), (True, True), (False, False)])
def test_step_lockstep_vs_head(blocks_t1_plan, monkeypatch, reik_balance_first, coupled_reik):
    # Each in-loop step, on clones: the dedicated never-reset FK sites run to run,
    # then vs HEAD's one shared instance, bitwise. The sites' buffers never move.
    # Coupled re-IK drops every FK-fed Conf from Adam, so no compared grad comes out
    # of an FK backward; with coupled_reik=False, Adam optimizes them and one does.
    from cutamp.config import TAMPConfiguration
    from cutamp.t1_domain import Conf

    world, skeleton, base = blocks_t1_plan
    # The cases share blocks_t1_plan's world, which keeps every site it builds.
    monkeypatch.setattr(world, "_site_kinematics", {})
    config = TAMPConfiguration(num_particles=64, num_opt_steps=50, early_stop_patience=0,
                               reik_balance_first=reik_balance_first, coupled_reik=coupled_reik,
                               force_adam=not coupled_reik, enable_visualizer=False)
    confs = {v for op in skeleton for v, prm in zip(op.values, op.operator.parameters) if prm.type == Conf}
    steps = _install_lockstep(monkeypatch, world)
    print("blocks_t1_plan:", " | ".join(op.name for op in skeleton),
          f"(PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')})")
    # Not the no-op Visualizer, so the body returns vis and each step compares costs, mask and values too.
    run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config, visualizer=SpyVis(config))
    assert len(steps) == 50
    names, got, _ = steps[0]
    if not coupled_reik:  # the gate sees an FK backward
        assert any(n in confs and g is not None and bool(g.ne(0).any()) for n, g in zip(names, got["grads"])), (
            f"no optimized Conf has a nonzero grad: {names}")
    ptrs = [p for _, _, p in steps]
    for k, p in enumerate(ptrs):  # a site that is never built fails by name
        want = {"rollout", "retract", "com"} | ({"gate"} if reik_balance_first and k >= 5 else set())
        assert p.keys() - {"config"} == want, f"step {k}: sites {sorted(p.keys() - {'config'})}, want {sorted(want)}"
    for site in ("rollout", "retract", "com", "config"):
        assert all(p[site] == ptrs[0][site] for p in ptrs), site
    if reik_balance_first:  # refresh #1, at step 4, builds the gate site
        assert all(p["gate"] == ptrs[5]["gate"] for p in ptrs[5:]), "gate"


@needs_cuda
@pytest.mark.parametrize("sc,balance_first", [
    (None, True), *((sc, True) for sc in SUPPORTED_SOFT_COSTS),
    *((sc, False) for sc in ("max_obj_dist", "min_obj_dist", "align_yaw")),
])
def test_step_body_matches_head(blocks_t1_plan, monkeypatch, sc, balance_first):
    # The step vs its twin and HEAD, as test_step_lockstep_vs_head checks it, for each soft
    # cost alone and for none. max_obj_dist, min_obj_dist and align_yaw also run in default
    # mode: the default soft costs never reach their triu gathers.
    from cutamp.config import TAMPConfiguration

    world, skeleton, base = blocks_t1_plan
    config = TAMPConfiguration(num_particles=64, num_opt_steps=5, early_stop_patience=0,
                               reik_balance_first=balance_first, enable_visualizer=False,
                               soft_cost=None if sc is None else [sc])
    steps = _install_lockstep(monkeypatch, world)
    print("blocks_t1_plan:", " | ".join(op.name for op in skeleton),
          f"(PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')})")
    run_optimizer(make_plan_info(world, skeleton, clone_particles(base), config), config, visualizer=SpyVis(config))
    assert len(steps) == 5


@needs_cuda
def test_frozen_sum_matches_production(blocks_t1_plan):
    # Dense inputs at the run's layouts and two more: production's goal_in_xy sum
    # equals the frozen copy bitwise, forward and grad with respect to x.
    from cutamp.config import TAMPConfiguration
    from cutamp.cost_function import CostFunction, _per_object_sum

    world, skeleton, _ = blocks_t1_plan
    layouts = CostFunction(skeleton, world, TAMPConfiguration(num_particles=64)).surface_to_sphere_counts.values()
    g = torch.Generator("cuda").manual_seed(0)
    for counts in [*layouts, (37, 50, 13), (200, 56)]:
        x = torch.rand(64, sum(counts), device="cuda", generator=g, requires_grad=True)
        gy = torch.rand(64, len(counts), device="cuda", generator=g)
        got, want = _per_object_sum(x, counts), _head_per_object_sum(x, counts)
        assert_bitwise([got, *torch.autograd.grad(got, x, gy)], [want, *torch.autograd.grad(want, x, gy)],
                       f"counts {counts}")
