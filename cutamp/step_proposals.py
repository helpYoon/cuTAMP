"""Forward-only GPU screening of nearby foot commands on the actual incoming stance."""

from dataclasses import dataclass, fields, is_dataclass
from functools import partial

import torch

from cutamp.landing_predictor import LandingPrediction
from cutamp.step_ik import solve_stance_ik
from cutamp.step_refinement import _check_deadline, _evaluate
from cutamp.step_set import RHO
from cutamp.step_validation import validate_step_geometry


def _rows(value, indices):
    """Select complete particles, including their physical traces and policy histories."""
    if torch.is_tensor(value):
        return value.index_select(0, indices)
    if is_dataclass(value):
        return type(value)(**{f.name: _rows(getattr(value, f.name), indices) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(_rows(v, indices) for v in value)
    return value


@dataclass
class LandingChoices:
    commands: torch.Tensor
    soles: torch.Tensor
    postures: torch.Tensor
    centering_m: torch.Tensor
    particle_indices: torch.Tensor
    predictions: LandingPrediction
    batch_prediction: LandingPrediction | None = None

    def prediction(self, index):
        """Return the exact evaluated rollout, retaining a singleton batch dimension."""
        if not 0 <= index < len(self.commands):
            raise IndexError(index)
        return _rows(self.predictions, self.particle_indices.new_tensor([index]))


def _scene_rows(poses, rows):
    return {name: value[rows] if value.ndim == 3 and len(value) > 1 else value for name, value in (poses or {}).items()}


@torch.no_grad()
def rank_landing_commands(
    world,
    adapter,
    initial,
    order,
    nominal,
    wrist,
    posture_seed,
    arm,
    *,
    object_poses=None,
    held=None,
    target_soles=None,
    count=64,
    max_choices=8,
    margin_m=0.05,
    other_arm_reference=None,
    endpoint_collides=None,
    eligible_sources=None,
    include_fallback=False,
    deadline=None,
):
    """Rank freshly sampled candidates for each particle in a model batch.

    `count` budgets the batch's total proposals, with at least one nominal per
    particle. All output rows pass command bounds, model tracking, stance IK,
    support and full-trace geometry. Supported manipulation paths still require
    endpoint collision and connected motion checks against the selected landing.
    """
    if order not in ("L", "R", "LR", "RL") or not len(initial.dynamics):
        raise ValueError("Expected one or two alternating swing sides from a nonempty state batch")
    if count < 1 or max_choices < 1:
        raise ValueError("Expected positive proposal count and shortlist size")
    context = initial.dynamics
    b = len(context)
    steps = len(order)

    def batch(x, tail):
        return torch.as_tensor(x, device=context.device, dtype=context.dtype).reshape(-1, *tail).expand(b, *tail)

    base = batch(nominal, (steps, 3))
    source_ids = torch.arange(b, device=context.device)
    if eligible_sources is not None:
        if eligible_sources.dtype != torch.bool or eligible_sources.shape != (b,):
            raise ValueError("Eligible sources must be a Boolean mask of the incoming particles")
        source_ids = source_ids[eligible_sources]
    width = len(source_ids)
    if width == 0:
        raise ValueError("No feasible prefix to extend")
    total = max(count, width)
    # Interleave survivors round-robin, keeping their indices in the original particle batch.
    local_owners = torch.arange(total, device=context.device) % width
    owners = source_ids[local_owners]
    sampler = torch.quasirandom.SobolEngine(3 * steps, scramble=True, seed=42)
    samples = sampler.draw(total).to(context).reshape(total, steps, 3)
    # Local perturbations of the planner's proposal in metres, metres, radians.
    commands = base[owners] + (2 * samples - 1) * context.new_tensor([0.09, 0.06, 0.30])
    commands[:width] = base[source_ids]
    if include_fallback:
        if max_choices != 1:
            raise ValueError("Reusing one prediction per particle requires one choice per source")
        # Immediately preceding failures stay in the diagnostics at their nominal command.
        # Evaluate them in this same model call so refinement can reuse every trace.
        nominal_indices = torch.empty(b, device=context.device, dtype=torch.long)
        nominal_indices[source_ids] = torch.arange(width, device=context.device)
        if width < b:
            missing = torch.nonzero(~eligible_sources).flatten()
            nominal_indices[missing] = torch.arange(total, total + len(missing), device=context.device)
            owners = torch.cat((owners, missing))
            commands = torch.cat((commands, base[missing]))
    state = _rows(initial, owners)
    predictor = adapter.predictor
    incoming = predictor.model.feet(state.tensors[:6])[0]
    solve = solve_stance_ik
    if other_arm_reference is not None:
        other = batch(other_arm_reference, (other_arm_reference.shape[-1],))[owners]
        solve = partial(solve_stance_ik, other_arm_reference=other)
    if endpoint_collides is not None:
        solve = partial(solve, collides=partial(endpoint_collides, owners=owners))
    _check_deadline(deadline)
    ev = _evaluate(
        predictor,
        state,
        incoming,
        order,
        commands,
        arm,
        batch(wrist, (4, 4))[owners],
        batch(posture_seed, (33,))[owners],
        margin_m,
        None,
        None,
        None,
        solve_ik=solve,
    )
    prediction = ev.prediction
    soles = prediction.steps[-1].achieved
    center = ev.center[:, 0].norm(dim=-1)
    score = center
    if target_soles is not None:
        delta = soles - batch(target_soles, (2, 3))[owners]
        turn = torch.atan2(delta[..., 2].sin(), delta[..., 2].cos())
        score = (delta[..., :2].square().sum(-1) + (RHO * turn).square()).sum(-1)
    # Sort independently per source particle, then check small equally ranked
    # groups together. Continue down a particle's list only after collisions.
    rounds = (total + width - 1) // width
    ranked_cost = score.new_full((rounds * width,), float("inf"))
    ranked_cost[:total] = torch.where(ev.ok[:total], score[:total], float("inf"))
    ranked_cost = ranked_cost.reshape(rounds, width)
    which = ranked_cost.argsort(dim=0, stable=True)
    ranked = which * width + torch.arange(width, device=context.device)
    eligible = ranked_cost.gather(0, which).isfinite()
    num_selected = torch.zeros(width, dtype=torch.long, device=context.device)
    selected = []
    for start in range(0, rounds, max(8, max_choices)):
        _check_deadline(deadline)
        sl = slice(start, start + max(8, max_choices))
        block = ranked[sl][eligible[sl] & (num_selected < max_choices)]
        if not len(block):
            break
        source = owners[block]
        carried = {
            arm: (name, pose[source] if pose.ndim == 3 and len(pose) > 1 else pose)
            for arm, (name, pose) in (held or {}).items()
        }
        geometry = validate_step_geometry(
            world, adapter, _rows(prediction, block), object_poses=_scene_rows(object_poses, source), held=carried
        )
        accepted = block[geometry["valid"]]
        selected.append(accepted)
        num_selected.scatter_add_(0, local_owners[accepted], torch.ones_like(accepted))
    accepted = torch.cat(selected) if selected else owners[:0]
    # Keep only the requested leading choices for each source (stable within particle).
    result = torch.cat([accepted[local_owners[accepted] == i][:max_choices] for i in range(width)])
    batch_prediction = None
    if include_fallback:
        nominal_indices[owners[result]] = result
        batch_prediction = _rows(prediction, nominal_indices)
    return LandingChoices(commands[result], soles[result], ev.ik.q[result], center[result], owners[result],
                          _rows(prediction, result), batch_prediction)
