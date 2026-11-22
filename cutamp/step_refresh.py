"""One refresh of a gap: the world model and controller run on every particle's commands, and the slope of the landings
with respect to the commands is measured by nudged rows of the same call (design section 6)."""

from dataclasses import dataclass, fields, is_dataclass

import torch
from torch import Tensor

from cutamp.landing_predictor import LandingPrediction, LandingState, _intended_pair, _wrap, prediction_precision
from cutamp.stance import abs_pose, rel_pose

NUDGE = 1.0e-3  # m and rad: the distance over which the slope is measured (design 6, call 8)


@dataclass
class GapRefresh:
    prediction: LandingPrediction  # the n particles' own rows, with their traces
    landed: Tensor         # [n, s, 2, 3] sole pair after each step, world (x, y, yaw); the last one at the end of the settling hold
    relative: Tensor       # [n, s, 2, 3] the same pairs in the frame of `anchor`
    slope: Tensor          # [n, s, 2, 3, 3 * s] change of `relative` per unit of each command number
    commands: Tensor       # [n, s, 3] the commands it ran at (a detached copy)
    ticks_changed: Tensor  # [n, 3 * s] bool: nudging that number changed some step's tick count


def _first(value, n):
    """``value`` cut to its first ``n`` rows, through dataclasses and tuples; strings and ``None`` pass through."""
    if torch.is_tensor(value):
        return value[:n].clone()
    if is_dataclass(value):
        return type(value)(**{f.name: _first(getattr(value, f.name), n) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(_first(v, n) for v in value)
    return value


def _relative(anchor, pairs):
    """``[B, s, 2, 3]`` sole pairs in the frame of the ``[B, 3]`` sole ``anchor``."""
    return rel_pose(anchor[:, None, None].expand_as(pairs).reshape(-1, 3), pairs.reshape(-1, 3)).reshape(pairs.shape)


def _slope(relative, rows):
    """``[n, s, 2, 3, rows - 1]`` from ``[rows * n, s, 2, 3]``: each nudged block's change from the first block, its yaw
    wrapped, per NUDGE."""
    r = relative.reshape(rows, -1, *relative.shape[1:])
    d = r[1:] - r[:1]
    d[..., 2] = _wrap(d[..., 2])
    return (d / NUDGE).movedim(0, -1)


def refresh_gap(predictor, start: LandingState, order: str, commands: Tensor, anchor: Tensor,
                previous_slope: Tensor | None = None) -> GapRefresh:
    """Run a gap's steps from ``start`` on every particle's commands, values only, and measure their slope.

    One call of ``predictor.rollout``, with its default settling hold, on ``1 + 3 s`` rows per particle: the commands as
    they are, then each command number plus NUDGE. ``anchor`` ``[n, 3]`` is the sole that stays planted in the gap's
    first step. A landing jumps where a step's tick count changes, so where a nudge changed one the slope entry is
    ``previous_slope``'s, or with none the slope of a landing exactly as commanded from the start's own feet.
    """
    with torch.no_grad(), prediction_precision():
        u = commands.detach().clone()
        n, s = u.shape[:2]
        k = 3 * s
        rows = 1 + k
        offsets = torch.cat((u.new_zeros(1, k), NUDGE * torch.eye(k, dtype=u.dtype, device=u.device)))
        big_u = (u.reshape(1, n, k) + offsets[:, None]).reshape(rows * n, s, 3)
        big = LandingState(
            tuple(t.repeat(rows, *([1] * (t.ndim - 1))) for t in start.tensors),
            start.dynamics.repeat(rows, 1),
            start.history_source,
        )
        out = predictor.rollout(big, order, big_u)
        big_anchor = anchor.repeat(rows, 1)
        landed = torch.stack([step.achieved for step in out.steps], 1)
        relative = _relative(big_anchor, landed)
        slope = _slope(relative, rows)
        clocks = torch.stack([step.clock for step in out.steps], -1).reshape(rows, n, s)
        ticks_changed = (clocks[1:] != clocks[:1]).any(-1).T
        fallback = previous_slope
        if fallback is None:
            pair, pairs = predictor.model.feet(start.tensors[:6])[0].repeat(rows, 1, 1), []
            for j, foot in enumerate(order):
                pair = _intended_pair(pair, "LR".index(foot), big_u[:, j])
                pairs.append(pair)
            fallback = _slope(_relative(big_anchor, torch.stack(pairs, 1)), rows)
        slope = torch.where(ticks_changed[:, None, None, None], fallback, slope)
    return GapRefresh(_first(out, n), landed[:n].clone(), relative[:n].clone(), slope, u, ticks_changed)


@dataclass
class StandIn:
    """Every gap's landings between two refreshes: relative pose at the refresh plus slope times the commands' movement,
    placed on the sole that stays planted in the gap's first step (design 6, call 10)."""
    orders: tuple[str, ...]          # per pick or place: "", "L", "R", "LR" or "RL" ("" where it takes no step)
    relative: list[Tensor | None]    # per gap [n, s, 2, 3]
    slope: list[Tensor | None]       # per gap [n, s, 2, 3, 3 * s]
    at: list[Tensor | None]          # per gap [n, s, 3]

    def set(self, gap: int, refresh: GapRefresh) -> None:
        self.relative[gap], self.slope[gap], self.at[gap] = refresh.relative, refresh.slope, refresh.commands

    def stances(self, incoming: Tensor, commands: list[Tensor | None]) -> tuple[list[Tensor | None], list[Tensor]]:
        """``(after_steps, final)``: ``after_steps[j]`` ``[n, s, 2, 3]`` is the world pair after each step of gap ``j``
        (``None`` where it takes no step); ``final[j]`` ``[n, 2, 3]`` is the pair pick or place ``j`` stands on.
        Differentiable in every gap's commands and in ``incoming``."""
        after, final, before = [], [], incoming
        for j, order in enumerate(self.orders):
            if not order:
                after.append(None)
                final.append(before)
                continue
            n, s = commands[j].shape[:2]
            moved = (commands[j] - self.at[j]).reshape(n, 3 * s)
            relative = self.relative[j] + torch.einsum("nsabk,nk->nsab", self.slope[j], moved)
            anchor = before[:, 1 - "LR".index(order[0])]
            world = abs_pose(anchor[:, None, None].expand_as(relative).reshape(-1, 3), relative.reshape(-1, 3))
            after.append(world.reshape(relative.shape))
            final.append(after[-1][:, -1])
            before = final[-1]
        return after, final
