"""One refresh of a gap: the world model and controller run on every particle's commands, and the slope of the landings
with respect to the commands is measured by nudged rows of the same call (design section 6)."""

from dataclasses import dataclass, fields, is_dataclass

import math

import torch
from torch import Tensor

from cutamp.landing_predictor import (
    LandingPrediction,
    LandingState,
    _intended_pair,
    _wrap,
    landing_errors,
    prediction_precision,
)
from cutamp.stance import STEP_TRAVEL, STEP_X_MAX, STEP_Y_RANGE, STEP_YAW_RANGE, abs_pose, rel_pose
from cutamp.step_bound import planner_step_bound, travel

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


REDRAW_ROUNDS = 1000  # a programming safeguard on the rejection draw, not a planner limit (design 4)


def as_commanded(orders, incoming: Tensor, commands: list[Tensor | None]) -> tuple[list[Tensor | None], list[Tensor]]:
    """:meth:`StandIn.stances`'s return shape for landings exactly where commanded: each step's pair is the intended
    pair of its command on the pair before it, through the steps and the gaps."""
    after, final, before = [], [], incoming
    for order, u in zip(orders, commands):
        if not order:
            after.append(None)
        else:
            pairs = []
            for k, foot in enumerate(order):
                before = _intended_pair(before, "LR".index(foot), u[:, k])
                pairs.append(before)
            after.append(torch.stack(pairs, 1))
        final.append(before)
    return after, final


def draw_commands(orders, incoming: Tensor, n: int, generator: torch.Generator) -> list[Tensor | None]:
    """Per gap ``[n, s, 3]`` float32 on ``incoming``'s device (``None`` where no step): each step uniform over the
    trained box (mirrored for a right swing), kept where the step bound allows it and its travel from the sole's pose
    before the step is within ``STEP_TRAVEL``, refused particles redrawn. A later step and a later gap start from the
    earlier draws taken as landed. Drawn on the generator's device."""
    device, dtype = generator.device, torch.float32
    bound = planner_step_bound(device, dtype)
    lo = torch.tensor([-STEP_X_MAX, STEP_Y_RANGE[0], STEP_YAW_RANGE[0]], device=device, dtype=dtype)
    hi = torch.tensor([STEP_X_MAX, STEP_Y_RANGE[1], STEP_YAW_RANGE[1]], device=device, dtype=dtype)
    mirror = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)
    before, out = incoming.to(device, dtype), []
    for j, order in enumerate(orders):
        if not order:
            out.append(None)
            continue
        steps = []
        for k, foot in enumerate(order):
            side = "LR".index(foot)
            left = side == 0
            start = rel_pose(before[:, 1 - side], before[:, side])
            swing_left = torch.full((n,), left, device=device)
            c = torch.empty(n, 3, device=device, dtype=dtype)
            todo = torch.ones(n, dtype=torch.bool, device=device)
            for _ in range(REDRAW_ROUNDS):
                cand = lo + torch.rand(n, 3, generator=generator, device=device, dtype=dtype) * (hi - lo)
                cand = cand if left else cand * mirror
                t = travel(start, cand)
                ok = (bound.g(cand, swing_left) <= 0) & (t >= STEP_TRAVEL[0]) & (t <= STEP_TRAVEL[1])
                take = todo & ok
                c[take] = cand[take]
                todo &= ~ok
                if not todo.any():
                    break
            else:
                raise ValueError(f"gap {j} step {k + 1}: {int(todo.sum())} particles still refused after {REDRAW_ROUNDS} rounds")
            steps.append(c)
            before = _intended_pair(before, side, c)
        out.append(torch.stack(steps, 1).to(incoming.device))
    return out


def command_rows(orders, incoming: Tensor, commands: list[Tensor | None], after_steps: list[Tensor | None]) -> dict[str, Tensor]:
    """The commands' cost rows: per gap ``j`` and step ``k`` (1-based) ``gap{j}_step{k}_`` ``min_travel``,
    ``max_travel``, ``command_bound``, ``achieved_bound``, ``xy_precision`` and ``yaw_precision``, and for a two-step gap
    ``gap{j}_final_xy_precision`` and ``_yaw_precision``. Each is the positive part of the row of the same name in
    :func:`cutamp.step_refinement.landing_constraint_rows`, so 0 where the requirement is met; the landed pairs come
    from ``after_steps``, a step's reference from the intended pair on the pair before it."""
    rows, before = {}, incoming
    for j, order in enumerate(orders):
        if not order:
            continue
        u, landed, intended = commands[j], after_steps[j], before
        bounds = planner_step_bound(u.device, u.dtype)
        for k, foot in enumerate(order):
            name, side = f"gap{j}_step{k + 1}_", "LR".index(foot)
            c = u[:, k]
            foot_left = torch.full((len(c),), side == 0, device=c.device)
            t = travel(rel_pose(before[:, 1 - side], before[:, side]), c)
            reference = _intended_pair(before, side, c)
            intended = _intended_pair(intended, side, c)
            achieved = landed[:, k]
            xy, yaw, _ = landing_errors(achieved, reference)
            rows[name + "min_travel"] = (STEP_TRAVEL[0] - t) / STEP_TRAVEL[0]
            rows[name + "max_travel"] = t / STEP_TRAVEL[1] - 1
            rows[name + "command_bound"] = bounds.g(c, foot_left) / 0.02
            rows[name + "achieved_bound"] = bounds.g(rel_pose(achieved[:, 1 - side], achieved[:, side]), foot_left) / 0.02
            rows[name + "xy_precision"] = xy.square() / 0.020**2 - 1
            rows[name + "yaw_precision"] = yaw.square() / math.radians(1) ** 2 - 1
            before = achieved
        if len(order) > 1:
            xy, yaw, _ = landing_errors(before, intended)
            rows[f"gap{j}_final_xy_precision"] = xy.square() / 0.020**2 - 1
            rows[f"gap{j}_final_yaw_precision"] = yaw.square() / math.radians(1) ** 2 - 1
    return {key: value.clamp_min(0) for key, value in rows.items()}
