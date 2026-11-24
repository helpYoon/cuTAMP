"""One refresh of a gap: the world model and controller run on every particle's commands, and the slope of the landings
with respect to the commands is measured by nudged rows of the same call (design section 6)."""

from dataclasses import dataclass, fields, is_dataclass

import math
from time import monotonic

import torch
from torch import Tensor

from cutamp.landing_predictor import (
    LandingPrediction,
    LandingState,
    _intended_pair,
    _wrap,
    prediction_precision,
)
from cutamp.stance import STEP_TRAVEL, STEP_X_MAX, STEP_Y_RANGE, STEP_YAW_RANGE, abs_pose, rel_pose
from cutamp.step_bound import planner_step_bound, travel, wrap
from cutamp.step_validation import _obb_gap

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


REDRAW_ROUNDS = 1000  # a programming safeguard on the draw's fallback rounds (design 6, step 5), not a planner limit


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


def grid_commands(sole_half, width: float) -> Tensor:
    """``[N, 3]`` float64 CPU left-swing commands (x, y, yaw) in the planted sole's frame, x ascending, then y, then
    turn: the half-foot lattice of the trained box, kept where the step bound allows it.

    Derivation (spec 5): x steps by half a sole length ``sole_half[0]`` within ``STEP_X_MAX`` either way; y steps by
    half a sole width ``sole_half[1]`` from the home stance's ``width`` within ``STEP_Y_RANGE``; the turn steps from 0
    to ``STEP_YAW_RANGE[1]`` in the fewest equal parts each no larger than ``sole_half[1] / sole_half[0]``, the turn
    that swings the toe by half a sole width."""
    hx, hy = float(sole_half[0]), float(sole_half[1])
    kx = math.floor(STEP_X_MAX / hx)
    ky = torch.arange(math.ceil((STEP_Y_RANGE[0] - width) / hy), math.floor((STEP_Y_RANGE[1] - width) / hy) + 1)
    xs = torch.arange(-kx, kx + 1, dtype=torch.float64) * hx
    ys = width + ky.double() * hy
    turns = torch.linspace(0.0, STEP_YAW_RANGE[1], math.ceil(STEP_YAW_RANGE[1] / (hy / hx)) + 1, dtype=torch.float64)
    c = torch.cartesian_prod(xs, ys, turns)
    left = torch.ones(len(c), dtype=torch.bool)
    return c[planner_step_bound(torch.device("cpu"), torch.float64).g(c, left) <= 0]


def feet_clear(feet: Tensor, boxes: Tensor, halves: Tensor, foot_half: Tensor) -> Tensor:
    """``[m]`` bool: no foot of ``feet`` ``[m, k, 3]`` (planar sole poses x, y, yaw) overlaps a box of ``boxes``
    ``[m, b, 4, 4]`` (world from box, half extents ``halves`` ``[b, 3]``) by more than 1e-6 m; boxes whose top is at
    most 1e-6 m are ignored. The refresh's own rule (:func:`cutamp.step_validation._step_geometry`): the foot box of
    half extents ``foot_half`` ``[3]`` lifted to stand on the floor, the same :func:`_obb_gap`, skip and tolerance, one
    box at a time as it takes one object at a time."""
    cos, sin = feet[..., 2].cos(), feet[..., 2].sin()
    foot = torch.zeros(*feet.shape[:-1], 4, 4, dtype=feet.dtype, device=feet.device)
    foot[..., 0, 0], foot[..., 0, 1], foot[..., 1, 0], foot[..., 1, 1] = cos, -sin, sin, cos
    foot[..., 2, 2] = foot[..., 3, 3] = 1.0
    foot[..., :2, 3] = feet[..., :2]
    foot[..., 2, 3] = foot_half[2]
    clear = torch.ones(feet.shape[0], dtype=torch.bool, device=feet.device)
    for i in range(boxes.shape[1]):
        box, oh = boxes[:, i], halves[i]
        top = box[..., 2, 3] + (box[..., 2, :3].abs() * oh).sum(-1)
        overlap = -_obb_gap(foot, foot_half, box[:, None], oh)
        clear &= (torch.where(top[:, None] <= 1e-6, 0.0, overlap).clamp_min(0) <= 1e-6).all(-1)
    return clear


def _check_deadline(deadline):
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("the starting-landing draw reached the planner deadline")


class NoStartingPath(ValueError):
    """No legal, feet-clear grid path was found within ``REDRAW_ROUNDS`` rounds."""


PATH_CANDIDATES = 64  # whole grid paths each particle without a path draws per round (design 6, step 2)
NOISE_TRIES = 16  # noisy copies of each particle's path, in one batch (design 6, step 6)


def draw_grid_paths(orders, incoming: Tensor, generator: torch.Generator, grid: Tensor, reach: list, scenes: list,
                    foot_half: Tensor, deadline: float | None = None) -> tuple[list[Tensor | None], dict]:
    """Each particle's starting commands as one whole grid path (design 6): per gap ``[n, s, 3]`` float32 on
    ``incoming``'s device, right swings mirrored (x, -y, -yaw), ``None`` where no step; and ``info`` ``{"rounds",
    "own", "reached", "noisy"}``: rounds drawn, particles that kept a path of their own, False when nothing reached and
    the paths were drawn without the reach test, particles whose noisy copy passed.

    A path is one left-swing row of ``grid`` (:func:`grid_commands`) per step of every gap of ``orders``, chained from
    the particle's feet ``incoming`` ``[n, 2, 3]`` with :func:`_intended_pair`. It passes when, in order: every step is
    legal (the step bound and both ends of the travel), the pair at the end of each gap ``j`` passes ``reach[j]``
    (``None``: not tested; a callable on ``[m, 2, 3]`` float64 numpy pairs returning bool ``[m]``), and the landed soles
    of each gap ``j``, with its incoming pair for every gap with steps after the first, are clear of ``scenes[j]``
    (``None``: not tested; else ``(boxes [n, b, 4, 4], halves [b, 3])``, :func:`feet_clear`). Every test runs in
    float64 on the CPU; random numbers come from ``generator`` on its own device.

    Rounds draw ``PATH_CANDIDATES`` paths per particle without one and keep each particle's first passing path, until
    every particle has one or a round keeps none; particles left without one take the kept paths in turn, each the first
    from its turn on that passes its own test, else the one its turn names. When the first
    round keeps none, every particle takes a path that passes without the reach test, drawing further rounds up to
    ``REDRAW_ROUNDS`` in all, else :class:`NoStartingPath`. Then each particle keeps the first of ``NOISE_TRIES`` copies
    of its path, each number uniform within half a grid cell cut to the trained box, that passes its own test.
    Before each further round, a ``deadline`` (``time.monotonic()`` seconds) that has passed raises ``TimeoutError``."""
    cpu, f64 = torch.device("cpu"), torch.float64
    start = incoming.detach().to(cpu, f64)
    n = len(start)
    s = sum(len(order) for order in orders)
    bound = planner_step_bound(cpu, f64)
    mirror = torch.tensor([1.0, -1.0, -1.0], dtype=f64)
    half_cell = torch.stack([grid[:, a].unique().diff().min() for a in range(3)]) / 2
    lo = torch.tensor([-STEP_X_MAX, STEP_Y_RANGE[0], STEP_YAW_RANGE[0]], dtype=f64)
    hi = torch.tensor([STEP_X_MAX, STEP_Y_RANGE[1], STEP_YAW_RANGE[1]], dtype=f64)

    def passes(rows: Tensor, paths: Tensor, with_reach: bool) -> Tensor:
        """``[m]`` bool: the path test of left-swing ``paths`` ``[m, s, 3]`` from the feet of particles ``rows``."""
        pair, left = start[rows], torch.ones(len(rows), dtype=torch.bool)
        ok, heads, landed, ends, i = left.clone(), [], [], [], 0
        for order in orders:
            heads.append(pair if i else None)  # a gap's incoming pair is tested only after the first gap with steps
            soles = []
            for foot in order:
                side, c = "LR".index(foot), paths[:, i]
                command = c if side == 0 else c * mirror
                was = rel_pose(pair[:, 1 - side], pair[:, side])
                turn = wrap(command[:, 2] - was[:, 2]).abs()
                near = (command[:, :2] - was[:, :2]).norm(dim=-1) + 0.1124 * turn  # apply_command's yaw factor
                # g <= 0 holds sole_gap >= STEP_SOLE_GAP (a term of trained_box); of the two travels apply_command's
                # (yaw at 0.1124) is the shorter, travel()'s (yaw at RHO) the longer, so each end takes the stricter
                ok &= (bound.g(c, left) <= 0) & (near >= STEP_TRAVEL[0]) & (travel(was, command) <= STEP_TRAVEL[1])
                pair = _intended_pair(pair, side, command)
                soles.append(pair[:, side])
                i += 1
            landed.append(torch.stack(soles, 1) if soles else None)
            ends.append(pair)
        for j, test in enumerate(reach):
            if with_reach and test is not None and ok.any():
                idx = ok.nonzero().squeeze(1)
                ok[idx] = torch.as_tensor(test(ends[j][idx].numpy()), dtype=torch.bool)
        for j, scene in enumerate(scenes):
            if landed[j] is not None and scene is not None and ok.any():
                idx = ok.nonzero().squeeze(1)
                feet = landed[j][idx] if heads[j] is None else torch.cat((heads[j][idx], landed[j][idx]), 1)
                ok[idx] = feet_clear(feet, scene[0][rows[idx]], scene[1], foot_half)
        return ok

    path, own = torch.empty(n, s, 3, dtype=f64), torch.zeros(n, dtype=torch.bool)

    def keep(todo: Tensor, paths: Tensor, with_reach: bool) -> int:
        """Give each particle of ``todo`` its first passing path of ``paths`` ``[len(todo) * PATH_CANDIDATES, s, 3]``;
        how many kept one."""
        hit = passes(todo.repeat_interleave(PATH_CANDIDATES), paths, with_reach).view(len(todo), PATH_CANDIDATES)
        found = hit.any(1)
        path[todo[found]] = paths.view(len(todo), PATH_CANDIDATES, s, 3)[found, hit[found].int().argmax(1)]
        own[todo[found]] = True
        return int(found.sum())

    def candidates(m: int) -> Tensor:
        pick = torch.randint(len(grid), (m * PATH_CANDIDATES, s), generator=generator, device=generator.device)
        return grid[pick.cpu()]

    everyone = torch.arange(n)
    first_round = candidates(n)
    rounds, kept = 1, keep(everyone, first_round, True)
    reached = kept > 0
    while kept and not own.all():  # each round that goes on kept a path, so at most n rounds
        _check_deadline(deadline)
        todo = (~own).nonzero().squeeze(1)
        rounds, kept = rounds + 1, keep(todo, candidates(len(todo)), True)
    mine = int(own.sum())
    if not reached:  # nothing reaches: paths that pass without the reach test
        keep(everyone, first_round, False)
        while not own.all():
            if rounds >= REDRAW_ROUNDS:
                raise NoStartingPath(f"{int((~own).sum())} of {n} particles found no legal, feet-clear grid path "
                                     f"in {REDRAW_ROUNDS} rounds")
            _check_deadline(deadline)
            todo = (~own).nonzero().squeeze(1)
            rounds += 1
            keep(todo, candidates(len(todo)), False)
        mine = n
    lacking = (~own).nonzero().squeeze(1)
    if len(lacking):  # the kept paths in turn: each takes the first from its turn on that passes its own test
        kept = path[own]
        queue = (torch.arange(len(lacking))[:, None] + torch.arange(mine)) % mine  # [L, mine] kept rows, search order
        hit = passes(lacking.repeat_interleave(mine), kept.repeat(len(lacking), 1, 1), True).view(len(lacking), mine)
        first = hit.gather(1, queue).int().argmax(1, keepdim=True)  # 0 where none passes: the plain turn
        path[lacking] = kept[queue.gather(1, first).squeeze(1)]

    centre = path[:, None].expand(n, NOISE_TRIES, s, 3)
    low, high = torch.maximum(centre - half_cell, lo), torch.minimum(centre + half_cell, hi)
    r = torch.rand(n, NOISE_TRIES, s, 3, generator=generator, device=generator.device, dtype=f64).cpu()
    copies = low + r * (high - low)
    hit = passes(everyone.repeat_interleave(NOISE_TRIES), copies.reshape(-1, s, 3), reached).view(n, NOISE_TRIES)
    noisy = hit.any(1)
    path[noisy] = copies[noisy, hit[noisy].int().argmax(1)]

    out, i = [], 0
    for order in orders:
        if not order:
            out.append(None)
            continue
        u = path[:, i:i + len(order)].clone()
        for k, foot in enumerate(order):
            if foot == "R":
                u[:, k] *= mirror
        out.append(u.to(incoming.device, torch.float32))
        i += len(order)
    return out, {"rounds": rounds, "own": mine, "reached": reached, "noisy": int(noisy.sum())}


def command_rows(orders, incoming: Tensor, commands: list[Tensor | None], after_steps: list[Tensor | None]) -> dict[str, Tensor]:
    """The commands' cost rows: per gap ``j`` and step ``k`` (1-based) ``gap{j}_step{k}_`` ``min_travel``,
    ``max_travel``, ``command_bound`` and ``achieved_bound``. Each is the positive part of the row of the same name in
    :func:`cutamp.step_loop.landing_constraint_rows`, so 0 where the requirement is met; the landed pairs come
    from ``after_steps``, a step's reference from the intended pair on the pair before it."""
    rows, before = {}, incoming
    for j, order in enumerate(orders):
        if not order:
            continue
        u, landed = commands[j], after_steps[j]
        bounds = planner_step_bound(u.device, u.dtype)
        for k, foot in enumerate(order):
            name, side = f"gap{j}_step{k + 1}_", "LR".index(foot)
            c = u[:, k]
            foot_left = torch.full((len(c),), side == 0, device=c.device)
            t = travel(rel_pose(before[:, 1 - side], before[:, side]), c)
            achieved = landed[:, k]
            rows[name + "min_travel"] = (STEP_TRAVEL[0] - t) / STEP_TRAVEL[0]
            rows[name + "max_travel"] = t / STEP_TRAVEL[1] - 1
            rows[name + "command_bound"] = bounds.g(c, foot_left) / 0.02
            rows[name + "achieved_bound"] = bounds.g(rel_pose(achieved[:, 1 - side], achieved[:, side]), foot_left) / 0.02
            before = achieved
    return {key: value.clamp_min(0) for key, value in rows.items()}
