"""Grasp-conditioned, model-constrained SQP for a fixed T1 step order.

The command vector holds all three coordinates of every step in the gap. A
candidate update is scored by rerunning its actual clocks and solving full
wrist/sole IK on the resulting stance. Returned masks describe individual rows.
"""

import math
from dataclasses import dataclass

import torch

from cutamp.com_polygon_cost import margin_from_support_planes, support_region_planes
from cutamp.landing_predictor import LandingPrediction, LandingState, prediction_precision
from cutamp.landing_qp import composite_step
from cutamp.robots.t1 import REAL_LEGS
from cutamp.stance import STEP_TRAVEL, abs_pose, rel_pose
from cutamp.step_bound import planner_step_bound, travel
from cutamp.step_ik import StanceIK, solve_stance_ik
from cutamp.step_set import RHO


@dataclass
class GapRefinement:
    commands: torch.Tensor
    initial_gradient: torch.Tensor
    initial_margin: torch.Tensor
    initial_achieved: torch.Tensor
    margin: torch.Tensor
    posture: torch.Tensor
    achieved: torch.Tensor
    prediction: LandingPrediction | None
    ik: StanceIK
    ok: torch.Tensor
    rows: dict[str, torch.Tensor]
    iterations: int


@dataclass
class _Evaluation:
    prediction: LandingPrediction | None
    ik: StanceIK
    rows: dict[str, torch.Tensor]
    ok: torch.Tensor

    def matrix(self, comfort_buffer=0.0):
        b = len(self.ok)

        def with_buffer(name, row):
            if name == "comfort" or name.endswith("_comfort"):
                return row + comfort_buffer
            if comfort_buffer and name.endswith(("_xy_precision", "_yaw_precision")):
                # A strict boundary must yield an inward correction at equality.
                return row + 0.001
            return row

        return torch.cat([with_buffer(k, row).reshape(b, -1) for k, row in self.rows.items()], dim=-1)


def stance_ik_rows(ik: StanceIK, margin_m: float) -> dict[str, torch.Tensor]:
    """Normalized inequalities for a manipulation at its achieved double-support stance."""
    rows = {"comfort": (margin_m - ik.margin) / max(margin_m, 0.01)}
    planes = support_region_planes(ik.soles.double(), REAL_LEGS.sole_box, keyframe=True)
    rows["supported"] = -margin_from_support_planes(ik.com[:, :2], *planes) / 0.02
    for key, tol in (
        ("wrist_pos_error", 0.005),
        ("wrist_rot_error", 0.05),
        ("sole_pos_error", 0.001),
        ("sole_rot_error", 0.01),
        ("optimality", 2e-6),
    ):
        rows[key] = getattr(ik, key) / tol - 1
    return rows


def _evaluate(predictor, initial, incoming, order, u, arm, wrist, seed, margin, extra_rows, anchor, geometry_rows):
    pred = predictor.rollout(initial, order, u) if predictor is not None else None
    before = incoming
    rows = {}
    hard = torch.ones(len(u), dtype=torch.bool, device=u.device)
    if order:
        bounds = planner_step_bound(u.device, u.dtype)
    for j, side in enumerate(order):
        prefix = f"step{j + 1}_"
        c = u[:, j]
        s = "LR".index(side)
        foot = torch.full((len(c),), s == 0, device=c.device)
        start = rel_pose(before[:, 1 - s], before[:, s])
        t = travel(start, c)
        rows[prefix + "min_travel"] = (STEP_TRAVEL[0] - t) / STEP_TRAVEL[0]
        rows[prefix + "max_travel"] = t / STEP_TRAVEL[1] - 1
        rows[prefix + "command_bound"] = bounds.g(c, foot) / 0.02
        dest = abs_pose(before[:, 1 - s], c)
        pair = torch.stack((dest, before[:, 1]) if s == 0 else (before[:, 0], dest), dim=1)
        if pred is not None:
            step = pred.steps[j]
            pair = step.achieved
            ph = step.phases
            profile = predictor.profile
            rows[prefix + "xy_precision"] = step.xy_error.square() / 0.020**2 - 1
            rows[prefix + "yaw_precision"] = step.yaw_error.square() / math.radians(1) ** 2 - 1
            for key in (
                "deadline_xy",
                "deadline_yaw",
                "deadline_height",
                "support_xy",
                "support_yaw",
                "hold_xy",
                "hold_yaw",
            ):
                threshold = getattr(profile, key)
                rows[prefix + key] = (getattr(ph, key) - threshold) / max(threshold, 1e-6)
            rows[prefix + "terminal_sole_height"] = step.sole_height[:, -1].abs() / 0.02 - 1
            rows[prefix + "terminal_sole_tilt"] = step.sole_tilt[:, -1] / 0.10 - 1
            hard = hard & ph.deadline_ok & ph.hold_ok & ph.support_ok
        actual = rel_pose(pair[:, 1 - s], pair[:, s])
        rows[prefix + "achieved_bound"] = bounds.g(actual, foot) / 0.02
        before = pair
    if pred is not None:
        hard = hard & pred.tracking_ok
        if len(order) > 1:
            rows["final_xy_precision"] = pred.final_xy_error.square() / 0.020**2 - 1
            rows["final_yaw_precision"] = pred.final_yaw_error.square() / math.radians(1) ** 2 - 1
    soles = before if anchor is None else anchor(before)
    ik = solve_stance_ik(soles, wrist, seed, arm)
    rows.update(stance_ik_rows(ik, margin))
    if extra_rows is not None:
        more = extra_rows(pred, ik)
        if rows.keys() & more.keys():
            raise ValueError("Scene constraints must have distinct names")
        rows.update(more)
    if geometry_rows is not None:
        geometry = geometry_rows(incoming, order, u)
        if rows.keys() & geometry.keys():
            raise ValueError("Geometric constraint names must be distinct")
        rows.update(geometry)
    hard = hard & ik.ok
    for r in rows.values():
        hard = hard & torch.isfinite(r.reshape(len(u), -1)).all(-1) & (r.reshape(len(u), -1) <= 0).all(-1)
    return _Evaluation(pred, ik, rows, hard)


def _jacobian(rows, u):
    """One reverse sweep per constraint, reusing each particle's realized trajectory.

    Differentiate each original tensor separately. A VJP of a concatenated tensor
    also schedules zero adjoints through every unrelated path and collision kernel.
    """
    b = len(u)
    jac = []
    for row in rows.values():
        for component in row.reshape(b, -1).unbind(1):
            grad = None
            if component.requires_grad:
                (grad,) = torch.autograd.grad(component.sum(), u, retain_graph=True, allow_unused=True)
            jac.append(torch.zeros_like(u) if grad is None else grad)
    return torch.stack(jac, dim=1).flatten(2)


@prediction_precision()
def refine_gap(
    predictor,
    initial,
    order,
    commands,
    arm,
    wrist,
    posture_seed,
    *,
    margin_m=0.05,
    max_iterations=6,
    extra_rows=None,
    anchor=None,
    geometry_rows=None,
):
    """Optimize K commanded soles from a common actual state with one bound wrist per row.

    `extra_rows(prediction, ik)` can append differentiable signed scene, payload or
    later fixed-stance task constraints normalized to <= 0. `anchor` optionally maps
    model-world planar soles into the planner world. With predictor=None, initial is
    a [B,2,3] planar stance and outputs describe a geometric ablation. The
    geometric sweep can append constraints with `geometry_rows(incoming, order, commands)`.
    """
    if order == "none":
        order = ""
    if order not in ("", "L", "R", "LR", "RL"):
        raise ValueError(f"Unsupported step order {order!r}")
    b = len(commands)
    if commands.shape != (b, len(order), 3) or not b:
        raise ValueError("Expected [particles,steps,3] commands")
    if margin_m < 0 or not math.isfinite(margin_m) or max_iterations < 0:
        raise ValueError("Invalid margin or iteration budget")
    if predictor is not None:
        if not isinstance(initial, LandingState):
            raise ValueError("Model-conditioned refinement requires the incoming physical state and history")
        incoming = predictor.model.feet(initial.tensors[:6])[0]
    else:
        incoming = initial
    if incoming.shape != (b, 2, 3):
        raise ValueError("Expected one actual incoming sole pair per particle")

    def evaluate(c):
        return _evaluate(
            predictor,
            initial,
            incoming,
            order,
            c,
            arm,
            wrist,
            posture_seed,
            margin_m,
            extra_rows,
            anchor,
            geometry_rows,
        )

    u = commands.detach().clone()
    first_grad = torch.zeros_like(u)
    iter_count = 0
    initial_margin = None
    initial_achieved = None
    # A feasible iterate needs no further steps in a feasibility-first refinement.
    for i in range(max_iterations + 1):
        with torch.enable_grad():
            u = u.detach().requires_grad_(bool(order))
            cur = evaluate(u)
            if i == 0:
                initial_margin = cur.ik.margin.detach()
                initial_achieved = cur.ik.soles.detach()
                if order:
                    (first_grad,) = torch.autograd.grad(cur.ik.margin.sum(), u, retain_graph=True)
            if cur.ok.all() or not order or i == max_iterations:
                break
            values = cur.matrix()
            jac = _jacobian(cur.rows, u)
        with torch.no_grad():
            width = len(order) * 3
            # The local QP targets a 0.2 mm margin buffer so a finite update can
            # clear the inclusive 5 cm requirement despite linearization error.
            g = cur.matrix(comfort_buffer=0.0002 / max(0.01, margin_m))
            metric = u.new_tensor([1.0, 1.0, RHO]).repeat(len(order)).double()
            limit = u.new_tensor([0.018, 0.018, math.radians(2)]).repeat(len(order)).expand(b, -1).double()
            eye = torch.diag(metric)[None].expand(b, -1, -1)
            offset = ((u - commands).reshape(b, width) * metric).double()
            d, _, _ = composite_step(g.double(), jac.double(), offset, eye, limit, metric)
            d = d.reshape_as(u).to(u)
            finite = torch.isfinite(d).all(-1).all(-1)
            d = torch.where((finite & ~cur.ok)[:, None, None], d, 0)
            score = values.detach().amax(-1).clamp_min(0)
            if not (d.abs() > 1e-8).any():
                break
            old = u.detach()
            chosen = old
            moved = torch.zeros_like(cur.ok)
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = old + scale * d
                fresh = evaluate(trial)
                merit = fresh.matrix().amax(-1).clamp_min(0)
                better = ~moved & ~cur.ok & (fresh.ok | (merit < score - 1e-7))
                chosen = torch.where(better[:, None, None], trial, chosen)
                moved |= better
                if (moved | cur.ok | ~finite).all():
                    break
            if not moved.any():
                break
            u = chosen
            iter_count += 1
    with torch.no_grad():
        u = u.detach()
        fresh = evaluate(u)
    k = fresh.ik
    return GapRefinement(
        u,
        first_grad.detach(),
        initial_margin,
        initial_achieved,
        k.margin,
        k.q,
        k.soles,
        fresh.prediction,
        k,
        fresh.ok,
        fresh.rows,
        iter_count,
    )
