"""Constraint rows of one gap of a fixed T1 step order.

The command vector holds all three coordinates of every step in the gap. A
command is checked by rerunning its actual clocks and solving full wrist/sole
IK on the resulting stance. Returned masks describe individual rows.
"""

import math
from dataclasses import dataclass
from time import monotonic

import torch

from cutamp.com_polygon_cost import margin_from_support_planes, physical_support_centroid, support_region_planes
from cutamp.landing_predictor import LandingPrediction, LandingState, prediction_precision
from cutamp.robots.t1 import REAL_LEGS
from cutamp.stance import STEP_TRAVEL, abs_pose, rel_pose
from cutamp.step_bound import planner_step_bound, travel
from cutamp.step_ik import StanceIK, solve_stance_ik


@dataclass
class GapRefinement:
    commands: torch.Tensor
    margin: torch.Tensor
    posture: torch.Tensor
    achieved: torch.Tensor
    prediction: LandingPrediction | None
    ik: StanceIK
    ok: torch.Tensor
    rows: dict[str, torch.Tensor]
    center_distance: torch.Tensor


@dataclass
class _Evaluation:
    prediction: LandingPrediction | None
    ik: StanceIK
    rows: dict[str, torch.Tensor]
    ok: torch.Tensor
    center: torch.Tensor  # [B, manipulations on this stance, xy]

    def worst_center_squared(self):
        return self.center.square().sum(-1).amax(-1)

    def matrix(self):
        b = len(self.ok)
        return torch.cat([row.reshape(b, -1) for row in self.rows.values()], dim=-1)


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


def stance_center_residual(ik: StanceIK) -> torch.Tensor:
    """Signed xy displacement of the projected robot CoM from the physical hull's area centroid."""
    return ik.com[:, :2] - physical_support_centroid(ik.soles.double(), REAL_LEGS.sole_box)


def landing_constraint_rows(predictor, incoming, order, u, pred):
    """Command domain, phase, per-landing and composed-stance checks on this actual issue state."""
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
    return rows, hard, before


def _evaluate(predictor, initial, incoming, order, u, arm, wrist, seed, margin, extra_rows,
              *, solve_ik=solve_stance_ik, prediction=None):
    pred = prediction
    if predictor is not None and pred is None:
        pred = predictor.rollout(initial, order, u)
    rows, hard, before = landing_constraint_rows(predictor, incoming, order, u, pred)
    ik = solve_ik(before, wrist, seed, arm)
    rows.update(stance_ik_rows(ik, margin))
    centers = [stance_center_residual(ik)]
    if extra_rows is not None:
        more = extra_rows(pred, ik)
        if isinstance(more, tuple):
            more, future_centers = more
            centers.extend(future_centers)
        if rows.keys() & more.keys():
            raise ValueError("Scene constraints must have distinct names")
        rows.update(more)
    hard = hard & ik.ok
    for r in rows.values():
        hard = hard & torch.isfinite(r.reshape(len(u), -1)).all(-1) & (r.reshape(len(u), -1) <= 0).all(-1)
    center = torch.stack(centers, dim=1)
    return _Evaluation(pred, ik, rows, hard & center.isfinite().all(dim=(1, 2)), center)


def _check_deadline(deadline):
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("T1 step refinement exceeded the planner deadline")


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
    extra_rows=None,
    deadline=None,
    solve_ik=solve_stance_ik,
):
    """Check K commanded soles from a common actual state with one bound wrist per row, in one forward evaluation.

    `extra_rows(prediction, ik)` can append signed scene, payload or later
    fixed-stance task constraints normalized to <= 0. It may return
    (hard_rows, future_center_residuals) for every grasp on the achieved
    stance; each residual is an unsquared [B,2] vector. With predictor=None,
    initial is the [B,2,3] planar stance of a gap without steps.
    An absolute monotonic deadline interrupts the evaluation.
    """
    _check_deadline(deadline)
    if order == "none":
        order = ""
    if order not in ("", "L", "R", "LR", "RL"):
        raise ValueError(f"Unsupported step order {order!r}")
    b = len(commands)
    if commands.shape != (b, len(order), 3) or not b:
        raise ValueError("Expected [particles,steps,3] commands")
    if margin_m < 0 or not math.isfinite(margin_m):
        raise ValueError("Invalid margin")
    if predictor is not None:
        if not isinstance(initial, LandingState):
            raise ValueError("Model-conditioned refinement requires the incoming physical state and history")
        incoming = predictor.model.feet(initial.tensors[:6])[0]
    else:
        incoming = initial
    if incoming.shape != (b, 2, 3):
        raise ValueError("Expected one actual incoming sole pair per particle")

    with torch.no_grad():
        u = commands.detach().clone()
        ev = _evaluate(predictor, initial, incoming, order, u, arm, wrist, posture_seed, margin_m, extra_rows,
                       solve_ik=solve_ik)
    k = ev.ik
    _check_deadline(deadline)
    return GapRefinement(u, k.margin, k.q, k.soles, ev.prediction, k, ev.ok, ev.rows, ev.center[:, 0].norm(dim=-1))
