"""The weights of the requirements a stepped plan keeps as costs (2026-11-23 design 4.3), the straight lines that
carry a step's checks between refreshes (design 4.4), and the plan's preferences (design 4.5)."""

import torch
from curobo.types import JointState
from torch import Tensor

from cutamp.stance import UNIT_M, UNIT_RAD
from cutamp.step_refresh import NUDGE

GROWTH = 2.0  # a violated requirement's weight at each refresh
WEIGHT_CAP = 1e6  # times the weight a row starts at
RANK_BASE = 4096.0  # one unit of side lean outranks this many units of off centre


def side_lean(q: Tensor, stance: Tensor, model) -> Tensor:
    """The sine of the side lean ``[...]`` of postures ``q`` ``[..., dof]`` on ``stance`` ``[..., 2, 3]`` (each sole's
    x, y, yaw): the trunk's own up axis (``base_j_pitch``, ``base_j_roll``) along the left of the two soles' mean
    heading. Positive toward the robot's left."""
    p = q[..., model.joint_names.index("base_j_pitch")]
    r = q[..., model.joint_names.index("base_j_roll")]
    yaw = stance[..., 2].to(q)
    h = torch.atan2(yaw.sin().mean(-1), yaw.cos().mean(-1))
    return -h.sin() * r.cos() * p.sin() + h.cos() * -r.sin()


def joint_effort(world, q: Tensor) -> Tensor:
    """Joint effort ``[n, H]`` of postures ``q`` ``[n, H, dof]``: at rest, each joint outside the base its torque
    (``world.segment_torques``) over its limit, squared, summed. The torque model gives no gradient, so where ``q``
    requires one the value carries its slope by central differences of ``h`` on every joint, all ``1 + 2 dof`` rows of
    every posture in one call."""
    torques, dof, h = world.segment_torques, q.shape[-1], 1e-3
    joints = [k for k in range(dof) if k not in range(dof)[world.model.base]]
    flat = q.detach().reshape(-1, dof)
    rows = flat
    if q.requires_grad:
        nudge = torch.eye(dof, device=q.device, dtype=q.dtype) * h
        rows = torch.cat((flat, (flat[:, None] + nudge).flatten(0, 1), (flat[:, None] - nudge).flatten(0, 1)))
    rows = rows.contiguous()
    rest = torch.zeros_like(rows)
    tau = torques(JointState(position=rows, velocity=rest, acceleration=rest.clone(),
                             joint_names=list(world.model.joint_names)))
    effort = (tau[:, joints] / torques.effort[joints]).square().sum(-1)
    value = effort[:len(flat)].reshape(q.shape[:-1])
    if not q.requires_grad:
        return value
    up, down = effort[len(flat):].reshape(2, *q.shape)
    slope = (up - down) / (2 * h)
    return value.detach() + ((q - q.detach()) * slope).sum(-1)


def plan_rank(lean: Tensor, off_centre: Tensor) -> Tensor:
    """A plan's place in the strict order of the preferences, smaller first: whole units of its worst side ``lean``
    (radians), then whole units of its worst ``off_centre`` (metres)."""
    return (lean / UNIT_RAD).floor() * RANK_BASE + (off_centre / UNIT_M).floor().clamp(max=RANK_BASE - 1)


def lines(values: Tensor, n: int, ticks_changed: Tensor, finite: Tensor) -> tuple[Tensor, Tensor]:
    """Each check's value and slope in the commands, from ``values`` ``[(1 + c) n, K]``: the ``K`` checks on the
    particles' own rows (block 0), then on each copy with command number ``k`` nudged by ``NUDGE`` (block ``k + 1``).

    Returns ``(value [n, K], slope [n, K, c])``: block 0 with entries not finite as zero, and each copy's change from it
    per ``NUDGE``, zero where that nudge changed a tick count (``ticks_changed`` ``[n, c]``), where the particle's own
    row or the copy is not ``finite`` (``[(1 + c) n]``), or where the change is not finite."""
    blocks, trusted = values.reshape(-1, n, values.shape[-1]), finite.reshape(-1, n)
    slope = ((blocks[1:] - blocks[:1]) / NUDGE).permute(1, 2, 0)
    keep = (~ticks_changed & trusted[1:].T & trusted[:1].T)[:, None] & slope.isfinite()
    return blocks[0].nan_to_num(0.0, posinf=0.0, neginf=0.0), torch.where(keep, slope, 0.0)


def start_weight(cost_type: str, name: str, reducer) -> float:
    """The weight a row starts at: 0 for a held row, one per unit for the rows measured in metres, else the configured
    multiplier (1 where there is none)."""
    if cost_type == "KinematicConstraint":
        return 0.0
    if (cost_type, name) in (("ComComfort", "margin_shortfall"), ("Motion", "self_collision"),
                             ("Motion", "retract_self_collision")):
        return 1 / UNIT_M
    multiplier = reducer._get_multiplier(cost_type, name)
    return 1.0 if multiplier is None else multiplier


class RequirementWeights:
    """A weight per particle and row for every requirement in the cost. :meth:`apply` writes them into the cost and
    remembers which rows are violated; :meth:`grow` doubles the weights of those, once per refresh."""

    def __init__(self, reducer, checker):
        self.reducer, self.checker = reducer, checker
        self.weight, self.start, self.violated = {}, {}, {}

    def apply(self, cost: dict) -> None:
        for cost_type, entry in cost.items():
            if entry["type"] != "constraint":
                continue
            entry["weights"] = weights = {}
            for name, values in entry["values"].items():
                key = (cost_type, name)
                if key not in self.weight or self.weight[key].shape != values.shape:
                    self.start[key] = start_weight(cost_type, name, self.reducer)
                    self.weight[key] = torch.full_like(values.detach(), self.start[key])
                tolerance = self.checker._get_tol(cost_type, name)
                self.violated[key] = (values.detach() - tolerance / 2).clamp_min(0) > 0
                weights[name] = self.weight[key]

    def grow(self) -> None:
        for key, violated in self.violated.items():
            cap = WEIGHT_CAP * self.start[key]
            self.weight[key] = torch.where(violated, self.weight[key] * GROWTH, self.weight[key]).clamp_max(cap)
