"""The weights of the requirements a stepped plan keeps as costs (2026-11-23 design 4.3), and the straight lines that
carry a step's checks between refreshes (design 4.4)."""

import torch
from torch import Tensor

from cutamp.stance import UNIT_M
from cutamp.step_refresh import NUDGE

GROWTH = 2.0  # a violated requirement's weight at each refresh
WEIGHT_CAP = 1e6  # times the weight a row starts at


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
