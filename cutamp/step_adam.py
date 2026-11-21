"""Joint Adam refinement of a skeleton's continuous foot commands and placements."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlateauStop:
    tolerance_m: float
    patience: int
    min_updates: int = 0

    def __post_init__(self):
        if not math.isfinite(self.tolerance_m) or self.tolerance_m <= 0:
            raise ValueError("CoM plateau tolerance must be finite and positive")
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("CoM plateau patience must be a positive update count")
        if type(self.min_updates) is not int or self.min_updates < 0:
            raise ValueError("CoM plateau minimum updates must be a nonnegative integer")


def optimize_particles(initial, evaluate, iterations, lr, *, plateau=None, center_weight=1., trace_center=False):
    """Keep the best entire iterate per particle, ordered by violation then centering."""
    x = {key: value.detach().clone().requires_grad_() for key, value in initial.items()}
    optimizer = torch.optim.Adam(x.values(), lr=lr)
    best = {k: v.detach().clone() for k, v in x.items()}
    first = next(iter(x.values()))
    minimum = first.new_full((len(first),), torch.inf, dtype=torch.float64)
    best_center = minimum.clone()
    which = torch.zeros_like(minimum, dtype=torch.long)
    report = {"invalid_gradients": torch.zeros_like(which)}
    if (plateau is not None or trace_center) and (not math.isfinite(center_weight) or center_weight <= 0):
        raise ValueError("CoM plateau tracking requires a finite positive centering weight")
    center_history = []
    anchor_m = math.inf
    last_gain = None
    reason = "max_updates"
    for i in range(iterations + 1):
        with torch.set_grad_enabled(i < iterations):
            violation, center = evaluate(x)
            if i == 0:
                report["initial_violation"] = violation.detach()
            better = (violation < minimum) | ((violation == minimum) & (center < best_center))
            with torch.no_grad():
                for key in x:
                    best[key][better] = x[key][better]
                minimum = torch.where(better, violation, minimum)
                best_center = torch.where(better, center, best_center)
                which[better] = i
            if plateau is not None or trace_center:
                # center is the weighted squared worst endpoint distance of each entire particle.
                valid_center = (minimum == 0) & best_center.isfinite()
                distance = (best_center.masked_fill(~valid_center, torch.inf).min() / center_weight).sqrt()
                if trace_center:
                    center_history.append(distance)
                if plateau is not None:
                    distance_m = distance.item()
                    if math.isfinite(distance_m) and anchor_m - distance_m >= plateau.tolerance_m:
                        anchor_m, last_gain = distance_m, i
                    if last_gain is not None and i >= plateau.min_updates and i - last_gain >= plateau.patience:
                        reason = "plateau"
                        break
            if i == iterations:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = torch.where(violation > 0, violation, center)
            loss.sum().backward()
            valid = torch.ones_like(which, dtype=torch.bool)
            for v in x.values():
                if v.grad is None:
                    raise RuntimeError("A continuous particle parameter is disconnected from the objective")
                valid &= v.grad.flatten(1).isfinite().all(1)
            report["invalid_gradients"] += (~valid).long()
            for v in x.values():
                v.grad[~valid] = 0
            if i == 0:
                report["initial_gradient"] = {k: v.grad.detach().clone() for k, v in x.items()}
            optimizer.step()
    report.update(iteration=which, violation=minimum, center=best_center, updates_completed=i, stop_reason=reason)
    if trace_center:
        report["best_feasible_center_m"] = [x if math.isfinite(x) else None for x in torch.stack(center_history).tolist()]
    return best, report
