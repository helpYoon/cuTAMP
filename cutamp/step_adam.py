"""Experimental joint Adam refinement of the continuous bindings of a skeleton."""

import torch


def optimize_particles(initial, evaluate, iterations, lr):
    """Keep the best entire iterate per particle, ordered by violation then centering."""
    x = {key: value.detach().clone().requires_grad_() for key, value in initial.items()}
    optimizer = torch.optim.Adam(x.values(), lr=lr)
    best = {k: v.detach().clone() for k, v in x.items()}
    first = next(iter(x.values()))
    minimum = first.new_full((len(first),), torch.inf, dtype=torch.float64)
    best_center = minimum.clone()
    which = torch.zeros_like(minimum, dtype=torch.long)
    report = {"invalid_gradients": torch.zeros_like(which)}
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
    report.update(iteration=which, violation=minimum, center=best_center)
    return best, report
