"""Benchmark diagnostics comparing the learned objective with dense native verification."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from unittest.mock import patch

import torch

from cutamp import step_adam
from cutamp.landing_predictor import LandingPredictor, landing_errors
from cutamp.step_ik import stance_ik_tolerance
from cutamp.step_sequence import predict_sequence
from cutamp.step_sequence_model import load_student

__all__ = ["load_student", "predict_sequence", "student_objectives"]


@dataclass
class Forecast:
    order: str
    commands: torch.Tensor
    achieved: torch.Tensor
    tracking: torch.Tensor
    contact: torch.Tensor
    phase: torch.Tensor
    head_xy_cm: torch.Tensor
    head_yaw_deg: torch.Tensor

    @classmethod
    def record(cls, order, commands, result):
        with torch.no_grad():
            contact, phase, xy_cm, yaw_deg = [], [], [], []
            for s in result.steps:
                grounded = ((s.sole_height >= -0.005) | ~s.valid_ticks[:, :, None]).all((1, 2))
                landed = (s.sole_height[:, -1].abs() <= 0.005) & (s.sole_tilt[:, -1] <= 0.05)
                contact.append(grounded & landed.all(1))
                ph = s.phases
                phase.append(ph.deadline_ok & ph.support_ok & ph.hold_ok)
                # The teacher measures both readouts with the same FK.
                fk = s.fk_soles[:, -1] if hasattr(s, "fk_soles") else s.achieved
                xy, yaw, _ = landing_errors(fk, s.achieved)
                xy_cm.append(100 * xy.amax(1))
                yaw_deg.append(yaw.rad2deg().amax(1))
            return cls(
                order,
                commands.detach().clone(),
                torch.stack([s.achieved for s in result.steps], 1),
                result.tracking_ok.detach(),
                torch.stack(contact, 1).all(1),
                torch.stack(phase, 1).all(1),
                torch.stack(xy_cm, 1).amax(1),
                torch.stack(yaw_deg, 1).amax(1),
            )


@contextmanager
def student_objectives(model, *, warm_start_ik=False, ik_objective_tol=None):
    """Use learned Adam evaluations and compare their best bindings against native verification."""
    diagnostics = []
    pending = []
    native_rollout = LandingPredictor.rollout
    native_optimize = step_adam.optimize_particles

    def verify(teacher, initial, order, commands, **kwargs):
        result = native_rollout(teacher, initial, order, commands, **kwargs)
        if pending:
            forecast, entry = pending[0]
            if forecast.order == order and torch.equal(forecast.commands, commands):
                pending.pop(0)
                actual = Forecast.record(order, commands, result)
                delta = forecast.achieved - actual.achieved
                xy = delta[..., :2].norm(dim=-1).flatten(1).amax(1) * 100
                turn = delta[..., 2]
                yaw = torch.atan2(turn.sin(), turn.cos()).abs().flatten(1).amax(1).rad2deg()
                entry["replayed_gaps"].append(
                    {
                        "order": order,
                        "landing_p50_xy_cm": xy.median().item(),
                        "landing_p90_xy_cm": xy.quantile(0.9).item(),
                        "landing_p50_yaw_deg": yaw.median().item(),
                        "landing_p90_yaw_deg": yaw.quantile(0.9).item(),
                        "student_tracking_pass": forecast.tracking.sum().item(),
                        "exact_tracking_pass": actual.tracking.sum().item(),
                        "false_pass": (forecast.tracking & ~actual.tracking).sum().item(),
                        "false_fail": (~forecast.tracking & actual.tracking).sum().item(),
                        "student_contact_pass": forecast.contact.sum().item(),
                        "exact_contact_pass": actual.contact.sum().item(),
                        "student_contact_false_fail": (~forecast.contact & actual.contact).sum().item(),
                        "student_contact_false_pass": (forecast.contact & ~actual.contact).sum().item(),
                        "student_phase_pass": forecast.phase.sum().item(),
                        "exact_phase_pass": actual.phase.sum().item(),
                        "student_phase_false_fail": (~forecast.phase & actual.phase).sum().item(),
                        "student_vs_body_fk_p90_xy_cm": forecast.head_xy_cm.quantile(0.9).item(),
                        "student_vs_body_fk_p90_yaw_deg": forecast.head_yaw_deg.quantile(0.9).item(),
                    }
                )
        return result

    def optimize(initial, evaluate, iterations, lr, **options):
        # Each selected row may come from a different Adam iterate. Compare matched command rows.
        history = []
        center_rows = []
        ik_starts = {} if warm_start_ik else None

        def approximate(teacher, state, order, commands):
            result = predict_sequence(model, teacher, state, order, commands)
            history[-1].append(Forecast.record(order, commands, result))
            return result

        def objective(values):
            history.append([])
            tolerance = nullcontext() if ik_objective_tol is None else stance_ik_tolerance(ik_objective_tol)
            with patch.object(LandingPredictor, "rollout", approximate), tolerance:
                violation, center = evaluate(values) if ik_starts is None else evaluate(values, ik_starts=ik_starts)
            center_rows.append((violation.detach() == 0).sum())
            return violation, center

        best, report = native_optimize(initial, objective, iterations, lr, **options)
        which = report["iteration"]
        rows = torch.arange(len(which), device=which.device)
        entry = {
            "batch": len(which),
            "updates": report["updates_completed"],
            "max_updates": iterations,
            "stop_reason": report["stop_reason"],
            "student_rollouts": sum(map(len, history)),
            "selected_iteration_histogram": which.bincount(minlength=report["updates_completed"] + 1).tolist(),
            "student_feasible_rows": int((report["violation"] == 0).sum()),
            "rows_on_centering_objective": torch.stack(center_rows).tolist(),
            "invalid_gradients": report["invalid_gradients"].sum().item(),
            "replayed_gaps": [],
        }
        if "best_feasible_center_m" in report:
            entry["best_feasible_center_m"] = report["best_feasible_center_m"]
        diagnostics.append(entry)
        pending.clear()
        for gap in zip(*history, strict=True):

            def selected(key, candidates=gap):
                return torch.stack([getattr(value, key) for value in candidates])[which, rows]

            keys = ("commands", "achieved", "tracking", "contact", "phase", "head_xy_cm", "head_yaw_deg")
            pending.append((Forecast(gap[0].order, *(selected(key) for key in keys)), entry))
        return best, report

    with patch.object(step_adam, "optimize_particles", optimize), patch.object(LandingPredictor, "rollout", verify):
        yield diagnostics
