"""Physical execution metrics. Input arrays are measured at common timestamps."""

import hashlib, json
from pathlib import Path
import numpy as np


def manifest(paths):
    return {
        str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in paths
    }


def evaluate_execution(
    time_s,
    tcp_position_error_m,
    tcp_rotation_error_rad,
    *,
    hold_end_s=None,
    position_tolerance_m=0.01,
    rotation_tolerance_deg=5.0,
    hold_duration_s=0.5,
    completed=True,
    stability_violation=False,
    sample_period_s=0.002,
    foot_error_m=None,
    torque=None,
    effort=None
):
    t = np.asarray(time_s)
    p = np.asarray(tcp_position_error_m)
    r = np.asarray(tcp_rotation_error_rad)
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("Strictly increasing time required")
    if p.shape != t.shape or r.shape != t.shape:
        raise ValueError("One active TCP error per timestamp required")
    if np.max(np.diff(t)) > sample_period_s * 1.01:
        raise ValueError("Missing samples or unexpected cadence")
    if foot_error_m is not None and not np.isfinite(foot_error_m).all():
        raise ValueError("Nonfinite feet")
    if torque is not None and (
        effort is None
        or not np.isfinite(torque).all()
        or not np.isfinite(effort).all()
        or np.any(np.asarray(effort) <= 0)
    ):
        raise ValueError("Invalid torque or effort")
    end = float(t[-1] if hold_end_s is None else hold_end_s)
    mask = (t >= end - hold_duration_s - 1e-9) & (t <= end + 1e-9)
    covered = bool(
        mask.sum() > 1 and t[mask][-1] - t[mask][0] >= hold_duration_s - 1e-6
    )
    finite = bool(np.isfinite(p).all() and np.isfinite(r).all())
    result = dict(
        hold_success=bool(
            completed
            and not stability_violation
            and covered
            and finite
            and (p[mask] <= position_tolerance_m).all()
            and (r[mask] <= np.deg2rad(rotation_tolerance_deg)).all()
        ),
        completed=bool(completed),
        hold_span_s=float(np.ptp(t[mask])) if mask.any() else 0.0,
        max_position_error_m=float(p[mask].max()) if mask.any() else None,
        max_rotation_error_deg=float(np.rad2deg(r[mask]).max()) if mask.any() else None,
        tcp_rms_m=float(np.sqrt(np.mean(p * p))),
        execution_duration_s=float(t[-1]),
        gate=dict(
            position_m=position_tolerance_m,
            rotation_deg=rotation_tolerance_deg,
            hold_s=hold_duration_s,
        ),
    )
    if foot_error_m is not None:
        result["foot_max_displacement_m"] = float(np.max(foot_error_m))
    if torque is not None:
        ratio = np.abs(torque) / np.asarray(effort)
        result.update(
            torque_saturation_fraction=float(np.mean(ratio >= 1 - 1e-6)),
            torque_peak_effort_ratio=float(ratio.max()),
        )
    return result
