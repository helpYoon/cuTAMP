"""The joint-limit soft cost, and the Trunk tilt's cosine.

:func:`joint_limit_margin_soft_cost` is cuTAMP's particle-side soft cost over
segment-endpoint configurations: a hinge² penalty for leaving the joint
limits, which are bfm_wm's soft limits cut by the envelope (robots/t1.py), so
it has no margin band.

:func:`trunk_tilt_cosine` is the cosine of the Trunk's tilt, by which GN IK ranks each target's accepted starts
(cutamp/particle_initialization.py).
"""

from __future__ import annotations

from typing import Dict

import torch


# Initial-state confs are frozen at the robot's actual current state: the
# initial state must never acquire limit gradients.
# Keep in sync with the initial-conf set in optimize_plan.py (q0/left_q0/
# right_q0).
_EXCLUDED_CONF_NAMES = ("q0", "left_q0", "right_q0")


def joint_limit_margin_soft_cost(
    particles: Dict[str, torch.Tensor],
    joint_limits: torch.Tensor,
    num_particles: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-particle limit penalty summed over Conf particles and DOFs: each conf's hinge², zero inside the limits
    and quadratic in the depth past them. Pure fixed-shape tensor ops — CUDA-graph capture-safe.

    Args:
        particles: particle dict. Conf entries are ``[B, dof]`` named
            ``left_q*``/``right_q*``/``q*`` (same filter convention as
            ``minimize_body_movement`` and the COM-polygon helpers), minus
            the excluded initial-state confs.
        joint_limits: ``[2, dof]`` position limits, row 0 = lower.
        num_particles: B, for the no-confs fallback.
        device: device of the returned tensor.

    Returns:
        ``[B]`` non-negative penalty (0 for every conf inside the limits).
    """
    config_params = [
        name
        for name, v in particles.items()
        if name.startswith(("left_q", "right_q", "q"))
        and v.ndim == 2
        and name not in _EXCLUDED_CONF_NAMES
    ]
    if not config_params:
        return torch.zeros(num_particles, device=device)
    qs = torch.stack([particles[name] for name in config_params], dim=1)  # [B, n_confs, dof]
    lower, upper = joint_limits[0], joint_limits[1]
    return (torch.relu(lower - qs).square() + torch.relu(qs - upper).square()).sum(dim=(1, 2))


def trunk_tilt_cosine(q: torch.Tensor, pitch_index: int, roll_index: int) -> torch.Tensor:
    """``q.shape[:-1]``: ``cos(pitch) cos(roll)``, the cosine of the Trunk's tilt from the vertical.

    The base's chain is pitch, then roll, then yaw (YXZ), so the Trunk's z axis keeps ``cos(pitch) cos(roll)``
    of the vertical, whatever the yaw. ``q`` is ``[..., dof]``. Capture-safe."""
    return torch.cos(q[..., pitch_index]) * torch.cos(q[..., roll_index])
