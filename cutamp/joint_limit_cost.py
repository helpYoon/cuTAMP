"""The joint-limit soft cost, and the Trunk tilt bound.

:func:`joint_limit_margin_soft_cost` is cuTAMP's particle-side soft cost over
segment-endpoint configurations: a hinge² penalty for leaving the joint
limits, which are bfm_wm's soft limits (robots/t1.py), so it has no margin
band. Empirical analysis of saved plans showed 100% of near-limit danger
comes from endpoint configs (IK/particle output), never from trajopt
excursions.

Spec: docs/superpowers/specs/2026-06-11-joint-limit-margin-design.md

The Trunk tilt bound (design call 22), which an envelope may set (route A's sets none), has three consumers of
:func:`trunk_tilt_cosine`: the Motion value ``trunk_tilt`` over the confs (cutamp/cost_function.py) and
:class:`TrunkTiltCost` on the motion planner, both through :func:`trunk_tilt_excess`, and GN's tilt row
(cutamp/gn_refresh.py), which projects the covered confs onto the bound. A box on the base's pitch and roll would
cut a pure bow to the bound over sqrt(2), so the tilt is a cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_base_cfg import BaseCostCfg


def shrunken_bounds_penalty(q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Elementwise hinge² penalty for leaving the bounds.

    Zero strictly inside ``[lower, upper]``; quadratic in the penetration
    depth outside. ``q`` is ``[..., dof]``; bounds are ``[dof]`` (broadcast).
    Pure fixed-shape tensor ops — CUDA-graph capture-safe.
    """
    lower_pen = torch.relu(lower - q)
    upper_pen = torch.relu(q - upper)
    return lower_pen.square() + upper_pen.square()


# Initial-state confs are frozen at the robot's actual current state, and
# Phase-2 LBFGS re-leafs every cloned particle with requires_grad_(True)
# (optimize_plan.py) — the initial state must never acquire limit gradients.
# Keep in sync with the initial-conf set in optimize_plan.py (q0/left_q0/
# right_q0) and cost_function.py — bare "q0" appears in single-arm domains.
_EXCLUDED_CONF_NAMES = ("q0", "left_q0", "right_q0")


def joint_limit_margin_soft_cost(
    particles: Dict[str, torch.Tensor],
    joint_limits: torch.Tensor,
    num_particles: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-particle limit penalty summed over Conf particles and DOFs.

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
    return shrunken_bounds_penalty(qs, joint_limits[0], joint_limits[1]).sum(dim=(1, 2))


def trunk_tilt_cosine(q: torch.Tensor, pitch_index: int, roll_index: int) -> torch.Tensor:
    """``q.shape[:-1]``: ``cos(pitch) cos(roll)``, the cosine of the Trunk's tilt from the vertical.

    The base's chain is pitch, then roll, then yaw (YXZ), so the Trunk's z axis keeps ``cos(pitch) cos(roll)``
    of the vertical, whatever the yaw. ``q`` is ``[..., dof]``. Capture-safe."""
    return torch.cos(q[..., pitch_index]) * torch.cos(q[..., roll_index])


def trunk_tilt_excess(q: torch.Tensor, pitch_index: int, roll_index: int, cos_max: float) -> torch.Tensor:
    """``q.shape[:-1]``: ``relu(cos_max - trunk_tilt_cosine(q))``, positive where the Trunk tilts past the bound
    whose cosine is ``cos_max``. Capture-safe."""
    return torch.relu(cos_max - trunk_tilt_cosine(q, pitch_index, roll_index))


@dataclass
class TrunkTiltCostCfg(BaseCostCfg):
    #: The base's pitch and roll columns in the planner's active cspace.
    pitch_index: int = field(kw_only=True)
    roll_index: int = field(kw_only=True)
    #: The cosine of the envelope's maximum tilt.
    cos_max: float = field(kw_only=True)

    class_type: Type = None  # set after TrunkTiltCost definition below


class TrunkTiltCost(BaseCost):
    def forward(self, state) -> torch.Tensor:
        """``[batch, horizon, 1]``: ``weight * trunk_tilt_excess(q)**2``, zero while the tilt is within the bound."""
        excess = trunk_tilt_excess(
            state.joint_state.position, self.config.pitch_index, self.config.roll_index, self.config.cos_max
        )
        return excess.square().unsqueeze(-1) * self._weight


TrunkTiltCostCfg.class_type = TrunkTiltCost
