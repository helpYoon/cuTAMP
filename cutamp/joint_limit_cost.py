"""Joint-limit margin costs, and the Trunk tilt bound.

Two consumers share the margin definition (the planning model's
``margin_lower`` / ``margin_upper``, cutamp/robots/t1.py) and the same hinge²
penalty:

* :func:`joint_limit_margin_soft_cost` — cuTAMP particle-side soft cost over
  segment-endpoint configurations. The load-bearing fix: empirical analysis
  of saved plans showed 100% of near-limit danger comes from endpoint
  configs (IK/particle output), never from trajopt excursions.
* :class:`JointLimitMarginCost` — cuRobo-side trajopt cost registered via
  ``cutamp._curobo_internals.add_extra_cost`` (same mechanism as the COM
  polygon cost). Insurance: keeps the trajectory interior away from limits
  between endpoints. Shapes the optimizer objective only — extra costs
  never reach cuRobo's feasibility/success gate.

Margins are per-joint AND per-side: the standing home posture sits exactly
ON the ankle_pitch-upper / knee_pitch-lower / Torso_Pitch-upper bounds, so
those sides carry zero margin (a symmetric margin would penalize standing).

Spec: docs/superpowers/specs/2026-06-11-joint-limit-margin-design.md

The Trunk tilt bound (design call 22) has two consumers of :func:`trunk_tilt_excess` too: the Motion value
``trunk_tilt`` over the real legs' confs (cutamp/cost_function.py), and :class:`TrunkTiltCost` on the motion
planner. A box on the base's pitch and roll would cut a pure bow to the bound over sqrt(2), so the tilt is a cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_base_cfg import BaseCostCfg


def shrunken_bounds_penalty(
    q: torch.Tensor, shrunk_lower: torch.Tensor, shrunk_upper: torch.Tensor
) -> torch.Tensor:
    """Elementwise hinge² penalty for entering the margin bands.

    Zero strictly inside ``[shrunk_lower, shrunk_upper]``; quadratic in the
    penetration depth outside. ``q`` is ``[..., dof]``; bounds are ``[dof]``
    (broadcast). Pure fixed-shape tensor ops — CUDA-graph capture-safe.
    """
    lower_pen = torch.relu(shrunk_lower - q)
    upper_pen = torch.relu(q - shrunk_upper)
    return lower_pen.square() + upper_pen.square()


# Initial-state confs are frozen at the robot's actual current state, and
# Phase-2 LBFGS re-leafs every cloned particle with requires_grad_(True)
# (optimize_plan.py) — the initial state must never acquire margin gradients.
# Keep in sync with the initial-conf set in optimize_plan.py (q0/left_q0/
# right_q0) and cost_function.py — bare "q0" appears in single-arm domains.
_EXCLUDED_CONF_NAMES = ("q0", "left_q0", "right_q0")


def joint_limit_margin_soft_cost(
    particles: Dict[str, torch.Tensor],
    joint_limits: torch.Tensor,
    margin_lower: torch.Tensor,
    margin_upper: torch.Tensor,
    num_particles: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-particle margin penalty summed over Conf particles and DOFs.

    Args:
        particles: particle dict. Conf entries are ``[B, 21]`` named
            ``left_q*``/``right_q*``/``q*`` (same filter convention as
            ``minimize_body_movement`` and the COM-polygon helpers), minus
            the excluded initial-state confs.
        joint_limits: ``[2, 21]`` position limits, row 0 = lower.
        margin_lower: ``[21]`` per-joint margin from the lower bound (rad).
        margin_upper: ``[21]`` per-joint margin from the upper bound (rad).
        num_particles: B, for the no-confs fallback.
        device: device of the returned tensor.

    Returns:
        ``[B]`` non-negative penalty (0 for every conf outside the bands).
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
    qs = torch.stack([particles[name] for name in config_params], dim=1)  # [B, n_confs, 21]
    shrunk_lower = joint_limits[0] + margin_lower
    shrunk_upper = joint_limits[1] - margin_upper
    return shrunken_bounds_penalty(qs, shrunk_lower, shrunk_upper).sum(dim=(1, 2))


@dataclass
class JointLimitMarginCostCfg(BaseCostCfg):
    #: ``[dof_active]`` lower bounds ALREADY shrunk by the per-side margins
    #: (``limits[0] + margin_lower``), in the planner's active cspace order.
    shrunk_lower: Optional[torch.Tensor] = None
    #: ``[dof_active]`` upper bounds already shrunk (``limits[1] - margin_upper``).
    shrunk_upper: Optional[torch.Tensor] = None

    class_type: Type = None  # set after JointLimitMarginCost definition below

    def __post_init__(self):
        super().__post_init__()
        if self.shrunk_lower is None or self.shrunk_upper is None:
            raise ValueError("JointLimitMarginCostCfg requires shrunk_lower and shrunk_upper")
        self.shrunk_lower = self.shrunk_lower.to(
            self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        self.shrunk_upper = self.shrunk_upper.to(
            self.device_cfg.device, dtype=self.device_cfg.dtype
        )


class JointLimitMarginCost(BaseCost):
    def forward(self, state) -> torch.Tensor:
        q = state.joint_state.position  # [batch, horizon, dof_active]
        cost = shrunken_bounds_penalty(
            q, self.config.shrunk_lower, self.config.shrunk_upper
        ).sum(dim=-1, keepdim=True)
        return cost * self._weight  # [batch, horizon, 1] (cost-collection format)


JointLimitMarginCostCfg.class_type = JointLimitMarginCost


def trunk_tilt_excess(q: torch.Tensor, pitch_index: int, roll_index: int, cos_max: float) -> torch.Tensor:
    """``q.shape[:-1]``: ``relu(cos_max - cos(pitch) cos(roll))``, positive where the Trunk tilts past the bound.

    The base turns by yaw, then pitch, then roll, so the Trunk's z axis keeps ``cos(pitch) cos(roll)`` of the
    vertical: the cosine of its tilt, whatever the yaw. ``q`` is ``[..., dof]``. Capture-safe."""
    return torch.relu(cos_max - torch.cos(q[..., pitch_index]) * torch.cos(q[..., roll_index]))


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
