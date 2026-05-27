"""Center-of-mass over mobile-base support-rectangle cost.

Penalizes the projected COM (in ``mobile_base_link`` frame) when it leaves an
axis-aligned support rectangle, with a soft inside-margin barrier so the
optimizer steers away from the edge instead of bouncing off it.

Three regions:

* Strictly inside the inset rectangle (more than ``inside_margin`` from any
  edge): cost = 0.
* Within ``inside_margin`` of any edge from inside: only the barrier term is
  active.
* Outside the rectangle: both the outside-quadratic and the saturated barrier
  contribute (steepest gradient back to feasibility).

Requires:

* The runtime kinematics built with ``compute_com=True`` so
  ``state.cuda_robot_model_state.robot_com`` is populated. cuTAMP plumbs this
  via ``RobotStateTransitionCfg.compute_com`` (a local cuRobo modification).
* ``mobile_base_link`` registered in ``tool_frames`` so its world pose is
  available from ``state.tool_poses``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_base_cfg import BaseCostCfg


@dataclass
class ComOverBasePolygonCostCfg(BaseCostCfg):
    #: Half-extents of the rectangle in ``mobile_base_link`` frame (X, Y) in
    #: meters. Default ``(0.10, 0.15)`` matches a 20 cm fore/aft × 30 cm
    #: lateral support polygon.
    half_extents: Optional[Union[torch.Tensor, List[float]]] = None
    #: Distance inside the edge at which the soft barrier starts (meters).
    #: ``0`` disables the barrier (outside-only penalization).
    inside_margin: float = 0.02
    #: Relative scale of the inside barrier vs the outside quadratic.
    inside_weight: float = 1.0
    #: Tool frame to use as the rectangle's local origin.
    base_link_name: str = "mobile_base_link"

    class_type: Type = None  # set after ComOverBasePolygonCost definition below

    def __post_init__(self):
        super().__post_init__()
        if self.half_extents is None:
            self.half_extents = [0.10, 0.15]
        if isinstance(self.half_extents, list):
            self.half_extents = torch.tensor(self.half_extents)
        self.half_extents = self.half_extents.to(
            self.device_cfg.device, dtype=self.device_cfg.dtype
        )


def com_polygon_penalty(
    com_world: torch.Tensor,
    base_T: torch.Tensor,
    half_extents: torch.Tensor,
    inside_margin: float,
    inside_weight: float,
) -> torch.Tensor:
    """Per-sample COM-over-rectangle penalty in ``mobile_base_link`` frame.

    Args:
        com_world: ``[N, 3]`` COM in world frame.
        base_T: ``[N, 4, 4]`` SE(3) world pose of ``mobile_base_link``.
        half_extents: ``[2]`` rectangle half-sizes (x, y) in base frame.
        inside_margin: width of the inside soft-barrier band (meters).
        inside_weight: relative scale of the inside barrier vs the outside
            quadratic.

    Returns:
        ``[N]`` non-negative penalty.
    """
    # Closed-form SE(3) inverse-transform: T_inv @ [p; 1] = R^T (p - t).
    R = base_T[:, :3, :3]
    t = base_T[:, :3, 3]
    com_in_base = (R.transpose(-1, -2) @ (com_world - t).unsqueeze(-1)).squeeze(-1)[:, :2]
    offset = torch.abs(com_in_base) - half_extents
    outside = torch.clamp(offset, min=0.0)
    inside = torch.clamp(offset + inside_margin, min=0.0)
    return (outside ** 2).sum(dim=-1) + inside_weight * (inside ** 2).sum(dim=-1)


class ComOverBasePolygonCost(BaseCost):
    def forward(self, state) -> torch.Tensor:
        # cuRobo uses inconsistent batch layouts across kinematics fields:
        # robot_com is [B, H, 4]; tool_poses' link matrices come back
        # flattened to [B*H, 4, 4]. Flatten everything to N=B*H, do the
        # rectangle math, then reshape to [B, H, 1] (cost-collection format).
        cfg = self.config
        b, h = state.joint_state.position.shape[:2]
        n = b * h
        com_world = state.cuda_robot_model_state.robot_com[..., :3].reshape(n, 3)
        base_T = (
            state.tool_poses.get_link_pose(cfg.base_link_name, make_contiguous=True)
            .get_matrix()
            .reshape(n, 4, 4)
        )
        cost = com_polygon_penalty(
            com_world, base_T, cfg.half_extents, cfg.inside_margin, cfg.inside_weight,
        )
        return cost.reshape(b, h, 1) * self._weight


ComOverBasePolygonCostCfg.class_type = ComOverBasePolygonCost


def compute_com_polygon_mask(
    world,
    q_batch: torch.Tensor,
    *,
    half_extents: Optional[List[float]] = None,
) -> torch.Tensor:
    """Batched COM-in-polygon check.

    Computes per-link mass-weighted world COM via ``world.kinematics_with_com``
    (which is built with ``compute_com=True``), projects it into the
    ``mobile_base_link`` frame, and returns a boolean mask of which batch
    elements have their COM inside an axis-aligned rectangle around the
    base origin.

    Args:
        world: TAMPWorld instance. Must have ``kinematics_with_com``
            available (cached property — first call builds it lazily).
        q_batch: ``[B, full_dof]`` joint positions in
            ``world.kinematics.joint_names`` order.
        half_extents: ``[2]``-list of (X, Y) half-sizes in meters. Default
            ``[0.05, 0.10]`` matches the safety rectangle used by the
            planner-side COM cost.

    Returns:
        ``[B]`` bool tensor; True where the projected COM is inside the
        rectangle (per-axis ``abs(com_in_base[i]) <= half_extents[i]``).
    """
    from curobo.types import JointState
    if half_extents is None:
        # Two-foot support hull per actual_robot.urdf: 22.3cm foot length
        # × 31.2cm stance width (left foot Y=+0.106 ± 0.05, right Y=-0.106 ± 0.05).
        # Must match the cost-side polygon set in get_t1_motion_planner /
        # get_t1_ik_solver so the post-IK verification agrees with the
        # gradient pull.
        half_extents = [0.1115, 0.156]
    kin = world.kinematics_with_com
    device = kin.device_cfg.device
    q_batch = q_batch.to(device)
    if q_batch.ndim == 1:
        q_batch = q_batch.unsqueeze(0)
    B = q_batch.shape[0]

    half = torch.tensor(half_extents, device=device, dtype=q_batch.dtype)

    # NOTE: bundled cuRobo's batched COM CUDA kernel has a correctness bug —
    # only batch index 0 of ``robot_com`` is populated reliably; subsequent
    # slots come back as zeros or garbage from buffer-reuse aliasing. Until
    # the kernel is fixed upstream we iterate per-sample with B=1 calls,
    # which always produces a correct COM. The retry wrapper that consumes
    # this helper operates over small candidate batches (≤ a handful), so
    # the perf hit is negligible.
    mask = torch.empty(B, dtype=torch.bool, device=device)
    joint_names = list(world.kinematics.joint_names)
    for i in range(B):
        q_i = q_batch[i:i + 1]  # [1, dof]
        js = JointState.from_position(q_i, joint_names=joint_names)
        active_js = kin.get_active_js(js)
        ks = kin.compute_kinematics(active_js)
        com = ks.robot_com[..., :3].reshape(1, 3)
        base_T = (
            ks.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True)
            .get_matrix()
            .reshape(1, 4, 4)
        )
        R = base_T[:, :3, :3]
        t = base_T[:, :3, 3]
        com_in_base = (R.transpose(-1, -2) @ (com - t).unsqueeze(-1)).squeeze(-1)[:, :2]
        mask[i] = (torch.abs(com_in_base) <= half).all(dim=-1)[0]
    return mask
