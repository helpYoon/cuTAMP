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
from functools import lru_cache
from typing import Type

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_base_cfg import BaseCostCfg


# --- Single source of truth for the COM-over-base-polygon geometry ---
# Two-foot support hull per actual_robot.urdf: 22.3 cm foot length (X, fore/aft)
# × 31.2 cm stance width (Y, lateral). The cost, the post-IK mask, the per-conf
# penalties helper, the ComPolygon tolerance, and the planner/IK cost cfgs all
# reference these so they cannot drift apart.
COM_HALF_EXTENTS = (0.1115, 0.156)   # (X, Y) half-extents in mobile_base_link frame, meters
COM_INSIDE_MARGIN = 0.02             # inside-edge soft-barrier band width, meters
COM_INSIDE_WEIGHT = 1.0              # inside-barrier scale vs the outside quadratic
COM_TOL = 4e-4                       # == COM_INSIDE_WEIGHT * COM_INSIDE_MARGIN**2 (on-boundary penalty)


@lru_cache
def _half_extents(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``COM_HALF_EXTENTS`` on ``device``, made once: ``torch.tensor`` from a
    Python tuple is a blocking host-to-device copy. Shared; never mutate."""
    return torch.tensor(COM_HALF_EXTENTS, device=device, dtype=dtype)


@dataclass
class ComOverBasePolygonCostCfg(BaseCostCfg):
    #: Half-extents of the rectangle in ``mobile_base_link`` frame (X, Y) in
    #: meters, as a ``[2]`` tensor. ``COM_HALF_EXTENTS`` = ``(0.1115, 0.156)``
    #: is the two-foot support hull per actual_robot.urdf.
    half_extents: torch.Tensor = None
    #: Distance inside the edge at which the soft barrier starts (meters).
    #: ``0`` disables the barrier (outside-only penalization).
    inside_margin: float = COM_INSIDE_MARGIN
    #: Relative scale of the inside barrier vs the outside quadratic.
    inside_weight: float = COM_INSIDE_WEIGHT
    #: Tool frame to use as the rectangle's local origin.
    base_link_name: str = "mobile_base_link"

    class_type: Type = None  # set after ComOverBasePolygonCost definition below

    def __post_init__(self):
        super().__post_init__()
        self.half_extents = self.half_extents.to(
            self.device_cfg.device, dtype=self.device_cfg.dtype
        )


def _project_com_to_base(com_world: torch.Tensor, base_T: torch.Tensor) -> torch.Tensor:
    """Project world-frame COM points into the base frame's XY plane.

    Closed-form SE(3) inverse-transform: ``T_inv @ [p; 1] = R^T (p - t)``.
    Single home of the projection, used by :func:`com_polygon_penalty`.

    Args:
        com_world: ``[N, 3]`` COM in world frame.
        base_T: ``[N, 4, 4]`` SE(3) world pose of the base link.

    Returns:
        ``[N, 2]`` (x, y) COM in base frame.
    """
    R = base_T[:, :3, :3]
    t = base_T[:, :3, 3]
    return (R.transpose(-1, -2) @ (com_world - t).unsqueeze(-1)).squeeze(-1)[:, :2]


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
    com_in_base = _project_com_to_base(com_world, base_T)
    offset = torch.abs(com_in_base) - half_extents
    outside = torch.clamp(offset, min=0.0)
    inside = torch.clamp(offset + inside_margin, min=0.0)
    # Inside barrier uses MAX over axes (distance to the NEAREST edge), not sum:
    # near a corner the nearest-edge margin — not the sum of both — is what
    # bounds tip-over. Summing double-counts corners and wrongly rejects
    # COM-feasible configs (see test_com_polygon_penalty_corner_*).
    return (outside ** 2).sum(dim=-1) + inside_weight * (inside ** 2).max(dim=-1).values


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


def _fk_state(world, q_batch: torch.Tensor, kin=None):
    """FK ``q_batch`` (``[B, full_dof]`` or ``[full_dof]``) through ``kin``
    (``None``: ``world.kinematics``); return its kinematics state.
    Differentiable through cuRobo's FK chain."""
    from curobo.types import JointState
    kin = world.kinematics if kin is None else kin
    q_batch = q_batch.to(kin.device_cfg.device)
    if q_batch.ndim == 1:
        q_batch = q_batch.unsqueeze(0)
    js = JointState.from_position(q_batch, joint_names=list(kin.joint_names))
    return kin.compute_kinematics(kin.get_active_js(js))


def _com_and_base_from_state(ks):
    """``(com_world [B,3], base_T [B,4,4])`` from a CoM-enabled kinematics
    state: the inputs every COM-polygon helper needs."""
    com_world = ks.robot_com[..., :3].reshape(-1, 3)
    base_T = (
        ks.tool_poses.get_link_pose("mobile_base_link", make_contiguous=True)
        .get_matrix()
        .reshape(-1, 4, 4)
    )
    return com_world, base_T


def _fk_com_and_base(world, q_batch: torch.Tensor, kin=None):
    """FK ``q_batch`` through ``kin`` (``None``: ``world.kinematics``); return
    ``(com_world [B,3], base_T [B,4,4])`` — the inputs every COM-polygon
    helper needs. Differentiable through cuRobo's FK chain."""
    return _com_and_base_from_state(_fk_state(world, q_batch, kin))


def com_polygon_mask_from_state(ks) -> torch.Tensor:
    """:func:`compute_com_polygon_mask` on a kinematics state that a CoM-enabled
    FK already computed, so a caller that needs the FK for other checks too
    runs it once. ``[B]`` bool."""
    com, base_T = _com_and_base_from_state(ks)
    # Gate through the SAME penalty the hard ComPolygon constraint uses, so the
    # post-IK COM-safe check and ConstraintChecker can never diverge. With the
    # max-over-axes barrier, (penalty <= COM_TOL) == "COM inside-or-on the hull".
    half = _half_extents(com.device, com.dtype)
    pen = com_polygon_penalty(com, base_T, half, COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT)
    return pen <= COM_TOL


def compute_com_polygon_mask(world, q_batch: torch.Tensor) -> torch.Tensor:
    """Batched COM-in-polygon check.

    Computes per-link mass-weighted world COM via ``world.kinematics``
    (which is built with ``compute_com=True``), projects it into the
    ``mobile_base_link`` frame, and returns a boolean mask of which batch
    elements have their COM inside the ``COM_HALF_EXTENTS`` rectangle around
    the base origin.

    Args:
        world: TAMPWorld instance. Must have ``world.kinematics`` (built with ``compute_com=True``)
            available (cached property — first call builds it lazily).
        q_batch: ``[B, full_dof]`` joint positions in
            ``world.kinematics.joint_names`` order.

    Returns:
        ``[B]`` bool tensor; True where the COM-over-polygon penalty is
        ``<= COM_TOL`` (COM inside-or-on the support hull) — the same gate
        the hard ComPolygon constraint uses, so the two cannot diverge.
    """
    return com_polygon_mask_from_state(_fk_state(world, q_batch))


def com_margin_from_base(com_in_base: torch.Tensor) -> torch.Tensor:
    """Static stability margin (McGhee & Frank): min over the two axes of
    ``h - |c|`` for base-frame CoM points ``[N, >=2]``. Positive inside the
    support rectangle, zero on an edge, negative outside. Metres."""
    half = _half_extents(com_in_base.device, com_in_base.dtype)
    return (half - com_in_base[..., :2].abs()).min(dim=-1).values


def compute_com_margin(world, q_batch: torch.Tensor) -> torch.Tensor:
    """``[B]`` stability margins for full-cspace configs via FK.

    Measure-only companion to :func:`compute_com_polygon_mask`; used by the
    ablation study's shared yardstick (recorded in every arm)."""
    com, base_T = _fk_com_and_base(world, q_batch)
    return com_margin_from_base(_project_com_to_base(com, base_T))


def compute_com_polygon_penalties(
    world,
    particles: "dict[str, torch.Tensor]",
) -> "dict[str, torch.Tensor]":
    """Per-conf differentiable COM-polygon penalty.

    For every Conf-shaped particle in ``particles`` (name starts with
    ``left_q``/``right_q``/``q`` and ``ndim == 2``), computes the
    per-particle COM-over-polygon penalty, from one batched FK, via
    :func:`com_polygon_penalty`. Returns a dict mapping conf name to a
    ``[B]`` tensor of penalties in m². Tensors are differentiable
    through cuRobo's FK chain — calling ``.backward()`` on a function of
    these values populates ``.grad`` on the input Conf particles.

    Used by the hard ``ComPolygon`` constraint (via
    :meth:`CostFunction.com_polygon_constraint`); the result is cached on
    the ``CostFunction`` instance so the FK runs once per evaluation.

    Args:
        world: TAMPWorld instance. ``world.kinematics`` must
            be available (cached property — first call builds it).
        particles: dict mapping particle name to ``[B, full_dof]``
            tensor. Non-Conf particles (poses, grasps) are skipped.

    Returns:
        ``{conf_name: [B] float tensor}`` of penalty values in m².
        Empty dict if no Conf particles are in ``particles``.
    """
    conf_names = [
        p for p in particles
        if p.startswith(("left_q", "right_q", "q")) and particles[p].ndim == 2
    ]
    if not conf_names:
        return {}
    # One FK for all confs, on its own site (TAMPWorld.site_kinematics).
    qs = [particles[name] for name in conf_names]
    q_all = torch.cat(qs)
    com_world, base_T = _fk_com_and_base(world, q_all, world.site_kinematics("com"))
    pen = com_polygon_penalty(
        com_world, base_T, _half_extents(com_world.device, com_world.dtype),
        COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT,
    )
    return dict(zip(conf_names, pen.split([q.shape[0] for q in qs])))
