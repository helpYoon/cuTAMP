"""Center-of-mass over the support rectangle: the cost and every balance check.

Penalizes the projected COM (in the model's support frame, ``T1Model.support.link``) when it leaves an
axis-aligned support rectangle about that frame's origin, with a soft inside-margin barrier so the
optimizer steers away from the edge instead of bouncing off it.

Three regions:

* Strictly inside the inset rectangle (more than ``COM_INSIDE_MARGIN`` from any
  edge): cost = 0.
* Within ``COM_INSIDE_MARGIN`` of any edge from inside: only the barrier term is
  active.
* Outside the rectangle: both the outside-quadratic and the saturated barrier
  contribute (steepest gradient back to feasibility).

Every helper takes the support (its frame and half-extents) or reads ``world.model.support``: confs
(masks, margins, penalties, GN, the trajopt cost) use ``keyframe_half``, trajectory frames (the balance gate)
``gate_half``.

Requires:

* The runtime kinematics built with ``compute_com=True`` so
  ``state.cuda_robot_model_state.robot_com`` is populated.
  ``floor_wrench.install_floor_wrench`` turns ``Kinematics.compute_com`` on in
  trajopt's transitions.
* The support frame registered in ``tool_frames`` so its world pose is
  available from ``state.tool_poses``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Type

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_base_cfg import BaseCostCfg


# --- The support region's margins, the CoM penalty's barrier and the balance tolerances ---
# The trajectory gate's rectangle (support.gate_half) shrinks the soles by COM_SAFETY_MARGIN for the model and
# tracking errors a real robot adds: every trajectory frame's CoM and ZMP stay inside it (motion_solver._balance_gate).
# The keyframes' rectangle (support.keyframe_half) sits COM_DYNAMIC_RESERVE further in, so a segment can decelerate
# into its end conf without the ZMP leaving. The reserve was set on the simplified model (branch ablation-baseline):
# a conf on the support edge needed 4-7x slowdowns, and a 1 cm reserve left the optimizer no satisfying conf on 2 of
# 4 blocks_t1 seeds. support_rectangles builds the region from them; consumers read it from the model's Support.
COM_SAFETY_MARGIN = 0.02               # model, tracking and sole-edge error budget, meters
COM_DYNAMIC_RESERVE = 0.005            # keyframe CoM room for an arrival's deceleration, meters
COM_INSIDE_MARGIN = 0.02             # inside-edge soft-barrier band width, meters
COM_INSIDE_WEIGHT = 1.0              # inside-barrier scale vs the outside quadratic
COM_TOL = COM_INSIDE_WEIGHT * COM_INSIDE_MARGIN**2  # on-boundary penalty, 4e-4 (known-bugs N5)
LOAD_GUARD_A = 0.1                   # each foot carries at least this share of the weight (design call 5)
# How far a planted sole may leave the stance (design 7.4, decision 5), the hand's IK tolerances: the kinematic
# constraint holds every conf to them (scripts/utils.py), and the balance gate every trajectory frame.
SOLE_POS_TOL = 0.005                 # meters
SOLE_ROT_TOL = 0.05                  # radians


def soles_planted(pos_err: torch.Tensor, rot_err: torch.Tensor, pos_tol: float = SOLE_POS_TOL,
                  rot_tol: float = SOLE_ROT_TOL) -> torch.Tensor:
    """Elementwise: a sole's pose error against the stance within ``pos_tol`` and ``rot_tol``. At the tolerance
    passes, as the constraint checker accepts a value at its tolerance; a NaN error fails."""
    return (pos_err <= pos_tol) & (rot_err <= rot_tol)


def support_rectangles(
    sole_box: Tuple[float, float], sole_y: float, a: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """``(gate_half, keyframe_half)`` about the mid-sole of two side-by-side soles, each a ``sole_box`` (x, y)
    rectangle whose centre sits ``sole_y`` to either side, under the load guard ``a`` (design 7.4).

    The soles' hull is ``|x| <= sole_box[0] / 2``, ``|y| <= sole_y + sole_box[1] / 2``. The load guard's region
    S_a, where some load split gives each foot at least ``a`` of the weight, keeps x and narrows y by
    ``2 sole_y a``. The gate is S_a inside the hull shrunk by ``COM_SAFETY_MARGIN`` (at a = 0.1 the margin binds
    only in x), and the keyframes' rectangle sits ``COM_DYNAMIC_RESERVE`` further in.
    Both soles stand on one stance, so the region is this rectangle in the stance frame."""
    hull = (sole_box[0] / 2, sole_y + sole_box[1] / 2)
    s_a_y = hull[1] - 2 * sole_y * a
    gate = (hull[0] - COM_SAFETY_MARGIN, min(s_a_y, hull[1] - COM_SAFETY_MARGIN))
    keyframe = tuple(h - COM_DYNAMIC_RESERVE for h in gate)
    return gate, keyframe


@lru_cache(maxsize=None)
def _half_extents(half: Tuple[float, float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A support's ``half`` (its ``(x, y)`` half-extents) on ``device``, made once per
    value: ``torch.tensor`` from a Python tuple is a blocking host-to-device copy.
    Shared; never mutate. Never evicted, since a captured CUDA graph keeps
    reading its address."""
    return torch.tensor(half, device=device, dtype=dtype)


@dataclass
class ComOverBasePolygonCostCfg(BaseCostCfg):
    #: Half-extents of the rectangle in the ``base_link_name`` frame (X, Y) in
    #: meters, as a ``[2]`` tensor, about the frame's origin: the model's
    #: ``support.keyframe_half``.
    half_extents: torch.Tensor = None
    #: Tool frame the rectangle is measured in: the model's ``support.link``.
    base_link_name: str = None

    class_type: Type = None  # set after ComOverBasePolygonCost definition below

    def __post_init__(self):
        super().__post_init__()
        if self.half_extents is None or self.base_link_name is None:
            raise ValueError("ComOverBasePolygonCostCfg takes half_extents and base_link_name from the model's support")
        self.half_extents = self.half_extents.to(
            self.device_cfg.device, dtype=self.device_cfg.dtype
        )


def _project_com_to_base(com_world: torch.Tensor, base_T: torch.Tensor) -> torch.Tensor:
    """Project world-frame COM points into the support frame's XY plane. The support rectangle is centred on the
    frame's origin (the stance sits at the home mid-sole).

    Closed-form SE(3) inverse-transform: ``T_inv @ [p; 1] = R^T (p - t)``.
    Single home of the projection, used by :func:`com_polygon_penalty`.

    Args:
        com_world: ``[N, 3]`` COM in world frame.
        base_T: ``[N, 4, 4]`` SE(3) world pose of the support frame.

    Returns:
        ``[N, 2]`` (x, y) COM offset from the support centre, in the support frame's axes.
    """
    R = base_T[:, :3, :3]
    t = base_T[:, :3, 3]
    return (R.transpose(-1, -2) @ (com_world - t).unsqueeze(-1)).squeeze(-1)[:, :2]


def com_polygon_penalty(com_world: torch.Tensor, base_T: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    """Per-sample COM-over-rectangle penalty in the support frame, with the inside barrier ``COM_INSIDE_MARGIN``
    wide at ``COM_INSIDE_WEIGHT`` times the outside quadratic.

    Args:
        com_world: ``[N, 3]`` COM in world frame.
        base_T: ``[N, 4, 4]`` SE(3) world pose of the support frame.
        half_extents: ``[2]`` rectangle half-sizes (x, y) in that frame.

    Returns:
        ``[N]`` non-negative penalty.
    """
    com_in_base = _project_com_to_base(com_world, base_T)
    offset = torch.abs(com_in_base) - half_extents
    outside = torch.clamp(offset, min=0.0)
    inside = torch.clamp(offset + COM_INSIDE_MARGIN, min=0.0)
    # Inside barrier uses MAX over axes (distance to the NEAREST edge), not sum:
    # near a corner the nearest-edge margin — not the sum of both — is what
    # bounds tip-over. Summing double-counts corners and wrongly rejects
    # COM-feasible configs (see test_com_polygon_penalty_corner_*).
    return (outside ** 2).sum(dim=-1) + COM_INSIDE_WEIGHT * (inside ** 2).max(dim=-1).values


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
        cost = com_polygon_penalty(com_world, base_T, cfg.half_extents)
        return cost.reshape(b, h, 1) * self._weight


ComOverBasePolygonCostCfg.class_type = ComOverBasePolygonCost


class ComOverAchievedStanceCost(ComOverBasePolygonCost):
    """cuRobo trajopt cost with device buffers that can follow each validated landing."""

    def __init__(self, cfg, initial_soles, sole_box):
        super().__init__(cfg)
        self.sole_box = sole_box
        self.normals, self.bounds = support_region_planes(initial_soles, sole_box, keyframe=True)

    def set_stance(self, soles):
        normals, bounds = support_region_planes(soles.to(self.normals), self.sole_box, keyframe=True)
        # An already captured cuRobo CUDA graph reads these same tensor addresses.
        self.normals.copy_(normals)
        self.bounds.copy_(bounds)

    def forward(self, state):
        xy = state.cuda_robot_model_state.robot_com[..., :2]
        outside = (xy @ self.normals.T - self.bounds).amax(-1)
        cost = outside.clamp_min(0).square() + COM_INSIDE_WEIGHT * (outside + COM_INSIDE_MARGIN).clamp_min(0).square()
        return cost.unsqueeze(-1) * self._weight


def _fk_state(world, q_batch: torch.Tensor, kin=None):
    """FK ``q_batch`` (``[B, full_dof]`` or ``[full_dof]``) through ``kin`` (default ``world.kinematics``);
    return its kinematics state. Differentiable through cuRobo's FK chain."""
    from curobo.types import JointState
    kin = world.kinematics if kin is None else kin
    q_batch = q_batch.to(kin.device_cfg.device)
    if q_batch.ndim == 1:
        q_batch = q_batch.unsqueeze(0)
    js = JointState.from_position(q_batch, joint_names=list(kin.joint_names))
    # No kin.get_active_js(js): js is already in kin's order, and the reorder index is a blocking H2D copy.
    return kin.compute_kinematics(js)


def _com_and_base_from_state(ks, link: str):
    """``(com_world [B,3], base_T [B,4,4])`` from a CoM-enabled kinematics
    state, ``base_T`` the world pose of the support frame ``link``: the inputs
    every COM-polygon helper needs."""
    com_world = ks.robot_com[..., :3].reshape(-1, 3)
    base_T = (
        ks.tool_poses.get_link_pose(link, make_contiguous=True)
        .get_matrix()
        .reshape(-1, 4, 4)
    )
    return com_world, base_T


def _fk_com_and_base(world, q_batch: torch.Tensor, kin=None):
    """``(com_world [B,3], base_T [B,4,4])`` for ``q_batch``, FK'd by :func:`_fk_state`, in the frame of
    ``world.model.support``."""
    return _com_and_base_from_state(_fk_state(world, q_batch, kin), world.model.support.link)


def com_polygon_mask_from_state(ks, support) -> torch.Tensor:
    """``[B]`` bool :func:`compute_com_polygon_mask` against ``support`` (a model's
    :class:`~cutamp.robots.t1.Support`) on a state a CoM-enabled FK already computed."""
    com, base_T = _com_and_base_from_state(ks, support.link)
    # Gate through the SAME penalty the hard ComPolygon constraint uses, so the
    # GN IK acceptance check and ConstraintChecker can never diverge. With the
    # max-over-axes barrier, (penalty <= COM_TOL) == "COM inside-or-on the hull".
    half = _half_extents(support.keyframe_half, com.device, com.dtype)
    pen = com_polygon_penalty(com, base_T, half)
    return pen <= COM_TOL


def compute_com_polygon_mask(world, q_batch: torch.Tensor) -> torch.Tensor:
    """Batched COM-in-polygon check.

    Computes per-link mass-weighted world COM via ``world.kinematics``
    (which is built with ``compute_com=True``), projects it into the
    support frame of ``world.model.support``, and returns a boolean mask of
    which batch elements have their COM inside its ``keyframe_half``
    rectangle.

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
    return com_polygon_mask_from_state(_fk_state(world, q_batch), world.model.support)


def com_margin_from_base(com_in_base: torch.Tensor, half_extents: Tuple[float, float]) -> torch.Tensor:
    """Static stability margin (McGhee & Frank): min over the two axes of
    ``h - |c|`` for CoM points ``[N, >=2]`` about a support's centre, in its
    frame, against its ``half_extents``. Positive inside the support
    rectangle, zero on an edge, negative outside. Metres."""
    half = _half_extents(tuple(map(float, half_extents)), com_in_base.device, com_in_base.dtype)
    return (half - com_in_base[..., :2].abs()).min(dim=-1).values


def physical_support_planes(soles_xyyaw: torch.Tensor, sole_box: Tuple[float, float]):
    """Return outward unit normals and offsets of the two-sole hull, for n·p <= b.

    All directed corner pairs occupy fixed slots. Nonfacets have infinite
    offsets so the same batched reduction works for any relative sole pose.
    """
    centre, yaw = soles_xyyaw[..., :2], soles_xyyaw[..., 2]
    cosine, sine = yaw.cos(), yaw.sin()
    forward = torch.stack((cosine, sine), -1) * (sole_box[0] / 2)
    lateral = torch.stack((-sine, cosine), -1) * (sole_box[1] / 2)
    corners = centre.unsqueeze(-2) + torch.stack(
        (forward + lateral, -forward + lateral, -forward - lateral, forward - lateral), dim=-2
    )
    corners = corners.flatten(-3, -2)

    # For each directed corner pair (i, j), its left normal is an inward
    # facet normal exactly when all eight corners lie on or left of its line.
    # Testing fixed-size tensors keeps the CUDA path batched and capture-safe.
    offsets = corners.unsqueeze(-3) - corners.unsqueeze(-2)
    length = offsets.norm(dim=-1)
    tolerance = 8 * torch.finfo(length.dtype).eps * length.amax(dim=(-2, -1), keepdim=True)
    safe_length = torch.where(length > tolerance, length, 1.0)
    inward = torch.stack((-offsets[..., 1], offsets[..., 0]), dim=-1) / safe_length.unsqueeze(-1)
    corner_clearance = (inward.unsqueeze(-2) * offsets.unsqueeze(-3)).sum(dim=-1)
    is_edge = (length > tolerance) & (corner_clearance.amin(dim=-1) >= -tolerance)
    normals = -inward
    bounds = (normals * corners.unsqueeze(-2)).sum(-1).masked_fill(~is_edge, torch.inf)
    bounds = bounds.masked_fill(~is_edge.any(dim=(-2, -1), keepdim=True), torch.nan)
    return normals.flatten(-3, -2), bounds.flatten(-2, -1)


def margin_from_support_planes(com_xy: torch.Tensor, normals: torch.Tensor, bounds: torch.Tensor):
    """Signed clearance in metres. A single fixed stance broadcasts over many CoM positions."""
    return (bounds - (normals * com_xy.unsqueeze(-2)).sum(-1)).amin(dim=-1)


def physical_support_margin(com_xy: torch.Tensor, soles_xyyaw: torch.Tensor,
                            sole_box: Tuple[float, float]) -> torch.Tensor:
    """Minimum signed clearance from the physical two-sole hull, in metres.

    `com_xy` and `soles_xyyaw` share world coordinates and batch dimensions,
    with trailing shapes [2] and [2, 3]. `sole_box` is the full sole length
    and width. The active edge and the CoM both carry derivatives. Facet
    choices are discrete at changes of hull topology or closest-edge ties.
    """
    if com_xy.shape[-1:] != (2,) or soles_xyyaw.shape != (*com_xy.shape[:-1], 2, 3):
        raise ValueError("Expected matching batches of CoM [..., 2] and sole poses [..., 2, 3]")
    return margin_from_support_planes(com_xy, *physical_support_planes(soles_xyyaw, sole_box))


def compute_com_margin(world, q_batch: torch.Tensor) -> torch.Tensor:
    """``[B]`` stability margins for full-cspace configs via FK, against the keyframes' rectangle
    (``world.model.support.keyframe_half``).

    Measure-only companion to :func:`compute_com_polygon_mask`, through :func:`particle_com_margin`."""
    support = world.model.support
    com, base_T = _fk_com_and_base(world, q_batch)
    return com_margin_from_base(_project_com_to_base(com, base_T), support.keyframe_half)


def particle_com_margin(world, particle: "dict[str, torch.Tensor]") -> float:
    """One particle's CoM margin: :func:`compute_com_margin`'s min over its confs (``[full_dof]`` entries named
    ``left_q``/``right_q``/``q``). Measure only: the best particle's ``best_com_margin`` (optimize_plan.py) and
    the planned one's ``planned_com_margin`` (algorithm.py)."""
    return min(
        compute_com_margin(world, v.unsqueeze(0)).item()
        for k, v in particle.items() if k.startswith(("left_q", "right_q", "q")) and v.ndim == 1
    )


def com_zmp_in_support(world, plan_js):
    """``(com_xy [T, 2], zmp_xy [T, 2])`` of a plan's frames (position, velocity and acceleration, by joint name) in
    the support frame, about its origin, the rectangle's centre and on the sole plane (test_real_legs_support.py).

    The ZMP is the multibody one, exact: the floor's wrench on the robot is RNEA's first-pass base wrench
    (``world.segment_torques.floor``), so with its moment about the origin, the ZMP is (-M_y, M_x) / F_z. That counts
    every link's acceleration and spin; a trunk swinging above the hips moves it centimetres past the CoM alone's
    cart-table ZMP.

    The floor can only push. A frame whose F_z is not positive would need it to pull: the soles would leave the floor
    and there is no ZMP, so that frame's is +inf, outside every support region."""
    force, moment, com = world.segment_torques.floor(plan_js)
    zmp = torch.stack([-moment[:, 1], moment[:, 0]], dim=-1) / force[:, 2:]
    return com[:, :2], torch.where(force[:, 2:] > 0, zmp, math.inf)


def compute_com_polygon_penalties(
    world,
    particles: "dict[str, torch.Tensor]",
    achieved_stances=None,
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
    :meth:`CostFunction.com_polygon_constraint`), once per evaluation.

    Args:
        world: TAMPWorld instance; the FK runs on its ``site_kinematics("com")``.
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
    support = world.model.support
    com_world, base_T = _fk_com_and_base(world, q_all, world.site_kinematics("com"))
    if achieved_stances is not None:
        soles = torch.cat([achieved_stances[name] for name in conf_names])
        n, b = support_region_planes(soles, world.model.sole_box, keyframe=True)
        g = -margin_from_support_planes(com_world[..., :2], n, b)
        pen = g.clamp_min(0).square() + COM_INSIDE_WEIGHT * (g + COM_INSIDE_MARGIN).clamp_min(0).square()
        return dict(zip(conf_names, pen.split([q.shape[0] for q in qs])))
    half = _half_extents(support.keyframe_half, com_world.device, com_world.dtype)
    pen = com_polygon_penalty(com_world, base_T, half)
    return dict(zip(conf_names, pen.split([q.shape[0] for q in qs])))


@lru_cache(maxsize=None)
def _corner_signs(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``[4, 2]``: the signs of a rectangle's four corners along its axes, made once per device and dtype (a tensor
    from a Python list is a host-to-device copy, which fails a CUDA graph capture). Shared; never mutate or evict it:
    a captured graph keeps reading its address."""
    return torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], device=device, dtype=dtype)


def _sole_corners(soles: torch.Tensor, sole_box: Tuple[float, float]) -> torch.Tensor:
    """``[N, 2, 4, 2]``: the corners of each ``sole_box`` rectangle, centred on its sole frame."""
    c, s = torch.cos(soles[..., 2]), torch.sin(soles[..., 2])
    ex, ey = torch.stack([c, s], -1), torch.stack([-s, c], -1)
    sg = _corner_signs(soles.device, soles.dtype)
    return (soles[..., None, :2] + sg[:, 0, None] * (sole_box[0] / 2) * ex[..., None, :]
            + sg[:, 1, None] * (sole_box[1] / 2) * ey[..., None, :])


def physical_support_centroid(soles_xyyaw: torch.Tensor, sole_box: Tuple[float, float]) -> torch.Tensor:
    """Area centroid of the physical convex sole hull, in world XY.

    Keep consecutive counterclockwise hull edges. Removing duplicate corners
    and subdividing collinear faces avoids double-counting aligned soles. The
    signed-area formula differentiates through both moving vertices of every
    selected edge. Center the coordinates for numerical stability.
    """
    vertices = _sole_corners(soles_xyyaw, sole_box).flatten(-3, -2)
    origin = vertices.mean(-2)
    v = vertices - origin.unsqueeze(-2)
    a, b = v.unsqueeze(-2), v.unsqueeze(-3)
    delta = b - a
    length = delta.norm(dim=-1)
    tol = 8 * torch.finfo(v.dtype).eps * length.amax(dim=(-1, -2), keepdim=True)
    distinct = ~(length <= tol).tril(-1).any(-1)
    unit = delta / torch.where(length > tol, length, 1.0).unsqueeze(-1)
    inward = torch.stack((-unit[..., 1], unit[..., 0]), dim=-1)
    # [i,j,k] tests corner k against directed segment i→j.
    side = (inward.unsqueeze(-2) * delta.unsqueeze(-3)).sum(-1)
    along = (unit.unsqueeze(-2) * delta.unsqueeze(-3)).sum(-1)
    eps = tol.unsqueeze(-1)
    between = (side.abs() <= eps) & (along > eps) & (along < length.unsqueeze(-1) - eps)
    edges = (length > tol) & distinct.unsqueeze(-1) & distinct.unsqueeze(-2)
    edges &= (side.amin(dim=-1) >= -tol) & ~between.any(-1)
    cross = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).masked_fill(~edges, 0.0)
    numerator = ((a + b) * cross.unsqueeze(-1)).sum(dim=(-2, -3))
    return origin + numerator / (3 * cross.sum(dim=(-1, -2), keepdim=False).unsqueeze(-1))


def _perp_unit(d: torch.Tensor):
    """Both unit normals of ``d`` ``[..., 2]`` (stacked on a new axis -2) and whether ``d`` is not degenerate."""
    length = d.norm(dim=-1, keepdim=True)
    u = d / length.clamp_min(1e-9)
    n = torch.stack([torch.stack([u[..., 1], -u[..., 0]], -1), torch.stack([-u[..., 1], u[..., 0]], -1)], -2)
    return n, (length > 1e-9).squeeze(-1)


def support_region_planes(soles: torch.Tensor, sole_box: Tuple[float, float], keyframe: bool):
    """``(n [..., 552, 2], b [..., 552])`` whose intersection is the support region of ``soles`` ``[..., 2, 3]``.
    The 552 candidate normals: the soles' 8 edges, the 32 segments from a corner of one sole to a corner of the other
    (the hull's edges) and the 512 segments from a vertex of a L + (1 - a) R to one of (1 - a) L + a R (the load
    guard's edges). For each, b is the smaller of the two sets' support values, the hull's moved in by
    ``COM_SAFETY_MARGIN``: an intersection of convex sets is the intersection of their supporting half-planes, and
    moving a set in by a distance lowers each support value by it. The keyframe region moves in by
    ``COM_DYNAMIC_RESERVE`` more. A degenerate candidate gets b = +inf."""
    lead = soles.shape[:-2]
    soles = soles.reshape(-1, 2, 3)
    N, a = soles.shape[0], LOAD_GUARD_A
    V = _sole_corners(soles, sole_box)                                   # [N, 2, 4, 2]
    VL, VR = V[:, 0], V[:, 1]
    c, s = torch.cos(soles[..., 2]), torch.sin(soles[..., 2])
    ex, ey = torch.stack([c, s], -1), torch.stack([-s, c], -1)          # [N, 2, 2]
    n_edge = torch.cat([ex, -ex, ey, -ey], 1)                           # [N, 8, 2]
    ok_edge = torch.ones(N, 8, dtype=torch.bool, device=soles.device)
    n_h, ok_h = _perp_unit((VR[:, None, :, :] - VL[:, :, None, :]).reshape(N, 16, 2))
    n_h, ok_h = n_h.reshape(N, 32, 2), ok_h[:, :, None].expand(N, 16, 2).reshape(N, 32)
    A_ = (a * VL[:, :, None, :] + (1 - a) * VR[:, None, :, :]).reshape(N, 16, 2)
    B_ = ((1 - a) * VL[:, :, None, :] + a * VR[:, None, :, :]).reshape(N, 16, 2)
    n_s, ok_s = _perp_unit((B_[:, None, :, :] - A_[:, :, None, :]).reshape(N, 256, 2))
    n_s, ok_s = n_s.reshape(N, 512, 2), ok_s[:, :, None].expand(N, 256, 2).reshape(N, 512)
    n = torch.cat([n_edge, n_h, n_s], 1)                                 # [N, 552, 2]
    ok = torch.cat([ok_edge, ok_h, ok_s], 1)
    # The two sole rectangles' support values along every normal.
    hx, hy = sole_box[0] / 2, sole_box[1] / 2
    h = ((n[:, None] * soles[:, :, None, :2]).sum(-1) + hx * (n[:, None] * ex[:, :, None]).sum(-1).abs()
         + hy * (n[:, None] * ey[:, :, None]).sum(-1).abs())             # [N, 2, K]
    hL, hR = h[:, 0], h[:, 1]
    h_guard = torch.maximum(a * hL + (1 - a) * hR, (1 - a) * hL + a * hR)
    b = torch.minimum(h_guard, torch.maximum(hL, hR) - COM_SAFETY_MARGIN) - (COM_DYNAMIC_RESERVE if keyframe else 0.0)
    b = torch.where(ok, b, torch.full_like(b, float("inf")))
    return n.reshape(*lead, -1, 2), b.reshape(*lead, -1)
