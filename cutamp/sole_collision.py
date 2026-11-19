"""The sole's footprint against statics, objects and the other sole (design 5.11, version 1; PROTO collision/
sole_collision.py, rectangle and disc tests only).

The landed sole is its rectangle on the floor; the floor never enters, since the test has no vertical axis. Obstacles
are the oriented footprints of the statics that meet the foot's height band, the movables' discs and the other sole's
rectangle. :func:`footprint_values` is the largest penetration of the sole into any of them, plus the margin: at most
zero is clear. Differentiable in the sole's pose, with no host sync, so it can be captured in a CUDA graph.
"""

import math
from functools import cache
from typing import NamedTuple

import torch
from curobo.scene import Cuboid

from cutamp.stance import STEP_LINE_POSES


@cache
def _const(values, dtype, device):
    """A constant made once per (values, dtype, device): building a tensor from Python numbers is a host-to-device
    copy, which a CUDA graph cannot record (the planner's cost_function._device_const does the same)."""
    return torch.tensor(values, dtype=dtype, device=device)


def _half(half, like):
    return half if torch.is_tensor(half) else _const(tuple(float(v) for v in half), like.dtype, like.device)


_SIGNS = ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0))


def _axes(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((torch.stack((c, s), -1), torch.stack((-s, c), -1)), -2)  # (..., 2 axes, 2)


def point_rect_sdf(pose, half, pts):
    """Signed distance of points pts (..., M, 2) to the rectangle at pose (..., 3) = (x, y, yaw) with half extents
    half (2,) or (..., 2): negative inside (minus the distance to the nearest edge).  -> (..., M)."""
    ax = _axes(pose[..., 2])  # (..., 2, 2)
    local = torch.einsum("...ij,...mj->...mi", ax, pts - pose[..., None, :2])  # (..., M, 2)
    h = _half(half, pose)
    q = local.abs() - (h[..., None, :] if h.dim() > 1 else h)
    outside = torch.sqrt(torch.clamp(q, min=0.0).square().sum(-1) + 1e-18)
    inside = torch.clamp(q.max(-1).values, max=0.0)
    return outside + inside


def rect_corners(pose, half):
    h = _half(half, pose)
    sg = _const(_SIGNS, pose.dtype, pose.device)
    loc = sg * (h[..., None, :] if h.dim() > 1 else h)  # (..., 4, 2)
    ax = _axes(pose[..., 2])
    return pose[..., None, :2] + torch.einsum("...ji,...mj->...mi", ax, loc.expand(*pose.shape[:-1], 4, 2))


def rect_rect_sat(pose_a, half_a, pose_b, half_b):
    """Separating-axis gap of two oriented rectangles (broadcast over leading shapes).  Negative = minus the
    penetration depth (the shortest translation that separates them; exact).  Positive = the largest gap along the four
    face normals: a lower bound of the distance, exact when the closest features are a corner and a side."""
    ha, hb = _half(half_a, pose_a), _half(half_b, pose_a)
    aa, ab = _axes(pose_a[..., 2]), _axes(pose_b[..., 2])
    shape = torch.broadcast_shapes(aa.shape, ab.shape)
    aa, ab = aa.expand(shape), ab.expand(shape)
    axes = torch.cat((aa, ab), -2)  # (..., 4, 2)
    d = (pose_b[..., :2] - pose_a[..., :2]).expand(*shape[:-2], 2)
    dist = torch.einsum("...kj,...j->...k", axes, d).abs()
    ra = (torch.einsum("...kj,...ij->...ki", axes, aa).abs() * (ha[..., None, :] if ha.dim() > 1 else ha)).sum(-1)
    rb = (torch.einsum("...kj,...ij->...ki", axes, ab).abs() * (hb[..., None, :] if hb.dim() > 1 else hb)).sum(-1)
    return (dist - ra - rb).max(-1).values


def rect_rect_signed_distance(pose_a, half_a, pose_b, half_b):
    """Exact signed distance of two oriented rectangles: the separating-axis depth where they overlap, else the
    smallest corner-to-rectangle distance (for two disjoint convex polygons one closest point is always a corner)."""
    sat = rect_rect_sat(pose_a, half_a, pose_b, half_b)
    shape = torch.broadcast_shapes(pose_a.shape, pose_b.shape)
    pa, pb = pose_a.expand(shape), pose_b.expand(shape)
    ha, hb = _half(half_a, pa), _half(half_b, pa)
    ha = ha.expand(*shape[:-1], 2) if ha.dim() > 1 else ha
    hb = hb.expand(*shape[:-1], 2) if hb.dim() > 1 else hb
    d1 = point_rect_sdf(pb, hb, rect_corners(pa, ha)).min(-1).values
    d2 = point_rect_sdf(pa, ha, rect_corners(pb, hb)).min(-1).values
    return torch.where(sat < 0, sat, torch.minimum(d1, d2))


def spheres_to_discs(spheres, band_top):
    """World spheres (..., M, 4) -> discs (..., M, 3) = (x, y, r) for the 2-D test: the sphere's cross-section at the
    height where it is widest inside the band [0, band_top]; radius 0 for a sphere wholly above the band."""
    z, r = spheres[..., 2], spheres[..., 3]
    dz = torch.clamp(z - band_top, min=0.0)  # how far the centre is above the band
    rr = torch.sqrt(torch.clamp(r * r - dz * dz, min=0.0))
    return torch.stack((spheres[..., 0], spheres[..., 1], rr), -1)


class Footprints(NamedTuple):
    """The world's footprint table: ``rects [R, 5]`` (x, y, yaw, half x, half y) of the statics whose height interval
    meets the foot's band, and the names of the movables (their discs are made per particle from their poses)."""

    rects: torch.Tensor
    disc_names: tuple[str, ...]


def sole_half(model=None):
    """Half extents of the sole's rectangle: a robot fact, ``sole_box / 2`` of the model."""
    if model is None:
        from cutamp.robots.t1 import REAL_LEGS as model
    return tuple(float(v) / 2 for v in model.sole_box)


def footprint_values(sole, rects, other, discs, margin, half=None):
    """The largest penetration of the sole rectangle into any rectangle in ``rects``, disc in ``discs`` or the other
    sole, plus ``margin`` (m): zero or less is clear.

    sole [N, 3] (x, y, yaw), rects [R, 5], other [N, 3], discs [N, M, 3] (x, y, r; r <= 0 is ignored), as the
    leading dimensions broadcast (``sole [N, P, 3]`` against ``other [N, 1, 3]`` and ``discs [N, 1, M, 3]`` gives
    ``[N, P]``). Exact signed distances; differentiable in ``sole``; no host sync.
    """
    half = sole_half() if half is None else half
    terms = [margin - rect_rect_signed_distance(sole, half, other, half)[..., None]]
    if rects.shape[0]:
        terms.append(margin - rect_rect_signed_distance(sole[..., None, :], half, rects[:, :3], rects[:, 3:]))
    if discs.shape[-2]:
        d = point_rect_sdf(sole, half, discs[..., :2]) - discs[..., 2]
        terms.append(torch.where(discs[..., 2] > 0, margin - d, torch.full_like(d, -1e9)))
    return _cat(terms).max(-1).values


def _cat(terms):
    shape = torch.broadcast_shapes(*[t.shape[:-1] for t in terms])
    return torch.cat([t.expand(*shape, t.shape[-1]) for t in terms], -1)


def line_poses(start, landing, n=STEP_LINE_POSES):
    """``n`` poses evenly spaced on the straight line from ``start`` [N, 3] to ``landing`` [N, 3], the start excluded
    and the landing included; yaw interpolated the shorter way. -> [N, n, 3]."""
    t = torch.linspace(1.0 / n, 1.0, n, dtype=start.dtype, device=start.device)
    dyaw = torch.remainder(landing[..., 2] - start[..., 2] + math.pi, 2 * math.pi) - math.pi
    xy = start[..., None, :2] + t[:, None] * (landing[..., None, :2] - start[..., None, :2])
    yaw = start[..., None, 2] + t * dyaw[..., None]
    return torch.cat((xy, yaw[..., None]), -1)


def build_footprints(statics, movables, foot_top: float) -> Footprints:
    """The table of a world: one rectangle per static whose height interval meets ``(0, foot_top]`` (a table top above
    the band is left out, its legs are kept when the scene models them; the floor's top is at 0). A cuboid yawed
    about the vertical is its oriented rectangle; any other shape, or a tilted cuboid, is the axis-aligned box of its
    posed mesh."""
    rows = []
    for o in statics:
        lo, hi = o.get_trimesh_mesh(transform_with_pose=True).bounds
        if not (hi[2] > 1e-6 and lo[2] < foot_top):
            continue
        w, qx, qy, qz = o.pose[3:7]
        if isinstance(o, Cuboid) and abs(qx) < 1e-9 and abs(qy) < 1e-9:
            rows.append([o.pose[0], o.pose[1], 2.0 * math.atan2(qz, w), o.dims[0] / 2, o.dims[1] / 2])
        else:
            rows.append([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0, (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2])
    return Footprints(torch.tensor(rows, dtype=torch.float32).reshape(-1, 5), tuple(o.name for o in movables))


def geometric_sweep_rows(world, object_poses, held):
    """Make the planar clearance constraint for a geometric-only refinement."""
    from cutamp.stance import abs_pose
    from cutamp.step_set import _facts
    from cutamp.utils.common import transform_spheres

    top = _facts(world.model)["foot_box"][2]
    rectangles = build_footprints(world.statics, (), top).rects.to(world.q_init)
    carried = {obj for obj, _ in held.values()}
    loose = [
        transform_spheres(world.get_collision_spheres(obj), pose)
        for obj, pose in object_poses.items()
        if obj not in carried
    ]
    moving_discs = spheres_to_discs(torch.cat(loose, dim=1), top) if loose else None

    def evaluate(feet, order, commands):
        rows = {}
        discs = moving_discs if moving_discs is not None else feet.new_empty(len(feet), 0, 3)
        for j, side in enumerate(order):
            i = "LR".index(side)
            planted = feet[:, 1 - i]
            landing = abs_pose(planted, commands[:, j])
            path = line_poses(feet[:, i], landing)
            penetration = footprint_values(path, rectangles, planted[:, None], discs[:, None], 0.0)
            rows[f"step{j + 1}_sweep"] = penetration.amax(-1) / 0.02
            feet = torch.stack((landing, planted) if i == 0 else (planted, landing), dim=1)
        return rows

    return evaluate
