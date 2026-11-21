"""The gap between two sole rectangles on the floor (design 5.11, version 1; PROTO collision/sole_collision.py).

The step bound reads :func:`rect_rect_sat` for the swung sole against the planted one. Differentiable in the poses,
with no host sync, so it can be captured in a CUDA graph.
"""

from functools import cache

import torch


@cache
def _const(values, dtype, device):
    """A constant made once per (values, dtype, device): building a tensor from Python numbers is a host-to-device
    copy, which a CUDA graph cannot record (the planner's cost_function._device_const does the same)."""
    return torch.tensor(values, dtype=dtype, device=device)


def _half(half, like):
    return half if torch.is_tensor(half) else _const(tuple(float(v) for v in half), like.dtype, like.device)


def _axes(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((torch.stack((c, s), -1), torch.stack((-s, c), -1)), -2)  # (..., 2 axes, 2)


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


def sole_half(model=None):
    """Half extents of the sole's rectangle: a robot fact, ``sole_box / 2`` of the model."""
    if model is None:
        from cutamp.robots.t1 import REAL_LEGS as model
    return tuple(float(v) / 2 for v in model.sole_box)
