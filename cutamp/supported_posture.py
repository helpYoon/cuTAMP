"""The virtual-base chart of an achieved T1 stance and the self-collision spheres of its postures."""

from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
import yaml

from cutamp.robots.t1 import REAL_LEGS


@lru_cache(maxsize=4)
def _read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def _kinematics_cfg(cfg):
    return cfg["robot_cfg"]["kinematics"] if "robot_cfg" in cfg else cfg


def set_stance_base_limits(cfg, soles):
    """Transport the builder's per-leg XY reach boxes and intersect their world-axis bounds."""
    cfg = _kinematics_cfg(cfg)
    geometry = _read_yaml(REAL_LEGS.asset_root / "sole_base_reach.json")
    soles = np.asarray(soles)
    if soles.shape != (2, 3) or not np.isfinite(soles).all():
        raise ValueError("Expected two finite sole poses (x, y, yaw)")
    corners = []
    for foot, local in zip(soles, geometry["sole_base_xy"]):
        c, s = np.cos(foot[2]), np.sin(foot[2])
        corners.append(np.array(list(product(*local))) @ [[c, s], [-s, c]] + foot[:2])
    corners = np.stack(corners)
    padding = geometry["base_margin"]
    lower = np.floor((corners.min(axis=1).max(axis=0) - padding) * 1000) / 1000
    upper = np.ceil((corners.max(axis=1).min(axis=0) + padding) * 1000) / 1000
    if (lower >= upper).any():
        raise ValueError("The two leg workspaces have no shared virtual-base XY range")
    limits = np.stack([lower, upper], axis=1)
    for axis, interval in zip("xy", limits):
        cfg["extra_links"]["base_link_" + axis]["joint_limits"] = interval.tolist()
    return limits


class PostureCollisionSpheres:
    """Padded active self-sphere pairs, using the same link transforms as CoM and wrist FK."""

    def __init__(self, cfg, kin):
        raw = cfg["collision_spheres"]
        spheres = _read_yaml(raw)["collision_spheres"] if isinstance(raw, (str, Path)) else raw
        names, offsets, radii = [], [], []
        for link in cfg["collision_link_names"]:
            if link not in kin.links:
                continue
            for sphere in spheres.get(link, ()):
                radius = sphere["radius"]
                if radius > 0:
                    names.append(link)
                    offsets.append(sphere["center"])
                    radii.append(radius + cfg["collision_sphere_buffer"].get(link, 0) + cfg["self_collision_buffer"].get(link, 0))
        ignored = {frozenset((a, b)) for a, others in cfg["self_collision_ignore"].items() for b in others}
        pairs = [(i, j) for i, j in combinations(range(len(names)), 2)
                 if names[i] != names[j] and frozenset((names[i], names[j])) not in ignored]
        device = kin.lower.device
        self.i, self.j = torch.tensor(pairs, dtype=torch.long, device=device).reshape(-1, 2).T
        self.links = torch.tensor([kin.links.index(link) for link in names], dtype=torch.long, device=device)
        self.offsets = torch.tensor(offsets, dtype=torch.double, device=device)
        radii = torch.tensor(radii, dtype=torch.double, device=device)
        self.radii_sum = radii[self.i] + radii[self.j]
        self.descendants = kin.descendants[6:, self.links]
        self.joint_links = kin.joint_links

    def residual(self, frames, repeats=1):
        """Pack every active 6 mm sphere-pair shortfall and its joint Jacobian per posture."""
        pos, rot, _, axes = frames
        points = pos[:, self.links] + (rot[:, self.links] @ self.offsets.to(rot.dtype)[..., None]).squeeze(-1)
        # Complex-step Hessian columns share the same active sphere pairs as their real posture.
        real = points.real[::repeats]
        delta = real[:, self.i] - real[:, self.j]
        limit = self.radii_sum + 0.006
        if axes is None and not pos.is_complex():
            # Trial scoring only uses the squared norm. Dense values avoid packing and a device sync.
            return (limit - torch.linalg.vector_norm(delta, dim=-1)).clamp_min(0), None
        active = delta.square().sum(-1) < limit.square()
        owner, pair = active.nonzero(as_tuple=True)
        if not len(pair):
            jac = None if axes is None else pos.new_empty(len(pos), 0, axes.shape[1] + 6)
            return pos.new_empty(len(pos), 0), jac
        counts = active.sum(1)
        width = int(counts.max())
        slot = torch.arange(len(pair), device=pos.device) - (counts.cumsum(0) - counts)[owner]
        if repeats > 1:
            owner = (owner[:, None] * repeats + torch.arange(repeats, device=pos.device)).flatten()
            pair = pair.repeat_interleave(repeats)
            slot = slot.repeat_interleave(repeats)
        i, j = self.i[pair], self.j[pair]
        delta = points[owner, i] - points[owner, j]
        squared = delta.square().sum(-1)
        distance = torch.where(squared.real > 1e-24, squared, 1e-24).sqrt()
        deficit = limit[pair] - distance
        value = pos.new_zeros(len(pos), width)
        value[owner, slot] = deficit
        if axes is None:
            return value, None
        # Both spheres rotate together about any shared ancestor, so its
        # contribution to their separation cancels (as do all six base DOFs).
        lever = points[owner, i, None] - pos[owner[:, None], self.joint_links]
        unit = (delta / distance[:, None])[:, None].expand_as(lever)
        joint = (torch.linalg.cross(lever, unit) * axes[owner]).sum(-1)
        mask = (self.descendants[:, j] - self.descendants[:, i]).T
        jac = pos.new_zeros(len(pos), width, axes.shape[1] + 6)
        jac[owner, slot, 6:] = joint * mask
        return value, jac
