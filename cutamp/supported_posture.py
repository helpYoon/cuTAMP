"""Fit supported T1 manipulation keyframes in the achieved stance."""

from copy import deepcopy
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import least_squares

from cutamp.com_polygon_cost import physical_support_centroid, physical_support_planes, support_region_planes
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

    def gaps(self, frames):
        pos, rot = frames[:2]
        points = pos[:, self.links] + (rot[:, self.links] @ self.offsets.to(rot.dtype)[..., None]).squeeze(-1)
        d = (points[0, self.i] - points[0, self.j]).real
        norm = torch.linalg.vector_norm(d, dim=-1)
        return norm - self.radii_sum, d / norm[:, None].clamp_min(1e-12), points

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


def fit_centered_posture(soles, wrist, seed, arm, *, cfg, head=None, other_arm_reference=None, centering_weight=2.0):
    """Return 33 supported joint coordinates for level soles and a full wrist pose.

    The idle arm is fixed in joint space. An occupied arm can pass its seven reference angles.
    Applies the stance-specific virtual base chart on a private config. The connected motion
    planner separately checks objects, obstacles and all interpolated states.
    """
    from cutamp.step_ik import _Kinematics, _targets

    if arm not in ("left", "right") or not np.isfinite(centering_weight) or centering_weight < 0:
        raise ValueError("Expected a left/right acting arm and a finite nonnegative centering weight")
    cfg = deepcopy(_kinematics_cfg(cfg))
    as_cpu = lambda v: torch.as_tensor(v, dtype=torch.double, device="cpu").detach()
    soles, wrist = as_cpu(soles).reshape(1, 2, 3), as_cpu(wrist).reshape(1, 4, 4)
    set_stance_base_limits(cfg, soles[0].numpy())
    if head is not None:
        cfg["lock_joints"].update(head)
    kin = _Kinematics(torch.device("cpu"), cfg)
    spheres = kin.self_spheres
    seed = as_cpu(seed).reshape(kin.n).clamp(kin.lower, kin.upper)
    idle = REAL_LEGS.right_arm if arm == "left" else REAL_LEGS.left_arm
    fixed = as_cpu(REAL_LEGS.home[idle] if other_arm_reference is None else other_arm_reference)
    seed[idle] = fixed
    if (fixed < kin.lower[idle]).any() or (fixed > kin.upper[idle]).any():
        raise ValueError("The other arm's fixed posture exceeds its supported joint limits")
    free = torch.ones(kin.n, dtype=torch.bool)
    free[idle] = False
    prior = as_cpu([0.11] * 3 + [0.045] * (kin.n - 3))[free]
    target = _targets(soles, wrist).flatten()
    centroid = physical_support_centroid(soles, kin.box)[0]
    n, b = physical_support_planes(soles, kin.box)
    ns, bs = support_region_planes(soles, kin.box, True)
    normals, bounds = torch.cat((n[0], ns[0])), torch.cat((b[0] - 0.055, bs[0] - 0.003))
    finite = bounds.isfinite()
    normals, bounds = normals[finite], bounds[finite]
    # Complex-step gives the constrained fit a pose, CoM and sphere Jacobian in one FK batch.
    perturb = 1j * 1e-25 * torch.eye(kin.n, dtype=torch.double)[free]
    cached = {}

    def evaluate(x):
        if "x" in cached and np.array_equal(cached["x"], x):
            return cached["residual"], cached["jacobian"]
        q = seed.clone()
        q[free] = as_cpu(x)
        batch = torch.cat((q[None], q[None] + perturb))
        frames = kin.link_frames(batch)
        pose, com = kin.fk(batch, arm, False, frames=frames)
        xy, dxy = com[0, :2].real, com[1:, :2].imag.T * 1e25
        gaps, direction, points = spheres.gaps(frames)
        near = gaps < 0.006
        deficit = 0.006 - gaps[near]
        collision = deficit.norm()
        dnear = (points[1:, spheres.i[near]] - points[1:, spheres.j[near]]).imag * 1e25
        dgap = (dnear * direction[near]).sum(-1)
        dcollision = -(dgap * deficit).sum(-1) / collision.clamp_min(1e-12)
        violation = normals @ xy - bounds
        i = int(violation.argmax())
        residual = torch.cat([50 * (pose[0].real - target), centering_weight * (xy - centroid),
                              prior * (q[free] - seed[free]), 24 * collision[None], 20 * violation[i:i+1].clamp_min(0)])
        jacobian = torch.cat([50 * pose[1:].imag.T * 1e25, centering_weight * dxy, prior.diag(),
                              24 * dcollision[None], 20 * (violation[i] > 0) * (normals[i:i+1] @ dxy)])
        cached.update(x=x.copy(), residual=residual.numpy(), jacobian=jacobian.numpy())
        return cached["residual"], cached["jacobian"]

    result = least_squares(lambda x: evaluate(x)[0], seed[free].numpy(), jac=lambda x: evaluate(x)[1],
                           bounds=(kin.lower[free].numpy(), kin.upper[free].numpy()), max_nfev=30,
                           ftol=1e-5, xtol=1e-5, gtol=1e-5)
    answer = seed.clone()
    answer[free] = as_cpu(result.x)
    frames = kin.link_frames(answer[None])
    pose, com = kin.fk(answer[None], arm, False, frames=frames)
    error = (pose.flatten() - target).reshape(3, 12)
    position = error[:, :3].norm(dim=1)
    angle = 2 * (error[:, 3:].norm(dim=1) / (0.2 * np.sqrt(8))).clamp_max(1).asin()
    clearance = spheres.gaps(frames)[0]
    if (not answer.isfinite().all() or (position[:2] > 0.001).any() or position[2] > 0.005
            or (angle[:2] > 0.01).any() or angle[2] > 0.05 or (clearance < 0).any()
            or (n[0] @ com[0, :2] > b[0] - 0.05).any() or (ns[0] @ com[0, :2] > bs[0]).any()):
        raise ValueError("Supported posture failed a wrist, sole, support or self-collision check")
    return answer.numpy()
