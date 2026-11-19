"""Batched T1 sole and wrist IK with implicit differentiation of its bounded optimum."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from cutamp.com_polygon_cost import physical_support_margin
from cutamp.robots.t1 import REAL_LEGS, t1_curobo_cfg


@dataclass
class StanceIK:
    q: torch.Tensor
    margin: torch.Tensor
    hull_only_margin: torch.Tensor
    com: torch.Tensor
    wrist_pos_error: torch.Tensor
    wrist_rot_error: torch.Tensor
    sole_pos_error: torch.Tensor
    sole_rot_error: torch.Tensor
    optimality: torch.Tensor
    ok: torch.Tensor
    soles: torch.Tensor  # achieved landing supplied to the solver


def _xyz(s):
    return np.fromstring(s, sep=" ")


class _Kinematics:
    """Level-ordered URDF FK for both arms, sole centers, and the complete robot mass model.

    Double precision and explicit geometric Jacobians also allow complex-step evaluation
    of the *full* Hessian of the pose and posture residual objective on its active branch.
    """

    def __init__(self, device):
        model = REAL_LEGS
        cfg = t1_curobo_cfg(model)["robot_cfg"]["kinematics"]
        tree = ET.parse(cfg["urdf_path"]).getroot()
        names = model.joint_names
        self.n = len(names)
        self.box = model.sole_box

        def tensor(values):
            return torch.as_tensor(np.asarray(values), device=device, dtype=torch.float64)

        def indices(values):
            return torch.as_tensor(values, device=device, dtype=torch.long)

        # Base translation is world-aligned. Its three remaining coordinates are intrinsic Y-X-Z.
        lower = np.zeros(self.n)
        upper = np.zeros(self.n)
        for i, sub in enumerate(("x", "y", "z", "pitch", "roll", "yaw")):
            lower[i], upper[i] = cfg["extra_links"]["base_link_" + sub]["joint_limits"]
        self.base_axes = tensor(np.eye(3)[[1, 0, 2]])
        self.eye = torch.eye(3, device=device, dtype=torch.float64)
        children = {}
        for joint in tree.findall("joint"):
            if joint.get("name") == "world_joint":
                continue
            children.setdefault(joint.find("parent").get("link"), []).append(joint)
        self.levels = []
        links = ["Trunk"]
        parents = links.copy()
        ancestors = {"Trunk": [3, 4, 5]}
        joint_links = [0] * self.n
        while parents:
            level = []
            limb = []
            for i, parent in enumerate(parents):
                for joint in children.get(parent, ()):
                    name = joint.get("name")
                    child = joint.find("child").get("link")
                    origin = joint.find("origin")
                    axis = joint.find("axis")
                    a = _xyz(axis.get("xyz")) if axis is not None else np.zeros(3)
                    r = Rotation.from_euler("xyz", _xyz(origin.get("rpy"))).as_matrix()
                    if name in cfg["lock_joints"]:
                        r = r @ Rotation.from_rotvec(a * cfg["lock_joints"][name]).as_matrix()
                    j = names.index(name) if name in names else -1
                    if j >= 0:
                        joint_links[j] = len(links) + len(limb)
                        bounds = joint.find("limit")
                        lower[j], upper[j] = float(bounds.get("lower")), float(bounds.get("upper"))
                    ancestors[child] = ancestors[parent] + ([j] if j >= 0 else [])
                    limb.append(child)
                    # For fixed links the zero axis gives the identity at every angle.
                    level.append((i, max(j, 0), _xyz(origin.get("xyz")), r, a if j >= 0 else np.zeros(3)))
            if level:
                ix, jx, p, r, a = zip(*level)
                self.levels.append((indices(ix), indices(jx), tensor(p), tensor(r), tensor(a)))
            links.extend(limb)
            parents = limb
        clip = dict(zip(cfg["cspace"]["joint_names"], cfg["cspace"]["position_limit_clip"]))
        widths = np.array([clip[name] for name in names])
        self.lower, self.upper = tensor(lower + widths), tensor(upper - widths)
        self.prior = tensor([0.035] * 3 + [0.008] * (self.n - 3))
        self.joint_links = indices(joint_links[6:])
        self.end = {}
        for arm in ("left", "right"):
            tools = ("left_foot_link", "right_foot_link", f"{arm}_base_link")
            mask = np.zeros((3, self.n, 1))
            for i, name in enumerate(tools):
                mask[i, ancestors[name], 0] = 1.0
            self.end[arm] = indices([links.index(x) for x in tools]), tensor(mask)
        self.offsets = tensor(
            [cfg["extra_links"][s + "_sole_link"]["fixed_transform"][:3] for s in ("left", "right")] + [[0, 0, 0]]
        )
        masses, centers = [], []
        for name in links:
            inert = tree.find(f"link[@name='{name}']/inertial")
            masses.append(float(inert.find("mass").get("value")) if inert is not None else 0)
            centers.append(_xyz(inert.find("origin").get("xyz")) if inert is not None else np.zeros(3))
        self.centers = tensor(centers)
        self.mass_fraction = tensor(masses) / sum(masses)

    def _rot(self, axis, angle):
        x, y, z = axis.unbind(-1)
        o = torch.zeros_like(x)
        K = torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*x.shape, 3, 3)
        return self.eye + angle.sin()[..., None, None] * K + (1 - angle.cos())[..., None, None] * (K @ K)

    def fk(self, q, arm, jacobian=True):
        b = len(q)
        # Map constants into the arithmetic type when a complex-step evaluation is requested.
        base_axes = self.base_axes.to(q.dtype)
        R = self.eye.to(q.dtype).expand(b, -1, -1)
        axes = []
        for i in range(3):
            a = base_axes[i]
            axes.append(R @ a)
            R = R @ self._rot(a, q[:, 3 + i])
        rotations = [R[:, None]]
        positions = [q[:, None, :3]]
        link_axes = [torch.zeros_like(positions[0])]
        for parents, joints, offset, rot, axis in self.levels:
            Rp = rotations[-1][:, parents]
            p = positions[-1][:, parents] + (Rp @ offset.to(q.dtype)[..., None]).squeeze(-1)
            R0 = Rp @ rot.to(q.dtype)
            a = axis.to(q.dtype)
            rotations.append(R0 @ self._rot(a, q[:, joints]))
            positions.append(p)
            if jacobian:
                link_axes.append((R0 @ a[..., None]).squeeze(-1))
        ps = torch.cat(positions, dim=1)
        rs = torch.cat(rotations, dim=1)
        end, mask = self.end[arm]
        er = rs[:, end]
        ep = ps[:, end] + (er @ self.offsets.to(q.dtype)[..., None]).squeeze(-1)
        pcom = ps + (rs @ self.centers.to(q.dtype)[..., None]).squeeze(-1)
        com = (pcom * self.mass_fraction[:, None]).sum(1)
        pose = torch.cat((ep, 0.2 * er.transpose(-1, -2).reshape(b, 3, 9)), -1).reshape(b, -1)
        if not jacobian:
            return pose, com
        directions = torch.cat(
            (
                torch.zeros_like(q[:, :3, None]).expand(-1, -1, 3),
                torch.stack(axes, dim=1),
                torch.cat(link_axes, dim=1)[:, self.joint_links],
            ),
            dim=1,
        )
        origins = torch.cat((q[:, None, :3].expand(-1, 6, -1), ps[:, self.joint_links]), dim=1)
        angular = directions[:, None] * mask
        dp = torch.linalg.cross(angular, ep[:, :, None] - origins[:, None])
        dp = torch.cat((self.eye.to(q.dtype).expand(b, 3, 3, 3), dp[:, :, 3:]), dim=2)
        dr = torch.linalg.cross(angular[:, :, :, None], er.transpose(-1, -2)[:, :, None]).reshape(b, 3, self.n, 9)
        J = torch.cat((dp, 0.2 * dr), -1).transpose(-1, -2).reshape(b, -1, self.n)
        return pose, J, com

    def fitted(self, q, target, seed, arm):
        pose, J, _ = self.fk(q, arm)
        gradient = (J.mT @ (pose - target)[..., None]).squeeze(-1) + self.prior.square() * (q - seed)
        return pose, J, gradient

    @torch.no_grad()
    def solve(self, target, seed, arm):
        q = seed.clamp(self.lower, self.upper)
        damping = q.new_full((len(q), 1), 1e-4)
        eye = torch.eye(self.n, device=q.device, dtype=q.dtype)
        for _ in range(100):
            pose, J, g = self.fitted(q, target, seed, arm)
            low = q <= self.lower + 1e-8
            high = q >= self.upper - 1e-8
            active = (low & (g >= 0)) | (high & (g <= 0))
            optimality = g.masked_fill(active, 0).abs().amax(-1)
            if (optimality < 2e-9).all():
                break
            A = J.mT @ J + (self.prior.square() + damping).diag_embed()
            # At a box bound solve the restricted least-squares system. Projecting an
            # unconstrained direction alone can stall once one of the knees reaches a stop.
            for _ in range(4):
                free = ~active
                system = A * (free[:, :, None] & free[:, None, :]) + eye * active[:, :, None]
                step = torch.linalg.solve(system, (-g * free)[..., None]).squeeze(-1)
                outward = (low & (step < 0)) | (high & (step > 0))
                if not (outward & free).any():
                    break
                active |= outward
            change = step * (0.4 / step.abs().amax(-1, keepdim=True).clamp_min(0.4))
            candidate = (q + change).clamp(self.lower, self.upper)
            other, _ = self.fk(candidate, arm, False)
            cost = (pose - target).square().sum(-1) + ((q - seed) * self.prior).square().sum(-1)
            trial = (other - target).square().sum(-1) + ((candidate - seed) * self.prior).square().sum(-1)
            better = (trial < cost)[:, None]
            q = torch.where(better, candidate, q)
            damping = torch.where(better, damping * 0.35, damping * 5).clamp(1e-9, 100.0)
        _, j, g = self.fitted(q, target, seed, arm)
        active = ((q <= self.lower + 5e-7) & (g >= 0)) | ((q >= self.upper - 5e-7) & (g <= 0))
        return q, j, active, g.masked_fill(active, 0).abs().amax(-1)

    @torch.no_grad()
    def hessian(self, q, target, seed, arm):
        # Complex-step columns of J(q)^T r(q) + D^2(q-seed) include residual curvature.
        b, n = q.shape
        eye = torch.eye(n, device=q.device, dtype=torch.float64)
        qc = (q[:, None, :] + 1j * 1e-25 * eye).reshape(-1, n)
        _, _, g = self.fitted(qc, target.repeat_interleave(n, 0), seed.repeat_interleave(n, 0), arm)
        return (g.imag / 1e-25).reshape(b, n, n).mT


@lru_cache(maxsize=4)
def _kinematics(device):
    return _Kinematics(device)


def _targets(soles, wrist):
    b = len(soles)
    yaw = soles[..., 2]
    c, s, o = yaw.cos(), yaw.sin(), torch.zeros_like(yaw)
    r = torch.stack((c, -s, o, s, c, o, o, o, torch.ones_like(yaw)), dim=-1).reshape(b, 2, 3, 3)
    end_r = torch.cat((r, wrist[:, None, :3, :3]), dim=1)
    p = torch.cat((torch.cat((soles[..., :2], o[..., None]), -1), wrist[:, None, :3, 3]), dim=1)
    return torch.cat((p, 0.2 * end_r.transpose(-1, -2).reshape(b, 3, 9)), -1).reshape(b, -1)


def solve_stance_ik(soles, wrist, posture_seed, arm):
    """Fit both horizontal soles and the full bound wrist pose, with box-constrained T1 joints.

    The rank-restoring posture reference is an explicit input. Implicit gradients propagate
    into all three inputs, including a reference derived from a preceding gap.
    """
    b = len(soles)
    if (
        not b
        or arm not in ("left", "right")
        or soles.shape != (b, 2, 3)
        or wrist.shape != (b, 4, 4)
        or posture_seed.shape != (b, 33)
    ):
        raise ValueError("Expected a left/right T1 wrist, [B,2,3] soles, [B,4,4] wrist and [B,33] posture")
    kin = _kinematics(soles.device)
    seed = posture_seed.to(soles.device, dtype=torch.float64)
    wrist = wrist.to(soles.device, dtype=torch.float64)
    valid = soles.isfinite().all(dim=(1, 2)) & wrist.isfinite().all(dim=(1, 2)) & seed.isfinite().all(dim=1)
    # LAPACK/cuSOLVER may fail an entire batch on a single NaN. Substitute a
    # finite problem before the linear algebra and retain the row's failed certificate.
    soles = torch.where(valid[:, None, None], soles, 0)
    wrist = torch.where(valid[:, None, None], wrist, 0)
    seed = torch.where(valid[:, None], seed, 0)
    target = _targets(soles.double(), wrist)
    q, j, active, optimality = kin.solve(target, seed, arm)
    optimality = torch.where(valid, optimality, torch.inf)
    conditioning_ok = torch.ones_like(optimality, dtype=torch.bool)
    if torch.is_grad_enabled() and (target.requires_grad or seed.requires_grad):
        H = kin.hessian(q, target, seed, arm)
        free = ~active
        H = H * (free[:, :, None] & free[:, None, :]) + torch.diag_embed(active.to(H.dtype))
        ev = torch.linalg.eigvalsh((H + H.mT) * 0.5)
        conditioning_ok = (ev[:, 0] > 1e-10) & (ev[:, -1] < 1e10 * ev[:, 0])
        I = torch.eye(kin.n, device=q.device, dtype=q.dtype)
        H = torch.where(conditioning_ok[:, None, None], H, I)
        rhs = (j.mT @ (target - target.detach())[..., None]).squeeze(-1) + kin.prior.square() * (seed - seed.detach())
        q = q + torch.linalg.solve(H, (rhs * free)[..., None]).squeeze(-1) * conditioning_ok[:, None]
    pose, com = kin.fk(q, arm, False)
    residual = (pose - target).reshape(b, 3, 12)
    pos = residual[..., :3].norm(dim=-1)
    angle = 2 * (residual[..., 3:].norm(dim=-1) / (0.2 * np.sqrt(8))).clamp_max(1).asin()
    margin = physical_support_margin(com[:, :2], soles.double(), kin.box)
    fixed = physical_support_margin(com[:, :2].detach(), soles.double(), kin.box)
    wp, wr = pos[:, 2], angle[:, 2]
    sp, sr = pos[:, :2].amax(-1), angle[:, :2].amax(-1)
    ok = conditioning_ok & (optimality < 2e-6) & (wp < 0.005) & (wr < 0.05) & (sp < 0.001) & (sr < 0.01)
    return StanceIK(q.to(posture_seed.dtype), margin, fixed, com, wp, wr, sp, sr, optimality, ok, soles)
