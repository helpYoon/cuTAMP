"""Batched T1 sole and wrist IK with implicit differentiation of its bounded optimum."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from cutamp.com_polygon_cost import physical_support_centroid, physical_support_margin
from cutamp.stance import stance_joint_limits
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

    def __init__(self, device, cfg=None):
        model = REAL_LEGS
        if cfg is None:
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
        self.eye = torch.eye(3, device=device, dtype=torch.float64)
        children = {}
        for joint in tree.findall("joint"):
            if joint.get("name") == "world_joint":
                continue
            children.setdefault(joint.find("parent").get("link"), []).append(joint)
        self.levels = []
        origin_frames, joint_axes, angle_columns = [], [], []
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
                first = len(origin_frames)
                for offset, rotation in zip(p, r):
                    frame = np.eye(4)
                    frame[:3, :3], frame[:3, 3] = rotation, offset
                    origin_frames.append(frame)
                joint_axes.extend(a)
                angle_columns.extend(jx)
                self.levels.append((indices(ix), slice(first, len(origin_frames))))
            links.extend(limb)
            parents = limb
        # Constant Rodrigues coefficients let every local joint transform be evaluated
        # in one batch. Only the parent/child products depend on the tree level.
        origins = np.stack(origin_frames)
        axes = np.array(joint_axes)
        skew = np.zeros_like(origins)
        skew[:, 0, 1], skew[:, 1, 0] = -axes[:, 2], axes[:, 2]
        skew[:, 2, 0], skew[:, 0, 2] = -axes[:, 1], axes[:, 1]
        skew[:, 1, 2], skew[:, 2, 1] = -axes[:, 0], axes[:, 0]
        self.origins = tensor(origins)
        self.sin_coeff = tensor(origins @ skew)
        self.cos_coeff = tensor(origins @ skew @ skew)
        self.angle_columns = indices(angle_columns)
        self.axes = tensor(axes[np.array(joint_links[6:]) - 1])
        clip = dict(zip(cfg["cspace"]["joint_names"], cfg["cspace"]["position_limit_clip"]))
        widths = np.array([clip[name] for name in names])
        self.lower, self.upper = tensor(lower + widths), tensor(upper - widths)
        self.prior = tensor([0.035] * 3 + [0.008] * (self.n - 3))
        self.joint_links = indices(joint_links[6:])
        self.links = tuple(links)
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
        descendant = [[j in ancestors[name] for name in links] for j in range(self.n)]
        self.descendant_mass = tensor(descendant) * self.mass_fraction
        self.subtree_mass = self.descendant_mass.sum(-1, keepdim=True)

    def link_frames(self, q, jacobian=False):
        b = len(q)
        sy, sx, sz = q[:, 3:6].sin().unbind(-1)
        cy, cx, cz = q[:, 3:6].cos().unbind(-1)
        # The floating base uses intrinsic Y-X-Z rotations.
        rotation = torch.stack((
            cy * cz + sy * sx * sz, -cy * sz + sy * sx * cz, sy * cx,
            cx * sz, cx * cz, -sx,
            -sy * cz + cy * sx * sz, sy * sz + cy * sx * cz, cy * cx,
        ), -1).reshape(b, 3, 3)
        base = q.new_zeros((b, 1, 4, 4))
        base[:, 0, :3, :3], base[:, 0, :3, 3], base[:, 0, 3, 3] = rotation, q[:, :3], 1
        angles = q[:, self.angle_columns, None, None]
        local = self.origins + angles.sin() * self.sin_coeff + (1 - angles.cos()) * self.cos_coeff
        chain = [base]
        for parents, level in self.levels:
            chain.append(chain[-1].index_select(1, parents) @ local[:, level])
        frames = torch.cat(chain, dim=1)
        rs = frames[:, :, :3, :3]
        axes = link_axes = None
        if jacobian:
            zero = torch.zeros_like(cy)
            axes = torch.stack((zero, torch.ones_like(cy), zero, cy, zero, -sy, sy * cx, -sx, cy * cx), -1)
            axes = axes.reshape(b, 3, 3)
            link_axes = (rs.index_select(1, self.joint_links) @ self.axes.to(q.dtype)[..., None]).squeeze(-1)
        return frames[:, :, :3, 3], rs, axes, link_axes

    def fk(self, q, arm, jacobian=True, *, frames=None, com_jacobian=False):
        b = len(q)
        ps, rs, axes, link_axes = self.link_frames(q, jacobian) if frames is None else frames
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
                axes,
                link_axes,
            ),
            dim=1,
        )
        origins = torch.cat((q[:, None, :3].expand(-1, 6, -1), ps[:, self.joint_links]), dim=1)
        angular = directions[:, None] * mask
        dp = torch.linalg.cross(angular, ep[:, :, None] - origins[:, None])
        dp = torch.cat((self.eye.to(q.dtype).expand(b, 3, 3, 3), dp[:, :, 3:]), dim=2)
        dr = torch.linalg.cross(angular[:, :, :, None], er.transpose(-1, -2)[:, :, None]).reshape(b, 3, self.n, 9)
        J = torch.cat((dp, 0.2 * dr), -1).transpose(-1, -2).reshape(b, -1, self.n)
        if com_jacobian:
            # The derivative of each moving mass is axis × (mass center - joint origin).
            # Sum over descendants before taking the cross product for each joint.
            moment = self.descendant_mass @ pcom - self.subtree_mass * origins
            angular_com = torch.linalg.cross(directions[:, 3:], moment[:, 3:])
            Jcom = torch.cat((self.eye.expand(b, -1, -1), angular_com.mT), dim=-1)
            return pose, J, com, Jcom
        return pose, J, com

    def fitted(self, q, target, seed, arm, center_weight=0.0):
        if center_weight:
            pose, J, com, Jcom = self.fk(q, arm, com_jacobian=True)
            pose = torch.cat((pose, center_weight * com[:, :2]), -1)
            J = torch.cat((J, center_weight * Jcom[:, :2]), 1)
        else:
            pose, J, _ = self.fk(q, arm)
        gradient = (J.mT @ (pose - target)[..., None]).squeeze(-1) + self.prior.square() * (q - seed)
        return pose, J, gradient

    @torch.no_grad()
    def solve(self, target, seed, arm, *, bounds=None, center_weight=0.0, iterations=100):
        lower, upper = (self.lower, self.upper) if bounds is None else bounds
        q = seed.clamp(lower, upper)
        damping = q.new_full((len(q), 1), 1e-4)
        eye = torch.eye(self.n, device=q.device, dtype=q.dtype)
        for _ in range(iterations):
            pose, J, g = self.fitted(q, target, seed, arm, center_weight)
            low = q <= lower + 1e-8
            high = q >= upper - 1e-8
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
            candidate = (q + change).clamp(lower, upper)
            other, com = self.fk(candidate, arm, False)
            if center_weight:
                other = torch.cat((other, center_weight * com[:, :2]), -1)
            cost = (pose - target).square().sum(-1) + ((q - seed) * self.prior).square().sum(-1)
            trial = (other - target).square().sum(-1) + ((candidate - seed) * self.prior).square().sum(-1)
            better = (trial < cost)[:, None]
            q = torch.where(better, candidate, q)
            damping = torch.where(better, damping * 0.35, damping * 5).clamp(1e-9, 100.0)
        _, j, g = self.fitted(q, target, seed, arm, center_weight)
        active = ((q <= lower + 5e-7) & (g >= 0)) | ((q >= upper - 5e-7) & (g <= 0))
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


@torch.no_grad()
def centered_stance_witnesses(soles, wrist, seeds, arm, other, weight):
    """Initialize all particles' wrist, sole and projected-CoM residuals in one bounded solve."""
    kin = _kinematics(soles.device)
    soles = soles.double()
    seed = seeds.double().clone()
    idle = REAL_LEGS.right_arm if arm == "left" else REAL_LEGS.left_arm
    nominal = seed.new_tensor(REAL_LEGS.home[idle]) if other is None else other
    seed[:, idle] = nominal.to(seed)
    lower, upper = stance_joint_limits(soles, kin.lower, kin.upper)
    lower[:, idle] = upper[:, idle] = seed[:, idle]
    # As in fit_centered_posture, the wrist and sole residuals have weight 50.
    center_weight = weight / 50.0
    center = physical_support_centroid(soles, kin.box)
    target = torch.cat((_targets(soles, wrist.double()), center_weight * center), -1)
    q = kin.solve(target, seed, arm, bounds=(lower, upper), center_weight=center_weight, iterations=30)[0]
    return torch.where(q.isfinite().all(-1, keepdim=True), q, seed).to(seeds)


def solve_stance_ik(soles, wrist, posture_seed, arm, *, other_arm_reference=None, collides=None, kinematics=None):
    """Fit both horizontal soles and the full bound wrist pose, with box-constrained T1 joints.

    The other arm uses its nominal joint posture, or the reference of an occupied hand.
    If nominal collides, try progressively larger excursions toward the supplied witness.
    Both arms contribute their actual mass and pose to the resulting CoM.
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
    kin = _kinematics(soles.device) if kinematics is None else kinematics
    seed = posture_seed.to(soles.device, dtype=torch.float64)
    other = REAL_LEGS.right_arm if arm == "left" else REAL_LEGS.left_arm
    alternate = seed[:, other]
    reference = REAL_LEGS.home[other] if other_arm_reference is None else other_arm_reference
    reference = torch.as_tensor(reference, device=seed.device, dtype=seed.dtype).expand_as(alternate)
    seed = seed.clone()
    seed[:, other] = reference
    wrist = wrist.to(soles.device, dtype=torch.float64)
    valid = soles.isfinite().all(dim=(1, 2)) & wrist.isfinite().all(dim=(1, 2)) & seed.isfinite().all(dim=1)
    valid &= alternate.isfinite().all(dim=1)
    valid &= (reference >= kin.lower[other]).all(dim=1) & (reference <= kin.upper[other]).all(dim=1)
    # LAPACK/cuSOLVER may fail an entire batch on a single NaN. Substitute a
    # finite problem before the linear algebra and retain the row's failed certificate.
    soles = torch.where(valid[:, None, None], soles, 0)
    wrist = torch.where(valid[:, None, None], wrist, 0)
    seed = torch.where(valid[:, None], seed, 0)
    target = _targets(soles.double(), wrist)
    bounds = stance_joint_limits(soles.double(), kin.lower, kin.upper) if kinematics is None else None
    q, j, active, optimality = kin.solve(target, seed, arm, bounds=bounds)
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
    if collides is not None:
        def blocked(candidate):
            with torch.no_grad():
                return collides(candidate.to(posture_seed.dtype))

        collision = blocked(q)
        if other_arm_reference is None and collision.any():
            # None of the two sole or active-wrist poses depends on the idle arm.
            # Its alternate can be checked on the fitted torso without another IK solve.
            alternate = torch.where(valid[:, None], alternate, 0).clamp(kin.lower[other], kin.upper[other])
            reference = torch.where(valid[:, None], reference, 0)
            for fraction in (0.25, 0.5, 1.0):
                candidate = q.clone()
                candidate[:, other] = torch.lerp(reference, alternate, fraction)
                cleared = collision & ~blocked(candidate)
                q = torch.where(cleared[:, None], candidate, q)
                collision = collision & ~cleared
                if not collision.any():
                    break
        valid = valid & ~collision
    pose, com = kin.fk(q, arm, False)
    residual = (pose - target).reshape(b, 3, 12)
    pos = residual[..., :3].norm(dim=-1)
    angle = 2 * (residual[..., 3:].norm(dim=-1) / (0.2 * np.sqrt(8))).clamp_max(1).asin()
    margin = physical_support_margin(com[:, :2], soles.double(), kin.box)
    fixed = physical_support_margin(com[:, :2].detach(), soles.double(), kin.box)
    wp, wr = pos[:, 2], angle[:, 2]
    sp, sr = pos[:, :2].amax(-1), angle[:, :2].amax(-1)
    ok = valid & conditioning_ok & (optimality < 2e-6) & (wp < 0.005) & (wr < 0.05) & (sp < 0.001) & (sr < 0.01)
    return StanceIK(q.to(posture_seed.dtype), margin, fixed, com, wp, wr, sp, sr, optimality, ok, soles)
