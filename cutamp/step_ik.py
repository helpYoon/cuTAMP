"""Batched T1 sole and wrist IK with implicit differentiation of its bounded optimum."""

import math
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property, lru_cache
from threading import local

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from cutamp.com_polygon_cost import physical_support_centroid, physical_support_margin
from cutamp.stance import stance_joint_limits
from cutamp.robots.t1 import REAL_LEGS, t1_curobo_cfg
from cutamp.supported_posture import PostureCollisionSpheres

# Match the supported endpoint fit's CoM-to-pose residual ratio (6 to 50).
_CENTER_WEIGHT = 6.0 / 50.0
_SELF_COLLISION_WEIGHT = 24.0 / 50.0


_CUDA_GRAPHS = ContextVar("stance_ik_cuda_graphs", default=True)
_IK_TOLERANCE = ContextVar("stance_ik_tolerance", default=2e-9)


@contextmanager
def stance_ik_graphs(enabled):
    """Scope graph reuse to the planner request and its calling thread."""
    token = _CUDA_GRAPHS.set(enabled)
    try:
        yield
    finally:
        _CUDA_GRAPHS.reset(token)


@contextmanager
def stance_ik_tolerance(value):
    """Scope the numerical solve tolerance. Pose and landing acceptance have separate gates."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Stance IK solve tolerance must be finite and positive")
    token = _IK_TOLERANCE.set(value)
    try:
        yield
    finally:
        _IK_TOLERANCE.reset(token)


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


class _StanceIterationGraph:
    """Capture one bounded damped least-squares iteration at a fixed batch shape."""

    def __init__(self, kin, target, seed, arm, lower, upper, prior, weight, fixed_idle=False):
        self.target = torch.empty_like(target)
        self.seed = torch.empty_like(seed)
        self.lower, self.upper, self.prior = [torch.empty_like(seed) for _ in range(3)]
        self.q = torch.empty_like(seed)
        self.damping = seed.new_empty((len(seed), 1))
        self.converged = torch.empty(len(seed), dtype=torch.bool, device=seed.device)
        self.invalid = self.converged.new_zeros(())
        # Equality-constrained coordinates take a zero step in every active-set pass.
        # Factor only the remaining principal block while evaluating FK and cost in full.
        self.variables = None
        width = kin.n
        if fixed_idle:
            idle = REAL_LEGS.right_arm if arm == "left" else REAL_LEGS.left_arm
            columns = [i for i in range(kin.n) if not idle.start <= i < idle.stop]
            self.variables = torch.tensor(columns, device=seed.device)
            width = len(columns)
        self.eye = torch.eye(width, device=seed.device, dtype=seed.dtype)
        self.pair_ids = torch.arange(len(kin.self_spheres.i), device=seed.device) if weight else None
        self.reset(target, seed, lower, upper, prior)
        caller = torch.cuda.current_stream(seed.device)
        stream = torch.cuda.Stream(device=seed.device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            self.iterate(kin, arm, weight)
        caller.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.iterate(kin, arm, weight)

    def reset(self, target, seed, lower, upper, prior):
        for buf, value in zip((self.target, self.seed, self.lower, self.upper, self.prior),
                              (target, seed, lower, upper, prior), strict=True):
            buf.copy_(value)
        self.q.copy_(seed.clamp(lower, upper))
        self.damping.fill_(1e-4)
        self.invalid.zero_()

    def collisions(self, spheres, frames):
        """Keep original pair order and flag overflow; eager fallback covers crowded cases."""
        pos, rotation, _, axes = frames
        points = pos[:, spheres.links] + (rotation[:, spheres.links] @ spheres.offsets[..., None]).squeeze(-1)
        delta = points[:, spheres.i] - points[:, spheres.j]
        limit = spheres.radii_sum + 0.006
        n = len(self.pair_ids)
        active = delta.square().sum(-1) < limit.square()
        ranks = torch.where(active, n - self.pair_ids, 0)
        k = min(48, n)
        values, pairs = ranks.topk(k, sorted=True)
        self.invalid.logical_or_((active.sum(-1) > k).any())
        rows = torch.arange(len(pos), device=pos.device)[:, None]
        i, j = spheres.i[pairs], spheres.j[pairs]
        delta = points[rows, i] - points[rows, j]
        distance = delta.square().sum(-1).clamp_min(1e-24).sqrt()
        val = (limit[pairs] - distance) * (values > 0)
        lever = points[rows, i, None] - pos[:, None, spheres.joint_links]
        unit = (delta / distance[..., None])[:, :, None].expand_as(lever)
        joint = (torch.linalg.cross(lever, unit) * axes[:, None]).sum(-1)
        mask = (spheres.descendants.mT[j] - spheres.descendants.mT[i]) * (values > 0)[..., None]
        jac = torch.cat((pos.new_zeros(len(pos), k, 6), joint * mask), -1)
        return val, jac

    def iterate(self, kin, arm, weight):
        q, seed, target = self.q, self.seed, self.target
        lo, hi, prior = self.lower, self.upper, self.prior
        if weight:
            frames = kin.link_frames(q, True)
            pose, J, com, Jcom = kin.fk(q, arm, frames=frames, com_jacobian=True)
            collision, Jcollision = self.collisions(kin.self_spheres, frames)
            residual = torch.cat((torch.cat((pose, weight * com[:, :2]), -1) - target,
                                  _SELF_COLLISION_WEIGHT * collision), -1)
            J = torch.cat((J, weight * Jcom[:, :2], _SELF_COLLISION_WEIGHT * Jcollision), 1)
            grad = (J.mT @ residual[..., None]).squeeze(-1) + prior.square() * (q - seed)
        else:
            residual, J, grad = kin.fitted(q, target, seed, arm, prior=prior)
        low, high = q <= lo + 1e-8, q >= hi - 1e-8
        active = (low & (grad >= 0)) | (high & (grad <= 0))
        done = grad.masked_fill(active, 0).abs().amax(-1) < _IK_TOLERANCE.get()
        self.converged.copy_(done)
        finite_grad = grad.isfinite().all(-1)
        damping_diag = prior.square() + self.damping
        if self.variables is not None:
            ids = self.variables
            J = J.index_select(-1, ids)
            damping_diag = damping_diag.index_select(-1, ids)
            grad, low, high, active = (v.index_select(-1, ids) for v in (grad, low, high, active))
        A = J.mT @ J + damping_diag.diag_embed()
        # Positive damping makes every free block positive definite. These native
        # CUDA operations support graph capture, including the restricted solves.
        for _ in range(4):
            free = ~active
            system = A * (free[:, :, None] & free[:, None, :]) + self.eye * active[:, :, None]
            chol, info = torch.linalg.cholesky_ex(system)
            self.invalid.logical_or_((info != 0).any())
            tmp = torch.linalg.solve_triangular(chol, (-grad * free)[..., None], upper=False)
            step = torch.linalg.solve_triangular(chol.mT, tmp, upper=True).squeeze(-1)
            active |= (low & (step < 0)) | (high & (step > 0))
        if self.variables is not None:
            step = torch.zeros_like(q).index_copy_(-1, self.variables, step)
        change = step * (0.4 / step.abs().amax(-1, keepdim=True).clamp_min(0.4))
        candidate = (q + change).clamp(lo, hi)
        frames = kin.link_frames(candidate)
        other, com = kin.fk(candidate, arm, False, frames=frames)
        if weight:
            collision, _ = kin.self_spheres.residual(frames)
            other = torch.cat((other, weight * com[:, :2]), -1)
        cost = residual.square().sum(-1) + ((q - seed) * prior).square().sum(-1)
        trial = (other - target).square().sum(-1) + ((candidate - seed) * prior).square().sum(-1)
        if weight:
            trial = trial + (_SELF_COLLISION_WEIGHT * collision).square().sum(-1)
        finite = step.isfinite().all(-1) & finite_grad & cost.isfinite() & trial.isfinite()
        self.invalid.logical_or_((~finite).any())
        better = ((trial < cost) & ~done)[:, None]
        self.q.copy_(torch.where(better, candidate, q))
        self.damping.copy_(torch.where(better, self.damping * 0.35, self.damping * 5.0).clamp(1e-9, 100.))

def _graph_pose(kin, target, seed, arm, lower, upper, prior, center_weight, iterations, initial=None):
    cache = getattr(kin._solve_graphs, "cache", None)
    if cache is None:
        cache = kin._solve_graphs.cache = {}
    stream = torch.cuda.current_stream(seed.device)
    idle = REAL_LEGS.right_arm if arm == "left" else REAL_LEGS.left_arm
    fixed_idle = bool(center_weight) and torch.equal(lower[..., idle], upper[..., idle])

    def setup(t, s, lo, hi, p):
        key = (len(s), arm, center_weight, fixed_idle, stream.cuda_stream, torch.is_inference_mode_enabled(), _IK_TOLERANCE.get())
        if key not in cache:
            if len(cache) >= 16:
                cache.pop(next(iter(cache)))
            cache[key] = _StanceIterationGraph(kin, t, s, arm, lo, hi, p, center_weight, fixed_idle)
        op = cache[key]
        op.reset(t, s, lo, hi, p)
        return op

    op = setup(target, seed, lower, upper, prior)
    if initial is not None:
        op.q.copy_(initial.clamp(lower, upper))
    rows = torch.arange(len(seed), device=seed.device)
    completed = None
    for start in range(0, iterations, 8):
        for _ in range(min(8, iterations - start)):
            op.graph.replay()
        unfinished = ~op.converged[:len(rows)]
        count = int(unfinished.sum())
        if not count:
            break
        # Each converged particle is frozen immediately. Drop it from later replay
        # once the working batch halves. Pad only with active rows to reuse graph sizes.
        if 0 < count <= len(op.q) // 2 and len(op.q) > 4 and start + 8 < iterations:
            if op.invalid:
                return None
            if completed is None:
                completed = op.q.clone()
            else:
                completed.index_copy_(0, rows, op.q[:len(rows)])
            keep = unfinished.nonzero().flatten()
            rows = rows[keep]
            size = max(4, 1 << (count - 1).bit_length())
            keep = torch.cat((keep, keep[-1:].expand(size - count)))
            q, damping = op.q[keep], op.damping[keep]
            op = setup(*(v[keep] for v in (op.target, op.seed, op.lower, op.upper, op.prior)))
            op.q.copy_(q)
            op.damping.copy_(damping)
    if op.invalid:
        return None
    if completed is None:
        return op.q.clone()
    completed.index_copy_(0, rows, op.q[:len(rows)])
    return completed


class _Kinematics:
    """Level-ordered URDF FK for both arms, sole centers, and the complete robot mass model.

    Double precision and explicit geometric Jacobians also allow complex-step evaluation
    of the *full* Hessian of the pose and posture residual objective on its active branch.
    """

    def __init__(self, device, cfg=None):
        model = REAL_LEGS
        if cfg is None:
            cfg = t1_curobo_cfg(model)["robot_cfg"]["kinematics"]
        self.cfg = cfg
        self._solve_graphs = local()
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
        # Allow redundant joints to center the CoM while retaining a local, well-conditioned posture fit.
        self.centered_prior = 0.5 * self.prior
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
        self.descendants = tensor([[j in ancestors[name] for name in links] for j in range(self.n)])
        self.descendant_mass = self.descendants * self.mass_fraction
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

    @cached_property
    def self_spheres(self):
        return PostureCollisionSpheres(self.cfg, self)

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
            moment = self.descendant_mass.to(q.dtype) @ pcom - self.subtree_mass * origins
            angular_com = torch.linalg.cross(directions[:, 3:], moment[:, 3:])
            Jcom = torch.cat((self.eye.expand(b, -1, -1), angular_com.mT), dim=-1)
            return pose, J, com, Jcom
        return pose, J, com

    def fitted(self, q, target, seed, arm, center_weight=0.0, prior=None, repeats=1):
        prior = self.prior if prior is None else prior
        if center_weight:
            frames = self.link_frames(q, True)
            pose, J, com, Jcom = self.fk(q, arm, frames=frames, com_jacobian=True)
            collision, Jcollision = self.self_spheres.residual(frames, repeats)
            pose = torch.cat((pose, center_weight * com[:, :2]), -1)
            residual = torch.cat((pose - target, _SELF_COLLISION_WEIGHT * collision), -1)
            J = torch.cat((J, center_weight * Jcom[:, :2], _SELF_COLLISION_WEIGHT * Jcollision), 1)
        else:
            pose, J, _ = self.fk(q, arm)
            residual = pose - target
        gradient = (J.mT @ residual[..., None]).squeeze(-1) + prior.square() * (q - seed)
        return residual, J, gradient

    @torch.no_grad()
    def solve(self, target, seed, arm, *, bounds=None, center_weight=0.0, iterations=100, prior=None, initial=None):
        prior = self.prior if prior is None else prior
        lower, upper = (self.lower, self.upper) if bounds is None else bounds
        if (_CUDA_GRAPHS.get() and seed.is_cuda and seed.dtype == target.dtype == torch.float64 and 16 <= len(seed) <= 128
                and iterations > 0 and not torch.cuda.is_current_stream_capturing()):
            q = _graph_pose(self, target, seed, arm, lower, upper, prior, center_weight, iterations, initial)
            if q is not None:
                _, j, g = self.fitted(q, target, seed, arm, center_weight, prior)
                active = ((q <= lower + 5e-7) & (g >= 0)) | ((q >= upper - 5e-7) & (g <= 0))
                return q, j[:, :target.shape[1]], active, g.masked_fill(active, 0).abs().amax(-1)
        full_problem = target, seed, lower.expand_as(seed), upper.expand_as(seed), prior.expand_as(seed)
        rows = torch.arange(len(seed), device=seed.device)
        completed = None
        q = (seed if initial is None else initial).clamp(lower, upper)
        damping = q.new_full((len(q), 1), 1e-4)
        eye = torch.eye(self.n, device=q.device, dtype=q.dtype)
        for _ in range(iterations):
            residual, J, g = self.fitted(q, target, seed, arm, center_weight, prior)
            low = q <= lower + 1e-8
            high = q >= upper - 1e-8
            active = (low & (g >= 0)) | (high & (g <= 0))
            optimality = g.masked_fill(active, 0).abs().amax(-1)
            converged = optimality < _IK_TOLERANCE.get()
            count = int((~converged).sum())
            if not count:
                break
            # Pack only when the batch halves to amortize GPU gathers. Converged rows stay fixed until then.
            compact = count <= len(q) // 2
            if compact:
                keep = (~converged).nonzero().flatten()
            A = J.mT @ J + (prior.square() + damping).diag_embed()
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
            frames = self.link_frames(candidate)
            other, com = self.fk(candidate, arm, False, frames=frames)
            if center_weight:
                collision, _ = self.self_spheres.residual(frames)
                other = torch.cat((other, center_weight * com[:, :2]), -1)
            cost = residual.square().sum(-1) + ((q - seed) * prior).square().sum(-1)
            trial = (other - target).square().sum(-1) + ((candidate - seed) * prior).square().sum(-1)
            if center_weight:
                trial += (_SELF_COLLISION_WEIGHT * collision).square().sum(-1)
            better = ((trial < cost) & ~converged)[:, None]
            q = torch.where(better, candidate, q)
            damping = torch.where(better, damping * 0.35, damping * 5).clamp(1e-9, 100.0)
            if compact:
                # Scatter the finished poses before shrinking the working set for the next iteration.
                if completed is None:
                    completed = q.clone()
                else:
                    completed.index_copy_(0, rows, q)
                rows = rows.index_select(0, keep)
                q = q.index_select(0, keep)
                damping = damping.index_select(0, keep)
                target, seed, lower, upper, prior = (v.index_select(0, rows) for v in full_problem)
        if completed is not None:
            completed.index_copy_(0, rows, q)
            q = completed
            target, seed, lower, upper, prior = full_problem
        _, j, g = self.fitted(q, target, seed, arm, center_weight, prior)
        active = ((q <= lower + 5e-7) & (g >= 0)) | ((q >= upper - 5e-7) & (g <= 0))
        return q, j[:, :target.shape[1]], active, g.masked_fill(active, 0).abs().amax(-1)

    @torch.no_grad()
    def hessian(self, q, target, seed, arm, center_weight=0.0, prior=None):
        # Complex-step columns of J(q)^T r(q) + D^2(q - seed) include residual curvature.
        b, n = q.shape
        eye = torch.eye(n, device=q.device, dtype=torch.float64)
        qc = (q[:, None, :] + 1j * 1e-25 * eye).reshape(-1, n)
        _, _, g = self.fitted(
            qc, target.repeat_interleave(n, 0), seed.repeat_interleave(n, 0), arm, center_weight, prior, repeats=n
        )
        return (g.imag / 1e-25).reshape(b, n, n).mT


@lru_cache(maxsize=4)
def _kinematics(device):
    return _Kinematics(device)


def _well_conditioned(hessian):
    """The implicit derivative requires a positive, well-conditioned free-joint Hessian."""
    symmetric = (hessian + hessian.mT) * 0.5
    # The maximum absolute row sum bounds the largest eigenvalue. Positive
    # definiteness after a conservative shift certifies both original tests.
    # The factor of two keeps near-threshold decisions on the eigenvalue path.
    upper = symmetric.abs().sum(-1).amax(-1)
    shift = 2 * (upper * 1e-10).clamp_min(1e-10)
    shifted = symmetric.clone()
    shifted.diagonal(dim1=-2, dim2=-1).sub_(shift[:, None])
    _, info = torch.linalg.cholesky_ex(shifted)
    certified = (info == 0) & upper.isfinite()
    unresolved = (~certified).nonzero().flatten()
    if len(unresolved):
        ev = torch.linalg.eigvalsh(symmetric.index_select(0, unresolved))
        certified[unresolved] = (ev[:, 0] > 1e-10) & (ev[:, -1] < 1e10 * ev[:, 0])
    return certified


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


def solve_stance_ik(soles, wrist, posture_seed, arm, *, other_arm_reference=None, collides=None, kinematics=None,
                    center_posture=False, initial=None):
    """Fit both horizontal soles and the full bound wrist pose, with box-constrained T1 joints.

    The other arm uses its nominal joint posture, or the reference of an occupied hand.
    If nominal collides, try progressively larger excursions toward the supplied witness.
    Both arms contribute their actual mass and pose to the resulting CoM.
    An optional numerical initial guess leaves the posture regularizer at posture_seed.
    At a centered manipulation endpoint the horizontal CoM-to-hull-centroid
    residual participates in both the bounded solve and its implicit derivative.
    """
    b = len(soles)
    if (
        not b
        or arm not in ("left", "right")
        or soles.shape != (b, 2, 3)
        or wrist.shape != (b, 4, 4)
        or posture_seed.shape != (b, 33)
        or (initial is not None and initial.shape != posture_seed.shape)
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
    if initial is not None:
        initial = initial.detach().to(seed)
        initial = torch.where(initial.isfinite().all(-1, keepdim=True), initial, seed)
    pose_target = _targets(soles.double(), wrist)
    target = pose_target
    weight = _CENTER_WEIGHT if center_posture else 0.0
    regularizer = kin.centered_prior if center_posture else kin.prior
    if center_posture:
        centroid = physical_support_centroid(soles.double(), kin.box)
        target = torch.cat((target, weight * centroid), dim=-1)
    bounds = (stance_joint_limits(soles.double(), kin.lower, kin.upper) if kinematics is None
              else (kin.lower.expand_as(seed), kin.upper.expand_as(seed)))

    def fit(reference_pose, start=None):
        lower, upper = bounds
        if center_posture:
            lower, upper = lower.clone(), upper.clone()
            lower[:, other] = upper[:, other] = reference_pose[:, other]
        start_args = {} if start is None else {"initial": start}
        q, j, active, optimality = kin.solve(target, reference_pose, arm, bounds=(lower, upper), center_weight=weight,
                                             prior=regularizer, **start_args)
        conditioning_ok = torch.ones_like(optimality, dtype=torch.bool)
        if not torch.is_grad_enabled() or not (target.requires_grad or reference_pose.requires_grad):
            return q, optimality, conditioning_ok
        H = kin.hessian(q, target, reference_pose, arm, weight, regularizer)
        free = ~active
        dref = reference_pose - reference_pose.detach()
        rhs = (j.mT @ (target - target.detach())[..., None]).squeeze(-1) + regularizer.square() * dref
        # Equality-constrained joints follow the prescribed arm reference. Their
        # mass couples into the free-joint optimum through the off-diagonal Hessian.
        dq_fixed = torch.zeros_like(q)
        if center_posture:
            dq_fixed[:, other] = dref[:, other]
            rhs = rhs - (H @ dq_fixed[..., None]).squeeze(-1)
        H = H * (free[:, :, None] & free[:, None, :]) + torch.diag_embed(active.to(H.dtype))
        conditioning_ok = _well_conditioned(H)
        I = torch.eye(kin.n, device=q.device, dtype=q.dtype)
        H = torch.where(conditioning_ok[:, None, None], H, I)
        dq = dq_fixed + torch.linalg.solve(H, (rhs * free)[..., None]).squeeze(-1)
        return q + dq * conditioning_ok[:, None], optimality, conditioning_ok

    q, optimality, conditioning_ok = fit(seed, initial)
    collision_free = torch.ones_like(valid)
    if collides is not None:
        def blocked(candidate):
            with torch.no_grad():
                return collides(candidate.to(posture_seed.dtype))

        collision = blocked(q)
        can_move = valid & (alternate.to(posture_seed.dtype) != reference.to(posture_seed.dtype)).any(-1)
        if other_arm_reference is None and (collision & can_move).any():
            # An idle-arm excursion changes the centered optimum through its mass.
            alternate = torch.where(valid[:, None], alternate, 0).clamp(kin.lower[other], kin.upper[other])
            reference = torch.where(valid[:, None], reference, 0)
            for fraction in (0.25, 0.5, 1.0):
                candidate = q.clone()
                candidate[:, other] = torch.lerp(reference, alternate, fraction)
                if center_posture:
                    changed = seed.clone()
                    changed[:, other] = candidate[:, other]
                    candidate, candidate_opt, candidate_condition = fit(changed)
                cleared = collision & ~blocked(candidate)
                q = torch.where(cleared[:, None], candidate, q)
                if center_posture:
                    optimality = torch.where(cleared, candidate_opt, optimality)
                    conditioning_ok = torch.where(cleared, candidate_condition, conditioning_ok)
                collision = collision & ~cleared
                if not (collision & can_move).any():
                    break
        collision_free = ~collision
    pose, com = kin.fk(q, arm, False)
    optimality = torch.where(valid, optimality, torch.inf)
    residual = (pose - pose_target).reshape(b, 3, 12)
    pos = residual[..., :3].norm(dim=-1)
    angle = 2 * (residual[..., 3:].norm(dim=-1) / (0.2 * np.sqrt(8))).clamp_max(1).asin()
    margin = physical_support_margin(com[:, :2], soles.double(), kin.box)
    fixed = physical_support_margin(com[:, :2].detach(), soles.double(), kin.box)
    wp, wr = pos[:, 2], angle[:, 2]
    sp, sr = pos[:, :2].amax(-1), angle[:, :2].amax(-1)
    pose_ok = (wp < 0.005) & (wr < 0.05) & (sp < 0.001) & (sr < 0.01)
    ok = valid & collision_free & conditioning_ok & (optimality < 2e-6) & pose_ok
    return StanceIK(q.to(posture_seed.dtype), margin, fixed, com, wp, wr, sp, sr, optimality, ok, soles)
