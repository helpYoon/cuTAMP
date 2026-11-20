"""T1 full wrist and stance IK for a grasp-to-foot derivative diagnostic.

IEEE double local bounded least squares; exact geometric Jacobians and complex-step
differentiation of the stationarity system for the active-set implicit derivative.
Scores the two achieved SE(2) sole poses as flat double support at the grasp.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def fromxyz(s):
    return np.fromstring(s, sep=" ")


def skew(axis):
    x, y, z = axis
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class GraspIK:
    def __init__(self, root=None):
        root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        self.root = root
        assets = root / "cutamp/robots/assets/t1_29dof"
        cfg = yaml.safe_load((assets / "t1_29dof.yml").read_text())
        task = cfg["cutamp"]
        self.names = task["joint_order"]
        self.home = np.array(task["home"])
        kin = cfg["robot_cfg"]["kinematics"]
        locks = kin["lock_joints"]
        self.foot_box = np.array(task["sole_box"])
        tree = ET.parse(assets / "t1_planner.urdf").getroot()
        self.joints = []
        self.link_names = ["Trunk"]
        self.chains = {"Trunk": list(range(6))}
        self.lo = np.zeros(33)
        self.hi = np.zeros(33)
        for i, sub in enumerate(("x", "y", "z", "pitch", "roll", "yaw")):
            self.lo[i], self.hi[i] = kin["extra_links"]["base_link_" + sub]["joint_limits"]
        for j in tree.findall("joint"):
            name = j.get("name")
            if name == "world_joint":
                continue
            child = j.find("child").get("link")
            par = j.find("parent").get("link")
            orig = j.find("origin")
            p = fromxyz(orig.get("xyz"))
            r = Rotation.from_euler("xyz", fromxyz(orig.get("rpy"))).as_matrix()
            ai = j.find("axis")
            axis = fromxyz(ai.get("xyz")) if ai is not None else np.zeros(3)
            idx = self.names.index(name) if name in self.names else -1
            if name in locks:
                r = r @ Rotation.from_rotvec(axis * locks[name]).as_matrix()
            if idx >= 0:
                self.lo[idx] = float(j.find("limit").get("lower"))
                self.hi[idx] = float(j.find("limit").get("upper"))
            k = skew(axis)
            self.joints.append((self.link_names.index(par), idx, p, r, axis, k, k @ k))
            self.link_names.append(child)
            self.chains[child] = self.chains[par] + ([idx] if idx >= 0 else [])
        clip = dict(zip(kin["cspace"]["joint_names"], kin["cspace"]["position_limit_clip"]))
        self.lo += np.array([clip[n] for n in self.names])
        self.hi -= np.array([clip[n] for n in self.names])
        self.endlinks = ["left_foot_link", "right_foot_link", "left_base_link"]
        self.endidx = np.array([self.link_names.index(x) for x in self.endlinks])
        self.endoffset = np.array(
            [kin["extra_links"][x + "_sole_link"]["fixed_transform"][:3] for x in ("left", "right")] + [[0.0, 0.0, 0.0]]
        )
        self.mask = np.zeros((3, 33, 1))
        for row, link in enumerate(self.endlinks):
            self.mask[row, self.chains[link], 0] = 1
        mass = []
        comloc = []
        for link in self.link_names:
            inert = tree.find("link[@name='" + link + "']/inertial")
            mass.append(float(inert.find("mass").get("value")) if inert is not None else 0.0)
            comloc.append(fromxyz(inert.find("origin").get("xyz")) if inert is not None else np.zeros(3))
        self.mass = np.array(mass)
        self.comloc = np.array(comloc)
        self.baseaxes = np.eye(3)[[1, 0, 2]]
        self.baseK = [skew(x) for x in self.baseaxes]
        self.orientation_scale = 0.20
        # Regularizes the null space about one shared crouched posture. The same
        # weights/seed are used for every grasp and command, including validation.
        self.prior = np.full(33, 0.008)
        self.prior[:3] = 0.035

    def _kin(self, q):
        """FK, geometric pose Jacobians, sole and wrist poses, and mass-weighted CoM.

        Complex batches support accurate objective Hessians through the same FK.
        """
        q = np.atleast_2d(q)
        b = len(q)
        typ = q.dtype
        R = np.broadcast_to(np.eye(3, dtype=typ), (b, 3, 3)).copy()
        axes = np.zeros((b, 33, 3), dtype=typ)
        origins = np.zeros_like(axes)
        origins[:, 3:6] = q[:, None, :3]
        for i, (a, k) in enumerate(zip(self.baseaxes, self.baseK)):
            t = q[:, i + 3, None, None]
            axes[:, i + 3] = R @ a
            R = R @ (np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k))
        ps = [q[:, :3]]
        rs = [R]
        for parent, j, offset, r, axis, k, k2 in self.joints:
            rot = rs[parent] @ r
            p = ps[parent] + rs[parent] @ offset
            if j >= 0:
                origins[:, j] = p
                axes[:, j] = rot @ axis
                t = q[:, j, None, None]
                rot = rot @ (np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k2)
            ps.append(p)
            rs.append(rot)
        ps = np.stack(ps, 1)
        rs = np.stack(rs, 1)
        er = rs[:, self.endidx]
        ep = ps[:, self.endidx] + (er @ self.endoffset[None, :, :, None])[..., 0]
        w = axes[:, None, :, :] * self.mask[None]
        jp = np.cross(w, ep[:, :, None, :] - origins[:, None, :, :])
        jp[:, :, :3, :] = np.eye(3)
        jr = np.cross(w[:, :, :, None, :], er.swapaxes(-1, -2)[:, :, None, :, :])
        # For each end link, row order is xyz followed by the 3 columns of R.
        pose = np.concatenate([ep, self.orientation_scale * er.swapaxes(-1, -2).reshape(b, 3, 9)], -1).reshape(b, -1)
        jac = (
            np.concatenate([jp, self.orientation_scale * jr.reshape(b, 3, 33, 9)], -1)
            .transpose(0, 1, 3, 2)
            .reshape(b, -1, 33)
        )
        cent = ps + (rs @ self.comloc[None, :, :, None])[..., 0]
        com = (cent * self.mass[None, :, None]).sum(1) / self.mass.sum()
        return pose, jac, ep, er, com

    def _targets(self, soles, wrist):
        s = np.asarray(soles).reshape(2, 3)
        ep = np.concatenate([np.c_[s[:, :2], np.zeros(2)], wrist[None, :3, 3]], axis=0)
        r = np.zeros((3, 3, 3), dtype=np.result_type(s, wrist))
        for i, (_, _, yaw) in enumerate(s):
            c = np.cos(yaw)
            si = np.sin(yaw)
            r[i] = [[c, -si, 0], [si, c, 0], [0, 0, 1]]
        r[2] = wrist[:3, :3]
        return np.concatenate([ep, self.orientation_scale * r.transpose(0, 2, 1).reshape(3, 9)], 1).ravel()

    def _solve(self, soles, wrist, seed):
        target = self._targets(soles, wrist)
        cache = [None, None, None]

        def both(x):
            if cache[0] is None or not np.array_equal(x, cache[0]):
                pose, j, *_ = self._kin(x)
                cache[:] = [
                    x.copy(),
                    np.r_[pose[0] - target, self.prior * (x - seed)],
                    np.vstack([j[0], np.diag(self.prior)]),
                ]
            return cache[1:]

        ans = least_squares(
            lambda x: both(x)[0],
            np.clip(seed, self.lo + 1e-9, self.hi - 1e-9),
            jac=lambda x: both(x)[1],
            bounds=(self.lo, self.hi),
            x_scale="jac",
            xtol=2e-11,
            ftol=2e-12,
            gtol=3e-12,
            max_nfev=150,
        )
        return ans, target

    def wrist_of(self, q):
        _, _, p, r, _ = self._kin(q)
        t = np.eye(4)
        t[:3, :3] = r[0, 2]
        t[:3, 3] = p[0, 2]
        return t

    def corners(self, s):
        local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * self.foot_box / 2
        cos = np.cos(s[:, 2])
        sin = np.sin(s[:, 2])
        R = np.stack([cos, -sin, sin, cos], axis=1).reshape(2, 2, 2)
        return ((local @ R.transpose(0, 2, 1)) + s[:, None, :2]).reshape(8, 2)

    def evaluate(self, soles, wrist, seed=None):
        """Return a local implicit-gradient margin. q and com are differentiable too."""
        if seed is None:
            seed = self.home
        device = soles.device
        s = soles.detach().cpu().double().numpy()
        w = np.asarray(wrist, dtype=float)
        ans, target = self._solve(s, w, np.asarray(seed))
        q0 = ans.x
        pose, j, _ep, _er, cm = self._kin(q0)
        # Fix the bounds with an active KKT multiplier. Complex-step derivatives
        # differentiate the mathematical gradient J^T r + D^2 (q-q_ref) and
        # retain residual-curvature terms omitted in a Gauss-Newton Hessian.
        # Threshold is 0.5 micrometers for translation or 0.5 microradians for rotation.
        free = np.minimum(q0 - self.lo, self.hi - q0) > 5e-7
        eps = 1e-25
        qc = q0.astype(complex) + 1j * eps * np.eye(len(q0))
        pc, jc, _, _, comc = self._kin(qc)
        gc = np.einsum("bri,br->bi", jc, pc - target) + self.prior**2 * (qc - seed)
        h = (np.imag(gc) / eps).T
        target_jac = np.array(
            [self._targets((s.ravel() + 1j * eps * d).reshape(2, 3), w).imag / eps for d in np.eye(6)]
        ).T
        dq = np.zeros((33, 6))
        dq[free] = np.linalg.solve(h[np.ix_(free, free)], j[0][:, free].T @ target_jac)

        def as_tensor(value):
            return torch.as_tensor(value, device=device, dtype=torch.float64)

        # Attach the local first-order IK solution to the external autograd graph.
        # A changed command re-solves the nonlinear problem before scoring.
        ds = (soles.double() - soles.double().detach()).reshape(-1)
        q = as_tensor(q0) + as_tensor(dq) @ ds
        com = as_tensor(cm[0]) + as_tensor(comc.imag.T / eps) @ (q - as_tensor(q0))
        corners = (
            torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], device=device, dtype=torch.float64)
            * as_tensor(self.foot_box)
            / 2
        )
        phi = soles[:, 2].double()
        co, si = phi.cos(), phi.sin()
        rotation = torch.stack([co, -si, si, co], -1).reshape(2, 2, 2)
        vertices = (corners @ rotation.transpose(-1, -2) + soles[:, None, :2].double()).reshape(8, 2)
        ids = ConvexHull(self.corners(s)).vertices.copy()
        polygon = vertices[ids]
        edge = polygon.roll(-1, 0) - polygon
        normal = torch.stack([-edge[:, 1], edge[:, 0]], -1) / edge.norm(dim=-1, keepdim=True)
        margins = ((com[:2] - polygon) * normal).sum(-1)
        frozen_com_margins = ((com[:2].detach() - polygon) * normal).sum(-1)
        e = (pose[0] - target).reshape(3, 12)
        rot_error = 2 * np.arcsin(
            np.minimum(1, np.linalg.norm(e[:, 3:], axis=1) / (self.orientation_scale * np.sqrt(8)))
        )
        pos_error = np.linalg.norm(e[:, :3], axis=1)
        return {
            "margin": margins.min(),
            "hull_only_margin": frozen_com_margins.min(),
            "q": q,
            "com": com,
            "edge": int(margins.detach().argmin()),
            "wrist_pos_err_m": float(pos_error[2]),
            "wrist_rot_err_rad": float(rot_error[2]),
            "sole_pos_err_m": float(pos_error[:2].max()),
            "sole_rot_err_rad": float(rot_error[:2].max()),
            "optimality": float(ans.optimality),
            "solved": ans.success,
            "iterations": ans.nfev,
            "active": int((~free).sum()),
        }

    def fixture(self, soles):
        """Call ONCE at shared nominal achieved stance. Returns fixed trial inputs.

        The crouch seed is a cuTAMP FK-map witness. Restore its world origin at
        the observed sole midpoint. Both wrist targets share a rotation and
        the incoming stance; their gripper centers differ by 4.3 cm on one
        fixed object frame. Grasp/contact geometry is a prescribed fixture.
        """
        if torch.is_tensor(soles):
            soles = soles.detach().cpu().double().numpy()
        ranking = json.loads((self.root / "data/t1-reachability/floor-forward/ranking.json").read_text())
        q = np.array(ranking["combined"][0]["q"])
        feet = self._kin(q)[2][0, :2, :2]
        q[:2] += np.asarray(soles)[:, :2].mean(0) - feet.mean(0)
        wrist = self.wrist_of(q)
        grasps = {}
        for name, dx in (("near", 0.170), ("far", 0.213)):
            grasps[name] = wrist.copy()
            grasps[name][0, 3] += dx
        sweep = json.loads((self.root / "cutamp/robots/assets/t1_description/graspgenx/sweep_volume.json").read_text())
        center = (np.array(sweep["base_from_G"]) @ np.r_[sweep["offset_mid"], 1.0])[:3]
        world_object = np.eye(4)
        world_object[:3, 3] = np.mean([T[:3, :3] @ center + T[:3, 3] for T in grasps.values()], 0)
        return {
            "seed": q,
            "near": grasps["near"],
            "far": grasps["far"],
            "world_object": world_object,
            "object_to_wrist": {k: np.linalg.inv(world_object) @ T for k, T in grasps.items()},
        }
