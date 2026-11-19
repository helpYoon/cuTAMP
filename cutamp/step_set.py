"""The kinematic set of a step: which relative stances the legs can stand in and swing to (design 5.5).

A relative stance ``c = (x, y, yaw)`` is the swung (left) sole's pose in the planted (right) sole's frame; a right swing
reads the mirror. The set is computed from the planner's own model (``REAL_LEGS``) on a grid and stored as a signed
distance (metres, negative inside, yaw counted at ``RHO`` per radian). Every robot fact it uses is read from the model's
files: the sole box and the foot box from ``t1_planner.urdf``, the home width and the mid-sole from ``t1_29dof.yml``,
the limits from cuRobo. :func:`step_set_fingerprint` hashes those files; :func:`load_step_set` refuses an asset built
from other ones.

Method (ported from the study's ``klib.py``, ``s02_stance_set.py``, ``s06_reach.py``). A damped Gauss-Newton on weighted
rows with an active set on the joint limits solves a stance: both soles on their targets, the centre of mass inside the
support region (:func:`~cutamp.com_polygon_cost._region_halfplanes`, keyframe), no overlap of the legs, foot boxes and
floor; small-weight rows choose among the solutions. A cell is in a set only when a configuration found passes
:func:`_check`, which recomputes every condition from forward kinematics (an existence proof). The sets: ``planner``
(the full test) and ``lift25`` (the left sole flat 2.5 cm up, no centre-of-mass test). The kinematic set K is the cells in
both and in the piece of the lifted set that holds home. :class:`ExactStance` is the exact test of a stance that is not
on the grid.
"""

import hashlib
import math
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import roma
import torch
from curobo.config_io import load_yaml
from curobo.kinematics import Kinematics
from curobo.types import JointState

from cutamp.com_polygon_cost import COM_DYNAMIC_RESERVE, COM_SAFETY_MARGIN, LOAD_GUARD_A
from cutamp.gn_refresh import _rotmat_to_rotvec
from cutamp.robots.envelope import ENVELOPES_DIR
from cutamp.robots.t1 import REAL_LEGS, T1Model, get_t1_kinematics, t1_curobo_cfg
from cutamp.stance import STEP_LIFT

DEFAULT_STEP_SET = Path(__file__).parent / "robots" / "assets" / "t1_29dof" / "step_set.npz"

#: m per rad: a yaw change counted as travel of the sole's toe; a robot fact (the sole's half length).
RHO = REAL_LEGS.sole_box[0] / 2
#: Tolerances of the exact test, tighter than the planner's sole tolerance (com_polygon_cost.SOLE_POS_TOL / ROT_TOL).
TOL_POS, TOL_ROT, TOL_PEN = 5e-4, 2e-3, 1e-6
#: Weights of the small rows, which choose among the solutions.
W_SOFT = dict(centre=0.05, upright=0.05, yaw=0.02, z=0.01, posture=0.003)


class StepSetMismatch(ValueError):
    """The asset was built from other model files than the ones in the tree."""


class StepSetGrid(NamedTuple):
    sdf: np.ndarray  # [nx, ny, nyaw] float32, metres, negative inside the kinematic set
    xs: np.ndarray  # [nx] m
    ys: np.ndarray  # [ny] m
    ps: np.ndarray  # [nyaw] rad
    rho: float  # m per rad


def step_set_fingerprint(model: T1Model = REAL_LEGS) -> str:
    """sha256 over the bytes of every file the model's cuRobo config names (``urdf_path``, ``collision_spheres``),
    ``t1_29dof.yml`` and every file of ``assets/envelopes/``."""
    h = hashlib.sha256()
    kin = load_yaml(str(model.cfg_path))["robot_cfg"]["kinematics"]
    files = [model.asset_root / kin["urdf_path"], model.asset_root / kin["collision_spheres"], model.cfg_path] + sorted(
        p for p in ENVELOPES_DIR.iterdir() if p.is_file()
    )
    for p in files:
        h.update(p.read_bytes())
    return h.hexdigest()


def load_step_set(path: Path = DEFAULT_STEP_SET) -> StepSetGrid:
    z = np.load(path)
    stored = str(z["fingerprint"])
    if stored != step_set_fingerprint():
        raise StepSetMismatch(
            f"{path}: built from model files with fingerprint {stored}, but the files in the tree "
            f"have {step_set_fingerprint()}; rebuild it with cutamp/scripts/build_step_set.py"
        )
    return StepSetGrid(z["sdf"], z["xs"], z["ys"], z["ps"], float(z["rho"]))


@lru_cache(maxsize=4)
def _facts(model: T1Model) -> dict:
    """The robot facts the set needs, read from the model's files."""
    block = load_yaml(str(model.cfg_path))["cutamp"]
    box = None
    for link in ET.parse(str(model.asset_root / "t1_planner.urdf")).getroot().findall("link"):
        if link.get("name") == "left_foot_link":
            box = [float(v) for v in link.find("collision").find("geometry").find("box").get("size").split()]
    return dict(
        sole_box=tuple(model.sole_box),
        foot_box=tuple(box),
        home_width=2.0 * float(block["sole_y"]),
        mid_x=float(block["home_mid_sole"][0]),
    )


@contextmanager
def _strict_fp32():
    """The solver's float32 FK runs without TF32 (the study's setting); restored on exit, so a process is unchanged."""
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


# ------------------------------------------------------------------------------------------------------ geometry
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _stance_world(c, mid_x, lift=0.0):
    """Relative stance c [..., 3] -> world poses (x, y, z, yaw) of the left and right soles, the stance frame at the home
    mid-sole, heading 0."""
    c = np.asarray(c, float)
    x, y, psi = c[..., 0], c[..., 1], c[..., 2]
    cs, sn = np.cos(-psi / 2), np.sin(-psi / 2)
    dx, dy = cs * x - sn * y, sn * x + cs * y
    z = np.zeros_like(x)
    return (
        np.stack([mid_x + dx / 2, dy / 2, z + lift, psi / 2], -1),
        np.stack([mid_x - dx / 2, -dy / 2, z, -psi / 2], -1),
    )


def _pose_mat(p, dev, dtype=torch.float32):
    """[N, 4] (x, y, z, yaw) -> [N, 4, 4] flat sole poses."""
    p = torch.as_tensor(p, dtype=torch.float64)
    T = torch.eye(4, dtype=torch.float64).repeat(p.shape[0], 1, 1)
    c, s = torch.cos(p[:, 3]), torch.sin(p[:, 3])
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = c, -s, s, c
    T[:, :3, 3] = p[:, :3]
    return T.to(dev, dtype)


def _rect_gap(c, sole_box):
    """Signed gap [m] between the two sole rectangles of relative stances c [N, 3]: positive apart, negative overlapping.
    Separating-axis test on the 4 edge directions; exact for the overlap depth, and for the distance unless the closest
    features are two corners (then a lower bound)."""
    c = np.atleast_2d(np.asarray(c, float))
    N = len(c)
    hl, hw = sole_box[0] / 2, sole_box[1] / 2
    loc = np.array([[sx * hl, sy * hw] for sx in (-1, 1) for sy in (-1, 1)])
    cs, sn = np.cos(c[:, 2]), np.sin(c[:, 2])
    Rl = np.stack([np.stack([cs, -sn], -1), np.stack([sn, cs], -1)], -2)
    A = np.broadcast_to(loc, (N, 4, 2))
    B = np.einsum("nij,kj->nki", Rl, loc) + c[:, None, :2]
    axes = np.concatenate([np.broadcast_to(np.eye(2), (N, 2, 2)), Rl.transpose(0, 2, 1)], 1)
    pa, pb = np.einsum("nki,nai->nak", A, axes), np.einsum("nki,nai->nak", B, axes)
    best = np.maximum(pb.min(-1) - pa.max(-1), pa.min(-1) - pb.max(-1)).max(-1)
    out = best.copy()
    idx = np.flatnonzero(best > 0)
    if len(idx):
        out[idx] = _poly_dist(A[idx], B[idx])
    return out


def _poly_dist(A, B):
    """Exact distance between disjoint rectangles A, B [M, 4, 2] (corner order (-,-), (-,+), (+,-), (+,+))."""
    order = [0, 1, 3, 2]
    A, B = A[:, order], B[:, order]

    def pt_seg(P, Q):
        d = np.roll(Q, -1, axis=1) - Q
        w = P[:, :, None, :] - Q[:, None, :, :]
        t = np.clip((w * d[:, None]).sum(-1) / (d * d).sum(-1)[:, None], 0, 1)
        cl = Q[:, None] + t[..., None] * d[:, None]
        return np.linalg.norm(P[:, :, None, :] - cl, axis=-1).min((1, 2))

    return np.minimum(pt_seg(A, B), pt_seg(B, A))


# ------------------------------------------------------------------------------------------------------ the solver
class _Legs:
    """The planner's model with Jacobians: the solver's state. Arms and Waist stay at the model's home; the 6 base joints
    and the leg joints are solved."""

    def __init__(self, model: T1Model, kinematics: Kinematics):
        self.model, self.kin = model, kinematics
        self.facts = _facts(model)
        d = self.dev = kinematics.device_cfg.device
        self.names = list(kinematics.joint_names)
        self.dof = len(self.names)
        self.home = torch.tensor(model.home, dtype=torch.float32, device=d)
        self.lim = kinematics.get_joint_limits().position.clone()  # [2, dof]: the planner's limits
        hard = {
            j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
            for j in ET.parse(str(model.asset_root / "t1_planner.urdf")).getroot().findall("joint")
            if j.find("limit") is not None
        }
        self.lim_hard = self.lim.clone()  # the URDF's hard limits, where it has them
        for i, n in enumerate(self.names):
            if n in hard:
                self.lim_hard[0, i], self.lim_hard[1, i] = hard[n]
        self.free = [
            i
            for i, n in enumerate(self.names)
            if n.startswith("base_j_") or any(k in n for k in ("Hip", "Knee", "Ankle"))
        ]
        self.free_t = torch.tensor(self.free, device=d)
        self.i = {n: i for i, n in enumerate(self.names)}
        tf = list(kinematics.tool_frames)
        self.sole_idx = tuple(tf.index(f) for f in model.sole_frames)
        kp = kinematics.kinematics_config
        links = {}
        for link in t1_curobo_cfg(model)["robot_cfg"]["kinematics"]["collision_link_names"]:
            try:
                links[link] = kp.get_sphere_index_from_link_name(link).to(d).long()
            except Exception:
                pass
        sc = kinematics.get_self_collision_config()
        rad = self.fk(self.home.unsqueeze(0)).robot_spheres.reshape(-1, 4)[:, 3] + sc.sphere_padding.to(d)
        pairs = sc.collision_pairs.long().to(d)
        pairs = pairs[(rad[pairs] >= 0).all(-1)]
        leg = torch.zeros(rad.shape[0], dtype=torch.bool, device=d)
        for link in ("Hip_Yaw_Left", "Shank_Left", "Hip_Yaw_Right", "Shank_Right"):
            leg[links[link]] = True
        arm = torch.zeros_like(leg)
        for link, idx in links.items():
            if link.startswith(("AL", "AR", "left_", "right_")):
                arm[idx] = True
        keep = leg[pairs].any(-1)  # pairs with a leg sphere (the others never move here)
        self.pairs = pairs[keep]
        self.radius = rad
        self.left_leg = torch.cat([links["Hip_Yaw_Left"], links["Shank_Left"]])
        self.right_leg = torch.cat([links["Hip_Yaw_Right"], links["Shank_Right"]])
        fx, fy, fz = self.facts["foot_box"]  # the foot's collision box, its bottom face on the sole plane
        self.box_c = torch.tensor([0.0, 0.0, fz / 2], device=d)
        self.box_h = torch.tensor([fx / 2, fy / 2, fz / 2], device=d)

    def fk(self, q):
        return self.kin.compute_kinematics(JointState.from_position(q.contiguous(), joint_names=self.names))

    def penetration(self, spheres, TL, TR):
        """Summed penetration depth [N] (m): the self-collision sphere pairs with a leg sphere, each foot's collision box
        against the other leg's spheres, and every sphere against the floor z = 0."""
        c, r = spheres[..., :3], spheres[..., 3]
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        total = torch.relu(r[:, i] + r[:, j] - (c[:, i] - c[:, j]).norm(dim=-1)).sum(-1)
        for T, legs in ((TL, self.right_leg), (TR, self.left_leg)):
            loc = torch.einsum("nji,nkj->nki", T[:, :3, :3], c[:, legs] - T[:, None, :3, 3])
            qd = (loc - self.box_c).abs() - self.box_h
            dist = torch.clamp(qd, min=0).norm(dim=-1) + torch.clamp(qd.max(-1).values, max=0)
            total = total + torch.relu(r[:, legs] - dist).sum(-1)
        return total + torch.relu(r - c[..., 2]).sum(-1)

    def seeds(self, N, kind):
        """Start configurations [N, dof]: home, or squats of increasing depth (legs folded, trunk lowered)."""
        q = self.home.repeat(N, 1).clone()
        depth = {"home": None, "squat1": (0.6, 0.57), "squat2": (1.0, 0.46), "squat3": (1.4, 0.36)}[kind]
        if depth is not None:
            knee, z = depth
            for s in ("Left", "Right"):
                q[:, self.i[f"{s}_Knee_Pitch"]] = knee
                q[:, self.i[f"{s}_Hip_Pitch"]] = -knee / 2
                q[:, self.i[f"{s}_Ankle_Pitch"]] = -knee / 2
            q[:, self.i["base_j_z"]] = z
        return q

    def regions(self, L, R, planes: int = 32):
        """Keyframe support planes (n [N, planes, 2], b [N, planes]) of the stances whose soles are L, R [N, 4]
        (x, y, z, yaw), padded to ``planes`` rows by repeating the first: the facets of the region the planner's confs are
        held to (the study's construction, which :func:`~cutamp.com_polygon_cost._region_halfplanes` shares its zero set
        with): the hull of the load-guard points S_a and the hull of both soles, shrunk by COM_SAFETY_MARGIN, then every
        plane by COM_DYNAMIC_RESERVE."""
        N = len(L)
        n_out, b_out = np.zeros((N, planes, 2)), np.zeros((N, planes))
        sole = self.facts["sole_box"]
        loc = np.array([[sx * sole[0] / 2, sy * sole[1] / 2] for sx in (-1, 1) for sy in (-1, 1)])

        def rect(x, y, th):
            c, s = math.cos(th), math.sin(th)
            return loc @ np.array([[c, s], [-s, c]]) + np.array([x, y])

        a = LOAD_GUARD_A
        for k in range(N):
            FA, FB = rect(L[k, 0], L[k, 1], L[k, 3]), rect(R[k, 0], R[k, 1], R[k, 3])
            Sa = np.array([lam * p + (1 - lam) * q for lam in (a, 1 - a) for p in FA for q in FB])
            n1, b1 = _hull_halfplanes(Sa)
            n2, b2 = _hull_halfplanes(np.vstack([FA, FB]))
            n, b = np.vstack([n1, n2]), np.concatenate([b1, b2 - COM_SAFETY_MARGIN]) - COM_DYNAMIC_RESERVE
            if len(n) > planes:
                raise ValueError(len(n))
            pad = planes - len(n)
            n_out[k], b_out[k] = np.vstack([n, np.repeat(n[:1], pad, 0)]), np.concatenate([b, np.repeat(b[:1], pad)])
        return torch.tensor(n_out, dtype=torch.float32, device=self.dev), torch.tensor(
            b_out, dtype=torch.float32, device=self.dev
        )


def _hull_halfplanes(pts):
    """Outward unit normals n [m, 2] and offsets b [m] of the convex hull of ``pts`` (monotone chain): inside is n . x <= b."""
    P = sorted(set(map(tuple, np.round(np.asarray(pts, float), 12))))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 1e-15:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 1e-15:
            up.pop()
        up.append(p)
    H = np.array(lo[:-1] + up[:-1])  # counter-clockwise
    E = np.roll(H, -1, axis=0) - H
    n = np.stack([E[:, 1], -E[:, 0]], 1)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return n, (n * H).sum(1)


def _solve(
    rb,
    q0,
    TL,
    TR,
    hp_n=None,
    hp_b=None,
    iters=60,
    use_com=True,
    use_coll=True,
    lim=None,
    polish=12,
    com_in=0.003,
    coll_margin=0.003,
    step_clamp=0.2,
    damping=1e-5,
):
    """Damped Gauss-Newton with a joint-limit active set, from q0 [N, dof]. TL, TR [N, 4, 4] sole targets; hp_n [N, P, 2],
    hp_b [N, P] the support region (world frame). Rows of weight 1: both soles on their targets, the two most violated
    support edges, no overlap. The small rows choose among the solutions; the last ``polish`` iterations run them at a
    hundredth, so the hard rows close. Returns q [N, dof]."""
    d, N, f64 = rb.dev, q0.shape[0], torch.float64
    lo, hi = rb.lim if lim is None else lim
    F, nF = rb.free_t, len(rb.free)
    lo_f, hi_f = lo[F].to(f64), hi[F].to(f64)
    q = q0.clone()
    q[:, F] = torch.minimum(torch.maximum(q[:, F], lo[F]), hi[F])
    ipitch, iroll, iyaw, iz = (
        rb.free.index(rb.i[n]) for n in ("base_j_pitch", "base_j_roll", "base_j_yaw", "base_j_z")
    )
    leg_cols = [k for k, i in enumerate(rb.free) if not rb.names[i].startswith("base_j_")]
    home_f = rb.home[F].to(f64)
    z_ref = float(rb.home[rb.i["base_j_z"]])
    eyeF = torch.eye(nF, device=d, dtype=f64)
    for it in range(iters):
        sc = 1.0 if it < iters - polish else 0.01
        qg = q.detach().requires_grad_(True)
        with torch.enable_grad():
            st = rb.fk(qg)
            com = st.robot_com[..., :3].reshape(N, 3)[:, :2]
            jcx = torch.autograd.grad(com[:, 0].sum(), qg, retain_graph=True)[0].clone()
            jcy = torch.autograd.grad(com[:, 1].sum(), qg, retain_graph=use_coll)[0].clone()
            if use_coll:
                pen = rb.penetration(st.robot_spheres.reshape(N, -1, 4), TL, TR)
                jpen = torch.autograd.grad(pen.sum(), qg)[0].clone()
        with torch.no_grad():
            com = com.detach().to(f64)
            jc = torch.stack([jcx, jcy], 1)[:, :, F].to(f64)
            pos = torch.stack([st.tool_poses.position[:, 0, i] for i in rb.sole_idx], 1)
            quat = torch.stack([st.tool_poses.quaternion[:, 0, i] for i in rb.sole_idx], 1)
            rot = roma.unitquat_to_rotmat(quat.roll(-1, dims=-1))
            goal = torch.stack([TL, TR], 1)
            r_sole = (
                torch.cat([goal[..., :3, 3] - pos, _rotmat_to_rotvec(goal[..., :3, :3] @ rot.transpose(-1, -2))], -1)
                .reshape(N, 12)
                .to(f64)
            )
            jacs = st.tool_jacobians[:, 0]
            j_sole = torch.stack([jacs[:, i] for i in rb.sole_idx], 1).reshape(N, 12, -1)[:, :, F].to(f64)
            rows_J, rows_r = [j_sole], [r_sole]
            if use_com and hp_n is not None:
                hn = hp_n.to(f64)
                viol = torch.einsum("npk,nk->np", hn, com) - hp_b.to(f64)  # > 0 outside
                # the two most violated distinct edges (a second plane with the first's normal is masked)
                v1, k1 = viol.max(-1)
                n1 = hn.gather(1, k1.view(N, 1, 1).expand(N, 1, 2)).squeeze(1)
                same = (hn * n1.unsqueeze(1)).sum(-1) > 1 - 1e-9
                v2, k2 = viol.masked_fill(same, -1e9).max(-1)
                n2 = hn.gather(1, k2.view(N, 1, 1).expand(N, 1, 2)).squeeze(1)
                for v, n in ((v1, n1), (v2, n2)):
                    act = (v > -com_in).to(f64)
                    rows_J.append((torch.einsum("nk,nkf->nf", n, jc) * act.unsqueeze(-1)).unsqueeze(1))
                    rows_r.append(((-com_in - v) * act).unsqueeze(-1))
            if use_coll:
                act = (pen > 0).to(f64)
                rows_J.append((jpen[:, F].to(f64) * act.unsqueeze(-1)).unsqueeze(1))
                rows_r.append(((-coll_margin - pen.to(f64)) * act).unsqueeze(-1))
            qf = q[:, F].to(f64)
            cxy = torch.zeros(N, 2, device=d, dtype=f64)
            cxy[:, 0] = rb.facts["mid_x"]
            rows_J.append(jc * (sc * W_SOFT["centre"]))
            rows_r.append((cxy - com) * (sc * W_SOFT["centre"]))
            for col, tgt, w in (
                (ipitch, 0.0, W_SOFT["upright"]),
                (iroll, 0.0, W_SOFT["upright"]),
                (iyaw, 0.0, W_SOFT["yaw"]),
                (iz, z_ref, W_SOFT["z"]),
            ):
                J = torch.zeros(N, 1, nF, device=d, dtype=f64)
                J[:, 0, col] = sc * w
                rows_J.append(J)
                rows_r.append(((tgt - qf[:, col]) * sc * w).unsqueeze(-1))
            Jp = torch.zeros(N, len(leg_cols), nF, device=d, dtype=f64)
            for k, col in enumerate(leg_cols):
                Jp[:, k, col] = sc * W_SOFT["posture"]
            rows_J.append(Jp)
            rows_r.append((home_f[leg_cols] - qf[:, leg_cols]) * sc * W_SOFT["posture"])
            J, r = torch.cat(rows_J, 1), torch.cat(rows_r, 1)
            lam = damping + 0.1 * (r_sole * r_sole).sum(-1)
            # active set on the joint limits: a joint whose step would cross a limit moves exactly onto it and its
            # column leaves the solve
            free = torch.ones(N, nF, device=d, dtype=f64)
            fixed = torch.zeros(N, nF, device=d, dtype=f64)
            for _ in range(3):
                Jm = J * free.unsqueeze(1)
                rr = r - torch.einsum("nrf,nf->nr", J, fixed)
                H = Jm.transpose(1, 2) @ Jm + lam.view(N, 1, 1) * eyeF + torch.diag_embed(1.0 - free)
                g = torch.einsum("nrf,nr->nf", Jm, rr)
                dq = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1) * free
                step = dq + fixed
                s = (step_clamp / step.abs().amax(-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
                step = step * s
                new = qf + step
                over, under = new > hi_f, new < lo_f
                cross = (over | under) & (free > 0)
                fixed = torch.where(cross, torch.where(over, hi_f - qf, lo_f - qf), fixed)
                free = free * (~cross)
            new = torch.minimum(torch.maximum(qf + step, lo_f), hi_f)
            q = q.clone()
            q[:, F] = new.to(q.dtype)
    return q.detach()


@torch.no_grad()
def _check(rb, q, TL, TR, hp_n=None, hp_b=None, use_coll=True, lim=None):
    """The exact test of one configuration per row, from forward kinematics: ``(soles, limits, com, collision)`` ok [N]."""
    N = q.shape[0]
    lo, hi = rb.lim if lim is None else lim
    st = rb.fk(q)
    pos = torch.stack([st.tool_poses.position[:, 0, i] for i in rb.sole_idx], 1)
    quat = torch.stack([st.tool_poses.quaternion[:, 0, i] for i in rb.sole_idx], 1)
    rot = roma.unitquat_to_rotmat(quat.roll(-1, dims=-1))
    goal = torch.stack([TL, TR], 1)
    e_pos = (goal[..., :3, 3] - pos).norm(dim=-1).amax(-1)
    e_rot = _rotmat_to_rotvec(goal[..., :3, :3] @ rot.transpose(-1, -2)).norm(dim=-1).amax(-1)
    F = rb.free_t
    lim_ok = torch.maximum(q[:, F] - hi[F], lo[F] - q[:, F]).amax(-1) <= 1e-6
    com = st.robot_com[..., :3].reshape(N, 3)
    if hp_n is not None:
        com_ok = (torch.einsum("npk,nk->np", hp_n.double(), com[:, :2].double()) - hp_b.double()).amax(-1) <= 0
    else:
        com_ok = torch.ones(N, dtype=torch.bool, device=q.device)
    coll_ok = (
        rb.penetration(st.robot_spheres.reshape(N, -1, 4), TL, TR) <= TOL_PEN
        if use_coll
        else torch.ones(N, dtype=torch.bool, device=q.device)
    )
    return (e_pos <= TOL_POS) & (e_rot <= TOL_ROT), lim_ok, com_ok, coll_ok


def _try(rb, cells, q0, use_com, lift, iters=50, use_coll=True, lim=None):
    """Solve ``cells`` [N, 3] numpy from ``q0`` [N, dof]; returns (pass [N] bool tensor, q [N, dof])."""
    d = rb.dev
    L, R = _stance_world(cells, rb.facts["mid_x"], lift=lift)
    TL, TR = _pose_mat(L, d), _pose_mat(R, d)
    hn, hb = rb.regions(L, R) if use_com else (None, None)
    q = _solve(rb, q0, TL, TR, hn, hb, iters=iters, use_com=use_com, use_coll=use_coll, lim=lim)
    soles, lim_ok, com_ok, coll_ok = _check(rb, q, TL, TR, hn, hb, use_coll=use_coll, lim=lim)
    return soles & lim_ok & coll_ok & (com_ok if use_com else True), q


_STARTS = ("home", "squat1", "squat2", "squat3")
_OFFS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _grid():
    xs = np.round(np.arange(-0.80, 0.8001, 0.02), 4)
    ys = np.round(np.arange(-0.30, 0.9001, 0.02), 4)
    ps = np.radians(np.arange(-120, 120.01, 5.0))
    return xs, ys, ps


def _search_set(
    rb, cells, shape, use_com, lift, iters=40, chunk=30000, log=print, use_coll=True, lim=None, domain=None
):
    """The cells of ``cells`` [M, 3] (the grid, flattened) that pass, from the 4 fixed starts and then by continuation: a
    failing cell next to a passing one is re-solved from that neighbour's solution, in each of the 6 grid directions,
    until nothing changes. Cells whose sole rectangles overlap are never searched."""
    gap = _rect_gap(cells, rb.facts["sole_box"])
    dom_idx = np.flatnonzero((gap > 0) if domain is None else ((gap > 0) & domain.ravel()))
    pos = -np.ones(len(cells), np.int64)
    pos[dom_idx] = np.arange(len(dom_idx))
    M = len(dom_idx)
    dc = cells[dom_idx]
    ok = np.zeros(M, bool)
    Q = np.zeros((M, rb.dof), np.float32)

    def attempt(rows, q0):
        res_ok, res_q = [], []
        for a in range(0, len(rows), chunk):
            good, q = _try(
                rb,
                dc[rows[a : a + chunk]],
                torch.as_tensor(q0[a : a + chunk], device=rb.dev),
                use_com,
                lift,
                iters,
                use_coll,
                lim,
            )
            res_ok.append(good.cpu().numpy())
            res_q.append(q.cpu().numpy())
        return np.concatenate(res_ok), np.concatenate(res_q)

    t0 = time.time()
    for kind in _STARTS:
        rows = np.flatnonzero(~ok)
        if not len(rows):
            break
        good, q = attempt(rows, rb.seeds(len(rows), kind).cpu().numpy())
        ok[rows[good]], Q[rows[good]] = True, q[good]
        log(
            f"    start {kind}: tried {len(rows)}, passed {int(good.sum())}, total {int(ok.sum())} ({time.time() - t0:.0f} s)"
        )
    ijk = np.stack(np.unravel_index(dom_idx, shape), 1)
    tried = np.zeros((M, 6), bool)
    for rnd in range(200):
        new = 0
        for di, off in enumerate(_OFFS):
            nb = ijk + np.array(off)
            inside = ((nb >= 0) & (nb < np.array(shape))).all(1)
            nbpos = np.where(inside, pos[np.ravel_multi_index(np.where(inside[:, None], nb, 0).T, shape)], -1)
            cand = (~ok) & (~tried[:, di]) & (nbpos >= 0)
            cand[cand] &= ok[nbpos[cand]]
            rows = np.flatnonzero(cand)
            if not len(rows):
                continue
            tried[rows, di] = True
            good, q = attempt(rows, Q[nbpos[rows]])
            ok[rows[good]], Q[rows[good]] = True, q[good]
            new += int(good.sum())
        log(f"    continuation round {rnd}: +{new}, total {int(ok.sum())} ({time.time() - t0:.0f} s)")
        if new == 0:
            break
    full = np.zeros(len(cells), bool)
    full[dom_idx] = ok
    return full.reshape(shape)


def _signed_distance(kin, samp):
    """Metres inside (negative) or outside (positive) the set, the zero level midway between an in-cell and an
    out-cell. The grid's outermost layer counts as outside: nothing beyond the grid was computed."""
    from scipy import ndimage

    kin = kin.copy()
    kin[0], kin[-1], kin[:, 0], kin[:, -1], kin[:, :, 0], kin[:, :, -1] = False, False, False, False, False, False
    return (
        ndimage.distance_transform_edt(~kin, sampling=samp) - ndimage.distance_transform_edt(kin, sampling=samp)
    ).astype(np.float32)


def build_step_set(out_path: Path, log=print) -> None:
    """Computes the kinematic set from ``REAL_LEGS`` and writes ``out_path`` (``sdf``, ``xs``, ``ys``, ``ps``, ``rho``,
    ``fingerprint``). The grid: x -0.80..0.80 m and y -0.30..0.90 m in 2 cm steps, yaw -120..120 deg in 5 deg steps."""
    from scipy import ndimage

    t0 = time.time()
    with _strict_fp32():
        kin0 = get_t1_kinematics(REAL_LEGS)
        rb = _Legs(REAL_LEGS, Kinematics(kin0.config, compute_com=True, compute_jacobian=True))
        xs, ys, ps = _grid()
        shape = (len(xs), len(ys), len(ps))
        X, Y, P = np.meshgrid(xs, ys, ps, indexing="ij")
        cells = np.stack([X.ravel(), Y.ravel(), P.ravel()], 1)
        sets = {}
        # the superset every other set is searched inside: the soles on their targets inside the URDF's hard limits
        log(f"[reach_hard] {len(cells)} cells")
        reach = _search_set(rb, cells, shape, False, 0.0, log=log, use_coll=False, lim=rb.lim_hard)
        log(f"[reach_hard] {int(reach.sum())} cells ({time.time() - t0:.0f} s)")
        for name, use_com, lift in (("planner", True, 0.0), ("lift25", False, STEP_LIFT)):
            log(f"[{name}] {len(cells)} cells")
            sets[name] = _search_set(rb, cells, shape, use_com, lift, log=log, domain=reach)
            log(f"[{name}] {int(sets[name].sum())} cells in the set ({time.time() - t0:.0f} s)")
    cell = tuple(int(np.argmin(np.abs(a - v))) for a, v in zip((xs, ys, ps), (0.0, rb.facts["home_width"], 0.0)))
    planner, lift25 = sets["planner"], sets["lift25"]
    if not (planner[cell] and lift25[cell]):
        raise RuntimeError("the home stance is not in the planner set and the lifted set")
    lab_a, _ = ndimage.label(planner)
    lab_l, _ = ndimage.label(lift25)
    kin = (lab_a == lab_a[cell]) & (lab_l == lab_l[cell])
    samp = (float(xs[1] - xs[0]), float(ys[1] - ys[0]), RHO * float(ps[1] - ps[0]))
    sdf = _signed_distance(kin, samp)
    log(
        f"K: {int(kin.sum())} cells; planner {int(planner.sum())}; lift25 {int(lift25.sum())}; home depth "
        f"{-float(sdf[cell]):.3f} m; {time.time() - t0:.0f} s"
    )
    np.savez_compressed(
        Path(out_path), sdf=sdf, xs=xs, ys=ys, ps=ps, rho=np.float64(RHO), fingerprint=np.array(step_set_fingerprint())
    )


class ExactStance:
    """The exact test of a stance ``c`` [N, 3] that need not be on the grid (the study's ``kexact``), on ``world``'s
    model. :meth:`stand` is the planner's test (soles, limits, centre of mass in the support region, no overlap);
    :meth:`lift` the same with the left sole flat ``STEP_LIFT`` up and no centre-of-mass test. Each first solves the
    nearest cell of the kinematic set from the 4 fixed starts, then solves ``c`` from that cell's solution and from the
    4 fixed starts; ``c`` passes when any of them passes ``_check``. Stances whose soles overlap fail."""

    def __init__(self, world, grid: StepSetGrid | None = None):
        from scipy import ndimage

        self.rb = _Legs(world.model, world.kinematics_jac)
        grid = load_step_set() if grid is None else grid
        self.grid = grid
        K = grid.sdf < 0
        _, ind = ndimage.distance_transform_edt(
            ~K, sampling=(0.02, 0.02, grid.rho * math.radians(5.0)), return_indices=True
        )
        X, Y, P = np.meshgrid(grid.xs, grid.ys, grid.ps, indexing="ij")
        self.near = np.stack([X[tuple(ind)], Y[tuple(ind)], P[tuple(ind)]], -1)  # [nx, ny, nyaw, 3]

    def _nearest_cell(self, c):
        g = self.grid
        i = [
            np.clip(np.round((c[:, k] - a[0]) / (a[1] - a[0])).astype(int), 0, len(a) - 1)
            for k, a in enumerate((g.xs, g.ys, g.ps))
        ]
        return self.near[i[0], i[1], i[2]]

    def _exact(self, c, use_com, lift):
        c = np.atleast_2d(np.asarray(c, float)).copy()
        c[:, 2] = _wrap(c[:, 2])
        out = np.zeros(len(c), bool)
        gap = _rect_gap(c, self.rb.facts["sole_box"]) > 0
        if not gap.any():
            return out
        c = c[gap]
        N = len(c)
        with _strict_fp32():
            cell = self._nearest_cell(c)
            q_cell = None
            for kind in _STARTS:
                good, q = _try(self.rb, cell, self.rb.seeds(N, kind), use_com, lift)
                if q_cell is None:
                    q_cell, found = q.clone(), good.clone()
                else:
                    q_cell[good & ~found], found = q[good & ~found], found | good
            starts = [self.rb.seeds(N, kind) for kind in _STARTS] + [q_cell]
            passed = torch.zeros(N, dtype=torch.bool, device=self.rb.dev)
            for q0 in starts:
                good, _ = _try(self.rb, c, q0, use_com, lift)
                passed |= good
        out[gap] = passed.cpu().numpy()
        return out

    def stand(self, c) -> np.ndarray:
        """Batched: ``c`` ``[K, 3]`` -> bool ``[K]``."""
        return self._exact(c, True, 0.0)

    def lift(self, c) -> np.ndarray:
        """Batched: ``c`` ``[K, 3]`` -> bool ``[K]``."""
        return self._exact(c, False, STEP_LIFT)
