"""The batched QP solver that moves the landings at every re-solve (design 5 and 6).

:func:`admm` solves one small convex QP per row of a batch by OSQP's ADMM iteration, with a fixed iteration count,
``inv_ex`` and no host sync, so a CUDA graph can hold it. :func:`composite_step` is the composite step of trust-region
SQP (Byrd and Omojokun; Nocedal and Wright 18.5) built from two such QPs: first the least violation the move limit
allows, then the least effort among the moves that reach it."""
from typing import Optional, Tuple

import torch
from torch import Tensor

ITERS, ADAPT, RHO0, SIGMA, ALPHA = 200, 25, 1.0, 1e-6, 1.6     # the tested settings
INVERSE_MAX = 16                                                # a CUDA graph cannot capture inv_ex above this
_ZERO_ROW = 1e-9   # a row (or a requirement's gradient) of smaller norm counts as zero: left unscaled, weight 1
TRAVEL_TOL = 1e-4  # m of the step bound's travel: what ITERS iterations leave a row or the box short of its bound


def _mv(M: Tensor, v: Tensor) -> Tensor:
    return (M @ v.unsqueeze(-1)).squeeze(-1)


def _mm(A: Tensor, B: Tensor) -> Tensor:
    """``A @ B`` without cuBLAS: cuBLAS runs some batched shapes in TF32 when the process allows it (the planner does),
    which cost the block inverse three digits (24 x 24: 4e-3 against 3e-6). Batched matrix-vector products kept full
    precision in every shape measured, so :func:`_mv` uses cuBLAS."""
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


def _inverse(K: Tensor) -> Tensor:
    """``[B, n, n]``: the inverse of each ``K``. ``inv_ex`` up to ``INVERSE_MAX``; above it the two-by-two block inverse
    over halves (through the Schur complement), recursively. ``K`` symmetric positive definite, so every block that is
    inverted is too."""
    n = K.shape[-1]
    if n <= INVERSE_MAX:
        return torch.linalg.inv_ex(K)[0]
    h = n // 2
    A, B, C, D = K[..., :h, :h], K[..., :h, h:], K[..., h:, :h], K[..., h:, h:]
    Ai = _inverse(A)
    AiB, CAi = _mm(Ai, B), _mm(C, Ai)
    Si = _inverse(D - _mm(C, AiB))
    top_right = -_mm(AiB, Si)
    return torch.cat([torch.cat([Ai - _mm(top_right, CAi), top_right], -1), torch.cat([-_mm(Si, CAi), Si], -1)], -2)


def admm(P: Tensor, q: Tensor, G: Tensor, lo: Tensor, hi: Tensor, soft: Optional[Tensor] = None,
         iters: int = ITERS) -> Tuple[Tensor, Tensor, Tensor]:
    """(z [B, n], lam [B, m], res [B, 2]) of  min 1/2 z'Pz + q'z + sum_j soft_j/2 * relu(G_j z - hi_j)^2
    s.t. lo <= Gz <= hi on the rows where soft is 0 (lo <= Gz only, where soft > 0). One problem per row of the batch.

    ``P`` ``[B, n, n]`` positive semi-definite, ``G`` ``[B, m, n]``, ``soft`` ``[B, m]`` at least 0 (``None``: every row
    hard); infinite bounds allowed. Each row of ``G`` is scaled to unit norm and the objective to unit size first.
    ``lam`` is the multiplier of the unscaled problem (positive on a row at its upper bound or above it on a soft row,
    negative at its lower bound); ``res`` the primal and dual residuals (infinity norms) of the scaled one."""
    B, _, n = G.shape
    eye = torch.eye(n, device=G.device, dtype=G.dtype)
    gn = G.norm(dim=-1)
    gn = torch.where(gn > _ZERO_ROW, gn, torch.ones_like(gn))                                # [B, m]
    G, lo, hi = G / gn.unsqueeze(-1), lo / gn, hi / gn
    size = torch.maximum(P.abs().amax(dim=(-1, -2)), q.abs().amax(-1))                        # [B]
    if soft is not None:
        soft = soft * gn.square()
        size = torch.maximum(size, soft.amax(-1))
    size = size.clamp_min(torch.finfo(G.dtype).tiny)  # all zero (no slope, no effort): nothing scales, z stays 0
    P, q = P / size[:, None, None], q / size[:, None]
    if soft is not None:
        soft = soft / size[:, None]
    Gt = G.mT
    GtG = _mm(Gt, G)  # rho (up to 1e4) multiplies it in K, so TF32's 3e-4 here outweighed P and moved the answer
    rho = torch.full((B, 1), RHO0, device=G.device, dtype=G.dtype)
    z, lam = torch.zeros_like(q), torch.zeros_like(lo)
    s = torch.minimum(torch.maximum(lam, lo), hi)
    for i in range(iters):
        if i % ADAPT == 0:
            if i:  # OSQP's residual balancing, on the residuals relative to their scales
                Gz, Pz, Gl = _mv(G, z), _mv(P, z), _mv(Gt, lam)
                rp = (Gz - s).abs().amax(-1) / torch.maximum(Gz.abs().amax(-1), s.abs().amax(-1)).clamp_min(1e-9)
                terms = [Pz, q, Gl]
                if soft is not None:  # a soft row's penalty is part of the objective, so its term is sized apart from
                    # the hard rows' (with P and q 0 the two cancel, and the relative dual residual would read 1)
                    Gls = _mv(Gt, torch.where(soft > 0, lam, torch.zeros_like(lam)))
                    terms = [Pz, q, Gls, Gl - Gls]
                rd_size = torch.stack([v.abs().amax(-1) for v in terms]).amax(0)
                rd = (Pz + q + Gl).abs().amax(-1) / rd_size.clamp_min(1e-9)
                rho = (rho * (rp / rd.clamp_min(1e-12)).sqrt().unsqueeze(-1)).clamp(1e-4, 1e4)
            Kinv = _inverse(P + SIGMA * eye + rho.unsqueeze(-1) * GtG)
        zt = _mv(Kinv, SIGMA * z - q + _mv(Gt, rho * s - lam))
        z = ALPHA * zt + (1 - ALPHA) * z
        t = ALPHA * _mv(G, zt) + (1 - ALPHA) * s + lam / rho
        s_hard = torch.minimum(torch.maximum(t, lo), hi)
        if soft is None:
            s = s_hard
        else:  # the soft row's prox: free below hi, pulled back toward hi above it; lo stays hard
            s_soft = torch.where(t <= hi, torch.maximum(t, lo), (rho * t + soft * hi) / (rho + soft))
            s = torch.where(soft > 0, s_soft, s_hard)
        lam = rho * (t - s)
    res = torch.stack([(_mv(G, z) - s).abs().amax(-1), (_mv(P, z) + q + _mv(Gt, lam)).abs().amax(-1)], -1)
    return z, lam * size[:, None] / gn, res


def composite_step(g: Tensor, n: Tensor, r: Tensor, J: Tensor, limit: Tensor,
                   metric: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """(d [B, 3K], left [B, m], price [B, m]). g [B, m] row values (<= 0 holds), n [B, m, 3K] their gradients,
    r [B, M] effort offsets, J [B, M, 3K] their slope, limit [B, 3K] the move limit per coordinate,
    metric [3K] metres per unit of each coordinate (1, 1, RHO per step).

    QP 1 is the least violation the move limit allows: ``g_j + n_j . d <= 0`` hard on the rows that hold, soft with
    weight ``1 / nu_j^2`` on the violated ones, so each row counts how far the landings would still have to move
    (``nu_j`` is ``|n_j|`` per metre of the step bound's travel). QP 2 is the least effort ``1/2 |r + J d|^2`` that
    leaves no violated row worse than QP 1 did (plus 0.1 mm of travel, ADMM's tolerance) and breaks no row that holds.
    ``left`` is the violation QP 1 leaves, ``price`` QP 2's multiplier of each row, in effort per unit of ``g``."""
    B, m, k3 = n.shape
    rn = n.norm(dim=-1)
    nu = torch.where(rn > _ZERO_ROW, (n / metric).norm(dim=-1), torch.ones_like(rn))           # [B, m]
    viol = g > 0
    eye = torch.eye(k3, device=n.device, dtype=n.dtype).expand(B, -1, -1)
    G = torch.cat([n, eye], 1)
    lo = torch.cat([torch.full_like(g, -float("inf")), -limit], 1)
    soft = torch.cat([torch.where(viol, nu.pow(-2), torch.zeros_like(nu)), torch.zeros_like(limit)], 1)
    zero = torch.zeros(B, k3, k3, device=n.device, dtype=n.dtype)
    d1, _, _ = admm(zero, torch.zeros_like(limit), G, lo, torch.cat([-g, limit], 1), soft=soft)
    d1 = torch.minimum(torch.maximum(d1, -limit), limit)  # ADMM's z may overshoot the box by its residual
    left = torch.where(viol, (g + _mv(n, d1)).relu(), torch.zeros_like(g))
    hi = torch.cat([torch.where(viol, left - g + TRAVEL_TOL * nu, -g), limit], 1)   # 0.1 mm of travel
    z, lam, _ = admm(J.mT @ J, _mv(J.mT, r), G, lo, hi)
    return torch.minimum(torch.maximum(z, -limit), limit), left, lam[:, :m]
