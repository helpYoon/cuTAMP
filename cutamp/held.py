"""Held requirements (design 2026-11-23, section 4.1): the hand on its target and both soles on the stance.

A held requirement is not a cost. Before the costs are evaluated, a posture is put back onto it (:func:`put_back`):
one Newton step on the held rows, ``dq = Jᵀ (J Jᵀ + EPS I)⁻¹ r``, taken twice, scaled so that no joint moves more than
``MAX_CORRECTION`` and clamped to the joint limits. Joints sitting on a limit are left where they are in both passes.
The correction prefers nothing: it does not centre, straighten or relax a posture, it only takes away the held rows'
residual. After the backward, a posture keeps the part of its gradient that moves neither the hand nor the feet, and
each held row's multiplier is handed on (:func:`split_gradient`): when a goal moves by ``d``, the posture put back on
it changes the costs by ``mu · d`` (section 4.2), the gradient of :func:`stance_share` and :func:`target_share`.
"""

import torch
from curobo.types import JointState

from cutamp.gn_refresh import _pose_residual_and_jacobian, _sole_residual_and_jacobian, stance_sole_goals
from cutamp.stance import stance_joint_limits

EPS = 1e-8  # on the diagonal of J Jᵀ, a numerical safeguard
MAX_CORRECTION = 0.2  # rad (m for the base's x and y): the most one pass moves a joint, a safeguard on a straight-line step


@torch.no_grad()
def held_rows(world, q: torch.Tensor, stance: torch.Tensor, target: torch.Tensor | None = None,
              arm: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(r [n, m], J [n, m, dof])`` in float64 for the postures ``q`` ``[n, dof]`` on the sole pairs ``stance``
    ``[n, 2, 3]``: goal minus current, world frame, and how it moves with ``q`` (``r`` changes by ``−J dq``). The 12 sole
    rows are left position, left rotation vector, right position, right rotation vector; with ``target`` ``[n, 4, 4]``
    (the tool frame of ``arm``) the hand's 6 rows come first (``m`` 18). A ``target`` needs its ``arm``."""
    if target is not None and arm is None:
        raise ValueError("a hand target needs its arm")
    kin = world.kinematics_jac
    tools = list(kin.tool_frames)
    q = q.contiguous()
    state = kin.compute_kinematics(JointState.from_position(q, joint_names=list(kin.config.kinematics_config.joint_names)))
    every = torch.ones_like(q)
    sole_idx = tuple(tools.index(frame) for frame in world.model.sole_frames)
    r, jac = _sole_residual_and_jacobian(state, sole_idx, stance_sole_goals(stance.to(q)), every)
    r, jac = r.to(torch.float64, copy=True), jac.to(torch.float64, copy=True)  # the kinematics reuses its buffers
    if target is not None:
        frames = tuple((frame, tools.index(frame))
                       for frame in (world.tool_frame_for_arm("left"), world.tool_frame_for_arm("right")))
        is_left = torch.full((len(q),), world.tool_frame_for_arm(arm) == frames[0][0], device=q.device)
        r_hand, jac_hand = _pose_residual_and_jacobian(state, frames, target.to(q), every, is_left)
        r, jac = torch.cat((r_hand.double(), r), 1), torch.cat((jac_hand.double(), jac), 1)
    return r, jac


@torch.no_grad()
def put_back(world, q: torch.Tensor, stance: torch.Tensor, target: torch.Tensor | None = None,
             arm: str | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(q_new [n, dof], J [n, m, dof], capped [n])``: ``q`` put back on its :func:`held_rows` by two Newton passes,
    each clamped to the joint limits (the base to the stance's reach box,
    :func:`~cutamp.stance.stance_joint_limits`); in both, a joint sitting on a limit stays there. ``J`` (float64)
    is the second pass's, those joints' columns zero; ``capped`` marks the rows whose correction ``MAX_CORRECTION``
    scaled down in either pass. A row with anything not finite (posture, stance or rows) keeps its ``q`` and gets a
    zero ``J``."""
    lower, upper = stance_joint_limits(stance.to(q), *world.joint_limits.to(q))
    capped = torch.zeros(len(q), dtype=torch.bool, device=q.device)
    for _ in range(2):
        r, jac = held_rows(world, q, stance, target, arm)
        ok = r.isfinite().all(1) & jac.isfinite().all((1, 2)) & q.isfinite().all(1) & stance.isfinite().all((1, 2))
        r, jac = r.nan_to_num(0.0) * ok[:, None], jac.nan_to_num(0.0) * ok[:, None, None]
        jac = jac * ~((q <= lower + 1e-7) | (q >= upper - 1e-7))[:, None, :]
        eye = torch.eye(jac.shape[1], dtype=jac.dtype, device=jac.device)
        dq = (jac.mT @ torch.linalg.solve_ex(jac @ jac.mT + EPS * eye, r[..., None])[0])[..., 0]
        largest = dq.abs().amax(1, keepdim=True)
        capped |= largest[:, 0] > MAX_CORRECTION
        dq = dq * (MAX_CORRECTION / largest.clamp_min(1e-12)).clamp(max=1.0)
        q = torch.where(ok[:, None], (q + dq.to(q)).clamp(lower, upper), q)
    return q, jac, capped


def split_gradient(grad: torch.Tensor, jac: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(kept [n, dof], mu [n, m])`` for the gradients ``grad`` ``[n, dof]`` of postures held by ``jac``
    ``[n, m, dof]`` (:func:`put_back`'s): one multiplier per held row, ``mu = (J Jᵀ + EPS I)⁻¹ J g`` in float64, and the
    part of the gradient that moves no held row, ``kept = g − Jᵀ mu`` in ``grad``'s dtype. A row whose gradient is not
    finite keeps it and gets ``mu`` 0."""
    fine = grad.isfinite().all(1)
    g = torch.where(fine[:, None], grad, 0).double()
    jac = jac.double()
    eye = torch.eye(jac.shape[1], dtype=jac.dtype, device=jac.device)
    mu = torch.linalg.solve_ex(jac @ jac.mT + EPS * eye, jac @ g[..., None])[0][..., 0]
    kept = torch.where(fine[:, None], (g - (jac.mT @ mu[..., None])[..., 0]).to(grad), grad)
    return kept, mu


def stance_share(mu: torch.Tensor, stance: torch.Tensor) -> torch.Tensor:
    """A scalar whose gradient is the soles' share for whatever ``stance`` ``[n, 2, 3]`` ((x, y, yaw) per sole)
    depends on: each sole's x, y and yaw times ``mu`` ``[n, m]`` (:func:`split_gradient`'s, detached) of its position
    x, position y and rotation about the vertical, the last 12 rows. A row of ``stance`` with anything not finite
    counts as zero."""
    rows = mu.detach()[:, -12:].reshape(-1, 2, 6)
    soles = torch.cat((rows[..., :2], rows[..., 5:]), -1).to(stance)  # columns 0, 1 and 5, without a host index
    fine = stance.isfinite().all((1, 2))
    return (soles * torch.where(fine[:, None, None], stance, 0)).sum()


def target_share(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A scalar whose gradient is the hand's share for whatever ``target`` ``[n, 4, 4]`` depends on: ``mu`` ``[n, 18]``
    (:func:`split_gradient`'s, detached) of the hand's position rows times the target's position, plus that of its
    rotation rows times ``θ = ½ vee(R R₀ᵀ − R₀ Rᵀ)``, ``R₀ = R`` detached: ``θ`` is zero, and its derivative is the
    target's rotation in the world frame, the hand rows' metric. A row of ``target`` with anything not finite counts as
    zero. Raises ValueError unless ``mu`` has the 18 rows of a pick or place."""
    if mu.shape[1] != 18:
        raise ValueError(f"a hand's share needs the 18 held rows of a pick or place; got {mu.shape[1]}")
    m = mu.detach()[:, :6].to(target)
    target = torch.where(target.isfinite().all((1, 2))[:, None, None], target, 0)
    rot = target[:, :3, :3]
    a = rot @ rot.detach().mT
    theta = 0.5 * torch.stack((a[:, 2, 1] - a[:, 1, 2], a[:, 0, 2] - a[:, 2, 0], a[:, 1, 0] - a[:, 0, 1]), -1)
    return (m[:, :3] * target[:, :3, 3]).sum() + (m[:, 3:] * theta).sum()
