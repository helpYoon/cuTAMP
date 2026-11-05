"""cuRobo v2's FK orientation gradients, corrected on their way to PyTorch autograd.

cuRobo v2's FK backward decodes dL/dquaternion with a body-frame map but dots the result with world-frame joint
axes, so the gradient of a link's rotation to the joint positions comes out wrong while positions are exact. The
error grows with the link's rotation from the world's axes. The FK rotations that autograd can differentiate pass
through ``_FKQuatGrad`` (``_link_matrix`` for a link pose's matrix): the rollout's hands and soles (rollout.py) and
each sole's rotation in the floor wrench (floor_wrench.py; trajopt's transition calls that hook under no_grad, so
only a caller that differentiates the hook relies on it). The CoM costs' support frame needs neither: it is
constant in q on the real legs and a pure yaw on the simplified model (com_polygon_cost.py).

A module of its own because rollout.py imports tamp_world.py, which imports floor_wrench.py; rollout.py re-exports
both names.
"""

import torch


class _FKQuatGrad(torch.autograd.Function):
    """Identity on an FK quaternion ``[..., 4]`` (wxyz) whose backward re-encodes dL/dq for cuRobo v2's FK backward.

    That backward decodes the quaternion gradient with the body-frame map (``quaternion_gradient_to_angular_velocity``
    in quaternion_util.cuh) but dots the result with world-frame joint axes, which need
    Omega = 0.5 (-g_w v + w g_v + v x g_v). Passing g' = 2 q (x) (0, Omega) makes its decode return Omega exactly
    (for unit q, at any angle). cuRobo's own pose costs already encode this way; only PyTorch autograd needs it.
    If the fork's decode is ever changed to the world-frame formula, drop this: test_fk_orientation_grad.py would
    then fail on the double correction.
    """

    @staticmethod
    def forward(ctx, quat: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(quat.detach().clone())  # cuRobo may reuse the FK output buffer before backward
        return quat.clone()

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> torch.Tensor:
        (q,) = ctx.saved_tensors
        w, v, g_v = q[..., :1], q[..., 1:], g[..., 1:]
        omega = 0.5 * (w * g_v - g[..., :1] * v + torch.cross(v, g_v, dim=-1))
        return 2.0 * torch.cat((-(v * omega).sum(-1, keepdim=True), w * omega + torch.cross(v, omega, dim=-1)), -1)


def _link_matrix(pose) -> torch.Tensor:
    """``pose.get_matrix()`` for a cuRobo FK link pose, with gradients correct w.r.t. the joint positions."""
    pose.quaternion = _FKQuatGrad.apply(pose.quaternion)
    return pose.get_matrix()
