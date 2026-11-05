"""The rollout's hand rotations must carry correct gradients to the configurations.

cuRobo v2's FK backward decodes dL/dquaternion with a body-frame formula but dots the result with world-frame
joint axes, so orientation gradients reaching PyTorch come out wrong (sign right on about half the particles) while
position gradients are exact. That made Adam alone unable to optimize the pick and place configurations.
"""

import torch

from cutamp.tests.conftest import clone_particles, needs_cuda


@needs_cuda
def test_rollout_hand_rotation_gradient_matches_finite_differences(blocks_t1_plan):
    from cutamp.config import TAMPConfiguration
    from cutamp.rollout import RolloutFunction

    world, skeleton, particles = blocks_t1_plan
    rollout_fn = RolloutFunction(skeleton, world, TAMPConfiguration(num_particles=64))
    p = clone_particles(particles)
    confs = [c for c in rollout_fn.conf_params if p[c].dtype.is_floating_point]
    gen = torch.Generator(device=p[confs[0]].device).manual_seed(0)
    W = torch.randn(64, len(confs), 3, 3, device=p[confs[0]].device, generator=gen)
    d = {c: torch.randn(p[c].shape, device=p[c].device, generator=gen) for c in confs}

    def per_particle(parts) -> torch.Tensor:
        rot = rollout_fn(parts)["world_from_ee"][..., :3, :3]
        return (rot * W).sum((1, 2, 3))

    for c in confs:
        p[c].requires_grad_(True)
    f = per_particle(p)
    grads = torch.autograd.grad(f.sum(), [p[c] for c in confs])
    autograd = sum((g * d[c]).sum(-1) for g, c in zip(grads, confs))

    eps = 1e-3
    with torch.no_grad():
        plus = per_particle({**p, **{c: p[c] + eps * d[c] for c in confs}}).clone()  # cuRobo reuses buffers
        minus = per_particle({**p, **{c: p[c] - eps * d[c] for c in confs}}).clone()
    fd = (plus - minus) / (2 * eps)

    rel = ((autograd - fd).abs() / fd.abs().clamp_min(1e-3)).median().item()
    agree = ((autograd * fd) > 0).float().mean().item()
    assert agree > 0.98 and rel < 1e-2, f"sign agrees on {agree:.0%} of particles, median relative error {rel:.3g}"
