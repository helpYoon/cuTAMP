"""The real legs' FK rotations must carry correct gradients to the configurations.

These checks guard cuRobo's FK backward, a CUDA kernel that the source guard does not cover. Besides the hands, the
rollout's soles (``sole_rot_err``, in Adam's loss) carry rotation gradients on the real legs; their CoM, a position,
is a sanity check. A wrong orientation gradient grows with the link's rotation from the world's axes, and planted soles
have none (the stance's soles sit at the identity), so the checks run on tilted confs (``_tilt``). One world,
``real_world``: a module using it must not build a second.
"""

import torch

from cutamp.tests.conftest import clone_particles, needs_cuda


def _fd_check(f, x: dict, d: dict, eps: float = 1e-3) -> tuple:
    """``(sign agreement, median relative error)`` of autograd's derivative of ``f(x)`` ``[N]`` along ``d`` against
    central differences, for inputs ``x`` ``{name: [N, dof]}``."""
    x = {k: v.detach().clone().requires_grad_(True) for k, v in x.items()}
    grads = torch.autograd.grad(f(x).sum(), list(x.values()))
    autograd = sum((g * d[k]).sum(-1) for g, k in zip(grads, x))
    with torch.no_grad():
        plus = f({k: v + eps * d[k] for k, v in x.items()}).clone()  # cuRobo reuses buffers
        minus = f({k: v - eps * d[k] for k, v in x.items()}).clone()
    fd = (plus - minus) / (2 * eps)
    rel = ((autograd - fd).abs() / fd.abs().clamp_min(1e-3)).median().item()
    agree = ((autograd * fd) > 0).float().mean().item()
    return agree, rel


def _assert_fd(results: dict) -> None:
    """Each ``{name: (agree, rel)}`` within the bounds (sign agreement above 98%, median relative error below 1e-2),
    all named in the message."""
    msg = "; ".join(f"{k}: sign agrees on {a:.1%}, median relative error {r:.3g}" for k, (a, r) in results.items())
    assert all(a > 0.98 and r < 1e-2 for a, r in results.values()), msg


def _tilt(q: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """``q`` plus N(0, 0.3 rad) noise on every joint, which tilts a real-leg conf's soles by about 0.8 rad (median)
    from the stance: at planted soles a body-frame decode is the world-frame one, so an FK backward that confused
    them would still pass."""
    return q.detach() + 0.3 * torch.randn(q.shape, device=q.device, generator=gen)


def _real_confs(real_plan, gen: torch.Generator) -> torch.Tensor:
    """``[N, dof]``: every configuration in ``real_plan``'s particles, stacked and tilted (``_tilt``)."""
    _, particles = real_plan
    q = torch.cat([v for k, v in particles.items() if k.startswith(("left_q", "right_q")) and v.ndim == 2])
    return _tilt(q, gen)


@needs_cuda
def test_real_legs_rollout_rotation_gradients_match_finite_differences(real_world, real_plan):
    from cutamp.rollout import RolloutFunction

    skeleton, particles = real_plan
    rollout_fn = RolloutFunction(skeleton, real_world)
    p = clone_particles(particles)
    confs = [c for c in rollout_fn.conf_params if p[c].dtype.is_floating_point]
    gen = torch.Generator(device=real_world.device).manual_seed(0)
    p.update({c: _tilt(p[c], gen) for c in confs})
    d = {c: torch.randn(p[c].shape, device=p[c].device, generator=gen) for c in confs}

    def check(key: str) -> tuple:
        with torch.no_grad():
            W = torch.randn(rollout_fn(p)[key][..., :3, :3].shape, device=real_world.device, generator=gen)
        return _fd_check(lambda x: (rollout_fn({**p, **x})[key][..., :3, :3] * W).flatten(1).sum(-1),
                         {c: p[c] for c in confs}, d)

    _assert_fd({key: check(key) for key in ("world_from_ee", "world_from_soles")})


@needs_cuda
def test_real_legs_com_gradient_matches_finite_differences(real_world, real_plan):
    # A sanity check: the CoM is a position, which cuRobo's FK backward gets right. All 33 joints move,
    # the floating base's six among them, on the FK that the ComPolygon constraint differentiates.
    from cutamp.com_polygon_cost import _fk_state

    gen = torch.Generator(device=real_world.device).manual_seed(0)
    q0 = _real_confs(real_plan, gen)
    kin = real_world.site_kinematics("com")
    W = torch.randn(q0.shape[0], 3, device=q0.device, generator=gen)

    def weighted_com(x) -> torch.Tensor:
        return (_fk_state(real_world, x["q"], kin).robot_com[..., :3].reshape(-1, 3) * W).sum(-1)

    d = torch.randn(q0.shape, device=q0.device, generator=gen)
    _assert_fd({"robot_com": _fd_check(weighted_com, {"q": q0}, {"q": d})})
