"""The real legs' FK rotations must carry correct gradients to the configurations, as the simplified model's hands
do (test_fk_orientation_grad.py).

cuRobo v2's FK backward gets orientation gradients wrong while positions are exact, and ``_FKQuatGrad`` corrects
the rotations that autograd differentiates. On the real legs two more rotations can reach autograd: the rollout's
soles (``sole_rot_err``, in Adam's loss) and each sole's in the floor wrench (trajopt's transition calls that hook
under no_grad; the test differentiates it directly). Their CoM, a position, is a sanity check. The error grows with
the link's rotation from the world's axes, and planted soles have none (the stance's soles sit at the identity), so
those checks run on tilted confs (``_tilt``). One world, ``real_world``: a module using it must not build a second.
"""

import torch

from cutamp.tests.conftest import clone_particles, needs_cuda


def _fd_check(f, x: dict, d: dict, eps: float = 1e-3) -> tuple:
    """``(sign agreement, median relative error)`` of autograd's derivative of ``f(x)`` ``[N]`` along ``d`` against
    central differences, for inputs ``x`` ``{name: [N, dof]}``: the measure of test_fk_orientation_grad.py's
    hand-rotation test."""
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
    """Each ``{name: (agree, rel)}`` within that hand-rotation test's bounds, all named in the message."""
    msg = "; ".join(f"{k}: sign agrees on {a:.1%}, median relative error {r:.3g}" for k, (a, r) in results.items())
    assert all(a > 0.98 and r < 1e-2 for a, r in results.values()), msg


def _tilt(q: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """``q`` plus N(0, 0.3 rad) noise on every joint, which tilts a real-leg conf's soles by about 0.8 rad (median)
    from the stance. The kernel's body-frame decode is the world-frame one at the identity rotation, so at planted
    soles their gradients come out right without the fix (sign right on every particle, median relative error 3e-4)."""
    return q.detach() + 0.3 * torch.randn(q.shape, device=q.device, generator=gen)


def _real_confs(real_plan, gen: torch.Generator) -> torch.Tensor:
    """``[N, dof]``: every configuration in ``real_plan``'s particles, stacked and tilted (``_tilt``)."""
    _, particles = real_plan
    q = torch.cat([v for k, v in particles.items() if k.startswith(("left_q", "right_q")) and v.ndim == 2])
    return _tilt(q, gen)


@needs_cuda
def test_real_legs_rollout_rotation_gradients_match_finite_differences(real_world, real_plan):
    from cutamp.config import TAMPConfiguration
    from cutamp.rollout import RolloutFunction

    skeleton, particles = real_plan
    rollout_fn = RolloutFunction(skeleton, real_world, TAMPConfiguration(robot="real_legs", num_particles=64))
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
def test_real_legs_floor_wrench_gradient_matches_finite_differences(real_plan):
    from curobo.types import DeviceCfg, JointState

    from cutamp.floor_wrench import SegmentTorques
    from cutamp.robots.t1 import REAL_LEGS

    device_cfg = DeviceCfg()
    seg = SegmentTorques(REAL_LEGS, device_cfg)  # the real legs' FK with the CoM, RNEA and the floor-wrench hook
    gen = torch.Generator(device=device_cfg.device).manual_seed(0)
    q0 = _real_confs(real_plan, gen)
    n, names = q0.shape[0], seg.joint_names
    zero = torch.zeros_like(q0).view(1, n, -1)
    with torch.no_grad():  # the first pass's support load at q0, held fixed: the check is of the hook and its FK
        seg.dyn.setup_batch_size(1, n)
        tau_free = seg.dyn.compute_inverse_dynamics(JointState(
            position=q0.view(1, n, -1), velocity=zero, acceleration=zero, jerk=zero, joint_names=names)).clone()
    W = torch.randn(1, n, seg.kin.config.kinematics_config.num_links, 6, device=q0.device, generator=gen)

    def weighted_wrench(x) -> torch.Tensor:
        js = JointState.from_position(x["q"].view(1, n, -1), joint_names=names)
        return (seg.fn(js, seg.kin.compute_kinematics(js), tau_free) * W).sum((-2, -1)).view(n)

    d = torch.randn(q0.shape, device=q0.device, generator=gen)
    _assert_fd({"f_ext": _fd_check(weighted_wrench, {"q": q0}, {"q": d})})


@needs_cuda
def test_real_legs_com_gradient_matches_finite_differences(real_world, real_plan):
    # A sanity check, not a fix: the CoM is a position, which cuRobo's FK backward gets right. All 33 joints move,
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
