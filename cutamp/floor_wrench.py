"""The floor's push on the planted soles, for cuRobo's RNEA (design 7.4a).

The robot hangs from six virtual base joints. RNEA's first pass, with nothing touching the robot,
puts the whole support on them. The floor supplies exactly that wrench instead, split between the
two soles by where the CoM sits between them, and the second pass leaves the base joints nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import torch
from curobo._src.geom.transform import torch_quaternion_to_matrix

if TYPE_CHECKING:
    from curobo.motion_planner import MotionPlanner


def base_wrench(q_base: torch.Tensor, tau_base: torch.Tensor):
    """The wrench the six base joints apply, in the world frame: ``(F, M, P)``, each ``[..., 3]``.

    The joints are x, y, z (prismatic along the world axes), then yaw (z), pitch (y), roll (x), all
    turning about the pelvis point ``P = q[..., :3]``. ``F`` is the force and ``M`` the moment about P.
    """
    yaw, pitch = q_base[..., 3], q_base[..., 4]
    cy, sy, cp, sp = yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin()
    zero = torch.zeros_like(yaw)
    a = torch.stack([zero, zero, zero + 1], -1)       # the yaw, pitch and roll axes in the world
    b = torch.stack([-sy, cy, zero], -1)
    c = torch.stack([cy * cp, sy * cp, -sp], -1)
    t = tau_base[..., 3:]
    # Solves [a b c]^T M = tau_rot in closed form (no linalg call, so it captures in a CUDA graph).
    # ponytail: singular at |pitch| = 90 deg, which a standing pelvis never reaches.
    moment = (t[..., 0:1] * torch.cross(b, c, dim=-1) + t[..., 1:2] * torch.cross(c, a, dim=-1)
              + t[..., 2:3] * torch.cross(a, b, dim=-1)) / -cp[..., None]
    return tau_base[..., :3], moment, q_base[..., :3]


def split_between_soles(F, M, P, s_left, s_right, com):
    """Split the wrench ``(F, M about P)`` between the soles: ``(f_l, m_l, f_r, m_r)``, world frame.

    Each sole's share is where the CoM's xy projection falls on the segment between the sole
    centres, clamped to it. Each moment is about its own sole centre, and the two wrenches add up to
    ``(F, M)`` exactly.
    """
    d = (s_right - s_left)[..., :2]
    t = (((com - s_left)[..., :2] * d).sum(-1) / (d * d).sum(-1)).clamp(0.0, 1.0)[..., None]
    share_l, share_r = 1.0 - t, t
    s_mean = share_l * s_left + share_r * s_right
    m = M + torch.cross(P - s_mean, F, dim=-1)
    return share_l * F, share_l * m, share_r * F, share_r * m


def make_floor_wrench_fn(
    base_joints: Sequence[int], sole_tools: tuple[int, int], sole_links: tuple[int, int], n_links: int
) -> Callable:
    """The hook for ``RobotStateTransition.set_external_wrench_fn``.

    ``base_joints`` index the six base joints in the joint state, ``sole_tools`` the soles in the
    tool frames, and ``sole_links`` the soles in Dynamics' links. It returns ``f_ext``
    ``[B, H, n_links, 6]``, zero but for the sole rows, each ``[torque, force]`` in that sole's frame.
    """
    base_joints = list(base_joints)

    def floor_wrench(state, kin_state, tau_free: torch.Tensor) -> torch.Tensor:
        q = torch.stack([state.position[..., j] for j in base_joints], -1)
        tau = torch.stack([tau_free[..., j] for j in base_joints], -1)
        F, M, P = base_wrench(q, tau)
        pos, quat = kin_state.tool_poses.position, kin_state.tool_poses.quaternion
        wrenches = split_between_soles(F, M, P, pos[..., sole_tools[0], :], pos[..., sole_tools[1], :],
                                       kin_state.robot_com[..., :3])
        f_ext = q.new_zeros(*q.shape[:-1], n_links, 6)
        for i in range(2):
            f, m = wrenches[2 * i], wrenches[2 * i + 1]
            r_t = torch_quaternion_to_matrix(quat[..., sole_tools[i], :]).transpose(-1, -2)
            f_ext[..., sole_links[i], :3] = (r_t @ m[..., None])[..., 0]
            f_ext[..., sole_links[i], 3:] = (r_t @ f[..., None])[..., 0]
        return f_ext

    return floor_wrench


def install_floor_wrench(planner: MotionPlanner, base_joint_names: Sequence[str], sole_frames: tuple[str, str]) -> None:
    """Hook the floor wrench into trajopt's own transitions. Call it before ``planner``'s first plan or
    warm-up: cuRobo's CUDA graphs keep the hook they were captured with.

    cuRobo gives each rollout two transitions: its cost's (``transition_model``) and its metrics'
    (``metrics_transition_model``, which the success test runs); the interpolated re-check runs on the
    extra metrics rollouts. All of them get the hook. The IK solver and the graph planner keep RNEA's
    single pass (7.4a). Raises ``ValueError`` if one has no dynamics or was built without ``compute_com``.
    """
    solver = planner.trajopt_solver
    rollouts = list(solver.get_all_rollout_instances()) + list(solver.additional_metrics_rollouts.values())
    transitions = [tm for r in rollouts for tm in (r.transition_model, r.metrics_transition_model)]
    for tm in transitions:
        if tm.robot_dynamics is None or not tm.config.compute_com:
            raise ValueError("install_floor_wrench needs trajopt built with load_dynamics and compute_com")
    names, tools = solver.joint_names, solver.tool_frames
    kp = transitions[0].robot_dynamics.kinematics_config
    fn = make_floor_wrench_fn([names.index(j) for j in base_joint_names],
                              tuple(tools.index(f) for f in sole_frames),
                              tuple(kp.link_name_to_idx_map[f] for f in sole_frames), kp.num_links)
    for tm in transitions:
        tm.set_external_wrench_fn(fn)
