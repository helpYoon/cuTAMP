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
    from curobo.types import DeviceCfg, JointState

    from cutamp.robots.t1 import T1Model


def base_wrench(q_base: torch.Tensor, tau_base: torch.Tensor):
    """The wrench the six base joints apply, in the world frame: ``(F, M, P)``, each ``[..., 3]``.

    The joints are x, y, z (prismatic along the world axes), then pitch (y), roll (x), yaw (z), all
    turning about the pelvis point ``P = q[..., :3]``. ``F`` is the force and ``M`` the moment about P.
    """
    pitch, roll = q_base[..., 3], q_base[..., 4]
    cp, sp, cr, sr = pitch.cos(), pitch.sin(), roll.cos(), roll.sin()
    zero = torch.zeros_like(pitch)
    a = torch.stack([zero, zero + 1, zero], -1)       # the pitch, roll and yaw axes in the world
    b = torch.stack([cp, zero, -sp], -1)
    c = torch.stack([cr * sp, -sr, cr * cp], -1)
    t = tau_base[..., 3:]
    # Solves [a b c]^T M = tau_rot in closed form (no linalg call, so it captures in a CUDA graph): a . (b x c) is
    # -cos(roll). Singular at |roll| = 90 deg, the Trunk on its side, which planted soles never reach (the base's
    # roll limit stops 0.1 rad short of it).
    moment = (t[..., 0:1] * torch.cross(b, c, dim=-1) + t[..., 1:2] * torch.cross(c, a, dim=-1)
              + t[..., 2:3] * torch.cross(a, b, dim=-1)) / -cr[..., None]
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
    """The hook for ``_curobo_internals.set_external_wrench_fn``.

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
    extra metrics rollouts. All of them get the hook, and their FK computes the CoM (the split reads it, and so
    does the planner's CoM cost). The IK solver and the graph planner keep RNEA's single pass (7.4a) and no CoM.
    Raises ``ValueError`` if a transition has no dynamics.
    """
    from cutamp._curobo_internals import set_external_wrench_fn

    solver = planner.trajopt_solver
    rollouts = list(solver.get_all_rollout_instances()) + list(solver.additional_metrics_rollouts.values())
    transitions = [tm for r in rollouts for tm in (r.transition_model, r.metrics_transition_model)]
    for tm in transitions:
        if tm.robot_dynamics is None:
            raise ValueError("install_floor_wrench needs trajopt built with load_dynamics")
        tm.robot_model.compute_com = True  # read on every FK call; its CoM buffer is always allocated
    names, tools = solver.joint_names, solver.tool_frames
    kp = transitions[0].robot_dynamics.kinematics_config
    fn = make_floor_wrench_fn([names.index(j) for j in base_joint_names],
                              tuple(tools.index(f) for f in sole_frames),
                              tuple(kp.link_name_to_idx_map[f] for f in sole_frames), kp.num_links)
    for tm in transitions:
        set_external_wrench_fn(tm, fn)


class SegmentTorques:
    """cuRobo's two RNEA passes with the floor's push, as trajopt's transitions run them, on a plan's frames: the
    balance gate's torque check, which with the speed limits sets each segment's pace, and the reason a trajopt
    failed (design 7.4a).

    Calling it on a ``plan_js`` (position, velocity and acceleration, by joint name) returns the second pass's
    ``[T, dof]`` torques in ``model.joint_names`` order. Eager. ``effort`` ``[dof]`` is cuRobo's bound on each
    joint's |torque|: the pinned URDF's effort, and 10,000 on the massless base joints (link_params.py:35).
    ``velocity`` ``[dof]`` is cuRobo's bound on each joint's |velocity| (``get_joint_limits()``, never the URDF's
    velocity times the YAML's velocity_scale: cuRobo applies that scale twice).
    """

    def __init__(self, model: T1Model, device_cfg: DeviceCfg):
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg

        from cutamp.robots.t1 import get_t1_kinematics

        self.joint_names = list(model.joint_names)
        # The robot config's load_dynamics only makes RobotCfg build a DynamicsCfg like this one from the same
        # kinematics (types/robot.py); KinematicsCfg reads the URDF's inertias either way.
        self.kin = get_t1_kinematics(model, device_cfg)  # compute_com, and the joint order checked against model's
        kp = self.kin.config.kinematics_config
        self.dyn = Dynamics(DynamicsCfg(kinematics_config=kp, device_cfg=device_cfg))
        tools = list(self.kin.tool_frames)
        self.fn = make_floor_wrench_fn([self.joint_names.index(j) for j in model.joint_names[model.base]],
                                       tuple(tools.index(f) for f in model.sole_frames),
                                       tuple(kp.link_name_to_idx_map[f] for f in model.sole_frames), kp.num_links)
        limits = self.kin.get_joint_limits()  # each [2, dof]: -bound, +bound
        self.effort, self.velocity = limits.effort[1].clone(), limits.velocity[1].clone()

    def __call__(self, plan_js: JointState) -> torch.Tensor:
        from curobo.types import JointState

        if plan_js.joint_names is not None and list(plan_js.joint_names) != self.joint_names:
            plan_js = plan_js.reorder(self.joint_names)  # a plan also carries the locked joints
        n = len(self.joint_names)
        p, v, a = (x.reshape(1, -1, n).contiguous() for x in (plan_js.position, plan_js.velocity, plan_js.acceleration))
        js = JointState(position=p, velocity=v, acceleration=a, jerk=torch.zeros_like(p), joint_names=self.joint_names)
        with torch.no_grad():
            self.dyn.setup_batch_size(1, p.shape[1])
            tau_free = self.dyn.compute_inverse_dynamics(js).clone()  # Dynamics and Kinematics hand back reused buffers
            ks = self.kin.compute_kinematics(js).clone()
            return self.dyn.compute_inverse_dynamics(js, f_ext=self.fn(js, ks, tau_free)).clone()[0]
