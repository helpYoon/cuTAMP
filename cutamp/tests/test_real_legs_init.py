"""A plan's starting particles on the stance (design 7.3). On the real legs:
- every pick, place and retract conf of a plan without steps (``real_plan``) stands on the stance, each pick and place
  with its CoM in the keyframe rectangle;
- the tests' noisy start (:func:`_noisy_start`) stands on the stance, its base re-solved there (the stance is the
  noise's root);
- every conf locks only its idle arm out of Adam, and retract confs join the joint-limit and self-collision
  constraints.
The GPU tests share conftest's module-scoped ``real_world`` and ``real_plan``: one world per module."""

import roma
import torch

from cutamp.robots.t1 import REAL_LEGS
from cutamp.tests.conftest import needs_cuda
from cutamp.tests.test_t1_real_legs_model import hips_crossed

#: The IK's success tolerances, and the balance gate's sole drift (decision 5).
POS_TOL, ROT_TOL = 0.005, 0.05


def _home(world) -> torch.Tensor:
    return torch.tensor(world.model.home, device=world.device)


def _poses(world, q: torch.Tensor, frame: str) -> torch.Tensor:
    """``[B, 4, 4]`` world poses of ``frame`` at the confs ``q`` ``[B, dof]``."""
    from curobo.types import JointState
    kin = world.kinematics
    state = kin.compute_kinematics(JointState.from_position(q.contiguous(), joint_names=list(kin.joint_names)))
    return state.tool_poses.get_link_pose(frame, make_contiguous=True).get_matrix().reshape(-1, 4, 4)


def _sole_errors(world, q: torch.Tensor):
    """``([B, 2] m, [B, 2] rad)``: each sole's pose error at ``q`` against its stance pose."""
    from cutamp.costs import curobo_pose_error
    soles = torch.stack([_poses(world, q, frame) for frame in world.model.sole_frames], dim=1)
    return curobo_pose_error(soles, world.stance_soles.expand_as(soles))


def _ops(skeleton, *action_types):
    """``(arm, conf)`` of each op in ``skeleton`` whose action type is one of ``action_types``: the conf is the op's
    last parameter (Pick, Place and both Retracts)."""
    return [(op.operator.metadata.arm, op.values[-1]) for op in skeleton
            if op.operator.metadata.action_type in action_types]


def _span(*slices) -> set:
    return {i for s in slices for i in range(s.start, s.stop)}


#: The leg joints :func:`_noisy_start` draws once per pair, and those it holds at home.
_MIRRORED_LEG_JOINTS = ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch")
_HELD_LEG_JOINTS = ("Hip_Roll", "Hip_Yaw", "Ankle_Roll")


def _noisy_start(world, batch_size: int, arm, noise_std: float) -> torch.Tensor:
    """``[batch_size, dof]``: confs that stand on the stance, for tests: ``world.q_init`` plus Gaussian noise of std
    ``noise_std`` rad on the body and ``arm`` (None: both arms), clamped to the joint limits; the other arm keeps
    ``q_init``. The model's root is the trunk, so that noise would swing the soles. The stance is the noise's root
    instead: the legs draw each pitch once per pair and hold roll and yaw, and the base is re-solved so that the first
    sole sits on its stance pose (:func:`_stand_on_the_stance`). The planner's GN IK drew its starts so before every
    plan went through the stepped route (2026-11-25)."""
    model = world.model
    seed = world.q_init.unsqueeze(0).expand(batch_size, -1).clone()
    noise = torch.randn(batch_size, seed.shape[-1], device=seed.device) * noise_std
    noise[:, model.base] = 0.0  # re-solved from the stance below
    if arm == "left":
        noise[:, model.right_arm] = 0.0
    elif arm == "right":
        noise[:, model.left_arm] = 0.0
    col = list(model.joint_names).index
    for joint in _MIRRORED_LEG_JOINTS:
        noise[:, col(f"Right_{joint}")] = noise[:, col(f"Left_{joint}")]
    for joint in _HELD_LEG_JOINTS:
        for side in ("Left", "Right"):
            noise[:, col(f"{side}_{joint}")] = 0.0
    lower, upper = world.joint_limits.to(seed.device, seed.dtype)  # equal left and right pitch limits: still mirrored
    return _stand_on_the_stance(world, torch.clamp(seed + noise, min=lower, max=upper), lower, upper)


def _stand_on_the_stance(world, seed: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """``seed`` with each row's base re-solved so that the first of ``model.sole_frames`` sits on its stance pose:
    ``B' = S @ inv(W) @ B``, with ``S`` the stance pose, ``W`` the sole's FK pose and ``B`` the base pose. With mirrored
    legs the other sole's pose relative to the first is unchanged, so it lands on its stance pose too. A row whose
    ``B'`` leaves the base joints' limits takes home's base and body, keeping only its arm's noise."""
    from curobo.types import JointState
    from cutamp.particle_initialization import _BASE_JOINTS
    model = world.model
    assert tuple(model.joint_names[model.base]) == _BASE_JOINTS, model.joint_names[model.base]
    kin = world.kinematics
    state = kin.compute_kinematics(JointState.from_position(seed, joint_names=list(kin.joint_names)))
    # get_matrix copies the pose out of the FK state: cuRobo reuses its output buffers.
    sole = state.tool_poses.get_link_pose(model.sole_frames[0], make_contiguous=True).get_matrix().reshape(-1, 4, 4)
    stance = world.stance_soles[0]
    q_base = seed[:, model.base]
    rot = stance[:3, :3] @ sole[:, :3, :3].transpose(-1, -2)
    new_rot = rot @ roma.euler_to_rotmat("YXZ", q_base[:, 3:])
    new_pos = (rot @ (q_base[:, :3] - sole[:, :3, 3]).unsqueeze(-1)).squeeze(-1) + stance[:3, 3]
    new_base = torch.cat([new_pos, roma.rotmat_to_euler("YXZ", new_rot)], dim=-1)
    inside = ((lower[model.base] <= new_base) & (new_base <= upper[model.base])).all(-1, keepdim=True)
    home = torch.tensor(model.home, device=seed.device, dtype=seed.dtype)
    out = seed.clone()
    out[:, model.base] = torch.where(inside, new_base, home[model.base])
    out[:, model.body] = torch.where(inside, seed[:, model.body], home[model.body])
    return out


@needs_cuda
def test_noisy_starts_stand_on_the_stance(real_world):
    # The legs draw their pitch once per pair and hold their roll and yaw, and the base puts the left sole
    # on the stance; the right sole follows. Standing on the stance, a seed's Trunk is in the legs' reach, so its base
    # could leave the box only by leaning within 0.1 rad of its side (the roll cap), which legs bent alike in pitch
    # never do. Under a box cut lower, a row whose base would leave it keeps home's base and body.
    world, m, n = real_world, real_world.model, 256
    names, home = list(m.joint_names), _home(real_world)
    torch.manual_seed(0)
    q = _noisy_start(world, n, "left", 0.3)
    pos, rot = _sole_errors(world, q)
    assert float(pos.max()) <= 1e-5 and float(rot.max()) <= 1e-4, (float(pos.max()), float(rot.max()))
    lo, hi = world.joint_limits
    assert bool(((lo <= q) & (q <= hi)).all())
    held = [names.index(f"{side}_{joint}") for side in ("Left", "Right")
            for joint in ("Hip_Roll", "Hip_Yaw", "Ankle_Roll")] + sorted(_span(m.right_arm))
    assert torch.equal(q[:, held], home[held].expand(n, -1))
    for joint in ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch"):
        assert torch.equal(q[:, names.index(f"Right_{joint}")], q[:, names.index(f"Left_{joint}")]), joint
    idle = world.tool_frame_for_arm("right")
    moved = (_poses(world, q, idle)[:, :3, 3] - _poses(world, home.view(1, -1), idle)[:, :3, 3]).norm(dim=-1)
    assert float(moved.median()) >= 0.05, float(moved.median())
    assert not bool((q[:, m.base] == home[m.base]).all(-1).any())
    z = names.index("base_j_z")
    cut = hi.clone()
    cut[z] = q[:, z].median()
    out = _stand_on_the_stance(world, q, lo, cut)
    fell_back = (out[:, m.base] == home[m.base]).all(-1)
    assert 0 < int(fell_back.sum()) < n, int(fell_back.sum())
    assert torch.equal(out[fell_back][:, m.body], home[m.body].expand(int(fell_back.sum()), -1))


@needs_cuda
def test_real_leg_locks(real_plan):
    # On the real legs every pick, place and retract conf locks its idle arm only: the put-back holds a retract on
    # its stance.
    from cutamp.conf_locking import get_conf_locked_dofs
    skeleton, _ = real_plan
    retracts, reaches = _ops(skeleton, "retract"), _ops(skeleton, "pick", "place")
    assert retracts and reaches
    for arm, conf in retracts + reaches:
        idle = REAL_LEGS.right_arm if arm == "left" else REAL_LEGS.left_arm
        assert set(get_conf_locked_dofs(skeleton, conf, REAL_LEGS)) == _span(idle), conf


@needs_cuda
def test_retract_confs_join_the_motion_constraints(real_world, real_plan):
    from cutamp.config import TAMPConfiguration
    from cutamp.cost_function import CostFunction
    from cutamp.rollout import RolloutFunction
    from cutamp.scripts.utils import default_constraint_to_tol
    from cutamp.task_planning.constraints import ComPolygon, Motion
    from cutamp.tests.conftest import clone_particles, fixed_stances
    skeleton, particles = real_plan
    retracts = [conf for _, conf in _ops(skeleton, "retract")]
    config = TAMPConfiguration(num_particles=64)
    names = list(real_world.model.joint_names)
    particles = clone_particles(particles)
    z = names.index("base_j_z")
    with torch.no_grad():
        particles[retracts[0]][0] = hips_crossed(real_world)
        particles[retracts[0]][1, z] = real_world.joint_limits[0][z] - 0.05  # under the base's floor, the legs' reach
        cost_dict = CostFunction(skeleton, real_world, config)(
            RolloutFunction(skeleton, real_world)(particles), particles,
            achieved_stances=fixed_stances(real_world, particles))
    assert ComPolygon.type in cost_dict  # the CoM is a hard constraint at every conf
    values = cost_dict[Motion.type]["values"]
    joint_limit, self_collision = values["retract_joint_limit"], values["retract_self_collision"]
    assert joint_limit.shape == self_collision.shape == (64, len(retracts))
    assert float(self_collision[0, 0]) > 0.0 and float(joint_limit[1, 0]) > 0.0
    others = torch.ones_like(joint_limit, dtype=torch.bool)
    others[:2, 0] = False
    assert not bool(joint_limit[others].any()) and not bool(self_collision[others].any())
    # Exact, as the other Motion values.
    tols = {k: default_constraint_to_tol[Motion.type][k] for k in values}
    assert tols == dict.fromkeys(tols, 0.0)
