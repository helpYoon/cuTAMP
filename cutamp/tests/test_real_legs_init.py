"""Particle init on the stance (design 7.3). On the real legs:
- every pick and place conf (GN IK, ``gn_ik``) stands on the stance with its CoM in the keyframe rectangle;
- a noisy start of GN IK stands on the stance, its base re-solved there (the stance is the noise's root);
- retract confs stand at home, their base and body locked out of Adam, and join the joint-limit and self-collision
  constraints.
The GPU tests share conftest's module-scoped ``real_world`` and ``real_plan``: one world per module."""

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


@needs_cuda
def test_noisy_starts_stand_on_the_stance(real_world):
    # The legs draw their pitch once per pair and hold their roll and yaw, and the base puts the left sole
    # on the stance; the right sole follows. Standing on the stance, a seed's Trunk is in the legs' reach, so its base
    # could leave the box only by leaning within 0.1 rad of its side (the roll cap), which legs bent alike in pitch
    # never do. Under a box cut lower, a row whose base would leave it keeps home's base and body.
    from cutamp.particle_initialization import _noisy_start, _stand_on_the_stance
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
def test_init_confs_stand_on_the_stance(real_world, real_plan):
    # GN IK holds the soles and the CoM first (level 1), so every row of every pick and place conf stands on the
    # stance with its CoM inside, the rows it accepted and the rest, which keep its closest balanced iterate.
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    skeleton, particles = real_plan
    home = _home(real_world)
    reaches, retracts = _ops(skeleton, "pick", "place"), _ops(skeleton, "retract")
    assert reaches and len(retracts) == len(reaches), skeleton
    for _, conf in reaches:
        q = particles[conf]
        pos, rot = _sole_errors(real_world, q)
        assert float(pos.max()) <= POS_TOL and float(rot.max()) <= ROT_TOL, (conf, float(pos.max()), float(rot.max()))
        assert bool(compute_com_polygon_mask(real_world, q).all()), conf
    for _, conf in retracts:
        assert torch.equal(particles[conf], home.expand_as(particles[conf])), conf


@needs_cuda
def test_real_leg_locks(real_plan):
    # On the real legs a retract conf keeps its stance (base and body) and its idle arm; a pick or place conf locks
    # its idle arm only.
    from cutamp.conf_locking import get_conf_locked_dofs
    skeleton, _ = real_plan
    retracts, reaches = _ops(skeleton, "retract"), _ops(skeleton, "pick", "place")
    assert retracts and reaches
    for arm, conf in retracts + reaches:
        def idle(m):
            return m.right_arm if arm == "left" else m.left_arm
        stance = (REAL_LEGS.base, REAL_LEGS.body) if (arm, conf) in retracts else ()
        assert set(get_conf_locked_dofs(skeleton, conf, REAL_LEGS)) == _span(*stance, idle(REAL_LEGS)), conf


@needs_cuda
def test_retract_confs_join_the_motion_constraints(real_world, real_plan):
    from cutamp.config import TAMPConfiguration
    from cutamp.cost_function import CostFunction
    from cutamp.rollout import RolloutFunction
    from cutamp.scripts.utils import default_constraint_to_tol
    from cutamp.task_planning.constraints import ComPolygon, Motion
    from cutamp.tests.conftest import clone_particles
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
            RolloutFunction(skeleton, real_world)(particles), particles)
    assert ComPolygon.type in cost_dict  # the CoM is a hard constraint at every conf
    values = cost_dict[Motion.type]["values"]
    joint_limit, self_collision = values["retract_joint_limit"], values["retract_self_collision"]
    assert joint_limit.shape == self_collision.shape == (64, len(retracts))
    assert float(self_collision[0, 0]) > 0.0 and float(joint_limit[1, 0]) > 0.0
    at_home = torch.ones_like(joint_limit, dtype=torch.bool)
    at_home[:2, 0] = False
    assert not bool(joint_limit[at_home].any()) and not bool(self_collision[at_home].any())
    # Exact, as the other Motion values (route A's envelope sets no tilt bound, so there is no trunk_tilt value).
    tols = {k: default_constraint_to_tol[Motion.type][k] for k in values}
    assert tols == dict.fromkeys(tols, 0.0)
