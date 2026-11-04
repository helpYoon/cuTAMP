"""Each in-loop FK site (rollout, retract, CoM, gate, and the real legs' retract_motion) has its own
never-reset cuRobo Kinematics (``TAMPWorld.site_kinematics``)."""

import pytest
import torch

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


@needs_cuda
def test_site_kinematics_are_dedicated():
    from cutamp.tamp_world import FK_SITES
    world = make_blocks_t1_world()
    kins = [world.site_kinematics(s) for s in FK_SITES]
    assert len({id(k) for k in (*kins, world.kinematics, world.kinematics_jac)}) == len(FK_SITES) + 2 == 7
    assert all(world.site_kinematics(s) is k for s, k in zip(FK_SITES, kins))
    assert all(k.config is world.kinematics.config and k.compute_com and not k.compute_jacobian for k in kins)
    with pytest.raises(ValueError, match="rollout"):
        world.site_kinematics("refresh")


@needs_cuda
def test_same_shape_sites_keep_their_grads():
    # Two same-shape FK calls on one instance before backward corrupt the first's gradient; sites keep them apart.
    from curobo.kinematics import Kinematics
    from curobo.types import JointState
    world = make_blocks_t1_world()
    torch.manual_seed(0)
    q1, q2 = [(world.q_init + 0.4 * torch.randn(48, world.q_init.numel(), device="cuda")).requires_grad_()
              for _ in range(2)]

    def fk_loss(kin, q):
        ks = kin.compute_kinematics(JointState.from_position(q))
        return ks.robot_spheres.square().sum() + ks.robot_com[..., :3].square().sum()

    solo1 = torch.autograd.grad(fk_loss(world.site_kinematics("rollout"), q1), q1)[0].clone()
    solo2 = torch.autograd.grad(fk_loss(world.site_kinematics("retract"), q2), q2)[0].clone()
    g1, g2 = torch.autograd.grad(
        fk_loss(world.site_kinematics("rollout"), q1) + fk_loss(world.site_kinematics("retract"), q2), (q1, q2))
    assert torch.equal(g1, solo1) and torch.equal(g2, solo2)
    one = Kinematics(world.kinematics.config, compute_com=True)  # positive control: both calls on one instance
    assert torch.equal(torch.autograd.grad(fk_loss(one, q1), q1)[0], solo1)
    h1, _ = torch.autograd.grad(fk_loss(one, q1) + fk_loss(one, q2), (q1, q2))
    assert not torch.equal(h1, solo1)
