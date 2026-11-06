"""Every balance check on the model's support (design 7.4, call 5): the masks, margins, penalties and the multibody
ZMP measure the CoM in the stance frame, against S_a inside the soles' hull shrunk by 2 cm. The GPU tests share
conftest's module-scoped ``real_world``: one world per module."""

import torch

from cutamp.tests.conftest import needs_cuda


def _home(world) -> torch.Tensor:
    return torch.tensor(world.model.home, device=world.device).view(1, -1)


def _mid_sole_xy(world) -> torch.Tensor:
    """The home mid-sole, world xy: the midpoint of the two planted soles."""
    return world.stance_soles[:, :2, 3].mean(0)


@needs_cuda
def test_home_margin_on_the_stance(real_world):
    # Home's CoM sits 1.20 cm ahead of the mid-sole, centred in y: 8.745 - 1.20 cm to the keyframes' front edge.
    from cutamp.com_polygon_cost import compute_com_margin, compute_com_polygon_mask
    home = _home(real_world)
    margin = compute_com_margin(real_world, home).item()
    assert abs(margin - 0.0754) <= 5e-4, margin
    assert bool(compute_com_polygon_mask(real_world, home).item())


@needs_cuda
def test_mask_matches_a_brute_force_rectangle(real_world):
    # The mask against the keyframes' rectangle about the home mid-sole, measured in world xy: the stance frame is
    # the mid-sole, unrotated, whatever the floating base does.
    from cutamp.com_polygon_cost import _fk_state, compute_com_polygon_mask
    lo, hi = real_world.kinematics.get_joint_limits().position
    u = torch.rand(512, lo.shape[0], generator=torch.Generator().manual_seed(0)).to(lo)
    q = (lo + u * (hi - lo)).contiguous()
    mask = compute_com_polygon_mask(real_world, q)
    com_xy = _fk_state(real_world, q).robot_com[..., :2].reshape(-1, 2)
    offset = (com_xy - _mid_sole_xy(real_world)).abs()
    half = offset.new_tensor(real_world.model.support.keyframe_half)
    inside = (offset <= half).all(-1)
    clear = ((offset - half).abs() > 1e-6).all(-1)  # rows on an edge, to float error, may go either way
    assert int(clear.sum()) >= 500 and 0 < int(inside[clear].sum()) < int(clear.sum()), (clear.sum(), inside.sum())
    assert torch.equal(mask[clear], inside[clear])


@needs_cuda
def test_zmp_frame_is_the_stance(real_world):
    # At rest the multibody ZMP is the CoM, and both are measured from the stance: the home mid-sole, z = 0 on the
    # sole plane. The stance is massless, so it moves neither.
    from cutamp.com_polygon_cost import _fk_state, com_zmp_in_support
    home = _home(real_world)
    com, zmp = com_zmp_in_support(real_world, home.expand(3, 3, -1).contiguous(), 0.01)
    want = _fk_state(real_world, home).robot_com[..., :2].reshape(1, 2) - _mid_sole_xy(real_world)
    torch.testing.assert_close(com, want.expand(3, 2), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(zmp, com, rtol=0.0, atol=1e-6)
    bodies = real_world.bodies
    assert bodies.kinematics.tool_frames[0] == "stance" and float(bodies.mass[0]) == 0.0
