import torch

from cutamp.stance import abs_pose, rel_pose
from cutamp.step_refresh import StandIn

N = 3


def incoming():
    return torch.tensor([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]).repeat(N, 1, 1)  # left sole, right sole


def standin():
    torch.manual_seed(0)
    s = StandIn(("L", "", "RL"), [None] * 3, [None] * 3, [None] * 3)
    s.relative[0] = torch.tensor([[[0.2, 0.2, 0.1], [0.0, 0.0, 0.0]]]).repeat(N, 1, 1, 1)
    s.slope[0] = torch.zeros(N, 1, 2, 3, 3)
    s.slope[0][:, 0, 0] = torch.eye(3)  # the swung sole follows its command exactly
    s.at[0] = torch.tensor([[[0.2, 0.2, 0.1]]]).repeat(N, 1, 1)
    s.relative[2] = torch.randn(N, 2, 2, 3) * 0.05
    s.slope[2] = torch.randn(N, 2, 2, 3, 6) * 0.5
    s.at[2] = torch.randn(N, 2, 3) * 0.05
    return s


def test_equal_to_the_refresh_at_the_refresh_commands():
    s, inc = standin(), incoming()
    after, final = s.stances(inc, [s.at[0], None, s.at[2]])
    for sole in range(2):
        torch.testing.assert_close(after[0][:, 0, sole], abs_pose(inc[:, 1], s.relative[0][:, 0, sole]))
        for k in range(2):  # gap 2 swings the right foot first: its anchor is the left sole it starts on
            torch.testing.assert_close(after[2][:, k, sole], abs_pose(final[1][:, 0], s.relative[2][:, k, sole]))
    torch.testing.assert_close(final[0], after[0][:, -1])
    torch.testing.assert_close(final[2], after[2][:, -1])


def test_gap_without_steps_stands_on_the_stance_before_it():
    s, inc = standin(), incoming()
    after, final = s.stances(inc, [s.at[0], None, s.at[2]])
    assert after[1] is None
    torch.testing.assert_close(final[1], final[0])


def test_later_gap_rides_on_the_earlier_gap():
    s, inc = standin(), incoming()
    _, base = s.stances(inc, [s.at[0], None, s.at[2]])
    moved_command = s.at[0] + torch.tensor([0.03, -0.02, 0.05])
    after, final = s.stances(inc, [moved_command, None, s.at[2]])
    assert not torch.allclose(final[2], base[2])
    for sole in range(2):  # seen from the moved left sole, the later gap's landing is where the refresh had it
        torch.testing.assert_close(rel_pose(final[0][:, 0], final[2][:, sole]), s.relative[2][:, -1, sole], atol=1e-6, rtol=0)


def test_gradient_reaches_every_gaps_commands():
    s, inc = standin(), incoming()
    first, third = s.at[0].clone().requires_grad_(), s.at[2].clone().requires_grad_()
    _, final = s.stances(inc, [first, None, third])
    g_first, g_third = torch.autograd.grad(final[2].sum(), (first, third))
    assert g_first.abs().sum() > 0 and g_third.abs().sum() > 0
