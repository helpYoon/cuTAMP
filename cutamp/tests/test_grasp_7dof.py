"""A grasp is 7 numbers: G's pose in the object frame, [x, y, z, qw, qx, qy, qz]."""

import pytest
import roma
import torch

from cutamp.robots.t1 import LEFT_TOOL_FRAME, get_t1_tool_from_ee, load_sweep_volume
from cutamp.utils.common import (
    action_4dof_to_mat4x4,
    grasp_7dof_to_mat4x4,
    mat4x4_to_grasp_7dof,
    pose_list_to_mat4x4,
)


def test_quaternion_order_is_curobos():
    q = roma.random_unitquat(16)  # xyzw
    wxyz = roma.quat_xyzw_to_wxyz(q)
    p = torch.randn(16, 3)
    m = grasp_7dof_to_mat4x4(torch.cat([p, wxyz], -1))
    ref = torch.stack([pose_list_to_mat4x4([*p[i].tolist(), *wxyz[i].tolist()]) for i in range(16)])
    assert torch.allclose(m, ref, atol=1e-6)


def test_round_trip():  # on a [2, 4] batch: two leading dims
    g = torch.cat([torch.randn(2, 4, 3), roma.quat_xyzw_to_wxyz(roma.random_unitquat((2, 4)))], -1)
    m = grasp_7dof_to_mat4x4(g)
    assert torch.allclose(grasp_7dof_to_mat4x4(mat4x4_to_grasp_7dof(m)), m, atol=1e-6)


def test_widths_are_checked():
    with pytest.raises(ValueError):
        grasp_7dof_to_mat4x4(torch.zeros(3, 4))
    with pytest.raises(ValueError):
        action_4dof_to_mat4x4(torch.zeros(3, 7))


def test_frame_pins_the_top_down_grasp():  # design 7.8's frame test, 5 cm block
    sv, T = load_sweep_volume(), get_t1_tool_from_ee()[LEFT_TOOL_FRAME].cpu()
    R = torch.tensor([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])  # columns: X_G = +y_obj (closing), Y_G = +x_obj, Z_G = -z_obj (down)
    obj_from_G = torch.eye(4)
    obj_from_G[:3, :3] = R
    obj_from_G[:3, 3] = torch.tensor([0, 0, 0.005]) - R @ torch.tensor(sv["offset_open"])  # open box centre 2 cm below the top
    obj_from_ee = obj_from_G @ T
    # Ry(+90 deg), today's yaw-0 hand rotation. Its column 1, the closing axis, is +y_obj: the check a visual
    # inspection misses, since upstream's Franka frame needs a 90 deg turn about Z.
    assert torch.allclose(obj_from_ee[:3, :3], torch.tensor([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]), atol=1e-6)
    old_point = obj_from_ee @ torch.tensor([0.090, 0, -0.010, 1])  # today's grasp point, in the hand frame
    assert (old_point[:3] - torch.tensor([0, 0, 0.005])).norm() < 3e-3  # the probes put it 2.2 mm from the TCP
