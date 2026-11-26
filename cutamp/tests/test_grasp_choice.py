import math
from types import SimpleNamespace

import torch

import cutamp.com_polygon_cost as cpc
from cutamp.algorithm import grasp_group_order
from cutamp.particle_initialization import _BASE_JOINTS, pick_place_confs, worst_posture


def _op(kind, *values):
    return SimpleNamespace(operator=SimpleNamespace(metadata=SimpleNamespace(action_type=kind)), values=values)


def _fake_world(home=(0.0, 0.0, 0.7, 0.0, 0.0, 0.0)):  # the floating base alone
    return SimpleNamespace(model=SimpleNamespace(joint_names=_BASE_JOINTS, home=home))


def test_free_placement_projection_matches_the_final_rotated_footprint_and_support_checks():
    from cutamp.cost_function import CostFunction
    from cutamp.samplers import PLACEMENT_SUPPORT_GAP_M, project_supported_placement
    from cutamp.utils.common import action_4dof_to_mat4x4, transform_spheres

    spheres = torch.tensor([[-.13, 0, .02, .05], [.07, 0, .08, .02]])
    bounds = torch.tensor([[.5, -.06, .05], [.9, .06, .1]])
    poses = torch.tensor([[.45, .1, .15, 0.], [.9, .1, .17, math.pi], [.72, .15, .25, math.pi / 2]])
    projected = project_supported_placement(poses, spheres, bounds)
    moved = transform_spheres(spheres, action_4dof_to_mat4x4(projected))
    cost = SimpleNamespace(surface_to_placements={"pad": [("part", "p")]},
                           surface_to_aabb={"pad": bounds[:, :2]}, surface_to_target_z={"pad": .1 + PLACEMENT_SUPPORT_GAP_M},
                           surface_to_sphere_counts={"pad": (2,)}, surface_to_sphere_idx_map={"pad": torch.zeros(2, dtype=torch.long)})
    rollout = {"num_particles": 3, "action_to_pose_ts": {"p": 0}}
    checked = CostFunction.stable_placement_costs(cost, rollout, {"part": moved[:, None]})["values"]
    assert checked["pad_in_xy"][:2].max() < 1e-6
    assert checked["pad_in_xy"][2] > .07  # rotating the long axis onto the narrow side cannot fit
    assert checked["pad_support"].max() < 1e-6
    torch.testing.assert_close(projected[:, 3], poses[:, 3])


def test_pick_place_confs_lists_every_pick_and_place_conf_in_order():
    skeleton = [_op("pick", "a", "g1", "q1"), _op("pick", "b", "g2", "q2"), _op("place", "b", "g2", "p2", "s", "q3"),
                _op("place", "a", "g1", "p1", "s", "q4"), _op("retract", "q4", "t", "r1"),
                _op("pick", "a", "g3", "q5"), _op("place", "a", "g3", "p3", "s", "q6"), _op("pick", "b", "g4", "q7")]
    assert pick_place_confs(skeleton) == ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]


def test_worst_posture_takes_the_worse_conf(monkeypatch):
    monkeypatch.setattr(cpc, "compute_com_margin", lambda world, q: q[:, 5])  # the yaw stands in for the margin
    pick = torch.tensor([[0.0, 0.0, 0.5, 0.3, 0.0, 0.08], [0.0, 0.0, 0.7, 0.0, 0.0, 0.02]])
    place = torch.tensor([[0.0, 0.0, 0.7, 0.0, 0.4, 0.05], [0.3, 0.4, 0.7, 0.0, 0.0, 0.06]])
    tilt, travel, margin = worst_posture(_fake_world(), [pick, place])
    torch.testing.assert_close(tilt, torch.tensor([0.4, 0.0]))
    torch.testing.assert_close(travel, torch.tensor([0.2, 0.5]))
    torch.testing.assert_close(margin, torch.tensor([0.05, 0.02]))


def test_groups_go_cutoff_first_then_upright():
    # Per row: its group's pool indices, soft cost, all at 0.7, tilt, travel, margin.
    rows = [((0, 0), 1.0, True, 0.3, 0.0, 0.0),     # group a, last by soft cost
            ((0, 0), 0.5, True, 0.1, 0.2, 0.08),    # group a, first
            ((1, 0), 2.0, True, 0.0, 0.2, 0.08),    # group b, first: the most upright
            ((2, 1), 0.1, False, 0.0, 0.0, 0.08),   # below the cutoff: after every group that clears it
            ((1, 0), 3.0, True, 0.9, 0.0, 0.0),     # group b, second (its own tilt doesn't count)
            ((3, 0), 0.2, True, 0.2, 0.1, 0.02),
            ((4, 0), 0.3, True, 0.2, 0.1, 0.08),    # a larger margin first
            ((5, 0), 0.4, True, 0.2, 0.05, 0.02),   # less travel first
            ((0, 2), 0.6, True, 0.2, 0.1, 0.02),    # all equal: pool order, (0, 1) < (0, 2) < (3, 0)
            ((0, 1), 0.7, True, 0.2, 0.1, 0.02),
            ((0, 0), 0.5, True, 0.1, 0.2, 0.08)]    # group a, second: row 1's soft cost, so after it by index
    idx, soft, hi, tilt, travel, margin = (torch.tensor(c) for c in zip(*rows))
    assert grasp_group_order(idx, soft, hi, tilt, travel, margin, 12) == [2, 1, 7, 6, 9, 8, 5, 3, 4, 10, 0]
    assert grasp_group_order(idx, soft, hi, tilt, travel, margin, 3) == [2, 1, 7]
