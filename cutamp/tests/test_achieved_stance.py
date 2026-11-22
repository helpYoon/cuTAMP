import torch

from cutamp.tests.conftest import needs_cuda


@needs_cuda
def test_stepped_pick_reports_signed_self_collision_and_its_gradient(real_world):
    from cutamp.config import TAMPConfiguration
    from cutamp.cost_function import CostFunction
    from cutamp.t1_domain import LeftPick
    from cutamp.tests.test_t1_real_legs_model import hips_crossed

    obj = real_world.movables[0].name
    pick = LeftPick.ground({"obj": obj, "grasp": "g", "q": "q"})
    grasp = real_world.grasps[obj].grasps[:1]
    costs = CostFunction([pick], real_world, TAMPConfiguration())
    home, crossed = real_world.q_init.flatten(), hips_crossed(real_world)

    def self_overlap(poses):
        values = costs.stepped_scene_costs({"q": poses, "g": grasp.expand(len(poses), -1)}, [pick])["values"]
        assert "self_collision" in values
        return values["self_collision"][:, 0]

    q = torch.stack([home, crossed]).requires_grad_()
    overlap = self_overlap(q)
    assert overlap[0] < 0 < overlap[1]
    grad = torch.autograd.grad(overlap[1], q)[0][1]
    joint = grad.abs().argmax()
    assert abs(grad[joint]) > 1e-3
    delta = torch.zeros_like(home)
    delta[joint] = 1e-4
    plus, minus = self_overlap(torch.stack([crossed + delta, crossed - delta]))
    torch.testing.assert_close((plus - minus) / 2e-4, grad[joint], rtol=0.02, atol=2e-4)
    particles = {"q": q, "g": grasp.expand(len(q), -1)}
    plain = costs.stepped_scene_costs(particles, [pick])["values"]
    sited = costs.stepped_scene_costs(particles, [pick], site="stepped_scene")["values"]
    assert sited.keys() == plain.keys()
    for name, value in plain.items():
        torch.testing.assert_close(sited[name], value, rtol=0, atol=0)


def test_guard_transports_physical_stance_and_matches_home_region():
    from cutamp.com_polygon_cost import margin_from_support_planes, support_rectangles, support_region_planes

    soles = torch.tensor([[[0.0, 0.10, 0.0], [0.0, -0.10, 0.0]]], dtype=torch.float64)
    _, half = support_rectangles((0.22, 0.10), 0.10, 0.10)
    x = torch.tensor([[0.02, 0.07]], dtype=soles.dtype)
    margin = margin_from_support_planes(x, *support_region_planes(soles, (0.22, 0.10), True))
    torch.testing.assert_close(margin, torch.tensor([min(half[0] - 0.02, half[1] - 0.07)], dtype=x.dtype))
    moved = soles.clone()
    moved[..., :2] += 0.35
    torch.testing.assert_close(
        margin, margin_from_support_planes(x + 0.35, *support_region_planes(moved, (0.22, 0.10), True))
    )


def test_real_step_grid_has_model_fingerprint_and_supports_paired_seed():
    from cutamp.step_bound import planner_step_bound

    b = planner_step_bound(torch.device("cpu"), torch.float32)
    c = torch.tensor([[0.12, 0.21, 0.0], [0.12, -0.21, 0.0]])
    sides = torch.tensor([True, False])
    assert (b.g(c, sides) <= 0).all()


def test_hull_area_centroid_matches_independent_polygon_and_command_derivative():
    import numpy as np
    from scipy.spatial import ConvexHull
    from cutamp.com_polygon_cost import physical_support_centroid

    box = (0.22, 0.10)
    stance = torch.tensor(
        [
            [[0.24, 0.15, 0.8], [-0.12, -0.07, -0.17]],
            [[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]],
            [[0.3, -0.1, 0.2], [0.3, -0.1, 0.2]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    def centroid(s):
        corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * np.array(box) / 2
        pts = []
        for x, y, yaw in s:
            c, t = np.cos(yaw), np.sin(yaw)
            pts.extend([x, y] + corners @ np.array([[c, t], [-t, c]]))
        hull = np.array(pts)[ConvexHull(pts).vertices]
        next_v = np.roll(hull, -1, axis=0)
        cross = hull[:, 0] * next_v[:, 1] - hull[:, 1] * next_v[:, 0]
        return ((hull + next_v) * cross[:, None]).sum(0) / (3 * cross.sum())

    computed = physical_support_centroid(stance, box)
    pts = stance.detach().numpy()
    np.testing.assert_allclose(computed.detach().numpy(), np.stack([centroid(s) for s in pts]), atol=1e-10)
    assert (computed[0] - stance[0, :, :2].mean(0)).norm() > 0.001
    direction = torch.tensor([[0.1, -0.15, 0.3], [-0.05, 0.1, -0.2]], dtype=stance.dtype)
    measured = (centroid(pts[0] + 1e-5 * direction.numpy()) - centroid(pts[0] - 1e-5 * direction.numpy())) / 2e-5
    jac = torch.stack([torch.autograd.grad(v, stance, retain_graph=True)[0][0] for v in computed[0]])
    np.testing.assert_allclose((jac * direction).sum((1, 2)).numpy(), measured, atol=1e-8)
