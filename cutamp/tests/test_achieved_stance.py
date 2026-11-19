import torch


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
    torch.testing.assert_close(b.hard_bound(c * torch.tensor([0.0, 1.0, 1.0]), c, sides), c)


def test_geometric_sweep_rejects_blocked_side_before_the_second_landing():
    from types import SimpleNamespace
    from curobo.scene import Cuboid
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.sole_collision import geometric_sweep_rows

    obstacle = Cuboid(name="brick", pose=[0.19, 0.10, 0.02, 1, 0, 0, 0], dims=[0.04, 0.04, 0.04])
    world = SimpleNamespace(model=REAL_LEGS, statics=[obstacle], q_init=torch.zeros(33))
    feet = torch.tensor([[[0.0, 0.10, 0.0], [0.0, -0.10, 0.0]]])
    u = torch.tensor([[[0.30, 0.20, 0.0], [-0.05, -0.21, 0.0]]], requires_grad=True)
    rows = geometric_sweep_rows(world, {}, {})(feet, "LR", u)
    assert rows["step1_sweep"].item() > 0
    r = geometric_sweep_rows(world, {}, {})(feet, "R", torch.tensor([[[0.30, -0.20, 0.0]]]))
    assert r["step1_sweep"].item() < 0
    assert torch.autograd.grad(rows["step1_sweep"].sum(), u)[0].isfinite().all()


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
