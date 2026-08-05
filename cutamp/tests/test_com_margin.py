"""Static stability margin: min-axis distance to the support-rectangle edge."""
import torch

from cutamp.com_polygon_cost import COM_HALF_EXTENTS, com_margin_from_base


def test_margin_center_edge_outside():
    hx, hy = COM_HALF_EXTENTS
    pts = torch.tensor([
        [0.0, 0.0],          # center: margin = min(hx, hy) = hx
        [hx, 0.0],           # on the fore edge: 0
        [0.0, hy + 0.03],    # 3 cm outside laterally: -0.03
        [hx - 0.01, hy - 0.05],  # 1 cm from fore edge, 5 cm from side: 0.01
    ])
    m = com_margin_from_base(pts)
    assert torch.allclose(m, torch.tensor([hx, 0.0, -0.03, 0.01]), atol=1e-6)


def test_margin_sign_matches_polygon_mask_semantics():
    hx, hy = COM_HALF_EXTENTS
    inside = torch.tensor([[hx - 1e-3, 0.0]])
    outside = torch.tensor([[hx + 1e-3, 0.0]])
    assert com_margin_from_base(inside).item() > 0
    assert com_margin_from_base(outside).item() < 0
