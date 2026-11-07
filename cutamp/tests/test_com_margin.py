"""Static stability margin: min-axis distance to the support-rectangle edge."""
import torch

from cutamp.com_polygon_cost import com_margin_from_base
from cutamp.robots.t1 import REAL_LEGS


def test_margin_center_edge_outside():
    hx, hy = REAL_LEGS.support.keyframe_half
    pts = torch.tensor([
        [0.0, 0.0],          # center: margin = min(hx, hy) = hx
        [hx, 0.0],           # on the fore edge: 0
        [0.0, hy + 0.03],    # 3 cm outside laterally: -0.03
        [hx - 0.01, hy - 0.05],  # 1 cm from fore edge, 5 cm from side: 0.01
    ])
    m = com_margin_from_base(pts, REAL_LEGS.support.keyframe_half)
    assert torch.allclose(m, torch.tensor([hx, 0.0, -0.03, 0.01]), atol=1e-6)


def test_margin_sign_matches_polygon_mask_semantics():
    hx, hy = REAL_LEGS.support.keyframe_half
    inside = torch.tensor([[hx - 1e-3, 0.0]])
    outside = torch.tensor([[hx + 1e-3, 0.0]])
    assert com_margin_from_base(inside, REAL_LEGS.support.keyframe_half).item() > 0
    assert com_margin_from_base(outside, REAL_LEGS.support.keyframe_half).item() < 0


def test_margin_takes_any_half_extents_sequence():
    # A list or a tensor gives the tuple's margin, and the same values again add no cached device tensor.
    import cutamp.com_polygon_cost as cpc
    half = REAL_LEGS.support.keyframe_half
    pts = torch.tensor([[0.01, 0.02], [0.2, 0.0]])
    want = com_margin_from_base(pts, half)
    for h in (list(half), torch.tensor(half)):
        assert torch.equal(com_margin_from_base(pts, h), want)
    cached = cpc._half_extents.cache_info().currsize
    for h in (list(half), torch.tensor(half)):
        com_margin_from_base(pts, h)
    assert cpc._half_extents.cache_info().currsize == cached
