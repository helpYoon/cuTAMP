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


def test_support_mask_uses_the_gate_rectangle(monkeypatch):
    # The trajectory audit counts a frame as outside only when it leaves the rectangle the gate enforces
    # (support.gate_half, 0.09245 x 0.135), not the keyframes' tighter one (0.08745 x 0.130): x 9.0 cm from the
    # centre is inside, x 9.3 cm and y 13.6 cm are outside.
    from types import SimpleNamespace

    import cutamp.com_polygon_cost as cpc
    com = torch.tensor([[0.090, 0.0, 0.5], [0.093, 0.0, 0.5], [0.0, 0.136, 0.5]])
    monkeypatch.setattr(cpc, "_fk_com_and_base", lambda world, q: (com, torch.eye(4).expand(3, 4, 4)))
    world = SimpleNamespace(model=REAL_LEGS)
    assert cpc.compute_support_mask(world, torch.zeros(3, 1)).tolist() == [True, False, False]


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
