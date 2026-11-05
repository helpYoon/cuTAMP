"""Static stability margin: min-axis distance to the support-rectangle edge."""
import torch

from cutamp.com_polygon_cost import COM_CENTER, COM_HALF_EXTENTS, com_margin_from_base


def test_margin_center_edge_outside():
    hx, hy = COM_HALF_EXTENTS
    pts = torch.tensor([
        [0.0, 0.0],          # center: margin = min(hx, hy) = hx
        [hx, 0.0],           # on the fore edge: 0
        [0.0, hy + 0.03],    # 3 cm outside laterally: -0.03
        [hx - 0.01, hy - 0.05],  # 1 cm from fore edge, 5 cm from side: 0.01
    ])
    m = com_margin_from_base(pts, COM_HALF_EXTENTS)
    assert torch.allclose(m, torch.tensor([hx, 0.0, -0.03, 0.01]), atol=1e-6)


def test_margin_sign_matches_polygon_mask_semantics():
    hx, hy = COM_HALF_EXTENTS
    inside = torch.tensor([[hx - 1e-3, 0.0]])
    outside = torch.tensor([[hx + 1e-3, 0.0]])
    assert com_margin_from_base(inside, COM_HALF_EXTENTS).item() > 0
    assert com_margin_from_base(outside, COM_HALF_EXTENTS).item() < 0


def test_support_rectangle_is_shrunk_sole():
    # The real soles (t1.urdf foot boxes, legs together) minus the 2 cm safety margin on every side: every
    # trajectory frame's CoM and ZMP stay in it. Keyframe CoMs sit 5 mm further in, so an arrival can
    # decelerate without its ZMP leaving.
    import pytest

    from cutamp.com_polygon_cost import SUPPORT_HALF_EXTENTS
    assert SUPPORT_HALF_EXTENTS == pytest.approx((0.0915, 0.13625))
    assert COM_HALF_EXTENTS == pytest.approx((0.0865, 0.13125))


def test_projection_subtracts_center():
    # The soles span x [-0.1015, +0.1215] about the ankle axis: the rectangle is centred 1 cm ahead
    # of mobile_base_link, so its front edge is at x 0.1015 and its back edge at -0.0815.
    from cutamp.com_polygon_cost import SUPPORT_HALF_EXTENTS, _project_com_to_base
    base_T = torch.eye(4).expand(3, 4, 4)
    com = torch.tensor([[0.01, 0.0, 0.5], [0.1015 + 1e-4, 0.0, 0.5], [-0.0815 + 1e-4, 0.0, 0.5]])
    c = _project_com_to_base(com, base_T, COM_CENTER)
    assert torch.allclose(c[0], torch.zeros(2), atol=1e-7)
    m = com_margin_from_base(c, SUPPORT_HALF_EXTENTS)
    assert m[1].item() < 0 and m[2].item() > 0


def test_support_mask_uses_the_gate_rectangle(monkeypatch):
    # The trajectory audit counts a frame as outside only when it leaves the rectangle the gate enforces
    # (SUPPORT_HALF_EXTENTS), not the keyframes' tighter one: x 8.9 cm from the centre is inside.
    from types import SimpleNamespace

    import cutamp.com_polygon_cost as cpc
    from cutamp.robots.t1 import SIMPLIFIED
    com = torch.tensor([[0.089 + 0.01, 0.0, 0.5], [0.092 + 0.01, 0.0, 0.5], [0.01, 0.137, 0.5]])
    monkeypatch.setattr(cpc, "_fk_com_and_base", lambda world, q: (com, torch.eye(4).expand(3, 4, 4)))
    world = SimpleNamespace(model=SIMPLIFIED)
    assert cpc.compute_support_mask(world, torch.zeros(3, 1)).tolist() == [True, False, False]


def test_margin_takes_any_half_extents_sequence():
    # A list or a tensor gives the tuple's margin, and the same values again add no cached device tensor.
    import cutamp.com_polygon_cost as cpc
    pts = torch.tensor([[0.01, 0.02], [0.2, 0.0]])
    want = com_margin_from_base(pts, COM_HALF_EXTENTS)
    for half in (list(COM_HALF_EXTENTS), torch.tensor(COM_HALF_EXTENTS)):
        assert torch.equal(com_margin_from_base(pts, half), want)
    cached = cpc._half_extents.cache_info().currsize
    for half in (list(COM_HALF_EXTENTS), torch.tensor(COM_HALF_EXTENTS)):
        com_margin_from_base(pts, half)
    assert cpc._half_extents.cache_info().currsize == cached
