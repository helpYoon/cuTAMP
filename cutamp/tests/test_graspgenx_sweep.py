from cutamp.robots.t1 import load_sweep_volume
from cutamp.scripts.graspgenx_sweep_volume import compute


def test_stored_numbers_match_the_meshes():
    assert load_sweep_volume() == compute()


def test_open_box_spans_the_open_pads():  # probes: 70.0 mm gap, pads x 0.0710-0.1045, mid-plane z -0.010
    sv = load_sweep_volume()
    assert abs(sv["extents_open"][0] - 0.070) < 5e-4 and abs(sv["offset_open"][0]) < 1e-6
    z_lo = sv["offset_open"][2] - sv["extents_open"][2] / 2
    z_hi = sv["offset_open"][2] + sv["extents_open"][2] / 2
    assert abs(z_lo - 0.0710) < 5e-4 and abs(z_hi - 0.1045) < 5e-4 and z_lo > 0.054  # clear of the palm (mesh 0.054)


def test_mid_box_spans_the_half_open_pads():
    assert abs(load_sweep_volume()["extents_mid"][0] - 0.035) < 5e-4
