import numpy as np

from cutamp.tests.conftest import needs_cuda


@needs_cuda
def test_crouch_witnesses_keep_both_feet_planted_and_lower_the_shoulders():
    from cutamp.reachability_fk import T1MapSampler

    sampler = T1MapSampler(samples=128, seed=3)
    soles = sampler.home_soles.copy()
    soles[:, :2] -= soles[:, :2].mean(0)
    batches = sampler.map_stance(soles)
    left = {b["posture"]: b for b in batches if b["arm"] == "left" and len(b["points"])}
    assert "standing" in left and len(left) > 1
    heights = {}
    for posture, batch in left.items():
        assert batch["margin"].min() >= .05
        q = batch["q"][0]
        marks = sampler.landmarks(q)
        np.testing.assert_allclose(marks["left_arm"][-1], batch["points"][0], atol=1e-5)
        for side, expected in zip(("left", "right"), soles):
            np.testing.assert_allclose(marks[side + "_leg"][-1], [*expected[:2], 0], atol=3e-4)
        heights[posture] = marks["left_arm"][0, 2]
    assert heights["standing"] - min(heights.values()) > .06
