import numpy as np

from cutamp.tests.conftest import needs_cuda


@needs_cuda
def test_crouch_witnesses_keep_both_feet_planted_and_lower_the_trunk():
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
        q = sampler.cfg.to_device(batch["q"][0])
        state = sampler._fk(sampler.kin, q)
        np.testing.assert_allclose(sampler._contact(state, "left")[0].cpu(), batch["points"][0], atol=1e-5)
        feet = sampler._feet(state)[0, :, :3, 3].cpu()
        np.testing.assert_allclose(feet, np.c_[soles[:, :2], np.zeros(2)], atol=3e-4)
        heights[posture] = q[2].item()  # The floating base's z carries the Trunk.
    assert heights["standing"] - min(heights.values()) > .06


@needs_cuda
def test_planted_crouch_reaches_a_floor_resting_box_with_a_real_hand_sample():
    from cutamp.reachability_fk import T1MapSampler, _sole_transforms

    sampler = T1MapSampler(samples=16384, seed=7)
    center = np.array([.30, .18, .025])
    goals = _sole_transforms(sampler.home_soles)
    for batch in sampler.map_stance(sampler.home_soles):
        if batch["arm"] != "left":
            continue
        inside = (abs(batch["points"] - center) <= .025).all(axis=1)
        if inside.any():
            i = np.flatnonzero(inside)[0]
            q = sampler.cfg.to_device(batch["q"][i])
            state = sampler._fk(sampler.kin, q)
            assert q[3] > .1  # The validated witness uses a hip hinge.
            assert abs(q[sampler.waist_index]) > .01
            np.testing.assert_allclose(sampler._feet(state)[0].cpu(), goals, atol=3e-4)
            spheres = state.robot_spheres
            assert (spheres[..., 2] - spheres[..., 3]).min() >= 0
            assert batch["margin"][i] >= .05
            return
    raise AssertionError("No actual left-hand sample inside the 5 cm cube resting on the floor")


@needs_cuda
def test_rotated_open_finger_crossing_floor_is_filtered_for_either_query_arm(monkeypatch):
    import torch

    from cutamp.reachability_fk import T1MapSampler

    sampler = T1MapSampler(samples=2, seed=7)
    q = sampler.home.copy()
    q[:6] = [-.14233524, .02253928, .45684937, .85, 0, -.36894161]
    q[sampler.model.left_arm] = [-.01790193, -.88036394, .35706404, -.56899089, -2.20802355, .10484624, .39322078]
    q[sampler.waist_index] = -q[5]
    for side in ("Left", "Right"):
        q[sampler.joint_names.index(f"{side}_Hip_Pitch")] = -2.45
        q[sampler.joint_names.index(f"{side}_Knee_Pitch")] = 1.8
    q = sampler.cfg.to_device(q)
    state = sampler._fk(sampler.kin, q)
    spheres = state.robot_spheres
    assert (spheres[..., 2] - spheres[..., 3]).min() > 0
    # A previous lowest-hand witness: the left finger penetrates the floor by 0.885 mm.
    np.testing.assert_allclose(sampler._gripper_clearance(state, "left").item(), -.000885, atol=1e-6)
    assert sampler._gripper_clearance(state, "right").item() > 0
    monkeypatch.setattr(sampler, "_arm_samples", lambda pose, arm: torch.stack([pose, q]))
    soles = sampler.home_soles.copy()
    soles[:, :2] -= soles[:, :2].mean(axis=0)
    batches = sampler.map_stance(soles)
    assert {b["arm"] for b in batches if len(b["q"])} == {"left", "right"}
    assert all(not np.any(np.all(b["q"] == q.cpu().numpy(), axis=1)) for b in batches)
