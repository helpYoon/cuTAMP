"""Tests for _attach_object / _detach_object: where the held block's collision
spheres go, and which blocks stay world obstacles."""
import cutamp.motion_solver as ms
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


@needs_cuda
def test_attached_spheres_sit_on_the_block():
    # Attach each arm's block at q, then FK at the same q: the link-local
    # spheres must land back on the block, not ~0.5 m off the hand.
    world = make_blocks_t1_world()
    planner = ms.build_motion_planner(world)
    kp = planner.attachment_manager.kinematics_params
    js = ms._planner_js_from_full(planner, world.q_init, world.kinematics.joint_names)
    for arm, block in (("left", "block3"), ("right", "block2")):
        ms._attach_object(planner, block, js, arm)
        idx = kp.get_sphere_index_from_link_name(ms._ATTACH_LINK_FOR_ARM[arm])
        live = kp.link_spheres[0, idx, 3] > 0
        assert live.any(), f"{arm}: no spheres attached"
        spheres = planner.kinematics.compute_kinematics(js).robot_spheres.view(-1, 4)[idx][live]
        err = float((spheres[:, :3].mean(0) - world.get_object_pose(block)[:3, 3]).norm())
        assert err < 0.01, f"{arm}: attached spheres are {err:.3f} m from {block}"


@needs_cuda
def test_place_reenables_only_the_placed_block():
    # Both arms hold a block, as after RightPick. Placing the left one must
    # re-enable block3 only: block2 is still in the right hand.
    world = make_blocks_t1_world()
    planner = ms.build_motion_planner(world)
    cub = planner.scene_collision_checker.data.cuboids
    js = ms._planner_js_from_full(planner, world.q_init, world.kinematics.joint_names)
    ms._attach_object(planner, "block3", js, "left")
    ms._attach_object(planner, "block2", js, "right")

    def enabled():
        return {b: bool(cub.enable[0, cub.get_idx(b)]) for b in ("block2", "block3")}

    ms._detach_object(planner, "block3", "left")
    assert enabled() == {"block2": False, "block3": True}, "after LeftPlace"
    ms._detach_object(planner, "block2", "right")
    assert enabled() == {"block2": True, "block3": True}, "after RightPlace"
