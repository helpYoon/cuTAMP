"""Tests for _attach_object / _detach_object: where the held block's collision
spheres go, and which blocks stay world obstacles."""
import cutamp.motion_solver as ms
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


@needs_cuda
def test_attached_spheres_sit_on_the_block():
    # Attach each arm's block at q, then FK at the same q: the link-local
    # spheres must land back on the block, not ~0.5 m off the hand.
    world = make_blocks_t1_world()
    planner = world.get_motion_planner()
    kp = ms._get_attachment_manager(planner).kinematics_params
    js = ms._planner_js_from_full(planner, world.q_init, world.kinematics.joint_names)
    for arm, block in (("left", "block3"), ("right", "block2")):
        ms._attach_object(planner, block, js, arm)
        idx = kp.get_sphere_index_from_link_name(ms._ATTACH_LINK_FOR_ARM[arm])
        live = kp.link_spheres[0, idx, 3] > 0
        assert live.any(), f"{arm}: no spheres attached"
        spheres = planner.kinematics.compute_kinematics(js).robot_spheres.view(-1, 4)[idx][live]
        err = float((spheres[:, :3].mean(0) - world.get_object_pose(block)[:3, 3]).norm())
        assert err < 0.01, f"{arm}: attached spheres are {err:.3f} m from {block}"

