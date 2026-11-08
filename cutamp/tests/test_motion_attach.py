"""Tests for _attach_object / _detach_object: where the held block's collision
spheres go, and which blocks stay world obstacles."""
import torch
from curobo.types import JointState

import cutamp.motion_solver as ms
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def _home_js(world):
    """``world.q_init`` as a ``[1, n]`` planner JointState."""
    return JointState.from_position(world.q_init.reshape(1, -1), joint_names=list(world.model.joint_names))


@needs_cuda
def test_attached_spheres_sit_on_the_block():
    # Attach each arm's block at q, then FK at the same q: the link-local
    # spheres must land back on the block, not ~0.5 m off the hand.
    world = make_blocks_t1_world()
    planner = ms.build_motion_planner(world)
    kp = planner.attachment_manager.kinematics_params
    js = _home_js(world)
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
    js = _home_js(world)
    ms._attach_object(planner, "block3", js, "left")
    ms._attach_object(planner, "block2", js, "right")

    def enabled():
        return {b: bool(cub.enable[0, cub.get_idx(b)]) for b in ("block2", "block3")}

    ms._detach_object(planner, "block3", "left")
    assert enabled() == {"block2": False, "block3": True}, "after LeftPlace"
    ms._detach_object(planner, "block2", "right")
    assert enabled() == {"block2": True, "block3": True}, "after RightPlace"


def _cup_world():
    """cup_t1 at its run height (0.5 m), with blocks_t1's block3 added on the floor for the other hand. Attaching needs
    no grasps, so the world is built without GraspGenX's."""
    from curobo.types import DeviceCfg
    from cutamp.envs.utils import load_env, raise_surface_height, resolve_env_path
    from cutamp.robots import REAL_LEGS, load_t1_container
    from cutamp.tamp_world import TAMPWorld
    env = raise_surface_height(load_env(resolve_env_path("cup_t1")), 0.5)
    block = next(o for o in load_env(resolve_env_path("blocks_t1")).movables if o.name == "block3")
    env.movables.append(block)
    env.type_to_objects["Movable"].append(block)
    dc = DeviceCfg()
    q_init = torch.as_tensor(REAL_LEGS.home, dtype=torch.float32, device=dc.device)
    return TAMPWorld(env=env, device_cfg=dc, robot=load_t1_container(REAL_LEGS, dc), q_init=q_init, grasps={})


def _enabled(planner) -> dict:
    """Each scene obstacle's collision flag, by name: the cuboids' and the meshes'."""
    data = planner.scene_collision_checker.data
    return {name: bool(store.enable[0, store.get_idx(name)]) for store in (data.cuboids, data.meshes)
            for name in store.get_names(0)}


@needs_cuda
def test_held_cup_sits_in_its_hull_and_its_place_reenables_only_it():
    # _attach_object's Mesh branch: the cup's spheres are fit on its convex hull. After the link-local rewrite, FK
    # at the attach conf puts every fitted sphere inside the cup's hull. The cup is out of the scene while held, and
    # its place re-enables it alone: the right hand's block stays out until its own place.
    world = _cup_world()
    planner = ms.build_motion_planner(world)
    kp = planner.attachment_manager.kinematics_params
    js, before = _home_js(world), _enabled(planner)
    assert set(before) >= {"cup", "block3", "table", "goal", "floor"} and all(before.values())
    ms._attach_object(planner, "cup", js, "left")
    ms._attach_object(planner, "block3", js, "right")
    idx = kp.get_sphere_index_from_link_name(ms._ATTACH_LINK_FOR_ARM["left"])
    live = kp.link_spheres[0, idx, 3] > 0
    spheres = planner.kinematics.compute_kinematics(js).robot_spheres.view(-1, 4)[idx][live].double().cpu().numpy()
    cup = next(o for o in world.movables if o.name == "cup")
    hull = cup.get_trimesh_mesh(transform_with_pose=True).convex_hull
    # A sphere inside a convex hull is at least its radius inside every face's plane.
    out = ((spheres[:, None, :3] - hull.triangles[None, :, 0]) * hull.face_normals[None]).sum(-1) + spheres[:, 3:]
    assert len(spheres) and out.max() <= 1e-4, f"{len(spheres)} spheres, up to {1000 * out.max():.2f} mm outside"
    assert _enabled(planner) == {**before, "cup": False, "block3": False}, "while held"
    ms._detach_object(planner, "cup", "left")
    assert _enabled(planner) == {**before, "block3": False}, "after LeftPlace"
    ms._detach_object(planner, "block3", "right")
    assert _enabled(planner) == before, "after RightPlace"
