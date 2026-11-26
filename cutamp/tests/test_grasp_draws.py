"""A plan's grasps come from its objects' GraspGenX grasps (design 7.8): the load-time drops of grasps whose hand
spheres reach below the object's bottom face or touch the object, per arm; each pick's pool of every grasp the hand can
take; and the place yaw that follows the held grasp. The GPU tests share conftest's module-scoped ``real_world``."""

import math
from types import SimpleNamespace

import pytest
import torch
from curobo.scene import Cuboid
from curobo.types import DeviceCfg

from cutamp.grasps import AZIMUTH_CONE, GraspSet, drop_grasps
from cutamp.robots.t1 import get_t1_gripper_spheres, load_sweep_volume
from cutamp.tests.conftest import needs_cuda
from cutamp.utils.common import grasp_7dof_to_mat4x4, mat4x4_to_grasp_7dof
from cutamp.utils.shapes import sample_greedy_surface_spheres

CPU = DeviceCfg(device=torch.device("cpu"))

#: Each hand's ``*_hand_link`` and wrist (AL6 / AR6) sphere in its ``*_base_link`` frame at home (Hand_Roll 0), worked
#: out by hand from t1_planner.urdf (the fixed joint at (0, ±0.084, 0), yaw ±90 deg, and Hand_Roll's origin) and
#: t1_29dof_spheres.yml's radii.
WRIST_SPHERES = {
    "left": [[-0.019, 0.005, 0.010, 0.035], [-0.084, 0.00475, -0.0015, 0.045]],
    "right": [[-0.019, -0.005, 0.010, 0.035], [-0.084, -0.00475, -0.0015, 0.045]],
}

BOTTOM = -0.025
"""The 5 cm block's bottom face, in its own frame."""


def _block():
    """``(block, spheres)``: blocks_t1's 5 cm block at the origin, on the CPU, with the 50 surface spheres TAMPWorld
    gives it."""
    block = Cuboid(name="block", pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dims=[0.05, 0.05, 0.05], device_cfg=CPU)
    return block, sample_greedy_surface_spheres(block, n_spheres=50, sphere_radius=0.005)


def _hand(arm: str) -> torch.Tensor:
    """``arm``'s 14 hand spheres in its tool frame, as ``TAMPWorld.hand_spheres`` builds them."""
    return torch.cat([get_t1_gripper_spheres(arm, CPU), torch.tensor(WRIST_SPHERES[arm])])


def _top_down() -> torch.Tensor:
    """``[4, 4]`` obj_from_G: a top-down grasp of the 5 cm block at yaw 0, closing along the object's y, with the open
    box's centre 2 cm below the top (test_grasp_7dof's frame test)."""
    rot = torch.tensor([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])  # columns: X_G = +y_obj, Y_G = +x_obj, Z_G = -z_obj
    pose = torch.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = torch.tensor([0, 0, 0.005]) - rot @ torch.tensor(load_sweep_volume()["offset_open"])
    return pose


def _moved(x=0.0, y=0.0, z=0.0, yaw=0.0) -> torch.Tensor:
    """``[4, 4]``: a turn by ``yaw`` about the object's z, then a shift by (``x``, ``y``, ``z``)."""
    pose = torch.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    pose[:2, :2] = torch.tensor([[c, -s], [s, c]])
    pose[:3, 3] = torch.tensor([x, y, z])
    return pose


def test_drop_below_bottom():
    # At the top-down grasp the fingertip spheres' lowest point is 13 mm below the block's centre, 12 mm above its
    # bottom face. 4 cm lower it is 53 mm below: past the bottom face, so the grasp would go into the support.
    _, spheres = _block()
    top, lower = _top_down(), _moved(z=-0.04) @ _top_down()
    for arm in ("left", "right"):
        assert drop_grasps(torch.stack([top, lower]), _hand(arm), spheres, BOTTOM).tolist() == [True, False], arm
        # The bottom rule alone (no object spheres; 4 cm lower the palm touches the top too): the lowest point,
        # z - r, against the bottom face.
        alone = drop_grasps(torch.stack([lower, _moved(z=-0.010) @ top, _moved(z=-0.015) @ top]), _hand(arm),
                            spheres[:0], BOTTOM)
        assert alone.tolist() == [False, True, False], arm


def test_drop_own_overlap():
    # Turned 45 deg about the vertical, the fingers close across the block's diagonal: 70.7 mm, wider than the finger
    # spheres' 65.1 mm opening. Its fingertips stay 12 mm above the bottom face, so only the overlap rule drops it.
    _, spheres = _block()
    top, turned = _top_down(), _moved(yaw=math.pi / 4) @ _top_down()
    for arm in ("left", "right"):
        assert drop_grasps(torch.stack([top, turned]), _hand(arm), spheres, BOTTOM).tolist() == [True, False], arm
        assert drop_grasps(turned[None], _hand(arm), spheres[:0], BOTTOM).tolist() == [True], arm


@needs_cuda
def test_drops_run_per_arm(real_world):
    # The two hands' sphere sets differ by up to 5 mm, so each arm drops its own grasps.
    world, obj = real_world, "block2"
    grasps = world.grasps[obj]
    obj_from_G = grasp_7dof_to_mat4x4(grasps.grasps)
    keeps = {}
    for arm in ("left", "right"):
        hand = world.hand_spheres(arm)
        assert torch.equal(hand[:12], world.gripper_spheres[world.tool_frame_for_arm(arm)]), arm
        torch.testing.assert_close(hand[12:].cpu(), torch.tensor(WRIST_SPHERES[arm]), atol=1e-6, rtol=0)
        keeps[arm] = drop_grasps(obj_from_G, hand, world.get_collision_spheres(obj), BOTTOM)
        candidates = world.grasp_candidates(obj, arm)
        assert torch.equal(candidates.grasps, grasps.grasps[keeps[arm]]), arm
        assert torch.equal(candidates.scores, grasps.scores[keeps[arm]]), arm
        assert world.grasp_candidates(obj, arm) is candidates, arm  # made once, then kept
    assert not torch.equal(world.hand_spheres("left"), world.hand_spheres("right"))


@needs_cuda
def test_pick_pool_is_every_grasp_the_hand_can_take(real_world):
    # No approach cone (user, 2026-11-08: "Top grasps are not always a good grasp"): each pick judges every grasp
    # grasp_candidates keeps, top or side, from any direction. With the cone, blocks_t1's pools were 222 and 221.
    from cutamp.particle_initialization import _pick_pools

    skeleton, _, _ = _first_plan(real_world)
    pools = _pick_pools(real_world, skeleton)
    picks = [op for op in skeleton if op.operator.metadata.action_type == "pick"]
    assert len(picks) == len(pools) == 2
    for op in picks:
        obj, grasp, _ = op.values
        want = real_world.grasp_candidates(obj, op.operator.metadata.arm)
        assert torch.equal(pools[grasp][0], want.grasps) and torch.equal(pools[grasp][1], want.scores), obj


def _first_plan(world):
    """``(skeleton, obj, arm)``: blocks_t1's task plan (``BLOCKS_T1_ACTIONS``), and its first pick's object and arm."""
    from cutamp.tests.conftest import BLOCKS_T1_ACTIONS, skeleton_through

    skeleton = skeleton_through(world.initial_state, world.goal_state, *BLOCKS_T1_ACTIONS)
    pick = next(op for op in skeleton if op.operator.metadata.action_type == "pick")
    return skeleton, pick.values[0], pick.operator.metadata.arm


@needs_cuda
def test_no_grasp_left_names_the_rules(real_world, monkeypatch):
    from cutamp.particle_initialization import NoGraspCandidates, _pick_pools

    world, top = real_world, _top_down()
    # Two grasps beside the block and 4 cm low reach below its bottom face without touching it; three turned ones
    # touch it without reaching below (both hands).
    below = [_moved(x=0.15, z=-0.04) @ top, _moved(y=0.15, z=-0.04) @ top]
    touching = [_moved(yaw=a) @ top for a in (math.pi / 4, 3 * math.pi / 4, -math.pi / 4)]
    bad = GraspSet(mat4x4_to_grasp_7dof(torch.stack(below + touching)).to(world.device),
                   torch.full((5,), 0.5, device=world.device))
    monkeypatch.setattr(world, "_grasps", {name: bad for name in world.grasps})
    monkeypatch.setattr(world, "_grasp_candidates", {})
    skeleton, obj, arm = _first_plan(world)
    with pytest.raises(NoGraspCandidates) as err:
        _pick_pools(world, skeleton)
    message = str(err.value)
    assert obj in message and f"{arm} hand" in message, message
    assert "5 grasps" in message and "each reaches below its bottom face or touches it" in message, message


def test_place_yaw_follows_the_grasp():
    # blocks_t1's goal pad and stance. A side grasp whose approach Z_G is +x_obj approaches the placed object along the
    # placement's yaw, which must lie within AZIMUTH_CONE of the direction from the stance to the placement. A
    # top-down grasp's yaw stays uniform. The scene is empty (zero collision), so each row keeps its first candidate.
    from cutamp.particle_initialization import _draw_placements

    block, spheres = _block()
    goal = Cuboid(name="goal", pose=[0.55, 0.0, 0.005, 1.0, 0.0, 0.0, 0.0], dims=[0.45, 0.45, 0.01], device_cfg=CPU)
    soles = torch.eye(4).repeat(2, 1, 1)
    soles[:, :2, 3] = torch.tensor([[0.0101, 0.10625], [0.0101, -0.10625]])
    world = SimpleNamespace(stance_soles=soles, collision_fn=lambda s: s.new_zeros(s.shape[:2]))
    side = torch.eye(4)
    side[:3, :3] = torch.tensor([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])  # columns: X_G = +y_obj, Y_G = +z_obj, Z_G = +x_obj
    side[:3, 3] = torch.tensor([-0.1, 0.0, 0.0])
    n = 256
    held = torch.stack([side, _top_down()]).repeat_interleave(n // 2, dim=0)
    torch.manual_seed(0)
    placements = _draw_placements(world, block, spheres, goal, held)
    assert placements.shape == (n, 4)
    toward = torch.atan2(placements[:, 1], placements[:, 0] - 0.0101)
    off = torch.remainder(placements[:, 3] - toward + math.pi, 2 * math.pi) - math.pi  # the side approach is the yaw
    assert float(off[: n // 2].abs().max()) <= AZIMUTH_CONE + 1e-5
    assert float(off[n // 2:].abs().max()) > AZIMUTH_CONE
