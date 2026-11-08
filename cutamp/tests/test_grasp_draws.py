"""Init draws each row's grasp from its object's GraspGenX grasps (design 7.8): the load-time drops of grasps whose
hand spheres reach below the object's bottom face or touch the object, per arm; the approach cone; the redraw of the
rows whose IK failed, at picks and places; and the place yaw that follows the held grasp. The GPU tests share
conftest's module-scoped ``real_world``."""

import math
from types import SimpleNamespace

import pytest
import torch
from curobo.scene import Cuboid
from curobo.types import DeviceCfg

from cutamp.grasps import AZIMUTH_CONE, VERTICAL_CONE, GraspSet, approach_in_cone, drop_grasps
from cutamp.robots.t1 import get_t1_gripper_spheres, load_sweep_volume
from cutamp.tests.conftest import default_checker, needs_cuda
from cutamp.utils.common import action_4dof_to_mat4x4, grasp_7dof_to_mat4x4, mat4x4_to_grasp_7dof
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
    # The left hand keeps 242 of blocks_t1's 4,100 grasps and the right 241; 7 are kept for one hand only (2026-11-07).
    assert int((keeps["left"] ^ keeps["right"]).sum()) > 0


def test_cone():
    # blocks_t1's stance and block3: a horizontal approach must come from the robot's side, within AZIMUTH_CONE of
    # the direction from the stance to the object; a near-vertical one may come from anywhere.
    stance, obj = torch.tensor([0.0101, 0.0]), torch.tensor([0.36, 0.32])
    away = math.atan2(0.32, 0.36 - 0.0101)

    def horizontal(azimuth):
        return math.cos(azimuth), math.sin(azimuth), 0.0

    def tilted(tilt, azimuth):  # ``tilt`` from straight down, leaning toward ``azimuth``
        return math.sin(tilt) * math.cos(azimuth), math.sin(tilt) * math.sin(azimuth), -math.cos(tilt)

    cases = {
        "from the far side": (horizontal(away + math.pi), False),
        "from the robot's side": (horizontal(away), True),
        "vertical": ((0.0, 0.0, -1.0), True),
        "from below": ((0.0, 0.0, 1.0), False),
        "just inside the cone": (horizontal(away + AZIMUTH_CONE - 0.02), True),
        "just outside it": (horizontal(away - AZIMUTH_CONE - 0.02), False),
        "near vertical, toward the robot": (tilted(VERTICAL_CONE - 0.02, away + math.pi), True),
        "past near vertical, toward the robot": (tilted(VERTICAL_CONE + 0.02, away + math.pi), False),
    }
    world_from_G = torch.eye(4).repeat(len(cases), 1, 1)
    for i, (z_axis, _) in enumerate(cases.values()):  # only Z_G, the approach, matters
        world_from_G[i, :3, 2] = torch.tensor(z_axis)
    got = approach_in_cone(world_from_G, stance, obj)
    assert dict(zip(cases, got.tolist())) == {name: want for name, (_, want) in cases.items()}


def _first_plan(world):
    """``(skeleton, obj, arm)``: blocks_t1's first task plan, and its first pick's object and arm."""
    from cutamp.t1_domain import all_t1_operators
    from cutamp.task_planning import task_plan_generator

    skeleton = next(task_plan_generator(world.initial_state, world.goal_state, all_t1_operators))
    pick = next(op for op in skeleton if op.operator.metadata.action_type == "pick")
    return skeleton, pick.values[0], pick.operator.metadata.arm


@needs_cuda
def test_no_grasp_left_names_the_rules(real_world, monkeypatch):
    from cutamp.config import TAMPConfiguration
    from cutamp.particle_initialization import ParticleInitializer

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
    with pytest.raises(ValueError) as err:
        ParticleInitializer(world, TAMPConfiguration(num_particles=8), default_checker())(skeleton)
    message = str(err.value)
    assert obj in message and f"{arm} hand" in message, message
    assert "2 reach below" in message and "3 touch" in message, message


@needs_cuda
def test_redraw_only_failed_rows(real_world, monkeypatch):
    # Every op's first gn_ik call fails rows 0-9 and a redraw's call accepts every row, so each pick and place redraws
    # once: rows 0-9 take the redraw's grasp or placement and conf, and rows 10 and up keep the first draw's.
    import cutamp.particle_initialization as pi
    from cutamp.config import TAMPConfiguration

    n, failed, calls, real_gn_ik = 32, 10, [], pi.gn_ik

    def spy(world, world_from_ee, arms, movables, tol):
        q, ok = real_gn_ik(world, world_from_ee, arms, movables, tol)
        ok = torch.arange(len(ok), device=ok.device) >= failed if len(ok) == n else torch.ones_like(ok)
        calls.append((world_from_ee.clone(), q.clone()))
        return q, ok

    monkeypatch.setattr(pi, "gn_ik", spy)
    skeleton, _, _ = _first_plan(real_world)
    torch.manual_seed(0)
    particles = pi.ParticleInitializer(real_world, TAMPConfiguration(num_particles=n), default_checker())(skeleton)
    ops = [op for op in skeleton if op.operator.metadata.action_type in ("pick", "place")]
    assert len(calls) == 2 * len(ops)
    for k, op in enumerate(ops):
        (first, q_first), (redraw, q_redraw) = calls[2 * k], calls[2 * k + 1]
        assert redraw.shape[0] == failed, op  # only the failed rows were solved again
        obj, grasp, q = op.values[0], op.values[1], op.values[-1]
        if op.operator.metadata.action_type == "pick":
            world_from_obj = real_world.get_object_pose(obj)
        else:
            world_from_obj = action_4dof_to_mat4x4(particles[op.values[2]])
        tool_from_ee = real_world.tool_from_ee[real_world.tool_frame_for_arm(op.operator.metadata.arm)]
        targets = world_from_obj @ grasp_7dof_to_mat4x4(particles[grasp]) @ tool_from_ee
        torch.testing.assert_close(targets[failed:], first[failed:])
        torch.testing.assert_close(targets[:failed], redraw)
        assert not torch.allclose(redraw, first[:failed]), op  # the failed rows drew anew
        assert torch.equal(particles[q][failed:], q_first[failed:]), op
        assert torch.equal(particles[q][:failed], q_redraw), op


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
