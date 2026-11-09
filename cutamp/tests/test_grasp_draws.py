"""Init gives each row a grasp from its object's GraspGenX grasps (design 7.8): the load-time drops of grasps whose
hand spheres reach below the object's bottom face or touch the object, per arm; each pick's pool of every grasp the
hand can take; each pick's chosen grasps and their pool confs, or the score draw where no grasp is balanced; the
redraw of the rows whose IK failed, at picks and places; and the place yaw that follows the held grasp. The GPU tests
share conftest's module-scoped ``real_world``."""

import math
from types import SimpleNamespace

import pytest
import torch
from curobo.scene import Cuboid
from curobo.types import DeviceCfg

from cutamp.grasps import AZIMUTH_CONE, GraspSet, drop_grasps
from cutamp.robots.t1 import get_t1_gripper_spheres, load_sweep_volume
from cutamp.tests.conftest import default_checker, needs_cuda
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
    # The left hand keeps 242 of blocks_t1's 4,100 grasps and the right 241; 7 are kept for one hand only (2026-11-07).
    assert int((keeps["left"] ^ keeps["right"]).sum()) > 0


@needs_cuda
def test_pick_pool_is_every_grasp_the_hand_can_take(real_world):
    # No approach cone (user, 2026-11-08: "Top grasps are not always a good grasp"): each pick judges every grasp
    # grasp_candidates keeps, top or side, from any direction. With the cone, blocks_t1's pools were 222 and 221.
    from cutamp.config import TAMPConfiguration
    from cutamp.particle_initialization import ParticleInitializer

    skeleton, _, _ = _first_plan(real_world)
    pools = ParticleInitializer(real_world, TAMPConfiguration(num_particles=8), default_checker())._pick_pools(skeleton)
    picks = [op for op in skeleton if op.operator.metadata.action_type == "pick"]
    assert len(picks) == len(pools) == 2
    for op in picks:
        obj, grasp, _ = op.values
        want = real_world.grasp_candidates(obj, op.operator.metadata.arm)
        assert torch.equal(pools[grasp][0], want.grasps) and torch.equal(pools[grasp][1], want.scores), obj


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


def _fake_ik(monkeypatch, accept):
    """gn_ik and ik_check stand-ins. A row's targets hold its drawn value v; gn_ik answers q = (v, v), accepts
    accept(v) and records each call's values; ik_check accepts a conf whose first entry is >= 0."""
    import cutamp.particle_initialization as pi
    calls = []

    def gn_ik(world, world_from_ee, arms, movables, tol):
        v = world_from_ee[:, 0, 0]
        calls.append(v.tolist())
        return torch.stack([v, v], -1), accept(v)

    def ik_check(world, q, world_from_ee, arms, movables, tol):
        return q[:, 0] >= 0, q[:, 0] >= 0, torch.zeros(len(q))

    monkeypatch.setattr(pi, "gn_ik", gn_ik)
    monkeypatch.setattr(pi, "ik_check", ik_check)
    return calls


def _solve(particles, draw, **kw):  # four rows on the CPU
    import cutamp.particle_initialization as pi
    inputs = lambda rows: (particles["x"][rows, None, None].expand(-1, 4, 4), torch.zeros(len(rows), 1, 4))
    return pi._draw_and_solve(SimpleNamespace(device="cpu"), particles, "x", 4, draw, inputs, "left", None, **kw)


def test_redraw_only_failed_rows(monkeypatch):
    calls = _fake_ik(monkeypatch, lambda v: (v % 2 == 1) | (v >= 10))
    particles = {}
    q, ok, accepted = _solve(particles, lambda rows: rows.float() + 10 * (len(rows) < 4))
    assert calls == [[0, 1, 2, 3], [10, 12]]  # only the failed rows drew again and were solved again
    assert particles["x"].tolist() == [10, 1, 12, 3] and q[:, 0].tolist() == [10, 1, 12, 3]
    assert ok.all() and accepted == [2, 4]
    calls.clear()
    q, ok, accepted = _solve({}, lambda rows: rows.float(), rounds=0)
    assert calls == [[0, 1, 2, 3]] and ok.tolist() == [False, True, False, True] and accepted == [2]


def test_check_the_start_before_solving(monkeypatch):
    # Rows 0 and 2 pass the check and keep their start conf; gn_ik solves rows 1 and 3, accepts 3; row 1 redraws
    # with `redraw`, not `draw` (G4).
    calls = _fake_ik(monkeypatch, lambda v: (v == 3) | (v >= 100))
    start = torch.tensor([[1.0, 7.0], [-1.0, 7.0], [1.0, 8.0], [-1.0, 8.0]])
    particles = {}
    q, ok, accepted = _solve(particles, lambda rows: rows.float(), redraw=lambda rows: 100 + rows.float(),
                             q_start=start)
    assert calls == [[1, 3], [101]]
    assert torch.equal(q[[0, 2]], start[[0, 2]])
    assert q[1].tolist() == [101, 101] and q[3].tolist() == [3, 3]
    assert particles["x"].tolist() == [0, 101, 2, 3] and ok.all() and accepted == [3, 4]


@needs_cuda
def test_floor_picks_choose_upright_grasps(real_world):
    # blocks_t1 on the floor (real_world), n 64, seed 0. Each pick's ten are its most upright balanced grasps at 0.7
    # or more (G2, G5), and every particle holds one of them. The first pick's scene is the pool walk's, so its
    # particles take the ten in turn with the pool walk's confs, accepted as they are (G4).
    import cutamp.particle_initialization as pi
    from cutamp.config import TAMPConfiguration
    from cutamp.grasps import GRASP_SCORE_CUTOFF

    skeleton, _, _ = _first_plan(real_world)
    torch.manual_seed(0)
    init = pi.ParticleInitializer(real_world, TAMPConfiguration(num_particles=64), default_checker())
    particles = init(skeleton)
    picks = [op for op in skeleton if op.operator.metadata.action_type == "pick"]
    assert [c.grasp for c in init.grasp_choice] == [op.values[1] for op in picks]
    for c in init.grasp_choice:
        m = len(c.scores)
        assert all(len(t) == m for t in (c.grasps, c.ok, c.tilt, c.travel, c.margin))
        assert len(c.chosen) == 10 and c.ok[c.chosen].all() and (c.scores[c.chosen] >= GRASP_SCORE_CUTOFF).all()
        tilts = c.tilt[c.chosen]
        assert (tilts[1:] >= tilts[:-1]).all()  # upright first
        rest = c.ok & (c.scores >= GRASP_SCORE_CUTOFF)
        rest[c.chosen] = False
        assert not rest.any() or tilts.max() <= c.tilt[rest].min()
        assert (particles[c.grasp][:, None] == c.grasps[c.chosen][None]).all(-1).any(-1).all()
    first, q = init.grasp_choice[0], picks[0].values[2]
    rank = first.chosen[torch.arange(64, device=first.chosen.device) % 10]
    assert torch.equal(particles[first.grasp], first.grasps[rank])  # in turn, nothing redrawn
    assert torch.equal(particles[q][10:], particles[q][:-10])  # the same grasp, the same copied conf


@needs_cuda
def test_no_balanced_grasp_falls_back_to_the_score_draw(real_world, monkeypatch, caplog):
    # Every pool-walk IK fails (its calls are the ones wider than n), so no grasp is balanced: each pick warns and
    # draws by score from its pool, as before (G6): torch.multinomial on the pick's own scores, not the pool walk's
    # draw in turn.
    import logging
    import cutamp.particle_initialization as pi
    from cutamp.config import TAMPConfiguration

    n, real_gn_ik, real_multinomial, draws = 8, pi.gn_ik, torch.multinomial, []

    def pool_fails(world, world_from_ee, arms, movables, tol):
        q, ok = real_gn_ik(world, world_from_ee, arms, movables, tol)
        return q, ok & (len(ok) <= n)

    def multinomial(input, *args, **kwargs):
        draws.append(input.clone())
        return real_multinomial(input, *args, **kwargs)

    monkeypatch.setattr(pi, "gn_ik", pool_fails)
    skeleton, _, _ = _first_plan(real_world)
    torch.manual_seed(0)
    init = pi.ParticleInitializer(real_world, TAMPConfiguration(num_particles=n), default_checker())
    with caplog.at_level(logging.WARNING, logger="cutamp.particle_initialization"), monkeypatch.context() as m:
        m.setattr(torch, "multinomial", multinomial)
        particles = init(skeleton)
    warnings = [r.getMessage() for r in caplog.records
                if r.name == "cutamp.particle_initialization" and r.levelno == logging.WARNING]
    assert init.grasp_choice
    for c in init.grasp_choice:
        assert c.chosen is None and not c.ok.any()
        assert any(c.obj in message for message in warnings)
        assert any(torch.equal(scores, c.scores) for scores in draws), c.obj
        assert (particles[c.grasp][:, None] == c.grasps[None]).all(-1).any(-1).all()


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
