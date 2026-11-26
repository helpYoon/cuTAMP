"""``gn_project_balanced`` on a stance per row (design 2026-11-23, section 4). GPU: conftest's ``real_world``."""
import math

import torch

from cutamp.com_polygon_cost import (_fk_com_and_base, margin_from_support_planes, physical_support_centroid,
                                     support_region_planes, support_region_sides)
from cutamp.costs import curobo_pose_error
from cutamp.gn_refresh import gn_project_balanced, stance_sole_goals
from cutamp.robots.t1 import REAL_LEGS
from cutamp.step_loop import _planar
from cutamp.tests.conftest import needs_cuda
from cutamp.tests.test_real_legs_gn import FAR, _first_pick, _hand_error, _moved_targets, _tilt
from cutamp.tests.test_real_legs_init import POS_TOL, ROT_TOL, _poses

#: A stance two steps from the world's own, outside the world's base box.
SHIFT = (0.60, -0.40)


def _home(world, n):
    return _planar(world.stance_soles).expand(n, -1, -1).clone()


def _shifted(q, targets, stances):
    d = torch.tensor(SHIFT, device=q.device)
    q, targets, stances = q.clone(), targets.clone(), stances.clone()
    q[:, :2] += d
    targets[:, :2, 3] += d
    stances[..., :2] += d
    return q, targets, stances


def _sole_errors(world, q, stances):
    soles = torch.stack([_poses(world, q, frame) for frame in world.model.sole_frames], dim=1)
    return curobo_pose_error(soles, stance_sole_goals(stances))


def _margin(world, q, stances):
    com, _ = _fk_com_and_base(world, q)
    return margin_from_support_planes(com[..., :2], *support_region_planes(stances, world.model.sole_box, keyframe=True))


def _side_by_side():
    """``[1, 2, 3]``: the world's stance from the model's own data, its home mid-sole and sole spacing."""
    from curobo.config_io import load_yaml
    block = load_yaml(str(REAL_LEGS.cfg_path))["cutamp"]
    x, y = block["home_mid_sole"][0], block["sole_y"]
    return torch.tensor([[[x, y, 0.0], [x, -y, 0.0]]])


def test_the_sides_keep_the_region():
    # Inside, the margin over the sides alone is the margin over every plane, and no point changes side. Outside, the
    # margin over every plane is no distance: a plane beyond a corner can be violated more than any side (from 27.6 mm
    # out on the side-by-side stance and 9.0 mm on the stepped one, 2026-11-23), so only the side is compared there.
    stepped = _side_by_side()
    stepped[:, 1, 0] += 0.15
    stepped[:, 1, 2] += math.radians(20)
    g = torch.linspace(-0.3, 0.3, 61)
    grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    for stances in (_side_by_side(), stepped):
        normals, bounds = support_region_planes(stances, REAL_LEGS.sole_box, keyframe=True)
        sides = support_region_sides(normals, bounds)
        assert int(sides.isfinite().sum()) < int(bounds.isfinite().sum())    # not vacuous: planes were dropped
        points = physical_support_centroid(stances, REAL_LEGS.sole_box) + grid
        want, got = (margin_from_support_planes(points, normals[0], b[0]) for b in (bounds, sides))
        inside = want >= 0
        assert 0 < int(inside.sum()) < len(points)
        torch.testing.assert_close(got[inside], want[inside], rtol=0, atol=1e-6)
        assert torch.equal(got >= 0, inside)


def test_the_side_by_side_stance_has_four_sides():
    normals, bounds = support_region_planes(_side_by_side(), REAL_LEGS.sole_box, keyframe=True)
    sides = normals[0, support_region_sides(normals, bounds)[0].isfinite()]
    axes = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    near = (sides.unsqueeze(1) - axes).norm(dim=-1) <= 1e-6                    # [sides, 4]
    assert bool(near.any(1).all()) and bool(near.any(0).all())


@needs_cuda
def test_the_worlds_stance_region_is_the_keyframe_rectangle(real_world):
    # Passes before the change: it pins the fact the stance path relies on (spec 4, "Checked").
    home = _home(real_world, 1)
    centre = home[0, :, :2].mean(0)
    normals, bounds = support_region_planes(home, real_world.model.sole_box, keyframe=True)
    hx, hy = real_world.model.support.keyframe_half
    for dx, dy, inside in ((hx, hy, True), (-hx, -hy, True), (hx + 1e-4, 0.0, False), (0.0, hy + 1e-4, False)):
        point = centre + torch.tensor([dx, dy], device=centre.device)
        margin = float(margin_from_support_planes(point[None], normals, bounds))
        assert (margin >= -1e-6) == inside, (dx, dy, margin)


@needs_cuda
def test_the_worlds_stance_as_sole_goals(real_world):
    torch.testing.assert_close(stance_sole_goals(_home(real_world, 1))[0], real_world.stance_soles, rtol=0, atol=1e-5)


@needs_cuda
def test_on_the_worlds_stance_levels_1_and_2_match(real_world, real_plan):
    arm, _, q0 = _first_pick(real_world, real_plan)
    targets, _ = _moved_targets(real_world, q0, arm)
    want = gn_project_balanced(real_world, q0, targets, arm, collision=False)
    got = gn_project_balanced(real_world, q0, targets, arm, collision=False, stances=_home(real_world, len(q0)))
    torch.testing.assert_close(got, want, rtol=0, atol=1e-4)


@needs_cuda
def test_a_moved_stance_gives_the_moved_posture(real_world, real_plan):
    # Every joint 0.01 rad inside its limits first: GN blocks a joint at its limit by the sign of its step, which on a
    # joint exactly at the limit (most of the plan's starting rows, at the ankle cap) moving the scene's float32
    # rounding can flip; then the two calls take different steps (2026-11-25: 8 of 60 rows, 0.048 rad).
    arm, _, q0 = _first_pick(real_world, real_plan)
    lower, upper = real_world.joint_limits
    q0 = torch.minimum(torch.maximum(q0, lower + 0.01), upper - 0.01)
    targets, _ = _moved_targets(real_world, q0, arm)
    home = _home(real_world, len(q0))
    want = gn_project_balanced(real_world, q0, targets, arm, collision=False, stances=home)
    want[:, :2] += torch.tensor(SHIFT, device=want.device)
    q1, targets1, stances1 = _shifted(q0, targets, home)
    got = gn_project_balanced(real_world, q1, targets1, arm, collision=False, stances=stances1)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-4)


@needs_cuda
def test_out_of_reach_on_a_moved_stance_keeps_soles_and_balance(real_world, real_plan):
    arm, _, q = _first_pick(real_world, real_plan)
    targets, _ = _moved_targets(real_world, q, arm, FAR)
    q, targets, stances = _shifted(q, targets, _home(real_world, len(q)))
    for _ in range(4):
        q = gn_project_balanced(real_world, q, targets, arm, collision=False, stances=stances)
        pos, rot = _sole_errors(real_world, q, stances)
        assert float(pos.max()) <= 1e-3 and float(rot.max()) <= 0.01      # as _assert_soles_held on the world's stance
        assert float(_margin(real_world, q, stances).min()) >= 0
        assert float(_hand_error(real_world, q, targets, arm).min()) > 0.02


@needs_cuda
def test_a_posture_follows_a_foot_that_moved(real_world, real_plan):
    # What an update meets after Adam moved a command: the right sole 3 cm ahead and turned 5 degrees.
    arm, _, q = _first_pick(real_world, real_plan)
    targets = _poses(real_world, q, real_world.tool_frame_for_arm(arm))
    stances = _home(real_world, len(q))
    stances[:, 1, 0] += 0.03
    stances[:, 1, 2] += math.radians(5)
    for _ in range(3):
        q = gn_project_balanced(real_world, q, targets, arm, stances=stances)
    pos, rot = _sole_errors(real_world, q, stances)
    assert float(pos.median()) <= POS_TOL and float(rot.median()) <= ROT_TOL
    assert float(_hand_error(real_world, q, targets, arm).median()) <= POS_TOL


@needs_cuda
def test_a_tilted_start_ends_less_tilted(real_world, real_plan):
    arm, _, q0 = _first_pick(real_world, real_plan)
    targets = _poses(real_world, q0, real_world.tool_frame_for_arm(arm))
    stances = _home(real_world, len(q0))
    stances[:, 1, 0] += 0.03
    tilted = q0.clone()
    roll = list(REAL_LEGS.joint_names).index("base_j_roll")
    tilted[:, roll] += 0.3 * torch.where(q0[:, roll] >= 0, 1.0, -1.0)      # further the way each conf already leans
    q = tilted
    for _ in range(5):
        q = gn_project_balanced(real_world, q, targets, arm, stances=stances)
    assert float(_tilt(q).median()) < float(_tilt(tilted).median())
    assert float(_hand_error(real_world, q, targets, arm).median()) <= POS_TOL
    assert float(_sole_errors(real_world, q, stances)[0].median()) <= POS_TOL


@needs_cuda
def test_feet_in_line_give_finite_postures(real_world, real_plan):
    arm, _, q = _first_pick(real_world, real_plan)
    targets = _poses(real_world, q, real_world.tool_frame_for_arm(arm))
    stances = _home(real_world, len(q))
    stances[:, 1, :2] = stances[:, 0, :2] + torch.tensor([0.25, 0.0], device=q.device)
    assert bool(gn_project_balanced(real_world, q, targets, arm, collision=False, stances=stances).isfinite().all())


@needs_cuda
def test_without_stances_no_region_is_built(real_world, real_plan, monkeypatch):
    import cutamp.com_polygon_cost as com_polygon_cost
    import cutamp.gn_refresh as gn_refresh

    def refuse(*args, **kwargs):
        raise AssertionError("the one-stance path builds no support region")

    for module in (com_polygon_cost, gn_refresh):
        if hasattr(module, "support_region_planes"):
            monkeypatch.setattr(module, "support_region_planes", refuse)
    arm, _, q0 = _first_pick(real_world, real_plan)
    targets, _ = _moved_targets(real_world, q0, arm)
    gn_project_balanced(real_world, q0, targets, arm)
