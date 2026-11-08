"""Each pick and place as an approach, a straight insert and a leave along the grasp's axis (design 7.8): the pre and
post hand targets, the FK line check, the active hand's criteria row, the held block's gap to the floor the lift starts
from, and the retry on a line miss. blocks_t1's segments end to end are test_motion_anchor's plan audit."""

import math

import pytest
import torch

from cutamp.tests.conftest import needs_cuda


def _hand(x_axis, y_axis, position) -> torch.Tensor:
    """``[4, 4]``: a hand pose whose +x (the approach) is ``x_axis`` and +y is ``y_axis``, at ``position``."""
    x, y = torch.tensor(x_axis, dtype=torch.float32), torch.tensor(y_axis, dtype=torch.float32)
    pose = torch.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, y, torch.linalg.cross(x, y)
    pose[:3, 3] = torch.tensor(position, dtype=torch.float32)
    return pose


def _moved(pose: torch.Tensor, offset) -> torch.Tensor:
    """``pose`` translated by the world-frame ``offset``, its rotation kept."""
    out = pose.clone()
    out[:3, 3] += torch.tensor(offset, dtype=torch.float32)
    return out


def test_approach_targets():
    from cutamp.grasps import APPROACH_DISTANCE, LIFT_HEIGHT, VERTICAL_CONE, approach_targets, is_vertical

    assert (APPROACH_DISTANCE, LIFT_HEIGHT) == (0.05, 0.05)
    # Top-down, the hand's +x straight down: the pick comes in from 5 cm straight above and leaves the same way.
    top = _hand((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (0.40, 0.10, 0.55))
    assert is_vertical(top)
    pre, post = approach_targets(top, "pick")
    torch.testing.assert_close(pre, _moved(top, (0.0, 0.0, 0.05)))
    assert torch.equal(post, pre)
    # Horizontal, the hand's +x forward: the pick inserts from 5 cm back along x and lifts 5 cm.
    side = _hand((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.40, 0.10, 0.55))
    assert not is_vertical(side)
    pre, post = approach_targets(side, "pick")
    torch.testing.assert_close(pre, _moved(side, (-0.05, 0.0, 0.0)))
    torch.testing.assert_close(post, _moved(side, (0.0, 0.0, 0.05)))
    # The place comes in from 5 cm above and backs out 5 cm along the hand's x.
    pre, post = approach_targets(side, "place")
    torch.testing.assert_close(pre, _moved(side, (0.0, 0.0, 0.05)))
    torch.testing.assert_close(post, _moved(side, (-0.05, 0.0, 0.0)))
    # The cone: within VERTICAL_CONE of straight down is vertical.
    for tilt, vertical in ((VERTICAL_CONE - 0.02, True), (VERTICAL_CONE + 0.02, False)):
        x = (math.sin(tilt), 0.0, -math.cos(tilt))
        assert is_vertical(_hand(x, (0.0, 1.0, 0.0), (0.4, 0.1, 0.55))) is vertical, tilt


def test_check_line():
    # A straight 5 cm insert: a straight path passes; a 6 mm sideways bulge, a 2 mm backtrack and a stop 6 mm short
    # each raise, naming the frame; 4 mm, 0.5 mm and 4 mm pass.
    from cutamp.motion_solver import check_line

    start, end = torch.tensor([0.40, 0.10, 0.60]), torch.tensor([0.40, 0.10, 0.55])
    line = end - start
    t = torch.linspace(0.0, 1.0, 41).unsqueeze(-1)
    straight = start + t * line
    check_line(straight, start, end)

    sideways = torch.tensor([1.0, 0.0, 0.0])
    check_line(straight + 0.004 * torch.sin(math.pi * t) * sideways, start, end)
    with pytest.raises(RuntimeError, match="6.0 mm .* frame 20 of 41"):
        check_line(straight + 0.006 * torch.sin(math.pi * t) * sideways, start, end)

    def back_by(metres):
        s = torch.cat([torch.linspace(0.0, 0.5, 21), torch.linspace(0.5 - metres / 0.05, 1.0, 20)]).unsqueeze(-1)
        return start + s * line

    check_line(back_by(0.0005), start, end)
    with pytest.raises(RuntimeError, match="2.0 mm .* frame 21 of 41"):
        check_line(back_by(0.002), start, end)

    check_line(start + t * line * (1 - 0.004 / 0.05), start, end)
    with pytest.raises(RuntimeError, match="6.0 mm .* frame 40 of 41"):
        check_line(start + t * line * (1 - 0.006 / 0.05), start, end)


@pytest.fixture(scope="module")
def planner(real_world):
    """``real_world``'s motion planner as solve_curobo builds it, built once."""
    from cutamp.motion_solver import build_motion_planner
    with build_motion_planner(real_world) as mp:
        yield mp


def _criteria_rows(planner) -> list:
    """``[(tool_frames, {field: [links, ...] tensor})]``: a copy of every tool-pose cost's stacked criteria in the
    planner's trajopt and IK rollouts, which ``planner.update_tool_pose_criteria`` writes."""
    rows = []
    for solver in (planner.trajopt_solver, planner.ik_solver):
        core = solver.core
        for rollout in [*core.get_all_rollout_instances(), *core.additional_metrics_rollouts.values()]:
            for cost in rollout.get_cost_component_by_name("tool_pose"):
                stack = cost._stacked_tool_pose_criteria
                rows.append((list(stack.tool_frames),
                             {name: getattr(stack, name).clone() for name in stack.tensor_field_names()}))
    return rows


def _row_is(rows, frame, criteria) -> bool:
    """Whether ``frame``'s row equals ``criteria`` in every stack of ``rows``, tensor by tensor."""
    return all(torch.equal(stack[name][frames.index(frame)], getattr(criteria, name))
               for frames, stack in rows for name in stack)


@needs_cuda
def test_line_motion_restores_the_row(planner):
    # Inside _line_motion the active hand's row is the line; the other hand's and the soles' rows never change. After
    # a normal exit, and after an exception inside the block, the hand's row is t1's terminal-only hand tracking again.
    from curobo.types import ToolPoseCriteria
    from cutamp.motion_solver import _line_motion

    default = ToolPoseCriteria.track_position_and_orientation(non_terminal_scale=0.0, device_cfg=planner.device_cfg)
    hand = "left_base_link"
    before = _criteria_rows(planner)
    assert before and _row_is(before, hand, default)

    def others_unchanged(rows):
        return all(torch.equal(stack[name][i], old[name][i])
                   for (frames, stack), (_, old) in zip(rows, before)
                   for i, frame in enumerate(frames) if frame != hand for name in stack)

    with _line_motion(planner, "left"):
        inside = _criteria_rows(planner)
        assert not _row_is(inside, hand, default) and others_unchanged(inside)
        assert _row_is(inside, hand, ToolPoseCriteria.linear_motion(axis="x", device_cfg=planner.device_cfg))
    assert _row_is(_criteria_rows(planner), hand, default)

    with pytest.raises(ValueError, match="inside the block"):
        with _line_motion(planner, "left", torch.tensor([0.0, 0.0, 1.0])):
            inside = _criteria_rows(planner)
            assert not _row_is(inside, hand, default) and others_unchanged(inside)
            raise ValueError("inside the block")
    after = _criteria_rows(planner)
    assert _row_is(after, hand, default) and others_unchanged(after)


@needs_cuda
def test_attached_block_clears_the_floor(real_world, real_plan, planner):
    # blocks_t1 on the floor (h 0.0): with the first pick's block attached at its grasp conf, where the lift starts,
    # the planner's start-state check (the graph planner's, on the planner's spheres) passes. The same spheres moved
    # just into the floor fail it, so the check sees the held block and the floor.
    import cutamp.motion_solver as ms
    from curobo.types import JointState
    from cutamp.tests.test_real_legs_gn import _first_pick

    skeleton, _ = real_plan
    obj = next(op for op in skeleton if op.operator.metadata.action_type == "pick").values[0]
    arm, _, q = _first_pick(real_world, real_plan)
    q = q[:1].clone()
    js = JointState.from_position(q, joint_names=list(planner.kinematics.joint_names))
    kp = planner.attachment_manager.kinematics_params
    idx = kp.get_sphere_index_from_link_name(ms._ATTACH_LINK_FOR_ARM[arm])
    ms._reset_attachment_state(planner, real_world)
    try:
        ms._attach_object(planner, obj, js, arm)
        idx = idx[kp.link_spheres[0, idx, 3] > 0]
        spheres = planner.kinematics.compute_kinematics(js).robot_spheres.view(-1, 4)[idx]
        lowest = float((spheres[:, 2] - spheres[:, 3]).min())  # the floor's top is z = 0
        assert bool(planner.graph_planner.check_samples_feasibility(q)), f"held block's lowest sphere at {lowest} m"
        # Into the floor by 1 mm: the attach link is the tool frame's identity child, so the shift is link-local.
        world_from_link = planner.kinematics.compute_kinematics(js).tool_poses.get_link_pose(ms.TOOL_FRAME_FOR_ARM[arm])
        down = world_from_link.get_matrix()[0, :3, :3].T @ torch.tensor([0.0, 0.0, -(lowest + 0.001)],
                                                                         device=q.device)
        kp.link_spheres[0, idx, :3] += down
        assert not bool(planner.graph_planner.check_samples_feasibility(q))
    finally:
        ms._reset_attachment_state(planner, real_world)


@needs_cuda
def test_line_miss_retries_next_particle(real_world, real_plan, planner, monkeypatch):
    # The first insert's line check misses: solve_curobo's RuntimeError carries the first pick's op_idx, and
    # _motion_plan_with_retries moves on to the next candidate. The skeleton is cut after the first pick, so the
    # approach IK has only that pick's confs, which the init's GN accepted. The gate's soles check is off, so the
    # test's miss is the first error the retry meets whatever the init drew: on the init whose first pick is block2's
    # (PYTHONHASHSEED 1) both candidates' approaches leave the stance by 6.5 and 9.3 mm (2026-11-08), which would fail
    # them first. test_real_legs_motion tests that check.
    import cutamp.algorithm as alg
    import cutamp.motion_solver as ms
    from cutamp.config import TAMPConfiguration
    from cutamp.particle_initialization import IKTol
    from cutamp.tests.conftest import default_checker
    from cutamp.tests.test_real_legs_gn import _accepted
    from cutamp.utils.timer import TorchTimer
    from cutamp.utils.visualizer import Visualizer

    skeleton, particles = real_plan
    k = next(i for i, op in enumerate(skeleton) if op.operator.metadata.action_type == "pick")
    rows = torch.nonzero(_accepted(real_world, real_plan, skeleton[k].values[-1])).flatten()[:2]
    candidates = [{name: v[r].detach().clone() for name, v in particles.items()} for r in rows.tolist()]
    assert len(candidates) == 2

    check_line, solve, lines, errors = ms.check_line, alg.solve_curobo, [], []

    def miss_first(palm, start, end):
        lines.append(len(palm))
        if len(lines) == 1:
            raise RuntimeError("palm strays 9.9 mm from its line at frame 0 (the test's miss)")
        check_line(palm, start, end)

    def solve_spy(plan_info, particle, *args, **kwargs):
        try:
            return solve(plan_info, particle, *args, **kwargs)
        except RuntimeError as e:
            errors.append(e)
            raise

    monkeypatch.setattr(ms, "check_line", miss_first)
    monkeypatch.setattr(ms, "_check_planted", lambda world, q: None)
    monkeypatch.setattr(alg, "solve_curobo", solve_spy)
    monkeypatch.setattr(alg, "_PLAN_ATTEMPTS", (20,))  # one pass: the next candidate plans, or the test fails here
    config = TAMPConfiguration(num_particles=64, curobo_plan=True, enable_visualizer=False)
    _, _, attempts, _, failures = alg._motion_plan_with_retries(
        {"plan_skeleton": list(skeleton[: k + 1])}, candidates, real_world, config, TorchTimer(),
        Visualizer(config, None), get_planner=lambda: planner, ik_tol=IKTol.from_checker(default_checker()))
    assert errors and errors[0].op_idx == k and "the test's miss" in str(errors[0]), errors
    assert "insert" in str(errors[0]) and failures[0] == str(errors[0])
    assert attempts == 2 and len(lines) >= 2  # the next candidate planned its own insert
