"""Tests for the Gauss-Newton reIK refresh (cutamp/gn_refresh.py).

CUDA-only: exercises the real T1 kinematics kernel — the Jacobian convention
check is the load-bearing regression (GN silently diverges if cuRobo ever
changes row order or frame). The tests run on the real legs.
"""
import inspect
from unittest import mock

import pytest
import roma
import torch

from cutamp.tests.conftest import cupti_sync_events, make_blocks_t1_world, needs_cuda, sync_debug_error


def _ik(world, target, arm):
    """GN IK's confs for ``target`` ``[B, 4, 4]`` on ``arm``, no objects to clear; every row accepted."""
    from cutamp.particle_initialization import IKTol, gn_ik
    from cutamp.tests.conftest import default_checker
    q, ok = gn_ik(world, target, arm, target.new_zeros(len(target), 0, 4), IKTol.from_checker(default_checker()))
    assert bool(ok.all()), (arm, int(ok.sum()))
    return q


def _blocks(world, n):
    """``[n, M, 4]``: blocks_t1's movables' spheres at their initial poses, on every row: what a conf sees before any
    place, for deps built by hand."""
    from cutamp.utils.common import transform_spheres
    spheres = [transform_spheres(world.get_collision_spheres(obj), world.get_object_pose(obj)) for obj in world.movables]
    return torch.cat(spheres).expand(n, -1, -1)


def _four_dep_stack(world):
    """``(q [32, dof], T [32, 4, 4], arms)``: four arm-sorted 8-row IK deps. Per arm, one IK
    solution for a target at (0.35, +-0.25, 0.25) feeds a warm dep (target drifted by mm) and a
    far dep (x = 0.9, out of balanced reach)."""
    B = 8
    qs, targets = [], []
    for arm, y in (("left", 0.25), ("right", -0.25)):
        target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
        target[:, :3, 3] = torch.tensor([0.35, y, 0.25], device="cuda")
        q0 = _ik(world, target, arm)
        moved = target.clone()
        moved[:, 0, 3] += 0.003
        moved[:, 2, 3] += 0.0025
        far = target.clone()
        far[:, 0, 3] = 0.9
        qs += [q0, q0]
        targets += [moved, far]
    return torch.cat(qs), torch.cat(targets), ("left", "left", "right", "right")


def _without_com_level(monkeypatch):
    """Until ``monkeypatch`` undoes it, GN without its CoM level: ``_balanced_step`` sees a zero CoM and CoM Jacobian,
    its positional arguments 3 and 4, so both CoM active sets stay empty and level 1 keeps only the sole rows. A
    reorder of those arguments fails here, by name or by shape, not silently."""
    import cutamp.gn_refresh as gn_refresh
    step = gn_refresh._balanced_step
    assert list(inspect.signature(step).parameters)[3:5] == ["jac_com", "com"]

    def no_com(q, r, jac, jac_com, com, *args, **kwargs):
        assert jac_com.shape == (q.shape[0], 2, q.shape[1]) and com.shape == (q.shape[0], 2)
        return step(q, r, jac, torch.zeros_like(jac_com), torch.zeros_like(com), *args, **kwargs)

    monkeypatch.setattr(gn_refresh, "_balanced_step", no_com)


def _pose_errors(world, q, target, arm):
    """``([B] pos_err, [B] rot_err)`` of ``q`` ``[B, dof]`` against ``target`` ``[B, 4, 4]`` on ``arm``'s tool frame,
    from one FK on ``world.kinematics``, in the constraint checker's metric (``curobo_pose_error``)."""
    from curobo.types import JointState, Pose
    from cutamp.costs import curobo_pose_error
    kin = world.kinematics
    state = kin.compute_kinematics(JointState.from_position(q, joint_names=list(kin.joint_names)))
    ee = state.tool_poses.get_link_pose(world.tool_frame_for_arm(arm))
    # get_link_pose returns strided views; get_matrix requires contiguous.
    ee = Pose(position=ee.position.contiguous(), quaternion=ee.quaternion.contiguous()).get_matrix()
    return curobo_pose_error(ee, target)


def _quat_to_mat(q_wxyz):
    return roma.unitquat_to_rotmat(q_wxyz[..., [1, 2, 3, 0]].contiguous())


def _fk_pose(world, q, frame):
    from curobo.types import JointState
    kin = world.kinematics_jac
    js = JointState.from_position(q, joint_names=list(kin.config.kinematics_config.joint_names))
    state = kin.compute_kinematics(js)
    pose = state.tool_poses.get_link_pose(frame)
    B = q.shape[0]
    # Clone: cuRobo reuses its FK buffers in place, so the views go stale on
    # the next compute_kinematics call.
    return pose.position.reshape(B, 3).clone(), pose.quaternion.reshape(B, 4).clone(), state


@needs_cuda
def test_tool_jacobian_matches_finite_differences():
    # Rows 0-2 = world-frame linear, rows 3-5 = world-frame angular. GN's
    # update law assumes exactly this; fail loudly if cuRobo changes it.
    world = make_blocks_t1_world()
    frame = world.tool_frame_for_arm("left")
    kin = world.kinematics_jac
    frame_idx = list(kin.tool_frames).index(frame)

    torch.manual_seed(0)
    q0 = world.q_init.unsqueeze(0).clone() + 0.05 * torch.randn(1, world.q_init.numel(), device="cuda")
    pos0, quat0, state = _fk_pose(world, q0, frame)
    J = state.tool_jacobians[0, 0, frame_idx].clone()  # [6, dof]
    R0 = _quat_to_mat(quat0)[0]

    eps = 1e-3
    for k in range(q0.shape[1]):  # every column: base, arms, legs and trunk
        dq = torch.zeros_like(q0)
        dq[0, k] = eps
        pos1, quat1, _ = _fk_pose(world, q0 + dq, frame)
        fd_lin = (pos1 - pos0)[0] / eps
        dR = _quat_to_mat(quat1)[0] @ R0.T
        fd_ang = 0.5 * torch.tensor([
            dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1],
        ], device="cuda") / eps
        assert torch.allclose(J[:3, k], fd_lin, atol=2e-3), f"linear rows, joint {k}"
        assert torch.allclose(J[3:, k], fd_ang, atol=2e-3), f"angular rows, joint {k}"


@needs_cuda
def test_gn_project_balanced_keeps_com_in_and_closes_pose():
    # Balance-first GN: from a warm start it converges; from a target well out
    # of balanced reach it ends CoM-inside and closer to the target than it
    # started (closest balanced pose, not an unbalanced exact one).
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    from cutamp.gn_refresh import gn_project_balanced

    world = make_blocks_t1_world()
    B = 8
    target = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
    target[:, :3, 3] = torch.tensor([0.35, 0.25, 0.25], device="cuda")
    q0 = _ik(world, target, "left")
    moved = target.clone()
    moved[:, 0, 3] += 0.003
    moved[:, 2, 3] += 0.0025
    q1 = gn_project_balanced(world, q0, moved, "left")
    pos, rot = _pose_errors(world, q1, moved, "left")
    assert bool((pos < 0.005).all() & (rot < 0.05).all()), f"pos {pos.max().item():.4f}"
    assert bool(compute_com_polygon_mask(world, q1).all())

    far = target.clone()
    far[:, 0, 3] = 0.9
    pos0, _ = _pose_errors(world, q0, far, "left")
    q2 = gn_project_balanced(world, q0, far, "left", iters=30)
    pos2, _ = _pose_errors(world, q2, far, "left")
    assert bool(compute_com_polygon_mask(world, q2).all())
    assert bool((pos2 < pos0 - 0.05).all()), f"{pos0.min().item():.3f} -> {pos2.max().item():.3f}"

    # Without its CoM level, GN trades the CoM for the pose, so it ends closer to
    # the far target and, on most rows, outside the polygon: level 1 is what holds the CoM. Under the ankle cap
    # (shins never pitch back past home) one row of the 8 ends inside without it too, at the pose error it has with
    # it (0.349 m, seen 2026-11-06): its joint limits stop it first. So half the rows, and the ones that leave.
    with pytest.MonkeyPatch.context() as m:
        _without_com_level(m)
        q3 = gn_project_balanced(world, q0, far, "left", iters=30)
    pos3, _ = _pose_errors(world, q3, far, "left")
    out = ~compute_com_polygon_mask(world, q3)
    assert int(out.sum()) >= B // 2
    assert bool((pos3[out] < pos2[out]).all()), f"{pos3[out].max().item():.3f} vs {pos2[out].min().item():.3f}"


@needs_cuda
def test_gn_project_balanced_pins_per_row():
    # Each row keeps its own dep's frozen columns and tool frame (a right row on the left frame
    # would chase a target half a metre away).
    from cutamp.gn_refresh import gn_project_balanced

    world = make_blocks_t1_world()
    model = world.model
    q, T, arms = _four_dep_stack(world)
    q1 = gn_project_balanced(world, q, T, arms)
    per_dep = torch.cat([gn_project_balanced(world, qi, Ti, a) for qi, Ti, a in zip(q.split(8), T.split(8), arms)])
    assert (q1 - per_dep).abs().max().item() <= 1e-5  # bitwise as measured; slack for batch-dependent GPU reductions
    left_only = gn_project_balanced(world, q[:16], T[:16], arms[:2])  # a stack with no right-arm rows
    assert (left_only - per_dep[:16]).abs().max().item() <= 1e-5
    assert torch.equal(q1[:16, model.right_arm], q[:16, model.right_arm])
    assert torch.equal(q1[16:, model.left_arm], q[16:, model.left_arm])
    assert not torch.equal(q1[:16, model.left_arm], q[:16, model.left_arm])
    for rows, arm in ((slice(0, 8), "left"), (slice(16, 24), "right")):
        pos, rot = _pose_errors(world, q1[rows], T[rows], arm)
        assert bool((pos < 2e-3).all() & (rot < 2e-2).all()), (
            f"{arm}: pos {pos.max().item():.4f} rot {rot.max().item():.4f}"
        )
    # A real blocks_t1 refresh interleaves the arms (left_q1, right_q1, left_q3, right_q3).
    perm = torch.cat([torch.arange(8, device="cuda") + 8 * b for b in (0, 2, 1, 3)])
    inter = ("left", "right", "left", "right")
    q_int = gn_project_balanced(world, q[perm], T[perm], inter)
    assert (q_int - q1[perm]).abs().max().item() <= 1e-5


@needs_cuda
def test_gn_gate_matches_checker_helpers():
    # One FK over the whole stack gives the checker helpers' masks, each row on its own arm and both soles against
    # the stance. Every third row leans out of balance: GN without its CoM level (_without_com_level, which keeps the
    # sole rows) toward a target 0.9 m ahead ends CoM-outside, as test_gn_project_balanced_keeps_com_in_and_closes_pose
    # finds.
    from cutamp.com_polygon_cost import _fk_state, compute_com_polygon_mask
    from cutamp.gn_refresh import gn_gate, gn_project_balanced, soles_on_stance

    world = make_blocks_t1_world()
    q, T, arms = _four_dep_stack(world)
    far = T.clone()
    far[:, 0, 3] = 0.9
    with pytest.MonkeyPatch.context() as m:
        _without_com_level(m)
        q[::3] = gn_project_balanced(world, q, far, arms, iters=30)[::3]
    # Rows 1 and 2 get rotation errors between and beyond the tolerance; leaned row 0 aims
    # at its own tool pose, so it is pose-ok but CoM-out.
    for row, ang in ((1, 0.02), (2, 0.1)):
        T[row, :3, :3] = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, ang], device="cuda")) @ T[row, :3, :3]
    pos, quat, _ = _fk_pose(world, q[:8], world.tool_frame_for_arm("left"))
    T[0, :3, :3] = _quat_to_mat(quat[0])
    T[0, :3, 3] = pos[0]
    kin = world.kinematics
    with mock.patch.object(kin, "compute_kinematics", wraps=kin.compute_kinematics) as fk:
        pose_ok, com_ok = gn_gate(world, q, T, arms, 0.005, 0.05)
    assert fk.call_count == 1  # both masks from one FK
    ref = [_pose_errors(world, qi, Ti, a) for qi, Ti, a in zip(q.split(8), T.split(8), arms)]
    want_pose = torch.cat([(p < 0.005) & (r < 0.05) for p, r in ref])
    want_pose &= soles_on_stance(world, _fk_state(world, q), 0.005, 0.05)
    assert torch.equal(pose_ok, want_pose)
    assert torch.equal(com_ok, compute_com_polygon_mask(world, q))
    assert torch.equal(com_ok, torch.cat([compute_com_polygon_mask(world, qi) for qi in q.split(8)]))
    assert pose_ok.any() and (~pose_ok).any() and com_ok.any() and (~com_ok).any()
    assert (pose_ok & ~com_ok).any()
    assert not com_ok[::3].any()  # every leaned row: without its CoM level GN leaves the polygon


@needs_cuda
# Sync debug mode warns that it misses some syncs; the profile is the oracle.
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")
def test_balanced_refresh_is_sync_free():
    # A steady-state balance-first refresh makes no host syncs or copies (its profile holds only
    # the profiler's own cudaDeviceSynchronize) and drops stale grads.
    from torch.profiler import ProfilerActivity, profile

    from cutamp.optimize_plan import IKDep, _ReikDriftGate, _refresh_balanced, _StaticRefresh

    world = make_blocks_t1_world()
    q, T, arms = _four_dep_stack(world)
    names = ["left_q1", "left_q2", "right_q3", "right_q4"]
    deps = [IKDep(n, a, target_fn=lambda p, Ti=Ti: Ti, movables_fn=lambda p, M=_blocks(world, 8): M)
            for n, a, Ti in zip(names, arms, T.split(8))]
    particles = {n: qi.clone() for n, qi in zip(names, q.split(8))}
    gate = _ReikDriftGate(pos_tol=0.005, rot_tol=0.05)
    for step in (4, 9):  # warm-up: lazy kinematics, caches, allocations
        _refresh_balanced(particles, deps, world, gate, step)
    for n in names:
        particles[n].grad = torch.ones_like(particles[n])  # stale buffers the refresh must drop
    torch.cuda.synchronize()
    sync_events = cupti_sync_events()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], **sync_events) as prof, sync_debug_error():
        _refresh_balanced(particles, deps, world, gate, 14)
    events = [e.name for e in prof.events()]
    assert [n for n in events if "Synchronize" in n] == ["cudaDeviceSynchronize"]
    assert not [n for n in events if "DtoH" in n or "HtoD" in n]
    if sync_events:  # the profiler's own cudaDeviceSynchronize is the one Context Sync
        assert [n for n in events if n.endswith(" Sync")] == ["Context Sync"]
    assert all(particles[n].grad is None for n in names)

    gate.flush()

    # Replayed as G_refresh: one graph launch plus the slot clone's copy, and no eager kernel launch.
    refresh = _StaticRefresh(particles, deps, world, gate)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], **sync_events) as prof, sync_debug_error():
        refresh(19)
    names = [e.name for e in prof.events()]
    assert names.count("cudaGraphLaunch") == 1 and names.count("cudaMemcpyAsync") == 1
    assert not [n for n in names if n.startswith(("cudaLaunch", "cuLaunch"))]
    refresh.graph.release()


@needs_cuda
def test_balanced_refresh_on_each_dep_layout():
    # One dep, a one-arm stack and blocks_t1's interleaved arm order: each refresh writes back its
    # stack's projection and flushes one record per dep, in dep order, counting its gate misses.
    from cutamp.gn_refresh import gn_gate, gn_project_balanced
    from cutamp.optimize_plan import IKDep, _ReikDriftGate, _refresh_balanced

    world = make_blocks_t1_world()
    q, T, arms = _four_dep_stack(world)
    q_blocks, T_blocks = q.split(8), T.split(8)
    all_names = ["left_q1", "left_q2", "right_q3", "right_q4"]
    for idx in ([0], [2, 3], [0, 2, 1, 3]):
        names = [all_names[i] for i in idx]
        arms_sub = tuple(arms[i] for i in idx)
        T_sub = torch.cat([T_blocks[i] for i in idx])
        deps = [IKDep(n, a, target_fn=lambda p, Ti=T_blocks[i]: Ti, movables_fn=lambda p, M=_blocks(world, 8): M)
                for n, a, i in zip(names, arms_sub, idx)]
        particles = {n: q_blocks[i].clone() for n, i in zip(names, idx)}
        gate = _ReikDriftGate(0.005, 0.05)
        _refresh_balanced(particles, deps, world, gate, 4)
        confs = [particles[n] for n in names]
        before = torch.cat(confs)
        _refresh_balanced(particles, deps, world, gate, 9)
        written = torch.cat(confs)
        movables = torch.cat([dep.movables_fn(particles) for dep in deps])
        assert torch.equal(written, gn_project_balanced(world, before, T_sub, arms_sub, movables=movables)), idx
        gate.flush()
        assert [(r["step"], r["conf"]) for r in gate.stats] == [(s, n) for s in (4, 9) for n in names], idx
        pose_ok, com_ok = gn_gate(world, written, T_sub, arms_sub, 0.005, 0.05)
        misses = (~(pose_ok & com_ok)).view(len(idx), 8).sum(-1).tolist()
        assert [r["gn_fail"] for r in gate.stats[len(idx):]] == misses, idx


def test_pose_error_resolves_sub_mrad_rotations_in_float32():
    # The checker's atan2 angle must see a 4e-4 rad rotation (an acos form collapses to 0 below ~5e-4 rad in float32).
    from cutamp.costs import curobo_pose_error

    a = torch.eye(4).expand(2, 4, 4).clone()
    b = a.clone()
    b[:, :3, :3] = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, 4e-4]))
    assert curobo_pose_error(a, b)[1].min().item() > 1e-4
