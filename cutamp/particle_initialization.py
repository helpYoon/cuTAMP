# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Particle initialization under cuRobo v0.8 single-MotionPlanner architecture.

Each conf parameter (``left_q0``, ``right_q1``, etc.) is a single 21-DOF
vector in T1's full cspace order.
"""

import logging
from typing import Dict, List, Optional

import torch

from curobo.scene import Cuboid
from curobo.types import GoalToolPose, JointState, Pose

from cutamp._curobo_internals import (
    build_pin_weights,
    restore_cspace_target_dof_weight,
    snapshot_cspace_target_dof_weight,
    write_cspace_target_dof_weight,
)
from cutamp.config import TAMPConfiguration
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.robots.t1 import (
    BASE_INDICES,
    LEFT_ARM_INDICES,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_INDICES,
    RIGHT_ARM_JOINT_NAMES,
    t1_home,
)
from cutamp.samplers import (
    grasp_4dof_sampler,
    place_4dof_sampler,
)
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    pose_list_to_mat4x4,
    transform_spheres,
)

_log = logging.getLogger(__name__)


def _build_multi_frame_goal(
    host,
    active_tool_frame: str,
    target_pose: Pose,
    current_state: JointState,
) -> GoalToolPose:
    """Build a GoalToolPose covering all of ``host.kinematics.tool_frames``.

    ``host`` may be a ``MotionPlanner`` or an ``IKSolver`` — both expose
    ``kinematics.tool_frames`` and ``compute_kinematics(state)``.

    The active frame gets ``target_pose``; every inactive frame gets its
    current FK pose at ``current_state``. This satisfies cuRobo IK's
    ``reorder_links`` requirement that the goal cover every kinematics
    tool frame, while keeping the inactive frames at their "stay put"
    pose so they don't drag the trajopt away from the cspace pin.
    """
    all_frames: List[str] = list(host.kinematics.tool_frames)
    poses: Dict[str, Pose] = {active_tool_frame: target_pose}
    inactive = [f for f in all_frames if f != active_tool_frame]
    if inactive:
        kin_state = host.compute_kinematics(current_state)
        for f in inactive:
            poses[f] = kin_state.tool_poses.get_link_pose(f)
    return GoalToolPose.from_poses(poses, ordered_tool_frames=all_frames)


def _ik_for_pose(
    world: TAMPWorld,
    world_from_ee: torch.Tensor,
    arm: Optional[str],
    *,
    current_state_q: Optional[torch.Tensor] = None,
    ik=None,
    seed_from_current: bool = False,
):
    """Solve IK for a batch of EE poses, targeting the active arm's tool frame.

    Uses the same multi-frame-goal + cspace-pin pattern the motion planner
    applies during arm operators: the inactive frame is targeted at its
    current FK pose, and its joints are pinned via ``cspace_target_dof_weight``.
    Without this, IK leaves the inactive arm unconstrained and it drifts to
    arbitrary joint values, polluting downstream particle optimization.

    ``current_state_q`` (full-cspace ``[B, len(world.kinematics.joint_names)]``)
    anchors the cspace pin and inactive-frame goal at a non-home pose. Note
    cuRobo's seed manager IGNORES ``current_state`` for seeding — pass
    ``seed_from_current=True`` to additionally inject it as an explicit
    optimization seed (true warm start; remaining seeds stay random). Used
    by the coupled-reIK refresh, where the previous solution is a near-answer.

    ``ik`` overrides the solver (default ``world.ik_solver``) — the refresh
    path passes ``world.ik_refresh_solver``.
    """
    if ik is None:
        ik = world.ik_solver
    active_frame = world.tool_frame_for_arm(arm)
    if active_frame not in ik.kinematics.tool_frames:
        raise RuntimeError(
            f"IK solver does not expose tool frame {active_frame!r}; "
            f"available: {ik.kinematics.tool_frames}"
        )
    target_pose = Pose.from_matrix(world_from_ee)
    batch_size = world_from_ee.shape[0]
    full_names = list(world.kinematics.joint_names)
    if current_state_q is None:
        seed_full = world.q_init.unsqueeze(0).expand(batch_size, -1)
    else:
        # Caller passes full-cspace q; trust their batch shape.
        seed_full = current_state_q
    full_js = JointState.from_position(seed_full, joint_names=full_names)
    current_state = ik.kinematics.get_active_js(full_js)
    goal = _build_multi_frame_goal(ik, active_frame, target_pose, current_state)

    snapshot = snapshot_cspace_target_dof_weight([ik])
    try:
        if arm is not None:
            # Pin the inactive arm's DOFs at their current values.
            write_cspace_target_dof_weight(
                [ik], build_pin_weights(
                    ik, RIGHT_ARM_JOINT_NAMES if arm == "left" else LEFT_ARM_JOINT_NAMES,
                ),
            )
        seed_config = (
            current_state.position.unsqueeze(1) if seed_from_current else None
        )
        return ik.solve_pose(
            goal_tool_poses=goal, current_state=current_state, seed_config=seed_config,
        )
    finally:
        restore_cspace_target_dof_weight([ik], snapshot)


def _ik_for_pose_com_safe(
    world: TAMPWorld,
    world_from_ee: torch.Tensor,
    arm: Optional[str],
    *,
    max_retries: int = 3,
    context: str = "",
):
    """Call ``_ik_for_pose``; verify COM-in-polygon; retry failed particles.

    Backstop behind the CoM-aware seed-IK residual (Layer 1, cuRobo
    fork): verifies COM-in-polygon and retries the failed subset. The
    residual makes violations rare; this wrapper guarantees
    gate-consistency of stored confs. cuRobo's seed sampler has internal
    randomization so retries may converge to different (feasible)
    solutions.

    Returns the last batch result (best effort). Logs final failure count
    at WARNING level so persistent failures are visible.
    """
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    result = _ik_for_pose(world, world_from_ee, arm)
    full_dof = world.q_init.shape[-1]
    device = world.q_init.device
    batch_size = world_from_ee.shape[0]
    for _attempt in range(max_retries):
        q_batch = _ik_solution_to_full_q(result, world)
        in_polygon = compute_com_polygon_mask(world, q_batch)
        if bool(in_polygon.all()):
            return result
        # Diversify the LBFGS starting region across retries: each retry
        # perturbs the home seed with gaussian noise whose stddev grows
        # with attempt count. cuRobo's internal seed sampler adds its own
        # noise on top of `current_state`, but if every retry hands it
        # the SAME home pose, all attempts explore the same neighbourhood
        # and find the same COM-infeasible local minimum (the "lean
        # forward + tip" basin). Larger perturbations push seeds into
        # the leaned-back / squatted basins where COM-feasible solutions
        # live. Noise scales 0.1, 0.2, 0.3, ... rad and is masked to
        # body + active arm only (base & inactive arm stay locked).
        noise_std = 0.1 * (_attempt + 1)
        seed = world.q_init.unsqueeze(0).expand(batch_size, -1).clone()
        noise = torch.randn(batch_size, full_dof, device=device) * noise_std
        noise[:, BASE_INDICES] = 0.0  # base locked
        if arm == "left":
            noise[:, RIGHT_ARM_INDICES] = 0.0  # right arm locked
        elif arm == "right":
            noise[:, LEFT_ARM_INDICES] = 0.0  # left arm locked
        seed = seed + noise
        # Clamp the perturbed seed to joint limits so large-sigma retries don't
        # hand cuRobo an out-of-limits current_state. Limits come from the same
        # full-cspace kinematics that defines world.q_init's joint order
        # (JointLimits.joint_names == kinematics.joint_names, verified [2, full_dof]).
        _limits = world.kinematics.get_joint_limits().position  # [2, full_dof]
        _lower = _limits[0].to(seed.device, seed.dtype)
        _upper = _limits[1].to(seed.device, seed.dtype)
        seed = torch.clamp(seed, min=_lower, max=_upper)
        retry_result = _ik_for_pose(world, world_from_ee, arm, current_state_q=seed)
        result.copy_at_batch_indices(retry_result, ~in_polygon)
    # Final check for logging.
    q_batch_final = _ik_solution_to_full_q(result, world)
    in_polygon_final = compute_com_polygon_mask(world, q_batch_final)
    n_fail = int((~in_polygon_final).sum().item())
    if n_fail > 0:
        prefix = f"[{context}] " if context else ""
        _log.warning(
            f"{prefix}_ik_for_pose_com_safe: {n_fail}/{len(q_batch_final)} particles "
            f"still out of COM polygon after {max_retries} retries"
        )
    return result


def _achieved_pose_errors(
    world: TAMPWorld, q_full: torch.Tensor, world_from_ee: torch.Tensor, arm: str,
):
    """``([B] pos_err, [B] rot_err)`` of ``q_full`` against ``world_from_ee``
    on ``arm``'s tool frame, in the constraint checker's own metric
    (``curobo_pose_error``).

    Exists because cuRobo v0.8's ``IKSolverResult.position_error`` /
    ``rotation_error`` are unusable on multi-tool-frame robots: ``_get_result``
    concatenates the per-frame convergence metrics blockwise
    ``[P0..PL-1, R0..RL-1]`` but then views them as interleaved
    ``(num_links, 2)`` pairs, cross-pairing metres with radians (the success
    flag uses the un-mixed values and is unaffected).
    """
    from cutamp.costs import curobo_pose_error
    js = JointState.from_position(
        q_full, joint_names=list(world.kinematics.joint_names),
    )
    kin_state = world.kinematics.compute_kinematics(js)
    ee_pose = kin_state.tool_poses.get_link_pose(world.tool_frame_for_arm(arm))
    # get_link_pose returns strided views; get_matrix requires contiguous.
    ee = Pose(
        position=ee_pose.position.contiguous(),
        quaternion=ee_pose.quaternion.contiguous(),
    ).get_matrix()
    return curobo_pose_error(ee, world_from_ee)


def _ik_solution_to_full_q(ik_result, world: TAMPWorld) -> torch.Tensor:
    """Build a full-cspace ``[B, len(world.kinematics.joint_names)]`` tensor
    from an IK result. Locked-out joints (e.g. mobile base) are filled from
    the IK kinematics' own ``lock_jointstate`` — same source the planner uses,
    so values stay consistent without an external ``t1_home`` reference.
    """
    ik = world.ik_solver
    sol_js = JointState.from_position(
        ik_result.solution[:, 0], joint_names=list(ik.kinematics.joint_names),
    )
    return ik.kinematics.get_full_js(sol_js).reorder(list(world.kinematics.joint_names)).position


def _ik_success_mask(ik_result) -> torch.Tensor:
    """``[B]`` bool success mask for the stored (seed-0) solution.

    cuRobo result fields are ``[B]`` or ``[B, return_seeds]``; the seed-0
    column pairs with ``_ik_solution_to_full_q``'s ``solution[:, 0]``.
    """
    s = ik_result.success
    return (s.reshape(s.shape[0], -1)[:, 0] if s.ndim > 1 else s).bool().reshape(-1)


def _store_ik_q(particles: dict, q_name: str, ik_result, world: TAMPWorld, arm: Optional[str]) -> None:
    """Store the IK result's q under ``q_name``, reordered to the full kin's joint order.

    Rows whose IK reported failure fall back to ``world.q_init``: cuRobo's joint
    limits are a soft residual + success filter (nothing clamps), so an
    unsuccessful solution may be out-of-limit and must not enter particles.
    """
    q_full = _ik_solution_to_full_q(ik_result, world)
    success = _ik_success_mask(ik_result).to(q_full.device)
    home = world.q_init.to(q_full.device).unsqueeze(0).expand_as(q_full)
    particles[q_name] = torch.where(success.unsqueeze(-1), q_full, home)


class ParticleInitializer:
    def __init__(self, world: TAMPWorld, config: TAMPConfiguration):
        self.world = world
        self.config = config
        self.q_init = world.q_init.repeat(config.num_particles, 1)

    def _initial_particles(self) -> Particles:
        return {
            "left_q0": self.q_init.clone(),
            "right_q0": self.q_init.clone(),
        }

    def __call__(self, plan_skeleton: PlanSkeleton, verbose: bool = True) -> Optional[Particles]:
        config = self.config
        num_particles = config.num_particles
        world = self.world
        device = world.device

        particles = self._initial_particles()
        deferred_params: set[str] = set()
        move_free_deferred: dict[str, str] = {}
        log_debug = _log.debug if verbose else lambda *args, **kwargs: None

        for idx, ground_op in enumerate(plan_skeleton):
            metadata = ground_op.operator.metadata
            params = ground_op.values
            header = f"{idx + 1}. {ground_op}"
            arm = metadata.arm

            if metadata.is_motion and metadata.action_type is None:
                if len(params) == 3:
                    q_start, _traj, q_end = params
                    if q_start not in particles:
                        raise ValueError(f"{q_start=} should already be bound")
                    deferred_params.add(q_end)
                    move_free_deferred[q_end] = q_start
                    log_debug(f"{header}. Deferred {q_end}")
                else:
                    obj, grasp, q_start, _traj, q_end = params
                    if not world.has_object(obj):
                        raise ValueError(f"{obj=} not found in world")
                    if grasp not in particles:
                        raise ValueError(f"{grasp=} should already be bound")
                    if q_start not in particles:
                        raise ValueError(f"{q_start=} should already be bound")
                    deferred_params.add(q_end)
                    log_debug(f"{header}. Deferred {q_end}")

            elif metadata.action_type == "pick":
                obj, grasp, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp in particles:
                    raise ValueError(f"{grasp=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                tool_frame = world.tool_frame_for_arm(arm)
                gripper_spheres = world.gripper_spheres[tool_frame]
                tool_from_ee = world.tool_from_ee[tool_frame]

                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                num_faces = 4 if isinstance(obj_curobo, Cuboid) else None

                sampled_grasps = grasp_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, num_faces=num_faces)
                obj_from_grasp = action_4dof_to_mat4x4(sampled_grasps)

                grasp_spheres = transform_spheres(gripper_spheres, obj_from_grasp)
                grasp_coll = sphere_to_sphere_overlap(obj_spheres, grasp_spheres)
                good_idxs = grasp_coll.topk(num_particles, largest=False).indices
                particles[grasp] = sampled_grasps[good_idxs]

                obj_from_grasp = obj_from_grasp[good_idxs]
                world_from_obj = pose_list_to_mat4x4(obj_curobo.pose).to(device)
                world_from_ee = world_from_obj @ obj_from_grasp @ tool_from_ee
                ik_result = _ik_for_pose_com_safe(
                    world, world_from_ee, arm,
                    context=f"pick({arm})",
                )
                log_debug(
                    f"{header}. IK success: {ik_result.success.sum()}/{num_particles}"
                )
                _store_ik_q(particles, q, ik_result, world, arm)
                deferred_params.discard(q)
                move_free_deferred.pop(q, None)

            elif metadata.action_type == "place":
                obj, grasp, placement, surface, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if placement in particles:
                    raise ValueError(f"{placement=} shouldn't already be bound")
                if not world.has_object(surface):
                    raise ValueError(f"{surface=} not found in world")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                tool_frame = world.tool_frame_for_arm(arm)
                tool_from_ee = world.tool_from_ee[tool_frame]

                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                surface_curobo = world.get_object(surface)
                sampled_placements = place_4dof_sampler(
                    num_particles * 4, obj_curobo, obj_spheres, surface_curobo,
                )

                world_from_obj = action_4dof_to_mat4x4(sampled_placements)
                obj_place_spheres = transform_spheres(obj_spheres, world_from_obj)
                # WorldCollisionCost returns [B, H] summed across spheres.
                place_coll = world.collision_fn(obj_place_spheres[:, None].contiguous())[:, 0]
                best_idxs = place_coll.topk(num_particles, largest=False).indices
                sampled_placements = sampled_placements[best_idxs]
                world_from_obj = world_from_obj[best_idxs]

                particles[placement] = sampled_placements
                obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                world_from_ee = world_from_obj @ obj_from_grasp @ tool_from_ee
                ik_result = _ik_for_pose_com_safe(
                    world, world_from_ee, arm,
                    context=f"place({arm})",
                )
                log_debug(
                    f"{header}. IK success: {ik_result.success.sum()}/{num_particles}"
                )
                _store_ik_q(particles, q, ik_result, world, arm)
                deferred_params.discard(q)
                move_free_deferred.pop(q, None)

            elif metadata.action_type == "retract":
                # RetractHolding: obj, grasp, q_start, traj, q_retract
                # RetractFree: q_start, traj, q_retract
                q_retract = params[-1]
                q_start_param = params[-3]
                if q_start_param not in particles:
                    raise ValueError(f"{q_start_param=} should already be bound")
                if q_retract in particles:
                    raise ValueError(f"{q_retract=} shouldn't already be bound")
                home = torch.tensor(t1_home, device=device)
                particles[q_retract] = home.expand(num_particles, -1).clone()
                log_debug(f"{header}. Initialized q_retract to t1_home")

            else:
                raise NotImplementedError(f"Unsupported operator {ground_op.operator.name}")

        # Resolve trailing MoveFree configs by copying the source.
        unresolved = deferred_params & set(move_free_deferred.keys())
        for q_end in unresolved:
            q_start = move_free_deferred[q_end]
            particles[q_end] = particles[q_start].clone()
            deferred_params.remove(q_end)

        if deferred_params:
            raise RuntimeError(f"Deferred parameters not resolved: {deferred_params}")

        return particles
