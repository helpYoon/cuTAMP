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

import roma
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
    fixed: Optional[Dict[str, Pose]] = None,
) -> GoalToolPose:
    """Build a GoalToolPose covering all of ``host.kinematics.tool_frames``.

    ``host`` may be a ``MotionPlanner`` or an ``IKSolver`` — both expose
    ``kinematics.tool_frames`` and ``compute_kinematics(state)``.

    The active frame gets ``target_pose``, and each frame in ``fixed`` its
    pose there (the planted soles' stance, ``TAMPWorld.sole_goals``); every
    other inactive frame gets its current FK pose at ``current_state``. This
    satisfies cuRobo IK's ``reorder_links`` requirement that the goal cover
    every kinematics tool frame, and holds each inactive hand where it is in
    the world.
    """
    all_frames: List[str] = list(host.kinematics.tool_frames)
    poses: Dict[str, Pose] = {**(fixed or {}), active_tool_frame: target_pose}
    inactive = [f for f in all_frames if f not in poses]
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

    On a model with soles (the real legs) the soles' goals are their stance
    poses (``world.sole_goals``), never their FK at the IK's start conf
    (design 7.3), so a retry's noisy seed cannot move them.

    The multi-frame goal targets the inactive frame at its current FK pose,
    so the inactive hand stays put in the world while its arm's joints
    counter-rotate as the body bends (about 1 rad per approach in blocks_t1).
    Without it, IK leaves the inactive arm unconstrained and it drifts to
    arbitrary joint values, polluting downstream particle optimization. This
    goal is the only hold on the inactive hand (known-bugs B15): the cspace
    pin below is a no-op (``T1State.pin_for_arm_action``), the GN refreshes
    keep the inactive arm's joints fixed while they move the body, and the
    motion planner holds nothing between confs.

    ``current_state_q`` (full-cspace ``[B, len(world.kinematics.joint_names)]``)
    anchors the inactive-frame goal at a non-home pose. cuRobo's
    ``solve_pose`` also makes it the first optimization seed when no
    ``seed_config`` is given (fork solver_ik.py:735-736; the remaining seeds
    are random), so ``seed_from_current=True``, which passes that seed
    explicitly, states the warm start rather than adding it. Used by the
    coupled-reIK refresh, where the previous solution is a near-answer.

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
    goal = _build_multi_frame_goal(
        ik, active_frame, target_pose, current_state, fixed=world.sole_goals(batch_size),
    )

    snapshot = snapshot_cspace_target_dof_weight([ik])
    try:
        if arm is not None:
            # Weight the inactive arm's DOFs in cuRobo's cspace target term: a
            # no-op (``cspace_target_weight`` is 0, and that term's target is a
            # zero joint vector, not these joints' current values). The inactive
            # frame's target in ``goal`` is what holds that hand in the world.
            model = world.model
            write_cspace_target_dof_weight(
                [ik], build_pin_weights(
                    ik, model.joint_names[model.right_arm if arm == "left" else model.left_arm],
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


#: A retry's leg noise on a model with soles (ruling R21): each pitch joint draws once per pair, left onto right,
#: and roll and yaw hold q_init's, so the legs stay mirrored.
_MIRRORED_LEG_JOINTS = ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch")
_HELD_LEG_JOINTS = ("Hip_Roll", "Hip_Yaw", "Ankle_Roll")
#: The floating base's joints in ``model.base`` order: x, y, z, then YXZ Euler (build_planner_model.py's ``BASE``).
_BASE_JOINTS = ("base_j_x", "base_j_y", "base_j_z", "base_j_pitch", "base_j_roll", "base_j_yaw")


def _retry_seed(world: TAMPWorld, batch_size: int, arm: Optional[str], noise_std: float) -> torch.Tensor:
    """``[batch_size, dof]``: a CoM-gate retry's start conf, ``world.q_init``
    plus Gaussian noise of std ``noise_std`` rad on the body and the active
    arm, clamped to the joint limits. The inactive arm keeps ``q_init``, and
    so does the base of the simplified model, which is its root.

    On a model with soles (the real legs) the root is the trunk, so that noise
    would swing the soles. There the stance is the noise's root instead (ruling
    R21): the legs draw each pitch once per pair and hold roll and yaw, and the
    base is re-solved so that the first sole sits on its stance pose
    (``_stand_on_the_stance``).
    """
    model = world.model
    full_dof = world.q_init.shape[-1]
    device = world.q_init.device
    seed = world.q_init.unsqueeze(0).expand(batch_size, -1).clone()
    noise = torch.randn(batch_size, full_dof, device=device) * noise_std
    noise[:, model.base] = 0.0  # base locked
    if arm == "left":
        noise[:, model.right_arm] = 0.0  # right arm locked
    elif arm == "right":
        noise[:, model.left_arm] = 0.0  # left arm locked
    if model.sole_frames:
        # ponytail: holds while q_init's legs mirror each other with zero roll and yaw (home, as setup_cutamp sets
        # it); walking (stage 5) revisits it.
        col = list(model.joint_names).index
        for joint in _MIRRORED_LEG_JOINTS:
            noise[:, col(f"Right_{joint}")] = noise[:, col(f"Left_{joint}")]
        for joint in _HELD_LEG_JOINTS:
            for side in ("Left", "Right"):
                noise[:, col(f"{side}_{joint}")] = 0.0
    seed = seed + noise
    # Clamp the perturbed seed to joint limits so large-sigma retries don't
    # hand cuRobo an out-of-limits current_state. Limits come from the same
    # full-cspace kinematics that defines world.q_init's joint order
    # (JointLimits.joint_names == kinematics.joint_names, verified [2, full_dof]).
    # The left and right pitch limits are equal, so the clamp keeps the legs mirrored.
    _limits = world.kinematics.get_joint_limits().position  # [2, full_dof]
    _lower = _limits[0].to(seed.device, seed.dtype)
    _upper = _limits[1].to(seed.device, seed.dtype)
    seed = torch.clamp(seed, min=_lower, max=_upper)
    if model.sole_frames:
        seed = _stand_on_the_stance(world, seed, _lower, _upper)
    return seed


def _stand_on_the_stance(
    world: TAMPWorld, seed: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
) -> torch.Tensor:
    """``seed`` with each row's base re-solved so that the first of
    ``model.sole_frames`` sits on its stance pose: ``B' = S @ inv(W) @ B``,
    with ``S`` the stance pose, ``W`` the sole's FK pose and ``B`` the base
    pose. With mirrored legs (``_retry_seed``) the other sole's pose relative
    to the first is unchanged, so it lands on its stance pose too. A row whose
    ``B'`` leaves the base joints' limits takes home's base and body, keeping
    only its arm's noise.
    """
    model = world.model
    if tuple(model.joint_names[model.base]) != _BASE_JOINTS:
        raise ValueError(f"expected the base joints {_BASE_JOINTS}, got {model.joint_names[model.base]}")
    kin = world.kinematics
    state = kin.compute_kinematics(JointState.from_position(seed, joint_names=list(kin.joint_names)))
    # get_matrix copies the pose out of the FK state: cuRobo reuses its output buffers.
    sole = state.tool_poses.get_link_pose(model.sole_frames[0], make_contiguous=True).get_matrix().reshape(-1, 4, 4)
    stance = world.stance_soles[0]
    q_base = seed[:, model.base]
    # B' = S @ inv(W) @ B, as rotation and translation.
    rot = stance[:3, :3] @ sole[:, :3, :3].transpose(-1, -2)
    new_rot = rot @ roma.euler_to_rotmat("YXZ", q_base[:, 3:])
    new_pos = (rot @ (q_base[:, :3] - sole[:, :3, 3]).unsqueeze(-1)).squeeze(-1) + stance[:3, 3]
    new_base = torch.cat([new_pos, roma.rotmat_to_euler("YXZ", new_rot)], dim=-1)
    inside = ((lower[model.base] <= new_base) & (new_base <= upper[model.base])).all(-1, keepdim=True)
    home = torch.tensor(model.home, device=seed.device, dtype=seed.dtype)
    out = seed.clone()
    out[:, model.base] = torch.where(inside, new_base, home[model.base])
    out[:, model.body] = torch.where(inside, seed[:, model.body], home[model.body])
    return out


def _ik_for_pose_com_safe(
    world: TAMPWorld,
    world_from_ee: torch.Tensor,
    arm: Optional[str],
    *,
    max_retries: int = 3,
    context: str = "",
):
    """Call ``_ik_for_pose``; verify COM-in-polygon; retry failed particles.

    A best-effort backstop behind the CoM-aware seed-IK residual (Layer 1,
    cuRobo fork). Each of up to ``max_retries`` rounds re-solves the batch
    from a noisier start conf (``_retry_seed``) and copies the new results,
    success flags included, into the rows still outside, whether or not
    they land inside. Rows that fail every retry are kept: the caller's
    ``_store_ik_q`` resets only IK failures, to home (``world.q_init``). On
    blocks_t1 about half the IK-solved place rows stay outside after 3
    retries, on either model (ruling R24). The CoM is enforced downstream,
    by GN's balance-first projection (``gn_refresh.gn_project_balanced``)
    and by the balance gate on every trajectory frame
    (``motion_solver._balance_gate``).

    Returns the last batch result. Logs the final count of rows outside at
    WARNING level so persistent failures are visible.
    """
    from cutamp.com_polygon_cost import compute_com_polygon_mask
    result = _ik_for_pose(world, world_from_ee, arm)
    batch_size = world_from_ee.shape[0]
    for _attempt in range(max_retries):
        q_batch = _ik_solution_to_full_q(result, world)
        in_polygon = compute_com_polygon_mask(world, q_batch)
        if bool(in_polygon.all()):
            return result
        # Retry from a noisier start conf each round: noise of 0.1, 0.2, 0.3,
        # ... rad on the body and the active arm (_retry_seed). It reaches the
        # IK two ways. solve_pose makes current_state the first of the LM seed
        # IK's starts (fork solver_ik.py:735-736, seed_ik_solver.py:511-517;
        # only the seed manager's own current_state argument is ignored,
        # manager_seed.py:98-100), the rest being random. And it moves the
        # goals that _build_multi_frame_goal builds from current_state: on both
        # models, the idle hand's target moves with the trunk. Both can send
        # the solver to other basins than the COM-infeasible "lean forward +
        # tip" one.
        seed = _retry_seed(world, batch_size, arm, 0.1 * (_attempt + 1))
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


def _ik_for_pose_no_gate(world, world_from_ee, arm, *, context: str = ""):
    """Ablation (enable_com_init_gate=False): plain IK, no CoM gate or
    retries — layer L2 off. Same call shape as _ik_for_pose_com_safe."""
    return _ik_for_pose(world, world_from_ee, arm)


def _select_init_ik(config):
    """The init-IK entry point for this run's config (L2 on/off)."""
    return _ik_for_pose_com_safe if config.enable_com_init_gate else _ik_for_pose_no_gate


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
    so values stay consistent without an external home reference.
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
                ik_result = _select_init_ik(config)(
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
                ik_result = _select_init_ik(config)(
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
                home = torch.tensor(world.model.home, device=device)
                particles[q_retract] = home.expand(num_particles, -1).clone()
                log_debug(f"{header}. Initialized q_retract to the model's home")

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
