# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
import math
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Optional, Tuple, Union

import roma
import torch
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo.types import JointState

from cutamp.config import TAMPConfiguration
from cutamp.costs import curobo_pose_error, dist_from_bounds_jit, sphere_to_sphere_overlap, trajectory_length
from cutamp.joint_limit_cost import trunk_tilt_excess
from cutamp.rollout import Rollout, get_retract_parameters
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import (
    Collision,
    CollisionFree,
    CollisionFreeGrasp,
    CollisionFreeHolding,
    CollisionFreePlacement,
    ComPolygon,
    KinematicConstraint,
    Motion,
    StablePlacement,
)
from cutamp.task_planning.costs import TrajectoryLength
from cutamp.utils.common import transform_spheres


@lru_cache(maxsize=None)
def _device_const(values: tuple, device, dtype=None) -> torch.Tensor:
    """``values`` on ``device``, made once: ``torch.tensor`` from a Python
    sequence is a blocking host-to-device copy. Shared; never mutate. Never
    evicted, since a captured CUDA graph keeps reading its address."""
    return torch.tensor(values, device=device, dtype=dtype)


def _pairwise_upper_sum(x: torch.Tensor) -> torch.Tensor:
    """``[B]``: sums of ``x`` ``[B, n, n]`` over each strict upper triangle, in the triu mask's
    row-major order, so the sums match it. An index gather: the mask would sync on its ``nonzero``."""
    n = x.shape[-1]
    iu = torch.triu_indices(n, n, 1, device=x.device)
    return x[:, iu[0], iu[1]].sum(-1)


def _per_object_sum(x: torch.Tensor, counts: Tuple[int, ...]) -> torch.Tensor:
    """``[B, len(counts)]``: sums of ``x`` ``[B, sum(counts)]`` over consecutive column blocks of
    ``counts[i]`` columns. A fixed-order reduction, bitwise run to run and under graph replay, unlike
    ``scatter_add_``'s atomics; over 128 columns its split also depends on each row's address mod 16
    bytes, fixed for a freshly allocated ``x``. ``counts`` are Python ints, so there is no host read."""
    return torch.stack([block.sum(-1) for block in x.split(counts, dim=1)], dim=1)


class CostFunction:
    """
    Cost Function for a given plan skeleton. Given a rollout, we compute the constraints and costs.
    The __init__ function caches constraints and costs for quicker indexing during rollout evaluation.
    """

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self._rollout_validated = False

        # Accumulate the constraints, so we can batch them up when computing the costs
        self.cfree_constraints = []
        self.kinematic_constraints = []
        self.motion_constraints = []
        self.stable_placement_constraints = []
        self.traj_length_costs = []

        type_to_list = {
            KinematicConstraint.type: self.kinematic_constraints,
            Motion.type: self.motion_constraints,
            CollisionFree.type: self.cfree_constraints,
            CollisionFreeHolding.type: self.cfree_constraints,
            CollisionFreeGrasp.type: self.cfree_constraints,
            CollisionFreePlacement.type: self.cfree_constraints,
            StablePlacement.type: self.stable_placement_constraints,
            TrajectoryLength.type: self.traj_length_costs,
        }
        for ground_op in plan_skeleton:
            for co in [*ground_op.constraints, *ground_op.costs]:
                if co.type in type_to_list:
                    type_to_list[co.type].append(co)
                else:
                    raise NotImplementedError(f"Unhandled constraint or cost: {co}")

        # All conf parameters for motion constraints, we check rollout is subset
        self.motion_conf_params = set(
            iter(itertools.chain.from_iterable(con.params for con in self.motion_constraints))
        )

        self.retract_info = get_retract_parameters(plan_skeleton)

        # Single self-collision cost over the whole robot — both arms in one
        # kinematic chain, so one cost per query covers both.
        self_collision_config = SelfCollisionCostCfg(
            weight=self.world.device_cfg.to_device([1.0]),
            device_cfg=self.world.device_cfg,
            self_collision_kin_config=self.world.kinematics.get_self_collision_config(),
        )
        self.self_collision_cost_fn = SelfCollisionCost(self_collision_config)
        # The retract confs join the joint-limit and self-collision constraints (design 7.3), on a cost of their
        # own: an instance rewrites its output and gradient buffers on every call of one shape.
        self.retract_self_collision_cost_fn = SelfCollisionCost(self_collision_config) if self.retract_info else None
        # The envelope's Trunk tilt bound (call 22), as trunk_tilt_excess's arguments after the confs: the base's pitch
        # and roll columns, and the cosine of the maximum tilt. None when the envelope sets no bound.
        envelope, names = world.model.envelope, world.model.joint_names
        self.trunk_tilt_args = None if envelope.trunk_tilt_max is None else (
            names.index("base_j_pitch"), names.index("base_j_roll"), math.cos(envelope.trunk_tilt_max)
        )

        # Conf parameters for kinematic constraints, order in rollout should match
        self.kinematic_confs, self.kinematic_actions = zip(*(con.params for con in self.kinematic_constraints))
        self.kinematic_confs = list(self.kinematic_confs)
        self.kinematic_actions = list(self.kinematic_actions)

        # Compute the AABB and surface z-position for the placement surfaces.
        # Also precompute the static per-surface placement grouping used by
        # stable_placement_costs: ordered (obj, placement_param) pairs, each
        # object's sphere count, and the per-sphere → per-object index map
        # (sphere counts are static).
        self.surface_to_aabb = {}
        self.surface_to_target_z = {}
        self.surface_to_placements = defaultdict(list)
        for con in self.stable_placement_constraints:
            obj, _, placement, surface = con.params
            self.surface_to_placements[surface].append((obj, placement))
            if surface in self.surface_to_aabb:
                continue

            aabb = world.get_aabb(surface)  # includes xyz
            aabb_xy = aabb[:, :2]
            self.surface_to_aabb[surface] = aabb_xy

            # For the target z, need to take collision activation distance into account
            surface_z = aabb[1, 2]
            target_z = surface_z + world.collision_activation_distance + 2e-3  # add some buffer
            self.surface_to_target_z[surface] = target_z

        self.surface_to_sphere_idx_map = {}
        self.surface_to_sphere_counts: Dict[str, Tuple[int, ...]] = {}
        for surface, placements in self.surface_to_placements.items():
            counts = tuple(world.get_collision_spheres(obj).shape[0] for obj, _ in placements)
            self.surface_to_sphere_counts[surface] = counts
            self.surface_to_sphere_idx_map[surface] = torch.tensor(
                [o_idx for o_idx, n in enumerate(counts) for _ in range(n)], dtype=torch.int64, device=world.device
            )

        # All conf parameters for trajectory length costs, we check rollout matches
        # No support for trajectories right now
        self.traj_length_confs = []
        for cost in self.traj_length_costs:
            q_start, traj, q_end = cost.params
            self.traj_length_confs.append(q_start)
            self.traj_length_confs.append(q_end)
        self.traj_length_confs = list(dict.fromkeys(self.traj_length_confs))  # remove duplicates

        # Drop T1's per-arm initial confs; trajectory length only counts movement
        # between actionable confs.
        self.traj_length_confs = [c for c in self.traj_length_confs if c not in ("left_q0", "right_q0")]

    def _validate_rollout(self, rollout: Rollout):
        """Checks structure of the rollout conforms to the assumptions we make in the cost function implementation."""
        if self._rollout_validated:
            return

        # Configurations should be subset of parameters involved in motion constraints
        if not set(rollout["conf_params"]).issubset(self.motion_conf_params):
            raise RuntimeError(
                f"Missing conf params in motion constraints: {rollout['conf_params'] - self.motion_conf_params}"
            )

        # Kinematic configuration and actions (i.e., poses) should match
        if self.kinematic_confs != rollout["conf_params"]:
            raise RuntimeError(f"Expected conf params {self.kinematic_confs} but got {rollout['conf_params']}")
        if self.kinematic_actions != rollout["action_params"]:
            raise RuntimeError(f"Expected action params {self.kinematic_actions} but got {rollout['action_params']}")

        # Trajectory length parameters: rollout conf_params should be subset of traj_length_confs
        # (traj_length_confs may include trailing MoveFree configs not in rollout)
        if not set(rollout["conf_params"]).issubset(set(self.traj_length_confs)):
            raise RuntimeError(
                f"Rollout conf_params {rollout['conf_params']} is not a subset of "
                f"traj_length_confs {self.traj_length_confs}"
            )

        self._rollout_validated = True

    def kinematic_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Kinematic constraints - i.e., pose error between actual and desired end-effector poses, and both
        soles' error against the stance at every conf, ``sole_pos_err`` and ``sole_rot_err`` ``[N, 2H]``: a conf
        whose feet drifted would pass the CoM gate on a support polygon it no longer stands on (design 7.4)."""
        pos_errs, rot_errs = curobo_pose_error(rollout["world_from_ee"], rollout["world_from_ee_desired"])
        values = {"pos_err": pos_errs, "rot_err": rot_errs}
        sole_pos, sole_rot = curobo_pose_error(rollout["world_from_soles"], self.world.stance_soles)  # [N, H, 2]
        values["sole_pos_err"], values["sole_rot_err"] = sole_pos.flatten(1), sole_rot.flatten(1)
        kinematic_cost = {
            "type": "constraint",
            "constraints": self.kinematic_constraints,
            "values": values,
        }
        return kinematic_cost

    def motion_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Motion constraints — joint-limit violation + self-collision cost, of the rollout's confs and of the
        retract confs too (``retract_joint_limit``, ``retract_self_collision``); with an envelope that bounds the
        tilt, also the rollout confs' Trunk tilt past its maximum (``trunk_tilt``)."""
        confs = rollout["confs"]
        robot_spheres = rollout["robot_spheres"]

        joint_limits = self.world.joint_limits
        dist_from_joint_lims = dist_from_bounds_jit(confs, joint_limits[0], joint_limits[1])

        # Single self-collision cost: setup once per shape, then evaluate.
        b, h = robot_spheres.shape[0], robot_spheres.shape[1]
        self.self_collision_cost_fn.setup_batch_tensors(b, h)
        self_coll_vals = self.self_collision_cost_fn.forward(robot_spheres).squeeze(-1)

        values = {"joint_limit": dist_from_joint_lims, "self_collision": self_coll_vals}
        if self.retract_self_collision_cost_fn is not None and self._particles is not None:
            values.update(self._retract_motion_values())
        if self.trunk_tilt_args is not None:
            # ponytail: the retract confs are left out: their base and body are locked at home, which stands upright.
            values["trunk_tilt"] = trunk_tilt_excess(confs, *self.trunk_tilt_args)
        motion_cost = {
            "type": "constraint",
            "constraints": self.motion_constraints,
            "values": values,
        }
        return motion_cost

    def _retract_motion_values(self) -> Dict[str, torch.Tensor]:
        """``retract_joint_limit`` and ``retract_self_collision``, each ``[N, R]`` over ``retract_info``'s confs,
        from one FK of every retract conf on its own site (TAMPWorld.site_kinematics)."""
        q = torch.stack([self._particles[name] for name, _, _ in self.retract_info], dim=1)  # [N, R, dof]
        n, r = q.shape[0], q.shape[1]
        kin_state = self.world.site_kinematics("retract_motion").compute_kinematics(
            JointState.from_position(q.view(n * r, -1))
        )
        joint_limits = self.world.joint_limits
        self.retract_self_collision_cost_fn.setup_batch_tensors(n, r)
        return {
            "retract_joint_limit": dist_from_bounds_jit(q, joint_limits[0], joint_limits[1]),
            "retract_self_collision": self.retract_self_collision_cost_fn.forward(
                kin_state.robot_spheres.view(n, r, -1, 4)
            ).squeeze(-1),
        }

    def stable_placement_costs(
        self, rollout: Rollout, obj_to_spheres: Dict[str, torch.Tensor]  # each [b, t, n, 4]
    ) -> Union[dict, None]:
        """
        Stable Placement constraints. Converted to two costs:
            1. Object sphere xy positions within surface AABB
            2. Minimum object sphere is supported by the surface (compute distance between surface and bottom of sphere)
        """
        if not self.stable_placement_constraints:
            return None

        num_particles = rollout["num_particles"]
        support_vals = {}
        for surface, placements in self.surface_to_placements.items():
            # Per-call pose work: each object's spheres at its placement timestep.
            surface_spheres = [
                obj_to_spheres[obj][:, rollout["action_to_pose_ts"][placement]]
                for obj, placement in placements
            ]
            # Static sphere-index → object-index map (precomputed in __init__).
            sphere_idx_map = self.surface_to_sphere_idx_map[surface]
            sphere_idx_map_expand = sphere_idx_map[None].expand(num_particles, -1)  # expand by batch size

            # Since objects can have different numbers of spheres, we need to concatenate instead of stack
            spheres = torch.cat(surface_spheres, dim=1)
            spheres_xy = spheres[..., :2]

            # Within goal xy bounds, summed per object in a fixed order (_per_object_sum)
            in_goal_xy = dist_from_bounds_jit(spheres_xy, *self.surface_to_aabb[surface])
            support_vals[f"{surface}_in_xy"] = _per_object_sum(in_goal_xy, self.surface_to_sphere_counts[surface])

            # Distance between bottom of spheres and z-position of the surface
            spheres_bottom = spheres[..., 2] - spheres[..., 3]
            obj_bottom = torch.full(
                (num_particles, len(placements)), float("inf"), dtype=spheres_bottom.dtype, device=spheres_bottom.device
            )
            obj_bottom.scatter_reduce_(1, sphere_idx_map_expand, spheres_bottom, reduce="amin")
            target_z = self.surface_to_target_z[surface]
            support_vals[f"{surface}_support"] = torch.abs(obj_bottom - target_z)

        stable_placement_cost = {
            "type": "constraint",
            "constraints": self.stable_placement_constraints,
            "values": support_vals,
        }
        return stable_placement_cost

    def trajectory_costs(self, rollout: Rollout) -> dict:
        """Trajectory costs, just joint space distance between configurations for now."""
        traj_cost = {
            "type": "cost",
            "costs": self.traj_length_costs,
            "values": {"traj_length": trajectory_length(rollout["confs"])},
        }
        return traj_cost

    def collision_costs(self, rollout: Rollout, obj_to_spheres: Dict[str, torch.Tensor]) -> dict:  # each [b, t, n, 4]
        """Collision costs. This could be tied better to the constraints and sped up significantly."""
        # Robot to world
        robot_spheres = rollout["robot_spheres"]
        coll_values = {"robot_to_world": self.world.collision_fn(robot_spheres)}

        # Collision between movables and world
        all_movable_spheres = torch.cat(list(obj_to_spheres.values()), dim=2)
        coll_values["movable_to_world"] = self.world.collision_fn(all_movable_spheres)

        # Collision between robot and movables, need to expand poses to full timesteps
        all_pose_ts = _device_const(tuple(rollout["ts_to_pose_ts"].values()), all_movable_spheres.device)
        coll_values["robot_to_movables"] = sphere_to_sphere_overlap(
            robot_spheres,
            all_movable_spheres[:, all_pose_ts],
            activation_distance=self.config.gripper_activation_distance,
        )

        # TODO: this is slow and a bottleneck, could consider using curobo's fast sphere-to-sphere kernel
        # Collision between movable objects
        for obj_1, obj_2 in itertools.combinations(self.world.movables, 2):
            obj_1_spheres = obj_to_spheres[obj_1.name]
            obj_2_spheres = obj_to_spheres[obj_2.name]
            coll_values[f"{obj_1.name}_to_{obj_2.name}"] = sphere_to_sphere_overlap(
                obj_1_spheres,
                obj_2_spheres,
                activation_distance=self.config.movable_activation_distance,
                use_aabb_check=True,
            )

        # Retract collision checking - ensures retract configs are collision-free
        # RetractFree: checks world + all movables
        # RetractHolding: checks world + movables excluding held object
        if self.retract_info and self._particles is not None:
            # One FK for every retract conf, on its own site (TAMPWorld.site_kinematics).
            q_retracts = torch.cat([self._particles[name] for name, _, _ in self.retract_info])
            num_particles = self._particles[self.retract_info[0][0]].shape[0]
            kin_model = self.world.site_kinematics("retract")
            all_retract_spheres = kin_model.compute_kinematics(JointState.from_position(q_retracts)).robot_spheres
            for (q_retract_name, is_holding, held_obj), retract_spheres in zip(
                self.retract_info, all_retract_spheres.split(num_particles)  # each (b, h, n, 4)
            ):
                # Robot to world collision - keep shape (num_particles, 1) for consistency
                coll_values[f"{q_retract_name}_world"] = self.world.collision_fn(retract_spheres)
                
                # Robot to movables collision (exclude held object for RetractHolding)
                movable_list = []
                for obj_name, obj_spheres in obj_to_spheres.items():
                    if is_holding and obj_name == held_obj:
                        continue  # Skip held object for RetractHolding
                    movable_list.append(obj_spheres[:, -1])  # Last pose timestep
                
                if movable_list:
                    all_movables = torch.cat(movable_list, dim=1).unsqueeze(1)
                    coll_values[f"{q_retract_name}_movables"] = sphere_to_sphere_overlap(
                        retract_spheres, all_movables,
                        activation_distance=self.config.gripper_activation_distance
                    )
                else:
                    coll_values[f"{q_retract_name}_movables"] = torch.zeros((num_particles, 1), device=q_retracts.device)

        coll_cost = {
            "type": "constraint",
            "constraints": self.cfree_constraints,
            "values": coll_values,
        }
        return coll_cost

    def _com_polygon_penalties(self) -> Dict[str, torch.Tensor]:
        """Lazily compute and cache the COM-polygon penalties dict.

        Sole consumer is ``com_polygon_constraint``; cached so the FK-based
        computation runs at most once per cost evaluation.
        """
        if self._com_polygon_penalties_cache is None:
            from cutamp.com_polygon_cost import compute_com_polygon_penalties
            self._com_polygon_penalties_cache = compute_com_polygon_penalties(
                self.world, self._particles,
            )
        return self._com_polygon_penalties_cache

    def com_polygon_constraint(self) -> Union[dict, None]:
        """Hard COM-over-base-polygon filter, per conf.

        Mirrors how Collision works: per-conf continuous penalty values
        (units: m²), ANDed into the overall satisfying mask by
        ``ConstraintChecker.get_mask``. The penalty is the same
        differentiable inside-barrier function used by the soft cost
        (sharing the FK computation via
        ``self._com_polygon_penalties_cache``), so Adam receives a real
        gradient toward COM-feasibility instead of a dead-weight 0/1
        Boolean.
        """
        if self._particles is None:
            return None

        pens = self._com_polygon_penalties()
        if not pens:
            return None

        return {
            "type": "constraint",
            "constraints": [],
            "values": dict(pens),  # shallow copy: ConstraintChecker iterates this dict
        }

    def soft_costs(self, rollout: Rollout) -> dict:
        """Soft costs defined on the goal state. Supports multiple soft costs."""
        values = {}
        
        for cost_name in self.config.soft_cost:
            cost_value = self._compute_soft_cost(cost_name, rollout)
            values[cost_name] = cost_value
        
        return {"type": "cost", "constraints": [], "values": values}
    
    def _compute_soft_cost(self, cost_name: str, rollout: Rollout) -> torch.Tensor:
        """Compute a single soft cost by name."""
        device = self.world.device
        num_particles = rollout["num_particles"]
        
        # Object position-based costs
        if cost_name in ("dist_from_origin", "place_close_to_base", "max_obj_dist", "min_obj_dist", "min_y", "max_y"):
            last_obj_position = [v[:, -1, :3, 3] for v in rollout["obj_to_pose"].values()]
            last_obj_position = torch.stack(last_obj_position, dim=1)

            if cost_name == "dist_from_origin":
                return -last_obj_position.norm(dim=-1).sum(dim=-1)
            elif cost_name == "place_close_to_base":
                # Pull placed objects toward the base (locked at the origin):
                # penalize each movable's final XY distance. Far placements
                # (hand reach >~0.57 m) are where the COM cannot be fully
                # centered even with leg recruitment; this biases the chosen
                # placement away from that tail while the hard constraints
                # keep it stable and in-region.
                return last_obj_position[..., :2].norm(dim=-1).sum(dim=-1)
            elif cost_name in ("max_obj_dist", "min_obj_dist"):
                all_obj_dists = torch.cdist(last_obj_position, last_obj_position, p=2)
                dists_sum = _pairwise_upper_sum(all_obj_dists)
                return -dists_sum if cost_name == "max_obj_dist" else dists_sum
            else:  # min_y or max_y
                last_y = last_obj_position[..., 1].sum(dim=-1)
                return last_y if cost_name == "min_y" else -last_y
        
        # Yaw alignment cost
        elif cost_name == "align_yaw":
            last_obj_mat3x3 = torch.stack([v[:, -1, :3, :3] for v in rollout["obj_to_pose"].values()], dim=1)
            last_obj_yaw = roma.rotmat_to_euler("XYZ", last_obj_mat3x3)[..., 2]
            yaw_diffs = last_obj_yaw[:, :, None] - last_obj_yaw[:, None, :]
            yaw_diffs = torch.atan2(torch.sin(yaw_diffs), torch.cos(yaw_diffs)).abs()
            return _pairwise_upper_sum(yaw_diffs)
        
        # Retract close to home — penalize body slice (post-base) deviation
        # from the model's home pose.
        elif cost_name == "retract_close_to_home":
            if self._particles is None:
                raise RuntimeError("Particles must be provided for retract_close_to_home soft cost")

            model = self.world.model
            home_t = _device_const(model.home, device)
            # Penalize body (leg+torso+arms) deviation; ignore the base, which comes first.
            body_slice = slice(model.base.stop, len(model.home))
            home_body = home_t[body_slice]

            retract_param_names = [info[0] for info in self.retract_info]
            total_dist = torch.zeros(num_particles, device=device)
            if retract_param_names:
                qs = torch.stack([self._particles[p][:, body_slice] for p in retract_param_names], dim=1)
                total_dist += (qs - home_body).abs().sum(dim=(1, 2))
            return total_dist

        # Minimize body movement — keep leg + torso joints close to home values.
        elif cost_name == "minimize_body_movement":
            if self._particles is None:
                raise RuntimeError("Particles must be provided for minimize_body_movement soft cost")

            body = self.world.model.body
            home_body = _device_const(self.world.model.home[body], device)

            config_params = [p for p in self._particles if p.startswith(("left_q", "right_q", "q"))]
            if config_params:
                all_configs = torch.stack([self._particles[p] for p in config_params], dim=1)
                return (all_configs[:, :, body] - home_body).pow(2).sum(dim=(1, 2))
            return torch.zeros(num_particles, device=device)

        # Joint limits — a hinge² penalty on endpoint configurations past the
        # position limits, which are already the policy's soft limits (no
        # margin band, call 17). q0s are excluded inside the helper: the
        # initial robot state must never acquire limit gradients (Phase-2
        # LBFGS re-leafs every cloned particle).
        elif cost_name == "joint_limit_margin":
            if self._particles is None:
                raise RuntimeError("Particles must be provided for joint_limit_margin soft cost")

            from cutamp.joint_limit_cost import joint_limit_margin_soft_cost

            limits = self.world.joint_limits  # [2, dof], row 0 = lower
            return joint_limit_margin_soft_cost(self._particles, limits, num_particles, device)

        else:
            raise ValueError(f"Unsupported soft cost: {cost_name}")

    def __call__(self, rollout: Rollout, particles: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, dict]:
        self._validate_rollout(rollout)
        cost_dict = {}
        
        self._particles = particles
        # Reset per-call cache for the shared COM-polygon penalty helper.
        # com_polygon_constraint() and _compute_soft_cost("com_polygon")
        # both consult this; lazy-populated on first use so we only run
        # the FK once per __call__ even when both are active.
        self._com_polygon_penalties_cache: Optional[Dict[str, torch.Tensor]] = None

        def add_cost(k_, v_):
            if v_ is not None:
                cost_dict[k_] = v_

        traj_cost = self.trajectory_costs(rollout)
        add_cost(TrajectoryLength.type, traj_cost)

        # Get collision spheres for movable objects
        obj_to_spheres = {}
        for obj in self.world.movables:
            if obj.name in obj_to_spheres:
                raise RuntimeError(f"Object {obj.name} already in obj_to_spheres")
            obj_pose = rollout["obj_to_pose"][obj.name]
            obj_spheres = transform_spheres(self.world.get_collision_spheres(obj), obj_pose)
            obj_to_spheres[obj.name] = obj_spheres

        collision_cost = self.collision_costs(rollout, obj_to_spheres)
        add_cost(Collision.type, collision_cost)

        stable_placement_cost = self.stable_placement_costs(rollout, obj_to_spheres)
        add_cost(StablePlacement.type, stable_placement_cost)

        # Valid motions don't exceed joint limits
        motion_cost = self.motion_costs(rollout)
        add_cost(Motion.type, motion_cost)

        kinematic_cost = self.kinematic_costs(rollout)
        add_cost(KinematicConstraint.type, kinematic_cost)

        # COM-over-base-polygon hard filter (mirrors Collision)
        com_polygon_cost = self.com_polygon_constraint()
        add_cost(ComPolygon.type, com_polygon_cost)

        # Soft costs
        if self.config.soft_cost:
            soft_cost = self.soft_costs(rollout)
            add_cost("soft", soft_cost)

        return cost_dict
