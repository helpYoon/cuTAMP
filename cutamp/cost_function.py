# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Tuple, Union

import torch
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo.types import JointState

from cutamp.com_polygon_cost import compute_com_polygon_penalties
from cutamp.config import TAMPConfiguration
from cutamp.costs import curobo_pose_error, dist_from_bounds_jit, sphere_to_sphere_overlap, trajectory_length
from cutamp.joint_limit_cost import joint_limit_margin_soft_cost
from cutamp.rollout import Rollout, get_retract_parameters
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import (
    Collision,
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


def _per_object_sum(x: torch.Tensor, counts: Tuple[int, ...]) -> torch.Tensor:
    """``[B, len(counts)]``: sums of ``x`` ``[B, sum(counts)]`` over consecutive column blocks of
    ``counts[i]`` columns. A fixed-order reduction, bitwise run to run and under graph replay, unlike
    ``scatter_add_``'s atomics; over 128 columns its split also depends on each row's address mod 16
    bytes, fixed for a freshly allocated ``x``. ``counts`` are Python ints, so there is no host read."""
    return torch.stack([block.sum(-1) for block in x.split(counts, dim=1)], dim=1)


class CostFunction:
    """
    Cost Function for a given plan skeleton. Given a rollout, we compute the constraints and costs.
    The __init__ function precomputes the plan's static placement grouping and retract confs.
    """

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config

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

        # Compute the AABB and surface z-position for the placement surfaces of the plan's StablePlacement
        # constraints. Also precompute the static per-surface placement grouping used by
        # stable_placement_costs: ordered (obj, placement_param) pairs, each
        # object's sphere count, and the per-sphere → per-object index map
        # (sphere counts are static).
        self.surface_to_aabb = {}
        self.surface_to_target_z = {}
        self.surface_to_placements = defaultdict(list)
        stable_placements = [con for op in plan_skeleton for con in op.constraints if con.type == StablePlacement.type]
        for con in stable_placements:
            obj, _, placement, surface = con.params
            self.surface_to_placements[surface].append((obj, placement))
            if surface in self.surface_to_aabb:
                continue

            aabb = world.get_aabb(surface)  # includes xyz
            aabb_xy = aabb[:, :2]
            self.surface_to_aabb[surface] = aabb_xy

            # The target z: the surface's top plus a buffer (the world collision cost activates at 0)
            surface_z = aabb[1, 2]
            target_z = surface_z + 2e-3
            self.surface_to_target_z[surface] = target_z

        self.surface_to_sphere_idx_map = {}
        self.surface_to_sphere_counts: Dict[str, Tuple[int, ...]] = {}
        for surface, placements in self.surface_to_placements.items():
            counts = tuple(world.get_collision_spheres(obj).shape[0] for obj, _ in placements)
            self.surface_to_sphere_counts[surface] = counts
            self.surface_to_sphere_idx_map[surface] = torch.tensor(
                [o_idx for o_idx, n in enumerate(counts) for _ in range(n)], dtype=torch.int64, device=world.device
            )

    def kinematic_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Kinematic constraints - i.e., pose error between actual and desired end-effector poses, and both
        soles' error against the stance at every conf, ``sole_pos_err`` and ``sole_rot_err`` ``[N, 2H]``: a conf
        whose feet drifted would pass the CoM gate on a support polygon it no longer stands on (design 7.4)."""
        pos_errs, rot_errs = curobo_pose_error(rollout["world_from_ee"], rollout["world_from_ee_desired"])
        values = {"pos_err": pos_errs, "rot_err": rot_errs}
        sole_pos, sole_rot = curobo_pose_error(rollout["world_from_soles"], self.world.stance_soles)  # [N, H, 2]
        values["sole_pos_err"], values["sole_rot_err"] = sole_pos.flatten(1), sole_rot.flatten(1)
        return {"type": "constraint", "values": values}

    def motion_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Motion constraints — joint-limit violation + self-collision cost, of the rollout's confs and of the
        retract confs too (``retract_joint_limit``, ``retract_self_collision``)."""
        confs = rollout["confs"]
        robot_spheres = rollout["robot_spheres"]

        joint_limits = self.world.joint_limits
        dist_from_joint_lims = dist_from_bounds_jit(confs, joint_limits[0], joint_limits[1])

        # Single self-collision cost: setup once per shape, then evaluate.
        b, h = robot_spheres.shape[0], robot_spheres.shape[1]
        self.self_collision_cost_fn.setup_batch_tensors(b, h)
        self_coll_vals = self.self_collision_cost_fn.forward(robot_spheres).squeeze(-1)

        values = {"joint_limit": dist_from_joint_lims, "self_collision": self_coll_vals}
        if self.retract_self_collision_cost_fn is not None:
            values.update(self._retract_motion_values())
        return {"type": "constraint", "values": values}

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
        if not self.surface_to_placements:
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

        return {"type": "constraint", "values": support_vals}

    def trajectory_costs(self, rollout: Rollout) -> dict:
        """Trajectory costs, just joint space distance between configurations for now."""
        return {"type": "cost", "values": {"traj_length": trajectory_length(rollout["confs"])}}

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
        coll_values["robot_to_movables"] = sphere_to_sphere_overlap(robot_spheres, all_movable_spheres[:, all_pose_ts])

        # TODO: this is slow and a bottleneck, could consider using curobo's fast sphere-to-sphere kernel
        # Collision between movable objects
        for obj_1, obj_2 in itertools.combinations(self.world.movables, 2):
            obj_1_spheres = obj_to_spheres[obj_1.name]
            obj_2_spheres = obj_to_spheres[obj_2.name]
            coll_values[f"{obj_1.name}_to_{obj_2.name}"] = sphere_to_sphere_overlap(
                obj_1_spheres, obj_2_spheres, use_aabb_check=True,
            )

        # Retract collision checking - ensures retract configs are collision-free
        # RetractFree: checks world + all movables
        # RetractHolding: checks world + movables excluding held object
        if self.retract_info:
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
                    coll_values[f"{q_retract_name}_movables"] = sphere_to_sphere_overlap(retract_spheres, all_movables)
                else:
                    coll_values[f"{q_retract_name}_movables"] = torch.zeros((num_particles, 1), device=q_retracts.device)

        return {"type": "constraint", "values": coll_values}

    def com_polygon_constraint(self) -> Union[dict, None]:
        """Hard COM-over-base-polygon filter, per conf.

        Mirrors how Collision works: per-conf continuous penalty values
        (units: m²), ANDed into the overall satisfying mask by
        ``ConstraintChecker.get_mask``. The penalty is a differentiable
        inside-barrier function, so Adam receives a real gradient toward
        COM-feasibility instead of a dead-weight 0/1 Boolean.
        """
        pens = compute_com_polygon_penalties(self.world, self._particles)
        if not pens:
            return None
        return {"type": "constraint", "values": pens}

    def soft_costs(self, rollout: Rollout) -> dict:
        """Soft costs defined on the goal state. Supports multiple soft costs."""
        values = {}

        for cost_name in self.config.soft_cost:
            cost_value = self._compute_soft_cost(cost_name, rollout)
            values[cost_name] = cost_value

        return {"type": "cost", "values": values}

    def _compute_soft_cost(self, cost_name: str, rollout: Rollout) -> torch.Tensor:
        """Compute a single soft cost by name."""
        device = self.world.device
        num_particles = rollout["num_particles"]

        if cost_name == "place_close_to_base":
            # Pull placed objects toward the base (locked at the origin):
            # penalize each movable's final XY distance. Far placements
            # (hand reach >~0.57 m) are where the COM cannot be fully
            # centered even with leg recruitment; this biases the chosen
            # placement away from that tail while the hard constraints
            # keep it stable and in-region.
            last_obj_position = torch.stack([v[:, -1, :3, 3] for v in rollout["obj_to_pose"].values()], dim=1)
            return last_obj_position[..., :2].norm(dim=-1).sum(dim=-1)

        # Retract close to home — penalize body slice (post-base) deviation
        # from the model's home pose.
        elif cost_name == "retract_close_to_home":
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
            body = self.world.model.body
            home_body = _device_const(self.world.model.home[body], device)

            config_params = [p for p in self._particles if p.startswith(("left_q", "right_q", "q"))]
            if config_params:
                all_configs = torch.stack([self._particles[p] for p in config_params], dim=1)
                return (all_configs[:, :, body] - home_body).pow(2).sum(dim=(1, 2))
            return torch.zeros(num_particles, device=device)

        # Joint limits — a hinge² penalty on endpoint configurations past the
        # position limits, which are already the policy's soft limits, cut by
        # the envelope (no margin band, call 17). q0s are excluded inside the helper: the
        # initial robot state must never acquire limit gradients.
        elif cost_name == "joint_limit_margin":
            limits = self.world.joint_limits  # [2, dof], row 0 = lower
            return joint_limit_margin_soft_cost(self._particles, limits, num_particles, device)

        else:
            raise ValueError(f"Unsupported soft cost: {cost_name}")

    def __call__(self, rollout: Rollout, particles: Dict[str, torch.Tensor]) -> Dict[str, dict]:
        cost_dict = {}
        self._particles = particles

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
