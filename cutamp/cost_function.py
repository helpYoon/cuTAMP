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
from typing import Dict, Optional, Union

import roma
import torch
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo.types import JointState
from jaxtyping import Float

from cutamp.config import TAMPConfiguration
from cutamp.costs import curobo_pose_error, dist_from_bounds_jit, sphere_to_sphere_overlap, trajectory_length
from cutamp.rollout import Rollout, get_conf_to_arm, get_retract_parameters
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
    ValidPush,
    ValidPushStick,
)
from cutamp.task_planning.costs import GraspCost, RetractCost, TrajectoryLength
from cutamp.utils.common import transform_spheres


class CostFunction:
    """
    Cost Function for a given plan skeleton. Given a rollout, we compute the constraints and costs.
    The __init__ function caches constraints and costs for quicker indexing during rollout evaluation.
    """

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectories not supported in cost function yet")

        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self._rollout_validated = False

        # Accumulate the constraints, so we can batch them up when computing the costs
        self.cfree_constraints = []
        self.kinematic_constraints = []
        self.motion_constraints = []
        self.stable_placement_constraints = []
        self.valid_push_constraints = []
        self.valid_push_stick_constraints = []
        self.traj_length_costs = []

        type_to_list = {
            KinematicConstraint.type: self.kinematic_constraints,
            Motion.type: self.motion_constraints,
            CollisionFree.type: self.cfree_constraints,
            CollisionFreeHolding.type: self.cfree_constraints,
            CollisionFreeGrasp.type: self.cfree_constraints,
            CollisionFreePlacement.type: self.cfree_constraints,
            StablePlacement.type: self.stable_placement_constraints,
            ValidPush.type: self.valid_push_constraints,
            ValidPushStick.type: self.valid_push_stick_constraints,
            TrajectoryLength.type: self.traj_length_costs,
        }
        # GraspCost / RetractCost are recognized but contribute zero (covered
        # by IK convergence + retract-init at t1_home). Skip silently.
        ignored_co = {GraspCost.type, RetractCost.type}
        for ground_op in plan_skeleton:
            for co in [*ground_op.constraints, *ground_op.costs]:
                if co.type in type_to_list:
                    type_to_list[co.type].append(co)
                elif co.type not in ignored_co:
                    raise NotImplementedError(f"Unhandled constraint or cost: {co}")

        # All conf parameters for motion constraints, we check rollout is subset
        self.motion_conf_params = set(
            iter(itertools.chain.from_iterable(con.params for con in self.motion_constraints))
        )

        # Tag confs by arm so soft costs / book-keeping can dispatch per-arm.
        self.conf_to_arm = get_conf_to_arm(plan_skeleton)
        self.retract_info = get_retract_parameters(plan_skeleton)

        # Single self-collision cost over the whole robot — both arms in one
        # kinematic chain, so one cost per query covers both.
        self_collision_config = SelfCollisionCostCfg(
            weight=self.world.device_cfg.to_device([1.0]),
            device_cfg=self.world.device_cfg,
            self_collision_kin_config=self.world.kinematics.get_self_collision_config(),
        )
        self.self_collision_cost_fn = SelfCollisionCost(self_collision_config)

        # Conf parameters for kinematic constraints, order in rollout should match
        self.kinematic_confs, self.kinematic_actions = zip(*(con.params for con in self.kinematic_constraints))
        self.kinematic_confs = list(self.kinematic_confs)
        self.kinematic_actions = list(self.kinematic_actions)

        # Compute the AABB and surface z-position for the placement surfaces.
        # Also precompute the static per-surface placement grouping used by
        # stable_placement_costs: ordered (obj, placement_param) pairs and the
        # per-sphere → per-object index map (sphere counts are static).
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
        for surface, placements in self.surface_to_placements.items():
            sphere_idx_map = []
            for o_idx, (obj, _) in enumerate(placements):
                num_spheres = world.get_collision_spheres(obj).shape[0]
                sphere_idx_map.extend([o_idx] * num_spheres)
            self.surface_to_sphere_idx_map[surface] = torch.tensor(
                sphere_idx_map, dtype=torch.int64, device=world.device
            )

        # Store the button AABBs for ValidPush
        self.button_to_action = {}
        self.button_aabbs = []
        for con in self.valid_push_constraints:
            button, action = con.params
            if not self.world.has_object(button):
                raise ValueError(f"{button=} not found in world")
            if button in self.button_to_action:
                raise NotImplementedError(f"We only support pushing a button once right now")

            # Set z to be 2cm buffer above surface of the button
            aabb = self.world.get_aabb(button).clone()
            aabb[0, 2] = aabb[1, 2] + world.collision_activation_distance + 2e-3
            aabb[1, 2] = aabb[1, 2] + world.collision_activation_distance + 0.02
            self.button_aabbs.append(aabb)
            self.button_to_action[button] = action
        self.button_aabbs = torch.stack(self.button_aabbs) if self.button_aabbs else None

        # Store the buttons and sticks for ValidPushStick
        self.button_stick_actions = {}
        self.button_stick_aabbs = []
        self.button_to_stick = {}
        for con in self.valid_push_stick_constraints:
            button, stick, action = con.params
            if not self.world.has_object(button):
                raise ValueError(f"{button=} not found in world")
            if not self.world.has_object(stick):
                raise ValueError(f"{stick=} not found in world")
            if button in self.button_to_stick:
                raise NotImplementedError(f"We only support pushing a button once right now")

            # Set z to be 2cm buffer above surface of the button
            button_aabb = self.world.get_aabb(button).clone()
            button_aabb[0, 2] = button_aabb[1, 2] + world.collision_activation_distance + 2e-3
            button_aabb[1, 2] = button_aabb[1, 2] + world.collision_activation_distance + 0.02
            self.button_stick_aabbs.append(button_aabb)

            self.button_to_stick[button] = stick
            self.button_stick_actions[button] = action
        self.button_stick_aabbs = torch.stack(self.button_stick_aabbs) if self.button_stick_aabbs else None

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
        """Kinematic constraints - i.e., pose error between actual and desired end-effector poses."""
        pos_errs, rot_errs = curobo_pose_error(rollout["world_from_ee"], rollout["world_from_ee_desired"])
        kinematic_cost = {
            "type": "constraint",
            "constraints": self.kinematic_constraints,
            "values": {"pos_err": pos_errs, "rot_err": rot_errs},
        }
        return kinematic_cost

    def motion_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Motion constraints — joint-limit violation + self-collision cost."""
        confs = rollout["confs"]
        robot_spheres = rollout["robot_spheres"]

        joint_limits = self.world.joint_limits
        dist_from_joint_lims = dist_from_bounds_jit(confs, joint_limits[0], joint_limits[1])

        # Single self-collision cost: setup once per shape, then evaluate.
        b, h = robot_spheres.shape[0], robot_spheres.shape[1]
        self.self_collision_cost_fn.setup_batch_tensors(b, h)
        self_coll_vals = self.self_collision_cost_fn.forward(robot_spheres).squeeze(-1)

        motion_cost = {
            "type": "constraint",
            "constraints": self.motion_constraints,
            "values": {"joint_limit": dist_from_joint_lims, "self_collision": self_coll_vals},
        }
        return motion_cost

    def valid_push_costs(
        self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]
    ) -> Union[dict, None]:
        """Valid Push constraints - i.e., distance from the button for push actions."""
        if not (self.valid_push_constraints or self.valid_push_stick_constraints):
            return None

        valid_push_cost = {"type": "constraint", "constraints": [], "values": {}}

        if self.valid_push_constraints:
            ts_idxs = []  # get timestep for the button push actions
            for button, action in self.button_to_action.items():
                ts = rollout["action_to_ts"][action]
                ts_idxs.append(ts)
            tool_poses = rollout["world_from_tool_desired"][:, ts_idxs]
            tool_xyz = tool_poses[:, :, :3, 3]
            dist_from_button = dist_from_bounds_jit(tool_xyz, self.button_aabbs[:, 0], self.button_aabbs[:, 1])
            valid_push_cost["constraints"].extend(self.valid_push_constraints)
            valid_push_cost["values"]["dist_from_button"] = dist_from_button

        if self.valid_push_stick_constraints:
            # Get the stick spheres for each button push action
            all_stick_spheres = []
            for button, action in self.button_stick_actions.items():
                pose_ts = rollout["action_to_pose_ts"][action]
                stick_name = self.button_to_stick[button]
                stick_spheres = obj_to_spheres[stick_name][:, pose_ts]
                all_stick_spheres.append(stick_spheres)
            all_stick_spheres = torch.stack(all_stick_spheres, dim=1)

            # We'll consider bottom of spheres, so subtract the radius
            stick_xyz = all_stick_spheres[..., :3].clone()  # important! clone so we don't modify original tensor
            stick_xyz[..., 2] -= all_stick_spheres[..., 3]

            # Compute distance between all stick spheres and button AABB, take the minimum
            push_stick_cost = dist_from_bounds_jit(
                stick_xyz, self.button_stick_aabbs[:, None, 0], self.button_stick_aabbs[:, None, 1]
            )
            push_stick_cost = push_stick_cost.min(-1).values

            valid_push_cost["constraints"].extend(self.valid_push_stick_constraints)
            valid_push_cost["values"]["stick_dist_from_button"] = push_stick_cost

        return valid_push_cost

    def stable_placement_costs(
        self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]
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

            # Within goal xy bounds, need to gather by the spheres for each object
            in_goal_xy = dist_from_bounds_jit(spheres_xy, *self.surface_to_aabb[surface])
            obj_in_goal_xy = torch.zeros(
                (num_particles, len(placements)), dtype=in_goal_xy.dtype, device=in_goal_xy.device
            )
            obj_in_goal_xy.scatter_add_(1, sphere_idx_map_expand, in_goal_xy)
            support_vals[f"{surface}_in_xy"] = obj_in_goal_xy

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

    def collision_costs(self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]) -> dict:
        """Collision costs. This could be tied better to the constraints and sped up significantly."""
        # Robot to world
        robot_spheres = rollout["robot_spheres"]
        coll_values = {"robot_to_world": self.world.collision_fn(robot_spheres)}

        # Collision between movables and world
        all_movable_spheres = torch.cat(list(obj_to_spheres.values()), dim=2)
        coll_values["movable_to_world"] = self.world.collision_fn(all_movable_spheres)

        # Collision between robot and movables, need to expand poses to full timesteps
        all_pose_ts = list(rollout["ts_to_pose_ts"].values())
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
            for q_retract_name, is_holding, held_obj in self.retract_info:
                q_retract = self._particles[q_retract_name]
                num_particles = q_retract.shape[0]
                
                # Single kinematics covers all arms in v0.8.
                kin_model = self.world.kinematics
                robot_state = kin_model.compute_kinematics(JointState.from_position(q_retract))
                retract_spheres = robot_state.robot_spheres  # already (b, h, n, 4) in v0.8
                
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
                    coll_values[f"{q_retract_name}_movables"] = torch.zeros((num_particles, 1), device=q_retract.device)

        coll_cost = {
            "type": "constraint",
            "constraints": self.cfree_constraints,
            "values": coll_values,
        }
        return coll_cost

    def _com_polygon_penalties(self) -> Dict[str, torch.Tensor]:
        """Lazily compute and cache the COM-polygon penalties dict.

        Shared by ``com_polygon_constraint`` and the ``com_polygon`` soft cost
        so the (expensive) FK-based computation runs at most once per cost
        evaluation.
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
        if not self.config.enable_com_polygon or self._particles is None:
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
                mask = torch.triu(torch.ones_like(all_obj_dists), diagonal=1) == 1
                dists_sum = all_obj_dists[mask].view(mask.shape[0], -1).sum(-1)
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
            mask = torch.triu(torch.ones_like(yaw_diffs), diagonal=1) == 1
            return yaw_diffs[mask].view(mask.shape[0], -1).sum(-1)
        
        # Retract close to home — penalize body slice (post-base) deviation
        # from the configured T1 home pose.
        elif cost_name == "retract_close_to_home":
            if self._particles is None:
                raise RuntimeError("Particles must be provided for retract_close_to_home soft cost")

            from cutamp.robots.t1 import BASE_DOF, t1_home
            home_t = torch.tensor(t1_home, device=device)
            # Penalize body (leg+torso+arms) deviation; ignore base x/y/yaw.
            body_slice = slice(BASE_DOF, len(t1_home))
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

            from cutamp.robots.t1 import BODY_INDICES, t1_home
            home_body = torch.tensor(
                t1_home[BODY_INDICES.start:BODY_INDICES.stop], device=device
            )

            config_params = [p for p in self._particles if p.startswith(("left_q", "right_q", "q"))]
            if config_params:
                all_configs = torch.stack([self._particles[p] for p in config_params], dim=1)
                return (all_configs[:, :, BODY_INDICES] - home_body).pow(2).sum(dim=(1, 2))
            return torch.zeros(num_particles, device=device)

        # COM-over-base-polygon — penalize particle configurations whose
        # mass-weighted COM projects outside the mobile-base support rectangle.
        # Same shape as the cuRobo-side cost (cutamp/com_polygon_cost.py); the
        # penalty math is shared via com_polygon_penalty.
        elif cost_name == "com_polygon":
            if self._particles is None:
                raise RuntimeError("Particles must be provided for com_polygon soft cost")
            pens = self._com_polygon_penalties()
            if not pens:
                return torch.zeros(num_particles, device=device)
            # Sum over confs to match the prior per-particle scalar shape.
            return torch.stack(list(pens.values()), dim=0).sum(dim=0)

        # Joint-limit margin — keep endpoint configurations a soft band away
        # from position limits. Per-joint/per-side margins from robots/t1.py
        # (zero on the home-side bounds the standing posture sits ON, so home
        # costs exactly 0). q0s are excluded inside the helper: the initial
        # robot state must never acquire margin gradients (Phase-2 LBFGS
        # re-leafs every cloned particle).
        elif cost_name == "joint_limit_margin":
            if self._particles is None:
                raise RuntimeError("Particles must be provided for joint_limit_margin soft cost")

            from cutamp.joint_limit_cost import joint_limit_margin_soft_cost
            from cutamp.robots.t1 import T1_LIMIT_MARGIN_LOWER, T1_LIMIT_MARGIN_UPPER

            limits = self.world.joint_limits  # [2, 21], row 0 = lower
            return joint_limit_margin_soft_cost(
                self._particles,
                limits,
                torch.tensor(T1_LIMIT_MARGIN_LOWER, device=device, dtype=limits.dtype),
                torch.tensor(T1_LIMIT_MARGIN_UPPER, device=device, dtype=limits.dtype),
                num_particles,
                device,
            )

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

        valid_push_cost = self.valid_push_costs(rollout, obj_to_spheres)
        add_cost(ValidPush.type, valid_push_cost)

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
