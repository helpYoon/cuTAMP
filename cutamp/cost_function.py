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
from functools import cached_property, lru_cache
from typing import Dict, Tuple, Union

import torch
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo.types import JointState

from cutamp.com_polygon_cost import (
    compute_com_polygon_penalties, physical_support_centroid,
    physical_support_margin,
)
from cutamp.config import TAMPConfiguration
from cutamp.samplers import PLACEMENT_SUPPORT_GAP_M
from cutamp.costs import curobo_pose_error, dist_from_bounds_jit, sphere_to_sphere_overlap, trajectory_length
from cutamp.stance import stance_joint_limits
from cutamp.rollout import Rollout, get_retract_parameters, require_fixed_stance
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import (
    Collision,
    ComComfort,
    ComPolygon,
    KinematicConstraint,
    Motion,
    StablePlacement,
)
from cutamp.task_planning.costs import TrajectoryLength
from cutamp.utils.common import action_4dof_to_mat4x4, grasp_7dof_to_mat4x4, transform_spheres


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
        require_fixed_stance(plan_skeleton)
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config

        self.retract_info = get_retract_parameters(plan_skeleton)
        self._body_ids: dict[str, torch.Tensor] = {}  # per arm, _body_spheres' answer

        # Single self-collision cost over the whole robot — both arms in one
        # kinematic chain, so one cost per query covers both. use_grad_input: the cost's weight applies only then
        # (SelfCollisionCost).
        self_collision_config = SelfCollisionCostCfg(
            weight=self.world.device_cfg.to_device([1.0]),
            device_cfg=self.world.device_cfg,
            self_collision_kin_config=self.world.kinematics.get_self_collision_config(),
            use_grad_input=True,
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
            target_z = surface_z + PLACEMENT_SUPPORT_GAP_M
            self.surface_to_target_z[surface] = target_z

        self.surface_to_sphere_idx_map = {}
        self.surface_to_sphere_counts: Dict[str, Tuple[int, ...]] = {}
        for surface, placements in self.surface_to_placements.items():
            counts = tuple(world.get_collision_spheres(obj).shape[0] for obj, _ in placements)
            self.surface_to_sphere_counts[surface] = counts
            self.surface_to_sphere_idx_map[surface] = torch.tensor(
                [o_idx for o_idx, n in enumerate(counts) for _ in range(n)], dtype=torch.int64, device=world.device
            )

    def kinematic_costs(self, rollout: Rollout, stance_xyyaw) -> Union[dict, None]:
        """Kinematic constraints - i.e., pose error between actual and desired end-effector poses, and both
        soles' error against the stance at every conf, ``sole_pos_err`` and ``sole_rot_err`` ``[N, 2H]``: a conf
        whose feet drifted would pass the CoM gate on a support polygon it no longer stands on (design 7.4)."""
        pos_errs, rot_errs = curobo_pose_error(rollout["world_from_ee"], rollout["world_from_ee_desired"])
        values = {"pos_err": pos_errs, "rot_err": rot_errs}
        x, y, yaw = stance_xyyaw.unbind(-1)
        target_soles = action_4dof_to_mat4x4(torch.stack((x, y, torch.zeros_like(x), yaw), dim=-1))
        sole_pos, sole_rot = curobo_pose_error(rollout["world_from_soles"], target_soles)
        values["sole_pos_err"], values["sole_rot_err"] = sole_pos.flatten(1), sole_rot.flatten(1)
        return {"type": "constraint", "values": values}

    def motion_costs(self, rollout: Rollout, achieved_stances) -> Union[dict, None]:
        """Motion constraints — joint-limit violation + self-collision cost, of the rollout's confs and of the
        retract confs too (``retract_joint_limit``, ``retract_self_collision``)."""
        confs = rollout["confs"]
        robot_spheres = rollout["robot_spheres"]

        soles = torch.stack([achieved_stances[name] for name in rollout["conf_params"]], dim=1)
        joint_limits = stance_joint_limits(soles, *self.world.joint_limits)
        dist_from_joint_lims = dist_from_bounds_jit(confs, *joint_limits)

        # Single self-collision cost: setup once per shape, then evaluate.
        b, h = robot_spheres.shape[0], robot_spheres.shape[1]
        self.self_collision_cost_fn.setup_batch_tensors(b, h)
        self_coll_vals = self.self_collision_cost_fn.forward(robot_spheres).squeeze(-1)

        values = {"joint_limit": dist_from_joint_lims, "self_collision": self_coll_vals}
        if self.retract_self_collision_cost_fn is not None:
            values.update(self._retract_motion_values(achieved_stances))
        return {"type": "constraint", "values": values}

    def _retract_motion_values(self, achieved_stances) -> Dict[str, torch.Tensor]:
        """``retract_joint_limit`` and ``retract_self_collision``, each ``[N, R]`` over ``retract_info``'s confs,
        from one FK of every retract conf on its own site (TAMPWorld.site_kinematics)."""
        q = torch.stack([self._particles[name] for name, _, _ in self.retract_info], dim=1)  # [N, R, dof]
        n, r = q.shape[0], q.shape[1]
        kin_state = self.world.site_kinematics("retract_motion").compute_kinematics(
            JointState.from_position(q.view(n * r, -1))
        )
        soles = torch.stack([achieved_stances[name] for name, _, _ in self.retract_info], dim=1)
        joint_limits = stance_joint_limits(soles, *self.world.joint_limits)
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

    @cached_property
    def _foot_half(self) -> torch.Tensor:
        """A foot's half extents ``[3]``, built once: a recorded Adam step cannot copy from the host."""
        from cutamp.step_set import _facts

        return self.world.q_init.new_tensor(_facts(self.world.model)["foot_box"]) / 2

    @cached_property
    def _movables(self) -> dict:
        """Per movable, its world pose ``[4, 4]`` and its box in its own frame (``step_validation._local_box``), built
        once, for the same reason."""
        from cutamp.step_validation import _local_box

        poses = {obj.name: self.world.get_object_pose(obj) for obj in self.world.movables}
        return {obj.name: (poses[obj.name], _local_box(obj, poses[obj.name])) for obj in self.world.movables}

    def _body_spheres(self, kp, arm: str, count: int) -> torch.Tensor:
        """The indices, among the robot's ``count`` spheres, of those outside ``arm``'s hand (its tool frame, its hand
        link and its gripper links): what an object that arm carries is checked against. Found once per arm and kept:
        finding them waits on the device, and a selection by a mask does too."""
        if arm not in self._body_ids:
            from cutamp.robots.t1 import GRIPPER_JOINT_NAMES

            keep = torch.ones(count, dtype=torch.bool, device=kp.link_sphere_idx_map.device)
            for link in (self.world.tool_frame_for_arm(arm), f"{arm}_hand_link", *GRIPPER_JOINT_NAMES[arm]):
                if link in kp.link_name_to_idx_map:
                    keep[kp.get_sphere_index_from_link_name(link).long()] = False
            self._body_ids[arm] = keep.nonzero().flatten()
        return self._body_ids[arm]

    def stepped_scene_costs(self, particles, ops=None, initial_loose=None, initial_held=None, *, include_self=True,
                            site: str = "stepped_scene"):
        """Check selected postures against the other links, loose objects and carried objects, with the FK run on
        the FK site ``site`` (TAMPWorld.site_kinematics)."""
        from cutamp.gn_refresh import self_sphere_overlaps
        from cutamp.step_validation import _obb_gap, _object_box

        if ops is None:
            ops = [op for op in self.plan_skeleton if op.operator.metadata.action_type in ("pick", "place", "retract")]
        key = {"step": "q_land", "retract": "q_retract", "pick": "q", "place": "q"}
        q = torch.stack([particles[op.arg(key[op.operator.metadata.action_type])] for op in ops], dim=1)
        b, h = q.shape[:2]
        kin = self.world.site_kinematics(site)
        fk = kin.compute_kinematics(JointState.from_position(q.flatten(0, 1).contiguous()))
        robot = fk.robot_spheres.reshape(b, h, -1, 4)
        values = {"robot_to_world": self.world.collision_fn(robot)}
        if include_self:
            values["self_collision"] = self_sphere_overlaps(self.world.kinematics, robot).amax(-1)
        kp = kin.kinematics_config
        held = dict(initial_held or {})
        loose = (dict(initial_loose) if initial_loose is not None else
                 {name: pose.expand(b, -1, -1) for name, (pose, _) in self._movables.items()})
        foot_half = self._foot_half
        feet = []
        for frame in self.world.model.sole_frames:
            foot = fk.tool_poses.get_link_pose(frame, make_contiguous=True).get_matrix().reshape(b, h, 4, 4).clone()
            foot[..., :3, 3] += foot[..., :3, 2] * foot_half[2]
            feet.append(foot)
        for t, op in enumerate(ops):
            kind = op.operator.metadata.action_type
            arm = op.operator.metadata.arm
            if kind == "place":
                held.pop(op.arg("obj"))
                loose[op.arg("obj")] = action_4dof_to_mat4x4(particles[op.arg("placement")])
            object_spheres = {}
            peaks = []
            for obj in self.world.movables:
                name = obj.name
                spheres = robot[:, t]
                if name in held:
                    owner, ee_from_obj = held[name]
                    frame = self.world.tool_frame_for_arm(owner)
                    ee = fk.tool_poses.get_link_pose(frame, make_contiguous=True).get_matrix().reshape(b, h, 4, 4)
                    pose = ee[:, t] @ ee_from_obj
                    spheres = spheres.index_select(1, self._body_spheres(kp, owner, spheres.shape[1]))
                else:
                    pose = loose[name]
                obj_s = transform_spheres(self.world.get_collision_spheres(obj), pose)
                object_spheres[name] = obj_s
                peaks.append(sphere_to_sphere_overlap(spheres, obj_s))
                peaks.append(self.world.collision_fn(obj_s[:, None].contiguous()).reshape(b))
                box, half = _object_box(obj, pose, self._movables[name][1])
                for foot in feet:
                    peaks.append((-_obb_gap(foot[:, t], foot_half, box, half)).clamp_min(0))
            for a, c in itertools.combinations(object_spheres.values(), 2):
                peaks.append(sphere_to_sphere_overlap(a, c))
            if kind == "pick":
                frame = self.world.tool_frame_for_arm(arm)
                g = grasp_7dof_to_mat4x4(particles[op.arg("grasp")])
                # inv_ex: inv's values without its error check, which waits on the device
                held[op.arg("obj")] = arm, torch.linalg.inv_ex(g @ self.world.tool_from_ee[frame])[0]
            values[f"{t}_movables"] = torch.stack(peaks, dim=1).amax(1, keepdim=True)
        return {"type": "constraint", "values": values}

    def com_polygon_constraint(self, stances) -> Union[dict, None]:
        """Hard COM-over-base-polygon filter, per conf.

        Mirrors how Collision works: per-conf continuous penalty values
        (units: m²), ANDed into the overall satisfying mask by
        ``ConstraintChecker.get_mask``. The penalty is a differentiable
        inside-barrier function, so Adam receives a real gradient toward
        COM-feasibility instead of a dead-weight 0/1 Boolean.
        """
        pens = compute_com_polygon_penalties(self.world, self._particles, stances)
        if not pens:
            return None
        return {"type": "constraint", "values": pens}

    def __call__(self, rollout: Rollout, particles: Dict[str, torch.Tensor], *, achieved_stances) -> Dict[str, dict]:
        """The costs of ``particles`` with each conf's (and retract's) stance ``achieved_stances[name]`` ``[N, 2, 3]``,
        both soles' (x, y, yaw)."""
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

        # motion_costs below already checks robot self-collision for the final particle.
        collision_cost = self.stepped_scene_costs(particles, include_self=False, site="stepped_scene")
        add_cost(Collision.type, collision_cost)

        stable_placement_cost = self.stable_placement_costs(rollout, obj_to_spheres)
        add_cost(StablePlacement.type, stable_placement_cost)

        # Valid motions don't exceed joint limits
        motion_cost = self.motion_costs(rollout, achieved_stances)
        add_cost(Motion.type, motion_cost)

        stances = torch.stack([achieved_stances[name] for name in rollout["conf_params"]], dim=1)
        kinematic_cost = self.kinematic_costs(rollout, stances)
        add_cost(KinematicConstraint.type, kinematic_cost)

        # COM-over-base-polygon hard filter (mirrors Collision)
        com_polygon_cost = self.com_polygon_constraint(achieved_stances)
        add_cost(ComPolygon.type, com_polygon_cost)

        margin = physical_support_margin(rollout["com_xy"], stances, self.world.model.sole_box)
        shortfall = (self.config.com_comfort_margin_m - margin).clamp_min(0)
        add_cost(ComComfort.type, {"type": "constraint", "values": {"margin_shortfall_squared": shortfall.square()}})

        # The soft cost: the CoM's distance from the support's centre
        if self.config.com_center_weight:
            soft_cost = {"type": "cost", "values": {}}
            center = physical_support_centroid(stances, self.world.model.sole_box)
            distance_squared = (rollout["com_xy"] - center).square().sum(dim=-1)
            soft_cost["max_center_distance_m"] = distance_squared.detach().amax(-1).sqrt()
            soft_cost["values"]["com_center_distance_squared"] = self.config.com_center_weight * distance_squared
            add_cost("soft", soft_cost)

        return cost_dict
