# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""TAMPWorld for cuTAMP (cuRobo single-MotionPlanner architecture).

The world wraps a TAMPEnvironment with: collision Scene and T1 Kinematics, on the planning model
``world.model``; ``motion_solver.build_motion_planner`` builds the MotionPlanner from it. The IK is GN's
(``particle_initialization.gn_ik``), on these kinematics. T1 exposes two tool frames in
``robot_container.tool_frames``.
"""

import logging
from typing import Dict, List, Tuple, Union

import torch

from curobo.kinematics import Kinematics
from curobo.scene import Obstacle, Scene
from curobo.types import DeviceCfg, JointState, Pose

from cutamp.envs import TAMPEnvironment
from cutamp.floor_wrench import SegmentTorques
from cutamp.grasps import GraspSet, drop_grasps
from cutamp.robots import RobotContainer
from cutamp.robots.t1 import TOOL_FRAME_FOR_ARM, T1Model
from cutamp.t1_domain import get_initial_state
from cutamp.task_planning import State
from cutamp.utils.collision import WorldCollisionCost
from cutamp.utils.common import (
    approximate_goal_aabb,
    get_world_cfg,
    grasp_7dof_to_mat4x4,
    pose_list_to_mat4x4,
    transform_spheres,
)
from cutamp.utils.shapes import sample_greedy_surface_spheres

_log = logging.getLogger(__name__)

#: The Adam loop's FK sites, each on its own Kinematics (TAMPWorld.site_kinematics). "retract_motion" is the real
#: legs' retract confs for the motion constraints (cost_function.motion_costs).
FK_SITES = ("rollout", "retract", "com", "gate", "retract_motion")


class TAMPWorld:
    """A TAMPEnvironment plus T1 kinematics and a Scene.

    T1 exposes two tool frames (left + right base_link) via the
    ``RobotContainer``. ``q_init`` is a single full-cspace vector of the
    container's model (``world.model``).
    """

    def __init__(
        self,
        env: TAMPEnvironment,
        device_cfg: DeviceCfg,
        robot: RobotContainer,
        q_init: torch.Tensor,
        grasps: Dict[str, GraspSet],
    ):
        self.env = env
        self.device_cfg = device_cfg
        self._grasps = grasps

        self._name_to_obj = {obj.name: obj for obj in env.movables + env.statics}

        # The statics' collision Scene: the costs and GN treat the movables as sphere overlaps (their spheres below).
        # The cost activates at 0.
        self.world_cfg: Scene = get_world_cfg(env, include_movables=False)
        self.collision_fn = WorldCollisionCost(self.world_cfg, device_cfg, 0.0)

        # Robot container (kinematics, joint_limits, tool frames, gripper spheres).
        self.robot_container = robot
        self._q_init = q_init

        self._kinematics_jac = None
        self._segment_torques = None
        self._site_kinematics: Dict[str, Kinematics] = {}

        # Each movable's surface spheres in its own frame, for the rollout, the costs, particle init and the
        # visualizer (the motion planner's attach fits its own).
        self._obj_to_spheres: Dict[str, torch.Tensor] = {
            obj.name: sample_greedy_surface_spheres(obj, n_spheres=50, sphere_radius=0.005).to(device_cfg.device)
            for obj in self.movables
        }

        self._obj_to_aabb: Dict[str, torch.Tensor] = {}

        # Arm-home end-effector positions in world frame. Used by the
        # arm-affinity priority function in plan-skeleton search to rank
        # which arm should pick which object. Uses palm frames (*_base_link)
        # because they're already in tool_frames per t1_29dof.yml;
        # exact frame doesn't affect ranking, only LEFT-vs-RIGHT direction
        # matters.
        home_js = JointState.from_position(
            self._q_init.to(self.kinematics.device_cfg.device).unsqueeze(0)
        )
        home_ks = self.kinematics.compute_kinematics(home_js)
        self.arm_home_ee_world: Dict[str, torch.Tensor] = {
            "left":  home_ks.tool_poses.get_link_pose(
                "left_base_link", make_contiguous=True,
            ).position.flatten().detach().cpu(),
            "right": home_ks.tool_poses.get_link_pose(
                "right_base_link", make_contiguous=True,
            ).position.flatten().detach().cpu(),
        }

        # The planted soles' world poses at the model's home: every action stands on this one stance.
        stance_sole_poses: Dict[str, Pose] = {}
        stance_js = JointState.from_position(device_cfg.to_device(self.model.home).unsqueeze(0))
        stance_ks = self.kinematics.compute_kinematics(stance_js)
        for frame in self.model.sole_frames:
            pose = stance_ks.tool_poses.get_link_pose(frame, make_contiguous=True)
            # Copies: a single conf's pose is a view of cuRobo's FK buffers, which the next FK call rewrites.
            stance_sole_poses[frame] = Pose(
                position=pose.position.detach().clone(), quaternion=pose.quaternion.detach().clone(),
            )
        self._stance_soles: torch.Tensor = torch.stack([
            pose.get_matrix().reshape(4, 4) for pose in stance_sole_poses.values()
        ])

        kp = self.kinematics.kinematics_config
        self._hand_spheres: Dict[str, torch.Tensor] = {}
        for arm in ("left", "right"):
            tool = self.tool_frame_for_arm(arm)
            links = (f"{arm}_hand_link", f"A{arm[0].upper()}6")
            idx = torch.cat([kp.get_sphere_index_from_link_name(link) for link in links])
            world_from_tool = stance_ks.tool_poses.get_link_pose(tool, make_contiguous=True).get_matrix().reshape(4, 4)
            # In float64: a float32 CUDA matmul runs in TF32 here, 0.2 mm off at these world coordinates.
            wrist = transform_spheres(stance_ks.robot_spheres.reshape(-1, 4)[idx].double(),
                                      torch.linalg.inv(world_from_tool.double()))
            self._hand_spheres[arm] = torch.cat([self.gripper_spheres[tool], wrist.float()])
        # Each (object, arm)'s grasp candidates, made at first use (grasp_candidates).
        self._grasp_candidates: Dict[Tuple[str, str], GraspSet] = {}

    # ---- Container delegates ------------------------------------------------

    @property
    def model(self) -> T1Model:
        """The planning model: joint names, slices, home and support region."""
        return self.robot_container.model

    @property
    def kinematics(self) -> Kinematics:
        """Single Kinematics for the robot (covers all tool frames)."""
        return self.robot_container.kinematics

    @property
    def tool_frames(self) -> List[str]:
        return self.robot_container.tool_frames

    def tool_frame_for_arm(self, arm: Union[str, None]) -> str:
        """Resolve an arm name ("left"/"right") to its tool-frame name.

        ``arm=None`` falls back to the left tool frame.
        """
        if arm is None:
            return self.tool_frames[0]
        return TOOL_FRAME_FOR_ARM.get(arm, self.tool_frames[0])

    @property
    def tool_from_ee(self) -> Dict[str, torch.Tensor]:
        """Dict keyed by tool-frame name. T1 has two entries; others have one."""
        return self.robot_container.tool_from_ee

    @property
    def gripper_spheres(self) -> Dict[str, torch.Tensor]:
        return self.robot_container.gripper_spheres

    @property
    def joint_limits(self) -> torch.Tensor:
        return self.robot_container.joint_limits

    @property
    def device(self) -> torch.device:
        return self.device_cfg.device

    @property
    def q_init(self) -> torch.Tensor:
        return self._q_init

    @property
    def stance_soles(self) -> torch.Tensor:
        """``[2, 4, 4]`` world poses of ``model.sole_frames`` at ``model.home`` (made once, at construction)."""
        return self._stance_soles

    # ---- Movables and indexing ---------------------------------------------

    @property
    def movables(self) -> List[Obstacle]:
        return self.env.movables

    @property
    def statics(self) -> List[Obstacle]:
        return self.env.statics

    def get_object(self, name: str) -> Obstacle:
        if name not in self._name_to_obj:
            raise ValueError(f"Object '{name}' not found in environment")
        return self._name_to_obj[name]

    def has_object(self, name: str) -> bool:
        return name in self._name_to_obj

    def get_object_pose(self, obj: Union[Obstacle, str]) -> torch.Tensor:
        obj = obj if isinstance(obj, Obstacle) else self.get_object(obj)
        return pose_list_to_mat4x4(obj.pose).to(self.device)

    def get_collision_spheres(self, obj: Union[Obstacle, str]) -> torch.Tensor:
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        return self._obj_to_spheres[obj_name]

    @property
    def grasps(self) -> Dict[str, GraspSet]:
        """Each movable's GraspGenX grasps by name (``grasps.grasp_sets``); same-shaped movables share one set."""
        return self._grasps

    def hand_spheres(self, arm: str) -> torch.Tensor:
        """``[14, 4]``: ``arm``'s hand spheres in its tool frame (``*_base_link``), for the grasp drops: the gripper's 12
        (``gripper_spheres``), then the ``*_hand_link`` and the wrist (AL6 / AR6) spheres, placed by FK at
        ``model.home``. Hand_Roll moves the wrist sphere less than 2 mm in that frame across its range, so the set is
        taken as rigid."""
        return self._hand_spheres[arm]

    def grasp_candidates(self, obj_name: str, arm: str) -> GraspSet:
        """``obj_name``'s grasps (``grasps``) that ``arm``'s hand can take: those ``grasps.drop_grasps`` keeps with
        ``hand_spheres(arm)``, the object's spheres and its bottom face. Made at first use and kept. Same-shaped
        objects get equal sets."""
        key = (obj_name, arm)
        if key not in self._grasp_candidates:
            grasps, hand, spheres = self.grasps[obj_name], self.hand_spheres(arm), self.get_collision_spheres(obj_name)
            bottom = float(self.get_object(obj_name).get_trimesh_mesh().bounds[0, 2])  # unit pose: the object frame
            obj_from_G = grasp_7dof_to_mat4x4(grasps.grasps)
            keep = drop_grasps(obj_from_G, hand, spheres, bottom)
            self._grasp_candidates[key] = GraspSet(grasps.grasps[keep], grasps.scores[keep])
            _log.debug(f"{obj_name}, {arm} hand: {int(keep.sum())} of {len(keep)} grasps kept")
        return self._grasp_candidates[key]

    def get_aabb(self, obj: Union[Obstacle, str]) -> torch.Tensor:
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        if obj_name not in self._obj_to_aabb:
            obj = self.get_object(obj_name)
            self._obj_to_aabb[obj_name] = approximate_goal_aabb(obj).to(self.device)
        return self._obj_to_aabb[obj_name]

    # ---- Initial / goal state ----------------------------------------------

    @property
    def initial_state(self) -> State:
        return get_initial_state(
            movables=[obj.name for obj in self.env.type_to_objects.get("Movable", [])],
            surfaces=[obj.name for obj in self.env.type_to_objects.get("Surface", [])],
        )

    @property
    def goal_state(self) -> State:
        return self.env.goal_state

    # ---- Solvers ------------------------------------------------------------

    @property
    def kinematics_jac(self) -> Kinematics:
        """Jacobian-enabled twin of ``kinematics`` (lazily built).

        Same config, but ``compute_jacobian=True`` so every FK call
        also fills ``tool_jacobians`` ``[b, h, num_tool_frames, 6, dof]``
        (rows 0-2 linear / 3-5 angular, world frame; verified against finite
        differences). Separate instance because the flag is
        baked into the kernel buffers at construction. Used by the GN reIK
        refresh (``cutamp/gn_refresh.py``).
        """
        if self._kinematics_jac is None:
            self._kinematics_jac = Kinematics(self.kinematics.config, compute_com=True, compute_jacobian=True)
        return self._kinematics_jac

    @property
    def segment_torques(self) -> SegmentTorques:
        """cuRobo's two RNEA passes with the floor's push on a plan's frames
        (:class:`cutamp.floor_wrench.SegmentTorques`, lazily built): the balance gate's torque, speed and ZMP
        (``floor``) checks, which set each segment's pace."""
        if self._segment_torques is None:
            self._segment_torques = SegmentTorques(self.model, self.kinematics.config)
        return self._segment_torques

    def site_kinematics(self, name: str) -> Kinematics:
        """The Adam loop's FK site ``name``: a dedicated, lazily built, CoM-enabled Kinematics
        (``kinematics`` stays for the eager callers). An instance reuses one buffer set per batch
        shape, so two same-shape calls on it before backward corrupt each other's gradients. Never
        reset a site: a CUDA graph bakes its buffers, and a reset (or a shared instance's other
        callers) would reallocate them under the graph. That is safe: cuRobo's FK backward
        (kinematics_backward_kernel.cuh) overwrites ``grad_out`` row by row, its only writable pointer,
        so no gradient reads what an earlier call left. The backward returns ``grad_out`` itself, so a
        ``.grad`` adopted from it holds only until the site's next backward: clear or clone it."""
        if name not in FK_SITES:
            raise ValueError(f"unknown FK site {name!r}: FK_SITES is {FK_SITES}")
        if name not in self._site_kinematics:
            self._site_kinematics[name] = Kinematics(self.kinematics.config, compute_com=True)
        return self._site_kinematics[name]


def check_tamp_world_not_in_collision(world: TAMPWorld, collision_tol: float = 1e-6):
    """Sanity check: no movable starts inside a collision."""
    for obj in world.movables:
        mat4x4 = pose_list_to_mat4x4(obj.pose).to(world.device)
        spheres = transform_spheres(world.get_collision_spheres(obj), mat4x4)
        spheres = spheres[None, None].contiguous()
        coll_cost = world.collision_fn(spheres).sum()
        if coll_cost > collision_tol:
            raise ValueError(f"Initial state in collision for object '{obj.name}' with cost {coll_cost}")
