# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import warnings
from functools import cached_property
from typing import List, Literal, Dict, Union

import torch
from jaxtyping import Float

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sdf.world import WorldCollision
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from cutamp.envs import TAMPEnvironment
from cutamp.robots import RobotContainer, DualArmRobotContainer, load_robot_container
from cutamp.robots.franka import franka_curobo_cfg, get_franka_ik_solver
from cutamp.robots.ur5 import ur5_curobo_cfg, get_ur5_ik_solver
from cutamp.robots.t1 import t1_curobo_cfg, get_t1_ik_solver
from cutamp.tamp_domain import get_initial_state as get_initial_state_single_arm
from cutamp.t1_domain import get_initial_state as get_initial_state_dual_arm
from cutamp.task_planning import State
from cutamp.utils.collision import get_world_collision_cost
from cutamp.utils.common import approximate_goal_aabb, transform_spheres
from cutamp.utils.common import sample_between_bounds, get_world_cfg, pose_list_to_mat4x4
from cutamp.utils.shapes import sample_greedy_surface_spheres

_log = logging.getLogger(__name__)


class TAMPWorld:
    """
    Represents a TAMP world that wraps a static TAMPEnvironment with robot-specific logic,
    object indexing utilities, collision checking, IK solvers, and motion generation support.
    """

    def __init__(
        self,
        env: TAMPEnvironment,
        tensor_args: TensorDeviceType,
        robot: Union[Literal["panda", "ur5", "t1"], RobotContainer, DualArmRobotContainer],
        q_init: Union[Float[torch.Tensor, "dof"], Dict[str, Float[torch.Tensor, "dof"]]],
        collision_activation_distance: float = 0.0,
        coll_n_spheres: int = 50,
        coll_sphere_radius: float = 0.005,
    ):
        self.env = env
        self.tensor_args = tensor_args

        # Dicts and sets for indexing
        self._movable_names = {obj.name for obj in env.movables}
        self._name_to_obj = {obj.name: obj for obj in env.movables + env.statics}

        # Setup collision function
        self.world_cfg = get_world_cfg(env, include_movables=False)  # doesn't include movables
        self.collision_fn = get_world_collision_cost(self.world_cfg, tensor_args, collision_activation_distance)
        self.collision_activation_distance = collision_activation_distance

        # Setup robot container
        if isinstance(robot, str):
            warnings.warn(f"RobotContainer not provided, loading based on robot name {robot}")
            self.robot_container = load_robot_container(robot, tensor_args)
        else:
            self.robot_container = robot
        self.robot_name = self.robot_container.name
        
        # Store q_init - for dual-arm robots, expect dict with 'left' and 'right' keys
        if isinstance(q_init, dict):
            self._left_q_init = q_init["left"]
            self._right_q_init = q_init["right"]
            self._q_init = None  # Not used for dual-arm
        else:
            self._q_init = q_init
            self._left_q_init = None
            self._right_q_init = None

        # Setup the IK solver, right now it needs WorldCfg and I don't know the behavior, can speed up later
        if self.robot_name == "panda":
            self.ik_solver = get_franka_ik_solver(self.world_cfg)
        elif self.robot_name == "ur5":
            self.ik_solver = get_ur5_ik_solver(self.world_cfg)
        elif self.robot_name == "t1":
            # Dual-arm robot has two IK solvers, one for each arm
            self.ik_solver_left = get_t1_ik_solver("left", self.world_cfg)
            self.ik_solver_right = get_t1_ik_solver("right", self.world_cfg)
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")

        # Sample collision spheres for all movables
        self._obj_to_spheres: Dict[str, Float[torch.Tensor, "n 4"]] = {}
        for obj in self.movables:
            spheres = sample_greedy_surface_spheres(obj, n_spheres=coll_n_spheres, sphere_radius=coll_sphere_radius)
            self._obj_to_spheres[obj.name] = spheres.to(tensor_args.device)

        # AABB cache
        self._obj_to_aabb = {}

    @property
    def movables(self) -> List[Obstacle]:
        return self.env.movables

    def is_movable(self, obj: Obstacle | str) -> bool:
        if isinstance(obj, Obstacle):
            obj = obj.name
        return obj in self._movable_names

    @property
    def statics(self) -> List[Obstacle]:
        return self.env.statics

    @property
    def is_dual_arm(self) -> bool:
        """Whether the robot is a dual-arm robot (e.g., T1)."""
        return isinstance(self.robot_container, DualArmRobotContainer)

    @property
    def kin_model(self) -> CudaRobotModel:
        """
        Get kinematics model (single-arm robots only).
        
        For dual-arm robots, use get_kin_model(arm) instead.
        
        Raises:
            RuntimeError: If called on a dual-arm robot.
        """
        if self.is_dual_arm:
            raise RuntimeError(
                "kin_model property is not available for dual-arm robots. "
                "Use get_kin_model('left') or get_kin_model('right') instead."
            )
        return self.robot_container.kin_model

    @property
    def tool_from_ee(self) -> Float[torch.Tensor, "4 4"]:
        """
        Transformation from EE frame to tool/grasp frame (single-arm robots only).
        
        For dual-arm robots, use get_tool_from_ee(arm) instead.
        
        Raises:
            RuntimeError: If called on a dual-arm robot.
        """
        if self.is_dual_arm:
            raise RuntimeError(
                "tool_from_ee property is not available for dual-arm robots. "
                "Use get_tool_from_ee('left') or get_tool_from_ee('right') instead."
            )
        return self.robot_container.tool_from_ee

    def get_kin_model(self, arm: Literal["left", "right"] = "left") -> CudaRobotModel:
        """
        Get kinematics model for specified arm.
        
        Args:
            arm: Which arm's kinematics model to get ("left" or "right").
                 Ignored for single-arm robots.
        
        Returns:
            CudaRobotModel for the specified arm.
        """
        if self.is_dual_arm:
            if arm == "left":
                return self.robot_container.left_kin_model
            elif arm == "right":
                return self.robot_container.right_kin_model
            else:
                raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")
        return self.robot_container.kin_model

    def get_tool_from_ee(self, arm: Literal["left", "right"] = "left") -> Float[torch.Tensor, "4 4"]:
        """
        Get tool_from_ee transformation for specified arm.
        
        Args:
            arm: Which arm's tool_from_ee to get ("left" or "right").
                 Ignored for single-arm robots.
        
        Returns:
            4x4 transformation matrix from EE frame to tool/grasp frame.
        """
        if self.is_dual_arm:
            if arm == "left":
                return self.robot_container.left_tool_from_ee
            elif arm == "right":
                return self.robot_container.right_tool_from_ee
            else:
                raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")
        return self.robot_container.tool_from_ee

    def get_ik_solver(self, arm: Literal["left", "right"] = "left"):
        """
        Get IK solver for specified arm.
        
        Args:
            arm: Which arm's IK solver to get ("left" or "right").
                 Ignored for single-arm robots.
        
        Returns:
            IKSolver for the specified arm.
        """
        if self.is_dual_arm:
            if arm == "left":
                return self.ik_solver_left
            elif arm == "right":
                return self.ik_solver_right
            else:
                raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")
        return self.ik_solver

    def get_gripper_spheres(self, arm: Literal["left", "right"] = "left") -> Float[torch.Tensor, "n 4"]:
        """
        Get gripper collision spheres for specified arm.
        
        Args:
            arm: Which arm's gripper spheres to get ("left" or "right").
                 Ignored for single-arm robots.
        
        Returns:
            Tensor of shape (N, 4) with sphere centers and radii.
        """
        if self.is_dual_arm:
            if arm == "left":
                return self.robot_container.left_gripper_spheres
            elif arm == "right":
                return self.robot_container.right_gripper_spheres
            else:
                raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")
        return self.robot_container.gripper_spheres

    def get_joint_limits(self, arm: Literal["left", "right"] = "left") -> Float[torch.Tensor, "2 d"]:
        """
        Get joint limits for specified arm.
        
        Args:
            arm: Which arm's joint limits to get ("left" or "right").
                 Ignored for single-arm robots.
        
        Returns:
            Tensor of shape (2, DOF) where [0] is lower limits and [1] is upper limits.
        """
        if self.is_dual_arm:
            if arm == "left":
                return self.robot_container.left_joint_limits
            elif arm == "right":
                return self.robot_container.right_joint_limits
            else:
                raise ValueError(f"Invalid arm: {arm}. Must be 'left' or 'right'.")
        return self.robot_container.joint_limits

    @property
    def device(self) -> torch.device:
        return self.tensor_args.device

    @property
    def q_init(self) -> Float[torch.Tensor, "dof"]:
        """
        Get initial configuration (single-arm robots only).
        
        For dual-arm robots, use left_q_init and right_q_init instead.
        
        Raises:
            RuntimeError: If called on a dual-arm robot.
        """
        if self.is_dual_arm:
            raise RuntimeError(
                "q_init property is not available for dual-arm robots. "
                "Use left_q_init and right_q_init instead."
            )
        return self._q_init
    
    @property
    def left_q_init(self) -> Float[torch.Tensor, "dof"]:
        """
        Get initial configuration for left arm (dual-arm robots only).
        
        Raises:
            RuntimeError: If called on a single-arm robot.
        """
        if not self.is_dual_arm:
            raise RuntimeError(
                "left_q_init property is only available for dual-arm robots."
            )
        return self._left_q_init
    
    @property
    def right_q_init(self) -> Float[torch.Tensor, "dof"]:
        """
        Get initial configuration for right arm (dual-arm robots only).
        
        Raises:
            RuntimeError: If called on a single-arm robot.
        """
        if not self.is_dual_arm:
            raise RuntimeError(
                "right_q_init property is only available for dual-arm robots."
            )
        return self._right_q_init

    @property
    def initial_state(self) -> State:
        get_initial_state_fn = get_initial_state_dual_arm if self.is_dual_arm else get_initial_state_single_arm
        initial_state = get_initial_state_fn(
            movables=self.get_objects_by_type("Movable", return_name=True),
            surfaces=self.get_objects_by_type("Surface", return_name=True),
            sticks=self.get_objects_by_type("Stick", return_name=True),
            buttons=self.get_objects_by_type("Button", return_name=True),
        )
        return initial_state

    @property
    def goal_state(self) -> State:
        return self.env.goal_state

    def get_objects_by_type(self, obj_type: str, return_name: bool = True) -> List[Union[Obstacle, str]]:
        if obj_type not in self.env.type_to_objects:
            return []
        objs = self.env.type_to_objects[obj_type]
        if return_name:
            objs = [obj.name for obj in objs]
        return objs

    def get_object(self, name: str) -> Obstacle:
        """Get cuRobo Obstacle for object with the given name."""
        if name not in self._name_to_obj:
            raise ValueError(f"Object '{name}' not found in environment")
        return self._name_to_obj[name]

    def has_object(self, name: str) -> bool:
        """Whether the object with the given name exists in the environment."""
        return name in self._name_to_obj

    def get_object_pose(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "4 4"]:
        """Get the object initial pose."""
        obj = obj if isinstance(obj, Obstacle) else self.get_object(obj)
        mat4x4 = pose_list_to_mat4x4(obj.pose).to(self.device)
        return mat4x4

    def get_collision_spheres(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "n 4"]:
        """Get the collision spheres for the object (by either name or the cuRobo Obstacle)."""
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        return self._obj_to_spheres[obj_name]

    def get_aabb(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "2 3"]:
        """Get AABB for the given object."""
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        # Compute AABB if not cached
        if obj_name not in self._obj_to_aabb:
            obj = self.get_object(obj_name)
            aabb = approximate_goal_aabb(obj).to(self.device)
            self._obj_to_aabb[obj_name] = aabb
        return self._obj_to_aabb[obj_name]

    @cached_property
    def world_aabb(self) -> Float[torch.Tensor, "2 3"]:
        """Get AABB for the entire world (i.e., union of all objects)"""
        aabbs = [self.get_aabb(obj) for obj in self.movables] + [self.get_aabb(obj) for obj in self.statics]
        aabbs = torch.stack(aabbs)
        union_lower = aabbs[:, 0].min(dim=0).values
        union_upper = aabbs[:, 1].max(dim=0).values
        union_aabb = torch.stack([union_lower, union_upper])
        return union_aabb

    def warmup_ik_solver(self, num_particles: int, arm: Literal["left", "right"] = "left"):
        """
        Warmup cuRobo IK solver by running a batch solve.
        
        Args:
            num_particles: Number of random configurations to use for warmup.
            arm: Which arm's IK solver to warmup ("left" or "right").
                 For single-arm robots, this is ignored.
        """
        joint_limits = self.get_joint_limits(arm)
        kin_model = self.get_kin_model(arm)
        ik_solver = self.get_ik_solver(arm)
        
        q = sample_between_bounds(num_particles, bounds=joint_limits)
        goal_pose = kin_model.get_state(q).ee_pose
        _ = ik_solver.solve_batch(goal_pose)

    def get_motion_gen(
        self,
        collision_activation_distance: float,
        use_cuda_graph: bool = True,
        arm: Literal["left", "right"] = "left",
        num_trajopt_seeds: int = 2,
        num_trajopt_noisy_seeds: int = 2,
        world_coll_checker: WorldCollision = None,
    ) -> MotionGen:
        """
        Get the cuRobo motion generator for the robot.
        
        Args:
            collision_activation_distance: Distance threshold for collision activation.
            use_cuda_graph: Whether to use CUDA graph optimization. Set to False for debugging.
            arm: Which arm's motion generator to get ("left" or "right").
                 For single-arm robots, this is ignored.
            num_trajopt_seeds: Number of trajectory optimization seeds for Cartesian planning.
                              Must be set at init time, not at plan time, due to CUDA graph constraints.
            num_trajopt_noisy_seeds: Number of noisy seeds per trajectory seed for Cartesian planning,
                                    and also the number of seeds for joint-space planning (retract).
                                    Must be set at init time due to CUDA graph constraints.
            world_coll_checker: Optional shared WorldCollision object. If provided, this collision
                               checker will be reused instead of creating a new one. This is useful
                               for dual-arm robots to share the same collision world between both
                               arms, significantly reducing GPU memory usage.
        
        Returns:
            MotionGen configured for the specified arm.
        """
        if self.robot_name == "panda":
            robot_cfg = franka_curobo_cfg()
        elif self.robot_name == "ur5":
            robot_cfg = ur5_curobo_cfg()
        elif self.robot_name == "t1":
            robot_cfg = t1_curobo_cfg(arm)
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")

        max_num_spheres = max([len(sphs) for sphs in self._obj_to_spheres.values()])
        robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = max_num_spheres
        _log.info(f"Setting number of spheres for attachments to {max_num_spheres}")

        # World config needs to include movables for cuRobo
        # If world_coll_checker is provided, we don't need to pass world_model (it will be ignored)
        world_cfg = get_world_cfg(self.env, include_movables=True) if world_coll_checker is None else None
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_cfg,
            world_coll_checker=world_coll_checker,
            use_cuda_graph=use_cuda_graph,
            collision_activation_distance=collision_activation_distance,
            # Trajectory optimization seeds (memory vs success tradeoff)
            num_trajopt_seeds=num_trajopt_seeds,        # Seeds for Cartesian planning
            num_trajopt_noisy_seeds=num_trajopt_noisy_seeds,  # Seeds for joint-space (retract) + noisy augmentation
            # IK and graph seeds (lower memory impact)
            num_ik_seeds=64,          
            num_graph_seeds=4,        # Graph planner seeds for fallback paths
        )
        motion_gen = MotionGen(motion_gen_cfg)
        return motion_gen


def check_tamp_world_not_in_collision(world: TAMPWorld, collision_tol: float = 1e-6):
    """Check that the initial state of the movable objects are not in collision."""
    for obj in world.movables:
        # Transform spheres to world frame
        mat4x4 = pose_list_to_mat4x4(obj.pose).to(world.device)
        spheres = transform_spheres(world.get_collision_spheres(obj), mat4x4)  # [n, 4]
        spheres = spheres[None, None].contiguous()  # [1, 1, n, 4]

        coll_cost = world.collision_fn(spheres).sum()
        if coll_cost > collision_tol:
            raise ValueError(f"Initial state in collision for object '{obj.name}' with cost {coll_cost}")

    # TODO: catch collisions between spheres for each movable objects here
