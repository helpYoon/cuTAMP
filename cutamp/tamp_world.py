# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""TAMPWorld for cuTAMP (cuRobo v0.8 single-MotionPlanner architecture).

The world wraps a TAMPEnvironment with: collision Scene, T1 Kinematics, a
single multi-tool-frame InverseKinematics solver, and a MotionPlanner
factory. T1 exposes two tool frames in ``robot_container.tool_frames``.
"""

import logging
from typing import Dict, List, Union

import torch

from curobo.inverse_kinematics import InverseKinematics
from curobo.kinematics import Kinematics
from curobo.motion_planner import MotionPlanner
from curobo.scene import Obstacle, Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState

from cutamp.envs import TAMPEnvironment
from cutamp.robots import RobotContainer
from cutamp.robots.t1 import TOOL_FRAME_FOR_ARM, get_t1_ik_solver, get_t1_motion_planner
from cutamp.t1_domain import get_initial_state
from cutamp.task_planning import State
from cutamp.utils.collision import WorldCollisionCost
from cutamp.utils.common import (
    approximate_goal_aabb,
    get_world_cfg,
    pose_list_to_mat4x4,
    sample_between_bounds,
    transform_spheres,
)
from cutamp.utils.shapes import sample_greedy_surface_spheres

_log = logging.getLogger(__name__)

#: The Adam loop's forward-kinematics sites, each on its own Kinematics
#: (:meth:`TAMPWorld.site_kinematics`).
FK_SITES = ("rollout", "retract", "com", "gate")


class TAMPWorld:
    """A TAMPEnvironment plus T1 kinematics, a Scene, and motion-planner factory.

    T1 exposes two tool frames (left + right base_link) via the
    ``RobotContainer``. ``q_init`` is a single 21-DOF vector.
    """

    def __init__(
        self,
        env: TAMPEnvironment,
        device_cfg: DeviceCfg,
        robot: RobotContainer,
        q_init: torch.Tensor,
        collision_activation_distance: float = 0.0,
        coll_n_spheres: int = 50,
        coll_sphere_radius: float = 0.005,
        enable_com_aware_ik: bool = True,  # Matches config.enable_com_aware_ik's default (validated ON).
        ik_max_batch_size: int = 512,  # Must be >= config.num_particles (particle init solves full-batch).
        reik_num_seeds: int = 0,  # >0: dedicated low-seed IK solver for coupled-reIK refreshes (see ik_refresh_solver).
        reik_lbfgs_iters: int = 0,  # >0: LBFGS budget for the refresh solver (0 = cuRobo default 100).
    ):
        self.env = env
        self.device_cfg = device_cfg

        self._movable_names = {obj.name for obj in env.movables}
        self._name_to_obj = {obj.name: obj for obj in env.movables + env.statics}

        # Static collision Scene (excludes movables; movables get attached at grasp time).
        self.world_cfg: Scene = get_world_cfg(env, include_movables=False)
        self.collision_fn = WorldCollisionCost(
            self.world_cfg, device_cfg, collision_activation_distance,
        )
        self.collision_activation_distance = collision_activation_distance

        # Robot container (kinematics, joint_limits, tool frames, gripper spheres).
        self.robot_container = robot
        self._q_init = q_init

        # Single multi-frame IK solver. Both tool frames are configured
        # (matches the planner). Particle init applies the same per-DOF
        # cspace pin the planner uses to keep the inactive arm at home,
        # so a "single-arm Pick" doesn't drift the other arm.
        self.ik_solver: InverseKinematics = get_t1_ik_solver(
            self.world_cfg, device_cfg=device_cfg, max_batch_size=ik_max_batch_size,
            enable_com_aware_ik=enable_com_aware_ik,
        )
        self._reik_num_seeds = reik_num_seeds
        self._reik_lbfgs_iters = reik_lbfgs_iters
        self._ik_max_batch_size = ik_max_batch_size
        self._enable_com_aware_ik = enable_com_aware_ik
        self._ik_refresh_solver = None
        self._kinematics_jac = None
        self._site_kinematics: Dict[str, Kinematics] = {}

        # Pre-fit collision spheres for each movable (used by attachment_manager).
        self._obj_to_spheres: Dict[str, torch.Tensor] = {}
        for obj in self.movables:
            spheres = sample_greedy_surface_spheres(
                obj, n_spheres=coll_n_spheres, sphere_radius=coll_sphere_radius,
            )
            self._obj_to_spheres[obj.name] = spheres.to(device_cfg.device)

        self._obj_to_aabb: Dict[str, torch.Tensor] = {}

        # Arm-home end-effector positions in world frame. Used by the
        # arm-affinity priority function in plan-skeleton search to rank
        # which arm should pick which object. Uses palm frames (*_base_link)
        # because they're already in tool_frames per t1_planar_base.yml;
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

    # ---- Container delegates ------------------------------------------------

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

    # ---- Movables and indexing ---------------------------------------------

    @property
    def movables(self) -> List[Obstacle]:
        return self.env.movables

    @property
    def statics(self) -> List[Obstacle]:
        return self.env.statics

    def is_movable(self, obj: Union[Obstacle, str]) -> bool:
        if isinstance(obj, Obstacle):
            obj = obj.name
        return obj in self._movable_names

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

        Same 21-DOF model, but ``compute_jacobian=True`` so every FK call
        also fills ``tool_jacobians``. Separate instance because the flag is
        baked into the kernel buffers at construction. Used by the GN reIK
        refresh (``cutamp/gn_refresh.py``).
        """
        if self._kinematics_jac is None:
            from cutamp.robots.t1 import get_t1_kinematics
            self._kinematics_jac = get_t1_kinematics(
                device_cfg=self.device_cfg, compute_jacobian=True,
            )
        return self._kinematics_jac

    def site_kinematics(self, name: str) -> Kinematics:
        """The dedicated Kinematics of the Adam loop's FK site ``name`` (one of
        ``FK_SITES``): on ``kinematics``' config and CoM-enabled like it, built
        lazily, once per name, and never reset. ``kinematics`` stays for the
        eager users: particle init, ``compute_com_polygon_mask``, the final
        refresh and the motion checks.

        1. One instance per site: an instance keeps its FK outputs, the state
           its backward reads and the gradient it returns in one buffer set,
           so same-shape calls on one instance before backward corrupt each
           other's gradients.
        2. Never reset, and that is safe: cuRobo's FK backward overwrites
           ``grad_out`` row by row, and that is the kernel's only writable
           pointer (``curobo/_src/curobolib/kernels/kinematics/
           kinematics_backward_kernel.cuh:138``, ``:153``). No gradient reads
           what an earlier call left, and the zero-filled gradient inputs the
           backward falls back on stay zero.
        3. Fixed buffers: a site's buffers change only with its batch shape, so
           a CUDA graph can bake them in whatever the call order. A graph that
           bakes buffers allocated before capture writes freed memory once an
           eager call reallocates them, as the gate FK in G_refresh, the
           refresh's graph, would on the shared ``kinematics``.
        """
        if name not in FK_SITES:
            raise ValueError(f"unknown FK site {name!r}: FK_SITES is {FK_SITES}")
        if name not in self._site_kinematics:
            self._site_kinematics[name] = Kinematics(self.kinematics.config, compute_com=True)
        return self._site_kinematics[name]

    @property
    def ik_refresh_solver(self) -> InverseKinematics:
        """IK solver for coupled-reIK refreshes. With ``reik_num_seeds > 0``,
        a dedicated low-seed solver (lazily built): refreshes are warm-started
        from the current particle q, so they don't need the 32-seed global
        search — cost scales ~linearly with seeds. Same capacity, collision
        checking, and CoM-aware residual as the main solver, which stays in
        use for particle init and the final pre-check refresh. Falls back to
        the main solver when ``reik_num_seeds == 0``.
        """
        if self._reik_num_seeds <= 0:
            return self.ik_solver
        if self._ik_refresh_solver is None:
            self._ik_refresh_solver = get_t1_ik_solver(
                self.world_cfg, device_cfg=self.device_cfg,
                max_batch_size=self._ik_max_batch_size,
                enable_com_aware_ik=self._enable_com_aware_ik,
                num_seeds=self._reik_num_seeds,
                lbfgs_iters=self._reik_lbfgs_iters,
                use_cuda_graph=True,  # captured by warmup_ik_solver(warm_refresh=True)
            )
        return self._ik_refresh_solver

    def warmup_ik_solver(self, num_particles: int, warm_refresh: bool = False):
        """Warmup the multi-frame IK solver with a multi-frame FK goal.

        When the world has a refresh solver (``reik_num_seeds > 0``),
        ``warm_refresh`` also captures its CUDA graphs with the exact call the
        coupled-reIK fallback makes; with 0 seeds (balance-first setup builds
        its world that way) there is nothing to capture. The capture must run
        before the planner thread starts: a capture concurrent with another
        thread's CUDA calls fails.
        """
        ik = self.ik_solver
        max_bs = getattr(ik.config, "max_batch_size", num_particles)
        n = min(num_particles, max_bs)
        q = sample_between_bounds(n, bounds=self.joint_limits)
        js = JointState.from_position(q)
        kin_state = self.kinematics.compute_kinematics(js)
        all_frames = list(ik.kinematics.tool_frames)
        poses = {f: kin_state.tool_poses.get_link_pose(f) for f in all_frames}
        goal = GoalToolPose.from_poses(poses, ordered_tool_frames=all_frames)
        _ = ik.solve_pose(goal_tool_poses=goal)
        if warm_refresh and self._reik_num_seeds > 0:
            from cutamp.particle_initialization import _ik_for_pose  # import cycle at module level

            refresh = self.ik_refresh_solver
            # The seed here is already the answer; without this the LBFGS stage
            # exits early, stays uncaptured, and captures mid-optimization.
            exit_early, refresh.config.exit_early = refresh.config.exit_early, False
            try:
                ee = kin_state.tool_poses.get_link_pose(
                    self.tool_frame_for_arm("left"), make_contiguous=True,
                ).get_matrix()
                _ik_for_pose(self, ee, "left", current_state_q=q, ik=refresh, seed_from_current=True)
            finally:
                refresh.config.exit_early = exit_early

    def get_motion_planner(
        self,
        collision_activation_distance: float = 0.01,
        use_cuda_graph: bool = True,
        enable_com_polygon: bool = True,
        enable_joint_limit_margin: bool = True,
    ) -> MotionPlanner:
        """Build a single MotionPlanner for the robot using the world's Scene
        with movables included (so attached objects are part of the collision
        world during planning).
        """
        scene_with_movables = get_world_cfg(self.env, include_movables=True)
        return get_t1_motion_planner(
            scene_with_movables,
            use_cuda_graph=use_cuda_graph,
            collision_activation_distance=collision_activation_distance,
            device_cfg=self.device_cfg,
            enable_com_polygon=enable_com_polygon,
            enable_joint_limit_margin=enable_joint_limit_margin,
        )


def check_tamp_world_not_in_collision(world: TAMPWorld, collision_tol: float = 1e-6):
    """Sanity check: no movable starts inside a collision."""
    for obj in world.movables:
        mat4x4 = pose_list_to_mat4x4(obj.pose).to(world.device)
        spheres = transform_spheres(world.get_collision_spheres(obj), mat4x4)
        spheres = spheres[None, None].contiguous()
        coll_cost = world.collision_fn(spheres).sum()
        if coll_cost > collision_tol:
            raise ValueError(f"Initial state in collision for object '{obj.name}' with cost {coll_cost}")
