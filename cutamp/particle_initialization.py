# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Pieces of the plan's starting particles (:class:`~cutamp.step_loop.SteppedPlan` builds them) and of the motion
stage: each pick's grasp pool (:func:`_pick_pools`), the placements drawn at each place (:func:`_draw_placements`),
the checker's IK tolerances (:class:`IKTol`), and the posture summary the motion stage orders its rows by without centring (:func:`worst_posture`).
"""

import math
from typing import Dict, List, NamedTuple, Sequence, Tuple

import torch

from curobo.scene import Obstacle

from cutamp.constraint_checker import ConstraintChecker
from cutamp.grasps import AZIMUTH_CONE, near_vertical
from cutamp.joint_limit_cost import trunk_tilt_cosine
from cutamp.samplers import place_4dof_sampler, project_supported_placement
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import KinematicConstraint
from cutamp.utils.common import action_4dof_to_mat4x4, transform_spheres


#: The floating base's joints in ``model.base`` order: x, y, z, then YXZ Euler (build_planner_model.py's ``BASE``).
_BASE_JOINTS = ("base_j_x", "base_j_y", "base_j_z", "base_j_pitch", "base_j_roll", "base_j_yaw")


#: The iterations of the balance-first IK projection (:func:`~cutamp.gn_refresh.gn_project_balanced`, collision row
#: on) on a posture that already reaches its target, as the plan's starting projection runs it (SteppedPlan). The
#: planner's earlier GN IK took 30 balance-first iterations, then these 10 with the collision row, whose split reached
#: every pick target of blocks_t1's first plan (64 of 64; 20 + 10: 52).
GN_IK_CLEAR_ITERS = 10


class IKTol(NamedTuple):
    """The constraint checker's tolerances an IK conf must meet: the hand's and the soles' pose errors
    (KinematicConstraint)."""
    pos: float
    rot: float
    sole_pos: float
    sole_rot: float

    @classmethod
    def from_checker(cls, checker: ConstraintChecker) -> "IKTol":
        kc = KinematicConstraint.type
        return cls(*(checker._get_tol(kc, n) for n in ("pos_err", "rot_err", "sole_pos_err", "sole_rot_err")))


def worst_posture(world: TAMPWorld, confs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per row of ``confs`` (each ``[B, dof]``): the Trunk's tilt in radians and its travel from home in metres, each
    the largest over the confs, and the CoM margin in metres, the smallest."""
    from cutamp.com_polygon_cost import compute_com_margin

    names = list(world.model.joint_names)
    pitch, roll = names.index("base_j_pitch"), names.index("base_j_roll")
    xyz = [names.index(n) for n in _BASE_JOINTS[:3]]
    tilts, travels, margins = [], [], []
    for q in confs:
        tilts.append(torch.acos(trunk_tilt_cosine(q, pitch, roll).clamp(-1, 1)))
        home = torch.as_tensor(world.model.home, dtype=q.dtype, device=q.device)
        travels.append((q[:, xyz] - home[xyz]).norm(dim=-1))
        margins.append(compute_com_margin(world, q))
    return (torch.stack(tilts).amax(0), torch.stack(travels).amax(0), torch.stack(margins).amin(0))


def pick_place_confs(plan_skeleton: PlanSkeleton) -> List[str]:
    """The conf name of every pick and place op, in skeleton order."""
    return [op.values[-1] for op in plan_skeleton if op.operator.metadata.action_type in ("pick", "place")]


#: Placement candidates per row at a place: the row keeps the one that collides least with the scene.
_PLACE_DRAWS = 4


def _draw_placements(
    world: TAMPWorld, obj: Obstacle, obj_spheres: torch.Tensor, surface: Obstacle, held: torch.Tensor,
    *, surface_bounds=None,
) -> torch.Tensor:
    """``[n, 4]``: a placement of ``obj`` on ``surface`` for each row of ``held`` ``[n, 4, 4]``, the row's grasp of
    ``obj`` (G in the object frame). Of ``_PLACE_DRAWS`` candidates whose xyz come from ``place_4dof_sampler``, the row
    keeps the one whose spheres ``obj_spheres`` collide least with the scene (``world.collision_fn``).

    Each candidate's yaw follows the grasp: ψ = φ_u - φ_g + U(-AZIMUTH_CONE, AZIMUTH_CONE), with φ_g the azimuth of
    the grasp's approach Z_G in the object frame and φ_u that of the candidate's xy minus the stance's (the mean of
    ``world.stance_soles``). The world approach, ψ + φ_g, then comes from the robot's side, within ``AZIMUTH_CONE`` of
    the direction from the stance to the placement. A near-vertical grasp keeps the sampler's uniform yaw."""
    n = held.shape[0]
    candidates = place_4dof_sampler(n * _PLACE_DRAWS, obj, obj_spheres, surface).view(n, _PLACE_DRAWS, 4)
    approach = held[:, :3, 2]
    away = candidates[..., :2] - world.stance_soles[:, :2, 3].mean(0)
    phi_u, phi_g = torch.atan2(away[..., 1], away[..., 0]), torch.atan2(approach[:, 1:2], approach[:, :1])
    psi = phi_u - phi_g + (2 * torch.rand_like(phi_u) - 1) * AZIMUTH_CONE
    candidates[..., 3] = torch.where(near_vertical(approach)[:, None], candidates[..., 3],
                                     torch.remainder(psi, 2 * math.pi))
    if surface_bounds is not None:
        candidates = project_supported_placement(candidates, obj_spheres, surface_bounds)
    world_from_obj = action_4dof_to_mat4x4(candidates)  # [n, k, 4, 4]
    collision = world.collision_fn(transform_spheres(obj_spheres, world_from_obj).contiguous())  # [n, k]
    return candidates[torch.arange(n, device=candidates.device), collision.argmin(-1)]


class NoGraspCandidates(ValueError):
    """The geometric gripper filters removed every grasp of the assigned hand."""


def _pick_pools(world: TAMPWorld, plan_skeleton: PlanSkeleton) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Each pick's grasps ``[m, 7]`` and scores ``[m]``, keyed by its grasp parameter: every grasp its hand can take
    (``world.grasp_candidates``), top or side, from any direction. Raises :class:`NoGraspCandidates` for a pick whose
    hand can take none."""
    pools = {}
    for idx, ground_op in enumerate(plan_skeleton):
        if ground_op.operator.metadata.action_type != "pick":
            continue
        arm = ground_op.operator.metadata.arm
        obj, grasp, _q = ground_op.values
        if not world.has_object(obj):
            raise ValueError(f"{obj=} not found in world")
        candidates = world.grasp_candidates(obj, arm)
        if not len(candidates.grasps):
            raise NoGraspCandidates(f"{idx + 1}. {ground_op}: none of {obj}'s {len(world.grasps[obj].grasps)} grasps "
                                    f"is left for the {arm} hand: each reaches below its bottom face or touches it")
        pools[grasp] = candidates.grasps, candidates.scores
    return pools
