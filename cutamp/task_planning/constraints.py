# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from .tamp_structs import Constraint


class Collision(Constraint):
    """This constraint is used more for typing purposes."""


class KinematicConstraint(Constraint):
    def __init__(self, conf, target_pose):
        super().__init__(conf, target_pose)


class Motion(Constraint):
    def __init__(self, q_start, traj, q_end):
        super().__init__(q_start, traj, q_end)


class StablePlacement(Constraint):
    def __init__(self, obj, grasp, placement, surface):
        super().__init__(obj, grasp, placement, surface)


class ComPolygon(Constraint):
    """Hard filter: particle's final-config COM must project inside the
    stance's support rectangle (``world.model.support``). Mirrors how
    Collision filters particles."""


class ComComfort(Constraint):
    """Minimum physical sole-hull clearance of the projected CoM at each pick and place."""


class StepReach(Constraint):
    """The relative landing command and swing travel must be legal."""

    def __init__(self, sole_from, sole_planted, sole_to):
        super().__init__(sole_from, sole_planted, sole_to)


class StandOnStance(Constraint):
    """The predicted whole-body landing is supported on the achieved pair of soles."""

    def __init__(self, q_stand, q_land, sole_l, sole_r):
        super().__init__(q_stand, q_land, sole_l, sole_r)


class Footprint(Constraint):
    """Check sole clearance against the other foot and the scene during the step."""

    def __init__(self, sole_from, sole_planted, sole_to):
        super().__init__(sole_from, sole_planted, sole_to)
