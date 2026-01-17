# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
TAMP Domain for T1 dual-arm robot.

This domain uses generic parameter names (q, q_start, q_end, traj) instead of
arm-prefixed names (left_q, right_q). The arm information is encoded in the
fluent/operator names (LeftAt vs RightAt, LeftPick vs RightPick), not in the
parameter names. This keeps the domain compatible with the existing task planner.
"""

from typing import Sequence

from cutamp.task_planning import Fluent, Parameter, TAMPOperator, State
from cutamp.task_planning.constraints import (
    CollisionFree,
    CollisionFreeGrasp,
    CollisionFreeHolding,
    CollisionFreePlacement,
    KinematicConstraint,
    Motion,
    StablePlacement,
    ValidPush,
    ValidPushStick,
)
from cutamp.task_planning.costs import GraspCost, TrajectoryLength

# Types
Conf = "conf"
Traj = "traj"
Pose = "pose"
Grasp = "grasp"

Movable = "movable"
Surface = "surface"
Button = "button"

# Fluents (aka predicates)
LeftAt = Fluent("LeftAt", [Parameter("q", Conf)])
RightAt = Fluent("RightAt", [Parameter("q", Conf)])
LeftHandEmpty = Fluent("LeftHandEmpty")
RightHandEmpty = Fluent("RightHandEmpty")

LeftCanMove = Fluent("LeftCanMove")
RightCanMove = Fluent("RightCanMove")

LeftJustMoved = Fluent("LeftJustMoved")
RightJustMoved = Fluent("RightJustMoved")

LeftHolding = Fluent("LeftHolding", [Parameter("obj", Movable)])
RightHolding = Fluent("RightHolding", [Parameter("obj", Movable)])

LeftHoldingWithGrasp = Fluent("LeftHoldingWithGrasp", [Parameter("obj", Movable), Parameter("grasp", Grasp)])
RightHoldingWithGrasp = Fluent("RightHoldingWithGrasp", [Parameter("obj", Movable), Parameter("grasp", Grasp)])

ButtonPushed = Fluent("ButtonPushed", [Parameter("button", "button")])
PushedWithStick = Fluent("PushedWithStick", [Parameter("button", "button"), Parameter("obj", Movable)])
CanPush = Fluent("CanPush", [Parameter("button", "button")])
IsMovable = Fluent("IsMovable", [Parameter("obj", Movable)])
IsButton = Fluent("IsButton", [Parameter("button", Button)])
IsSurface = Fluent("IsSurface", [Parameter("surface", Surface)])
IsStick = Fluent("IsStick", [Parameter("obj", Movable)])
HasNotPickedUp = Fluent("HasNotPickedUp", [Parameter("obj", Movable)])
On = Fluent("On", [Parameter("obj", Movable), Parameter("surface", Surface)])

all_tamp_fluents = [
    LeftAt,
    RightAt,
    LeftHandEmpty,
    RightHandEmpty,
    LeftCanMove,
    RightCanMove,
    LeftJustMoved,
    RightJustMoved,
    LeftHolding,
    RightHolding,
    LeftHoldingWithGrasp,
    RightHoldingWithGrasp,
    ButtonPushed,
    PushedWithStick,
    CanPush,
    IsMovable,
    IsButton,
    IsSurface,
    IsStick,
    HasNotPickedUp,
    On,
]


# Parameters - generic names, arm info comes from fluent/operator context
q = Parameter("q", Conf)
q_start = Parameter("q_start", Conf)
q_end = Parameter("q_end", Conf)
traj = Parameter("traj", Traj)

obj = Parameter("obj", Movable)
button = Parameter("button", Button)
surface = Parameter("surface", Surface)

grasp = Parameter("grasp", Grasp)
pose = Parameter("pose", Pose)
placement = Parameter("placement", Pose)


# Operators - this is the important part!
# Left arm operators use LeftAt, LeftHandEmpty, etc.
# Right arm operators use RightAt, RightHandEmpty, etc.
# The operator name (LeftMoveFree vs RightMoveFree) tells downstream code which arm to use.

LeftMoveFree = TAMPOperator(
    "LeftMoveFree",
    [q_start, traj, q_end],
    preconditions=[LeftAt(q_start), LeftHandEmpty(), LeftCanMove()],
    add_effects=[LeftAt(q_end), LeftJustMoved()],
    del_effects=[LeftAt(q_start), LeftCanMove()],
    constraints=[CollisionFree(q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)

RightMoveFree = TAMPOperator(
    "RightMoveFree",
    [q_start, traj, q_end],
    preconditions=[RightAt(q_start), RightHandEmpty(), RightCanMove()],
    add_effects=[RightAt(q_end), RightJustMoved()],
    del_effects=[RightAt(q_start), RightCanMove()],
    constraints=[CollisionFree(q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)


LeftMoveHolding = TAMPOperator(
    "LeftMoveHolding",
    [obj, grasp, q_start, traj, q_end],
    preconditions=[
        LeftAt(q_start),
        LeftHolding(obj),
        LeftHoldingWithGrasp(obj, grasp),
        LeftCanMove(),
    ],
    add_effects=[LeftAt(q_end), LeftJustMoved()],
    del_effects=[LeftAt(q_start), LeftCanMove()],
    constraints=[CollisionFreeHolding(obj, grasp, q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)

RightMoveHolding = TAMPOperator(
    "RightMoveHolding",
    [obj, grasp, q_start, traj, q_end],
    preconditions=[
        RightAt(q_start),
        RightHolding(obj),
        RightHoldingWithGrasp(obj, grasp),
        RightCanMove(),
    ],
    add_effects=[RightAt(q_end), RightJustMoved()],
    del_effects=[RightAt(q_start), RightCanMove()],
    constraints=[CollisionFreeHolding(obj, grasp, q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)

LeftPick = TAMPOperator(
    "LeftPick",
    [obj, grasp, q],
    preconditions=[
        LeftAt(q),
        LeftHandEmpty(),
        IsMovable(obj),
        LeftJustMoved(),
        HasNotPickedUp(obj),
    ],
    add_effects=[
        LeftHolding(obj),
        LeftHoldingWithGrasp(obj, grasp),
        LeftCanMove(),
    ],
    del_effects=[LeftHandEmpty(), LeftJustMoved(), HasNotPickedUp(obj)],
    constraints=[KinematicConstraint(q, grasp), CollisionFreeGrasp(obj, grasp)],
    costs=[GraspCost(obj, grasp)],
)

RightPick = TAMPOperator(
    "RightPick",
    [obj, grasp, q],
    preconditions=[
        RightAt(q),
        RightHandEmpty(),
        IsMovable(obj),
        RightJustMoved(),
        HasNotPickedUp(obj),
    ],
    add_effects=[
        RightHolding(obj),
        RightHoldingWithGrasp(obj, grasp),
        RightCanMove(),
    ],
    del_effects=[RightHandEmpty(), RightJustMoved(), HasNotPickedUp(obj)],
    constraints=[KinematicConstraint(q, grasp), CollisionFreeGrasp(obj, grasp)],
    costs=[GraspCost(obj, grasp)],
)

LeftPlace = TAMPOperator(
    "LeftPlace",
    [obj, grasp, placement, surface, q],
    preconditions=[
        LeftAt(q),
        LeftHolding(obj),
        LeftHoldingWithGrasp(obj, grasp),
        IsSurface(surface),
        LeftJustMoved(),
    ],
    add_effects=[LeftHandEmpty(), LeftCanMove(), On(obj, surface)],
    del_effects=[
        LeftHolding(obj),
        LeftHoldingWithGrasp(obj, grasp),
        LeftJustMoved(),
    ],
    constraints=[
        KinematicConstraint(q, placement),
        StablePlacement(obj, grasp, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)

RightPlace = TAMPOperator(
    "RightPlace",
    [obj, grasp, placement, surface, q],
    preconditions=[
        RightAt(q),
        RightHolding(obj),
        RightHoldingWithGrasp(obj, grasp),
        IsSurface(surface),
        RightJustMoved(),
    ],
    add_effects=[RightHandEmpty(), RightCanMove(), On(obj, surface)],
    del_effects=[
        RightHolding(obj),
        RightHoldingWithGrasp(obj, grasp),
        RightJustMoved(),
    ],
    constraints=[
        KinematicConstraint(q, placement),
        StablePlacement(obj, grasp, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)


LeftPush = TAMPOperator(
    "LeftPush",
    [button, pose, q],
    preconditions=[
        LeftAt(q),
        IsButton(button),
        LeftHandEmpty(),
        CanPush(button),
        LeftJustMoved(),
    ],
    add_effects=[ButtonPushed(button), LeftCanMove()],
    del_effects=[LeftJustMoved(), CanPush(button)],
    constraints=[KinematicConstraint(q, pose), ValidPush(button, pose)],
    costs=[],
)

RightPush = TAMPOperator(
    "RightPush",
    [button, pose, q],
    preconditions=[
        RightAt(q),
        IsButton(button),
        RightHandEmpty(),
        CanPush(button),
        RightJustMoved(),
    ],
    add_effects=[ButtonPushed(button), RightCanMove()],
    del_effects=[RightJustMoved(), CanPush(button)],
    constraints=[KinematicConstraint(q, pose), ValidPush(button, pose)],
    costs=[],
)

LeftPushStick = TAMPOperator(
    "LeftPushStick",
    [button, obj, grasp, pose, q],
    preconditions=[
        LeftAt(q),
        LeftHolding(obj),
        LeftHoldingWithGrasp(obj, grasp),
        IsButton(button),
        IsStick(obj),
        CanPush(button),
        LeftJustMoved(),
    ],
    add_effects=[ButtonPushed(button), LeftCanMove(), PushedWithStick(button, obj)],
    del_effects=[LeftJustMoved(), CanPush(button)],
    constraints=[KinematicConstraint(q, pose), ValidPushStick(button, obj, pose)],
    costs=[],
)

RightPushStick = TAMPOperator(
    "RightPushStick",
    [button, obj, grasp, pose, q],
    preconditions=[
        RightAt(q),
        RightHolding(obj),
        RightHoldingWithGrasp(obj, grasp),
        IsButton(button),
        IsStick(obj),
        CanPush(button),
        RightJustMoved(),
    ],
    add_effects=[ButtonPushed(button), RightCanMove(), PushedWithStick(button, obj)],
    del_effects=[RightJustMoved(), CanPush(button)],
    constraints=[KinematicConstraint(q, pose), ValidPushStick(button, obj, pose)],
    costs=[],
)

all_t1_operators = [
    LeftMoveFree, RightMoveFree,
    LeftMoveHolding, RightMoveHolding,
    LeftPick, RightPick,
    LeftPlace, RightPlace,
    LeftPush, RightPush,
    LeftPushStick, RightPushStick,
]


def get_initial_state(
    movables: Sequence[str] = (), surfaces: Sequence[str] = (), sticks: Sequence[str] = (), buttons: Sequence[str] = ()
) -> State:
    """Ground the initial state of the T1 TAMP domain.
    
    Note: left_q0 is the initial left arm configuration, right_q0 is the initial right arm configuration.
    The fluent (LeftAt vs RightAt) determines which arm the configuration belongs to.
    """
    initial_state = {
        LeftAt.ground("left_q0"),
        RightAt.ground("right_q0"),
        LeftHandEmpty.ground(),
        RightHandEmpty.ground(),
        LeftCanMove.ground(),
        RightCanMove.ground(),
    }
    for movable in movables:
        initial_state.add(IsMovable.ground(movable))
        initial_state.add(HasNotPickedUp.ground(movable))

    for surface in surfaces:
        initial_state.add(IsSurface.ground(surface))

    for stick in sticks:
        initial_state.add(IsStick.ground(stick))
        initial_state.add(IsMovable.ground(stick))
        initial_state.add(HasNotPickedUp.ground(stick))

    for button in buttons:
        initial_state.add(IsButton.ground(button))
        initial_state.add(CanPush.ground(button))

    initial_state = frozenset(initial_state)
    return initial_state
