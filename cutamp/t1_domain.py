# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Domain file like in PDDL. The task planner can be improved, but it suffices for now.
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


# Parameters - used for naming in operators, so it's easier to read when debugging
left_q = Parameter("left_q", Conf)
right_q = Parameter("right_q", Conf)
left_q_start = Parameter("left_q_start", Conf)
right_q_start = Parameter("right_q_start", Conf)
left_q_end = Parameter("left_q_end", Conf)
right_q_end = Parameter("right_q_end", Conf)
left_traj = Parameter("left_traj", Traj)
right_traj = Parameter("right_traj", Traj)

obj = Parameter("obj", Movable)
button = Parameter("button", Button)
surface = Parameter("surface", Surface)

grasp = Parameter("grasp", Grasp)
pose = Parameter("pose", Pose)
placement = Parameter("placement", Pose)


# Operators - this is the important part!
LeftMoveFree = TAMPOperator(
    "MoveFree",
    [left_q_start, left_traj, left_q_end],
    preconditions=[LeftAt(left_q_start), LeftHandEmpty(), LeftCanMove()],
    add_effects=[LeftAt(left_q_end), LeftJustMoved()],
    del_effects=[LeftAt(left_q_start), LeftCanMove()],
    constraints=[CollisionFree(left_q_start, left_traj, left_q_end), Motion(left_q_start, left_traj, left_q_end)],
    costs=[TrajectoryLength(left_q_start, left_traj, left_q_end)],
)

RightMoveFree = TAMPOperator(
    "MoveFree",
    [right_q_start, right_traj, right_q_end],
    preconditions=[RightAt(right_q_start), RightHandEmpty(), RightCanMove()],
    add_effects=[RightAt(right_q_end), RightJustMoved()],
    del_effects=[RightAt(right_q_start), RightCanMove()],
    constraints=[CollisionFree(right_q_start, right_traj, right_q_end), Motion(right_q_start, right_traj, right_q_end)],
    costs=[TrajectoryLength(right_q_start, right_traj, right_q_end)],
)


LeftMoveHolding = TAMPOperator(
    "MoveHolding",
    [obj, grasp, left_q_start, left_traj, left_q_end],
    preconditions=[
        LeftAt(left_q_start),
        LeftHolding(obj),
        LeftHoldingWithGrasp(obj, grasp),
        LeftCanMove(),
    ],
    add_effects=[LeftAt(left_q_end), LeftJustMoved()],
    del_effects=[LeftAt(left_q_start), LeftCanMove()],
    constraints=[CollisionFreeHolding(obj, grasp, left_q_start, left_traj, left_q_end), Motion(left_q_start, left_traj, left_q_end)],
    costs=[TrajectoryLength(left_q_start, left_traj, left_q_end)],
)

RightMoveHolding = TAMPOperator(
    "MoveHolding",
    [obj, grasp, right_q_start, right_traj, right_q_end],
    preconditions=[
        RightAt(right_q_start),
        RightHolding(obj),
        RightHoldingWithGrasp(obj, grasp),
        RightCanMove(),
    ],
    add_effects=[RightAt(right_q_end), RightJustMoved()],
    del_effects=[RightAt(right_q_start), RightCanMove()],
    constraints=[CollisionFreeHolding(obj, grasp, right_q_start, right_traj, right_q_end), Motion(right_q_start, right_traj, right_q_end)],
    costs=[TrajectoryLength(right_q_start, right_traj, right_q_end)],
)

LeftPick = TAMPOperator(
    "Pick",
    [obj, grasp, left_q],
    preconditions=[
        LeftAt(left_q),
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
    constraints=[KinematicConstraint(left_q, grasp), CollisionFreeGrasp(obj, grasp)],
    costs=[GraspCost(obj, grasp)],
)

RightPick = TAMPOperator(
    "Pick",
    [obj, grasp, right_q],
    preconditions=[
        RightAt(right_q),
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
    constraints=[KinematicConstraint(right_q, grasp), CollisionFreeGrasp(obj, grasp)],
    costs=[GraspCost(obj, grasp)],
)

LeftPlace = TAMPOperator(
    "Place",
    [obj, grasp, placement, surface, left_q],
    preconditions=[
        LeftAt(left_q),
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
        KinematicConstraint(left_q, placement),
        StablePlacement(obj, grasp, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)

RightPlace = TAMPOperator(
    "Place",
    [obj, grasp, placement, surface, right_q],
    preconditions=[
        RightAt(right_q),
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
        KinematicConstraint(right_q, placement),
        StablePlacement(obj, grasp, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)


LeftPush = TAMPOperator(
    "Push",
    [button, pose, left_q],
    preconditions=[
        LeftAt(left_q),
        IsButton(button),
        LeftHandEmpty(),
        CanPush(button),
        LeftJustMoved(),
    ],
    add_effects=[ButtonPushed(button), LeftCanMove()],
    del_effects=[LeftJustMoved(), CanPush(button)],  # Note: CFree for Push already encoded in the Move operator
    constraints=[KinematicConstraint(left_q, pose), ValidPush(button, pose)],
    costs=[],
)

RightPush = TAMPOperator(
    "Push",
    [button, pose, right_q],
    preconditions=[
        RightAt(right_q),
        IsButton(button),
        RightHandEmpty(),
        CanPush(button),
        RightJustMoved(),
    ],
    add_effects=[ButtonPushed(button), RightCanMove()],
    del_effects=[RightJustMoved(), CanPush(button)],  # Note: CFree for Push already encoded in the Move operator
    constraints=[KinematicConstraint(right_q, pose), ValidPush(button, pose)],
    costs=[],
)

LeftPushStick = TAMPOperator(
    "PushStick",
    [button, obj, grasp, pose, left_q],
    preconditions=[
        LeftAt(left_q),
        LeftHolding(obj),
        LeftHoldingWithGrasp(obj, grasp),
        IsButton(button),
        IsStick(obj),
        CanPush(button),
        LeftJustMoved(),
    ],
    add_effects=[ButtonPushed(button), LeftCanMove(), PushedWithStick(button, obj)],
    del_effects=[LeftJustMoved(), CanPush(button)],
    # CFree is automatically handled right now within the operator
    constraints=[KinematicConstraint(left_q, pose), ValidPushStick(button, obj, pose)],
    costs=[],
)

RightPushStick = TAMPOperator(
    "PushStick",
    [button, obj, grasp, pose, right_q],
    preconditions=[
        RightAt(right_q),
        RightHolding(obj),
        RightHoldingWithGrasp(obj, grasp),
        IsButton(button),
        IsStick(obj),
        CanPush(button),
        RightJustMoved(),
    ],
    add_effects=[ButtonPushed(button), RightCanMove(), PushedWithStick(button, obj)],
    del_effects=[RightJustMoved(), CanPush(button)],
    # CFree is automatically handled right now within the operator
    constraints=[KinematicConstraint(right_q, pose), ValidPushStick(button, obj, pose)],
    costs=[],
)

all_tamp_operators = [LeftMoveFree, RightMoveFree, LeftMoveHolding, RightMoveHolding, LeftPick, RightPick, LeftPlace, RightPlace, LeftPush, RightPush, LeftPushStick, RightPushStick]


def get_initial_state(
    movables: Sequence[str] = (), surfaces: Sequence[str] = (), sticks: Sequence[str] = (), buttons: Sequence[str] = ()
) -> State:
    """Ground the initial state of the TAMP domain."""
    initial_state = {LeftAt.ground("left_q0"), RightAt.ground("right_q0"), LeftHandEmpty.ground(), RightHandEmpty.ground(), LeftCanMove.ground(), RightCanMove.ground()}
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
