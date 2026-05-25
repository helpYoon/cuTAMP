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

from cutamp.task_planning import Fluent, OperatorMetadata, Parameter, TAMPOperator, State
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
from cutamp.task_planning.costs import GraspCost, RetractCost, TrajectoryLength

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

LeftJustPicked = Fluent("LeftJustPicked")
RightJustPicked = Fluent("RightJustPicked")
LeftJustPlaced = Fluent("LeftJustPlaced")
RightJustPlaced = Fluent("RightJustPlaced")

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

# Tracks that the planar base is positioned within reach of the named target.
# Added by MoveBaseTo(target). Pick/Place can require this in a follow-up pass
# to force the planner to navigate before reaching.
BaseNearby = Fluent("BaseNearby", [Parameter("target", Movable)])

all_t1_fluents = [
    LeftAt,
    RightAt,
    LeftHandEmpty,
    RightHandEmpty,
    LeftCanMove,
    RightCanMove,
    LeftJustMoved,
    RightJustMoved,
    LeftJustPicked,
    RightJustPicked,
    LeftJustPlaced,
    RightJustPlaced,
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
    BaseNearby,
]


# Parameters - generic names, arm info comes from fluent/operator context
q = Parameter("q", Conf)
q_start = Parameter("q_start", Conf)
q_end = Parameter("q_end", Conf)
q_retract = Parameter("q_retract", Conf)
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
# The arm information is now stored in metadata.arm, not encoded in the operator name.

LeftMoveFree = TAMPOperator(
    "LeftMoveFree",
    [q_start, traj, q_end],
    preconditions=[LeftAt(q_start), LeftHandEmpty(), LeftCanMove(), RightCanMove()],  # Require RightCanMove to prevent interleaving
    add_effects=[LeftAt(q_end), LeftJustMoved()],
    del_effects=[LeftAt(q_start), LeftCanMove(), RightCanMove()],  # Block right arm until LeftPick completes
    constraints=[CollisionFree(q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
    metadata=OperatorMetadata(is_motion=True, arm="left"),
)

RightMoveFree = TAMPOperator(
    "RightMoveFree",
    [q_start, traj, q_end],
    preconditions=[RightAt(q_start), RightHandEmpty(), RightCanMove(), LeftCanMove()],  # Require LeftCanMove to prevent interleaving
    add_effects=[RightAt(q_end), RightJustMoved()],
    del_effects=[RightAt(q_start), RightCanMove(), LeftCanMove()],  # Block left arm until RightPick completes
    constraints=[CollisionFree(q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
    metadata=OperatorMetadata(is_motion=True, arm="right"),
)


LeftMoveHolding = TAMPOperator(
    "LeftMoveHolding",
    [obj, grasp, q_start, traj, q_end],
    preconditions=[
        LeftAt(q_start),
        LeftHolding(obj),
        LeftHoldingWithGrasp(obj, grasp),
        LeftCanMove(),
        RightCanMove(),  # Require RightCanMove to prevent interleaving
    ],
    add_effects=[LeftAt(q_end), LeftJustMoved()],
    del_effects=[LeftAt(q_start), LeftCanMove(), RightCanMove()],  # Block right arm until LeftPlace completes
    constraints=[CollisionFreeHolding(obj, grasp, q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
    metadata=OperatorMetadata(is_motion=True, arm="left"),
)

RightMoveHolding = TAMPOperator(
    "RightMoveHolding",
    [obj, grasp, q_start, traj, q_end],
    preconditions=[
        RightAt(q_start),
        RightHolding(obj),
        RightHoldingWithGrasp(obj, grasp),
        RightCanMove(),
        LeftCanMove(),  # Require LeftCanMove to prevent interleaving
    ],
    add_effects=[RightAt(q_end), RightJustMoved()],
    del_effects=[RightAt(q_start), RightCanMove(), LeftCanMove()],  # Block left arm until RightPlace completes
    constraints=[CollisionFreeHolding(obj, grasp, q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
    metadata=OperatorMetadata(is_motion=True, arm="right"),
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
        LeftJustPicked(),  # Trigger retract after pick
    ],
    del_effects=[LeftHandEmpty(), LeftJustMoved(), HasNotPickedUp(obj)],
    constraints=[KinematicConstraint(q, grasp), CollisionFreeGrasp(obj, grasp)],
    costs=[GraspCost(obj, grasp)],
    metadata=OperatorMetadata(is_actionable=True, action_type="pick", arm="left"),
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
        RightJustPicked(),  # Trigger retract after pick
    ],
    del_effects=[RightHandEmpty(), RightJustMoved(), HasNotPickedUp(obj)],
    constraints=[KinematicConstraint(q, grasp), CollisionFreeGrasp(obj, grasp)],
    costs=[GraspCost(obj, grasp)],
    metadata=OperatorMetadata(is_actionable=True, action_type="pick", arm="right"),
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
    add_effects=[LeftHandEmpty(), LeftJustPlaced(), On(obj, surface)],  # Trigger retract after place
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
    metadata=OperatorMetadata(is_actionable=True, action_type="place", arm="left"),
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
    add_effects=[RightHandEmpty(), RightJustPlaced(), On(obj, surface)],  # Trigger retract after place
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
    metadata=OperatorMetadata(is_actionable=True, action_type="place", arm="right"),
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
    metadata=OperatorMetadata(is_actionable=True, action_type="push", arm="left"),
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
    metadata=OperatorMetadata(is_actionable=True, action_type="push", arm="right"),
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
    metadata=OperatorMetadata(is_actionable=True, action_type="push_stick", arm="left"),
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
    metadata=OperatorMetadata(is_actionable=True, action_type="push_stick", arm="right"),
)

# Retract operators - move arm to home configuration after pick/place
# These use soft costs to find the closest collision-free configuration to home

LeftRetractHolding = TAMPOperator(
    "LeftRetractHolding",
    [obj, grasp, q_start, traj, q_retract],
    preconditions=[
        LeftAt(q_start),
        LeftHolding(obj),
        LeftHoldingWithGrasp(obj, grasp),
        LeftJustPicked(),
    ],
    add_effects=[LeftAt(q_retract), LeftCanMove(), RightCanMove()],
    del_effects=[LeftAt(q_start), LeftJustPicked()],
    constraints=[CollisionFreeHolding(obj, grasp, q_start, traj, q_retract), Motion(q_start, traj, q_retract)],
    costs=[RetractCost(q_retract)],
    metadata=OperatorMetadata(is_motion=True, action_type="retract", arm="left"),
)

RightRetractHolding = TAMPOperator(
    "RightRetractHolding",
    [obj, grasp, q_start, traj, q_retract],
    preconditions=[
        RightAt(q_start),
        RightHolding(obj),
        RightHoldingWithGrasp(obj, grasp),
        RightJustPicked(),
    ],
    add_effects=[RightAt(q_retract), RightCanMove(), LeftCanMove()],
    del_effects=[RightAt(q_start), RightJustPicked()],
    constraints=[CollisionFreeHolding(obj, grasp, q_start, traj, q_retract), Motion(q_start, traj, q_retract)],
    costs=[RetractCost(q_retract)],
    metadata=OperatorMetadata(is_motion=True, action_type="retract", arm="right"),
)

LeftRetractFree = TAMPOperator(
    "LeftRetractFree",
    [q_start, traj, q_retract],
    preconditions=[
        LeftAt(q_start),
        LeftHandEmpty(),
        LeftJustPlaced(),
    ],
    add_effects=[LeftAt(q_retract), LeftCanMove(), RightCanMove()],
    del_effects=[LeftAt(q_start), LeftJustPlaced()],
    constraints=[CollisionFree(q_start, traj, q_retract), Motion(q_start, traj, q_retract)],
    costs=[RetractCost(q_retract)],
    metadata=OperatorMetadata(is_motion=True, action_type="retract", arm="left"),
)

RightRetractFree = TAMPOperator(
    "RightRetractFree",
    [q_start, traj, q_retract],
    preconditions=[
        RightAt(q_start),
        RightHandEmpty(),
        RightJustPlaced(),
    ],
    add_effects=[RightAt(q_retract), RightCanMove(), LeftCanMove()],
    del_effects=[RightAt(q_start), RightJustPlaced()],
    constraints=[CollisionFree(q_start, traj, q_retract), Motion(q_start, traj, q_retract)],
    costs=[RetractCost(q_retract)],
    metadata=OperatorMetadata(is_motion=True, action_type="retract", arm="right"),
)

# MoveBaseTo: planar-base navigation bound to a target object.
#
# Both arms must be at the same starting config (LeftAt(q_start) AND
# RightAt(q_start)) — initially this is satisfied via shared "q0" in
# get_initial_state. After MoveBaseTo, both LeftAt and RightAt point to the
# same q_end (which differs from q_start only in base[0:3]). Subsequent arm
# operators may diverge LeftAt/RightAt as before (this is fine — once the base
# is positioned, arm ops do not move it).
#
# The "Bound to target" semantics: q_end's base must place the target object
# within reach of either tool frame. The cost function enforces reachability
# via a BaseReachable check (added separately).
MoveBaseTo = TAMPOperator(
    "MoveBaseTo",
    [obj, q_start, traj, q_end],
    preconditions=[
        LeftAt(q_start), RightAt(q_start),
        LeftCanMove(), RightCanMove(),
        IsMovable(obj),  # binds `obj` to an existing movable literal
    ],
    add_effects=[
        LeftAt(q_end), RightAt(q_end),
        LeftJustMoved(), RightJustMoved(),
        BaseNearby(obj),
    ],
    del_effects=[LeftAt(q_start), RightAt(q_start)],
    constraints=[
        CollisionFree(q_start, traj, q_end),
        Motion(q_start, traj, q_end),
    ],
    costs=[TrajectoryLength(q_start, traj, q_end)],
    metadata=OperatorMetadata(is_motion=True, action_type="navigate", arm=None),
)


all_t1_operators = [
    LeftMoveFree, RightMoveFree,
    LeftMoveHolding, RightMoveHolding,
    LeftPick, RightPick,
    LeftPlace, RightPlace,
    LeftRetractHolding, RightRetractHolding,
    LeftRetractFree, RightRetractFree,
    LeftPush, RightPush,
    LeftPushStick, RightPushStick,
    MoveBaseTo,
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
