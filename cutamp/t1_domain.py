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
One factory builds each arm's fluents and one its operators, so the two sides stay mirror images.
"""

from typing import Sequence

from cutamp.task_planning import Fluent, OperatorMetadata, Parameter, TAMPOperator, State
from cutamp.task_planning.constraints import KinematicConstraint, Motion, StablePlacement
from cutamp.task_planning.costs import TrajectoryLength

# Types
Conf = "conf"
Traj = "traj"
Pose = "pose"
Grasp = "grasp"

Movable = "movable"
Surface = "surface"


# Fluents (aka predicates)
def _arm_fluents(arm: str) -> dict[str, Fluent]:
    """One arm's fluents, named <Arm><key> (LeftAt, RightAt, LeftHandEmpty, ...)."""
    s = arm.capitalize()
    return {
        "At": Fluent(f"{s}At", [Parameter("q", Conf)]),
        "HandEmpty": Fluent(f"{s}HandEmpty"),
        "CanMove": Fluent(f"{s}CanMove"),
        "JustMoved": Fluent(f"{s}JustMoved"),
        "JustPicked": Fluent(f"{s}JustPicked"),
        "JustPlaced": Fluent(f"{s}JustPlaced"),
        # Deliberately redundant with *HoldingWithGrasp: these are the grasp-free goal predicates
        # reachable from env YAML goals (which can name an object but not a grasp literal).
        "Holding": Fluent(f"{s}Holding", [Parameter("obj", Movable)]),
        "HoldingWithGrasp": Fluent(f"{s}HoldingWithGrasp", [Parameter("obj", Movable), Parameter("grasp", Grasp)]),
    }


_ARM_FLUENTS = {arm: _arm_fluents(arm) for arm in ("left", "right")}
_LEFT, _RIGHT = _ARM_FLUENTS["left"], _ARM_FLUENTS["right"]

IsMovable = Fluent("IsMovable", [Parameter("obj", Movable)])
IsSurface = Fluent("IsSurface", [Parameter("surface", Surface)])
HasNotPickedUp = Fluent("HasNotPickedUp", [Parameter("obj", Movable)])
On = Fluent("On", [Parameter("obj", Movable), Parameter("surface", Surface)])

# LeftAt, RightAt, LeftHandEmpty, RightHandEmpty, ..., then the arm-free fluents
all_t1_fluents = [f for pair in zip(_LEFT.values(), _RIGHT.values()) for f in pair] + [
    IsMovable,
    IsSurface,
    HasNotPickedUp,
    On,
]


# Parameters - generic names, arm info comes from fluent/operator context
q = Parameter("q", Conf)
q_start = Parameter("q_start", Conf)
q_end = Parameter("q_end", Conf)
q_retract = Parameter("q_retract", Conf)
traj = Parameter("traj", Traj)

obj = Parameter("obj", Movable)
surface = Parameter("surface", Surface)

grasp = Parameter("grasp", Grasp)
placement = Parameter("placement", Pose)


# Operators - this is the important part!
def _arm_operators(arm: str) -> list[TAMPOperator]:
    """One arm's six operators, named <Arm><op>: MoveFree, MoveHolding, Pick, Place, RetractHolding, RetractFree.

    They use only this arm's fluents (LeftAt, LeftHandEmpty, ...), except that a move needs and clears both arms'
    CanMove, so the arms never interleave: the other arm waits until this arm's retract, after its pick or place, gives
    CanMove back to both. metadata.arm records the arm.
    """
    me, other = _ARM_FLUENTS[arm], _ARM_FLUENTS["right" if arm == "left" else "left"]
    s = arm.capitalize()
    at, empty, can_move, other_can_move = me["At"], me["HandEmpty"], me["CanMove"], other["CanMove"]
    holding, holding_grasp = me["Holding"], me["HoldingWithGrasp"]
    just_moved, just_picked, just_placed = me["JustMoved"], me["JustPicked"], me["JustPlaced"]

    move_free = TAMPOperator(
        f"{s}MoveFree",
        [q_start, traj, q_end],
        preconditions=[at(q_start), empty(), can_move(), other_can_move()],
        add_effects=[at(q_end), just_moved()],
        del_effects=[at(q_start), can_move(), other_can_move()],
        constraints=[Motion(q_start, traj, q_end)],
        costs=[TrajectoryLength(q_start, traj, q_end)],
        metadata=OperatorMetadata(is_motion=True, arm=arm),
    )
    move_holding = TAMPOperator(
        f"{s}MoveHolding",
        [obj, grasp, q_start, traj, q_end],
        preconditions=[at(q_start), holding(obj), holding_grasp(obj, grasp), can_move(), other_can_move()],
        add_effects=[at(q_end), just_moved()],
        del_effects=[at(q_start), can_move(), other_can_move()],
        constraints=[Motion(q_start, traj, q_end)],
        costs=[TrajectoryLength(q_start, traj, q_end)],
        metadata=OperatorMetadata(is_motion=True, arm=arm),
    )
    pick = TAMPOperator(
        f"{s}Pick",
        [obj, grasp, q],
        preconditions=[at(q), empty(), IsMovable(obj), just_moved(), HasNotPickedUp(obj)],
        add_effects=[holding(obj), holding_grasp(obj, grasp), just_picked()],  # JustPicked triggers the retract
        del_effects=[empty(), just_moved(), HasNotPickedUp(obj)],
        constraints=[KinematicConstraint(q, grasp)],
        costs=[],
        metadata=OperatorMetadata(is_actionable=True, action_type="pick", arm=arm),
    )
    place = TAMPOperator(
        f"{s}Place",
        [obj, grasp, placement, surface, q],
        preconditions=[at(q), holding(obj), holding_grasp(obj, grasp), IsSurface(surface), just_moved()],
        add_effects=[empty(), just_placed(), On(obj, surface)],  # JustPlaced triggers the retract
        del_effects=[holding(obj), holding_grasp(obj, grasp), just_moved()],
        constraints=[KinematicConstraint(q, placement), StablePlacement(obj, grasp, placement, surface)],
        costs=[],
        metadata=OperatorMetadata(is_actionable=True, action_type="place", arm=arm),
    )
    # Retracts move the arm back towards home after a pick or place; the optimizer's soft costs pick the closest
    # collision-free configuration to home.
    retract_holding = TAMPOperator(
        f"{s}RetractHolding",
        [obj, grasp, q_start, traj, q_retract],
        preconditions=[at(q_start), holding(obj), holding_grasp(obj, grasp), just_picked()],
        add_effects=[at(q_retract), can_move(), other_can_move()],
        del_effects=[at(q_start), just_picked()],
        constraints=[Motion(q_start, traj, q_retract)],
        costs=[],
        metadata=OperatorMetadata(is_motion=True, action_type="retract", arm=arm),
    )
    retract_free = TAMPOperator(
        f"{s}RetractFree",
        [q_start, traj, q_retract],
        preconditions=[at(q_start), empty(), just_placed()],
        add_effects=[at(q_retract), can_move(), other_can_move()],
        del_effects=[at(q_start), just_placed()],
        constraints=[Motion(q_start, traj, q_retract)],
        costs=[],
        metadata=OperatorMetadata(is_motion=True, action_type="retract", arm=arm),
    )
    return [move_free, move_holding, pick, place, retract_holding, retract_free]


LeftMoveFree, LeftMoveHolding, LeftPick, LeftPlace, LeftRetractHolding, LeftRetractFree = _arm_operators("left")
RightMoveFree, RightMoveHolding, RightPick, RightPlace, RightRetractHolding, RightRetractFree = _arm_operators("right")

all_t1_operators = [
    LeftMoveFree, RightMoveFree,
    LeftMoveHolding, RightMoveHolding,
    LeftPick, RightPick,
    LeftPlace, RightPlace,
    LeftRetractHolding, RightRetractHolding,
    LeftRetractFree, RightRetractFree,
]


def get_initial_state(movables: Sequence[str] = (), surfaces: Sequence[str] = ()) -> State:
    """Ground the initial state of the T1 TAMP domain.
    
    Note: left_q0 is the initial left arm configuration, right_q0 is the initial right arm configuration.
    The fluent (LeftAt vs RightAt) determines which arm the configuration belongs to.
    """
    initial_state = {
        _LEFT["At"].ground("left_q0"),
        _RIGHT["At"].ground("right_q0"),
        _LEFT["HandEmpty"].ground(),
        _RIGHT["HandEmpty"].ground(),
        _LEFT["CanMove"].ground(),
        _RIGHT["CanMove"].ground(),
    }
    for movable in movables:
        initial_state.add(IsMovable.ground(movable))
        initial_state.add(HasNotPickedUp.ground(movable))

    for surface in surfaces:
        initial_state.add(IsSurface.ground(surface))

    initial_state = frozenset(initial_state)
    return initial_state
