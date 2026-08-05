"""Pin lifecycle invariants for T1State."""
import pytest

from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


def _make_state():
    """Build a real T1State with a real MotionPlanner.

    Uses the smallest feasible setup so the test runs in a few seconds.
    Mirrors how ``motion_solver._motion_plan_for_particle`` constructs the
    state: planner from ``world.get_motion_planner()`` and current_js from
    ``_planner_js_from_full(planner, q_init, full_joint_names)``.
    """
    from curobo.types import JointState

    from cutamp.t1_state import T1State

    world = make_blocks_t1_world()
    planner = world.get_motion_planner()

    # Mirror motion_solver._planner_js_from_full: build full-cspace JointState
    # with joint_names, then project to the planner's active cspace.
    full_joint_names = list(world.kinematics.joint_names)
    full_js = JointState.from_position(
        world.q_init[None].to(planner.kinematics.device_cfg.device),
        joint_names=full_joint_names,
    )
    active_js = planner.kinematics.get_active_js(full_js)

    return T1State(
        planner=planner,
        kinematics=world.kinematics,
        tool_from_ee=world.tool_from_ee,
        current_js=active_js,
    )


@needs_cuda
def test_double_pin_for_arm_action_raises():
    """Calling pin_for_arm_action while a pin is already active must raise."""
    state = _make_state()
    state.pin_for_arm_action("left")
    try:
        with pytest.raises(RuntimeError, match="pin"):
            state.pin_for_arm_action("left")
    finally:
        state.unpin()


@needs_cuda
def test_pin_unpin_cycle_repeats():
    """After a successful pin + unpin cycle, calling pin_for_arm_action a
    second time must work (snapshot is None again), proving unpin fully
    cleared state."""
    state = _make_state()
    state.pin_for_arm_action("left")
    state.unpin()
    # Second cycle — must not raise.
    state.pin_for_arm_action("left")
    state.unpin()
