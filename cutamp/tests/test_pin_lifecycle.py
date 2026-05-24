"""Pin lifecycle invariants for T1State."""
import os

import pytest

needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _make_state():
    """Build a real T1State with a real MotionPlanner.

    Uses the smallest feasible setup so the test runs in a few seconds.
    Mirrors how ``motion_solver._motion_plan_for_particle`` constructs the
    state: planner from ``world.get_motion_planner()`` and current_js from
    ``_planner_js_from_full(planner, q_init, full_joint_names)``.
    """
    from curobo.types import DeviceCfg, JointState

    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.robots import get_q_home, load_robot_container
    from cutamp.t1_state import T1State
    from cutamp.tamp_world import TAMPWorld

    device_cfg = DeviceCfg()
    robot_container = load_robot_container(device_cfg=device_cfg)
    q_init = device_cfg.to_device(get_q_home())
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    world = TAMPWorld(env, device_cfg, robot=robot_container, q_init=q_init)
    planner = world.get_motion_planner()

    # Mirror motion_solver._planner_js_from_full: build full-cspace JointState
    # with joint_names, then project to the planner's active cspace.
    full_joint_names = list(world.kinematics.joint_names)
    full_js = JointState.from_position(
        q_init[None].to(planner.kinematics.device_cfg.device),
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
def test_double_pin_for_movebase_raises():
    """Calling pin_for_movebase while a pin is active must raise, matching
    pin_for_arm_action's guard."""
    state = _make_state()
    state.pin_for_movebase()
    try:
        with pytest.raises(RuntimeError, match="pin"):
            state.pin_for_movebase()
    finally:
        state.unpin()


@needs_cuda
def test_unpin_re_enables_tool_pose_after_partial_apply():
    """Even if pin_for_movebase succeeds in disabling tool_pose criteria,
    unpin must always re-enable them. Specifically: after a successful pin
    + unpin cycle, calling pin_for_movebase a second time must work
    (snapshot is None again), proving unpin fully cleared state."""
    state = _make_state()
    state.pin_for_movebase()
    state.unpin()
    # Second cycle — must not raise.
    state.pin_for_movebase()
    state.unpin()
