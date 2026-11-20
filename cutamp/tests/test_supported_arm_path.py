import torch

from curobo.types import JointState
from cutamp.robots.t1 import REAL_LEGS


def test_direct_supported_path_keeps_planted_legs_and_joins_a_moving_arm():
    from cutamp.motion_solver import _direct_supported_path
    from cutamp.step_ik import _kinematics

    home = torch.tensor(REAL_LEGS.home)
    start = home.clone()
    start[REAL_LEGS.right_arm.start] -= 0.24
    velocity = torch.zeros_like(start)
    velocity[REAL_LEGS.right_arm.start] = 0.07
    end = home.clone()
    end[REAL_LEGS.left_arm.start] -= 0.7
    names = list(REAL_LEGS.joint_names)
    js = JointState(start[None], velocity[None], torch.zeros_like(start)[None], joint_names=names)
    plan = _direct_supported_path(js, end, REAL_LEGS, 0.025, "left")
    torch.testing.assert_close(plan.position[0], start)
    torch.testing.assert_close(plan.velocity[0], velocity)
    torch.testing.assert_close(plan.position[-1], end)
    torch.testing.assert_close(plan.velocity[-1], torch.zeros_like(start))
    return_to_home = plan.position[:, REAL_LEGS.right_arm.start]
    assert torch.all(return_to_home.diff() >= -1e-6)
    feet, _ = _kinematics(torch.device("cpu")).fk(plan.position.double(), "left", False)
    torch.testing.assert_close(feet[:, :24], feet[:1, :24].expand_as(feet[:, :24]), atol=2e-5, rtol=0)


def test_idle_arm_reference_removes_an_unnecessary_between_endpoint_detour():
    from cutamp.motion_solver import _smooth_arm_segment

    names = list(REAL_LEGS.joint_names)
    q = torch.tensor(REAL_LEGS.home).repeat(81, 1)
    q[40, REAL_LEGS.right_arm.start] += 0.7
    q[40, REAL_LEGS.left_arm.start] -= 0.3
    path = JointState.from_position(q, joint_names=names)
    preferred = _smooth_arm_segment(path, names[REAL_LEGS.right_arm], 0.025)
    torch.testing.assert_close(
        preferred.position[:, REAL_LEGS.right_arm], q[:1, REAL_LEGS.right_arm].expand(len(q), 7)
    )
    torch.testing.assert_close(preferred.position[:, REAL_LEGS.left_arm], q[:, REAL_LEGS.left_arm])
    torch.testing.assert_close(path.position, q)


def test_idle_arm_moves_only_as_far_as_needed_to_clear_a_blocked_path():
    from cutamp.motion_solver import _select_idle_arm_route

    names = list(REAL_LEGS.joint_names)
    q = torch.tensor(REAL_LEGS.home).repeat(61, 1)
    idle = names[REAL_LEGS.left_arm]
    shoulder = names.index(idle[0])
    q[:, shoulder] += torch.sin(torch.linspace(0, torch.pi, len(q))) * 0.8
    path = JointState.from_position(q, joint_names=names)

    def collides(candidate):
        return candidate.position[30, shoulder] < q[0, shoulder] + 0.30

    cleared, reason = _select_idle_arm_route(path, idle, 0.025, collides)
    assert not collides(cleared)
    assert cleared.position[30, shoulder] - q[0, shoulder] < 0.50
    assert reason == "curobo_collision_avoidance"
    nominal, reason = _select_idle_arm_route(path, idle, 0.025, lambda _: False)
    torch.testing.assert_close(nominal.position[:, shoulder], q[:1, shoulder].expand(len(q)), atol=1e-5, rtol=0)
    assert reason == "curobo_nominal_arm"


def test_direct_crouch_keeps_both_soles_planted_between_feasible_endpoints():
    from cutamp.motion_solver import _direct_supported_path
    from cutamp.step_ik import _kinematics

    home = torch.tensor(REAL_LEGS.home, dtype=torch.float64)
    target = home.clone()
    names = list(REAL_LEGS.joint_names)
    kin = _kinematics(home.device)

    def feet(q):
        return kin.fk(q, "left", False)[0].reshape(-1, 3, 12)[:, :2]

    for side in ("Left", "Right"):
        for joint, delta in [("Hip_Pitch", -0.35), ("Knee_Pitch", 0.7), ("Ankle_Pitch", -0.35)]:
            target[names.index(f"{side}_{joint}")] += delta
    start_feet = feet(home[None])
    target[:3] += (start_feet - feet(target[None]))[0, :, :3].mean(0)
    torch.testing.assert_close(feet(target[None]), start_feet)
    middle = feet(((home + target) * 0.5)[None])
    assert (middle - start_feet)[..., :3].abs().max() > 0.005
    plan = _direct_supported_path(
        JointState.from_position(home[None], joint_names=names), target, REAL_LEGS, 0.025, "left"
    )
    torch.testing.assert_close(
        feet(plan.position), start_feet.expand(len(plan.position), -1, -1), atol=0.0001, rtol=0
    )
