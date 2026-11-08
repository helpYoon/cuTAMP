"""The stage4-raw plan's ``dt`` label (``motion_solver._plan_dt``): it must be the spacing of the positions
``get_interpolated_plan()`` returns, cuRobo's ``interpolation_dt``. The optimizer's
``js_solution.dt`` (0.0145-0.0246 s) played plans 1.02-1.54x too fast."""

import numpy as np
import torch

from curobo.types import JointState

import cutamp.motion_solver as ms
from cutamp.tests.conftest import make_blocks_t1_world, needs_cuda


@needs_cuda
def test_plan_dt_labels_interpolated_spacing():
    world = make_blocks_t1_world()
    planner = ms.build_motion_planner(world)
    for obj in world.movables:
        planner.scene_collision_checker.enable_obstacle(name=obj.name, enable=False)
    names = list(world.kinematics.joint_names)
    start = world.q_init.view(1, -1).clone()
    goal = start.clone()
    # Arm joints only: every model has them, and they leave the real legs' soles where they stand.
    for joint, delta in (("Right_Shoulder_Pitch", -1.2), ("Left_Shoulder_Pitch", -0.6),
                         ("Left_Elbow_Yaw", 0.4), ("Right_Elbow_Yaw", -0.3)):
        goal[0, names.index(joint)] += delta
    result = planner.plan_cspace(JointState.from_position(goal, joint_names=names),
                                 JointState.from_position(start, joint_names=names))
    assert bool(result.success.any())
    plan = result.get_interpolated_plan()
    dt = ms._plan_dt(planner)
    p, v, a = (x.reshape(-1, x.shape[-1]).cpu().numpy()
               for x in (plan.position, plan.velocity, plan.acceleration))
    # Cubic-Hermite position step from the stored velocities and accelerations; k is the
    # ratio of the real spacing to the label (1.00000 at 0.025 s on this plan).
    h = dt * (v[:-1] + v[1:]) / 2 + dt**2 * (a[:-1] - a[1:]) / 12
    dp = np.diff(p, axis=0)
    k = float((dp * h).sum() / (h * h).sum())
    assert abs(k - 1.0) < 1e-3, f"positions are spaced {k:.4f}x the label {dt}"
