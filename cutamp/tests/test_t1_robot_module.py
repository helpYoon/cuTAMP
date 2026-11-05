"""Tests for the T1 robot module under cuRobo v0.8 single-MotionPlanner.

The dual-arm-specific assertions from the v0.7 era (DualArmRobotContainer,
two kinematics models, etc.) are gone. This file now exercises the single-
container shape: one Kinematics, two tool frames, gripper-spheres dict.
It also covers the outputs of real-leg runs (decision 9): the raw plan, the
refusal in check_plan_margins and the Rerun viewer.
"""

import copy
import pickle

import numpy as np
import pytest

import torch

from cutamp.tests.conftest import needs_cuda


@pytest.fixture
def t1_container():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from curobo.types import DeviceCfg
    from cutamp.robots import SIMPLIFIED, load_t1_container
    return load_t1_container(SIMPLIFIED, DeviceCfg())


def test_t1_container_is_single_robot(t1_container):
    from cutamp.robots import RobotContainer
    assert isinstance(t1_container, RobotContainer)
    assert t1_container.name == "t1"


def test_t1_has_two_tool_frames(t1_container):
    from cutamp.robots.t1 import LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME
    assert set(t1_container.tool_frames) == {LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME}
    assert LEFT_TOOL_FRAME in t1_container.tool_from_ee
    assert RIGHT_TOOL_FRAME in t1_container.tool_from_ee
    assert LEFT_TOOL_FRAME in t1_container.gripper_spheres
    assert RIGHT_TOOL_FRAME in t1_container.gripper_spheres


def test_t1_joint_limits_shape(t1_container):
    from cutamp.robots.t1 import CUROBO_DOF
    assert t1_container.joint_limits.shape == (2, CUROBO_DOF)


def test_t1_kinematics_runs(t1_container):
    from cutamp.robots.t1 import CUROBO_DOF, t1_home
    from curobo.types import JointState

    q = torch.tensor(t1_home, device=t1_container.kinematics.device_cfg.device).unsqueeze(0)
    js = JointState.from_position(q)
    state = t1_container.kinematics.compute_kinematics(js)
    # ToolPose covers both arm tool frames plus mobile_base_link (tracked for
    # the COM-over-base cost; not an IK target).
    assert state.tool_poses.num_links >= 2


def _raw_plan_input():
    """A two-segment plan as solve_curobo returns one, every value distinct: each JointState a batch of one, in the
    real legs' joint order reversed plus the head's two locked joints. The balance gate slowed the second segment."""
    from curobo.types import JointState
    from cutamp.robots.t1 import REAL_LEGS
    names = list(REAL_LEGS.joint_names)[::-1] + ["AAHead_yaw", "Head_pitch"]

    def frames(n, start):
        x = start + torch.arange(n * len(names), dtype=torch.float32).reshape(1, n, len(names))
        return JointState(position=x, velocity=x + 1000, acceleration=x + 2000, joint_names=names)

    return [{"dt": 0.025, "arm": "left", "plan": frames(4, 0.0), "held_objs": {}},
            {"dt": 0.025, "arm": None, "plan": frames(6, 1e4), "held_objs": {},
             "planner_timing": {"plan": frames(3, 2e4), "held_objs": {}}}]


def test_raw_motion_plan_in_model_order():
    # A real-leg plan is saved raw (decision 9): float32 [T, 33] in the model's joint order, whatever order cuRobo's
    # JointStates carry, and a slowed segment keeps its motion planner's timing.
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.utils.plan_processor import raw_motion_plan
    plan = _raw_plan_input()
    raw = raw_motion_plan(plan, REAL_LEGS)
    names = list(REAL_LEGS.joint_names)
    assert (raw["schema_version"], raw["model"], raw["joint_names"]) == ("stage3-raw", "real_legs", names)
    assert len(raw["segments"]) == len(plan)
    for seg, entry in zip(raw["segments"], plan):
        assert set(seg) == {"arm", "dt", "position", "velocity", "acceleration", "planner_timing"}
        assert (seg["arm"], seg["dt"]) == (entry["arm"], entry["dt"])
        pairs = [(seg, entry["plan"])]
        if "planner_timing" in entry:
            pairs.append((seg["planner_timing"], entry["planner_timing"]["plan"]))
        else:
            assert seg["planner_timing"] is None
        for out, js in pairs:
            cols = [js.joint_names.index(n) for n in names]
            for key in ("position", "velocity", "acceleration"):
                assert out[key].dtype == np.float32
                np.testing.assert_array_equal(out[key], getattr(js, key)[0][:, cols].numpy())


def test_mpc_example_refuses_a_raw_plan_until_the_export(tmp_path):
    # The MPC example reads schema 3, the simplified model's. Regenerating a raw real-leg plan saves the same schema,
    # so for one the example says real-leg plans await stage 6's export, not "regenerate".
    import importlib.util
    import pickle
    from pathlib import Path

    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.utils.plan_processor import raw_motion_plan
    path = Path(__file__).resolve().parents[2] / "examples" / "load_motion_plan_for_mpc.py"
    spec = importlib.util.spec_from_file_location("load_motion_plan_for_mpc", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    with open(tmp_path / "motion_plan.pkl", "wb") as f:
        pickle.dump(raw_motion_plan(_raw_plan_input(), REAL_LEGS), f)
    with pytest.raises(RuntimeError, match="real-leg plans await stage 6's export") as refused:
        example.load_for_mpc(tmp_path / "motion_plan.pkl")
    assert "egenerate" not in str(refused.value)


def test_check_plan_margins_refuses_raw(tmp_path):
    # check_plan_margins gates the simplified model's margin bands: the real legs have none (decision 6), and a raw
    # plan has no schema-3 keys. It refuses one by name, with a non-zero exit.
    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.scripts.check_plan_margins import main
    from cutamp.utils.plan_processor import raw_motion_plan
    path = tmp_path / "raw.pkl"
    path.write_bytes(pickle.dumps(raw_motion_plan(_raw_plan_input(), REAL_LEGS)))
    with pytest.raises(SystemExit) as stop:
        main(["--plan", str(path)])
    assert stop.value.code not in (0, None)
    assert "simplified model's schema 3 only" in str(stop.value.code)


def _poses(translation, mat3x3) -> np.ndarray:
    """``[n, 4, 4]`` from Rerun's Transform3D translations and 3x3 matrices, which Rerun stores column-major."""
    pose = np.tile(np.eye(4), (len(translation), 1, 1))
    pose[:, :3, 3] = np.asarray(translation, dtype=np.float64).reshape(-1, 3)
    pose[:, :3, :3] = np.asarray(mat3x3, dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
    return pose


def _logged_poses(path, entity: str) -> np.ndarray:
    """``[n, 4, 4]``: each Transform3D logged at ``entity`` in the Rerun recording saved at ``path``, in log order."""
    import rerun as rr
    table = rr.dataframe.load_recording(str(path)).view(index="log_tick", contents=f"/{entity}").select().read_all()
    return _poses(table[f"/{entity}:Transform3D:translation"].to_pylist(),
                  table[f"/{entity}:Transform3D:mat3x3"].to_pylist())


def _column_poses(columns) -> np.ndarray:
    """``[n, 4, 4]`` from Transform3D columns (``get_rr_columns``'s values)."""
    values = {str(c.component_descriptor()): c.as_arrow_array().to_pylist() for c in columns}
    return _poses(values["Transform3D:translation"], values["Transform3D:mat3x3"])


def _record(path, make_robot, *calls):
    """Run ``make_robot()`` and then each ``call(robot)`` into an in-memory Rerun recording, saved at ``path``.
    Returns the robot."""
    import rerun as rr
    rec = rr.RecordingStream("cutamp_test_viewer")
    memory = rec.memory_recording()
    with rec:
        robot = make_robot()
        for call in calls:
            call(robot)
    rec.flush()
    path.write_bytes(memory.drain_as_bytes())
    return robot


#: yourdfpy's scene build calls float() on a 1-element array once per actuated joint: numpy's deprecation, not ours.
yourdfpy_float = pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0:DeprecationWarning")


@needs_cuda
@yourdfpy_float
def test_rerun_robot_real_legs(tmp_path):
    # The real-leg viewer (decision 9). The root's Transform3D is the base pose, the Trunk's world pose: at a tilt
    # with pitch, roll and yaw all non-zero, cuRobo's FK of the Trunk, so a wrong order of the base's YXZ angles
    # fails (each other order is checked to differ). Every actuated URDF joint gets a value by name: the model's
    # from q, the fingers from the gripper state, the head 0; both hands then land where cuRobo puts them. A conf
    # takes exactly the model's joints. The trajectory path (get_rr_columns) logs the same root.
    import roma
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import DeviceCfg, JointState
    from cutamp.robots.t1 import GRIPPER_CLOSED, GRIPPER_OPEN, REAL_LEGS, load_t1_rerun, t1_curobo_cfg

    names = list(REAL_LEGS.joint_names)
    q = torch.tensor(REAL_LEGS.home, dtype=torch.float64)
    base = {"base_j_x": 0.05, "base_j_y": -0.03, "base_j_z": -0.02, "base_j_pitch": 0.4, "base_j_roll": -0.3,
            "base_j_yaw": 0.6}
    assert sorted(base) == sorted(names[REAL_LEGS.base])
    for joint, delta in base.items():
        q[names.index(joint)] += delta
    others = [i for i, n in enumerate(names) if n not in base]
    q[others] += 0.01 * torch.arange(1, len(others) + 1, dtype=q.dtype)  # every other joint a distinct value

    path = tmp_path / "viewer.rrd"
    robot = _record(path, lambda: load_t1_rerun(REAL_LEGS, load_mesh=False),
                    lambda r: r.set_grippers(left=GRIPPER_CLOSED, right=GRIPPER_OPEN),
                    lambda r: r.set_joint_positions(q.tolist()))
    for wrong in (q.tolist()[:-1], q.tolist() + list(GRIPPER_CLOSED)):
        with pytest.raises(ValueError, match="33 joints"):
            robot.set_joint_positions(wrong)

    frames = ["Trunk", "left_base_link", "right_base_link"]
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(t1_curobo_cfg(REAL_LEGS)), tool_frames=frames,
                                                        device_cfg=DeviceCfg()))
    q_kin = q[[names.index(n) for n in kin.joint_names]].float().to(kin.device_cfg.device)
    ks = kin.compute_kinematics(JointState.from_position(q_kin[None]))
    fk = {f: ks.tool_poses.get_link_pose(f, make_contiguous=True).get_matrix().reshape(4, 4).double().cpu().numpy()
          for f in frames}

    root = _logged_poses(path, "t1")[-1]
    np.testing.assert_allclose(root, fk["Trunk"], rtol=0, atol=1e-5)
    angle = {axis: q[names.index(joint)].item() for axis, joint in
             (("Y", "base_j_pitch"), ("X", "base_j_roll"), ("Z", "base_j_yaw"))}
    for order in ("XYZ", "XZY", "YZX", "ZXY", "ZYX"):  # the other orders of the same three rotations
        wrong = roma.euler_to_rotmat(order, torch.tensor([angle[a] for a in order], dtype=torch.float64))
        assert np.abs(wrong.numpy() - fk["Trunk"][:3, :3]).max() > 0.05, order

    cfg = dict(zip(robot.urdf.actuated_joint_names, robot.urdf.cfg))
    fingers = {f"{side}_{link}": value for side, state in (("left", GRIPPER_CLOSED), ("right", GRIPPER_OPEN))
               for link, value in zip(("Link1", "Link11", "Link2", "Link22"), state)}
    want = {**{names[i]: q[i].item() for i in others}, **fingers, "AAHead_yaw": 0.0, "Head_pitch": 0.0}
    assert sorted(cfg) == sorted(want)
    for joint, value in want.items():
        assert cfg[joint] == pytest.approx(value, abs=1e-9), joint
    for hand in frames[1:]:
        np.testing.assert_allclose(root @ robot.urdf.get_transform(hand), fk[hand], rtol=0, atol=1e-5, err_msg=hand)

    columns = robot.get_rr_columns(torch.stack([torch.tensor(REAL_LEGS.home, dtype=torch.float64), q]))
    np.testing.assert_allclose(_column_poses(columns["t1"])[1], fk["Trunk"], rtol=0, atol=1e-5)


@yourdfpy_float
def test_rerun_robot_simplified_unchanged(tmp_path):
    # The simplified viewer keeps its positional mapping: a 21-DOF conf logs the planar base as the root's
    # Transform3D and fills the URDF through curobo_to_urdf_joints.
    from cutamp.robots.t1 import GRIPPER_CLOSED, SIMPLIFIED, base_se3_from_q, curobo_to_urdf_joints, load_t1_rerun
    q = [0.1, -0.2, 0.3] + [0.01 * k for k in range(1, len(SIMPLIFIED.joint_names) - 2)]
    path = tmp_path / "viewer.rrd"
    robot = _record(path, lambda: load_t1_rerun(SIMPLIFIED, load_mesh=False),
                    lambda r: r.set_grippers(left=GRIPPER_CLOSED), lambda r: r.set_joint_positions(q))
    np.testing.assert_allclose(_logged_poses(path, "t1")[-1], base_se3_from_q(q).numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(robot.urdf.cfg, curobo_to_urdf_joints(q, left_gripper=GRIPPER_CLOSED), atol=1e-6)
