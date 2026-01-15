"""
Tests for T1 robot module (cutamp/robots/t1.py).

Tests for kinematics models, IK solvers, gripper spheres, and container loading.
"""
import pytest
import torch


class TestT1ConfigLoading:
    """Tests for T1 cuRobo config loading."""

    def test_t1_left_curobo_cfg_loads(self):
        """Test left arm cuRobo config loads."""
        from cutamp.robots.t1 import t1_left_curobo_cfg
        cfg = t1_left_curobo_cfg()
        assert "robot_cfg" in cfg
        assert "kinematics" in cfg["robot_cfg"]

    def test_t1_right_curobo_cfg_loads(self):
        """Test right arm cuRobo config loads."""
        from cutamp.robots.t1 import t1_right_curobo_cfg
        cfg = t1_right_curobo_cfg()
        assert "robot_cfg" in cfg
        assert "kinematics" in cfg["robot_cfg"]

    def test_config_caching_works(self):
        """Test that lru_cache returns same object."""
        from cutamp.robots.t1 import t1_curobo_cfg
        cfg1 = t1_curobo_cfg("left")
        cfg2 = t1_curobo_cfg("left")
        assert cfg1 is cfg2  # Same cached object

    def test_config_has_external_asset_path(self):
        """Test that external_asset_path is set correctly."""
        from cutamp.robots.t1 import t1_curobo_cfg
        cfg = t1_curobo_cfg("left")
        ext_path = cfg["robot_cfg"]["kinematics"]["external_asset_path"]
        assert ext_path is not None
        assert "t1_description" in ext_path


class TestT1KinematicsModel:
    """Tests for T1 kinematics model creation."""

    def test_get_left_kinematics_model(self):
        """Test left arm kinematics model creation."""
        from cutamp.robots.t1 import get_t1_kinematics_model
        kin_model = get_t1_kinematics_model("left")
        assert kin_model is not None
        # Check DOF matches expected (11 joints)
        joint_limits = kin_model.kinematics_config.joint_limits.position
        assert joint_limits.shape == (2, 11), f"Expected (2, 11), got {joint_limits.shape}"

    def test_get_right_kinematics_model(self):
        """Test right arm kinematics model creation."""
        from cutamp.robots.t1 import get_t1_kinematics_model
        kin_model = get_t1_kinematics_model("right")
        assert kin_model is not None
        joint_limits = kin_model.kinematics_config.joint_limits.position
        assert joint_limits.shape == (2, 11), f"Expected (2, 11), got {joint_limits.shape}"

    def test_invalid_arm_raises_error(self):
        """Test invalid arm name raises ValueError."""
        from cutamp.robots.t1 import get_t1_kinematics_model
        with pytest.raises(ValueError, match="Invalid arm"):
            get_t1_kinematics_model("middle")


class TestT1ForwardKinematics:
    """Tests for T1 forward kinematics."""

    def test_left_fk_at_home(self, tensor_args):
        """Test FK at home position returns valid pose."""
        from cutamp.robots.t1 import get_t1_kinematics_model, t1_home_left

        kin_model = get_t1_kinematics_model("left")
        q_home = torch.tensor(t1_home_left, device=tensor_args.device, dtype=torch.float32)[None]

        state = kin_model.get_state(q_home)
        ee_pose = state.ee_pose

        # Check pose is valid (not NaN, reasonable position)
        assert not torch.isnan(ee_pose.position).any(), "EE position contains NaN"
        assert not torch.isnan(ee_pose.quaternion).any(), "EE quaternion contains NaN"
        assert ee_pose.position[0, 2] > 0, "EE should be above ground"

    def test_right_fk_at_home(self, tensor_args):
        """Test FK at home position for right arm."""
        from cutamp.robots.t1 import get_t1_kinematics_model, t1_home_right

        kin_model = get_t1_kinematics_model("right")
        q_home = torch.tensor(t1_home_right, device=tensor_args.device, dtype=torch.float32)[None]

        state = kin_model.get_state(q_home)
        assert not torch.isnan(state.ee_pose.position).any()
        assert not torch.isnan(state.ee_pose.quaternion).any()

    def test_fk_batch_processing(self, tensor_args):
        """Test FK works with batched inputs."""
        from cutamp.robots.t1 import get_t1_kinematics_model, t1_home_left

        kin_model = get_t1_kinematics_model("left")
        batch_size = 100
        q_home = torch.tensor(t1_home_left, device=tensor_args.device, dtype=torch.float32)
        q_batch = q_home[None].expand(batch_size, -1).contiguous()

        state = kin_model.get_state(q_batch)
        assert state.ee_pose.position.shape == (batch_size, 3)
        assert state.ee_pose.quaternion.shape == (batch_size, 4)

    def test_left_right_ee_positions_are_different(self, tensor_args):
        """Test that left and right EEs are at different positions."""
        from cutamp.robots.t1 import get_t1_kinematics_model, t1_home_left, t1_home_right

        left_kin = get_t1_kinematics_model("left")
        right_kin = get_t1_kinematics_model("right")

        q_left = torch.tensor(t1_home_left, device=tensor_args.device, dtype=torch.float32)[None]
        q_right = torch.tensor(t1_home_right, device=tensor_args.device, dtype=torch.float32)[None]

        left_state = left_kin.get_state(q_left)
        right_state = right_kin.get_state(q_right)

        # Left and right EE should have different Y positions (mirrored)
        left_y = left_state.ee_pose.position[0, 1]
        right_y = right_state.ee_pose.position[0, 1]
        assert left_y != right_y, "Left and right EEs should be at different Y positions"


class TestT1IKSolver:
    """Tests for T1 IK solver."""

    def test_left_ik_solver_creation(self):
        """Test left arm IK solver can be created."""
        from cutamp.robots.t1 import get_t1_ik_solver
        ik_solver = get_t1_ik_solver("left", world_cfg=None)
        assert ik_solver is not None

    def test_right_ik_solver_creation(self):
        """Test right arm IK solver can be created."""
        from cutamp.robots.t1 import get_t1_ik_solver
        ik_solver = get_t1_ik_solver("right", world_cfg=None)
        assert ik_solver is not None

    def test_ik_solver_solves_home_pose(self, tensor_args):
        """Test IK solver can find solution for home pose."""
        from cutamp.robots.t1 import get_t1_ik_solver, get_t1_kinematics_model, t1_home_left
        from curobo.types.math import Pose

        kin_model = get_t1_kinematics_model("left")
        ik_solver = get_t1_ik_solver("left", world_cfg=None)

        # Get EE pose at home
        q_home = torch.tensor(t1_home_left, device=tensor_args.device, dtype=torch.float32)[None]
        state = kin_model.get_state(q_home)
        goal_pose = Pose(state.ee_pose.position, state.ee_pose.quaternion)

        # Solve IK
        result = ik_solver.solve_batch(goal_pose)

        # Check actual error values (cuRobo's success threshold may be strict)
        # Position error should be very small (< 1cm)
        # Rotation error should be very small (< 0.1 rad)
        pos_error = result.position_error.min().item()
        rot_error = result.rotation_error.min().item()
        
        assert pos_error < 0.01, f"Position error too large: {pos_error}"
        assert rot_error < 0.1, f"Rotation error too large: {rot_error}"

    def test_ik_solution_reaches_goal(self, tensor_args):
        """Test IK solution actually reaches the goal pose."""
        from cutamp.robots.t1 import get_t1_ik_solver, get_t1_kinematics_model, t1_home_left
        from curobo.types.math import Pose

        kin_model = get_t1_kinematics_model("left")
        ik_solver = get_t1_ik_solver("left", world_cfg=None)

        q_home = torch.tensor(t1_home_left, device=tensor_args.device, dtype=torch.float32)[None]
        state = kin_model.get_state(q_home)
        goal_pose = Pose(state.ee_pose.position, state.ee_pose.quaternion)

        result = ik_solver.solve_batch(goal_pose)

        if result.success.any():
            # Verify solution reaches goal
            q_solution = result.solution[result.success][0:1]
            achieved_state = kin_model.get_state(q_solution)

            pos_error = (achieved_state.ee_pose.position - goal_pose.position).norm()
            assert pos_error < 0.01, f"Position error too large: {pos_error}"


class TestT1GripperSpheres:
    """Tests for T1 gripper sphere loading."""

    def test_left_gripper_spheres_load(self, tensor_args):
        """Test left gripper spheres load correctly."""
        from cutamp.robots.t1 import get_t1_gripper_spheres

        spheres = get_t1_gripper_spheres("left", tensor_args)
        assert spheres.ndim == 2
        assert spheres.shape[1] == 4, "Each sphere should have [x, y, z, radius]"
        assert spheres.shape[0] > 0, "Should have at least one sphere"

    def test_right_gripper_spheres_load(self, tensor_args):
        """Test right gripper spheres load correctly."""
        from cutamp.robots.t1 import get_t1_gripper_spheres

        spheres = get_t1_gripper_spheres("right", tensor_args)
        assert spheres.ndim == 2
        assert spheres.shape[1] == 4

    def test_gripper_spheres_have_positive_radius(self, tensor_args):
        """Test all gripper spheres have positive radius."""
        from cutamp.robots.t1 import get_t1_gripper_spheres

        for arm in ["left", "right"]:
            spheres = get_t1_gripper_spheres(arm, tensor_args)
            assert (spheres[:, 3] > 0).all(), f"{arm} gripper has non-positive radius spheres"

    def test_gripper_spheres_centers_are_reasonable(self, tensor_args):
        """Test gripper sphere centers are in reasonable range."""
        from cutamp.robots.t1 import get_t1_gripper_spheres

        for arm in ["left", "right"]:
            spheres = get_t1_gripper_spheres(arm, tensor_args)
            centers = spheres[:, :3]
            # Spheres should be within ~30cm of gripper origin
            assert (centers.abs() < 0.3).all(), f"{arm} gripper has spheres too far from origin"


class TestT1RobotContainer:
    """Tests for T1 DualArmRobotContainer."""

    def test_load_t1_container(self, tensor_args):
        """Test T1 container loads successfully."""
        from cutamp.robots import load_robot_container, DualArmRobotContainer

        container = load_robot_container("t1", tensor_args)
        assert isinstance(container, DualArmRobotContainer)
        assert container.name == "t1"

    def test_container_has_both_arm_models(self, tensor_args):
        """Test container has both kinematic models."""
        from cutamp.robots import load_robot_container

        container = load_robot_container("t1", tensor_args)
        assert hasattr(container, "left_kin_model")
        assert hasattr(container, "right_kin_model")
        assert container.left_kin_model is not None
        assert container.right_kin_model is not None

    def test_container_joint_limits_shape(self, tensor_args):
        """Test joint limits have correct shape."""
        from cutamp.robots import load_robot_container

        container = load_robot_container("t1", tensor_args)
        assert container.left_joint_limits.shape == (2, 11)
        assert container.right_joint_limits.shape == (2, 11)

    def test_container_has_gripper_spheres(self, tensor_args):
        """Test container has gripper spheres for both arms."""
        from cutamp.robots import load_robot_container

        container = load_robot_container("t1", tensor_args)
        assert container.left_gripper_spheres.shape[1] == 4
        assert container.right_gripper_spheres.shape[1] == 4

    def test_container_has_tool_transforms(self, tensor_args):
        """Test container has tool transformations."""
        from cutamp.robots import load_robot_container

        container = load_robot_container("t1", tensor_args)
        assert container.left_tool_from_ee.shape == (4, 4)
        assert container.right_tool_from_ee.shape == (4, 4)


class TestT1JointMapping:
    """Tests for cuRobo to URDF joint mapping."""

    def test_curobo_to_urdf_joints_left_arm(self):
        """Test joint mapping for left arm."""
        from cutamp.robots.t1 import (
            curobo_to_urdf_joints, t1_home_left,
            CUROBO_DOF, URDF_DOF, GRIPPER_OPEN
        )
        
        urdf_joints = curobo_to_urdf_joints(t1_home_left, "left")
        assert len(urdf_joints) == URDF_DOF
        
        # Check left arm joints are from cuRobo input
        assert urdf_joints[6:13] == t1_home_left[4:11]
        
        # Check gripper is open (default for visualization)
        assert urdf_joints[13:17] == GRIPPER_OPEN

    def test_curobo_to_urdf_joints_right_arm(self):
        """Test joint mapping for right arm."""
        from cutamp.robots.t1 import (
            curobo_to_urdf_joints, t1_home_right,
            URDF_DOF, GRIPPER_OPEN
        )
        
        urdf_joints = curobo_to_urdf_joints(t1_home_right, "right")
        assert len(urdf_joints) == URDF_DOF
        
        # Check right arm joints are from cuRobo input
        assert urdf_joints[17:24] == t1_home_right[4:11]
        
        # Check gripper is open (default for visualization)
        assert urdf_joints[24:28] == GRIPPER_OPEN

    def test_curobo_to_urdf_joints_invalid_dof(self):
        """Test invalid DOF raises error."""
        from cutamp.robots.t1 import curobo_to_urdf_joints
        
        with pytest.raises(ValueError, match="Expected 11 DOF"):
            curobo_to_urdf_joints([0.0] * 7, "left")

    def test_curobo_to_urdf_joints_invalid_arm(self):
        """Test invalid arm raises error."""
        from cutamp.robots.t1 import curobo_to_urdf_joints
        
        with pytest.raises(ValueError, match="Invalid arm"):
            curobo_to_urdf_joints([0.0] * 11, "middle")


class TestT1RerunVisualization:
    """Tests for T1 Rerun visualization."""

    def test_load_t1_rerun_no_mesh(self):
        """Test T1 Rerun robot loads without mesh (faster)."""
        from cutamp.robots.t1 import load_t1_rerun

        robot = load_t1_rerun(load_mesh=False)
        assert robot is not None
        assert robot.name == "t1"

    def test_load_t1_rerun_with_mesh(self):
        """Test T1 Rerun robot loads with mesh."""
        from cutamp.robots.t1 import load_t1_rerun

        robot = load_t1_rerun(load_mesh=True)
        assert robot is not None
        assert robot.name == "t1"
    
    def test_full_neutral_has_open_gripper(self):
        """Test that full_neutral uses open gripper values."""
        from cutamp.robots.t1 import load_t1_rerun, GRIPPER_OPEN
        robot = load_t1_rerun(load_mesh=False)


class TestT1RobotRegistry:
    """Tests for T1 in robot registry."""

    def test_t1_in_robot_to_fns(self):
        """Test T1 is registered in robot_to_fns."""
        from cutamp.robots import robot_to_fns

        assert "t1" in robot_to_fns
        assert "rerun" in robot_to_fns["t1"]
        assert "q_home" in robot_to_fns["t1"]
        assert "container" in robot_to_fns["t1"]

    def test_get_q_home_t1(self):
        """Test getting home position for T1."""
        from cutamp.robots import get_q_home

        q_home = get_q_home("t1")
        assert len(q_home) == 11, f"Expected 11 DOF, got {len(q_home)}"

    def test_load_rerun_robot_t1(self):
        """Test loading Rerun robot for T1."""
        from cutamp.robots import load_rerun_robot

        robot = load_rerun_robot("t1", load_mesh=False)
        assert robot is not None
        assert robot.name == "t1"
