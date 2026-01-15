# cuTAMP with Booster T1

## Quick Links

The full README with installation instructions, examples, and detailed documentation can be found in [README_DETAILED.md](README_DETAILED.md).

## Modifications

📁 **[Robots Folder](cutamp/robots/)**
- **t1_description/** - T1 robot assets and configuration files:
  - `t1_simplified.urdf` - Simplified URDF model for T1 dual-arm humanoid robot
  - `t1_left_11dof.yml` / `t1_right_11dof.yml` - cuRobo configuration files for left/right arm (11 DOF each: 2 lift + 2 torso + 7 arm)
  - `t1_spheres.yml` - Collision sphere definitions for motion planning
  - `left_gripper_spheres.pt` / `right_gripper_spheres.pt` - Pre-computed gripper collision spheres for grasp planning
  - `meshes/` - STL mesh files for robot visualization and collision checking
- [`__init__.py`](cutamp/robots/__init__.py) - Robot container factory and registry. Provides `DualArmRobotContainer` dataclasses, and `load_robot_container()` function for T1 robot. Includes tool frame transformations (`tool_from_ee`) for top-down grasping.
- [`t1.py`](cutamp/robots/t1.py) - T1 dual-arm humanoid robot module. Provides cuRobo integration including kinematics models, IK solvers, gripper collision spheres, and joint mapping between cuRobo's 11-DOF model and URDF's 28-DOF representation. Supports Rerun visualization.

📁 **[Scripts Folder](cutamp/scripts/)**

Available scripts:
- [`gripper_sphere_editor.py`](cutamp/scripts/gripper_sphere_editor.py) - Interactive gripper sphere editor for grasp sampling.
- [`robot_sphere_editor.py`](cutamp/scripts/robot_sphere_editor.py) - Interactive robot sphere editor for curobo.
