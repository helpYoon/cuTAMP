## Robot Assets

T1 humanoid (the only supported robot) lives under
[`t1_description/`](t1_description/):

- `t1_simplified.urdf` — URDF used for planning + visualization.
- `actual_robot.urdf` — Booster T1 reference URDF (sim/real reference; not used at runtime).
- `t1_planar_base.yml` — cuRobo robot config (kinematics, cspace, lock_joints, extra_links for the planar base).
- `t1_spheres.yml` — collision spheres per link, in each link's local frame.
- `left_gripper_spheres.pt` / `right_gripper_spheres.pt` — serialized PyTorch tensors of gripper collision spheres
  (used by particle initialization for approximate collision checking).
- `meshes/`, `actual_meshes/` — visual + collision STLs.
