## Robot Assets

T1 humanoid (the only supported robot), on one planning model, the real legs:

- [`t1_29dof/`](t1_29dof/) — a pinned copy of bfm_wm's T1 and the planner model derived from it.
  - `urdf/T1_29dof.urdf`, `meshes/`, `t1_actuators.npz`, `LICENSE` — the vendor files, copied verbatim;
    `PROVENANCE.yml` gives each one's source and sha1.
  - `build_planner_model.py` — writes `t1_planner.urdf` (the vendor URDF plus a `world` link and cuTAMP's gripper),
    `t1_29dof.yml` (the cuRobo robot config, with a `cutamp:` block cuRobo never reads) and `t1_29dof_spheres.yml`
    from the pinned copy, `../t1_description/gripper.urdf` and `upper_body_spheres.yml`; a rerun reproduces all
    three byte for byte.
  - `upper_body_spheres.yml` — the hand-kept upper-body collision spheres (vendor frame), which the sphere editors
    edit.
  - `write_mujoco_fixture.py` — writes `mujoco_fixture.npz` (MuJoCo's FK, masses, inertias and soft limits on
    bfm_wm's `t1.xml`), which the tests check the planner model against.
- [`t1_description/`](t1_description/) — cuTAMP's gripper.
  - `gripper.urdf` — the gripper links and joints the builder appends to each hand, weightless.
  - `meshes/` — the gripper's 10 STLs.
  - `left_gripper_spheres.pt` / `right_gripper_spheres.pt` — serialized PyTorch tensors of gripper collision spheres
    (used by particle initialization for approximate collision checking).
  - `actual_meshes/` — 60 STLs of Booster's T1 that no URDF or code references.
- [`envelopes/`](envelopes/) — `route_a_start.yml`, route A's envelope (`cutamp/robots/envelope.py`).
