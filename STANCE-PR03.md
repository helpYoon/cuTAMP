# Persistent stance support — PR03

Base helpYoon/cuTAMP 890579f; isolated 5080 checkout /home/robocup/aditya/last-mile-stance, branch research/persistent-stance. Original repositories untouched.

## Integrated behavior

`StanceState(soles, contacts)` stores batched world sole transforms and contact flags. `StanceSchedule` binds whole-body configuration names to persistent stance IDs and rejects manipulation actions that silently change them.

`world.set_planning_stance(state, q_seed, sole_half)` binds one explicitly initialized double-support stance for a complete standing plan before optimizer/planner construction. It verifies flat/coplanar feet, full-body seed/sole FK agreement within .1mm, finite values and joint bounds. It updates initialization, arm-affinity home positions, sole targets, support halfspaces and reference home together. Both arms consume the same whole-body seed.

`solve_stance_seed(world, desired_soles, seed=None)` solves native sole Jacobian IK within joint limits to initialize staggered feet. Failure is explicit. This is a simulator initialization/reference operation, not a step execution.

Planning paths migrated: particle initialization/fallback home/retract references, balance-first GN, keyframe CoM costs/gates/margins, native trajopt CoM penalty, floor wrench in world frame, analytic ZMP slowdown inequalities, final trajectory sole/CoM/ZMP/torque/velocity checks. Exported segment metadata includes sole transforms, contact mode, full seed, support geometry and stance hash.

Generic support region is convex load-share support aL+(1-a)R through (1-a)L+aR (a=.1), with .02m safety inset and .005m extra keyframe reserve. Rotated/staggered feet use planar halfspaces, not an axis-aligned bounding rectangle. GN uses signed support distance in its hard hierarchy; its old rectangle centering/uprighting secondary objectives are disabled on this new path. This can change posture quality; native solver success does not establish physical success.

The original home-stance path remains unchanged unless set_planning_stance is called. No-schedule sole targets return the old tensor directly to preserve capture behavior.

## Validation completed on 5080

- Five tensor tests pass: nominal inside/outside, rigid invariance, staggered batching, gradients, persistent schedule/action guard.
- Two original real-leg model tests pass: sole-plane geometry and support rectangle.
- Three original native GN tests pass: sole finite-difference Jacobians, sole preservation for reachable and unreachable targets.
- Native explicit-home, translated/yawed, and coplanar staggered GN tests each pass all 4 rows, CoM/sole gates and CUDA graph replay.
- Actual T2 two-box native planning: legacy-home passes 16 segments; translated(.02,.01)/yaw(.08rad) passes 16 segments with 14/16 satisfying particles, ~6.02s optimization loop; staggered left+4cm/right-4cm x passes 16 segments,14/16 satisfying,~5.75s. These use n=16, seed0. Full pipeline timings are in timer_metrics.json; loop times are not complete wall clock.
- Nonhome pilot support sole half sizes are (.11,.05)m, which require matching to independently verified physical CAD before physical claims.

Evidence: integration-runs/*/{overall_metrics.json,timer_metrics.json,motion_plan.pkl}; native-stance-results.json; native-candidate-results.json; fixed-regression.log. Full native plans are planned references, not physically executed rollouts. Some selected postures have large trunk tilt and substantial ZMP retiming; this is not a performance improvement claim.

## Limits and next integration

One persistent stance per complete native planning solve is enabled. Batched state/schedule representation and per-configuration sole costs exist, but differing stance IDs inside one manipulation skeleton still require an explicit transition and are rejected by default. No Step operator, swing trajectory, one-step controller, or stance-gradient optimization is claimed. The deprecated experimental_keyframe_only argument does not bypass this guard.

Input validation synchronizes and belongs before graph capture. Runtime GN, costs and trajectory planning have been capture-tested. Single-support geometry exists but whole-plan initialization currently requires double support. Support halfspaces are a support/load model, not general friction/contact-wrench feasibility. Model-based torque and ZMP gates retain upstream approximations.
