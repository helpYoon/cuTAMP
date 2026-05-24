# Sim-to-Real Trajectory Mapping

This document enumerates every discrepancy between the trajectory cuTAMP saves
(planned against `t1_simplified.urdf`) and what an MPC tracking that
trajectory on the real robot (using `actual_robot.urdf`) needs to consume.
For each item: what it is, why it exists, the symptoms if ignored, and
exactly what to do.

> **Naming**: in this doc, "sim" = our planning URDF (`t1_simplified.urdf`),
> "real" = the on-robot URDF (`actual_robot.urdf`). They share the same
> upper-body chain (shoulders, elbows, wrists, hands, head, Trunk mesh,
> Waist mesh) and intentionally differ in the legs.

> **Status legend**: ✅ resolved · ⚠️ documented compensation required · 🛠 design accepted

---

## 1. Trunk link frame is offset 6.25 cm in +X between the two URDFs

**Status**: ✅ Resolved by Commit 3 (schema_version=2): plan_processor.py
applies -0.0625 X to saved trunk_xyz and emits hand poses in WORLD frame.
Consumer needs no compensation.

**What**: The `Waist_Yaw` joint's origin convention places the Trunk link
frame at a different physical point in each URDF.

**Real** (`actual_robot.urdf:1019-1034`):
```xml
<joint name="Waist" type="revolute">
  <origin xyz="0.0625 0 -0.1155"/>   <!-- parent=Trunk, child=Waist -->
```
**Sim** (`t1_simplified.urdf`):
```xml
<joint name="Waist_Yaw" type="revolute">
  <origin xyz="0 0 0.1155"/>          <!-- parent=Waist, child=Trunk -->
```

Sim's Trunk LINK FRAME sits **+6.25 cm in X** (forward) relative to where
real URDF places it.

**Why we don't patch the joint origin to match real**:

Because our chain is rooted at `world` (bottom-up) while real's is rooted
at `Trunk` (top-down), the Waist_Yaw joint has its parent/child reversed.
For a joint with a non-zero translational offset, parent/child reversal
changes **where the rotation axis lives in world**:

* Real: rotation anchor at `(0.0625, 0, -0.1155)` in Trunk's frame.
* Sim (if we tried to align Trunk frames): rotation anchor at the Trunk
  origin itself.

If we shift the sim's Waist_Yaw origin to `(-0.0625, 0, 0.1155)` to align
the Trunk frame with real's, then when `Waist_Yaw != 0` the upper body
swings around an axis at a **different physical location** than the real
robot would use. That off-axis rotation produces wrong hand positions
during torso twist motions — silently and much worse than the static
6.25 cm offset.

So we keep the joint origin at `(0, 0, 0.1155)` (rotation axis matches
real's behavior) and accept that the Trunk LINK FRAME is offset.

The Trunk link's `visual`/`collision`/`inertial` origins all carry a
matching `-0.0625` X compensation so the rendered geometry and mass
distribution land at the right world positions. Likewise, the Trunk
spheres in `t1_spheres.yml` have `-0.0625` X compensations baked in.

**What changed in schema v2**: plan_processor.py subtracts `0.0625` from
saved `trunk_xyz[:, 0]` so the value represents real-URDF's Trunk world
pose. Hand poses are now emitted in WORLD frame directly (FK target
switched from `*_base_link` to `*_hand_link`), so a "place hand 0.30 m in
front" target now refers to the real-URDF world frame with no consumer
math.

---

## 2. World frame doesn't exist on the real robot

**Status**: ✅ Resolved by Commit 3 (schema_version=2): plan_processor.py
applies -0.0625 X to saved trunk_xyz and emits hand poses in WORLD frame.
Consumer needs no compensation.

**What**: Sim planner uses `world` as root with virtual planar-base joints
locked at zero; mobile_base bottom sits at world z=0. The real URDF is
rooted at Trunk, no world frame natively. The robot estimates its world
pose from IMU + leg odometry.

**Fix**: Schema v2 emits hand poses in **WORLD frame** (real-URDF-native)
alongside the Trunk's own world pose (`trunk_xyz`, `trunk_quat_wxyz`).
The MPC anchors via its world estimator (IMU + leg odometry).

---

## 3. Trunk full world pose now exposed (xyz + quat)

**Status**: ✅ resolved

`position.trunk_xyz` (`[T, 3]`, world) and `position.trunk_quat_wxyz`
(`[T, 4]`, world) are now in the schema. `trunk_height` kept as a
convenience alias for `trunk_xyz[:, 2]`.

In schema v2 the saved `trunk_xyz` is **real-URDF-native** (the 6.25 cm
X offset from #1 is already subtracted in plan_processor.py).

---

## 4. Hand poses now in WORLD frame

**Status**: ✅ Resolved by Commit 3 (schema_version=2): plan_processor.py
applies -0.0625 X to saved trunk_xyz and emits hand poses in WORLD frame.
Consumer needs no compensation.

`right_hand_xyz`, `right_hand_quat_wxyz`, `left_hand_xyz`,
`left_hand_quat_wxyz` are in **WORLD frame** (real-URDF-native). Velocity
fields are likewise in WORLD frame.

FK target switched from `*_base_link` (sim-only) to `*_hand_link` (in
both URDFs), so the poses are directly consumable by an MPC tracking the
real T1.

---

## 5. Leg DOFs missing on sim side (3 per side: Hip_Roll, Hip_Yaw, Ankle_Roll)

**Status**: 🛠 design accepted (MPC handles)

Sim cspace per leg: 2 DOF (`ankle_pitch`, `knee_pitch`) plus the abstracted
`Torso_Pitch` (= both hip pitches). Real robot per leg: 6 DOF
(`Hip_Pitch`, `Hip_Roll`, `Hip_Yaw`, `Knee_Pitch`, `Ankle_Pitch`,
`Ankle_Roll`).

The MPC's balance controller is expected to choose values for the missing
DOFs around our commanded Trunk world pose.

---

## 6. `ankle_pitch` and `knee_pitch` not in saved data

**Status**: 🛠 design accepted (MPC solves leg IK from trunk_xyz/quat)

We intentionally do NOT save the leg joint values. The MPC is expected to
solve its own leg IK given the saved Trunk world pose, choosing
`Hip_Roll`/`Hip_Yaw`/`Ankle_Roll` for balance as part of that solve.

This is documented in the schema's `note_legs` field.

---

## 7. `Torso_Pitch` broadcast to both hip pitches

**Status**: ✅ documented in schema

`trunk_pitch` is sim's `Torso_Pitch` joint value (frame-independent). The
schema's `joint_name_groups.trunk_pitch` label spells out the broadcast:

```
"Torso_Pitch (broadcast to BOTH Left_Hip_Pitch + Right_Hip_Pitch on real)"
```

Same convention applies if we ever expose `ankle_pitch` and `knee_pitch`
(both real-side joints get the same sim value).

---

## 8. Quaternion convention is now explicit

**Status**: ✅ resolved

All quaternion fields renamed to `*_quat_wxyz`. Top-level schema field
`quaternion_convention` documents cuRobo's wxyz and gives the conversion
to xyzw:

```python
quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
```

---

## 9. Trunk *world* orientation ≠ `trunk_pitch` joint value

**Status**: ⚠️ both available, documented

`trunk_pitch` = `Torso_Pitch` joint value (frame-independent; correct
command for real's `Hip_Pitch` joints). The Trunk link's actual *world-
frame* orientation is the cumulative rotation through the leg chain, which
differs from the joint value when the legs are bent.

For consumers needing the Trunk's world orientation (e.g., IMU balance
targets), the schema exposes `trunk_quat_wxyz` (full world orientation,
extracted from FK).

---

## 10. Trunk height matches real robot

**Status**: ✅ resolved (clarification of earlier wording)

Earlier this doc said "Trunk height fixed" which was confusing — "fixed"
meant *repaired* (past-tense), not *static*. To be clear:

- The Trunk's z value VARIES along the trajectory (squat motions take it
  as low as ~0.40 m and back to ~0.67 m standing).
- The URDF was corrected so the standing-at-home Trunk world Z is
  **0.6735 m**, matching `actual_robot.urdf` to sub-mm. Previously it was
  0.7255 m (52 mm too tall) due to wrong leg segment lengths.

Specific URDF changes that remain in effect:
- `mobile_base_link` box height `0.05` → `0.042` m
- `ankle_pitch` joint origin z `0.05` → `0.042`
- `thigh_link` cylinder length `0.28` → `0.236` m
- `Torso_Pitch` joint origin z `0.28` → `0.236`
- Corresponding visual / collision / inertial / sphere centers shifted

---

## Severity summary

| # | Item | Status | Action required |
|---|---|---|---|
| 1 | Trunk frame +6.25 cm X offset | ✅ resolved (Commit 3) | None — applied in plan_processor.py |
| 2 | No world frame on real | ✅ resolved (Commit 3) | None — hand poses are world-frame, real-URDF-native |
| 3 | Trunk full world pose exposed | ✅ done | — |
| 4 | Hand poses in WORLD frame | ✅ resolved (Commit 3) | None — emitted in world directly |
| 5 | Missing 3 leg DOFs | 🛠 accepted | MPC chooses Hip_Roll / Hip_Yaw / Ankle_Roll |
| 6 | `ankle_pitch`/`knee_pitch` not saved | 🛠 accepted | MPC solves leg IK from Trunk world pose |
| 7 | `Torso_Pitch` broadcast | ✅ documented | — |
| 8 | Quaternion convention | ✅ documented | — |
| 9 | Trunk pitch joint vs world | ✅ both available | — |
| 10 | Trunk height | ✅ fixed | — |

---

## TL;DR for the MPC consumer (schema_version=2)

1. Load `motion_plan.pkl` (schema_version=2). Hand poses are world-frame,
   real-URDF-native — no compensation needed.
2. Broadcast `trunk_pitch` to both `Left_Hip_Pitch` and `Right_Hip_Pitch`.
3. Solve your own leg IK (Hip_Roll, Hip_Yaw, Ankle_Pitch, Ankle_Roll,
   Knee_Pitch) from the saved `trunk_xyz` + `trunk_quat_wxyz`.

---

## File pointers

- Sim URDF: `cutamp/robots/assets/t1_description/t1_simplified.urdf`
- Real URDF: `cutamp/robots/assets/t1_description/actual_robot.urdf`
- Save processor / schema docstring: `cutamp/utils/plan_processor.py`
- Save CLI flag: `cutamp/scripts/run_cutamp.py:--save_plan`
- Sphere geometry: `cutamp/robots/assets/t1_description/t1_spheres.yml`
