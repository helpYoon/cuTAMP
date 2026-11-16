# T2 physical execution and evaluation

These command-line tools execute and measure admitted T2 loco-manipulation references in MuJoCo. They do not replace the cuTAMP planner or automatically select footsteps. They are isolated examples; existing T1 planner/controller code is unchanged.

## Dependencies and inputs

Use Python 3.12 on Linux and `python -m pip install -r requirements.txt`. Physics was qualified with MuJoCo 3.8.1, NumPy 2.2.6, SciPy 1.15.3 and ProxSuite 0.8.2. Rendering additionally uses imageio/Pillow and a working EGL/OpenGL context. Set `MUJOCO_GL=egl` for headless rendering. The renderer is separate from execution.

The sibling `../stepping` directory supplies `contact_wbc.py` and `run_step.py` (`ContactWholeBodyQP` and `StepReference`). Pass its path explicitly with `--step-source`; there are no host-specific imports.

Small configurations included here:

- `configs/t2-robot.xml`: complete physical robot and gripper definition, with relative mesh paths. It is the qualified robot XML with asset paths sanitized; no physics values were changed.
- `configs/t2-assets-manifest.json`: required 53 STL basenames and SHA256 hashes. Mesh binaries are external model assets and are not included here. Supply the exact files, not arbitrary URDF mesh substitutes.
- `configs/initial-standing.json`: named robot-only root/joint seed. Object state is deliberately absent; scene settling owns it.
- `configs/farther-low-scene.json`: source cube(.95,.25,.18), 15cm pedestal, floor target region centered(.80,.25,0). The target region is noncolliding visualization; the actual floor supports the cube.
- `configs/grasp1.json`: frozen native grasp used in the demonstrated episode.
- `t2-actuator-map.json`: named 31 body-joint mapping. Gripper actuators are handled separately. Runtime indices are resolved by name.

`native_to_bundle.py` consumes the trusted planner's `motion_plan.pkl` and `overall_metrics.json`. Python pickle inputs must be obtained from your own trusted planning run. `known_payload.py` derives the known mass/hand-frame geometry from the planned grasp, without reading measured object contact forces.

## Build a fresh scene

Given the exact mesh files in `$T2_ASSETS`:

```sh
python settle_scene.py --scene configs/farther-low-scene.json \
  --robot configs/t2-robot.xml --assets "$T2_ASSETS" \
  --asset-inventory configs/t2-assets-manifest.json \
  --compile-mjb --output work/scene
```

The builder performs 2 seconds of object-only settling and combines those settled object poses with the robot. Asset relocation changes filenames, not geometry or physics. A newly compiled model can have a different binary SHA; generate fresh audit receipts against that model. Do not edit an existing admission's hashes to bypass the model checks.

## Native plans, references and admission

The sequence used for the 113.112 second demonstration is: five prescribed empty 15 cm steps; explicit double-support bridge; low pick/lift; bridge; native `RetractHolding` into front carry; one loaded right 15 cm step; bridge; proper initially-held floor place. Manipulation segments came from `wb_manifold_v2_projected_optimizer` inside cuTAMP. The sequence has no external trajectory-repair wrapper or fabricated native-success flag.

Use `python TOOL.py --help` for the exact exporter/generator options. `native_to_bundle.py` exports physical named-joint references and actual gripper events. Step references must explicitly keep the holding gripper closed. `make_bridge.py` generates a checked double-support handoff, never a state reset.

`qualify_reference_batch.py` accepts a JSON manifest with `model`, `audit`, optional `known_payload`, and `references:[{label,reference_npz}]`. It runs independent dense collision, anatomical posture and kinematic checks. The collision audit uses the actual MuJoCo collision geometry: some links are box proxies, so “CAD” in receipt names does not imply exact concave visual meshes everywhere.

`build_admission.py` accepts a JSON spec containing `model`, `scene`, `scope`, and ordered `stages`. Every stage names `reference_npz`, `label`, `kind`, `planner_generator_version`, `robot_qpos_indices`, `cad` and `comfort` receipts. Manipulation stages also require `native_gates`; steps require `kinematic_gates`; bridges require `bridge`. Include `required_closed_arm:0` for a left-hand carrying stage (`1` for right). Each receipt must actually pass. The builder checks robot and gripper boundary continuity, composes the reference, and binds all inputs/receipts by SHA256.

```sh
python build_admission.py --spec work/composition.json \
  --output-reference work/reference.npz --output-admission work/admission.json
python execute_admitted.py --cache-reference --step-source ../stepping \
  --model work/scene/model.mjb --scene work/scene/scene.json \
  --reference work/reference.npz --admission-report work/admission.json \
  --known-payload work/known-payload.json --audit t2-actuator-map.json \
  --output work/actual
```

Physics runs at 0.5 ms with 2 ms control. The robot is initialized once from the robot reference; the free object is initialized once from the independently settled scene. No object/robot poses are overwritten after initialization, and no object attachment weld is created. Actual closed-finger contacts hold the cube. Known payload compensation is active only while the corresponding gripper is commanded closed. The reference-packet cache preserves values and avoids repeated spline work; it does not slow the physics clock.

## Evaluate and render actual execution

```sh
python analyze_composed.py --run work/actual --scene work/scene/scene.json \
  --audit t2-actuator-map.json --support-body floor --target-region goal
python audit_body_posture.py --run work/actual --stride 1 --qualification-tolerance-deg 0
python audit_actuators.py --run work/actual --audit t2-actuator-map.json
python audit_floor_clearance.py --run work/actual --stride 1
MUJOCO_GL=egl python render_contact.py --run work/actual \
  --scene work/scene/scene.json --camera-azimuth 90 \
  --camera-lookat .65 .15 .55 --camera-distance 2.3 --goal-inset
```

The `../stepping` contact-replay audits additionally verify actual support wrenches and unexpected whole-body contacts. A video is an actual-state replay; it is not a kinematic preview.

Object success requires exactly one grasp/release lifecycle, retained relative TCP pose through the last release transition, physical support contact, complete rotated footprint containment, target height and settling checks. An early opening cannot truncate the hold metric and hide a drop. Post-release withdrawal is excluded from hold drift. Posture and contact audits are separate gates from object-task success; inspect all reports.

## Replay the frozen evidence bundle

The source repository intentionally contains no binary references, traces, videos, model binaries or checkpoints. If you have the separately supplied `comfortable-full-actual002` evidence archive:

```sh
python restore_evidence.py --archive "$EVIDENCE_ARCHIVE" --output work/restored
```

This verifies every member against `SHA256_MANIFEST.json`, maps the original host paths to the local archive, and builds a new local reference/admission from unchanged checked receipts. It never edits the original signed inputs and does not run physics. Use the printed model/scene/reference/admission paths with `execute_admitted.py`; the known payload is `composition/known-payload.json` inside the archive. This is the exact replay path. Regenerating plans from seeds is a separate experiment; optimizer nondeterminism can produce different valid trajectories.

The frozen model SHA256 is `56792be83ce98656db9c5521556ba0a787462d8e504b7e071e97e9c2fb17edd0`. The successful run used 113.112 seconds of physical time and 80.77 seconds of physics-loop wall time on its original CPU. This excludes reference/cache preparation and is not a planning-time or hard-real-time guarantee.

## Qualification and limits

The deterministic demonstrated episode achieved maximum active-foot error 2.766 mm, held-object drift 1.045 mm, 472.13 mm object lift, 100% final floor contact and stable placement. Minimum finger collision-geometry floor clearance was 30.81 mm. Full replay found zero unexpected robot/self contacts. All six swing feet had zero interior ground force, with 34.77–34.95 mm clearance. Minimum combined support CoP margin was 13.18 mm.

Footsteps are prescribed, not selected by the TAMP search. A finite same-scene search found 0/225 standing solutions, 3/225 after two steps, and 193/225 at the selected five-step stance under the declared posture policy. This is not global infeasibility or minimal-step evidence. The loaded step demonstrates capability and is not necessary for this floor target. No stochastic robustness or hardware validation is claimed. One body-actuator hit its limit for one control frame; joint-bound overshoot was 84 microradians. A conservative individual lightly loaded foot CoP bound grazed −18.46 micrometers during transitions; combined support stayed positive. The successful controller remains the baseline contact QP; an experimental COM-feedforward change was not needed.

Run the focused regression tests with `python -m unittest discover -p 'test_*.py'`. They cover admission/hash failures, silent gripper opening, grasp/release lifecycle, model-asset integrity and portable archive relocation.
