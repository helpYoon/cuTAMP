# T2 prescribed contact transitions

This CPU package supplies the torque QP and reference interface used by the T2 example. It supports prescribed, quasi-static flat-ground foot transitions. It does not choose footsteps, establish general walking, or certify hardware safety. The planner extension is in `cutamp/`; complete scene construction, native-plan conversion, admission, continuous execution and object-success checks are in `../execution/`.

## Environment and model

Use a separate CPU environment from the planner's Python 3.10/CUDA environment:

```sh
uv venv --python 3.12 .venv-t2-controller
uv pip install --python .venv-t2-controller/bin/python -r examples/t2_loco_manipulation/stepping/requirements.txt
```

The qualified stack is CPython 3.12.3, MuJoCo 3.8.1, NumPy 2.2.6, SciPy 1.15.3 and ProxSuite 0.8.2. MJB files are version-specific; rebuild them with that MuJoCo version. External robot/gripper meshes are required: follow the asset-inventory and model-builder instructions in `../execution/README.md`. No mesh/checkpoint/model binaries or recorded rollouts are included here.

`configs/t2-body-map.json` fixes the 31-body-joint ordering. The first joint must be the T2 trunk's free root; free-root qpos starts at zero. The 31 body entries are **not** a contiguous qpos slice: finger joints interleave them. The physical sole reference is `[0.042485,0,-0.032506]` in each ankle-roll frame, with sole half-sizes `[0.10791,0.040886]` m. Those dimensions must match the supplied model; do not enlarge them to make a plan pass.

The qualified controller rate is 500 Hz (2 ms), with four 0.5 ms physics steps. Changing these rates requires requalification. The QP uses CPU ProxSuite; this is not a hard-real-time scheduling qualification.

## Reference generation

From this directory, using an externally built scene:

```sh
python make_step_reference.py \
  --model /path/to/scene.mjb \
  --initial ../execution/configs/initial-standing.json \
  --audit configs/t2-body-map.json \
  --terminal-joints-json configs/terminal-arm-home.json \
  --foot left --dx .15 --knee-bump .30 \
  --output /path/to/output/empty-step1.npz
```

For subsequent steps, pass the previous reference with `--initial-index -1`. `configs/prescribed-gaits.json` records the five-empty-step sequence and the loaded-step settings used in the example. The initial JSON sets named robot joints, including passive finger joints, while the scene owns object initial poses. The optional waist upper bound is an optimization restriction; it does not alter the physical XML.

For a loaded step, use the exact planned front-carry endpoint, its declared `--payload-json`, `--foot right --dx .15 --knee-bump .10 --support-bias=-.07`, and omit final arm-home targets. The support bias is measured along the **stationary support foot's** x axis; right swing means left support. The command is rejected outside the unchanged sole's 15 mm inset. The example's heelward point leaves 37.91 mm raw longitudinal margin, or 22.91 mm relative to that inset; this is reference geometry, not a measured dynamic guarantee.

The generator writes explicit gripper commands: declared held payloads remain closed. Missing commands must never silently open a carrying hand. Known payload mass/local COM comes from the planned grasp and object model, not measured object forces.

`ik_errors[:,0]` is maximum sole-position error in metres; `ik_errors[:,1]` is combined-COM target error in metres, **not rotation**. Generated references still require full geometry/anatomical/velocity/dynamics admission and actual physical qualification. Object qpos entries inside a reference are not instructions to reset an object during execution.

## Execution interface and audits

`run_step.StepReference` provides `packet(t)` and `feet_packet(t)` to `contact_wbc.ContactWholeBodyQP`. The QP retains floating-base dynamics, active-foot acceleration constraints, zero swing-foot wrench, friction/CoP cones, effort and joint-viability bounds. A selected contact mode and known payload are explicit inputs. Default controller failure dumps are disabled; set `controller.failure_dump` to an output path when diagnosing a rejection.

Use `../execution/execute_admitted.py` for an admitted continuous task. The standalone `run_step.py` CLI is only a component diagnostic: a genuine held-object fixture must contain the actual held object state at initialization. A planned carry robot pose alone does not physically grasp an object. After initialization, continuous execution must not overwrite body/object state or stitch separate rollouts into a physical-success claim.

`audit_contacts.py` replays saved actual motor/gripper commands and measures true sole-frame forces, CoP, friction and swing clearance. `audit_all_contacts.py` reports all robot-involving geometry contacts, separating intended finger/block contacts. The latter defaults to the example object name `block`. Neither audit reruns physics or substitutes reference poses for measured states. Small conservative individual-foot boundary grazes are reported separately from combined support margins.

## Validation

```sh
python -m pytest -q tests
```

These asset-free tests cover packet derivatives/contact switching, robot-only initialization preserving scene objects, and refusal of an incompatible body mapping. Release parity tests additionally compare the controller and packets against the frozen qualified implementation on external T2 model/reference inputs. See `RELEASE_VALIDATION.md` for measured results and portability limits. A passing unit test does not replace full physical task qualification.
