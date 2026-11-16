# T2 last-mile planner example

This preserves the native planning inputs for the physically tested five-empty-step → low pick → front carry → loaded step → floor-place experiment. Stepping and execution require the separate controller/example integration. These files alone generate manipulation references; they do not implement autonomous footstep selection.

## Setup

Use Linux x86_64, Python 3.10, CUDA-capable PyTorch and the repository's pinned dependency lock:

```sh
git submodule update --init --recursive
uv sync --locked
```

The integrated manifold optimizer additionally requires `scipy==1.15.3`; the branch includes its dependency lock. The validated Quadro RTX6000 environment used Python3.10.22, torch2.9.1+cu128, numpy2.2.6, scipy1.15.3, warp-lang1.16.0, cuda-core0.7.0 and cuda-bindings12.9.6. FP32 is used, with TF32 disabled. T2 URDFs and collision-sphere assets are included under `t2-port`; they do not require host-specific mesh paths. Native fresh grasp generation uses the existing GraspGen-X installation and separately downloaded weights; this example freezes a previously generated G1 grasp, so no weights or tokens are committed.

## Generate the final manipulation stages

Run from the repository root. Output folders are created separately from source.

```sh
export PYTHONPATH="$PWD:$PWD/curobo:$PWD/examples/t2_last_mile:$PWD/examples/t2_last_mile/connected:$PWD/examples/t2_last_mile/low-task"
export CUTAMP_WB_SEED_VERSION=2 T2_KNOTS=16 T2_POSE_WEIGHT=100000000
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export STUDY_FROZEN_GRASP="$PWD/examples/t2_last_mile/low-task/grasp1.json"
export CONNECTED_STANCE_JSON="$PWD/examples/t2_last_mile/low-task/comfortable-stance.json"
export LOW_TASK_PICK_HINT="$PWD/examples/t2_last_mile/low-task/comfortable-hint.json"
export LOW_TASK_MODEL_DIR="$PWD/outputs/t2-pick/model"
uv run python examples/t2_last_mile/low-task/comfortable_entry.py --env examples/t2_last_mile/low-task/comfortable-walk-scene.yml -n 64 --motion_plan --disable_visualizer --seed 102 --save_plan --experiment_root outputs --experiment_id t2-pick
```

For front carry, unset `LOW_TASK_PICK_HINT`; set `CONNECTED_STANCE_JSON`, `CONNECTED_HELD_JSON` and `--env` to the corresponding files in `low-task/comfortable-held-carry-state`, and `LOW_TASK_HELD_CARRY=1`. Use a new output/model folder. For floor placement, unset `LOW_TASK_HELD_CARRY`, and use `low-task/comfortable-S6-floor-state` instead. The held records retain the same grasp and exact planned post-step robot configuration. They are planning initialization records, not simulator state resets.

## What changed

- Persistent actual sole poses and exact two-sole support hulls.
- GPU projected whole-body trajectory optimization with explicit generator provenance, whole-path sole/task/COM/contact/anatomical constraints, C2 derivatives and analytic joint-bound checks.
- Native dynamics/ZMP/torque/velocity and Cartesian-line gates remain independent admission checks.
- Exact cuboid contact and orientation-aware placement helpers; phase-specific attached-object support-contact semantics preserve ordinary robot-floor collision.
- Genuine initial-Holding and motion-only continuation, front-carry targets and optional checked IK hints.

`WB-Manifold-2` is a custom cuTAMP trajectory generator, not unmodified cuRobo TrajOpt success. It uses GPU FK/projection and CPU SciPy spline/bound calculations. Historical `WB-Seed-1` is retained as experimental source; it is not the generator used by this final example. Exact physical CAD, comfort, model/reference hashes and contact-rich execution qualification are still required before treating a new reference as successful.

Five empty steps and the loaded right step are prescribed. Bounded same-scene initialization found S0=0/225, S2=3/225 and S5=193/225 accepted queries. This is not a proof of global unreachability or a minimal step count. The loaded step is optional for this floor target.

## Tests

```sh
uv run pytest -q cutamp/tests/test_anatomical_capture.py cutamp/tests/test_cubic_bounds.py cutamp/tests/test_intrinsic_stance.py cutamp/tests/test_plan_provenance.py
uv run pytest -q examples/t2_last_mile/low-task/test_motion_only_rollout.py
```

The source snapshot passed the first five tests on Robovision. CUDA/GPU availability is required for the motion-only integration tests.
