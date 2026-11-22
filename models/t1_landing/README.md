# T1 step models

The learned step models the planner uses.
cuTAMP loads these files from `cutamp.landing_accuracy` (`LANDING_BUNDLE`,
`LANDING_ACCURACY`) and `cutamp.step_sequence_model`
(`DEFAULT_STEP_SEQUENCE_MODEL`).

| File | What it is |
| --- | --- |
| `step-sequence-v1.pt` | TorchScript archive: the step-sequence model that learned step Adam differentiates through. From the issue state, the foot order and the foot commands, it predicts the body and both soles at six knots of each step (quarter, half and three quarters of the deadline, the deadline, mid hold and the end of the 75-tick hold). It was trained against `landing-seed3.pt`, whose checkpoint and deployment hashes its `extra/step_sequence.json` records. |
| `landing-seed3.pt` | TorchScript archive: the frozen bfm_wm step policy and world model, sole forward kinematics and the deployed 50 Hz clock (10 history ticks). It produces the post-retract history and checks the landings Adam chose. It runs on logical CUDA device 0 and needs no bfm_wm import or simulator. |
| `home-model.json` | The model's measured landing accuracy from the nominal home stance at actuator delays of 2, 5 and 8 physics substeps; task search reads it to rank step schedules. |
| `bfm-wm-notices.md` | bfm_wm's third-party notices: the MIT licenses of the BlinkMimic 2-derived code (bfm_wm's `src/`) compiled into both archives, and the terms of the motion data the controller was trained on. |

## Provenance

All three model files come from bfm_wm, the repository that trains the T1
controller and its world model. **bfm_wm is not public yet**, so none of them
can be rebuilt from this repository. They are frozen exports:

- checkpoint sha256 `766d602d8cabf1ccacfad7b59d7d09baac53ad17b59b9f6ead57e09739435f8b`
  (seed 3, step 4000), bfm_wm revision `76aba69c` plus the uncommitted step-command code;
- `landing-seed3.pt` was exported with PyTorch 2.9.0+cu128; the archive's
  `extra/landing.json` and the table's `provenance` record the checkpoint,
  revision and source hashes;
- `step-sequence-v1.pt` was trained by bfm_wm's step-sequence code, which is
  uncommitted there too (epoch 95; `extra/step_sequence.json` holds its schema,
  teacher and deployment hashes);
- the archives' debug records (source line maps holding local paths) were removed
  after export; their code, weights and metadata are unchanged;
- `cutamp/landing_accuracy.py` pins the checkpoint and source hashes, and
  `cutamp/step_sequence_model.py` refuses a step-sequence model trained against
  another teacher or deployment. With a bundle exported from another checkpoint,
  the planner plans no steps and records why, except that a scene starting from
  the synthetic standing start (such as `two_steps_floor_transfer`) stops with
  the error. An accuracy table measured on another model is not used, and the
  run records why.

Earlier versions of `experiments/export_t1_landing.py` and
`experiments/score_t1_landing_accuracy.py` made `landing-seed3.pt` and
`home-model.json` (the hashes the files record are of those versions); the
scripts run with bfm_wm's Python and write under `data/`, which is not tracked.

```bash
sha256sum -c <<'EOF'
61413e2a0c8ecbbd398803a9466d3aafea2c66fe8f2acf404726a1af8ad5f686  models/t1_landing/home-model.json
2f8002e1dcf16957a7f64d37aacb89ca10e3a579cfc1c1756d154381b90f6c47  models/t1_landing/landing-seed3.pt
e6558a4779e89c0ed68ae29a87a3668e2db95d214355741eb0d8e9d1526615fb  models/t1_landing/step-sequence-v1.pt
EOF
```
