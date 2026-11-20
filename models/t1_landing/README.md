# T1 step model

The learned step model the planner uses by default (`--step_mode model`).
cuTAMP loads these files from `cutamp.landing_accuracy` (`LANDING_BUNDLE`,
`LANDING_ACCURACY`); `--landing_bundle` and `--step_accuracy_table` point it
at others.

| File | What it is |
| --- | --- |
| `landing-seed3.pt` | TorchScript archive: the frozen bfm_wm step policy and world model, sole forward kinematics and the deployed 50 Hz clock (10 history ticks). It runs on logical CUDA device 0 and needs no bfm_wm import or simulator. |
| `home-model.json` | The model's measured landing accuracy from the nominal home stance at actuator delays of 2, 5 and 8 physics substeps; task search reads it to rank step schedules. |

## Provenance

Both files come from bfm_wm, the repository that trains the T1 controller and
its world model. **bfm_wm is not public yet**, so neither file can be rebuilt
from this repository. They are frozen exports:

- checkpoint sha256 `766d602d8cabf1ccacfad7b59d7d09baac53ad17b59b9f6ead57e09739435f8b`
  (seed 3, step 4000), bfm_wm revision `76aba69c` plus the uncommitted step-command code;
- exported with PyTorch 2.9.0+cu128; the archive's `extra/landing.json` and the
  table's `provenance` record the checkpoint, revision and source hashes. The
  archive's debug records (source line maps holding local paths) were removed
  after export; its code, weights and metadata are unchanged;
- `cutamp/landing_accuracy.py` pins the checkpoint and source hashes. With a
  bundle exported from another checkpoint, the planner plans no steps and records
  why, except that a scene starting from the synthetic standing start (such as
  `two_steps_floor_transfer`) stops with the error. An accuracy table measured
  on another model is not used, and the run records why.

Earlier versions of `experiments/export_t1_landing.py` and
`experiments/score_t1_landing_accuracy.py` made these files (the hashes the files
record are of those versions); the scripts run with bfm_wm's Python and write
under `data/`, which is not tracked.

```bash
sha256sum -c <<'EOF'
61413e2a0c8ecbbd398803a9466d3aafea2c66fe8f2acf404726a1af8ad5f686  models/t1_landing/home-model.json
2f8002e1dcf16957a7f64d37aacb89ca10e3a579cfc1c1756d154381b90f6c47  models/t1_landing/landing-seed3.pt
EOF
```
