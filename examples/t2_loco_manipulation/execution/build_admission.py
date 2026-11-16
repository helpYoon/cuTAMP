"""Compile checked stages and issue a hash-bound execution admission receipt.
The spec supplies model/scene paths and ordered stages with actual gate receipts.
Nothing is marked qualified unless all required receipts are present and passed.
"""

import argparse, json
from pathlib import Path
import numpy as np
from compose_reference import compose
from admission import sha256, verify

p = argparse.ArgumentParser()
p.add_argument("--spec", required=True)
p.add_argument("--output-reference", required=True)
p.add_argument("--output-admission", required=True)
a = p.parse_args()
spec_path = Path(a.spec).resolve()
spec = json.loads(spec_path.read_text())
model = Path(spec["model"]).resolve()
scene = Path(spec["scene"]).resolve()
output = Path(a.output_reference).resolve()
out = Path(a.output_admission).resolve()
stages = []
for row in spec["stages"]:
    stage = {
        "kind": row["kind"],
        "planner_timing_mode": row.get("planner_timing_mode", "not_recorded"),
        "reference_path": str(Path(row["reference_npz"]).resolve()),
        "reference_sha256": sha256(row["reference_npz"]),
        "model_sha256": sha256(model),
        "scene_path": str(Path(row.get("scene_path", scene)).resolve()),
        "scene_sha256": sha256(row.get("scene_path", scene)),
        "planner_generator_version": row["planner_generator_version"],
    }
    for gate in ["native_gates", "kinematic_gates", "bridge", "cad", "comfort"]:
        if gate not in row:
            continue
        rp = Path(row[gate]).resolve()
        result = json.loads(rp.read_text())
        accepted = result.get(
            "passed", result.get("qualified", result.get("motion_plan_success"))
        )
        if accepted is not True:
            raise ValueError("Unpassed gate " + gate + " for " + row["label"])
        stage[gate] = {
            "passed": True,
            "receipt": {"path": str(rp), "sha256": sha256(rp)},
        }
    stages.append(stage)
compiled = compose(spec["stages"])
np.savez_compressed(output, **compiled)
record = {
    "schema_version": "execution_admission_v1",
    "scope": spec.get("scope", ""),
    "qualified": True,
    "model_sha256": sha256(model),
    "scene_sha256": sha256(scene),
    "reference_sha256": sha256(output),
    "composition_spec": {"path": str(spec_path), "sha256": sha256(spec_path)},
    "compiler_sha256": sha256(__file__),
    "stages": stages,
}
candidate = out.with_name("." + out.name + ".candidate")
candidate.write_text(json.dumps(record, indent=2))
try:
    verify(candidate, model, output, scene)
except Exception:
    record["qualified"] = False
    candidate.write_text(json.dumps(record, indent=2))
    candidate.replace(out)
    raise
candidate.replace(out)
print(
    json.dumps(
        {
            "qualified": True,
            "stages": len(stages),
            "reference": str(output),
            "admission": str(out),
            "duration_s": float(compiled["time_s"][-1]),
        },
        indent=2,
    )
)
