"""Check actual recorded motor effort and physical joint limits."""

import argparse, json
from pathlib import Path
import numpy as np, mujoco

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--audit", required=True)
a = p.parse_args()
r = Path(a.run)
model_path = (r / "model-path.txt").read_text().strip()
m = (
    mujoco.MjModel.from_binary_path(model_path)
    if model_path.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(model_path)
)
z = dict(np.load(r / "trace.npz"))
rows = json.loads(Path(a.audit).read_text())["mapping"]
ids = [m.actuator(x["joint"]).id for x in rows]
bounds = np.maximum(
    abs(m.actuator_ctrlrange[ids, 0]), abs(m.actuator_ctrlrange[ids, 1])
)
assert np.all(bounds > 0)
ratio = abs(z["torques"]) / bounds
violation = 0.0
for x in rows:
    j = m.joint(x["joint"])
    q = z["qpos"][:, int(j.qposadr[0])]
    lo, hi = j.range
    violation = max(violation, float(max(lo - q.min(), q.max() - hi, 0)))
result = dict(
    peak_actuator_effort_fraction=float(ratio.max()),
    saturated_actuator_frames=int(np.any(ratio >= 1 - 1e-8, axis=1).sum()),
    maximum_joint_limit_violation_rad=violation,
    finite_actual_state_and_torque=bool(
        all(np.isfinite(z[key]).all() for key in ["qpos", "qvel", "torques"])
    ),
)
(r / "actuator-audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
