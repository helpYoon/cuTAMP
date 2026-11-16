"""Declared task-posture envelope on the exact exported physical reference.
These are research task constraints, not universal human/robot safety limits.
"""

import argparse, json
from pathlib import Path
import numpy as np, mujoco
from admission import sha256

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--output", required=True)
p.add_argument("--stride", type=int, default=1)
a = p.parse_args()
m = (
    mujoco.MjModel.from_binary_path(a.model)
    if a.model.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(a.model)
)
d = mujoco.MjData(m)
z = dict(np.load(a.reference))
d.qpos[:] = z["qpos"][0]
mujoco.mj_kinematics(m, d)
feet = [m.body(s + "_ankle_roll_link").id for s in ["left", "right"]]
f = sum(d.xmat[b].reshape(3, 3)[:, 0] for b in feet)
f[2] = 0
f /= np.linalg.norm(f)
left = np.cross([0, 0, 1], f)
rows = []
for i in range(0, len(z["qpos"]), a.stride):
    d.qpos[:] = z["qpos"][i]
    mujoco.mj_kinematics(m, d)
    u = d.xmat[m.body("trunk").id].reshape(3, 3)[:, 2]
    v = d.xmat[m.body("waist_yaw_link").id].reshape(3, 3)[:, 2]
    tp = np.degrees(np.arctan2(u @ f, u[2]))
    pp = np.degrees(np.arctan2(v @ f, v[2]))
    lean = np.degrees(np.arctan2(u @ left, u[2]))
    rows.append([z["time_s"][i], tp, pp, tp - pp, lean])
x = np.array(rows)
eps = 1e-4
checks = {
    "finite": bool(np.isfinite(x).all()),
    "torso_forward_pitch_minus5_to60deg": bool(
        x[:, 1].min() >= -5 - eps and x[:, 1].max() <= 60 + eps
    ),
    "torso_left_lean_12deg": bool(abs(x[:, 4]).max() <= 12 + eps),
    "hinge_minus3_to45deg": bool(
        x[:, 3].min() >= -3 - eps and x[:, 3].max() <= 45 + eps
    ),
}
result = {
    "passed": all(checks.values()),
    "model_sha256": sha256(a.model),
    "reference_sha256": sha256(a.reference),
    "scope": "Declared task comfort on exact exported reference; not physical execution or universal safety certification",
    "sample_stride": a.stride,
    "checks": checks,
    "torso_forward_pitch_min_max_deg": [float(x[:, 1].min()), float(x[:, 1].max())],
    "pelvis_forward_pitch_min_max_deg": [float(x[:, 2].min()), float(x[:, 2].max())],
    "forward_hinge_min_max_deg": [float(x[:, 3].min()), float(x[:, 3].max())],
    "torso_left_lean_abs_max_deg": float(abs(x[:, 4]).max()),
    "definition": "Positive forward pitch means link local +z leans toward initial actual foot +x heading; torso=trunk, pelvis=waist_yaw_link",
}
out = Path(a.output)
out.write_text(json.dumps(result, indent=2))
np.savez_compressed(
    out.with_suffix(".npz"),
    time_s=x[:, 0],
    torso_forward_pitch_deg=x[:, 1],
    pelvis_forward_pitch_deg=x[:, 2],
    forward_hinge_deg=x[:, 3],
    torso_left_lean_deg=x[:, 4],
)
print(json.dumps(result, indent=2))
