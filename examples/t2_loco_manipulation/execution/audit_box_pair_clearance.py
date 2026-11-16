"""Exact box SAT separation at exported reference samples.
Positive maximum-axis gap is a conservative lower bound on Euclidean clearance.
Negative gap measures the minimum separating translation on SAT axes.
"""

import argparse, json
from pathlib import Path
import numpy as np, mujoco

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--geom1", required=True)
p.add_argument("--geom2", required=True)
p.add_argument("--output", required=True)
p.add_argument("--pitch-error", type=float, default=0.0)
p.add_argument("--roll-error", type=float, default=0.0)
a = p.parse_args()
m = (
    mujoco.MjModel.from_binary_path(a.model)
    if a.model.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(a.model)
)
d = mujoco.MjData(m)
ids = [m.geom(n).id for n in [a.geom1, a.geom2]]
assert all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in ids)
z = dict(np.load(a.reference))
gaps = []
worst_errors = []
side = "right" if a.geom1.startswith("right") else "left"
pitch_id = int(m.joint(side + "_ankle_pitch_joint").qposadr[0])
roll_id = int(m.joint(side + "_ankle_roll_joint").qposadr[0])
perturbations = [
    (dp, dr)
    for dp in ([-a.pitch_error, 0.0, a.pitch_error] if a.pitch_error else [0.0])
    for dr in ([-a.roll_error, 0.0, a.roll_error] if a.roll_error else [0.0])
]
for q in z["qpos"]:
    local = []
    for dp, dr in perturbations:
        d.qpos[:] = q
        d.qpos[pitch_id] += dp
        d.qpos[roll_id] += dr
        mujoco.mj_kinematics(m, d)
        A = d.geom_xmat[ids[0]].reshape(3, 3)
        B = d.geom_xmat[ids[1]].reshape(3, 3)
        axes = np.concatenate(
            [A.T, B.T, np.cross(A.T[:, None, :], B.T[None, :, :]).reshape(9, 3)]
        )
        axes = axes[np.linalg.norm(axes, axis=1) > 1e-8]
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        delta = d.geom_xpos[ids[1]] - d.geom_xpos[ids[0]]
        r1 = abs(axes @ A) @ m.geom_size[ids[0]]
        r2 = abs(axes @ B) @ m.geom_size[ids[1]]
        local.append(float((abs(axes @ delta) - r1 - r2).max()))
    index = int(np.argmin(local))
    gaps.append(local[index])
    worst_errors.append(perturbations[index])
gaps = np.array(gaps)
worst = int(gaps.argmin())
report = {
    "minimum_SAT_separation_m": float(gaps.min()),
    "time_at_minimum_s": float(z["time_s"][worst]),
    "penetration_frames": int((gaps < 0).sum()),
    "geom_pair": [a.geom1, a.geom2],
    "ankle_error_screen_rad": {"pitch": a.pitch_error, "roll": a.roll_error},
    "worst_perturbation_rad": worst_errors[worst],
    "scope": "Exact OBB separation at reference knots; positive gap lower-bounds Euclidean clearance, not dynamic tracking robustness",
}
Path(a.output).write_text(json.dumps(report, indent=2))
np.savez_compressed(
    Path(a.output).with_suffix(".npz"), time_s=z["time_s"], SAT_separation_m=gaps
)
print(json.dumps(report, indent=2))
