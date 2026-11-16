"""Kinematic admission for a prescribed contact-transition reference.
Does not certify dynamic step execution; forces/torques are evaluated in physics.
"""

import argparse, json
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation
from admission import sha256

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--audit", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
m = (
    mujoco.MjModel.from_binary_path(a.model)
    if a.model.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(a.model)
)
d = mujoco.MjData(m)
z = dict(np.load(a.reference))
names = [r["joint"] for r in json.loads(Path(a.audit).read_text())["mapping"]]
violation = 0.0
feet = [m.body(s + "_ankle_roll_link").id for s in ["left", "right"]]
sole = np.array([0.042485, 0, -0.032506])
maxpos = 0.0
maxrot = 0.0
for name in names:
    j = m.joint(name)
    v = z["qpos"][:, int(j.qposadr[0])]
    lo, hi = j.range
    violation = max(violation, float(max(lo - v.min(), v.max() - hi, 0.0)))
for i, q in enumerate(z["qpos"]):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    for k, b in enumerate(feet):
        if not z["contacts"][i, k]:
            continue
        R = d.xmat[b].reshape(3, 3)
        pos = d.xpos[b] + R @ sole
        maxpos = max(maxpos, float(np.linalg.norm(pos - z["feet"][i, k, :3, 3])))
        maxrot = max(
            maxrot,
            float(Rotation.from_matrix(R.T @ z["feet"][i, k, :3, :3]).magnitude()),
        )
checks = {
    "finite_reference": bool(
        all(np.isfinite(z[k]).all() for k in ["qpos", "feet", "normal_caps", "time_s"])
    ),
    "monotonic_time": bool(np.all(np.diff(z["time_s"]) > 0)),
    "joint_position_limits": violation <= 1e-6,
    "active_sole_position_5mm": maxpos <= 0.005,
    "active_sole_rotation_0p05rad": maxrot <= 0.05,
    "nonnegative_normal_caps": bool((z["normal_caps"] >= 0).all()),
    "support_contact_present": bool(np.all(np.any(z["contacts"], axis=1))),
}
result = dict(
    passed=all(checks.values()),
    checks=checks,
    max_joint_limit_violation_rad=violation,
    max_active_sole_position_error_m=maxpos,
    max_active_sole_rotation_error_rad=maxrot,
    model_sha256=sha256(a.model),
    reference_sha256=sha256(a.reference),
    scope="Dense prescribed-reference kinematics only; dynamic contact/torque qualification requires actual physical execution",
)
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
