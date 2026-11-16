"""Signed actual-body posture relative to initial planted-foot heading.
Positive forward pitch means link local +z leans toward robot forward (+x).
Trunk and pelvis are measured separately; root is the upper trunk, not pelvis.
"""

import argparse, json
from pathlib import Path
import numpy as np, mujoco

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--output")
p.add_argument("--stride", type=int, default=5)
p.add_argument(
    "--qualification-tolerance-deg",
    type=float,
    help="Explicit tracking allowance around the declared reference comfort envelope",
)
a = p.parse_args()
r = Path(a.run)
z = dict(np.load(r / "trace.npz"))
model = (r / "model-path.txt").read_text().strip()
m = (
    mujoco.MjModel.from_binary_path(model)
    if model.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(model)
)
d = mujoco.MjData(m)
d.qpos[:] = z["qpos"][0]
mujoco.mj_kinematics(m, d)
feet = [m.body(s + "_ankle_roll_link").id for s in ["left", "right"]]
forward = sum(d.xmat[b].reshape(3, 3)[:, 0] for b in feet)
forward[2] = 0
forward /= np.linalg.norm(forward)
left = np.cross([0, 0, 1], forward)
bodies = ["trunk", "waist_yaw_link"]
jointnames = [
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_pitch_joint",
    "right_knee_pitch_joint",
]
jids = [int(m.joint(n).qposadr[0]) for n in jointnames]
rows = []
times = z["time"]
phases = z.get("phases", np.full(len(times), "episode"))
for i in range(0, len(times), a.stride):
    d.qpos[:] = z["qpos"][i]
    mujoco.mj_kinematics(m, d)
    row = {"time_s": float(times[i]), "phase": str(phases[i])}
    for label, name in [("torso", "trunk"), ("pelvis", "waist_yaw_link")]:
        b = m.body(name).id
        up = d.xmat[b].reshape(3, 3)[:, 2]
        row[label + "_forward_pitch_deg"] = float(
            np.degrees(np.arctan2(up @ forward, up[2]))
        )
        row[label + "_left_lean_deg"] = float(np.degrees(np.arctan2(up @ left, up[2])))
        row[label + "_height_m"] = float(d.xpos[b, 2])
    row["forward_hinge_deg"] = (
        row["torso_forward_pitch_deg"] - row["pelvis_forward_pitch_deg"]
    )
    row.update(
        {
            name + "_deg": float(np.degrees(d.qpos[j]))
            for name, j in zip(jointnames, jids)
        }
    )
    rows.append(row)
keys = [k for k in rows[0] if k not in ["time_s", "phase"]]
summary = {}
for phase in dict.fromkeys(row["phase"] for row in rows):
    rr = [row for row in rows if row["phase"] == phase]
    summary[phase] = {
        k: {
            "min": min(x[k] for x in rr),
            "max": max(x[k] for x in rr),
            "final": rr[-1][k],
        }
        for k in keys
    }
report = {
    "definition": "Relative to initial actual foot heading; positive pitch means local +z tilts forward. trunk is upper torso; waist_yaw_link is pelvis. Not Euler pitch and not a comfort-success assertion.",
    "sample_stride": a.stride,
    "initial_forward_world": forward.tolist(),
    "by_phase": summary,
}
if a.qualification_tolerance_deg is not None:
    tol = a.qualification_tolerance_deg
    if not 0 <= tol <= 1:
        raise ValueError("Tracking allowance must be declared within0..1degree")
    tp = np.array([x["torso_forward_pitch_deg"] for x in rows])
    lat = np.array([x["torso_left_lean_deg"] for x in rows])
    hinge = np.array([x["forward_hinge_deg"] for x in rows])
    checks = {
        "torso_forward_pitch": bool(tp.min() >= -5 - tol and tp.max() <= 60 + tol),
        "torso_lateral_lean": bool(abs(lat).max() <= 12 + tol),
        "forward_hinge": bool(hinge.min() >= -3 - tol and hinge.max() <= 45 + tol),
    }
    report.update(
        passed=all(checks.values()),
        checks=checks,
        actual_tracking_allowance_deg=tol,
        reference_envelope_deg={
            "torso_forward_pitch": [-5, 60],
            "torso_lateral_lean": [-12, 12],
            "forward_hinge": [-3, 45],
        },
    )
out = Path(a.output) if a.output else r / "body-posture-audit.json"
out.write_text(json.dumps(report, indent=2))
np.savez_compressed(
    out.with_suffix(".npz"), **{k: np.array([row[k] for row in rows]) for k in rows[0]}
)
print(json.dumps(report, indent=2))
