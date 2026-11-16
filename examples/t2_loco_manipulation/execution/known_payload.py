"""Known payload parameters from planned grasp FK and the known scene model."""

import argparse, json, pickle, hashlib
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--plan-dir", required=True)
p.add_argument("--arm", choices=["left", "right"], required=True)
p.add_argument("--object", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
m = (
    mujoco.MjModel.from_binary_path(a.model)
    if a.model.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(a.model)
)
d = mujoco.MjData(m)
folder = Path(a.plan_dir)
raw = pickle.load(open(folder / "motion_plan.pkl", "rb"))
metrics = json.load(open(folder / "overall_metrics.json"))
matching = [
    s
    for s in raw["segments"]
    if s.get("gripper") == "close"
    and metrics["final_plan_skeleton"][s["op_idx"]].startswith(a.arm.capitalize())
]
assert matching, "No requested-arm planned close event"
q = np.asarray(matching[0]["position"][-1])
d.qpos[:3] = q[:3]
d.qpos[3:7] = Rotation.from_euler("YXZ", q[3:6]).as_quat()[[3, 0, 1, 2]]
for n, x in zip(raw["joint_names"][6:], q[6:]):
    d.qpos[m.joint(n).qposadr[0]] = x
mujoco.mj_forward(m, d)
hand = m.body(a.arm + "_gripper/base").id
obj = m.body(a.object).id
R = d.xmat[hand].reshape(3, 3)
H = np.eye(4)
H[:3, :3] = R.T @ d.xmat[obj].reshape(3, 3)
H[:3, 3] = R.T @ (d.xpos[obj] - d.xpos[hand])
com = R.T @ (d.xipos[obj] - d.xpos[hand])
mass = float(m.body_mass[obj])
rows = np.zeros((2, 4))
rows[0 if a.arm == "left" else 1] = np.r_[mass, mass * com]
r = dict(
    source="Known object inertia/geometry and planned grasp FK; no measured object state or force",
    arm=a.arm,
    object_name=a.object,
    local_com_m=com.tolist(),
    payload_rows=rows.tolist(),
    hand_from_object=H.tolist(),
    model_sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
    native_plan_sha256=hashlib.sha256(
        (folder / "motion_plan.pkl").read_bytes()
    ).hexdigest(),
)
Path(a.output).write_text(json.dumps(r, indent=2))
print(json.dumps(r))
