"""Dense exported-reference CAD admission including planned object lifecycle.
Kinematic attachment is used ONLY in this audit; execution retains a free object.
Actual finger motion/forces are qualified separately from recorded simulation.
"""

import argparse, json
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation
from admission import sha256

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--known-payload")
p.add_argument("--initial-held", action="store_true")
p.add_argument("--arm", choices=["left", "right"], default="left")
p.add_argument("--object", default="block")
p.add_argument("--output", required=True)
a = p.parse_args()
m = (
    mujoco.MjModel.from_binary_path(a.model)
    if a.model.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(a.model)
)
d = mujoco.MjData(m)
z = dict(np.load(a.reference))
oid = int(m.joint(a.object + "_free").qposadr[0])
obj = m.body(a.object).id
base = m.body(a.arm + "_gripper/base").id
H = (
    np.array(json.loads(Path(a.known_payload).read_text())["hand_from_object"])
    if a.known_payload
    else None
)
held = a.initial_held
O = np.eye(4)
O[:3, 3] = m.qpos0[oid : oid + 3]
O[:3, :3] = Rotation.from_quat(m.qpos0[oid + 3 : oid + 7][[1, 2, 3, 0]]).as_matrix()
bad = {}
floor_bad = {}
foot_bad = {}
foot_ids = {m.body(side + "_ankle_roll_link").id for side in ["left", "right"]}
maxpayload = 0.0
side = 0 if a.arm == "left" else 1
for i, q in enumerate(z["qpos"]):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    closed = bool(z["gripper"][i, side] >= 128) if "gripper" in z else held
    if closed:
        if H is None:
            raise ValueError(
                "Closing reference requires known planned hand_from_object"
            )
        B = np.eye(4)
        B[:3, :3] = d.xmat[base].reshape(3, 3)
        B[:3, 3] = d.xpos[base]
        O = B @ H
    held = closed
    d.qpos[oid : oid + 3] = O[:3, 3]
    d.qpos[oid + 3 : oid + 7] = Rotation.from_matrix(O[:3, :3]).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(m, d)
    for c in d.contact:
        b1, b2 = [int(m.geom_bodyid[int(g)]) for g in c.geom]
        bn = [m.body(b1).name, m.body(b2).name]
        depth = float(-c.dist)
        if (
            depth > 1e-5
            and (b1 in foot_ids or b2 in foot_ids)
            and "floor" not in bn
            and "goal" not in bn
        ):
            pair = " / ".join(sorted(bn))
            foot_bad[pair] = max(foot_bad.get(pair, 0.0), depth)
        if obj in [b1, b2]:
            other = bn[1] if b1 == obj else bn[0]
            if other.startswith(a.arm + "_gripper/"):
                continue
            maxpayload = max(maxpayload, depth)
            if depth > 1e-5:
                bad["payload / " + other] = max(
                    bad.get("payload / " + other, 0.0), depth
                )
        elif "floor" in bn and depth > 1e-5:
            other = bn[1] if bn[0] == "floor" else bn[0]
            if not other.endswith("_ankle_roll_link") and other != "goal":
                floor_bad[other] = max(floor_bad.get(other, 0.0), depth)
        elif depth > 0.002:
            if (
                m.body_weldid[b1] == m.body_weldid[b2]
                or m.body_parentid[b1] == b2
                or m.body_parentid[b2] == b1
            ):
                continue
            if (
                "_gripper/" in bn[0]
                and "_gripper/" in bn[1]
                and bn[0].split("_gripper/")[0] == bn[1].split("_gripper/")[0]
            ):
                continue
            key = " / ".join(sorted(bn))
            bad[key] = max(bad.get(key, 0.0), depth)
result = dict(
    passed=not bad and not floor_bad and not foot_bad,
    frames=len(z["qpos"]),
    model_sha256=sha256(a.model),
    reference_sha256=sha256(a.reference),
    unintended_penetrations_m=bad,
    unintended_robot_floor_penetrations_m=floor_bad,
    unintended_nonground_foot_penetrations_m=foot_bad,
    max_payload_nonhand_penetration_m=maxpayload,
    scope="Dense kinematic exported reference; physical jaw motion/contact forces/object retention require actual rollout",
    tolerances_m={
        "payload_nonhand": 1e-5,
        "nonfoot_robot_floor": 1e-5,
        "nonground_foot": 1e-5,
        "other_nonadjacent_robot": 0.002,
    },
)
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
