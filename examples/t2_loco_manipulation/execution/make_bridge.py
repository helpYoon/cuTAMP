"""Explicit, audited double-support quintic handoff; never changes simulator state."""

import argparse, json
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import ConvexHull
from admission import sha256

p = argparse.ArgumentParser()
p.add_argument("--known-payload")
p.add_argument("--model", required=True)
p.add_argument("--before", required=True)
p.add_argument("--after", required=True)
p.add_argument("--audit", required=True)
p.add_argument("--output", required=True)
p.add_argument("--duration", type=float, default=1.0)
p.add_argument("--gripper", nargs=2, type=float, default=[0, 0])
a = p.parse_args()
m = (
    mujoco.MjModel.from_binary_path(a.model)
    if a.model.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(a.model)
)
d = mujoco.MjData(m)
q0 = np.load(a.before)["qpos"][-1]
q1 = np.load(a.after)["qpos"][0]
names = [x["joint"] for x in json.load(open(a.audit))["mapping"]]
ids = np.array([int(m.joint(n).qposadr[0]) for n in names])
feet_ids = [m.body(s + "_ankle_roll_link").id for s in ["left", "right"]]
r = Slerp([0, 1], Rotation.from_quat(np.array([q0, q1])[:, [4, 5, 6, 3]]))
t = np.linspace(0, a.duration, int(np.ceil(a.duration / 0.01)) + 1)
qs = []
feet = []
margins = []
bad = {}
corners = np.array(
    [[x, y, 0] for x in [-0.10791, 0.10791] for y in [-0.040886, 0.040886]]
)
payload = json.load(open(a.known_payload)) if a.known_payload else None
for ti in t:
    u = ti / a.duration
    b = 10 * u**3 - 15 * u**4 + 6 * u**5
    q = m.qpos0.copy()
    q[:3] = (1 - b) * q0[:3] + b * q1[:3]
    q[3:7] = r(b).as_quat()[[3, 0, 1, 2]]
    q[ids] = (1 - b) * q0[ids] + b * q1[ids]
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    if payload:
        bid = m.body(payload["arm"] + "_gripper/base").id
        H = np.array(payload["hand_from_object"])
        Rb = d.xmat[bid].reshape(3, 3)
        obj = m.body(payload["object_name"]).id
        adr = int(m.joint(payload["object_name"] + "_free").qposadr[0])
        d.qpos[adr : adr + 3] = d.xpos[bid] + Rb @ H[:3, 3]
        d.qpos[adr + 3 : adr + 7] = Rotation.from_matrix(Rb @ H[:3, :3]).as_quat()[
            [3, 0, 1, 2]
        ]
        mujoco.mj_forward(m, d)
    ff = []
    points = []
    for bid in feet_ids:
        R = d.xmat[bid].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = d.xpos[bid] + R @ np.array([0.042485, 0, -0.032506])
        ff.append(T)
        points.extend((T[:3, 3] + corners @ R.T)[:, :2])
    hull = ConvexHull(points)
    root = int(m.jnt_bodyid[0])
    com = d.subtree_com[root, :2].copy()
    if payload:
        com = (com * m.body_subtreemass[root] + d.xipos[obj, :2] * m.body_mass[obj]) / (
            m.body_subtreemass[root] + m.body_mass[obj]
        )
    margins.append(float(-(hull.equations[:, :2] @ com + hull.equations[:, 2]).max()))
    feet.append(ff)
    qs.append(q)
    for c in d.contact:
        if c.dist >= -0.002:
            continue
        bs = m.geom_bodyid[np.array(c.geom)]
        bn = [m.body(int(x)).name for x in bs]
        if (
            m.body_weldid[bs[0]] == m.body_weldid[bs[1]]
            or m.body_parentid[bs[0]] == bs[1]
            or m.body_parentid[bs[1]] == bs[0]
        ):
            continue
        if (
            payload
            and payload["object_name"] in bn
            and any("_gripper/" in name for name in bn)
        ):
            continue
        if "floor" in bn and any("ankle" in n for n in bn):
            continue
        if (
            all("_gripper/" in n for n in bn)
            and bn[0].split("_gripper/")[0] == bn[1].split("_gripper/")[0]
        ):
            continue
        bad[" / ".join(sorted(bn))] = max(
            bad.get(" / ".join(sorted(bn)), 0.0), float(-c.dist)
        )
feet = np.array(feet)
drift = float(np.linalg.norm(feet[:, :, :3, 3] - feet[:1, :, :3, 3], axis=-1).max())
receipt = dict(
    duration_s=a.duration,
    source="explicit quintic DS reference bridge",
    max_sole_motion_m=drift,
    min_com_support_margin_m=min(margins),
    unintended_penetrations=bad,
    qualified=drift <= 0.005 and min(margins) >= 0.005 and not bad,
)
Path(a.output + ".audit.json").write_text(json.dumps(receipt, indent=2))
assert receipt["qualified"], receipt
np.savez_compressed(
    a.output,
    time_s=t,
    qpos=qs,
    feet=feet,
    contacts=np.ones((len(t), 2), bool),
    normal_caps=np.full((len(t), 2), 1000.0),
    phases=np.full(len(t), "double_support_handoff"),
    gripper=np.tile(a.gripper, (len(t), 1)),
    initial_qpos=qs[0],
)
receipt.update(model_sha256=sha256(a.model), reference_sha256=sha256(a.output))
Path(a.output + ".audit.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
