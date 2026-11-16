"""Export native cuTAMP segments into the shared timed physical-FK reference schema."""

import argparse, pickle, json
from pathlib import Path
import numpy as np, mujoco
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument("--through-retract-holding", action="store_true")
p.add_argument("--through-pick-lift", action="store_true")
p.add_argument("--model", required=True)
p.add_argument("--plan-dir", required=True)
p.add_argument("--output", required=True)
p.add_argument("--start-segment", type=int, default=0)
p.add_argument("--end-segment", type=int)
p.add_argument("--tail-hold", type=float, default=1.0)
p.add_argument("--initial-gripper", type=float, nargs=2, default=[0.0, 0.0])
a = p.parse_args()
m = (
    mujoco.MjModel.from_binary_path(a.model)
    if a.model.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(a.model)
)
d = mujoco.MjData(m)
raw = pickle.load(open(Path(a.plan_dir) / "motion_plan.pkl", "rb"))
metrics = json.load(open(Path(a.plan_dir) / "overall_metrics.json"))
names = raw["joint_names"]
parts = []
events = []
clock = 0.0
grip = np.array(a.initial_gripper)
grips = []
if a.through_pick_lift:
    indices = [
        i
        for i, s in enumerate(raw["segments"])
        if "RetractHolding" in metrics["final_plan_skeleton"][int(s["op_idx"])]
    ]
    if indices:
        a.end_segment = min(indices)
    else:
        assert all(
            "Place" not in metrics["final_plan_skeleton"][int(s["op_idx"])]
            for s in raw["segments"]
        ), "Missing holding-retract delimiter before placement"
        a.end_segment = len(raw["segments"])
if a.through_retract_holding:
    indices = [
        i
        for i, s in enumerate(raw["segments"])
        if "RetractHolding" in metrics["final_plan_skeleton"][int(s["op_idx"])]
    ]
    assert indices, "Missing native holding-retract stage"
    a.end_segment = max(indices) + 1
segments = raw["segments"][a.start_segment : a.end_segment]
for seg in segments:
    q, v, acc = [np.array(seg[k]) for k in ["position", "velocity", "acceleration"]]
    t = np.arange(len(q)) * seg["dt"]
    parts.append(
        (clock, clock + t[-1], BPoly.from_derivatives(t, np.stack((q, v, acc), 1)))
    )
    clock += t[-1]
    if seg.get("gripper"):
        op = metrics["final_plan_skeleton"][int(seg["op_idx"])]
        side = 0 if op.startswith("Left") else 1
        grip[side] = 255 if seg["gripper"] == "close" else 0
        events.append((clock, grip.copy()))
        duration = 0.7 if seg["gripper"] == "close" else 0.4
        parts.append((clock, clock + duration, q[-1]))
        clock += duration
first = np.array(segments[0]["position"][0])
last = np.array(segments[-1]["position"][-1])
duration = clock + a.tail_hold
time = np.linspace(0, duration, int(np.ceil(duration / 0.01)) + 1)
qpos = []
feet = []
commands = []
for t in time:
    q = first if t <= 0 else last
    for lo, hi, fn in parts:
        if lo <= t < hi:
            q = fn(t - lo) if callable(fn) else fn
            break
    d.qpos[:] = m.qpos0
    d.qpos[:3] = q[:3]
    d.qpos[3:7] = Rotation.from_euler("YXZ", q[3:6]).as_quat()[[3, 0, 1, 2]]
    for n, x in zip(names[6:], q[6:]):
        d.qpos[m.joint(n).qposadr[0]] = x
    mujoco.mj_kinematics(m, d)
    poses = []
    for s in ["left", "right"]:
        b = m.body(s + "_ankle_roll_link").id
        R = d.xmat[b].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = d.xpos[b] + R @ np.array([0.042485, 0, -0.032506])
        poses.append(T)
    qpos.append(d.qpos.copy())
    feet.append(poses)
    g = np.array(a.initial_gripper)
    for et, eg in events:
        if t >= et:
            g = eg
    commands.append(g)
np.savez_compressed(
    a.output,
    time_s=time,
    qpos=qpos,
    feet=feet,
    contacts=np.ones((len(time), 2), bool),
    normal_caps=np.full((len(time), 2), 1000.0),
    gripper=commands,
    phases=np.full(len(time), "native_manipulation"),
    initial_qpos=qpos[0],
)
print(
    json.dumps(
        dict(
            duration_s=duration,
            segments=len(segments),
            events=[(t, g.tolist()) for t, g in events],
        )
    )
)
