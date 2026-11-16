"""Physical object-task metrics from one continuous composed simulation."""

import argparse, json
from pathlib import Path
import numpy as np, mujoco

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--scene", required=True)
p.add_argument(
    "--audit", required=True, help="Named body-actuator map used for execution"
)
p.add_argument("--arm", choices=["left", "right"], default="left")
p.add_argument(
    "--support-body",
    default="goal",
    help="Actual physical support body; use floor for a floor placement region",
)
p.add_argument(
    "--target-region",
    default="goal",
    help="Scene cuboid defining placement region and target top height",
)
a = p.parse_args()
run = Path(a.run)
z = dict(np.load(run / "trace.npz"))
model_path = (run / "model-path.txt").read_text().strip()
m = (
    mujoco.MjModel.from_binary_path(model_path)
    if model_path.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(model_path)
)
d = mujoco.MjData(m)
scene = json.load(open(a.scene))
names = scene["types"]["Movable"]
assert len(names) == 1
name = names[0]
obj = m.body(name).id
goal = m.body(a.support_body).id
site = m.site(a.arm + "_grasp_tcp").id
side = 0 if a.arm == "left" else 1
acts = [m.actuator(row["joint"]).id for row in json.load(open(a.audit))["mapping"]]
gacts = [m.actuator(s + "_gripper/fingers_actuator").id for s in ["left", "right"]]
positions = []
relative = []
speeds = []
goal_contact = []
grasp_contact = []
for i, (q, v) in enumerate(zip(z["qpos"], z["qvel"])):
    d.qpos[:] = q
    d.qvel[:] = v
    d.ctrl[acts] = z["torques"][i]
    d.ctrl[gacts] = z["gripper"][i]
    mujoco.mj_kinematics(m, d)
    positions.append(d.xpos[obj].copy())
    relative.append(
        d.site_xmat[site].reshape(3, 3).T @ (d.xpos[obj] - d.site_xpos[site])
    )
    adr = int(m.joint(name + "_free").dofadr[0])
    speeds.append(np.linalg.norm(v[adr : adr + 3]))
    gc = 0
    hc = 0
    if i % 10 == 0 or z["time"][i] >= z["time"][-1] - 1.0:
        mujoco.mj_forward(m, d)
        contacts_now = d.contact
    else:
        contacts_now = []
    for c in contacts_now:
        bs = m.geom_bodyid[np.array(c.geom)]
        gc += set(bs) == {obj, goal}
        hc += obj in bs and any("_gripper/" in m.body(int(b)).name for b in bs)
    goal_contact.append(gc)
    grasp_contact.append(hc)
time = z["time"]
positions = np.array(positions)
relative = np.array(relative)
tail = time >= time[-1] - 1.0
g = scene["geometries"]["cuboid"][a.target_region]
dims = np.array(scene["geometries"]["cuboid"][name]["dims"])
from scipy.spatial.transform import Rotation

gR = Rotation.from_quat(np.array(g["pose"][3:])[[1, 2, 3, 0]]).as_matrix()
corners = (
    np.array([[x, y, v] for x in [-1, 1] for y in [-1, 1] for v in [-1, 1]]) * dims / 2
)
local = (
    positions[-1] + corners @ d.xmat[obj].reshape(3, 3).T - np.array(g["pose"][:3])
) @ gR
contained = bool(
    (np.max(abs(local[:, :2]), axis=0) <= np.array(g["dims"][:2]) / 2 + 0.003).all()
)
bottom = positions[-1, 2] - (abs(d.xmat[obj].reshape(3, 3)) @ (dims / 2))[2]
height_error = abs(bottom - (g["pose"][2] + g["dims"][2] / 2))
from object_lifecycle import hold_lifecycle

close, release, carry, drift = hold_lifecycle(time, z["gripper"][:, side], relative)
lift = float(positions[:, 2].max() - scene["geometries"]["cuboid"][name]["pose"][2])
contact_fraction = float(np.mean(np.array(goal_contact)[tail] > 0))
speed = float(np.max(np.array(speeds)[tail]))
settle_drift = float(np.ptp(positions[tail], axis=0).max())
execution = json.load(open(run / "execution.json"))
close_events = len(close)
open_events = len(release)
checks = dict(
    single_grasp_release_lifecycle=close_events == 1 and open_events == 1,
    completed=execution["completed"],
    active_soles_1cm=execution["max_active_sole_error_m"] <= 0.01,
    lift_3cm=lift >= 0.03,
    gripper_contact=any(grasp_contact),
    retained=drift is not None and drift <= 0.02,
    released=bool(len(release)),
    goal_footprint=contained,
    height_5mm=height_error <= 0.005,
    goal_contact_95pct=contact_fraction >= 0.95,
    settled_speed_2cm_s=speed <= 0.02,
    settled_drift_5mm=settle_drift <= 0.005,
)
checks = {k: bool(v) for k, v in checks.items()}
report = dict(
    gripper_lifecycle=dict(
        close_events=close_events,
        open_events=open_events,
        retention_interval="First close+0.7s through LAST release; intervening openings cannot hide a drop",
    ),
    actual_support_body=a.support_body,
    target_region=a.target_region,
    support_contact_fraction_final_second=contact_fraction,
    contact_sampling="Every10th frame plus every final-second frame; TCP/object retention allframes",
    qualified=all(checks.values()),
    qualification_checks=checks,
    objects={
        name: dict(
            max_lift_m=lift,
            object_relative_tcp_drift_max_m=drift,
            goal_footprint_contained=contained,
            settled_bottom_height_error_m=float(height_error),
            goal_contact_fraction_final_second=contact_fraction,
            maximum_speed_final_second_m_s=speed,
            position_range_final_second_m=np.ptp(positions[tail], axis=0).tolist(),
            final_position=positions[-1].tolist(),
        )
    },
    foot_max_drift_m=execution["max_active_sole_error_m"],
    continuous_mjdata=True,
    object_pose_writes_after_initialization=0,
    scope="Measured physical object state; one continuous simulation; references never set object pose",
)
(run / "summary.json").write_text(
    json.dumps(report, indent=2, default=lambda x: x.item())
)
print(json.dumps(report, indent=2, default=lambda x: x.item()))
