"""One continuous physical execution of a composed contact/manipulation reference."""

import argparse, json, sys, time
from pathlib import Path
import numpy as np, mujoco

p = argparse.ArgumentParser()
p.add_argument("--admission-report", required=True)
p.add_argument("--scene", required=True)
p.add_argument("--cache-reference", action="store_true")
p.add_argument("--known-payload")
p.add_argument("--step-source", required=True)
p.add_argument("--model", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--audit", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
sys.path.insert(0, a.step_source)
from admission import verify

admission_record = verify(a.admission_report, a.model, a.reference, a.scene)
from run_step import StepReference
from contact_wbc import ContactWholeBodyQP

out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
m = (
    mujoco.MjModel.from_binary_path(a.model)
    if Path(a.model).suffix == ".mjb"
    else mujoco.MjModel.from_xml_path(a.model)
)
m.opt.timestep = 0.0005
d = mujoco.MjData(m)
z = dict(np.load(a.reference))
d.qpos[:] = m.qpos0
d.qpos[:7] = z["initial_qpos"][:7]
audit = json.load(open(a.audit))
names = [x["joint"] for x in audit["mapping"]]
active = [
    (m.joint(n).id, int(m.joint(n).qposadr[0]), int(m.joint(n).dofadr[0]))
    for n in names
]
d.qpos[[x[1] for x in active]] = z["initial_qpos"][[x[1] for x in active]]
mujoco.mj_forward(m, d)
initial = d.qpos.copy()
acts = [m.actuator(n).id for n in names]
gripacts = [m.actuator(s + "_gripper/fingers_actuator").id for s in ["left", "right"]]
ref = StepReference(m, z, dict(active=active, actuators=acts))
if a.cache_reference:
    from cached_packets import CachedPackets

    ref = CachedPackets(ref)
ctrl = ContactWholeBodyQP(m, d, dict(active=active, actuators=acts), ref)
ctrl.friction_mu = 0.7
ctrl.friction_facets = 8
ctrl.foot_kp = 400.0
ctrl.foot_kd = 40.0
fault = None
rows = []
known_payload = (
    np.array(json.load(open(a.known_payload))["payload_rows"])
    if a.known_payload
    else np.zeros((2, 4))
)
started = time.monotonic()
from evaluation import manifest
import platform, shutil

sources = [
    __file__,
    a.admission_report,
    a.scene,
    a.reference,
    a.model,
    a.audit,
    str(Path(a.step_source) / "contact_wbc.py"),
    str(Path(a.step_source) / "run_step.py"),
]
if a.known_payload:
    sources.append(a.known_payload)
for helper_name in [
    "admission.py",
    "cached_packets.py",
    "evaluation.py",
    "object_lifecycle.py",
    "compose_reference.py",
]:
    helper_path = Path(__file__).resolve().parent / helper_name
    if helper_path.is_file():
        sources.append(str(helper_path))
receipt = dict(
    sources=manifest(sources),
    host=platform.node(),
    mujoco=mujoco.__version__,
    physics_dt_s=m.opt.timestep,
    control_dt_s=0.002,
    controller=dict(friction_mu=0.7, friction_facets=8, foot_kp=400.0, foot_kd=40.0),
    initial_state_source="robot reference plus settled scene object coordinates",
    state_writes_after_initialization=0,
    admission_verified=True,
    fall_check="actual trunk height below reference by15cm or absolute trunk height below18cm",
)
from importlib.metadata import version

receipt["versions"] = {
    name: version(name) for name in ["numpy", "scipy", "mujoco", "proxsuite"]
}
receipt["python"] = sys.version
(out / "provenance.json").write_text(json.dumps(receipt, indent=2))
shutil.copyfile(str(Path(a.step_source) / "contact_wbc.py"), out / "controller.py")
snapshot = out / "source_snapshot"
snapshot.mkdir()
for filename in sources:
    source_path = Path(filename)
    if source_path.suffix in [".py", ".json"]:
        shutil.copyfile(source_path, snapshot / source_path.name)

for k in range(round(ref.duration / 0.002)):
    t = k * 0.002
    i = max(
        0, min(np.searchsorted(z["time_s"], t, side="right") - 1, len(z["time_s"]) - 1)
    )
    g = z["gripper"][i] if "gripper" in z else np.zeros(2)
    ctrl.payload_rows = known_payload * (np.asarray(g) >= 128)[:, None]
    try:
        tau = ctrl.step(d, t, state_is_current=True)
    except Exception as e:
        fault = str(e)
        break
    d.ctrl[acts] = tau
    d.ctrl[gripacts] = g
    for _ in range(4):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    feet = ref.feet_packet(d.time)
    actual = np.array(
        [d.xpos[b] + d.xmat[b].reshape(3, 3) @ ctrl.sole for b in ctrl.feet]
    )
    force = np.zeros((2, 3))
    for ci, c in enumerate(d.contact):
        bs = m.geom_bodyid[np.array(c.geom)]
        if not any(m.body(int(b)).name == "floor" or int(b) == 0 for b in bs):
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, ci, f)
        world = c.frame.reshape(3, 3).T @ f[:3]
        for fi, fb in enumerate(ctrl.feet):
            if fb in bs:
                force[fi] += world if bs[1] == fb else -world
    rows.append(
        (
            d.qpos.copy(),
            d.qvel.copy(),
            tau.copy(),
            g.copy(),
            actual,
            feet[0][:, :3, 3],
            feet[3],
            force,
            str(z["phases"][i]),
            feet[4],
        )
    )
    if (
        not np.isfinite(d.qpos).all()
        or d.qpos[2] < z["qpos"][i, 2] - 0.15
        or d.qpos[2] < 0.18
    ):
        fault = "fall relative to planned crouch or nonfinite state"
        break
    if k % 1000 == 0:
        print(
            json.dumps(
                dict(
                    t=t,
                    phase=str(z["phases"][i]),
                    foot_error=np.linalg.norm(
                        actual - feet[0][:, :3, 3], axis=1
                    ).tolist(),
                )
            ),
            flush=True,
        )
if not rows:
    (out / "execution.json").write_text(
        json.dumps(dict(completed=False, fault=fault, sim_seconds=0.0))
    )
    raise SystemExit(1)
keys = [
    "qpos",
    "qvel",
    "torques",
    "gripper",
    "soles",
    "desired_soles",
    "contacts",
    "measured_foot_forces",
    "phases",
    "normal_caps",
]
trace = {key: np.array([row[i] for row in rows]) for i, key in enumerate(keys)}
trace["time"] = np.arange(1, len(rows) + 1) * 0.002
np.savez_compressed(out / "trace.npz", **trace)
(out / "model-path.txt").write_text(a.model)
errors = np.linalg.norm(trace["soles"] - trace["desired_soles"], axis=-1)
active_errors = errors[trace["contacts"]]
r = dict(
    completed=fault is None and len(rows) * 0.002 >= ref.duration - 0.003,
    fault=fault,
    sim_seconds=len(rows) * 0.002,
    wall_seconds=time.monotonic() - started,
    max_active_sole_error_m=float(active_errors.max()),
    actual_physics=True,
    continuous_mjdata=True,
    object_pose_writes_after_initialization=0,
    body_state_writes_after_initialization=0,
    admission_verified=True,
    fall_check="actual trunk height below reference by15cm or absolute trunk height below18cm",
)
(out / "execution.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r))
