"""Settle free objects against static environment, before planning or execution."""

import argparse, json, copy, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, mujoco

p = argparse.ArgumentParser()
p.add_argument("--scene", required=True)
p.add_argument("--robot", required=True)
p.add_argument("--output", required=True)
p.add_argument(
    "--assets", help="Directory containing mesh basenames; no geometry conversion"
)
p.add_argument("--asset-inventory", help="Verify exact mesh hashes before building")
p.add_argument("--compile-mjb", action="store_true")
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
spec = json.loads(Path(a.scene).read_text())
spec = spec.get("planner_environment", spec)
visual_regions = set(spec.get("physical_scene", {}).get("noncolliding_regions", []))
assert visual_regions <= set(spec["geometries"]["cuboid"]), "Unknown visual region"
assert not visual_regions.intersection(
    spec["types"]["Movable"]
), "A movable object cannot be made noncolliding"
# Settle object/environment alone. Robot configuration is initialized by the controller later.
root = ET.Element("mujoco")
ET.SubElement(root, "option", timestep=".0005", gravity="0 0 -9.81")
w = ET.SubElement(root, "worldbody")
for n, g in spec["geometries"]["cuboid"].items():
    pose = g["pose"]
    b = ET.SubElement(
        w,
        "body",
        name=n,
        pos=" ".join(map(str, pose[:3])),
        quat=" ".join(map(str, pose[3:])),
    )
    moving = n in spec["types"]["Movable"]
    if moving:
        ET.SubElement(b, "freejoint", name=n + "_free")
    attrs = dict(
        type="box",
        size=" ".join(str(x / 2) for x in g["dims"]),
        friction="1 .01 .001",
        condim="4",
        solref=".005 1",
        solimp=".99 .999 .001",
    )
    if moving:
        attrs["mass"] = str(g.get("mass_kg", 0.1))
    if n in visual_regions:
        attrs.update(contype="0", conaffinity="0")
    ET.SubElement(b, "geom", **attrs)
m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
d = mujoco.MjData(m)
for _ in range(4000):
    mujoco.mj_step(m, d)
assert (
    np.isfinite(d.qpos).all() and np.max(np.abs(d.qvel), initial=0) < 0.005
), "Object did not settle"
for n in spec["types"]["Movable"]:
    j = m.joint(n + "_free")
    i = int(j.qposadr[0])
    spec["geometries"]["cuboid"][n]["pose"] = d.qpos[i : i + 7].tolist()
receipt = dict(
    settled=True,
    procedure="2s object-only MuJoCo gravity/contact settling; robot absent",
    physics_dt=0.0005,
    max_speed=float(np.max(np.abs(d.qvel), initial=0)),
    physical_noncolliding_regions=sorted(visual_regions),
    planner_environment=spec,
)
(out / "settled-scene.json").write_text(json.dumps(receipt, indent=2))
spec["settled_scene_receipt"] = receipt["procedure"]
(out / "scene.json").write_text(json.dumps(spec, indent=2))
# Build execution model using exactly settled poses.
from model_assets import resolve_model_assets, verify_asset_inventory

if a.asset_inventory:
    if not a.assets:
        raise ValueError("--asset-inventory requires --assets")
    verify_asset_inventory(a.assets, json.loads(Path(a.asset_inventory).read_text()))
robot = resolve_model_assets(ET.parse(a.robot).getroot(), a.robot, a.assets)
rw = robot.find("worldbody")
for geom in list(rw.findall("geom")):
    if geom.get("name") == "floor" or geom.get("type") == "plane":
        rw.remove(geom)
for body in w.findall("body"):
    rw.append(copy.deepcopy(body))
for b in rw.findall("body"):
    if b.get("name") in spec["types"]["Movable"]:
        b.set(
            "pos",
            " ".join(map(str, spec["geometries"]["cuboid"][b.get("name")]["pose"][:3])),
        )
        b.set(
            "quat",
            " ".join(map(str, spec["geometries"]["cuboid"][b.get("name")]["pose"][3:])),
        )
ET.ElementTree(robot).write(out / "scene.xml", encoding="unicode")
compiled = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
if a.compile_mjb:
    mujoco.mj_saveModel(compiled, str(out / "model.mjb"), None)
