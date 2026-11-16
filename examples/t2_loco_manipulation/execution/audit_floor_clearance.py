"""Exact MuJoCo collision distance/contact audit against the actual floor.
Reads recorded physical finger joints, so opening/closing geometry is included.
This does not move robot/object poses in an execution; it replays recorded states.
"""

import argparse, json
from pathlib import Path
import numpy as np, mujoco

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--arm", choices=["left", "right"], default="left")
p.add_argument("--floor-body", default="floor")
p.add_argument("--stride", type=int, default=10)
p.add_argument("--output")
a = p.parse_args()
r = Path(a.run)
z = dict(np.load(r / "trace.npz"))
path = (r / "model-path.txt").read_text().strip()
m = (
    mujoco.MjModel.from_binary_path(path)
    if path.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(path)
)
d = mujoco.MjData(m)
floor = m.body(a.floor_body).id
floor_geoms = np.flatnonzero(m.geom_bodyid == floor)
hand_geoms = np.array(
    [
        i
        for i in range(m.ngeom)
        if m.body(int(m.geom_bodyid[i])).name.startswith(a.arm + "_gripper/")
        and (m.geom_contype[i] or m.geom_conaffinity[i])
    ]
)
assert len(floor_geoms) and len(hand_geoms)
local_vertices = {}
for g in hand_geoms:
    typ = m.geom_type[g]
    size = m.geom_size[g]
    if typ == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        adr = int(m.mesh_vertadr[mid])
        count = int(m.mesh_vertnum[mid])
        local_vertices[g] = np.array(m.mesh_vert[adr : adr + count])
    elif typ == mujoco.mjtGeom.mjGEOM_BOX:
        local_vertices[g] = np.array(
            [
                [x, y, v]
                for x in [-size[0], size[0]]
                for y in [-size[1], size[1]]
                for v in [-size[2], size[2]]
            ]
        )
    else:
        raise ValueError("Unsupported exact hand-envelope primitive: " + str(typ))
mindistance = np.inf
min_signed_height = np.inf
min_collision_distance = np.inf
min_time = None
hits = {}
rows = []
query = np.zeros(6)
for i in range(0, len(z["time"]), a.stride):
    d.qpos[:] = z["qpos"][i]
    d.qvel[:] = z["qvel"][i]
    mujoco.mj_forward(m, d)
    clearance = 0.25
    floor_top = max(
        float(d.geom_xpos[f, 2] + abs(d.geom_xmat[f].reshape(3, 3)[2]) @ m.geom_size[f])
        for f in floor_geoms
    )
    signed_height = min(
        float(
            (vertices @ d.geom_xmat[g].reshape(3, 3)[2] + d.geom_xpos[g, 2]).min()
            - floor_top
        )
        for g, vertices in local_vertices.items()
    )
    for h in hand_geoms:
        for f in floor_geoms:
            clearance = min(
                clearance,
                float(mujoco.mj_geomDistance(m, d, int(h), int(f), 0.25, query)),
            )
    min_collision_distance = min(min_collision_distance, clearance)
    min_signed_height = min(min_signed_height, signed_height)
    clearance = signed_height
    if clearance < mindistance:
        mindistance = clearance
        min_time = float(z["time"][i])
    for c in d.contact:
        b1, b2 = [int(m.geom_bodyid[int(g)]) for g in c.geom]
        if floor not in [b1, b2]:
            continue
        other = b2 if b1 == floor else b1
        name = m.body(other).name
        if name.endswith("_ankle_roll_link") or name in ["block", "goal"]:
            continue
        if c.dist < -0.00001:
            hits[name] = max(hits.get(name, 0.0), float(-c.dist))
    rows.append([z["time"][i], clearance])
result = dict(
    scope="Exact signed collision-mesh/box vertex height above horizontal floor at every sampled recorded physical state; auxiliary geomDistance may return zero without contact",
    sample_stride=a.stride,
    distance_query_cap_m=0.25,
    floor_body=a.floor_body,
    arm=a.arm,
    minimum_collision_distance_m=float(min_collision_distance),
    minimum_signed_gripper_height_above_floor_m=float(min_signed_height),
    minimum_gripper_floor_clearance_m=float(mindistance),
    minimum_clearance_time_s=min_time,
    unintended_robot_floor_penetrations_m=hits,
    passed=bool(mindistance >= -0.00001 and not hits),
)
out = Path(a.output) if a.output else r / "floor-clearance-audit.json"
out.write_text(json.dumps(result, indent=2))
np.savez_compressed(
    out.with_suffix(".npz"),
    time_s=np.array(rows)[:, 0],
    gripper_floor_clearance_m=np.array(rows)[:, 1],
)
print(json.dumps(result, indent=2))
