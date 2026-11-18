"""Sparse T1 hand reach map and geometric low-pick query. All distances in meters, yaw in radians."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

ORDERS = ("none", "L", "R", "LR", "RL")
# Mirror y and yaw for a right swing. A relative command's terminal foot shape is
# reusable wherever the support sole lies, which keeps the offline map finite.
COMMANDS = (
    (0.0, 0.21, 0.0),
    (0.12, 0.21, 0.0),
    (0.24, 0.21, 0.0),
    (0.32, 0.21, 0.0),
    (-0.12, 0.21, 0.0),
    (0.0, 0.30, 0.0),
    (0.18, 0.28, math.radians(15)),
)
_CORNER_SIGNS = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])


def _rot(yaw):
    """Rotation matrix for row-vector points."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])


def _wrap(yaw):
    return np.arctan2(np.sin(yaw), np.cos(yaw))


def _soles_array(soles):
    soles = np.asarray(soles, dtype=np.float64)
    if soles.shape != (2, 3) or not np.isfinite(soles).all():
        raise ValueError("soles must be a finite left/right pair of (x, y, yaw)")
    return soles


def canonical_stance(soles):
    """Floor poses in the stance frame and the world pose of that frame."""
    soles = _soles_array(soles)
    frame = soles.mean(axis=0)
    frame[2] = _wrap(soles[0, 2] + 0.5 * _wrap(soles[1, 2] - soles[0, 2]))
    local = soles.copy()
    local[:, :2] = (soles[:, :2] - frame[:2]) @ _rot(-frame[2])
    local[:, 2] = _wrap(soles[:, 2] - frame[2])
    return local, frame


def stance_key(soles):
    # Only absorb floating point roundoff under rigid transforms. Unbaked
    # geometries are left uncovered rather than substituting a different stance.
    local, _ = canonical_stance(soles)
    return tuple(np.round(local.ravel(), 7))


def apply_command(soles, side, command):
    """Apply one support-relative endpoint satisfying bfm_wm's landing envelope, or return None."""
    if side not in ("L", "R"):
        raise ValueError("side must be L or R")
    old = _soles_array(soles)
    idx = 0 if side == "L" else 1
    x, y, yaw = command
    sign = 1 if idx == 0 else -1
    if not (-0.33 <= x <= 0.33 and 0.14 <= sign * y <= 0.48 and math.radians(-20) <= sign * yaw <= math.radians(50)):
        return None
    # bfm_wm/src/landing.py::_separated, in the planted sole's frame.
    half = np.array([0.112434, 0.05])
    axes = np.vstack((np.eye(2), _rot(yaw)))
    project = np.abs(axes @ np.array([x, y]))
    extents = np.abs(axes) @ half + np.abs(axes @ _rot(-yaw)) @ half
    if np.max(project - extents) < 0.02:
        return None
    support = old[1 - idx]
    dest = np.r_[support[:2] + np.array([x, y]) @ _rot(support[2]), _wrap(support[2] + yaw)]
    travel = np.linalg.norm(dest[:2] - old[idx, :2]) + 0.1124 * abs(_wrap(dest[2] - old[idx, 2]))
    if not 0.05 <= travel <= 0.60:
        return None
    pair = old.copy()
    pair[idx] = dest
    return pair


def endpoint_orders(home_soles):
    """One empty order and all legal prefixes of length one or two from the mirrored command menu."""
    choices = {"none": [_soles_array(home_soles).copy()]}
    for order in ORDERS[1:]:
        side = order[-1]
        sign = 1 if side == "L" else -1
        pairs = []
        for previous in choices[order[:-1] or "none"]:
            for x, y, yaw in COMMANDS:
                dest = apply_command(previous, side, (x, sign * y, sign * yaw))
                if dest is not None:
                    pairs.append(dest)
        choices[order] = pairs
    return choices


class ReachMap:
    """One FK witness per occupied cell, arm, posture and relative sole pair."""

    def __init__(self, voxel_size, metadata=None):
        self.voxel_size = float(voxel_size)
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        self.metadata = dict(metadata or {})
        self._stances = {}

    def add_stance(self, soles, batches):
        """Insert FK batches expressed in the local stance frame."""
        key = stance_key(soles)
        if key in self._stances:
            raise ValueError("stance has already been added")
        layers = []
        seen = set()
        for batch in batches:
            arm, posture = batch["arm"], batch["posture"]
            points = np.asarray(batch["points"], dtype=np.float64)
            q = np.asarray(batch["q"])
            margin = np.asarray(batch["margin"]).reshape(-1)
            if points.shape != (len(points), 3) or q.ndim != 2 or not len(points) == len(q) == len(margin):
                raise ValueError("wrong FK batch dimensions")
            if not all(np.isfinite(arr).all() for arr in (points, q, margin)):
                raise ValueError("nonfinite FK sample")
            if (arm, posture) in seen:
                raise ValueError("duplicate arm/posture batch in this stance")
            seen.add((arm, posture))
            cells = np.floor(points / self.voxel_size).astype(np.int32)
            ijk, index = np.unique(cells, axis=0, return_index=True)
            layers.append(
                {
                    "arm": arm,
                    "posture": posture,
                    "ijk": ijk,
                    "points": points[index],
                    "q": q[index].astype(np.float32),
                    "margin": margin[index].astype(np.float32),
                }
            )
        self._stances[key] = layers

    def layers(self, soles, arm):
        return [layer for layer in self._stances.get(stance_key(soles), ()) if layer["arm"] == arm]

    def save(self, path):
        """Write compressed numeric arrays and JSON schema metadata without pickle."""
        info = {"version": 1, "voxel_size": self.voxel_size, "metadata": self.metadata, "stances": []}
        arrays = {}
        for i, (key, layers) in enumerate(self._stances.items()):
            desc = {"soles": np.asarray(key).reshape(2, 3).tolist(), "layers": []}
            for j, layer in enumerate(layers):
                desc["layers"].append([layer["arm"], layer["posture"]])
                for name in ("ijk", "points", "q", "margin"):
                    arrays[f"{i}_{j}_{name}"] = layer[name]
            info["stances"].append(desc)
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as out:
            tmp = Path(out.name)
            try:
                np.savez_compressed(out, manifest=json.dumps(info), **arrays)
                out.close()
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            info = json.loads(data["manifest"].item())
            if info["version"] != 1:
                raise ValueError("unsupported reach map format")
            atlas = cls(info["voxel_size"], info["metadata"])
            for i, desc in enumerate(info["stances"]):
                layers = []
                for j, (arm, posture) in enumerate(desc["layers"]):
                    arrays = {n: data[f"{i}_{j}_{n}"] for n in ("ijk", "points", "q", "margin")}
                    layers.append(dict(arm=arm, posture=posture, **arrays))
                atlas._stances[tuple(np.asarray(desc["soles"]).ravel())] = layers
        return atlas


def _box_gaps(relative, half, half_cell, yaw):
    """Euclidean distances between axis-aligned cubic cells and an upright, yawed box.

    relative is cell center minus box center, expressed in cell-frame axes.
    In 2D disjoint rectangles attain their minimum at a vertex and an edge.
    The separating-axis test also handles edge crossings where no vertex is inside.
    """
    turn = _rot(yaw)
    half_xy, half_z = half[:2], half[2]
    vertical = np.maximum(np.abs(relative[:, 2]) - half_cell - half_z, 0.0)
    xy = relative[:, :2]
    if abs(turn[0, 1]) < 1e-12 or abs(turn[0, 0]) < 1e-12:
        extent = half_xy @ np.abs(turn)
        planar = np.maximum(np.abs(xy) - half_cell - extent, 0.0)
        return np.sqrt(np.sum(planar**2, axis=1) + vertical**2)

    in_box = xy @ turn.T
    intersects = (np.abs(xy) <= half_cell + half_xy @ np.abs(turn)).all(axis=1) & (
        np.abs(in_box) <= half_xy + half_cell * np.abs(turn).sum(axis=1)
    ).all(axis=1)
    planar_sq = np.zeros(len(xy))
    if np.any(~intersects):
        outside = xy[~intersects]
        best = np.full(len(outside), np.inf)
        for sign in _CORNER_SIGNS:
            cell_vertex_in_box = (outside + half_cell * sign) @ turn.T
            gap = np.maximum(np.abs(cell_vertex_in_box) - half_xy, 0.0)
            best = np.minimum(best, np.sum(gap**2, axis=1))
            box_vertex_in_cell = (half_xy * sign) @ turn - outside
            gap = np.maximum(np.abs(box_vertex_in_cell) - half_cell, 0.0)
            best = np.minimum(best, np.sum(gap**2, axis=1))
        planar_sq[~intersects] = best
    return np.sqrt(planar_sq + vertical**2)


def _nearest_cell(layer, voxel_size, center, half, yaw):
    """Closest occupied voxel; use its representative FK sample to break voxel-distance ties."""
    centers = (layer["ijk"] + 0.5) * voxel_size
    relative = centers - center
    # The gap to the enclosing AABB is a lower bound. One exact distance gives
    # an upper bound, so most cells need no oriented-box calculation.
    bbox = np.r_[half[:2] @ np.abs(_rot(yaw)), half[2]]
    lower = np.maximum(np.abs(relative) - 0.5 * voxel_size - bbox, 0.0)
    lower_sq = np.einsum("ij,ij->i", lower, lower)
    seed = lower_sq.argmin()
    (upper,) = _box_gaps(relative[seed : seed + 1], half, 0.5 * voxel_size, yaw)
    candidate = np.flatnonzero(lower_sq <= (upper + 1e-12) ** 2)
    distances = _box_gaps(relative[candidate], half, 0.5 * voxel_size, yaw)
    distance = distances.min()
    tied = candidate[distances <= distance + 1e-13]
    samples = layer["points"][tied] - center
    samples[:, :2] = samples[:, :2] @ _rot(-yaw)
    point_gaps = np.linalg.norm(np.maximum(np.abs(samples) - half, 0.0), axis=1)
    best = np.lexsort((np.linalg.norm(samples, axis=1), point_gaps))[0]
    # Closed boxes still touch after rotating the complete scene, up to roundoff.
    return float(distance) if distance > 1e-12 else 0.0, float(point_gaps[best]), int(tied[best])


def _world_point(local_point, frame):
    return [*(local_point[:2] @ _rot(frame[2]) + frame[:2]).tolist(), float(local_point[2])]


def rank_orders(atlas, home_soles, target_center, target_size, arm="left", posture="all", target_yaw=0.0, orders=None):
    """Score exact orders by distance to a pick box given its center, full side lengths and world yaw.

    A zero distance means the volume of an occupied voxel overlaps the box. The
    actual FK witness may differ by up to one voxel diagonal; its position and
    distance are reported separately. A missing distance means no covered samples.
    """
    center = np.asarray(target_center, dtype=float)
    size = np.asarray(target_size, dtype=float)
    if center.shape != (3,) or size.shape != (3,) or not np.isfinite([*center, *size, target_yaw]).all():
        raise ValueError("provide finite target center, size and yaw")
    if np.any(size <= 0) or arm not in ("left", "right") or posture not in ("standing", "crouching", "all"):
        raise ValueError("invalid target size, arm or posture selection")
    if orders is None:
        orders = endpoint_orders(home_soles)
    results = []
    for order, endpoints in orders.items():
        steps = 0 if order == "none" else len(order)
        winner = None
        best = (np.inf, np.inf)
        covered = 0
        for endpoint_idx, soles in enumerate(endpoints):
            _, frame = canonical_stance(soles)
            local_center = np.r_[(center[:2] - frame[:2]) @ _rot(-frame[2]), center[2]]
            local_yaw = target_yaw - frame[2]
            layers = atlas.layers(soles, arm)
            supported = False
            for layer in layers:
                standing = layer["posture"] == "standing"
                if posture == "standing" and not standing or posture == "crouching" and standing:
                    continue
                if len(layer["ijk"]) == 0:
                    continue
                supported = True
                gap, point_gap, idx = _nearest_cell(layer, atlas.voxel_size, local_center, size / 2, local_yaw)
                if (gap, point_gap) >= best:
                    continue
                best = (gap, point_gap)
                world_cell = _world_point((layer["ijk"][idx] + 0.5) * atlas.voxel_size, frame)
                winner = {
                    "distance_m": gap,
                    "point_distance_m": point_gap,
                    "posture": layer["posture"],
                    "endpoint_index": endpoint_idx,
                    "soles": np.asarray(soles).tolist(),
                    "q": layer["q"][idx].tolist(),
                    "ijk": layer["ijk"][idx].tolist(),
                    "point_world": _world_point(layer["points"][idx], frame),
                    "cell_center_world": world_cell,
                    "margin_m": float(layer["margin"][idx]),
                }
            covered += int(supported)
        if winner is None:
            winner = dict.fromkeys(
                (
                    "distance_m",
                    "point_distance_m",
                    "posture",
                    "endpoint_index",
                    "soles",
                    "q",
                    "ijk",
                    "point_world",
                    "cell_center_world",
                    "margin_m",
                )
            )
        status = "uncovered" if not covered else ("hit" if winner["distance_m"] == 0.0 else "miss")
        results.append(
            dict(
                order=order,
                steps=steps,
                status=status,
                endpoint_count=len(endpoints),
                covered_count=covered,
                uncovered_count=len(endpoints) - covered,
                **winner,
            )
        )

    def priority(row):
        distance = row["distance_m"]
        return (
            row["status"] != "hit",
            row["steps"],
            math.inf if distance is None else round(distance, 12),
            ORDERS.index(row["order"]),
        )

    results.sort(key=priority)
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    return results
