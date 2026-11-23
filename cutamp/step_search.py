"""Landing options for one pick or place: every menu step sequence from the current feet that lands legally and off
the pick object, with the hand reach map's distance to the target and the landing-accuracy evidence. The A* in
``cutamp.task_planning.search`` chooses among them.

World geometry uses meters and radians, with left/right sole poses in that order.
An edge's cost is (unresolved manipulation, number of steps).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from cutamp.landing_accuracy import check_profile, handoff_precision
from cutamp.reachability import (
    COMMANDS,
    ORDERS,
    ReachMap,
    _box_gaps,
    _box_gaps_rows,
    _rectangle_separation,
    _rot,
    _soles_array,
    _wrap,
    apply_command,
    canonical_stance,
)


def _tuples(rows):
    return tuple(tuple(float(v) for v in row) for row in rows)


def _local_bounds(geometry):
    """cuRobo mesh bounds use the local object frame and include its scale."""
    if geometry is None:
        return None
    dims = getattr(geometry, "dims", None)
    if dims is not None:
        half = np.asarray(dims, dtype=float) / 2
        bounds = np.array([-half, half])
    elif callable(getattr(geometry, "get_trimesh_mesh", None)):
        bounds = np.asarray(geometry.get_trimesh_mesh().bounds, dtype=float)
    else:
        return None
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        return None
    return bounds


@dataclass(frozen=True)
class TargetRegion:
    """Upright world box; unknown center or extent leaves the estimate unresolved."""

    arm: str
    center: tuple | None
    size: tuple | None
    yaw: float = 0.0
    solid_pick: bool = False
    loaded: bool = False
    coarse: bool = False

    @classmethod
    def pick(cls, arm, geometry, *, loaded=False):
        """Transform the object's local box once; a tilted box uses its world AABB."""
        bounds = _local_bounds(geometry)
        pose = getattr(geometry, "pose", None)
        if bounds is None or pose is None:
            return cls(arm, None, None, loaded=loaded)
        pose = np.asarray(pose, dtype=float)
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        center = rotation @ bounds.mean(axis=0) + pose[:3]
        size = bounds[1] - bounds[0]
        upright = np.allclose(rotation[2], (0, 0, 1), atol=1e-9, rtol=0)
        if upright:
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            size = np.abs(rotation) @ size
            yaw = 0.0
        solid = upright and getattr(geometry, "dims", None) is not None
        return cls(arm, center, size, yaw, solid_pick=solid, loaded=loaded, coarse=not solid)

    @classmethod
    def place(cls, arm, geometry, surface, *, loaded=True):
        """Extrude a flat destination patch's top by the upright object's local height."""
        bounds = _local_bounds(geometry)
        if bounds is None or surface is None:
            return cls(arm, None, None, loaded=loaded)
        patch = cls.pick(arm, surface, loaded=loaded)
        if patch.center is None or patch.coarse:
            # A mesh AABB or tilted plane does not establish a usable flat top.
            return cls(arm, None, None, loaded=loaded)
        height = bounds[1, 2] - bounds[0, 2]
        center = (*patch.center[:2], patch.center[2] + (patch.size[2] + height) / 2)
        return cls(
            arm,
            center,
            (*patch.size[:2], height),
            patch.yaw,
            loaded=loaded,
            coarse=getattr(geometry, "dims", None) is None,
        )

    def __post_init__(self):
        if self.arm not in ("left", "right") or not np.isfinite(self.yaw):
            raise ValueError("expected a left/right arm and finite box yaw")
        for name in ("center", "size"):
            value = getattr(self, name)
            if value is None:
                continue
            v = np.asarray(value, dtype=float)
            if v.shape != (3,) or not np.isfinite(v).all() or (name == "size" and np.any(v <= 0)):
                raise ValueError(f"invalid target {name}")
            object.__setattr__(self, name, tuple(v))


@dataclass(frozen=True)
class LandingEdge:
    order: str
    commands: tuple
    soles: tuple
    steps: tuple  # achieved in the *search geometry*, one nominal sole pair per command
    distance_m: float | None
    posture: str | None
    witness: tuple | None  # nominal, stance-local configuration
    accuracy: str  # "profiled" matches the nominal sole pair; actual state/history is assessed in refinement
    passes: int
    trials: int

    @property
    def cost(self):
        supported = self.accuracy == "no_step" or self.passes > 0
        return (int(self.distance_m != 0 or not supported), len(self.commands))


@dataclass(frozen=True)
class Schedule:
    edges: tuple[LandingEdge, ...]
    cost: tuple[int, int]


class _HandIndex:
    """Index occupied nominal voxel centers; run exact cell-box geometry on a bounded candidate set."""

    def __init__(self, layers, voxel_size):
        self.voxel = voxel_size
        # Distinct sole pairs can put the same target at exactly the same hand-map coordinates.
        self.nearest = lru_cache(maxsize=2048)(self.nearest)
        self._scaled = lru_cache()(self._scaled)
        self.layers = [layer for layer in layers if len(layer["ijk"])]
        if not self.layers:
            self.tree = None
            return
        cells = np.concatenate([layer["ijk"] for layer in self.layers])
        unique, self.source = np.unique(cells, axis=0, return_index=True)
        self.ends = np.cumsum([len(layer["ijk"]) for layer in self.layers])
        self.tree = cKDTree((unique + 0.5) * self.voxel)

    def nearest(self, center, half, yaw):
        if self.tree is None:
            return None, None, None
        half_cell = self.voxel / 2
        _, seed = self.tree.query(center)
        (upper,) = _box_gaps(self.tree.data[seed : seed + 1] - center, half, half_cell, yaw)
        # Triangular inequality about the two centers. Cells outside this ball
        # cannot improve on the initially found cell-to-box distance.
        cell, gap = int(seed), upper
        if upper > 1e-12:
            radius = np.linalg.norm(half) + np.sqrt(3) * half_cell + upper + 1e-12
            ids = self.tree.query_ball_point(center, radius, return_sorted=True)
            gaps = _box_gaps(self.tree.data[ids] - center, half, half_cell, yaw)
            i = int(np.argmin(gaps))
            cell, gap = ids[i], float(gaps[i])
        row = int(self.source[cell])
        layer_id = int(np.searchsorted(self.ends, row, side="right"))
        layer = self.layers[layer_id]
        row -= int(self.ends[layer_id - 1]) if layer_id else 0
        return (0.0 if gap <= 1e-12 else gap, layer["posture"], tuple(layer["q"][row].tolist()))

    def within(self, centers, half, yaws, bar):
        """Per row, whether some cell's gap to the box is <= bar + 1e-12: ``nearest``'s answer without its minimum."""
        half_cell, limit = self.voxel / 2, bar + 1e-12

        def passes(rows, cells):
            return _box_gaps_rows(self.tree.data[cells] - centers[rows], half, half_cell, yaws[rows]) <= limit

        _, seed = self.tree.query(centers)
        passed = passes(np.arange(len(centers)), seed)
        rest = np.flatnonzero(~passed)
        if len(rest):
            # The nearest cells to 9 box points at the centre's height: corners, edge midpoints and the centre.
            u, v = np.meshgrid((-1, 0, 1), (-1, 0, 1))
            u, v = u.ravel() * half[0], v.ravel() * half[1]
            c, s = np.cos(yaws[rest])[:, None], np.sin(yaws[rest])[:, None]
            points = np.repeat(centers[rest, None], 9, axis=1)
            points[..., 0] += u * c - v * s
            points[..., 1] += u * s + v * c
            _, near = self.tree.query(points.reshape(-1, 3))
            passed[rest] = passes(np.repeat(rest, 9), near).reshape(-1, 9).any(axis=1)
            rest = rest[~passed[rest]]
        if len(rest):
            # A cell within the limit has its centre this close to the upright box's centre in x, y and in z.
            r_xy = np.linalg.norm(half[:2]) + half_cell + limit
            r_z = half[2] + half_cell + limit
            ids = self._scaled(r_xy / r_z).query_ball_point(
                centers[rest] * (1, 1, r_xy / r_z), r_xy, p=np.inf, return_sorted=False
            )
            rows = np.repeat(rest, [len(i) for i in ids])
            if len(rows):
                passed[rows[passes(rows, np.concatenate(ids).astype(np.intp))]] = True
        return passed

    def _scaled(self, z_scale):
        return cKDTree(self.tree.data * (1, 1, z_scale))


class StepSearch:
    """One immutable map/menu/accuracy profile shared by all queries in a planning run."""

    def __init__(self, atlas, *, accuracy=None, lag=None, commands=COMMANDS):
        self.lag = lag
        meta = atlas.metadata
        if "home_soles" not in meta:
            raise ValueError("a transported map requires the nominal home_soles in its metadata")
        self.home = _tuples(meta["home_soles"])
        self.sole_half = np.asarray(meta.get("sole_box_m", (0.224868, 0.10)), dtype=float) / 2
        # Older T1 maps record only the planar footprint. Their URDF foot box is 43.66 mm tall.
        self.foot_height = float(meta.get("foot_height_m", 0.04366))
        self.hands = {arm: _HandIndex(atlas.layers(self.home, arm), atlas.voxel_size) for arm in ("left", "right")}
        if self.sole_half.shape != (2,) or not np.isfinite(self.sole_half).all() or (self.sole_half <= 0).any():
            raise ValueError("invalid sole footprint")
        template = _tuples(commands)
        if any(len(row) != 3 or not np.isfinite(row).all() for row in template):
            raise ValueError("a command requires finite x, y, yaw")
        signed = {"L": template, "R": tuple((x, -y, -yaw) for x, y, yaw in template)}
        self.menu = [
            (order, sequence)
            for order in ORDERS
            for sequence in (product(*(signed[side] for side in order)) if order != "none" else [()])
        ]
        self.records = {}
        self.profile_stance = None
        self.provenance = {}
        if accuracy is not None:
            self.provenance = dict(accuracy["provenance"])
            check_profile(self.provenance)
            self.profile_stance = _tuples(_soles_array(accuracy["home_soles"]))
            for schedule in accuracy["schedules"]:
                key = (schedule["order"], _tuples(schedule["commands"]))
                matched = [
                    handoff_precision(t.get("stages"), schedule["order"]) for t in schedule["trials"] if t["lag"] == lag
                ]
                hits, total = self.records.get(key, (0, 0))
                self.records[key] = hits + matched.count(True), total + sum(flag is not None for flag in matched)
        # Caches belong to this instance and expire with its immutable data/profile.
        self._cached_edges = lru_cache(maxsize=256)(self._edges)

    @classmethod
    def load(cls, map_path, *, accuracy_path=None, lag=None, commands=COMMANDS):
        atlas = ReachMap.load(map_path, nominal_only=True)
        accuracy = json.loads(Path(accuracy_path).read_text()) if accuracy_path is not None else None
        return cls(atlas, accuracy=accuracy, lag=lag, commands=commands)

    def _accuracy(self, order, commands, exact_start, loaded):
        if order == "none":
            return "no_step", 0, 0
        passes, trials = self.records.get((order, commands), (0, 0))
        if loaded or not trials:
            return "unknown", 0, 0
        return ("profiled" if exact_start else "transferred"), passes, trials

    def edges(self, incoming_soles, target):
        """Legal, geometrically clear endpoints with evidence specific to that command sequence."""
        return self._cached_edges(_tuples(_soles_array(incoming_soles)), target)

    def blocked_feet(self, soles, target):
        """Soles known to overlap a solid pick object within the physical sole height band."""
        if not target.solid_pick or target.center is None or target.size is None:
            return ()
        half = np.asarray(target.size) / 2
        if target.center[2] - half[2] >= self.foot_height or target.center[2] + half[2] <= 0:
            return ()
        poses = np.asarray(soles)
        xy = (poses[:, :2] - target.center[:2]) @ _rot(-target.yaw)
        return tuple(
            side
            for side, offset, yaw in zip("LR", xy, poses[:, 2])
            if _rectangle_separation(offset, half[:2], self.sole_half, yaw - target.yaw) < -1e-12
        )

    def reaches(self, soles, target, bar):
        """Per [n, 2, 3] sole pair, whether ``_edges``' hand-map gap to the target is <= bar + 1e-12.

        The target needs a centre, a size and a non-empty map for its arm.
        """
        soles = np.asarray(soles, dtype=np.float64)
        # canonical_stance's frame, one row per stance.
        xy = soles[:, :, :2].mean(axis=1)
        yaw = _wrap(soles[:, 0, 2] + 0.5 * _wrap(soles[:, 1, 2] - soles[:, 0, 2]))
        center = np.asarray(target.center)
        local = np.empty((len(soles), 3))
        local[:, :2] = np.matmul((center[:2] - xy)[:, None], np.moveaxis(_rot(-yaw), -1, 0))[:, 0]
        local[:, 2] = center[2]
        return self.hands[target.arm].within(local, np.asarray(target.size) / 2, target.yaw - yaw, bar)

    def _edges(self, incoming, target):
        known = target.center is not None and target.size is not None
        if known:
            center, half = np.asarray(target.center), tuple(np.asarray(target.size) / 2)
        arm_map = self.hands[target.arm]
        exact_start = self.profile_stance is not None and np.allclose(incoming, self.profile_stance, atol=1e-7, rtol=0)
        candidates = []
        # Share composed prefixes, including between a one-step and its two-step alternatives.
        prefixes = {("none", ()): ()}
        for order, commands in self.menu:
            steps = ()
            if commands:
                prefix = (order[:-1] or "none", commands[:-1])
                if prefix not in prefixes:
                    continue
                previous = prefixes[prefix]
                soles = apply_command(previous[-1] if previous else incoming, order[-1], commands[-1])
                if soles is None:
                    continue
                steps = (*previous, _tuples(soles))
                prefixes[order, commands] = steps
            feet = steps[-1] if steps else incoming
            if self.blocked_feet(feet, target):
                continue
            gap, posture, q = None, None, None
            if known:
                _, frame = canonical_stance(feet)
                local = (*((center[:2] - frame[:2]) @ _rot(-frame[2])), center[2])
                gap, posture, q = arm_map.nearest(local, half, target.yaw - frame[2])
            evidence = self._accuracy(order, commands, exact_start, target.loaded)
            candidates.append(LandingEdge(order, commands, feet, steps, gap, posture, q, *evidence))
        return tuple(candidates)
