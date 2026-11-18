"""Build the offline T1 hand map and score a cuboid pick directly from it.

From the worktree, with the cuTAMP environment active:
    python -m cutamp.scripts.t1_reachability build
    python -m cutamp.scripts.t1_reachability score --plot

The score command queries the saved map on CPU. Plotting uses the visual URDF
to draw and independently verify the two stored fixed-foot configurations.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cutamp.reachability import ReachMap, canonical_stance, endpoint_orders, rank_orders, stance_key

DEFAULT_DIR = Path("data/t1-reachability")


def build_map(path, samples, seed, voxel_size, device):
    from cutamp.reachability_fk import T1MapSampler

    sampler = T1MapSampler(samples=samples, seed=seed, device=device)
    home = np.asarray(sampler.home_soles)
    orders = endpoint_orders(home)
    stances = {}
    for endpoints in orders.values():
        for soles in endpoints:
            stances.setdefault(stance_key(soles), canonical_stance(soles)[0])
    metadata = dict(
        sampler.metadata, home_soles=home.tolist(),
        order_endpoints={order: [soles.tolist() for soles in endpoints] for order, endpoints in orders.items()},
    )
    atlas = ReachMap(voxel_size, metadata=metadata)
    start = time.perf_counter()
    accepted = []
    for i, local in enumerate(stances.values(), 1):
        batches = sampler.map_stance(local)
        counts = {f"{row['arm']}:{row['posture']}": len(row["points"]) for row in batches}
        accepted.append({"local_soles": local.tolist(), "accepted_samples": counts})
        atlas.add_stance(local, batches)
        print(f"Stance {i}/{len(stances)}: {sum(counts.values())} accepted configurations", flush=True)
    for arm in ("left", "right"):
        available = {layer["posture"] for layer in atlas.layers(home, arm) if len(layer["ijk"])}
        if "standing" not in available or len(available) < 2:
            raise RuntimeError(f"The {arm} fixed-foot map needs standing and crouching samples")
    atlas.metadata["stance_samples"] = accepted
    atlas.metadata["build_seconds"] = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(path)
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def _world_xyz(points, frame):
    xyz = np.asarray(points).copy()
    cos, sin = np.cos(frame[2]), np.sin(frame[2])
    xyz[..., :2] = xyz[..., :2] @ np.array([[cos, sin], [-sin, cos]]) + frame[:2]
    return xyz


def render(atlas, result, output):
    from cutamp.scripts.plot_t1_reachability import plot_report
    from cutamp.scripts.plot_t1_witness_meshes import plot_witness_meshes

    # Ranking is already complete. CPU visual FK only draws/verifies saved witnesses.
    poses = plot_witness_meshes(result, output / "t1-fixed-feet-poses.png")
    home = np.asarray(result["home_soles"])
    _, frame = canonical_stance(home)
    rows = {mode: {r["order"]: r for r in result[mode]} for mode in ("standing", "crouching", "combined")}
    data = {key: result[key] for key in ("home_soles", "target_center", "target_size", "arm", "voxel_size")}
    data["frame_rows"] = []
    for order, union in rows["combined"].items():
        upright, bent = rows["standing"][order], rows["crouching"][order]
        data["frame_rows"].append({
            "order": order, "standing_m": upright["distance_m"], "crouching_m": bent["distance_m"],
            "combined_m": union["distance_m"], "rank": union["rank"], "rank_standing": upright["rank"],
            "rank_crouching": bent["rank"], "best_posture": union["posture"], "feet": union["soles"],
        })
    layers = atlas.layers(home, result["arm"])
    skeletons = {}
    for mode in ("standing", "crouching"):
        cells = [layer["ijk"] for layer in layers if (layer["posture"] == "standing") == (mode == "standing")]
        if not cells or rows[mode]["none"]["q"] is None:
            raise ValueError(f"Map has no fixed-foot {mode} witness for {result['arm']}")
        centers = (np.unique(np.concatenate(cells), axis=0) + 0.5) * atlas.voxel_size
        data[f"{mode}_points"] = _world_xyz(centers, frame)
        lines = {key: np.asarray(line) for key, line in poses[mode]["landmarks_world"].items()}
        shoulders = np.stack([lines["left_arm"][0], lines["right_arm"][0]])
        lines["shoulders"] = shoulders
        lines["torso"] = np.stack([shoulders.mean(axis=0), lines["spine"][0]])
        skeletons[mode] = [line.tolist() for line in lines.values()]
    data["skeletons"] = skeletons
    plot_report(data, output)


def score(path, center, size, arm, output, plot=False):
    atlas = ReachMap.load(path)
    home = np.asarray(atlas.metadata["home_soles"])
    # Reuse the exact sampled foot endpoints recorded when this map was built.
    orders = atlas.metadata.get("order_endpoints")
    if orders is None:
        orders = endpoint_orders(home)
    result = {
        "map": path.name, "arm": arm, "voxel_size": atlas.voxel_size, "target_center": list(center),
        "target_size": list(size), "home_soles": home.tolist(), "coordinates": "world meters; axis-aligned cuboid",
        "sampling": {key: atlas.metadata[key] for key in
                     ("samples_per_arm_posture", "seed", "posture_bend_rad", "physical_margin_m", "device")
                     if key in atlas.metadata},
    }
    result["query_seconds"] = {}
    for label, posture in (("standing", "standing"), ("crouching", "crouching"), ("combined", "all")):
        start = time.perf_counter()
        result[label] = rank_orders(atlas, home, center, size, arm=arm, posture=posture, orders=orders)
        result["query_seconds"][label] = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    (output / "ranking.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["combined"]:
        miss = row["distance_m"]
        distance = "uncovered" if miss is None else f"{100 * miss:.2f} cm"
        print(f"{row['rank']}. {row['order']:4s}  {distance}  {row['posture']}")
    if plot:
        render(atlas, result, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Sample T1 configurations offline")
    build.add_argument("--samples", type=int, default=65536, help="Samples per arm, stance and posture")
    build.add_argument("--seed", type=int, default=7)
    build.add_argument("--voxel-size", type=float, default=0.015, help="Voxel edge in meters")
    build.add_argument("--device", default="cuda", help="CUDA device for offline FK sampling")
    query = commands.add_parser("score", help="Rank no step, L, R, LR, RL for a world-frame pick box")
    query.add_argument("--arm", choices=("left", "right"), default="left")
    query.add_argument("--center", nargs=3, type=float, default=(0.30, 0.18, 0.22), metavar=("X", "Y", "Z"))
    query.add_argument("--size", nargs=3, type=float, default=(0.05, 0.05, 0.05), metavar=("DX", "DY", "DZ"))
    query.add_argument("--output", type=Path, default=DEFAULT_DIR)
    query.add_argument("--plot", action="store_true", help="Draw the stored witnesses and write PNG and HTML")
    for operation in (query, build):
        operation.add_argument("--map", type=Path, default=DEFAULT_DIR / "t1-hand-map.npz")
    args = parser.parse_args()
    if args.command == "build":
        build_map(args.map, args.samples, args.seed, args.voxel_size, args.device)
    else:
        score(args.map, args.center, args.size, args.arm, args.output, args.plot)


if __name__ == "__main__":
    main()
