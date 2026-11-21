"""Build the offline T1 hand map.

From the repository root, with the cuTAMP environment active:
    python -m cutamp.scripts.t1_reachability build
    python -m cutamp.scripts.t1_reachability build --nominal-only

build writes the full map (every foot order's stances, which experiments/score_t1_landing_accuracy.py reads)
under data/t1-reachability/. With --nominal-only it writes only the home stance, the map step search reads
(shipped as t1-nominal-hand-map.npz).
"""

import argparse
import time
from pathlib import Path

import numpy as np

from cutamp.reachability import ReachMap, canonical_stance, endpoint_orders, stance_key

DEFAULT_DIR = Path("data/t1-reachability")


def build_map(path, samples, seed, voxel_size, device, *, nominal_only=False):
    from cutamp.reachability_fk import T1MapSampler

    sampler = T1MapSampler(samples=samples, seed=seed, device=device)
    home = np.asarray(sampler.home_soles)
    orders = endpoint_orders(home)
    stances = {}
    for endpoints in ([orders["none"]] if nominal_only else orders.values()):
        for soles in endpoints:
            stances.setdefault(stance_key(soles), canonical_stance(soles)[0])
    metadata = dict(
        sampler.metadata, home_soles=home.tolist(), nominal_only=nominal_only,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Sample T1 configurations offline")
    build.add_argument("--samples", type=int, default=65536, help="Samples per arm, stance and posture")
    build.add_argument("--seed", type=int, default=7)
    build.add_argument("--voxel-size", type=float, default=0.015, help="Voxel edge in meters")
    build.add_argument("--nominal-only", action="store_true", help="Build only the nominal sole pair for transported-map search")
    build.add_argument("--device", default="cuda", help="CUDA device for offline FK sampling")
    build.add_argument("--map", type=Path, help="Output NPZ (defaults to a separate t1-nominal-hand-map.npz with --nominal-only)")
    args = parser.parse_args()
    path = args.map or DEFAULT_DIR / ("t1-nominal-hand-map.npz" if args.nominal_only else "t1-hand-map.npz")
    build_map(path, args.samples, args.seed, args.voxel_size, args.device, nominal_only=args.nominal_only)


if __name__ == "__main__":
    main()
