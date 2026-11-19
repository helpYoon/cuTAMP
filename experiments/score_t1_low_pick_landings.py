"""From the cuTAMP worktree, rerun this file with the cuTAMP Python environment."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "data/t1-reachability/landing-accuracy"
sys.path.insert(0, str(ROOT))
from cutamp.landing_accuracy import handoff_precision
from cutamp.reachability import ORDERS, ReachMap, rank_orders


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(rows, lags, standing):
    hits = [r for r in rows if r["geometry"] == "hit"]
    result = {
        "landing_count": len(rows),
        "geometry_hit_count": len(hits),
        "geometry_hit_ids": [r["id"] for r in hits],
    }
    if standing:
        result["landing_accuracy"] = "exempt; no commanded step"
        return result
    any_ids = [r["id"] for r in hits if r["passing_lags"]]
    all_ids = [r["id"] for r in hits if len(r["passing_lags"]) == len(lags)]
    result.update(
        joint_any_count=len(any_ids), joint_any_ids=any_ids, joint_all_count=len(all_ids), joint_all_ids=all_ids
    )
    by_lag = {}
    for lag in lags:
        matching = [r["id"] for r in hits if lag in r["passing_lags"]]
        by_lag[str(lag)] = {"count": len(matching), "ids": matching}
    result["joint_by_lag"] = by_lag
    return result


def main():
    fixture_path = HERE.parent / "floor-forward" / "ranking.json"
    fixture = json.loads(fixture_path.read_text())
    model_path = HERE / "home-model.json"
    model = json.loads(model_path.read_text())
    map_path = HERE.parent / fixture["map"]
    map_sha = digest(map_path)
    if model["provenance"]["catalog_sha256"] != map_sha:
        raise ValueError("accuracy pilot and reach map must share the same catalogue")
    atlas = ReachMap.load(map_path)
    home = atlas.metadata["home_soles"]
    np.testing.assert_allclose(home, fixture["home_soles"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(home, model["home_soles"], atol=1e-7, rtol=0)
    lags = sorted(model["provenance"]["lags"])
    tested = {(s["order"], s["id"]): s for s in model["schedules"]}
    menu = atlas.metadata["order_endpoints"]
    expected = {(o, i) for o in ORDERS[1:] for i in range(len(menu[o]))}
    if set(tested) != expected or len(tested) != len(model["schedules"]):
        raise ValueError("accuracy pilot and reach map have different order/ID keys")
    orders = {}
    for order in ORDERS:
        rows = []
        for idx, soles in enumerate(menu[order]):
            (geom,) = rank_orders(
                atlas,
                home,
                fixture["target_center"],
                fixture["target_size"],
                arm=fixture["arm"],
                orders={order: [soles]},
                solid_pick=True,
            )
            row = {
                "id": idx,
                "geometry": geom["status"],
                "voxel_gap_m": geom["distance_m"],
                "sample_gap_m": geom["point_distance_m"],
            }
            if order != "none":
                trials = tested[order, idx]["trials"]
                if sorted(t["lag"] for t in trials) != lags:
                    raise ValueError(f"missing or duplicate lag trial at {order}/{idx}")
                passed = []
                for trial in trials:
                    stages = trial["stages"]
                    if "".join(s["foot"] for s in stages) != order:
                        raise ValueError(f"wrong sequence at {order}/{idx}")
                    np.testing.assert_allclose(stages[-1]["nominal_soles"], soles, atol=1e-7, rtol=0)
                    joint_ok = handoff_precision(stages, order)
                    if joint_ok is None:
                        raise ValueError(f"incomplete handoff measurements at {order}/{idx}")
                    if joint_ok:
                        passed.append(trial["lag"])
                row["passing_lags"] = sorted(passed)
            rows.append(row)
        orders[order] = {**summary(rows, lags, order == "none"), "rows": rows}
    output = {
        "task": {
            "fixture": "../floor-forward/ranking.json",
            "arm": fixture["arm"],
            "target_center_m": fixture["target_center"],
            "target_size_m": fixture["target_size"],
            "posture": "all",
            "solid_pick": True,
        },
        "model": "home-model.json",
        "model_file_sha256": digest(model_path),
        "map_sha256": map_sha,
        "lags_physics_substeps": lags,
        "joint_definition": "The same (order, id) has an occupied hand voxel touching the pick box, clear final sole footprints, legal commands and both soles within strict XY/yaw limits of the intended pair at issue for every handoff and the original nominal final pair.",
        "interpretation": "Counts are per catalogue endpoint and tested delay condition. Full grasp, achieved-stance reach and the next motion's handoff require refinement. Settling is diagnostic.",
        "orders": orders,
    }
    (HERE / "low-pick-joint.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for order, data in orders.items():
        print(order, data["geometry_hit_count"], data.get("joint_any_count"), data.get("joint_all_count"))


if __name__ == "__main__":
    main()
