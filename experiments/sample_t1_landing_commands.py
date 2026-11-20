"""Draw paired single- and two-step schedules using the bfm_wm free-command training sampler.

Run with bfm_wm's Python environment. Commands are support-relative (meters and radians).
Each L or R sample supplies the identical first command to its LR or RL counterpart.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sample(bfm_root, home, per_order, seed):
    sys.path.insert(0, str(bfm_root.resolve()))
    from src.landing import LandingCfg, compose, land_sample

    cfg = LandingCfg()
    rng = torch.Generator().manual_seed(seed)

    def draw(poses, swing):
        pending = torch.arange(len(poses))
        commands = torch.empty((len(poses), 3), dtype=poses.dtype)
        while len(pending):
            u = torch.rand((len(pending), 3 + 3 * cfg.candidates), dtype=poses.dtype, generator=rng)
            u[:, 0] = 0.75 if swing else 0.25
            _, c, _, ok = land_sample(cfg, poses[pending], u)
            commands[pending[ok]] = c[ok]
            pending = pending[~ok]
        return commands

    schedules = []
    for swing, order in enumerate(("LR", "RL")):
        poses = torch.tensor(home, dtype=torch.float64).repeat(per_order, 1, 1)
        first = draw(poses, swing)
        poses[:, swing] = compose(poses[:, 1 - swing], first)
        second = draw(poses, 1 - swing)
        for i, (c1, c2) in enumerate(zip(first.tolist(), second.tolist())):
            schedules.append({"order": order[0], "id": i, "commands": [c1]})
            schedules.append({"order": order, "id": i, "commands": [c1, c2]})
    return {
        "home_soles": home,
        "schedules": schedules,
        "source": "bfm_wm.src.landing.land_sample",
        "seed": seed,
        "per_order": per_order,
        "clock": "scorer chooses deployment clock at issue",
        "pairing": "L[i] and LR[i] share a first command; R[i] and RL[i] share a first command",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfm-root", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=Path("data/t1-reachability/t1-hand-map.npz"))
    parser.add_argument("--per-order", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_order <= 0:
        parser.error("--per-order must be positive")
    with np.load(args.map, allow_pickle=False) as atlas:
        home = json.loads(atlas["manifest"].item())["metadata"]["home_soles"]
    result = sample(args.bfm_root, home, args.per_order, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
