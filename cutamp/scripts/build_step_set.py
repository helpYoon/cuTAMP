"""Builds the kinematic set of a step from the planner's model and writes it as the asset ``load_step_set`` reads.

python -B -m cutamp.scripts.build_step_set [--out PATH]
"""

import argparse
import time

from cutamp.step_set import DEFAULT_STEP_SET, build_step_set

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_STEP_SET))
    args = ap.parse_args()
    t0 = time.time()
    build_step_set(args.out)
    print(f"wrote {args.out} in {time.time() - t0:.0f} s")
