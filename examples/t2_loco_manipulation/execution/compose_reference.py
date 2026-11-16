"""Compose prescribed reference stages without resetting the physical scene.

Input JSON: {stages:[{reference_npz:path,label:str}], initial_qpos_npz:path}.
Every stage follows StepReference's schema. Free-object coordinates after the
first initialization are reference-only and never supplied to the simulator.
"""

import argparse, json
from pathlib import Path
import numpy as np


def compose(stages):
    output = {
        k: []
        for k in [
            "time_s",
            "qpos",
            "feet",
            "contacts",
            "normal_caps",
            "phases",
            "gripper",
        ]
    }
    clock = 0.0
    previous = None
    previous_gripper = None
    for i, stage in enumerate(stages):
        z = np.load(stage["reference_npz"])
        t = z["time_s"]
        q = z["qpos"]
        assert len(t) == len(q) and np.all(np.diff(t) > 0)
        if previous is not None:
            # Full qpos includes freely simulated objects; check robot-only continuity.
            ids = np.array(stage["robot_qpos_indices"])
            jump = np.max(abs(q[0, ids] - previous[ids]))
            assert jump <= 0.01, f"Robot stage discontinuity {jump}"
        grip = (
            z["gripper"]
            if "gripper" in z
            else np.tile(stage.get("gripper", [0.0, 0.0]), (len(t), 1))
        )
        assert (
            grip.shape == (len(t), 2) and np.isfinite(grip).all()
        ), "Invalid gripper commands"
        if previous_gripper is not None:
            assert (
                np.max(abs(grip[0] - previous_gripper)) < 1e-9
            ), f'Gripper stage discontinuity at {stage["label"]}; explicit audited manipulation event required'
        if stage.get("required_closed_arm") is not None:
            assert np.all(
                grip[:, int(stage["required_closed_arm"])] > 128
            ), "Held stage opens its gripper"
        previous_gripper = grip[-1].copy()
        keep = slice(None) if i == 0 else slice(1, None)
        output["time_s"].append(t[keep] - t[0] + clock)
        for key in ["qpos", "feet", "contacts", "normal_caps"]:
            output[key].append(z[key][keep])
        output["phases"].append(np.full(len(t[keep]), stage["label"]))
        output["gripper"].append(grip[keep])
        clock += float(t[-1] - t[0])
        previous = q[-1]
    result = {k: np.concatenate(v) for k, v in output.items()}
    result["initial_qpos"] = result["qpos"][0].copy()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stages", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    spec = json.loads(Path(a.stages).read_text())
    np.savez_compressed(a.output, **compose(spec["stages"]))
