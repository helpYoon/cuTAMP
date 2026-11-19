"""Export one differentiable CUDA tick for cuTAMP with bfm_wm's Python.

Example from the cuTAMP checkout:
  <bfm_wm>/.venv/bin/python experiments/export_t1_landing.py --bfm-root <bfm_wm>

The archive contains frozen networks, sole FK, the deployed clock and provenance.
cuTAMP loads it without importing the training repository or a simulator.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from score_t1_landing_accuracy import ACTUATORS, CHECKPOINT, URDF, Model, sha

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASH = "766d602d8cabf1ccacfad7b59d7d09baac53ad17b59b9f6ead57e09739435f8b"


class LandingTick(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.policy = model.policy
        self.wm = model.wm
        self.register_buffer("home_joints", model.q0)
        self.model = model

    def forward(self, state, command, context):
        return self.model.tick(state, command, context)

    def feet(self, physical):
        values = self.model.landing.soles(*self.model.kin.forward(*physical))
        return values["pose"], values["h"], values["tilt"]


def export(bfm_root, output):
    if torch.cuda.current_device() != 0:
        raise ValueError("Export on logical CUDA device 0; the traced FK uses device-specific constants")
    checkpoint = bfm_root / CHECKPOINT
    if sha(checkpoint) != EXPECTED_HASH:
        raise ValueError(f"Unexpected T1 landing checkpoint: {checkpoint}")
    model = Model(bfm_root, checkpoint, "cuda")
    module = LandingTick(model).eval().requires_grad_(False)
    state = model.reset(2)
    cmd = torch.zeros(2, 5, device="cuda")
    ctx = torch.zeros(2, 38, device="cuda")
    traced = torch.jit.trace_module(module, {"forward": (state, cmd, ctx), "feet": (state[:6],)}, check_trace=False)
    # Verify that a different batch shape really advances both deployed foot clocks.
    state = model.reset(3)
    cmd = cmd[:1].repeat(3, 1)
    ctx = ctx[:1].repeat(3, 1)
    for _ in range(50):
        state = model.tick(state, cmd, ctx)
    cmd[:, 0] = 1.0
    cmd[1, 1] = 1.0
    cmd[:, 2:] = cmd.new_tensor([[0.12, 0.21, 0.0], [0.12, -0.21, 0.0], [0.24, 0.21, 0.0]])
    expected = model.tick(state, cmd, ctx)
    actual = traced(state, cmd, ctx)
    for value, ref in zip(actual, expected):
        torch.testing.assert_close(value, ref, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(traced.feet(actual[:6])[0], model.pose(expected[:6]), rtol=2e-5, atol=2e-6)
    source_files = [
        "src/networks.py",
        "src/dr.py",
        "src/mdp.py",
        "src/landing.py",
        "src/motion.py",
        "src/sensing.py",
        "utils/rotations.py",
        "utils/export_landing.py",
        "robot/t1.py",
        URDF,
        ACTUATORS,
    ]
    tree = ast.parse((bfm_root / "robot/t1.py").read_text())
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "T1_JOINT_NAMES"
    )
    metadata = {
        "format": 2,
        "cuda_index": 0,
        "checkpoint_sha256": EXPECTED_HASH,
        "joint_names": names,
        "history_ticks": model.hist,
        "control_dt": 0.02,
        "export_torch": str(torch.__version__),
        "bfm_revision": subprocess.check_output(["git", "-C", str(bfm_root), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: sha(bfm_root / name) for name in source_files},
        "adapter_sha256": sha(Path(__file__).with_name("score_t1_landing_accuracy.py")),
        "exporter_sha256": sha(Path(__file__)),
    }
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata["deployment_sha256"] = fingerprint
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as temp:
        path = Path(temp.name)
    try:
        torch.jit.save(traced, str(path), _extra_files={"landing.json": json.dumps(metadata)})
        path.replace(output)
    finally:
        path.unlink(missing_ok=True)
    print(f"Exported {output} ({output.stat().st_size:,} bytes), deployment {fingerprint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/t1-reachability/landing-accuracy/landing-seed3.pt")
    args = parser.parse_args()
    export(args.bfm_root.resolve(), args.output)
