"""One differentiable, deployed-clock T1 left landing from a common seeded start.

Inputs are batched (x, y, yaw) commands in the warmed right sole frame with units
m/m/rad. World outputs use the native physical frame of the model, in L/R order.
No physics simulator is constructed.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CUTAMP_ROOT", HERE.parent if HERE.name == "experiments" else Path.cwd()))
BFM = Path(os.environ.get("BFM_ROOT", Path.home() / "bfm_wm"))
sys.path.insert(0, str(REPO / "experiments"))
from score_t1_landing_accuracy import CHECKPOINT, HOLD, SETTLED_WINDOW, WARMUP, Model, sha

CHECKPOINT_HASH = "766d602d8cabf1ccacfad7b59d7d09baac53ad17b59b9f6ead57e09739435f8b"


class Rollout:
    def __init__(self, lag=2, device="cuda", hold_ticks=HOLD):
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The paired rollout experiment requires CUDA")
        path = BFM / CHECKPOINT
        if sha(path) != CHECKPOINT_HASH:
            raise ValueError(f"Checkpoint hash mismatch at {path}")
        self.model = Model(BFM, path, device)
        self.device = device
        self.lag = lag
        if hold_ticks < SETTLED_WINDOW:
            raise ValueError(f"hold_ticks must cover the {SETTLED_WINDOW}-tick settling check")
        self.hold_ticks = hold_ticks
        with torch.no_grad():
            self.context = torch.zeros(1, 38, device=device)
            self.context[:, 0] = (1.0 - 0.75) / 0.45
            self.context[:, -5:] = (lag - 5.0) / 3
            self.start = self.model.reset(1)
            hold = torch.zeros(1, 5, device=device)
            for _ in range(WARMUP):
                self.start = self.model.tick(self.start, hold, self.context)
            assert (self.start[-1][:, 2:4] == 0).all()
            self.start_soles = self.model.pose(self.start[:6])

    def __call__(self, command):
        assert command.ndim == 2 and command.shape[0] and command.shape[1] == 3
        assert command.device.type == "cuda" and command.dtype == torch.float32
        b = len(command)
        start = tuple(t.expand(b, *t.shape[1:]).contiguous() for t in self.start)
        context = self.context.expand(b, -1).contiguous()
        zero = command.new_zeros(b, 5)
        issue = torch.cat([command.new_ones(b, 1), command.new_zeros(b, 1), command], dim=1)
        initial_feet = self.start_soles.expand(b, -1, -1)
        reference = torch.stack((self.model.landing.compose(initial_feet[:, 1], command), initial_feet[:, 1]), dim=1)
        state = self.model.tick(start, issue, context)
        clocks = state[-1][:, 2].detach().round().long() + 1
        tick_by_row = clocks + self.hold_ticks
        t_min, t_max = (int(t) for t in torch.aminmax(tick_by_row))
        assert t_min - self.hold_ticks >= 40 and t_max - self.hold_ticks <= 62
        window_start = t_min - SETTLED_WINDOW
        max_joint = command.new_zeros(b)
        max_root = command.new_zeros(b)
        final = None
        for tick in range(2, t_max + 1):
            state = self.model.tick(state, zero, context)
            if tick > window_start:
                with torch.no_grad():
                    inside = (tick_by_row - SETTLED_WINDOW < tick) & (tick <= tick_by_row)
                    max_joint = torch.where(
                        inside, torch.maximum(max_joint, state[5].square().mean(-1).sqrt()), max_joint
                    )
                    max_root = torch.where(inside, torch.maximum(max_root, state[2].norm(dim=-1)), max_root)
            if tick == t_min:
                final = state[:6]
            elif tick > t_min:
                final = tuple(
                    torch.where((tick_by_row == tick).view(b, *([1] * (src.ndim - 1))), src, dst)
                    for src, dst in zip(state[:6], final)
                )

        link_xyz, link_R, v, w = self.model.kin.forward(*final)
        feet = self.model.landing.soles(link_xyz, link_R, v, w)
        xyz = torch.cat([feet["pose"][..., :2], feet["zc"][..., None]], dim=-1)
        diff = feet["pose"] - reference
        return {
            "soles": feet["pose"],
            "sole_xyz": xyz,
            "sole_R": link_R,
            "physical": final,
            "clock": clocks,
            "reference": reference,
            "start_soles": initial_feet,
            "xy_error": diff[..., :2].norm(dim=-1),
            "yaw_error_deg": torch.rad2deg(torch.atan2(torch.sin(diff[..., 2]), torch.cos(diff[..., 2]))).abs(),
            "height": feet["h"],
            "tilt": feet["tilt"],
            "max_root_speed": max_root,
            "max_joint_rms": max_joint,
        }


if __name__ == "__main__":
    torch.set_num_threads(1)
    r = Rollout()
    u = torch.tensor([[0.12, 0.21, 0], [0.1202, 0.21, 0], [0.1198, 0.21, 0]], device="cuda", requires_grad=True)
    t = time.time()
    out = r(u)
    si = out["soles"][:, 0, 0]
    g = torch.autograd.grad(si.sum(), u)[0]

    def values(t):
        return t.detach().cpu().tolist()

    result = {
        k: values(out[k][0])
        for k in (
            "soles",
            "sole_xyz",
            "reference",
            "start_soles",
            "xy_error",
            "yaw_error_deg",
            "clock",
            "max_root_speed",
            "max_joint_rms",
            "tilt",
            "height",
        )
    }
    result.update(
        grad=values(g[0]),
        dx_fd=float(((si[1] - si[2]) / 0.0004).detach()),
        runtime_s=time.time() - t,
        gpu=torch.cuda.get_device_name(),
    )
    assert not any(k.split(".")[0] in {"mujoco", "mjlab", "mujoco_warp"} for k in sys.modules)
    assert out["xy_error"][0].max().item() < 0.02 and out["yaw_error_deg"][0].max().item() < 1.0
    print(json.dumps(result, indent=2))
