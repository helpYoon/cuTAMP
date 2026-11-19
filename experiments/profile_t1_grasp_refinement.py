"""Measure the T1 command derivative, geometric ablation and closed-loop step costs.

Run from the root with PYTHONPATH=. python experiments/profile_t1_grasp_refinement.py.
Timings bracket synchronized CUDA work after loading the model and preparing the shared
measured history. The first and repeated measurements use identical targets.
"""

import argparse
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

from cutamp.landing_predictor import LandingPredictor, prediction_precision
from cutamp.step_ik import solve_stance_ik
from cutamp.step_refinement import refine_gap
from cutamp.tests.test_grasp_stance_ik import paired_inputs


def now():
    torch.cuda.synchronize()
    return perf_counter()


def as_list(x):
    return x.detach().cpu().tolist()


def hull_gradient_ik(*a, **kw):
    # The nonlinear margin and all feasibility checks are identical. Only the
    # derivative of the manipulation comfort row stops at the solved CoM.
    ik = solve_stance_ik(*a, **kw)
    return replace(ik, margin=ik.hull_only_margin)


def profile_order(predictor, start, commands, order, wrists, seeds):
    with prediction_precision():
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        c = commands.clone().requires_grad_()
        t0 = now()
        pred = predictor.rollout(start, order, c)
        ik = solve_stance_ik(pred.steps[-1].achieved, wrists, seeds, "left")
        t1 = now()
        (grad,) = torch.autograd.grad(ik.margin.sum(), c)
        t2 = now()
        return {
            "forward_s": t1 - t0,
            "backward_s": t2 - t1,
            "batch_per_s": len(c) / (t2 - t0),
            "peak_MiB": torch.cuda.max_memory_allocated() / 2**20,
            "incremental_peak_MiB": (torch.cuda.max_memory_allocated() - baseline) / 2**20,
            "dmargin_du": as_list(grad),
            "ticks_per_row_including_hold": [as_list(x.valid_ticks.sum(1)) for x in pred.steps],
            "ik_ok": as_list(ik.ok),
            "tracking_ok": as_list(pred.tracking_ok),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/t1-reachability/landing-accuracy/landing-seed3.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = LandingPredictor(args.bundle)
    wrists, seeds = paired_inputs(p.device, torch.float32)
    d = wrists.new_zeros(2, 38)
    d[:, 0] = 0.25 / 0.45
    d[:, -5:] = -1.0
    start = p.standing_start(d)
    standing = p.model.feet(start.tensors[:6])[0]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "metadata": p.metadata,
        "order_profiles": {},
        "ablations": {},
    }
    for order in ("L", "R", "LR", "RL"):
        u = d.new_tensor([[0.12, 0.21 if s == "L" else -0.21, 0.0] for s in order])[None].repeat(2, 1, 1)
        report["order_profiles"][order] = {
            name: profile_order(p, start, u, order, wrists, seeds) for name in ("first", "repeat")
        }
    u = d.new_tensor([[[0.12, 0.21, 0.0]]]).repeat(2, 1, 1)
    with torch.no_grad():
        none = refine_gap(p, start, "", u[:, :0], "left", wrists, seeds, max_iterations=0)
    report["no_step"] = {"margin_m": as_list(none.margin), "feasible": as_list(none.ok)}
    for name in ("full", "geometric_then_model_filter", "fixed_com_gradient"):
        geometric = name == "geometric_then_model_filter"
        context = (
            patch("cutamp.step_refinement.solve_stance_ik", hull_gradient_ik)
            if name == "fixed_com_gradient"
            else nullcontext()
        )
        torch.cuda.reset_peak_memory_stats()
        t0 = now()
        with context:
            r = refine_gap(
                None if geometric else p,
                standing if geometric else start,
                "L",
                u,
                "left",
                wrists,
                seeds,
                max_iterations=4,
            )
        elapsed = now() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        with torch.no_grad():
            checked = refine_gap(p, start, "L", r.commands, "left", wrists, seeds, max_iterations=0)
        report["ablations"][name] = {
            "elapsed_s": elapsed,
            "peak_MiB": peak,
            "iterations": r.iterations,
            "commands": as_list(r.commands),
            "dmargin_du": as_list(r.initial_gradient),
            "optimized_margin_m": as_list(r.margin),
            "fresh_model_margin_m": as_list(checked.margin),
            "fresh_model_feasible": as_list(checked.ok),
            "fresh_achieved_soles": as_list(checked.achieved),
            "fresh_xy_m": as_list(checked.prediction.final_xy_error),
            "fresh_yaw_deg": as_list(checked.prediction.final_yaw_error.rad2deg()),
            "fresh_wrist_m": as_list(checked.ik.wrist_pos_error),
            "fresh_wrist_rad": as_list(checked.ik.wrist_rot_error),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
