"""Time one real T1 run_cutamp call with synchronized, exclusive phase timings."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cutamp.algorithm as planner
import cutamp.step_particles as particles
import cutamp.step_refinement as refinement
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tests.paired_grasp_scene import paired_setup


class Timings:
    def __init__(self):
        self.stack = []
        self.seconds = Counter()
        self.calls = Counter()
        self.vjps = defaultdict(list)

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize()
        self.stack.append([name, perf_counter(), 0.0])
        try:
            yield
        finally:
            torch.cuda.synchronize()
            name, start, children = self.stack.pop()
            duration = perf_counter() - start
            self.seconds[name] += duration - children
            self.calls[name] += 1
            if self.stack:
                self.stack[-1][2] += duration

    def wrap(self, stack, obj, name, category):
        original = getattr(obj, name)

        @wraps(original)
        def timed(*args, **kwargs):
            key = category(*args, **kwargs) if callable(category) else category
            with self.stage(key):
                return original(*args, **kwargs)

        stack.enter_context(patch.object(obj, name, timed))

    def instrument(self, stack):
        for obj, name, category in (
            (planner, "setup_cutamp", "world_setup"),
            (planner, "_load_step_search", "rank_assets"),
            (planner, "_load_predictor", "load_model"),
            (particles, "_placements_and_grasps", "particle_bindings"),
            (refinement, "_evaluate", "other_evaluation"),
            (refinement, "solve_stance_ik", "ik"),
            (particles, "solve_stance_ik", "ik"),
            (particles.CostFunction, "stepped_scene_costs", "keyframe_collision"),
            (particles, "sweep_constraint_rows", "swept_collision"),
            (particles, "validate_step_geometry", "final_sweep"),
            (refinement, "composite_step", "qp"),
            (
                refinement,
                "_jacobian",
                lambda rows, *args, **kw: "center_jacobian" if "centering" in rows else "hard_jacobian",
            ),
        ):
            self.wrap(stack, obj, name, category)
        from cutamp.landing_predictor import LandingPredictor
        from cutamp.step_planner import ScheduleSearch

        def rollout_kind(predictor, initial, order, commands):
            dual = torch.autograd.forward_ad.unpack_dual(commands)
            return "directional_policy_wm" if dual.tangent is not None else "learned_rollout"

        self.wrap(stack, LandingPredictor, "rollout", rollout_kind)
        if hasattr(LandingPredictor, "linearize"):
            self.wrap(stack, LandingPredictor, "linearize", "learned_linearization")
        self.wrap(stack, ScheduleSearch, "next", "schedule_ranking")
        grad = torch.autograd.grad

        @wraps(grad)
        def timed_grad(*args, **kwargs):
            frame = sys._getframe(1)
            if frame.f_code.co_name != "_jacobian":
                return grad(*args, **kwargs)
            local = frame.f_locals
            name = next((k for k, v in local["rows"].items() if v is local.get("row")), "other")
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = grad(*args, **kwargs)
            end.record()
            self.vjps[name].append((start, end))
            return result

        stack.enter_context(patch.object(torch.autograd, "grad", timed_grad))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pick-only", action="store_true")
    parser.add_argument("--label", default="far", choices=("far", "near"))
    parser.add_argument("--updates", default=4, type=int, help="Maximum SQP updates for the single grounded schedule")
    parser.add_argument("--clock-only", action="store_true", help="Measure end-to-end latency without phase probes")
    parser.add_argument(
        "--reverse-reference", action="store_true", help="Use the original full reverse tape at each SQP linearization"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    env, q, inputs = paired_setup(args.label, place=not args.pick_only)
    torch.cuda.synchronize()
    fixture_seconds = perf_counter() - started
    cfg = TAMPConfiguration(
        step_mode="model",
        num_particles=1,
        step_refine_iterations=args.updates,
        max_step_schedules=1,
        enable_visualizer=False,
        curobo_plan=False,
        experiment_root=str(args.output),
        landing_bundle=str(ROOT / "data/t1-reachability/landing-accuracy/landing-seed3.pt"),
    )
    torch.manual_seed(641)
    bytes_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    timings = Timings()
    with ExitStack() as stack:
        if args.reverse_reference:
            from cutamp.landing_predictor import LandingPredictor

            stack.enter_context(patch.object(LandingPredictor, "linearize", LandingPredictor.rollout, create=True))
        if not args.clock_only:
            timings.instrument(stack)
        with timings.stage("other_planner"):
            _, n, m = planner.run_cutamp(
                env,
                cfg,
                CostReducer(default_constraint_to_mult.copy()),
                ConstraintChecker(default_constraint_to_tol.copy()),
                q_init=q,
                step_inputs=inputs,
                experiment_id="plan",
            )
    wall = sum(timings.seconds.values())
    report = {
        "gpu": torch.cuda.get_device_name(),
        "scene": env.name,
        "grasp": args.label,
        "step_mode": cfg.step_mode,
        "particles": cfg.num_particles,
        "orders": inputs.orders,
        "updates": cfg.step_refine_iterations,
        "accepted": n,
        "status": m["status"],
        "wall_s": wall,
        "extra_peak_mib": (torch.cuda.max_memory_allocated() - bytes_before) / 2**20,
        "fixture_setup_s_excluded": fixture_seconds,
        "clock_only": args.clock_only,
        "reverse_reference": args.reverse_reference,
        "planner_elapsed_s": m["elapsed_s"],
        "first_accepted_s": m["time_to_first_accepted_s"],
        "phase_exclusive_s": dict(timings.seconds),
        "phase_calls": dict(timings.calls),
        "vjp_gpu_stream_s": {
            k: {"calls": len(v), "s": sum(a.elapsed_time(b) for a, b in v) / 1000} for k, v in timings.vjps.items()
        },
        "gaps": m.get("gaps", []),
    }
    (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], round(wall, 2), "s", dict(timings.seconds))
    if n == 0:
        raise RuntimeError("Profile did not yield an accepted particle")


if __name__ == "__main__":
    main()
