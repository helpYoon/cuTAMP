"""Collect T1 controller issue states and particle commands on disjoint floor-transfer layouts."""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from cutamp import step_adam
from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import load_env, resolve_env_path
from cutamp.grasps import _SETS
from cutamp.landing_predictor import LandingPredictor
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

# Pick and destination centers in meters. The shipped example remains a separate planner benchmark.
LAYOUTS = [
    ("train", "forward", 711, (0.80, 0.14), (1.12, 0.17)),
    ("train", "diagonal", 712, (0.84, 0.20), (1.12, 0.24)),
    ("train", "right", 713, (0.80, -0.15), (1.11, -0.18)),
    ("train", "shallow", 714, (0.75, 0.12), (1.05, 0.13)),
    ("val", "offset", 821, (0.805, 0.18), (1.12, 0.205)),
    ("test", "held-out", 913, (0.815, 0.15), (1.14, 0.17)),
]


def collect(root):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    original_optimize = step_adam.optimize_particles
    original_rollout = LandingPredictor.rollout
    for split, name, seed, pick, place in LAYOUTS:
        template = yaml.safe_load(Path(resolve_env_path("two_steps_floor_transfer")).read_text())
        template["demo_seed"] = seed
        template["name"] = "step-sequence-" + name
        bodies = template["geometries"]["cuboid"]
        bodies["part"]["pose"][:2], bodies["pad"]["pose"][:2] = list(pick), list(place)
        scene = root / (name + ".yml")
        scene.write_text(yaml.safe_dump(template, sort_keys=False))
        env = load_env(scene)
        torch.manual_seed(seed)
        _SETS.clear()
        group_start = len(rows)

        def record(predictor, issue, order, commands, *, group=name, source_split=split, **kwargs):
            if order and order != "none":
                rows.append(
                    {
                        "tensors": tuple(t.detach().cpu() for t in issue.tensors),
                        "dynamics": issue.dynamics.detach().cpu(),
                        "commands": commands.detach().cpu().clone(),
                        "order": order,
                        "group": group,
                        "history_source": issue.history_source,
                        "split": source_split,
                    }
                )
            return original_rollout(predictor, issue, order, commands, **kwargs)

        def capture(initial, evaluate, iterations, lr, **options):
            with patch.object(LandingPredictor, "rollout", record):
                return original_optimize(initial, evaluate, iterations, lr, **options)

        cfg = TAMPConfiguration(
            step_optimizer="adam",
            step_refine_iterations=1,
            max_step_schedules=1,
            num_particles=64,
            enable_visualizer=False,
            curobo_plan=False,
            experiment_root=str(root / "collection"),
        )
        with patch.object(step_adam, "optimize_particles", capture):
            run_cutamp(
                env,
                cfg,
                CostReducer(default_constraint_to_mult.copy()),
                ConstraintChecker(default_constraint_to_tol.copy()),
                experiment_id=name,
            )
        print(name, [(r["order"], r["history_source"], len(r["commands"])) for r in rows[group_start:]], flush=True)
        partial = root / "starts.partial.pt"
        torch.save(rows, partial)
        partial.replace(root / "starts.pt")
    assert all(any(row["split"] == split for row in rows) for split in ("train", "val", "test"))
    assert any("hold" in row["history_source"] and "home" not in row["history_source"] for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args().output)
