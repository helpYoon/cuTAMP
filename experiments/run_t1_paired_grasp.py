"""Paired model-only T1 grasp experiment. Run from cuTAMP with bfm_wm's Python.

OPENBLAS_NUM_THREADS=1 <bfm_wm>/.venv/bin/python experiments/run_t1_paired_grasp.py

The shared learned-rollout Jacobian and the implicit IK Jacobian are composed at
the achieved soles. Centered differences rerun the deployed clock and solve IK.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from t1_paired_ik import GraspIK
from t1_paired_rollout import BFM, CHECKPOINT, CHECKPOINT_HASH, Rollout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cutamp"))
from reachability import apply_command

sys.path.pop(0)

START = np.array([0.12, 0.21, 0.0])
MARGIN = 0.05
XY_LIMIT = 0.02
YAW_LIMIT = np.pi / 180
SCALE = np.array([0.012, 0.012, np.deg2rad(1.5)])
RESULT = ROOT / "data/t1-reachability/grasp-landing/paired-model.json"


def array(value):
    return value.detach().cpu().double().numpy()


class Experiment:
    def __init__(self):
        self.rollout = Rollout(lag=2)
        self.ik = GraspIK(ROOT)
        self.cache = {}

    def landing(self, command, derivative=False):
        key = tuple(np.asarray(command, dtype=np.float32).tolist())
        if key in self.cache and (not derivative or "J" in self.cache[key]):
            return self.cache[key]
        with torch.set_grad_enabled(derivative):
            u = torch.tensor(key, device="cuda", dtype=torch.float32, requires_grad=derivative)
            raw = self.rollout(u[None])
            feet = raw["soles"][0]
            record = {
                k: array(raw[k][0]).tolist()
                for k in (
                    "soles",
                    "reference",
                    "start_soles",
                    "xy_error",
                    "yaw_error_deg",
                    "clock",
                    "height",
                    "tilt",
                    "max_root_speed",
                    "max_joint_rms",
                )
            }
            record["command"] = list(key)
            if derivative:

                def jac(values):
                    return np.stack(
                        [array(torch.autograd.grad(v, u, retain_graph=True)[0]) for v in values.reshape(-1)]
                    )

                record["J"] = jac(feet)
                record["J_reference"] = jac(raw["reference"][0])
        self.cache[key] = record
        return record

    def evaluate(self, command, wrist, seed, derivative=False, fresh=False):
        if fresh:
            self.cache.pop(tuple(np.asarray(command, dtype=np.float32).tolist()), None)
        r = self.landing(command, derivative)
        feet = torch.tensor(r["soles"], device="cuda", dtype=torch.float64, requires_grad=derivative)
        k = self.ik.evaluate(feet, wrist, seed)
        margin = k["margin"].item()
        row = {key: v for key, v in r.items() if key not in ("J", "J_reference")}
        row.update(
            {
                key: float(k[key])
                for key in ("wrist_pos_err_m", "wrist_rot_err_rad", "sole_pos_err_m", "sole_rot_err_rad", "optimality")
            }
        )
        row.update(
            margin_m=margin,
            loss_m2=max(0.0, MARGIN - margin) ** 2,
            com=array(k["com"]).tolist(),
            posture=array(k["q"]).tolist(),
            edge=int(k["edge"]),
            active=int(k["active"]),
            ik_converged=bool(k["solved"]),
            ik_iterations=int(k["iterations"]),
        )
        row["tracking_ok"] = max(row["xy_error"]) < XY_LIMIT and max(row["yaw_error_deg"]) < 1.0
        row["settled"] = row["max_root_speed"] < 0.02 and row["max_joint_rms"] < 0.01
        row["ik_ok"] = bool(
            row["ik_converged"]
            and row["wrist_pos_err_m"] < 0.005
            and row["wrist_rot_err_rad"] < 0.05
            and row["sole_pos_err_m"] < 0.001
            and row["sole_rot_err_rad"] < 0.01
        )
        if derivative:
            gradients = {}
            for key in ("margin", "hull_only_margin"):
                cotangent = array(torch.autograd.grad(k[key], feet, retain_graph=True)[0]).ravel()
                gradients[key] = cotangent @ r["J"]
            g = gradients["margin"]
            row.update(
                dm_du=g.tolist(),
                hull_only_dm_du=gradients["hull_only_margin"].tolist(),
                posture_dm_du=(g - gradients["hull_only_margin"]).tolist(),
                dloss_du=(-2 * max(0.0, MARGIN - margin) * g).tolist(),
                sole_pose_jacobian=r["J"].tolist(),
            )
        return row

    def propose(self, row):
        # Trust-region SQP step in dimensionless command coordinates. Leave a
        # strict interior tracking buffer; every accepted trial passes a fresh rollout.
        u = row["command"]
        r = self.landing(u, True)
        diff = np.array(r["soles"]) - r["reference"]
        diff[:, 2] = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
        d_diff = (r["J"] - r["J_reference"]).reshape(2, 3, 3) * SCALE
        dm = np.array(row["dm_du"]) * SCALE
        need = 0.0502 - row["margin_m"]

        def objective(x):
            return max(0.0, need - dm @ x) ** 2 / max(need, 0.0002) ** 2 + 1e-4 * x @ x

        def feasible(x):
            residual = diff + np.einsum("ijk,k->ij", d_diff, x)
            return np.r_[
                1 - np.sum((residual[:, :2] / (0.01995)) ** 2, axis=1),
                1 - (residual[:, 2] / np.deg2rad(0.997)) ** 2,
                1 - x @ x,
            ]

        opt = minimize(
            objective,
            np.zeros(3),
            method="SLSQP",
            bounds=[(-1, 1)] * 3,
            constraints=[{"type": "ineq", "fun": feasible}],
            options={"maxiter": 100, "ftol": 1e-12},
        )
        if not opt.success or not np.isfinite(opt.x).all():
            raise RuntimeError(f"Local constrained optimization failed: {opt.message}")
        return opt.x * SCALE

    def optimize(self, name, wrist, seed):
        current = self.evaluate(START, wrist, seed, True)
        assert current["tracking_ok"] and current["ik_ok"], (name, current)
        initial = current.copy()
        history = []
        for n in range(10):
            if current["margin_m"] >= MARGIN:
                break
            delta = self.propose(current)
            for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
                command = np.asarray(current["command"]) + factor * delta
                if apply_command(np.array(current["start_soles"]), "L", command) is None:
                    continue
                trial = self.evaluate(command, wrist, seed)
                ok = trial["tracking_ok"] and trial["ik_ok"] and trial["margin_m"] > current["margin_m"] + 1e-7
                history.append(
                    {
                        "iteration": n + 1,
                        "accepted": ok,
                        **{
                            k: trial[k]
                            for k in ("command", "margin_m", "clock", "xy_error", "yaw_error_deg", "wrist_pos_err_m")
                        },
                    }
                )
                if ok:
                    current = self.evaluate(command, wrist, seed, True)
                    break
            else:
                break
        final = self.evaluate(current["command"], wrist, seed, fresh=True)
        assert final["tracking_ok"] and final["ik_ok"]
        return {"name": name, "wrist_world": wrist.tolist(), "initial": initial, "final": final, "trials": history}

    def direct_margin_gradient(self, wrist, seed):
        """One reverse pass from grasp margin to command, through the actual rollout."""
        u = torch.tensor(START, dtype=torch.float32, device="cuda", requires_grad=True)
        landing = self.rollout(u[None])
        ik = self.ik.evaluate(landing["soles"][0], wrist, seed)
        return array(torch.autograd.grad(ik["margin"], u)[0])

    def probe(self, direction, wrist, seed, eps=0.0005):
        plus = self.evaluate(START + eps * direction, wrist, seed)
        minus = self.evaluate(START - eps * direction, wrist, seed)
        return {
            "direction": direction.tolist(),
            "epsilon": eps,
            "dm_du": (plus["margin_m"] - minus["margin_m"]) / (2 * eps),
            "dloss_du": (plus["loss_m2"] - minus["loss_m2"]) / (2 * eps),
            "clocks": [minus["clock"], plus["clock"]],
            "edges": [minus["edge"], plus["edge"]],
            "active_counts": [minus["active"], plus["active"]],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    start_time = time.monotonic()
    torch.set_num_threads(1)
    exp = Experiment()
    nominal = exp.landing(START)
    fixed = exp.ik.fixture(np.array(nominal["soles"]))
    seed = fixed["seed"]
    fixture = {
        "seed": seed.tolist(),
        "world_object": fixed["world_object"].tolist(),
        "object_to_wrist": {k: v.tolist() for k, v in fixed["object_to_wrist"].items()},
        "wrists": {k: fixed[k].tolist() for k in ("near", "far")},
    }
    runs = []
    for name, wrist in fixture["wrists"].items():
        wrist = np.asarray(wrist)
        run = exp.optimize(name, wrist, seed)
        run["finite_difference"] = [exp.probe(direction, wrist, seed) for direction in np.eye(3)]
        # The yaw derivative is much smaller than either translation derivative;
        # also probe a larger yaw increment against float32 rollout roundoff.
        run["yaw_fd_sensitivity"] = [exp.probe(np.eye(3)[2], wrist, seed, eps=e) for e in (0.001, 0.002)]
        direct = exp.direct_margin_gradient(wrist, seed)
        np.testing.assert_allclose(direct, run["initial"]["dm_du"], rtol=0.002, atol=2e-5)
        run["initial"]["direct_autograd_dm_du"] = direct.tolist()
        for axis in (0, 1):
            fd = run["finite_difference"][axis]
            assert fd["clocks"] == [run["initial"]["clock"]] * 2
            np.testing.assert_allclose(fd["dm_du"], direct[axis], rtol=0.01, atol=1e-4)
        runs.append(run)
        print(
            name,
            "initial/final margin",
            run["initial"]["margin_m"],
            run["final"]["margin_m"],
            "initial gradient",
            run["initial"]["dm_du"],
            "final command",
            run["final"]["command"],
            flush=True,
        )
    assert len(runs) == 2 and runs[0]["initial"]["soles"] == runs[1]["initial"]["soles"]
    # These are fixture assertions, separate from contact or motion validation.
    assert runs[0]["initial"]["margin_m"] >= MARGIN > runs[1]["initial"]["margin_m"]
    for run in runs:
        assert run["final"]["margin_m"] >= MARGIN and run["final"]["tracking_ok"]
    paths = [
        Path(__file__),
        Path(__file__).with_name("t1_paired_ik.py"),
        Path(__file__).with_name("t1_paired_rollout.py"),
        Path(__file__).with_name("score_t1_landing_accuracy.py"),
        BFM / "src/networks.py",
        BFM / "src/landing.py",
        BFM / "utils/export_landing.py",
        ROOT / "cutamp/robots/assets/t1_29dof/t1_planner.urdf",
        ROOT / "cutamp/robots/assets/t1_29dof/t1_29dof.yml",
    ]
    sources = {
        str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    result = {
        "loss_definition": "max(0, 0.05 - margin_m)^2 in m^2",
        "command_units": ["m", "m", "rad"],
        "reference_frame": "warmed right support at issue; soles reported in native model world",
        "scope": "single unloaded L step, prescribed object-relative wrist frames, flat double support and fixed crouch reference for keyframe IK",
        "fixture": fixture,
        "runs": runs,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint": str(BFM / CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_HASH,
        "lag_substeps": 2,
        "hold_ticks": exp.rollout.hold_ticks,
        "warm_ticks": 50,
        "control_hz": 50,
        "friction": 1.0,
        "payload": "unloaded",
        "initial_history": "synthetic standing plus 50 HOLD ticks",
        "timing_s": time.monotonic() - start_time,
        "source_sha256": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
