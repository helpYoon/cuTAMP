"""The real-leg plans' torque, speed and balance report (design 7.4a and section 8).

For every real-leg raw plan (``run_cutamp --save_plan``, schema ``stage3-raw``) under a study
directory, it runs cuRobo's two RNEA passes with the floor's push (``SegmentTorques``) on every segment at its saved
timing, FKs every frame, and writes ``PREFIX.json`` and ``PREFIX.md``:

- per joint, the peak |torque| against its effort, and the peak |velocity| against its motor's knee speed (bfm_wm's
  ``t1_actuators.npz``, ``v_knee``: above it the motor's torque falls) and cuRobo's velocity limit;
- per run, the largest Trunk tilt (from the base's pitch and roll) and sole drift from the stance (by FK); how many
  segments the balance gate slowed and sped up (a retimed segment keeps the planner's timing), its smallest CoM and
  ZMP margins, its time scales and the checks that bound them (``overall_metrics.json``); and every segment the gate
  left at the planner's timing whose torque passes an effort. The gate checks those segments' torques too, so one
  here means a check missed it.

Usage:
    python -m cutamp.scripts.real_legs_report --study /home/yoonwoo/cutamp-runs/<study> --out <prefix>
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PLAN_FILE = "motion_plan.pkl"  # run_cutamp's --save_plan auto, in each run's experiment directory


def find_plans(study: Path) -> list:
    """``[(run directory, raw plan)]`` of every real-leg raw plan under ``study``, in path order."""
    from cutamp.utils.plan_processor import RAW_SCHEMA_VERSION

    plans = []
    for path in sorted(Path(study).rglob(PLAN_FILE)):
        with open(path, "rb") as f:
            plan = pickle.load(f)
        if plan.get("schema_version") == RAW_SCHEMA_VERSION and plan.get("model") == "real_legs":
            plans.append((path.parent, plan))
    return plans


class _RealLegs:
    """cuRobo's view of the real legs: the balance gate's torques (``SegmentTorques``) and FK of the soles, the
    stance soles at home, and each joint's effort, velocity limit and knee speed."""

    def __init__(self, device_cfg=None):
        from curobo.types import DeviceCfg

        from cutamp.floor_wrench import SegmentTorques
        from cutamp.robots.t1 import REAL_LEGS, T1_29DOF_DIR, get_t1_kinematics

        device_cfg = DeviceCfg() if device_cfg is None else device_cfg
        self.model, self.names, self.device = REAL_LEGS, list(REAL_LEGS.joint_names), device_cfg.device
        self.torques = SegmentTorques(REAL_LEGS, get_t1_kinematics(REAL_LEGS, device_cfg).config)
        self.effort = self.torques.effort.cpu().numpy()
        self.velocity_limit = self.torques.velocity.cpu().numpy()  # cuRobo's own, as the gate reads it
        with torch.no_grad():
            self.stance = self._soles(device_cfg.to_device(REAL_LEGS.home).unsqueeze(0)).clone()  # [1, 2, 4, 4]
        # t1_actuators.npz's arrays follow bfm_wm's T1_JOINT_NAMES, which the MuJoCo fixture stores (as the asset tests
        # read it). The base joints are virtual and have no motor.
        actuators = np.load(T1_29DOF_DIR / "t1_actuators.npz")
        order = [str(j) for j in np.load(T1_29DOF_DIR / "mujoco_fixture.npz")["joint_names"]]
        self.actuated = [j for j in self.names if j in order]
        npz = [order.index(j) for j in self.actuated]
        self.cols = [self.names.index(j) for j in self.actuated]
        if not np.allclose(actuators["tau_max"][npz], self.effort[self.cols], rtol=0, atol=1e-4):
            raise RuntimeError("t1_actuators.npz's tau_max disagrees with cuRobo's effort: its joint order is not "
                               "the MuJoCo fixture's")
        self.knee_speed = actuators["v_knee"][npz].astype(float)

    def _soles(self, q: torch.Tensor) -> torch.Tensor:
        """``[T, 2, 4, 4]`` world poses of the model's sole frames at confs ``q`` ``[T, dof]``."""
        from curobo.types import JointState

        ks = self.torques.kin.compute_kinematics(JointState.from_position(q, joint_names=self.names))
        return torch.stack([ks.tool_poses.get_link_pose(f, make_contiguous=True).get_matrix()
                            for f in self.model.sole_frames], dim=1)

    def planted(self, q: torch.Tensor):
        """The largest sole drift from the stance (m, rad) and Trunk tilt from the vertical (rad) over confs ``q``.
        The tilt is the base's (``trunk_tilt_cosine``), in float64."""
        from cutamp.costs import curobo_pose_error
        from cutamp.joint_limit_cost import trunk_tilt_cosine

        soles = self._soles(q)
        pos, rot = curobo_pose_error(soles, self.stance.expand_as(soles))
        cos = trunk_tilt_cosine(q.double(), self.names.index("base_j_pitch"), self.names.index("base_j_roll"))
        return pos.max().item(), rot.max().item(), torch.acos(cos.clamp(-1.0, 1.0)).max().item()

    def run(self, run_dir: Path, plan: dict) -> dict:
        """One run's peaks over every segment at its saved timing, and its balance gate's numbers."""
        from curobo.types import JointState

        if plan["joint_names"] != self.names:
            raise ValueError(f"{run_dir}: the plan's joints are not the real-leg model's")
        torque, speed = np.zeros(len(self.names)), np.zeros(len(self.names))
        drift_m = drift_rad = tilt = 0.0
        unslowed_over = []
        for k, seg in enumerate(plan["segments"]):
            p, v, a = (torch.as_tensor(seg[key], device=self.device)
                       for key in ("position", "velocity", "acceleration"))
            with torch.no_grad():
                tau = self.torques(JointState(position=p, velocity=v, acceleration=a, joint_names=self.names))
                peak = tau.abs().amax(0).cpu().numpy()
                pos, rot, tl = self.planted(p)
            torque, speed = np.maximum(torque, peak), np.maximum(speed, np.abs(seg["velocity"]).max(0))
            drift_m, drift_rad, tilt = max(drift_m, pos), max(drift_rad, rot), max(tilt, tl)
            if seg["planner_timing"] is None:  # at the planner's timing
                unslowed_over += [{"segment": k, "joint": self.names[j], "torque": float(peak[j]),
                                   "effort": float(self.effort[j])} for j in np.flatnonzero(peak > self.effort)]
        metrics_path = run_dir / "overall_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        frames = [(len(seg["position"]), len(seg["planner_timing"]["position"]))
                  for seg in plan["segments"] if seg["planner_timing"] is not None]  # both at the same dt
        binding = None if metrics.get("retime_binding") is None else {}  # segments per check, joints summed
        for label, count in (metrics.get("retime_binding") or {}).items():  # "velocity <joint>", "torque <joint>", ...
            check = label.split()[0]
            binding[check] = binding.get(check, 0) + count
        return {
            "run": run_dir.name, "path": str(run_dir), "segments": len(plan["segments"]),
            "slowed": sum(n > n_planner for n, n_planner in frames),
            "sped_up": sum(n < n_planner for n, n_planner in frames),
            "retime_min": metrics.get("retime_min"), "retime_max": metrics.get("retime_max"), "binding": binding,
            "com_min_margin": metrics.get("traj_com_min_margin"), "zmp_min_margin": metrics.get("traj_zmp_min_margin"),
            "max_tilt": tilt, "max_sole_drift_m": drift_m, "max_sole_drift_rad": drift_rad,
            "peak_torque": {j: float(torque[c]) for j, c in zip(self.actuated, self.cols)},
            "peak_speed": {j: float(speed[c]) for j, c in zip(self.actuated, self.cols)},
            "unslowed_over_effort": unslowed_over,
        }

    def joints(self, runs: list) -> list:
        """Per actuated joint, its peaks over ``runs`` beside its effort, knee speed and velocity limit."""
        return [{"joint": j, "peak_torque": max(r["peak_torque"][j] for r in runs), "effort": float(self.effort[c]),
                 "peak_speed": max(r["peak_speed"][j] for r in runs), "knee_speed": float(knee),
                 "velocity_limit": float(self.velocity_limit[c])}
                for j, c, knee in zip(self.actuated, self.cols, self.knee_speed)]


def build_report(study: Path) -> dict:
    plans = find_plans(study)
    if not plans:
        raise SystemExit(f"no real-leg raw plans ({PLAN_FILE}, schema stage3-raw) under {study}")
    legs = _RealLegs()
    runs = [legs.run(run_dir, plan) for run_dir, plan in plans]

    def extreme(fn, key):
        values = [r[key] for r in runs if r[key] is not None]
        return fn(values) if values else None

    return {
        "study": str(study), "runs": runs, "joints": legs.joints(runs),
        "summary": {
            "runs": len(runs), "max_tilt": extreme(max, "max_tilt"),
            "max_sole_drift_m": extreme(max, "max_sole_drift_m"),
            "max_sole_drift_rad": extreme(max, "max_sole_drift_rad"), "min_com_margin": extreme(min, "com_min_margin"),
            "min_zmp_margin": extreme(min, "zmp_min_margin"), "retime_min": extreme(min, "retime_min"),
            "retime_max": extreme(max, "retime_max"),
            "unslowed_over_effort": sum(len(r["unslowed_over_effort"]) for r in runs),
        },
    }


def _markdown(report: dict) -> str:
    def num(x, digits=2, scale=1.0):
        return "—" if x is None else f"{scale * x:.{digits}f}"

    s, joints = report["summary"], report["joints"]
    lines = [
        "# Real-leg torque and speed report", "",
        f"{s['runs']} real-leg plan{'s' * (s['runs'] != 1)} under `{report['study']}`. Torques are cuRobo's two RNEA "
        "passes with the floor's push (SegmentTorques) on every segment at its saved timing, and speeds the saved "
        "velocities. Knee speed is t1_actuators.npz's v_knee, past which the motor's torque falls; the velocity limit "
        "is cuRobo's.", "",
        "## Joints (peaks over every run)", "",
        "| joint | peak torque [N m] | effort [N m] | torque / effort | peak speed [rad/s] | knee speed [rad/s] "
        "| speed / knee | velocity limit [rad/s] | speed / limit |",
        "|" + "---|" * 9,
    ]
    for j in joints:
        lines.append(
            f"| {j['joint']} | {num(j['peak_torque'])} | {num(j['effort'], 1)} | {num(j['peak_torque'] / j['effort'])} "
            f"| {num(j['peak_speed'])} | {num(j['knee_speed'])} | {num(j['peak_speed'] / j['knee_speed'])} "
            f"| {num(j['velocity_limit'])} | {num(j['peak_speed'] / j['velocity_limit'])} |"
        )
    past_knee = [j["joint"] for j in joints if j["peak_speed"] > j["knee_speed"]]
    over = [f"{r['run']} segment {o['segment']} {o['joint']} {o['torque']:.2f} of {o['effort']:.1f} N m"
            for r in report["runs"] for o in r["unslowed_over_effort"]]
    lines += [
        "",
        f"Joints past their knee speed: {', '.join(past_knee) or 'none'}.",
        f"Segments at the planner's timing past an effort: {'; '.join(over) or 'none'}.",
        "", "## Runs", "",
        "| run | segments | slowed | sped up | time scale min | time scale max | binding | CoM margin [cm] "
        "| ZMP margin [cm] | max tilt [rad] | sole drift [mm] | sole drift [rad] | top torque / effort |",
        "|" + "---|" * 13,
    ]
    effort = {j["joint"]: j["effort"] for j in joints}
    for r in report["runs"]:
        top = max(effort, key=lambda j: r["peak_torque"][j] / effort[j])
        binding = "—" if r["binding"] is None else ", ".join(f"{k} {v}" for k, v in sorted(r["binding"].items()))
        lines.append(
            f"| {r['run']} | {r['segments']} | {r['slowed']} | {r['sped_up']} | {num(r['retime_min'])} "
            f"| {num(r['retime_max'])} | {binding} "
            f"| {num(r['com_min_margin'], scale=100)} | {num(r['zmp_min_margin'], 3, 100)} | {num(r['max_tilt'], 3)} "
            f"| {num(r['max_sole_drift_m'], 2, 1000)} | {num(r['max_sole_drift_rad'], 4)} "
            f"| {r['peak_torque'][top] / effort[top]:.2f} ({top}) |"
        )
    lines += [
        "",
        f"Over every run: max tilt {num(s['max_tilt'], 3)} rad; "
        f"max sole drift {num(s['max_sole_drift_m'], 2, 1000)} mm "
        f"and {num(s['max_sole_drift_rad'], 4)} rad; min CoM margin {num(s['min_com_margin'], scale=100)} cm; min ZMP "
        f"margin {num(s['min_zmp_margin'], 3, 100)} cm; time scales from {num(s['retime_min'])}x to "
        f"{num(s['retime_max'])}x.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, type=Path,
                    help="a directory whose tree holds the runs' experiment directories (run_cutamp's experiments)")
    ap.add_argument("--out", required=True, type=Path, help="output prefix: writes PREFIX.json and PREFIX.md")
    a = ap.parse_args(argv)
    report = build_report(a.study)
    md = _markdown(report)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.json").write_text(json.dumps(report, indent=1))
    Path(f"{a.out}.md").write_text(md)
    sys.stdout.write(md)
    print(f"\n{a.out}.json, {a.out}.md")


if __name__ == "__main__":
    main()
