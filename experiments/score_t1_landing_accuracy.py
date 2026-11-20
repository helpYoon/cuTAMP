"""Score T1 support-relative step schedules on the frozen bfm_wm policy and dynamics.

Run with bfm_wm's Python, e.g. python experiments/score_t1_landing_accuracy.py
--bfm-root <bfm_wm>. All distances are in meters, world sole headings
are in radians, and reported angular errors are absolute wrapped degrees.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cutamp"))
from landing_accuracy import handoff_precision
from reachability import COMMANDS, ORDERS, apply_command, canonical_stance

sys.path.pop(0)  # bfm_wm's namespace package utils has a namesake inside cutamp.

WARMUP = 50
HOLD = 75
SETTLED_WINDOW = 25
CHECKPOINT = "logs/land_r4_d10_al2_s3/ckpt_004000.pt"
URDF = "assets/t1/t1_description/urdf/T1_29dof.urdf"
ACTUATORS = "assets/t1/mujoco/t1_actuators.npz"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def catalog(map_file):
    with np.load(map_file, allow_pickle=False) as raw:
        metadata = json.loads(raw["manifest"].item())["metadata"]
    home = metadata["home_soles"]
    starts = {"none": [([], np.asarray(home))]}
    schedules = []
    for order in ORDERS[1:]:
        sign = 1 if order[-1] == "L" else -1
        entries = []
        for prefix, stance in starts[order[:-1] or "none"]:
            for x, y, yaw in COMMANDS:
                cmd = [x, sign * y, sign * yaw]
                target = apply_command(stance, order[-1], cmd)
                if target is not None:
                    entries.append((prefix + [cmd], target))
        starts[order] = entries
        assert np.allclose([s for _, s in entries], metadata["order_endpoints"][order], atol=1e-7, rtol=0)
        schedules.extend({"order": order, "id": i, "commands": c} for i, (c, _) in enumerate(entries))
    return {"home_soles": home, "schedules": schedules}


class FootFK:
    """Both foot links in bfm_wm joint order, from the vendor URDF."""

    def __init__(self, root, device, rotations):
        self.rotations = rotations
        tree = ast.parse((root / "robot/t1.py").read_text())
        names = next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "T1_JOINT_NAMES"
        )
        joints = {j.find("child").get("link"): j for j in ET.parse(root / URDF).findall("joint")}
        chains = []
        for side in ("left", "right"):
            link = side + "_ankle_roll_link"
            chain = []
            while link != "trunk":
                j = joints[link]
                assert np.allclose(np.fromstring(j.find("origin").get("rpy"), sep=" "), 0)
                raw_name = j.get("name").removesuffix("_joint")
                name = "Waist" if raw_name == "waist_yaw" else raw_name.title()
                chain.append(
                    (
                        names.index(name),
                        np.fromstring(j.find("origin").get("xyz"), sep=" "),
                        np.fromstring(j.find("axis").get("xyz"), sep=" "),
                    )
                )
                link = j.find("parent").get("link")
            chains.append(chain[::-1])
        self.indices = torch.tensor([[j[0] for j in c] for c in chains], device=device)
        self.offset, self.axis = [
            torch.tensor(np.array([[j[k] for j in c] for c in chains]), dtype=torch.float32, device=device)
            for k in (1, 2)
        ]

    def forward(self, p, R, v, w, q, qd):
        p, R, v, w = (a.unsqueeze(1) for a in (p, R, v, w))
        joint_R = self.rotations.axis_angle_to_mat(self.axis, q[:, self.indices])
        rates = qd[:, self.indices]
        for i in range(self.indices.shape[1]):
            offset = self.rotations.mv3(R, self.offset[:, i])
            p = p + offset
            v = v + torch.cross(w.expand_as(offset), offset, dim=-1)
            w = w + self.rotations.mv3(R, self.axis[:, i]) * rates[:, :, i, None]
            R = R @ joint_R[:, :, i]
        return p, R, v, w


class Model:
    def __init__(self, root, checkpoint, device):
        sys.path.insert(0, str(root))
        # motion and mdp import Kinematics for annotations. The default class loads
        # a simulator; this isolated command supplies the vendor-URDF implementation.
        kin_module = ModuleType("robot.kinematics")
        kin_module.Kinematics = FootFK
        sys.modules[kin_module.__name__] = kin_module
        from src import landing
        from src.mdp import obs_dim
        from src.motion import CONTROL_DT
        from src.networks import Policy, WorldModel
        from utils import rotations
        from utils.export_landing import LandNet

        ck = torch.load(checkpoint, weights_only=True, map_location="cpu")
        args = ck["args"]
        self.device = device
        self.landing = landing
        self.rotations = rotations
        self.hist = args["hist"]
        self.training_window = args["wm_window"]
        with np.load(root / ACTUATORS) as data:
            self.q0 = torch.as_tensor(data["default_q"].copy(), dtype=torch.float32, device=device)
        J = len(self.q0)
        self.kin = FootFK(root, device, rotations)

        def widths(name):
            return [int(s) for s in args[name].split(",")]

        pi = Policy(
            obs_dim(J),
            args["goal_frames"] * (15 + 2 * J),
            J,
            hist=self.hist,
            enc_hidden=widths("enc_hidden"),
            dec_hidden=widths("dec_hidden"),
            vel=True,
            land=True,
        )
        self.wm = WorldModel(J, 38, 2, widths("wm_hidden"), CONTROL_DT)
        pi.load_state_dict(ck["pi"])
        self.wm.load_state_dict(ck["wm"])
        self.wm.eval().requires_grad_(False).to(device)
        self.policy = (
            LandNet(pi, self.kin, [0, 1], self.q0, landing.LandingCfg(), 20.0).eval().requires_grad_(False).to(device)
        )

    def observation(self, physical, previous):
        _, R, _, w, q, qd = physical
        return torch.cat(
            (q - self.q0, qd, self.rotations.mv3(R.transpose(-1, -2), w), -R[:, 2, :], previous[:, : len(self.q0)]), -1
        )

    def pose(self, physical):
        return self.landing.soles(*self.kin.forward(*physical))["pose"]

    def reset(self, count):
        def zero(*shape):
            return torch.zeros(*shape, device=self.device)

        joint = self.q0.expand(count, -1)
        pos = zero(count, 3)
        pos[:, 2] = 0.688
        physical = (
            pos,
            torch.eye(3, device=self.device).repeat(count, 1, 1),
            zero(count, 3),
            zero(count, 3),
            joint,
            torch.zeros_like(joint),
        )
        previous = zero(count, 2 * len(self.q0))
        history = self.observation(physical, previous)[:, None].repeat(1, self.hist, 1)
        return (*physical, previous, history, zero(count, 8))

    def tick(self, trajectory, cmd, context):
        *physical, previous, history, state = trajectory
        action, next_state = self.policy(history, cmd, state)
        physical = self.wm.step(physical, action, previous, context)
        previous = torch.cat((action, previous[:, : len(self.q0)]), -1)
        history = torch.cat((history[:, 1:], self.observation(physical, previous)[:, None]), 1)
        return (*physical, previous, history, next_state)


def score_snapshot(achieved, nominal, issued, foot, command, joint_rms, root_speed):
    def errors(reference):
        dif = achieved - reference
        xy = np.linalg.norm(dif[:, :2], axis=-1)
        yaw = np.abs(np.rad2deg(np.arctan2(np.sin(dif[:, 2]), np.cos(dif[:, 2]))))
        return xy, yaw

    xy, yaw = errors(nominal)
    issued_xy, issued_yaw = errors(issued)
    within = (xy < 0.02) & (yaw < 1.0)
    support = achieved[1 - foot]
    d = achieved[foot, :2] - support[:2]
    cs, sn = np.cos(support[2]), np.sin(support[2])
    local_xy = np.array([cs * d[0] + sn * d[1], -sn * d[0] + cs * d[1]])
    local_dyaw = achieved[foot, 2] - support[2] - command[2]
    return {
        "achieved_soles": achieved.tolist(),
        "nominal_xy_error_m": xy.tolist(),
        "nominal_yaw_error_deg": yaw.tolist(),
        "issued_xy_error_m": issued_xy.tolist(),
        "issued_yaw_error_deg": issued_yaw.tolist(),
        "local_xy_error_m": float(np.linalg.norm(local_xy - command[:2])),
        "local_yaw_error_deg": float(abs(np.rad2deg(np.arctan2(np.sin(local_dyaw), np.cos(local_dyaw))))),
        "swing_pass": bool(within[foot]),
        "stance_pass": bool(within.all()),
        "root_speed_m_s": float(root_speed),
        "joint_speed_rms_rad_s": float(joint_rms),
    }


def warmed(model, home, lags):
    device = model.device
    count = len(lags)
    context = torch.zeros(count, 38, device=device)
    context[:, 0] = (1.0 - 0.75) / 0.45
    context[:, -5:] = torch.tensor([(lag - 5.0) / 3 for lag in lags], device=device)[:, None]
    empty = torch.zeros(count, 5, device=device)
    traj = model.reset(count)
    for _ in range(WARMUP):
        traj = model.tick(traj, empty, context)
    # Register the warmed stance to the map's world frame by one rigid SE(2)
    # transform. Retain the achieved width, stagger and relative-yaw discrepancy.
    measured = model.pose(traj[:6])
    as_pose = lambda values: torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)
    start_frame = as_pose([canonical_stance(p)[1] for p in measured.cpu().numpy()])
    home_frame = as_pose(canonical_stance(home)[1])

    def world(pose):
        return model.landing.compose(home_frame, model.landing.relative(pose, start_frame[:, None]))

    return traj, context, empty, world


def hold_control(model, home, lags):
    traj, context, empty, world = warmed(model, home, lags)
    start = world(model.pose(traj[:6])).cpu().numpy()
    samples = []
    for tick in range(251):
        if tick in (0, 125, 250):
            pose = world(model.pose(traj[:6])).cpu().numpy()
            for lag, initial, current in zip(lags, start, pose):
                row = {"lag": lag, "ticks_after_warmup": tick, "soles": current.tolist()}
                for key, reference in (("nominal", home), ("warm_start", initial)):
                    delta = current - reference
                    row[key + "_xy_error_m"] = np.linalg.norm(delta[:, :2], axis=1).tolist()
                    row[key + "_yaw_error_deg"] = np.abs(
                        np.rad2deg(np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2])))
                    ).tolist()
                samples.append(row)
        if tick < 250:
            traj = model.tick(traj, empty, context)
    return samples


def score_batch(model, home, entries):
    device = model.device
    count = len(entries)
    batch_ids = torch.arange(count, device=device)
    tensor = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)
    to_numpy = lambda a: a.detach().cpu().numpy()
    traj, context, empty, world = warmed(model, home, [lag for _, lag in entries])
    measured = to_numpy(world(model.pose(traj[:6])))
    trials = [{"lag": lag, "start_soles": measured[i].tolist(), "stages": []} for i, (_, lag) in enumerate(entries)]
    expected = np.repeat(np.asarray(home)[None], count, axis=0)
    num_steps = len(entries[0][0]["order"])
    for stage in range(num_steps):
        hold_ticks = HOLD if stage == num_steps - 1 else 0
        foot = torch.as_tensor([int(e["order"][stage] == "R") for e, lag in entries], device=device)
        commands = tensor([e["commands"][stage] for e, lag in entries])
        before = world(model.pose(traj[:6]))
        issued = before.clone()
        issued[batch_ids, foot] = model.landing.compose(before[batch_ids, 1 - foot], commands)
        before, issued = to_numpy(before), to_numpy(issued)
        for i, (e, _) in enumerate(entries):
            side, command = e["order"][stage], e["commands"][stage]
            expected[i] = apply_command(expected[i], side, command)
            trials[i]["stages"].append(
                {
                    "foot": side,
                    "hold_ticks": hold_ticks,
                    "issue_legal": apply_command(before[i], side, command) is not None,
                    "initial_soles": before[i].tolist(),
                    "nominal_soles": expected[i].tolist(),
                    "issued_world_soles": issued[i].tolist(),
                }
            )
        active = empty.clone()
        active[:, 0] = 1.0
        active[:, 1] = foot
        active[:, 2:] = commands
        traj = model.tick(traj, active, context)
        clocks = to_numpy(traj[-1][:, 2]).astype(int) + 1
        assert np.all((clocks >= 40) & (clocks <= 62))
        for i, c in enumerate(clocks):
            trials[i]["stages"][-1]["clock_ticks"] = int(c)
        handoff = tuple(torch.empty_like(x) for x in traj)
        foot_np = to_numpy(foot)
        handoff_tick = torch.as_tensor(clocks + hold_ticks, device=device)
        max_root_speed = torch.zeros_like(foot, dtype=commands.dtype)
        max_joint_rms = torch.zeros_like(max_root_speed)
        for t in range(2, int(clocks.max()) + hold_ticks + 1):
            traj = model.tick(traj, empty, context)
            if hold_ticks and t > int(clocks.min()) + hold_ticks - SETTLED_WINDOW:
                final_window = (handoff_tick - SETTLED_WINDOW < t) & (t <= handoff_tick)
                root_speed_now = traj[2].norm(dim=-1)
                joint_rms_now = traj[5].square().mean(-1).sqrt()
                max_root_speed = torch.where(
                    final_window, torch.maximum(max_root_speed, root_speed_now), max_root_speed
                )
                max_joint_rms = torch.where(final_window, torch.maximum(max_joint_rms, joint_rms_now), max_joint_rms)
            for name, times in (("deadline", clocks), ("handoff", clocks + hold_ticks)):
                ids = np.flatnonzero(times == t)
                if not len(ids):
                    continue
                gpu_ids = torch.as_tensor(ids, device=device)
                achieved = to_numpy(world(model.pose(traj[:6]))[gpu_ids])
                root_speed = to_numpy(traj[2][gpu_ids].norm(dim=-1))
                joint_rms = to_numpy(traj[5][gpu_ids].square().mean(-1).sqrt())
                if name == "handoff":
                    for dest, src in zip(handoff, traj):
                        dest[gpu_ids] = src[gpu_ids]
                for j, i in enumerate(ids):
                    result = trials[i]["stages"][-1]
                    result[name] = score_snapshot(
                        achieved[j],
                        expected[i],
                        issued[i],
                        foot_np[i],
                        entries[i][0]["commands"][stage],
                        joint_rms[j],
                        root_speed[j],
                    )
        for trial, v, qd in zip(trials, to_numpy(max_root_speed), to_numpy(max_joint_rms)):
            trial["stages"][-1]["settling"] = {
                "max_root_speed_m_s": float(v),
                "max_joint_speed_rms_rad_s": float(qd),
                "settled": bool(v < 0.02 and qd < 0.01),
            } if hold_ticks else None
        traj = handoff
    for tr, (entry, _) in zip(trials, entries):
        tr["final_stance_pass"] = tr["stages"][-1]["handoff"]["stance_pass"]
        tr["all_handoffs_pass"] = handoff_precision(tr["stages"], entry["order"])
        if tr["all_handoffs_pass"] is None:
            raise ValueError(f"Incomplete or nonfinite handoff evidence for {entry['order']}")
        tr["all_swings_pass"] = all(s["issue_legal"] and s["handoff"]["swing_pass"] for s in tr["stages"])
        tr["all_settled"] = tr["stages"][0]["settling"]["settled"] if num_steps == 1 else None
    return trials


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--bfm-root", type=Path, default=Path.home() / "bfm_wm")
    cli.add_argument("--checkpoint", type=Path)
    cli.add_argument("--map-file", type=Path, default=REPO / "data/t1-reachability/t1-hand-map.npz")
    cli.add_argument("--schedule-file", type=Path)
    cli.add_argument(
        "--hold-control-output",
        type=Path,
        help="Optional matched zero-step control at 0, 125, and 250 ticks after warm-up.",
    )
    cli.add_argument("--output", type=Path, default=REPO / "data/t1-reachability/landing-accuracy/home-model.json")
    cli.add_argument("--lags", type=int, nargs="+", default=[2, 5, 8], choices=range(2, 9))
    cli.add_argument("--batch-size", type=int, default=240)
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = cli.parse_args()
    if args.batch_size <= 0 or len(set(args.lags)) != len(args.lags):
        cli.error("batch size must be positive; list each delay only once")
    root = args.bfm_root.resolve()
    checkpoint = args.checkpoint or root / CHECKPOINT
    data = json.loads(args.schedule_file.read_text()) if args.schedule_file else catalog(args.map_file)
    home, schedules = data["home_soles"], data["schedules"]
    ids = {(e["order"], e["id"]) for e in schedules}
    if not schedules or len(ids) != len(schedules):
        raise ValueError("expected nonempty schedules with unique (order,id)")
    for e in schedules:
        order, cmds = e["order"], e["commands"]
        if order not in ORDERS[1:] or len(cmds) != len(order):
            raise ValueError(f"invalid schedule {e}")
        nominal = np.asarray(home)
        for foot, c in zip(order, cmds):
            nominal = apply_command(nominal, foot, c)
            if nominal is None:
                raise ValueError(f"command outside the planner envelope: {e}")
        e["trials"] = []
    model = Model(root, checkpoint, args.device)
    with torch.inference_mode():
        control_samples = hold_control(model, home, args.lags) if args.hold_control_output else None
        for length in (1, 2):
            cases = [(e, d) for e in schedules if len(e["order"]) == length for d in args.lags]
            for i in range(0, len(cases), args.batch_size):
                batch = cases[i : i + args.batch_size]
                print(f"{length}-step rows {i + 1}–{i + len(batch)} of {len(cases)}", flush=True)
                for (entry, _), trial in zip(batch, score_batch(model, home, batch)):
                    entry["trials"].append(trial)
    excluded = {"mujoco", "mjlab", "warp", "mujoco_warp"}
    assert not excluded.intersection(k.split(".")[0] for k in sys.modules)
    files = [
        URDF,
        ACTUATORS,
        "src/mdp.py",
        "src/motion.py",
        "src/networks.py",
        "src/landing.py",
        "src/dr.py",
        "utils/export_landing.py",
        "utils/rotations.py",
        "robot/t1.py",
    ]
    data["provenance"] = {
        "model_only": True,
        "torch_version": str(torch.__version__),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha(checkpoint),
        "bfm_git": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: sha(root / p) for p in files},
        "scorer_sha256": sha(Path(__file__)),
        "precision_sha256": sha(REPO / "cutamp/landing_accuracy.py"),
        "catalog_source": str((args.schedule_file or args.map_file).resolve()),
        "catalog_sha256": sha(args.schedule_file or args.map_file),
        "lags": args.lags,
        "training_window_ticks": model.training_window,
        "initialization": "Repeated default upright observations, zero velocities and previous actions, root z=.688 m; then closed-loop HOLD warm-up.",
        "registration": "Warm-start measured sole midpoint and mean heading rigidly aligned to nominal home; relative foot shape preserved.",
        "context": "All five actuator groups use the reported constant delay in physics substeps. Friction 1.0, zero encoder and torso CoM offsets, zero sensor noise, no payload.",
        "warmup_ticks": WARMUP,
        "hold_ticks": HOLD,
        "inter_step_hold_ticks": 0,
        "settled_window_ticks": SETTLED_WINDOW,
        "settling": "Separately scored: maximum joint-speed RMS < 0.01 rad/s and root speed < 0.02 m/s throughout the final 25 HOLD ticks.",
        "control_hz": 50,
        "xy_threshold_m": 0.02,
        "yaw_threshold_deg": 1.0,
        "acceptance": "Each command must be legal at issue. Each sole must be within strict tracking tolerances of its issued target at every handoff and its original nominal target at the final handoff.",
        "tracking": "issued_* errors vs frozen actual planted sole at issue; local_* errors vs opposite achieved sole at measurement.",
        "units": "sole poses [x m,y m,yaw rad], per-sole lists [left,right]",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as out:
        json.dump(data, out, allow_nan=False, separators=(",", ":"))
        out.write("\n")
    Path(out.name).replace(args.output)
    if args.hold_control_output:
        args.hold_control_output.parent.mkdir(parents=True, exist_ok=True)
        report = {"home_soles": home, "samples": control_samples, "provenance": data["provenance"]}
        args.hold_control_output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    print(f"Saved {len(schedules)} schedules to {args.output}", flush=True)


if __name__ == "__main__":
    main()
