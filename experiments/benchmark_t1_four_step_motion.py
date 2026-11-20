"""Generate and time a connected model-predicted LR/pick/LR/place motion on the local GPU.

python -m experiments.benchmark_t1_four_step_motion --output /tmp/t1-four-step --runs 3

Each request searches both foot-target pairs and solves fresh manipulation postures and trajectories.
The carried object participates in geometry checks. Seed-3 predicts unloaded robot dynamics.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
import json
from pathlib import Path
from time import perf_counter

PROCESS_START = perf_counter()

import numpy as np
import torch
from curobo.scene import Cuboid
from curobo.types import JointState, ToolPoseCriteria

from cutamp import motion_solver as ms
from cutamp.algorithm import setup_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.envs import TAMPEnvironment
from cutamp.landing_accuracy import LANDING_BUNDLE
from cutamp.landing_predictor import LandingPredictor
from cutamp.particle_initialization import IKTol
from cutamp.robots import load_t1_container
from cutamp.robots.t1 import REAL_LEGS, t1_curobo_cfg
from cutamp.scripts.utils import default_constraint_to_tol
from cutamp.step_ik import _kinematics
from cutamp.step_particles import _retract_on_stance
from cutamp.step_proposals import rank_landing_commands
from cutamp.step_validation import (
    SteppedMotion,
    StepStateAdapter,
    boundary_errors,
    compatible_boundary,
    model_step_record,
)
from cutamp.t1_domain import all_t1_fluents, all_t1_operators, all_t1_step_operators
from cutamp.utils.common import action_4dof_to_mat4x4, grasp_7dof_to_mat4x4
from cutamp.supported_posture import fit_centered_posture, set_stance_base_limits

ROOT = Path(__file__).resolve().parents[1]
HEAD = ("AAHead_yaw", "Head_pitch")


@contextmanager
def robot_config(config):
    # The worktree's FK reads its model YAML through a process-local cache. This benchmark
    # runs serially and restores the base model even when a candidate fails.
    active = t1_curobo_cfg(REAL_LEGS)
    old = active.copy()
    active.update(config)
    _kinematics.cache_clear()
    try:
        yield
    finally:
        active.update(old)
        _kinematics.cache_clear()


def make_world_config(base, head, soles, *, physical):
    cfg = deepcopy(base)
    k = cfg["robot_cfg"]["kinematics"]
    k["lock_joints"].update(head)
    set_stance_base_limits(k, soles)
    if physical:
        # The learned landing may lie outside the narrower supported IK posture envelope.
        k["urdf_path"] = str(REAL_LEGS.asset_root / "t1_planner.urdf")
        k["cspace"]["position_limit_clip"] = [0.0] * len(k["cspace"]["joint_names"])
    return cfg


class IncomingHead(StepStateAdapter):
    def __init__(self, predictor, head):
        super().__init__(predictor)
        self.head = head

    def physical(self, q, qd=None):
        state = super().physical(q, qd)
        for name, angle in self.head.items():
            state[4][..., self.names.index(name)] = angle
        return state


class ConnectedSteps(SteppedMotion):
    """Frozen policy/world-model predictions joined to checked supported segments for this benchmark."""

    def __init__(self, ops, world, predictor, start, bindings, fixture, counts):
        super().__init__(ops, {0: None, 1: None}, StepStateAdapter(predictor), world, validator=None)
        self.ops = ops
        self.start, self.bindings, self.fixture, self.counts = start, bindings, fixture, counts
        self.base_cfg = deepcopy(t1_curobo_cfg(REAL_LEGS))
        self.neutral_robot = world.robot_container
        self.rollouts, self.heads, self.configs, self.choices, self.diagnostics = {}, {}, {}, {}, []
        self.initial_states = {}
        self.preparation = None
        self.motion_planner = None

    def set_robot(self, robot):
        w = self.world
        w.robot_container = robot
        w._kinematics_jac = w._segment_torques = None
        w._site_kinematics.clear()

    def choose(self, gap, initial, poses, held):
        w = self.world
        key = "left_q2" if gap == 0 else "left_q5"
        target_obj = poses["part"] if gap == 0 else action_4dof_to_mat4x4(self.bindings["pose1"])
        wrist = (
            target_obj @ grasp_7dof_to_mat4x4(self.bindings["grasp1"]) @ w.tool_from_ee[w.tool_frame_for_arm("left")]
        )
        inp = self.fixture["gaps"][gap]
        t = perf_counter()
        choices = rank_landing_commands(
            w,
            self.adapter,
            initial,
            "LR",
            inp["commands"],
            wrist,
            inp["posture_seed"],
            "left",
            held=held,
            object_poses=poses,
            count=self.counts[gap],
            max_choices=1,
        )
        if not len(choices.commands):
            raise RuntimeError(f"No feasible landing from the actual incoming state for gap {gap}")
        # Use the exact row that passed tracking, phase, support and collision checks.
        command = choices.commands[:1]
        prediction = choices.prediction(0)
        self.initial_states[gap] = initial
        self.rollouts[gap] = prediction
        self.choices[gap] = command[0].tolist()
        head = {n: float(prediction.terminal.tensors[4][0, self.adapter.names.index(n)]) for n in HEAD}
        self.heads[gap] = head
        soles = prediction.steps[-1].achieved[0].tolist()
        cfg = make_world_config(self.base_cfg, head, soles, physical=False)
        self.configs[gap] = cfg
        pose = fit_centered_posture(
            soles,
            wrist.reshape(4, 4).tolist(),
            choices.postures[0].tolist(),
            "left",
            cfg=cfg,
            head=head,
            centering_weight=6.0,
        )
        self.bindings[key] = w.q_init.new_tensor(pose)
        self.diagnostics.append(
            {
                "gap": gap,
                "sample_count": self.counts[gap],
                "shortlisted": len(choices.commands),
                "selection_and_fit_s": perf_counter() - t,
                "wrist": wrist.reshape(4, 4).tolist(),
            }
        )

    def configure_motion(self, gap, held):
        w = self.world
        soles = self.rollouts[gap].steps[-1].achieved[0]
        physical = make_world_config(self.base_cfg, self.heads[gap], soles.tolist(), physical=True)
        old = self.motion_planner
        with robot_config(physical):
            self.set_robot(load_t1_container(w.model, device_cfg=w.device_cfg))
            planner = ms.build_motion_planner(w, initial_soles=soles)
        for arm, (obj, _) in held.items():
            # Activate the attachment on the new chart and transfer the exact collision spheres
            # fitted at grasp acquisition, including cuRobo's padding slots.
            js = JointState.from_position(self.bindings["left_q2"][None], joint_names=list(w.model.joint_names))
            ms._attach_object(planner, obj, js, arm)
            a, b = (p.attachment_manager.kinematics_params for p in (old, planner))
            link = ms._ATTACH_LINK_FOR_ARM[arm]
            src, dst = (k.get_sphere_index_from_link_name(link) for k in (a, b))
            b.link_spheres[:, dst] = a.link_spheres[:, src]
            if not torch.equal(b.link_spheres[:, dst], a.link_spheres[:, src]):
                raise RuntimeError("Carried object spheres changed across support charts")
        planner.update_tool_pose_criteria(
            {
                f: ToolPoseCriteria.track_position_and_orientation(non_terminal_scale=(8.0 if gap == 0 else 24.0))
                for f in w.model.sole_frames
            }
        )
        self.motion_planner = planner
        self.physical_cfg = physical

    def __getattr__(self, name):
        # solve_curobo keeps one planner handle while the support and locked head change.
        return getattr(self.motion_planner, name)

    def approach(self, idx, op, particle, object_poses, cfg, ik_tol):
        gap = 0 if op.operator.metadata.action_type == "pick" else 1
        with robot_config(self.configs[gap]):
            return super().approach(idx, op, particle, object_poses, cfg, ik_tol)

    def initial_velocity(self):
        return self.adapter.planner(self.start.tensors[:6])[1]

    def segment(self, physical, incoming, phase):
        p, v = self.adapter.planner(physical)
        q, qd = p[0].contiguous(), v[0].contiguous()
        if not compatible_boundary(boundary_errors(q[:1], qd[:1], incoming.position, incoming.velocity)).all():
            raise ValueError(f"{phase} does not join incoming position and velocity")
        js = JointState(
            position=q,
            velocity=qd,
            acceleration=torch.gradient(qd, spacing=0.02, dim=0)[0].contiguous(),
            joint_names=list(self.world.model.joint_names),
        )
        return js, {"phase": phase, "dt": 0.02, "source": "frozen_policy_world_model"}

    def step_segments(self, idx, last, poses, held):
        gap, j = self.by_op[idx]
        output = []
        if gap == 1 and j == 0:
            w = self.world
            bound = {
                arm: (obj, torch.linalg.inv(w.tool_from_ee[w.tool_frame_for_arm(arm)]) @ tf)
                for arm, (obj, tf) in held.items()
            }
            self.set_robot(self.neutral_robot)
            ad = IncomingHead(self.adapter.predictor, self.heads[0])
            with robot_config(self.base_cfg):
                prepared = ad.initialize(
                    last.position, self.start.dynamics, last.velocity, world=w, object_poses=poses, held=bound
                )
                if not prepared.compatible.all():
                    raise ValueError(f"Actual pick retract cannot issue steps: {prepared.diagnostics}")
                self.choose(1, prepared.state, poses, bound)
            init = prepared.state
            self.preparation = prepared.preparation
            prep = prepared.preparation.steps[0]
            output.append(self.segment(prep.trajectory, last, "model_preparation"))
            last = JointState(
                position=output[-1][0].position[-1:],
                velocity=output[-1][0].velocity[-1:],
                joint_names=list(w.model.joint_names),
            )
        else:
            init = self.start if gap == 0 and j == 0 else self.last_state
        s = self.rollouts[gap].steps[j]
        ticks = int(s.valid_ticks.sum())
        physical = tuple(torch.cat((a[:, None], v[:, :ticks]), dim=1) for a, v in zip(init.tensors, s.trajectory))
        output.append(self.segment(physical, last, "model_step_" + s.side))
        self.stance, self.last_state = s.achieved[0], s.terminal
        if gap == 1 and j == 1:
            self.configure_motion(1, held)
            t1_curobo_cfg(REAL_LEGS).update(self.physical_cfg)
            _kinematics.cache_clear()
        return output


def summarize(motion, entries, world, materialize):
    """Measure actual gripper-event poses; optionally serialize the sampled motion after timing."""
    from cutamp.com_polygon_cost import physical_support_centroid, physical_support_margin

    elapsed, frame = 0.0, 0
    segments, events = [], {}
    fields = {k: [] for k in ("t", "q", "head", "phase", "held_left", "held_right", "object_part")}
    object_pose = world.get_object_pose(world.movables[0]).cpu().numpy()
    last_head = motion.start.tensors[4][0, [motion.adapter.names.index(n) for n in HEAD]].cpu().numpy()
    step_index = 0
    approach_rise = {}
    for item in entries:
        js, phase, dt = item["plan"], item["phase"], item["dt"]
        q = ms._full_positions(world, js)
        n = len(q)
        op = motion.ops[item["op_idx"]].operator.metadata.action_type
        if op in ("pick", "place"):
            phase = op + "_" + phase
        times = elapsed + np.arange(n) * dt
        select = slice(0 if not segments else 1, None)
        end = frame + n - 1
        segments.append(
            dict(phase=phase, end=end, dt=dt, source=item.get("motion_source", item.get("step", {}).get("source")))
        )
        if materialize:
            if phase.startswith("model_step_"):
                gap, step = divmod(step_index, 2)
                s = motion.rollouts[gap].steps[step]
                heads = s.trajectory[4][0, : n - 1, [motion.adapter.names.index(x) for x in HEAD]].cpu().numpy()
                heads = np.vstack([last_head, heads])
                step_index += 1
            elif phase == "model_preparation":
                heads = (
                    motion.preparation.steps[0]
                    .trajectory[4][0, :, [motion.adapter.names.index(x) for x in HEAD]]
                    .cpu()
                    .numpy()
                )
            else:
                heads = np.broadcast_to(last_head, (n, 2))
            last_head = heads[-1]
            held = item.get("held_objs", {})
            obj = next((v[1].detach().cpu().numpy().reshape(-1, 4, 4) for v in held.values()), None)
            if obj is None:
                obj = np.broadcast_to(object_pose, (n, 4, 4))
            object_pose = obj[-1].copy()
            for key, v in dict(
                t=times,
                q=q.detach().cpu().numpy(),
                phase=np.repeat(phase, n),
                head=heads,
                held_left=np.repeat("left" in held, n),
                held_right=np.repeat("right" in held, n),
                object_part=obj,
            ).items():
                fields[key].append(v[select])
        if phase == "pick_approach":
            kin = _kinematics(q.device)
            for arm in ("left", "right"):
                fk, _ = kin.fk(q.double(), arm, False)
                height = fk.reshape(n, 3, 12)[:, -1, 2]
                approach_rise[arm] = 100 * float(height.max() - height[0])
        if item["gripper"]:
            gap = 0 if op == "pick" else 1
            with robot_config(motion.configs[gap]):
                kin = _kinematics(q.device)
                pose, com = kin.fk(q[-1:].double(), "left", False)
                pose = pose.reshape(1, 3, 12)
                r = pose[:, :2, 3:].reshape(1, 2, 3, 3).mT
                soles = torch.cat((pose[:, :2, :2], torch.atan2(r[:, :, 1, 0], r[:, :, 0, 0])[..., None]), dim=-1)
                offset = (com[:, :2] - physical_support_centroid(soles, kin.box)).norm()
                idle = world.model.right_arm
                error = (q[-1, idle] - q.new_tensor(world.model.home[idle])).abs().max().rad2deg()
                margin = physical_support_margin(com[:, :2], soles, kin.box)
                if not (error < 0.1 and margin >= 0.05):
                    raise ValueError("Nominal arm or support margin failed on this fixture")
                events[op] = dict(
                    frame=end,
                    time_s=float(times[-1]),
                    centroid_distance_cm=float(offset) * 100,
                    margin_cm=float(margin) * 100,
                    max_idle_joint_error_deg=float(error),
                )
        elapsed, frame = times[-1], end
    assert set(events) == {"pick", "place"} and segments[-1]["phase"] == "place_back_out"
    return events, segments, approach_rise, {k: np.concatenate(v) for k, v in fields.items()} if materialize else {}


def generate(fixture, predictor, config, out, counts):
    part, floor, pad = (Cuboid(**obj) for obj in fixture["scene"])
    on = next(f for f in all_t1_fluents if f.name == "On")
    env = TAMPEnvironment(
        "two_steps_floor_transfer",
        [part],
        [floor, pad],
        {"Movable": [part], "Surface": [pad]},
        frozenset([on.ground("part", "pad")]),
    )
    dyn = torch.zeros(1, 38, device=predictor.device)
    dyn[:, 0], dyn[:, -5:] = 0.25 / 0.45, -1.0
    initial = predictor.standing_start(dyn)
    adapter = StepStateAdapter(predictor)
    q, v = adapter.planner(initial.tensors[:6])
    _, vis, timer, w = setup_cutamp(env, config, q[0].tolist(), experiment_id=out.name)
    start = adapter.initialize(w.q_init[None], dyn, v, observed=initial, world=w)
    if not start.compatible.all():
        raise RuntimeError("Initial robot and controller history mismatch")
    ops_by_name = {op.name: op for op in (*all_t1_operators, *all_t1_step_operators)}
    ops = []
    for text in fixture["skeleton"]:
        name, values = text[:-1].split("(")
        op = ops_by_name[name]
        ops.append(op.ground(dict(zip((p.name for p in op.parameters), values.split(", ")))))
    bindings = dict(left_q0=q[0], grasp1=q.new_tensor(fixture["grasp"]), pose1=q.new_tensor(fixture["placement"]))
    st = ConnectedSteps(ops, w, predictor, initial, bindings, fixture, counts)
    st.choose(0, initial, {"part": w.get_object_pose(part)}, {})
    pred = st.rollouts[0]
    arrived = adapter.planner(pred.terminal.tensors[:6])[0]
    bindings["left_q3"] = _retract_on_stance(pred.steps[-1].achieved, arrived, "left", w.joint_limits).q[0]
    st.configure_motion(0, {})
    # solve_curobo copies/updates planned keyframes as it reaches each action; place is fitted at the actual handoff.
    tol = IKTol.from_checker(ConstraintChecker(default_constraint_to_tol.copy()))
    torch.manual_seed(641)
    with robot_config(st.physical_cfg):
        entries, metrics = ms.solve_curobo(
            {"plan_skeleton": ops}, bindings, w, config, timer, vis, ik_tol=tol, step_motion=st, get_planner=lambda: st
        )
    steps = {}
    for k, pred in st.rollouts.items():
        xy = max(float(s.xy_error.max()) for s in pred.steps)
        yaw = max(float(s.yaw_error.max()) for s in pred.steps)
        steps[("pick", "place")[k]] = dict(
            commands=st.choices[k],
            max_xy_cm=100 * max(xy, float(pred.final_xy_error.max())),
            max_yaw_deg=float(np.rad2deg(max(yaw, float(pred.final_yaw_error.max())))),
            hold_ticks=[int(s.valid_ticks.sum() - s.clock.item()) for s in pred.steps],
        )
    if any(g["hold_ticks"] != [0, 75] or g["max_xy_cm"] >= 2 or g["max_yaw_deg"] >= 1 for g in steps.values()):
        raise ValueError("Incomplete four-step tracking evidence")
    return st, entries, w, dict(motion=metrics, steps=steps, candidates=st.diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--pick-count", type=int, default=192)
    parser.add_argument("--place-count", type=int, default=128)
    parser.add_argument("--bundle", type=Path, default=LANDING_BUNDLE)
    args = parser.parse_args()
    if min(args.pick_count, args.place_count) < 1:
        parser.error("Candidate counts must be positive")
    if args.runs < 1:
        parser.error("--runs must be positive")
    torch.set_num_threads(1)
    fixture = json.loads((ROOT / "cutamp/tests/fixtures/t1_four_step_transfer.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    predictor = LandingPredictor(args.bundle)
    torch.cuda.synchronize()
    setup_s = perf_counter() - start
    config = TAMPConfiguration(
        step_mode="model",
        curobo_plan=True,
        enable_visualizer=False,
        experiment_root=str(args.output),
        landing_bundle=str(args.bundle),
        step_lag=2,
    )
    rows = []
    for i in range(args.runs):
        start = perf_counter()
        st, entries, w, result = generate(
            fixture, predictor, config, args.output / f"run-{i}", (args.pick_count, args.place_count)
        )
        torch.cuda.synchronize()
        result["planning_s"] = perf_counter() - start
        if i == 0:
            result["process_to_first_plan_s"] = perf_counter() - PROCESS_START
        events, segments, approach_rise, arrays = summarize(st, entries, w, i == args.runs - 1)
        result["events"] = events
        result["pick_approach_wrist_rise_cm"] = approach_rise
        rows.append(result)
        print(
            f"Request {i + 1}: {result['planning_s']:.3f} s, pick {events['pick']['centroid_distance_cm']:.2f} cm, "
            f"place {events['place']['centroid_distance_cm']:.2f} cm",
            flush=True,
        )
        if arrays:
            records = []
            for gap, prediction in st.rollouts.items():
                previous = st.initial_states[gap]
                for j, step in enumerate(prediction.steps):
                    command = previous.dynamics.new_tensor(st.choices[gap][j])[None]
                    records.append(
                        model_step_record(
                            st.adapter,
                            previous,
                            step,
                            command,
                            preparation=st.preparation if (gap, j) == (1, 0) else None,
                        )
                    )
                    previous = step.terminal
            with gzip.open(args.output / "controller-traces.json.gz", "wt") as file:
                json.dump(records, file, separators=(",", ":"))
            np.savez_compressed(args.output / "trajectory.npz", **arrays)
            meta = dict(
                segments=segments,
                events={k: v["frame"] for k, v in events.items()},
                joint_names=list(w.model.joint_names),
                metrics=result["motion"],
            )
            (args.output / "trajectory-segments.json").write_text(json.dumps(meta, indent=2))
            (args.output / "scene.json").write_text(json.dumps(fixture["scene"], indent=2))
        del st, entries, w
    report = dict(
        device=torch.cuda.get_device_name(),
        model_startup_s=setup_s,
        scope="Fresh fixed-skeleton request from saved HOLD-era geometric seeds, including scene setup, search, "
        "actual step rollout and supported motion. Excludes process startup, offline audit and export.",
        runs=rows,
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
