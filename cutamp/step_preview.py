"""Connect the searched foot commands to supported pick/place trajectories in the learned model."""

from copy import copy, deepcopy
from dataclasses import replace

import torch
from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo.kinematics import Kinematics
from curobo.types import JointState, Pose, ToolPoseCriteria

from cutamp import motion_solver as ms
from cutamp.com_polygon_cost import physical_support_centroid
from cutamp.robots.t1 import MOTION_INTERPOLATION_DT, get_t1_kinematics, t1_curobo_cfg
from cutamp.step_ik import _Kinematics
from cutamp.step_refinement import _check_deadline, landing_constraint_rows
from cutamp.step_validation import (
    StepStateAdapter,
    SteppedMotion,
    boundary_errors,
    compatible_boundary,
    validate_step_geometry,
)
from cutamp.supported_posture import fit_centered_posture, set_stance_base_limits
from cutamp.utils.common import action_4dof_to_mat4x4, get_world_cfg, grasp_7dof_to_mat4x4


_HEAD = ("AAHead_yaw", "Head_pitch")


def _check_issued_landings(predictor, initial, order, commands, prediction):
    """Check the searched controller domain and the strict per-sole and composed landing limits again at issue."""
    if not prediction.tracking_ok.all():
        raise ValueError("Predicted step or composed stance exceeds the 2 cm or 1 degree landing limit")
    before = predictor.model.feet(initial.tensors[:6])[0]
    rows, phase_ok, _ = landing_constraint_rows(predictor, before, order, commands, prediction)
    if not phase_ok.all() or not (torch.cat([v.flatten() for v in rows.values()]) <= 0).all():
        raise ValueError("Reissued commands violate the controller domain, phase or landing limits")


class _HandoffAdapter(StepStateAdapter):
    """Carry the supported planner's two locked head angles into its next controller issue state."""

    def __init__(self, predictor, model):
        super().__init__(predictor, model)
        self.head = None
        self.columns = [self.names.index(n) for n in _HEAD]

    def physical(self, q, qd=None):
        state = super().physical(q, qd)
        if self.head is not None:
            state[4][..., self.columns] = self.head
        return state


class PredictedSteppedMotion(SteppedMotion):
    """One model-only attempt. All mutable FK and obstacle state belongs to this attempted trajectory."""

    direct_dt = MOTION_INTERPOLATION_DT

    def __init__(self, skeleton, particle, inputs, adapter, world, deadline):
        private_world = copy(world)
        private_world._site_kinematics = {}
        super().__init__(skeleton, inputs, _HandoffAdapter(adapter.predictor, world.model), private_world, None)
        self.neutral_world = world
        self.skeleton = skeleton
        self.bindings = particle
        self.deadline = deadline
        self.charts = t1_curobo_cfg(world.model)
        self.supported_cfg = self.charts["robot_cfg"]["kinematics"]
        self.replayed = {}
        self.physical_cfg = None
        self._fallback = None
        self.stance = self.adapter.predictor.model.feet(self.adapter.physical(world.q_init[None]))[0][0]

    def get_planner(self):
        world = self.world
        self.scene_collision_checker = SceneCollision.from_config(
            SceneCollisionCfg(device_cfg=world.device_cfg, scene_model=get_world_cfg(world.env, include_movables=True))
        )
        self._install_collision_kinematics()
        return self

    def _install_collision_kinematics(self):
        # Motion collision FK has its own workspace and writable attachment slots. The
        # rest of its robot geometry is the exact immutable geometry of this stance.
        cfg = self.world.kinematics.config
        params = copy(cfg.kinematics_config)
        params.link_spheres = params.link_spheres.clone()
        self.kinematics = Kinematics(replace(cfg, kinematics_config=params), compute_com=True)
        self.attachment_manager = AttachmentManager(self.kinematics, self.scene_collision_checker, self.world.device_cfg)

    def set_support_stance(self, stance):
        self.stance = stance
        if self._fallback is not None:
            self._fallback.set_support_stance(stance)

    def fallback_planner(self, object_poses, held):
        """Build trajopt on demand with the exact head, foot bounds, scene and attachments checked by the direct path."""
        if self._fallback is None:
            self._fallback = ms.build_motion_planner(self.world, initial_soles=self.stance, robot_cfg=self.physical_cfg)
            if self.physical_cfg is not None:
                self._fallback.update_tool_pose_criteria({
                    f: ToolPoseCriteria.track_position_and_orientation(non_terminal_scale=self.sole_tracking_scale)
                    for f in self.world.model.sole_frames
                })
        planner = self._fallback
        for name, pose in object_poses.items():
            planner.scene_collision_checker.update_obstacle_pose(
                name=name, w_obj_pose=Pose.from_matrix(pose.reshape(1, 4, 4))
            )
            planner.scene_collision_checker.enable_obstacle(name=name, enable=True)
        for name, _ in held.values():
            planner.scene_collision_checker.enable_obstacle(name=name, enable=False)
        # The real attachment fit was done at grasp time. Copy its link-local sphere slots,
        # including any subsequently detached arm, into trajopt's FK buffers.
        planner.attachment_manager.kinematics_params.link_spheres.copy_(self.attachment_manager.kinematics_params.link_spheres)
        return planner

    def configure_stance(self, q, object_poses, held):
        world = self.world
        angles = self.last_state.tensors[4][0, self.adapter.columns]
        cfg = deepcopy(self.charts)
        kin_cfg = cfg["robot_cfg"]["kinematics"]
        kin_cfg["lock_joints"].update(dict(zip(_HEAD, angles.tolist())))
        set_stance_base_limits(kin_cfg, self.stance.tolist())
        self.supported_cfg = kin_cfg
        self.supported_kinematics = _Kinematics(q.device, kin_cfg)

        # The supported endpoints respect the operating envelope. The joined trajectory starts
        # at the controller's actual terminal state and is bounded by the source joint ranges.
        physical_cfg = deepcopy(cfg)
        physical = physical_cfg["robot_cfg"]["kinematics"]
        physical["urdf_path"] = str(world.model.asset_root / "t1_planner.urdf")
        physical["cspace"]["position_limit_clip"] = [0.0] * len(physical["cspace"]["joint_names"])
        kin = get_t1_kinematics(world.model, world.device_cfg, robot_cfg=physical_cfg)
        world.robot_container = replace(
            world.robot_container, kinematics=kin, joint_limits=kin.get_joint_limits().position
        )
        world._kinematics_jac = world._segment_torques = None
        world._site_kinematics.clear()
        self.physical_cfg = physical_cfg
        self.sole_tracking_scale = 24.0 if held else 8.0
        self._fallback = None
        old = self.attachment_manager.kinematics_params
        self._install_collision_kinematics()
        new = self.attachment_manager.kinematics_params
        for arm in held:
            link = ms._ATTACH_LINK_FOR_ARM[arm]
            new_ids = new.get_sphere_index_from_link_name(link)
            new.link_spheres[:, new_ids] = old.link_spheres[:, old.get_sphere_index_from_link_name(link)]
        self.adapter.head = angles

    def check_supported_start(self):
        _check_deadline(self.deadline)
        if self.last_state is not None:
            rates = self.last_state.tensors[5][0, self.adapter.columns]
            if not (rates.abs() <= 0.05).all():
                raise ValueError("Predicted head velocity cannot join supported motion")

    def _segment(self, trace, arrival, phase, **details):
        q, v = self.adapter.planner(trace)
        q, v = q[0].contiguous(), v[0].contiguous()
        arriving_v = arrival.velocity if arrival.velocity is not None else torch.zeros_like(arrival.position)
        if not compatible_boundary(boundary_errors(q[:1], v[:1], arrival.position, arriving_v)).all():
            raise ValueError(f"{phase} cannot join incoming position and velocity")
        plan = JointState(
            position=q,
            velocity=v,
            acceleration=torch.gradient(v, spacing=0.02, dim=0)[0].contiguous(),
            joint_names=list(self.world.model.joint_names),
        )
        return plan, dict(
            phase=phase,
            dt=0.02,
            source="frozen_policy_world_model",
            head=trace[4][0, :, self.adapter.columns].detach(),
            **details,
        )

    def step_segments(self, idx, last, object_poses, held):
        _check_deadline(self.deadline)
        gap, j = self.by_op[idx]
        initial, commands, saved, prep = self.inputs[gap]
        segments = []
        if j == 0:
            bound = {
                arm: (name, torch.linalg.inv(self.world.tool_from_ee[self.world.tool_frame_for_arm(arm)]) @ tf)
                for arm, (name, tf) in held.items()
            }
            velocity = last.velocity if last.velocity is not None else torch.zeros_like(last.position)
            observed = initial if gap == 0 and prep is None else None
            start = self.adapter.initialize(
                last.position,
                initial.dynamics,
                velocity,
                observed=observed,
                world=self.neutral_world,
                object_poses=object_poses,
                held=bound,
            )
            if not start.compatible.all():
                raise ValueError("Controller history cannot be issued from the actual arriving configuration")
            if start.preparation is not None:
                segments.append(self._segment(start.preparation.steps[0].trajectory, last, "model_preparation"))
                q, v = self.adapter.planner(start.state.tensors[:6])
                last = JointState(position=q, velocity=v, joint_names=list(self.world.model.joint_names))
            order = "".join(s.side for s in saved.steps)
            prediction = self.adapter.predictor.rollout(start.state, order, commands)
            _check_issued_landings(self.adapter.predictor, start.state, order, commands, prediction)
            geometry = validate_step_geometry(
                self.neutral_world, self.adapter, prediction, object_poses=object_poses, held=bound
            )
            if not geometry["valid"].all():
                raise ValueError("Predicted step trajectory violates collision or support constraints")
            self.replayed[gap] = (start.state, commands, prediction, start.preparation)
        start, commands, prediction, _ = self.replayed[gap]
        state = self.last_state if j else start
        step = prediction.steps[j]
        ticks = int(step.valid_ticks.sum())
        trace = tuple(torch.cat((x[:, None], y[:, :ticks]), dim=1) for x, y in zip(state.tensors, step.trajectory))
        plan, summary = self._segment(
            trace,
            last,
            "model_step_" + step.side,
            foot=step.side,
            command=commands[0, j].tolist(),
            xy_error_m=step.xy_error[0].tolist(),
            yaw_error_rad=step.yaw_error[0].tolist(),
            hold_ticks=ticks - int(step.clock[0]),
        )
        segments.append((plan, summary))
        self.last_state, self.stance = step.terminal, step.achieved[0]
        op = self.skeleton[idx]
        self.bindings[op.arg("q_land")] = plan.position[-1]
        self.bindings[op.arg("sole_to")] = self.stance["LR".index(step.side)]
        if j + 1 == len(prediction.steps):
            self.configure_stance(plan.position[-1:], object_poses, held)
        return segments

    def approach(self, idx, op, particle, object_poses, cfg, ik_tol):
        from cutamp.step_particles import _retract_on_stance

        self.check_supported_start()
        arm = op.operator.metadata.arm
        obj = (
            object_poses[op.arg("obj")]
            if op.operator.metadata.action_type == "pick"
            else action_4dof_to_mat4x4(particle[op.arg("placement")])
        )
        wrist = (
            obj
            @ grasp_7dof_to_mat4x4(particle[op.arg("grasp")])
            @ self.world.tool_from_ee[self.world.tool_frame_for_arm(arm)]
        ).reshape(4, 4)
        if cfg.com_center_weight:
            key = op.arg("q")
            try:
                pose = fit_centered_posture(
                    self.stance,
                    wrist,
                    particle[key],
                    arm,
                    cfg=self.supported_cfg,
                    other_arm_reference=self.occupied_arm,
                    centering_weight=6.0,
                )
                if (
                    self.endpoint_collides is None
                    or not self.endpoint_collides(particle[key].new_tensor(pose)[None]).any()
                ):
                    particle[key] = particle[key].new_tensor(pose)
            except ValueError:
                # Keep the searched endpoint available when the extra centering preference fails.
                pass
        approach = super().approach(idx, op, particle, object_poses, cfg, ik_tol)
        if self.last_state is not None and idx + 1 < len(self.skeleton):
            following = self.skeleton[idx + 1]
            if following.operator.metadata.action_type == "retract":
                rest, _ = self.adapter.planner(self.last_state.tensors[:6])
                ik = _retract_on_stance(
                    self.stance[None],
                    rest,
                    arm,
                    torch.stack((self.supported_kinematics.lower, self.supported_kinematics.upper)).to(rest),
                    other_arm_reference=self.occupied_arm,
                    kinematics=self.supported_kinematics,
                )
                if not ik.ok.all():
                    raise ValueError("Cannot retract on the reached support")
                particle[following.arg("q_retract")] = ik.q[0]
        return approach


@torch.no_grad()
def plan_predicted_step_motion(
    skeleton, row, predictions, adapter, world, config, *, timer, visualizer, ik_tol, deadline
):
    """Return and label a model-only preview after actual handoff, geometry and supported motion checks."""
    from cutamp.step_particles import _evidence

    bindings = dict(row.particle)
    motion = PredictedSteppedMotion(skeleton, bindings, predictions, adapter, world, deadline)
    try:
        plan, metrics = ms.solve_curobo(
            {"plan_skeleton": skeleton},
            bindings,
            motion.world,
            config,
            timer,
            None,
            step_motion=motion,
            get_planner=motion.get_planner,
            ik_tol=ik_tol,
        )
    except (ValueError, RuntimeError, TimeoutError) as exc:
        return None, {"motion_validation": "rejected", "reason": str(exc)}
    centers, margins = [], []
    for ik in motion.planned_endpoints.values():
        centers.append(float((ik.com[:, :2] - physical_support_centroid(ik.soles, world.model.sole_box)).norm()))
        margins.append(float(ik.margin[0]))
    connected = []
    for i, (idx, ik) in enumerate(motion.planned_endpoints.items()):
        gap = dict(
            op_idx=idx,
            center_distance_m=centers[i],
            margin_m=margins[i],
            com_xy=ik.com[0, :2].tolist(),
            achieved=ik.soles[0].tolist(),
            steps=[],
        )
        if i in motion.replayed:
            _, commands, pred, _ = motion.replayed[i]
            gap.update(
                commands=commands[0].tolist(),
                tracking_ok=bool(pred.tracking_ok[0]),
                final_xy_error_m=pred.final_xy_error[0].tolist(),
                final_yaw_error_deg=pred.final_yaw_error[0].rad2deg().tolist(),
            )
            gap["steps"] = [
                dict(foot=s.side, xy_error_m=s.xy_error[0].tolist(), yaw_error_deg=s.yaw_error[0].rad2deg().tolist())
                for s in pred.steps
            ]
        connected.append(gap)
    head = world.q_init.new_zeros(2)
    for entry in plan:
        if "step" in entry:
            series = entry["step"].pop("head")
            head = series[-1]
        else:
            series = head.expand(len(entry["plan"].position), -1)
        entry["head"] = series
    ms.play_curobo_motion(motion.world, plan, visualizer)
    row.particle.update(bindings)
    row.evidence = _evidence(motion.adapter, motion.replayed)
    return plan, {
        **metrics,
        "motion_validation": "model_predicted",
        "independent_motion": "untested",
        "com_center_distances_m": centers,
        "connected_motion_gaps": connected,
        "max_predicted_landing_xy_m": max(
            max(pred.final_xy_error.max().item(), *(s.xy_error.max().item() for s in pred.steps))
            for _, _, pred, _ in motion.replayed.values()
        ),
        "max_predicted_landing_yaw_deg": max(
            max(pred.final_yaw_error.rad2deg().max().item(), *(s.yaw_error.rad2deg().max().item() for s in pred.steps))
            for _, _, pred, _ in motion.replayed.values()
        ),
        "com_margins_m": margins,
        "best_com_margin": min(margins),
    }
