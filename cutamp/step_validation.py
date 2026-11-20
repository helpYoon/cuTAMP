"""Map the T1 planner and frozen controller to one floor/world frame and check state handoffs."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import roma
import torch
from torch import Tensor

from cutamp.landing_predictor import LandingState, _hold_trace, near_rest, prediction_precision
from cutamp.robots.t1 import REAL_LEGS


@dataclass
class StepStart:
    state: LandingState
    compatible: Tensor
    diagnostics: dict
    preparation: object | None = None


def _angular_axes(angles):
    """World axes of the planner's serial pitch(Y), roll(X), yaw(Z) joints."""
    pitch, roll, _ = angles.unbind(-1)
    zero, cp, sp, cr, sr = torch.zeros_like(roll), pitch.cos(), pitch.sin(), roll.cos(), roll.sin()
    return torch.stack(
        (
            torch.stack((zero, zero + 1, zero), -1),
            torch.stack((cp, zero, -sp), -1),
            torch.stack((sp * cr, -sr, cp * cr), -1),
        ),
        -1,
    )


def boundary_errors(current, velocity, expected, expected_velocity):
    """Per-row start/finish discrepancies in the planner's coordinates."""
    delta = current - expected
    turn = torch.atan2(delta[..., 3:].sin(), delta[..., 3:].cos())
    return {
        "base_position_m": delta[..., :3].norm(dim=-1),
        "angle_rad": turn.abs().amax(-1),
        "linear_velocity_m_s": (velocity[..., :3] - expected_velocity[..., :3]).norm(dim=-1),
        "angular_velocity_rad_s": (velocity[..., 3:] - expected_velocity[..., 3:]).abs().amax(-1),
    }


def compatible_boundary(errors):
    # Pose tolerances match the supported-arm IK/sole checks. Velocity continuity
    # compares *differences* at the splice, allowing nonzero matching velocities.
    return (
        (errors["base_position_m"] <= 0.005)
        & (errors["angle_rad"] <= 0.05)
        & (errors["linear_velocity_m_s"] <= 0.02)
        & (errors["angular_velocity_rad_s"] <= 0.05)
    )


class StepStateAdapter:
    """BFM root = world_from_Trunk. Six planner base coordinates use intrinsic Y-X-Z.

    Both URDFs use identically signed revolute joint angles (rad). The controller
    actuates two additional head joints, the planner pins them at zero. Its two
    sole offsets differ from the planner's rounded offsets by at most 0.05 mm.
    No re-centering of the measured feet or base is applied.
    """

    def __init__(self, predictor, model=REAL_LEGS):
        self.predictor = predictor
        self.robot = model
        self.names = tuple(predictor.metadata["joint_names"])
        base = tuple("base_j_" + n for n in ("x", "y", "z", "pitch", "roll", "yaw"))
        if model.joint_names[:6] != base or len(self.names) != 29 or len(set(self.names)) != 29:
            raise ValueError("Unsupported T1 base or joint set")
        if set(self.names) - set(model.joint_names) != {"AAHead_yaw", "Head_pitch"}:
            raise ValueError("Unsupported T1 head/actuator mapping")
        self.bfm_columns = [self.names.index(n) for n in model.joint_names[6:]]

    def physical(self, q, qd=None):
        """Convert (...,33) generalized coordinates and rates to p,R,v,w,q29,qd29."""
        if q.shape[-1] != len(self.robot.joint_names) or (qd is not None and qd.shape != q.shape):
            raise ValueError("Expected T1 generalized coordinates and matching velocities")
        qd = torch.zeros_like(q) if qd is None else qd
        joint = q.new_zeros(*q.shape[:-1], len(self.names))
        joint_vel = torch.zeros_like(joint)
        joint[..., self.bfm_columns] = q[..., 6:]
        joint_vel[..., self.bfm_columns] = qd[..., 6:]
        angle = q[..., 3:6]
        R = roma.euler_to_rotmat("YXZ", angle)
        w = (_angular_axes(angle) @ qd[..., 3:6, None]).squeeze(-1)
        return q[..., :3], R, qd[..., :3], w, joint, joint_vel

    def planner(self, physical):
        """Project physical states, including full-time traces, into the 33 named planner coordinates."""
        p, R, v, w, joint, joint_vel = physical
        # In the model's joint box |roll|<pi/2. atan2 avoids asin at the poles.
        roll = torch.atan2(-R[..., 1, 2], torch.linalg.vector_norm(R[..., 1, :2], dim=-1))
        pitch = torch.atan2(R[..., 0, 2], R[..., 2, 2])
        yaw = torch.atan2(R[..., 1, 0], R[..., 1, 1])
        angle = torch.stack((pitch, roll, yaw), -1)
        axes = _angular_axes(angle)
        rates = torch.linalg.solve(axes, w.unsqueeze(-1)).squeeze(-1)
        return (
            torch.cat((p, angle, joint[..., self.bfm_columns]), -1),
            torch.cat((v, rates, joint_vel[..., self.bfm_columns]), -1),
        )

    def raw_context(self, dynamics):
        """Invert the seed-3 XiLayout normalization. Delay units are 5 ms physics substeps."""
        return torch.cat(
            (
                0.75 + 0.45 * dynamics[..., :1],
                0.01 * dynamics[..., 1:30],
                0.05 * dynamics[..., 30:33],
                5.0 + 3.0 * dynamics[..., 33:],
            ),
            dim=-1,
        )

    @prediction_precision()
    def initialize(self, q, dynamics, qd=None, *, observed=None, world=None, object_poses=None, held=None):
        """Issue state for one gap; caller must honor the per-row compatibility mask.

        An observed state contains its ten raw observations, two previous actions
        (newest first) and the controller's live clock and anchors. When unobserved,
        advance a synthetic history for 50 deployed HOLD ticks, then check the actual
        predicted body, soles and velocities against the incoming planner state.
        """
        q = q.reshape(-1, len(self.robot.joint_names))
        qd = torch.zeros_like(q) if qd is None else qd.reshape_as(q)
        if dynamics.shape != (q.shape[0], 38):
            raise ValueError("Expected normalized [B,38] seed-3 dynamics")
        physical = self.physical(q, qd)
        preparation = None
        if observed is None:
            _, R, _, w, joints, rates = physical
            previous = joints.new_zeros(len(q), 2 * joints.shape[-1])
            raw = torch.cat(
                (
                    joints - self.predictor.model.home_joints,
                    rates,
                    (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1),
                    -R[:, 2, :],
                    previous[:, : joints.shape[-1]],
                ),
                dim=-1,
            )
            hist = raw[:, None, :].repeat(1, self.predictor.metadata["history_ticks"], 1)
            full = (*physical, previous, hist, q.new_zeros(len(q), 8))
            initial = LandingState(full, dynamics, "synthetic_hold_issue")
            ticks = torch.full((len(q),), 50, dtype=torch.long, device=q.device)
            full, following, actions = _hold_trace(self.predictor, full, dynamics, ticks, 50)
            state = LandingState(full, dynamics, "synthetic_retract_50_hold")
            if world is not None:
                # The planner may explicitly include the predicted preparation motion.
                # Its first state is exactly the requested pose/rate and its final state
                # is exactly the one used to issue the step. No pose or velocity reset.
                state.history_source = "synthetic_supported_hold_50"
                trajectory = tuple(torch.cat((start[:, None], tail), dim=1) for start, tail in zip(physical, following))
                sole, height, tilt = self.predictor.model.feet(tuple(v.flatten(0, 1) for v in trajectory))
                b, t = trajectory[0].shape[:2]
                sole, height, tilt = sole.reshape(b, t, 2, 3), height.reshape(b, t, 2), tilt.reshape(b, t, 2)
                xy = (sole[..., :2] - sole[:, :1, :, :2]).norm(dim=-1).amax((1, 2))
                yaw_delta = sole[..., 2] - sole[:, :1, :, 2]
                yaw = torch.atan2(yaw_delta.sin(), yaw_delta.cos()).abs().amax((1, 2))
                prepared = HoldPreparation(
                    trajectory,
                    sole,
                    height,
                    tilt,
                    torch.ones((b, t), device=q.device, dtype=torch.bool),
                    (xy < 0.020) & (yaw < torch.pi / 180),
                    actions,
                )
                preparation = HoldPrediction((prepared,), prepared.tracking_ok, initial)
            peak = torch.stack((following[5][:, -25:].square().mean(-1).sqrt().amax(1),
                                following[2][:, -25:].norm(dim=-1).amax(1)), dim=-1)
            diagnostics = {
                "max_joint_rms": peak[:, 0],
                "max_root_speed": peak[:, 1],
                "near_rest": near_rest(peak[:, 0], peak[:, 1]),
            }
        else:
            state = observed
            if (
                len(state.tensors) != 9
                or state.dynamics.shape != dynamics.shape
                or not torch.equal(state.dynamics, dynamics)
            ):
                raise ValueError("Observed state and requested dynamics differ")
            n, h, j = len(q), self.predictor.metadata["history_ticks"], len(self.names)
            shapes = ((n, 3), (n, 3, 3), (n, 3), (n, 3), (n, j), (n, j), (n, 2 * j), (n, h, 3 * j + 6), (n, 8))
            if any(
                v.shape != shape or v.device != q.device or v.dtype != q.dtype
                for v, shape in zip(state.tensors, shapes)
            ):
                raise ValueError("Observed state is missing named joints, history, clock or delay queue")
            diagnostics = {}
        _, R, _, w, joints, joint_rates, previous, history, _ = state.tensors
        actual_obs = torch.cat(
            (
                joints - self.predictor.model.home_joints,
                joint_rates,
                (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1),
                -R[:, 2, :],
                previous[:, : len(self.names)],
            ),
            -1,
        )
        synchronized = (actual_obs - history[:, -1]).abs().amax(-1) < 1.0e-4
        synchronized &= torch.stack([v.reshape(len(q), -1).isfinite().all(-1) for v in state.tensors]).all(0)
        diagnostics["synchronized_history"] = synchronized
        diagnostics["joint_hard_limit_rad"] = _physical_joint_excess(self.robot, self.names, joints)
        current, velocity = self.planner(state.tensors[:6])
        diagnostics.update(boundary_errors(current, velocity, q, qd))
        now, _, _ = self.predictor.model.feet(state.tensors[:6])
        expected, _, _ = self.predictor.model.feet(physical)
        delta = now - expected
        diagnostics["sole_xy_m"] = delta[..., :2].norm(dim=-1).amax(-1)
        diagnostics["sole_yaw_rad"] = torch.atan2(delta[..., 2].sin(), delta[..., 2].cos()).abs().amax(-1)
        phase = state.tensors[8]
        # The exported tick's observation feedback assumes unbiased joint encoders.
        applicable = (dynamics[..., 1:30] == 0).all(-1)
        usable = (
            compatible_boundary(diagnostics)
            & (diagnostics["sole_xy_m"] <= 0.005)
            & (diagnostics["sole_yaw_rad"] <= 0.05)
            & (phase[:, 0] > 0.5)
            & (phase[:, 2:4] == 0).all(-1)
            & applicable
        )
        if preparation is not None:
            geom = validate_step_geometry(world, self, preparation, object_poses=object_poses, held=held)
            usable = geom["valid"] & applicable & (phase[:, 0] > 0.5) & (phase[:, 2:4] == 0).all(-1)
            diagnostics.update(
                preparation_ok=geom["valid"],
                preparation_frames=preparation.steps[0].valid_ticks.shape[1],
                preparation_contact_and_drift=preparation.tracking_ok,
            )
        usable &= synchronized & (diagnostics["joint_hard_limit_rad"] <= 1.0e-6)
        diagnostics.update(history_source=state.history_source, encoder_context_supported=applicable)
        return StepStart(state, usable, diagnostics, preparation)


@dataclass
class HoldPreparation:
    trajectory: tuple
    sole_trace: Tensor
    sole_height: Tensor
    sole_tilt: Tensor
    valid_ticks: Tensor
    tracking_ok: Tensor
    actions: Tensor
    side: str = "HOLD"


@dataclass
class HoldPrediction:
    steps: tuple
    tracking_ok: Tensor
    initial: LandingState


def _obb_gap(A, ah, B, bh):
    """Signed separating-axis gap of two oriented boxes in 3D (positive = separated)."""
    # world_from_box; axes are the rotation matrix's columns.
    ar, br = torch.broadcast_tensors(A[..., :3, :3].transpose(-1, -2), B[..., :3, :3].transpose(-1, -2))
    cross = torch.linalg.cross(ar[..., :, None, :], br[..., None, :, :], dim=-1).flatten(-3, -2)
    axes = torch.cat((ar, br, cross), dim=-2)
    length = axes.norm(dim=-1)
    unit = axes / length.clamp_min(1.0e-8)[..., None]
    a_radius = (unit @ ar.transpose(-1, -2)).abs() @ ah
    b_radius = (unit @ br.transpose(-1, -2)).abs() @ bh
    delta = (A[..., :3, 3] - B[..., :3, 3])[..., None, :]
    return ((unit * delta).sum(-1).abs() - a_radius - b_radius).masked_fill(length < 1.0e-6, -torch.inf).amax(-1)


def _object_box(obj, world_from_obj):
    from cutamp.step_search import _local_bounds

    bounds = _local_bounds(obj)
    if bounds is None:
        raise ValueError(f"No bounds for scene object {obj.name}")
    a, b = world_from_obj.new_tensor(bounds)
    transform = world_from_obj.clone()
    transform[..., :3, 3] += (transform[..., :3, :3] @ ((a + b) / 2)[..., None]).squeeze(-1)
    return transform, (b - a) / 2


def _moving_head(spheres, p, R, joints, adapter, inds, offset):
    """Replace the head centers in cuRobo's locked-head sphere model by their measured FK."""
    # URDF Trunk -> H1 (yaw), H1 -> H2 (pitch). The head's two centers are taken
    # from the exact locked FK, so the collision sphere model stays identical at q_head=0.
    y = joints[..., adapter.names.index("AAHead_yaw")]
    pitch = joints[..., adapter.names.index("Head_pitch")]
    rot = roma.euler_to_rotmat("ZY", torch.stack((y, pitch), dim=-1))
    origin = p + (R @ offset).squeeze(-1)
    local = (spheres[:, inds, :3] - origin[:, None]) @ R
    moving = (local @ (R @ rot).transpose(-1, -2)) + origin[:, None]
    spheres = spheres.clone()
    spheres[:, inds, :3] = moving
    return spheres


@lru_cache(maxsize=8)
def _physical_joint_limits(robot, names):
    """The source URDF preserves physical joint ranges before supported-posture limits are applied."""
    tree = ET.parse(robot.asset_root / "t1_planner.urdf")
    by_name = {j.get("name"): j.find("limit") for j in tree.getroot().findall("joint")}
    return tuple((float(by_name[n].get("lower")), float(by_name[n].get("upper"))) for n in names)


def _physical_joint_excess(robot, names, positions):
    limits = positions.new_tensor(_physical_joint_limits(robot, names))
    return torch.maximum(limits[:, 0] - positions, positions - limits[:, 1]).amax(-1).clamp_min(0.0)


@lru_cache(maxsize=8)
def _foot_half_extents(robot):
    from cutamp.step_set import _facts

    return tuple(size * 0.5 for size in _facts(robot)["foot_box"])


@prediction_precision()
def _step_geometry(world, adapter, prediction, *, object_poses=None, held=None):
    """Fresh, batched, 50-Hz trajectory checks on real achieved whole-body states.

    `held` maps an arm to (object name, world-tool-frame-relative object pose).
    Check the carried rigid object's geometry against the scene, every other movable
    and the robot outside its holding hand. The model contains no payload dynamics.
    """
    from curobo.types import JointState

    from cutamp.costs import sphere_to_sphere_overlap
    from cutamp.gn_refresh import collision_penetration
    from cutamp.robots.t1 import GRIPPER_JOINT_NAMES, t1_curobo_cfg
    from cutamp.utils.common import transform_spheres

    held = dict(held or {})
    carried = {name: arm for arm, (name, _) in held.items()}
    poses = {obj.name: world.get_object_pose(obj) for obj in world.movables + world.statics}
    poses.update(object_poses or {})
    B = len(prediction.tracking_ok)
    kp = world.kinematics.kinematics_config

    def excluding(names):
        # Use the model's actual sphere-to-link map. Materialize the indices once;
        # repeated nonzero/Boolean indexing would synchronize each trajectory chunk.
        link_ids = kp.link_sphere_idx_map.flatten()
        mask = torch.ones_like(link_ids, dtype=torch.bool)
        for name in names:
            if name in kp.link_name_to_idx_map:
                mask &= link_ids != kp.link_name_to_idx_map[name]
        return mask.nonzero().flatten()

    foot_body_ids = {
        arm: excluding((f"Shank_{arm.title()}", f"Ankle_Cross_{arm.title()}", f"{arm}_foot_link"))
        for arm in ("left", "right")
    }
    payload_body_ids = {
        arm: excluding((world.tool_frame_for_arm(arm), f"{arm}_hand_link", *GRIPPER_JOINT_NAMES[arm]))
        for arm in held
    }
    head_ids = None
    if "H2" in t1_curobo_cfg(world.model)["robot_cfg"]["kinematics"]["collision_link_names"]:
        head_ids = kp.get_sphere_index_from_link_name("H2").long()
    like = world.q_init
    head_offset = like.new_tensor([0.0625, 0, 0.243 + 0.06185]).reshape(3, 1)
    physical_limits = like.new_tensor(_physical_joint_limits(adapter.robot, adapter.names))
    half = like.new_tensor(_foot_half_extents(world.model))
    totals = {}
    history = []
    for idx, step in enumerate(prediction.steps):
        q, qd = adapter.planner(step.trajectory)
        T = q.shape[1]

        def at(poses, rows):
            poses = poses.reshape(-1, 4, 4)
            return poses.expand(B, -1, -1)[rows]

        peaks = {}
        held_world = {arm: [] for arm in held}
        moments = []
        # A cuRobo instance owns writable FK buffers. Give each concurrently differentiated
        # step its own instance; run one FK per trace before its backward.
        if torch.is_grad_enabled():
            from curobo.kinematics import Kinematics

            kin = Kinematics(world.kinematics.config)
        else:
            kin = world.kinematics
        chunk_size = B * T if torch.is_grad_enabled() else 256
        for start in range(0, B * T, chunk_size):
            stop = min(start + chunk_size, B * T)
            rows = torch.arange(start, stop, device=q.device) // T
            frames = slice(start, stop)
            valid = step.valid_ticks.flatten()[frames]
            qpart = q.flatten(0, 1)[frames].contiguous()
            fk = kin.compute_kinematics(JointState.from_position(qpart))
            if step.side == "HOLD":
                moments.append(fk.robot_com.reshape(-1, 4)[:, :2].clone())

            def link(name, state=fk):
                return state.tool_poses.get_link_pose(name, make_contiguous=True).get_matrix().reshape(-1, 4, 4)

            spheres = fk.robot_spheres.reshape(len(qpart), -1, 4)
            sphere_poses, boxes = {}, {}
            actual_joints = step.trajectory[4].flatten(0, 1)[frames]
            if head_ids is not None:
                p, R = (v.flatten(0, 1)[frames] for v in step.trajectory[:2])
                spheres = _moving_head(spheres, p, R, actual_joints, adapter, head_ids, head_offset)
            pen = {"joint_hard_limit_rad": torch.maximum(
                physical_limits[:, 0] - actual_joints, actual_joints - physical_limits[:, 1]
            ).amax(-1).clamp_min(0.)}
            for obj in world.movables + world.statics:
                if obj.name in carried:
                    arm = carried[obj.name]
                    body_pose = link(world.tool_frame_for_arm(arm)) @ at(held[arm][1], rows)
                    held_world[arm].append(body_pose)
                else:
                    body_pose = at(poses[obj.name], rows)
                boxes[obj.name] = _object_box(obj, body_pose)
                if obj.name in carried or obj in world.movables:
                    sphere_poses[obj.name] = transform_spheres(world.get_collision_spheres(obj), body_pose)
            fixed = [s for name, s in sphere_poses.items() if name not in carried]
            pen["robot_collision_m"] = collision_penetration(world, spheres, torch.cat(fixed, 1) if fixed else None)
            pen["body_floor_m"] = (spheres[..., 3] - spheres[..., 2]).amax(-1).clamp_min(0)
            for name, arm in carried.items():
                obj_s = sphere_poses[name]
                # Only suppress the grasping hand, never the opposite arm or torso.
                pen["payload_robot_" + arm] = sphere_to_sphere_overlap(obj_s, spheres.index_select(1, payload_body_ids[arm]), 0.0)
                pen["payload_scene_" + arm] = world.collision_fn(obj_s[:, None].contiguous()).reshape(-1)
                others = [sp for other, sp in sphere_poses.items() if other != name]
                if others:
                    pen["payload_movables_" + arm] = sphere_to_sphere_overlap(obj_s, torch.cat(others, 1), 0.0)
                pen["payload_floor_" + arm] = (obj_s[..., 3] - obj_s[..., 2]).amax(1).clamp_min(0)
            # The planner's collision spheres leave out the feet. Use both physical
            # URDF foot boxes at each tick, including the swing over floor objects.
            feet = []
            for instr in world.model.sole_frames:
                foot = link(instr).clone()
                foot[..., :3, 3] += foot[..., :3, 2] * half[2]
                feet.append(foot)
                bottom = foot[..., 2, 3] - (foot[..., 2, :3].abs() * half).sum(-1)
                pen[f"{instr}_floor_m"] = (-0.005 - bottom).clamp_min(0)
                for obj in world.movables + world.statics:
                    box, oh = boxes[obj.name]
                    # Floor boxes entirely at/below the support plane are accounted
                    # by the 5 mm contact band below. Elevated objects stay active.
                    top = box[..., 2, 3] + (box[..., 2, :3].abs() * oh).sum(-1)
                    overlap = -_obb_gap(foot, half, box, oh)
                    pen[f"{instr}_{obj.name}"] = torch.where(top <= 1e-6, 0.0, overlap).clamp_min(0)
            pen["foot_to_foot_m"] = (-_obb_gap(feet[0], half, feet[1], half)).clamp_min(0)
            for arm, box in zip(("left", "right"), feet):
                # Mated ankle/leg is exempt; all upper body, opposite leg and arm
                # poses use actual spheres. Transform those sphere centers into foot.
                body = spheres.index_select(1, foot_body_ids[arm])
                rel = ((body[..., :3] - box[:, None, :3, 3]) @ box[..., :3, :3]).abs() - half
                dist = rel.clamp_min(0.0).norm(dim=-1) + rel.amax(-1).clamp_max(0.0)
                pen["foot_body_" + arm] = (body[..., 3] - dist).clamp_min(0).amax(-1)
            for name, value in pen.items():
                if name not in peaks:
                    peaks[name] = q.new_zeros(B)
                peaks[name] = peaks[name].scatter_reduce(0, rows, value.masked_fill(~valid, 0.0), reduce="amax")
        # The two feet stay within the 5 mm sole contact band throughout and finish
        # flat on the floor to the same tolerance that joins a supported arm motion.
        peaks["floor_penetration_m"] = (
            (-step.sole_height - 0.005).masked_fill(~step.valid_ticks[:, :, None], 0.0).amax((1, 2)).clamp_min(0.0)
        )
        peaks["landed_height_m"] = (step.sole_height[:, -1].abs() - 0.005).amax(-1).clamp_min(0.0)
        peaks["landed_tilt_rad"] = (step.sole_tilt[:, -1] - 0.05).amax(-1).clamp_min(0.0)
        violations = torch.stack(tuple(peaks.values()), dim=1).amax(1)
        # Measured slight penetration is admissible at ground contact within the
        # same positional tolerance used by planted-sole cuRobo trajectories.
        ground_ok = ((step.sole_height >= -0.005) | ~step.valid_ticks[:, :, None]).all((1, 2))
        landed = (step.sole_height[:, -1].abs() <= 0.005).all(-1) & (step.sole_tilt[:, -1] <= 0.05).all(-1)
        ok = (violations <= 1e-6) & ground_ok & landed & step.tracking_ok
        if step.side == "HOLD":
            ok &= (step.sole_height.abs() <= 0.005).all((1, 2)) & (step.sole_tilt <= 0.05).all((1, 2))
            # Remain in double support during the preparatory HOLD.
            from cutamp.com_polygon_cost import margin_from_support_planes, support_region_planes

            com = torch.cat(moments).reshape(B, T, 2)
            support = support_region_planes(step.sole_trace[:, 0], world.model.sole_box, False)
            ok &= (margin_from_support_planes(com, support[0][:, None], support[1][:, None]) >= 0).all(1)
        else:
            p = step.phases
            ok &= p.deadline_ok & p.support_ok & p.hold_ok
        for name, value in peaks.items():
            totals[name] = torch.maximum(totals.get(name, torch.zeros_like(value)), value)
        history.append(
            {
                "position": q,
                "velocity": qd,
                "valid_ticks": step.valid_ticks,
                "valid": ok,
                "grounded": ground_ok & landed,
                "peak_penetration": violations,
                "held_objects": {arm: torch.cat(held_world[arm]).reshape(B, T, 4, 4) for arm in held},
            }
        )
    valid = prediction.tracking_ok.clone()
    for entry in history:
        valid &= entry["valid"]
    return {
        "valid": valid,
        "steps": history,
        "penetration": totals,
        "load_status": "load_unmodeled" if held else "unloaded",
    }


@torch.no_grad()
def validate_step_geometry(world, adapter, prediction, *, object_poses=None, held=None):
    return _step_geometry(world, adapter, prediction, object_poses=object_poses, held=held)


def sweep_constraint_rows(world, adapter, prediction, object_poses=None, held=None):
    """Largest measured path violation per particle in the same model/scene as fresh scoring."""
    if prediction is None or not prediction.steps:
        return {}
    peaks = _step_geometry(world, adapter, prediction, object_poses=object_poses, held=held)["penetration"]
    return sweep_penetration_rows(peaks)


def sweep_penetration_rows(peaks):
    # Each underlying gate must pass. The maximum gives the QP the active
    # constraint's gradient and keeps the number of full-dynamics VJPs bounded.
    limits = [(v - 1.0e-6) / (0.05 if k.endswith("_rad") else 0.02) for k, v in peaks.items()]
    return {"swept_geometry": torch.stack(limits, dim=-1).amax(-1)}


@dataclass
class TransitionRequest:
    foot: str
    command: Tensor
    initial: LandingState
    predicted: object
    initial_position: Tensor
    initial_velocity: Tensor
    world: object
    object_poses: dict
    held: dict
    dt: float = 0.02


@dataclass
class TransitionEvidence:
    """Independent validator's heard/executed trace in model joint order, including the issue state.

    A step path contains `clock + hold_ticks + 1` frames at 50 Hz, including the issue
    state and the deadline. Consecutive alternating steps can join there with zero hold ticks. A preparatory
    HOLD contains its 51 consecutive frames, starting at the arriving state. A measured terminal
    state with actual observation/action histories is required before another step.
    The source must certify continuous swept clearance and contact on this scene.
    """

    source: str
    position: Tensor
    velocity: Tensor
    acceleration: Tensor
    dt: float
    contact_ok: bool
    swept_clearance_ok: bool
    load_ok: bool
    terminal: LandingState | None = None


class TransitionUnavailable(RuntimeError):
    """No independent evidence at the requested validation level."""


def model_step_record(adapter, initial, step, command, row=0, *, preparation=None):
    """A CPU replay record for a selected row, including its actual control and whole-body traces."""
    labels = (
        "root_position",
        "root_rotation",
        "root_linear_velocity",
        "root_angular_velocity",
        "joint_position",
        "joint_velocity",
        "previous_actions",
        "observation_history",
        "command_state",
    )

    def array(value):
        # Lists keep the result compatible with both the usual JSON metrics and raw plan pickle.
        return value.detach().cpu().tolist()

    def snapshot(state):
        return {name: array(value[row]) for name, value in zip(labels, state.tensors)}

    ticks = int(step.valid_ticks[row].sum())
    model = adapter.predictor
    record = {
        "foot": step.side,
        "command": array(command[row]),
        "dt_s": 0.02,
        "step_ticks": int(step.clock[row]),
        "hold_ticks": ticks - int(step.clock[row]),
        "joint_names": list(adapter.names),
        "checkpoint_sha256": model.metadata["checkpoint_sha256"],
        "deployment_sha256": model.metadata["deployment_sha256"],
        "history_source": initial.history_source,
        "dynamics_normalized": array(initial.dynamics[row]),
        "dynamics_raw": array(adapter.raw_context(initial.dynamics[row])),
        "profile": dict(vars(model.profile)),
        "xy_limit_m": 0.020,
        "yaw_limit_rad": torch.pi / 180,
        "initial": snapshot(initial),
        "terminal": snapshot(step.terminal),
        "actions": array(step.actions[row, :ticks]),
        "trajectory": {name: array(value[row, :ticks]) for name, value in zip(labels, step.trajectory)},
        "issued_soles": array(step.reference[row]),
        "sole_trace": array(step.sole_trace[row, :ticks]),
        "sole_height": array(step.sole_height[row, :ticks]),
        "sole_tilt": array(step.sole_tilt[row, :ticks]),
    }
    if preparation is not None:
        hold = preparation.steps[0]
        frames = int(hold.valid_ticks[row].sum())
        record["preparation"] = {
            "mode": "HOLD",
            "hold_ticks": frames - 1,
            "dt_s": 0.02,
            "history_source": preparation.initial.history_source,
            "initial": snapshot(preparation.initial),
            "terminal": record["initial"],
            "actions": array(hold.actions[row, : frames - 1]),
            "trajectory": {name: array(value[row, :frames]) for name, value in zip(labels, hold.trajectory)},
            "sole_trace": array(hold.sole_trace[row, :frames]),
            "sole_height": array(hold.sole_height[row, :frames]),
            "sole_tilt": array(hold.sole_tilt[row, :frames]),
            "sole_drift_ok": bool(hold.tracking_ok[row]),
        }
    return record


def verify_transition(request, evidence, adapter, intended_final=None):
    """Check independent execution in its own issue frame; preserve the actual end velocity."""
    from curobo.types import JointState

    from cutamp.landing_predictor import _intended_pair, landing_errors, phase_checks

    if evidence is None:
        raise TransitionUnavailable("Independent step validator supplied no transition")
    if not isinstance(evidence, TransitionEvidence) or not evidence.source:
        raise ValueError("Expected source-labelled TransitionEvidence")
    step = request.predicted
    hold = request.foot == "HOLD"
    n = int(step.valid_ticks[0].sum()) + (0 if hold else 1)
    values = (evidence.position, evidence.velocity, evidence.acceleration)
    if abs(evidence.dt - request.dt) > 1.0e-7 or any(
        x.shape != (n, len(adapter.robot.joint_names)) or not x.isfinite().all() for x in values
    ):
        raise ValueError("Independent step trace must include every STEP and HOLD frame at 50 Hz")
    if not (evidence.contact_ok and evidence.swept_clearance_ok and (not request.held or evidence.load_ok)):
        raise ValueError("Independent contact, clearance or carried-object check failed")
    q, qd, acc = [v.to(request.initial_position) for v in values]
    if not compatible_boundary(boundary_errors(q[0], qd[0], request.initial_position, request.initial_velocity)).all():
        raise ValueError("Independent step trace does not begin at the arriving state and velocity")
    if (_physical_joint_excess(adapter.robot, adapter.robot.joint_names[6:], q[:, 6:]) > 1.0e-6).any():
        raise ValueError("Independent step trace exceeds a physical URDF joint limit")
    trace, height, tilt = adapter.predictor.model.feet(adapter.physical(q, qd))
    js = JointState(position=q, velocity=qd, acceleration=acc, joint_names=list(adapter.robot.joint_names))
    if evidence.terminal is not None:
        checked = adapter.initialize(q[-1:], request.initial.dynamics, qd[-1:], observed=evidence.terminal)
        if not checked.compatible.all():
            raise ValueError("Independent terminal state history is inconsistent with the delivered trajectory")
    if hold:
        if evidence.terminal is None:
            raise TransitionUnavailable("Preparatory HOLD requires measured terminal histories before issuing a step")
        _, _, planted = landing_errors(trace, trace[:1])
        if not (planted.all() and (height.abs() <= 0.005).all() and (tilt <= 0.05).all()):
            raise ValueError("Independent preparatory HOLD exceeds the planted-sole limits")
        return (
            js,
            trace[-1],
            {
                "phase": "step_prepare",
                "foot": "HOLD",
                "source": evidence.source,
                "dt": request.dt,
                "history_source": request.initial.history_source,
                "load_validated": bool(request.held and evidence.load_ok),
                "issued": trace[0].detach(),
                "achieved": trace[-1].detach(),
                "clock": n - 1,
            },
        )
    side = "LR".index(request.foot)
    ref = _intended_pair(trace[:1], side, request.command.reshape(1, 3))
    xy, yaw, tracks = landing_errors(trace[-1:], ref)
    checks = phase_checks(
        trace[None, 1:],
        height[None, 1:],
        ref,
        side,
        step.clock,
        torch.ones((1, n - 1), dtype=torch.bool, device=q.device),
        adapter.predictor.profile,
    )
    if not (tracks & checks.deadline_ok & checks.support_ok & checks.hold_ok).item():
        raise ValueError("Independent landing missed the issued precision or phase limits")
    if not ((height >= -0.005).all() and (height[-1].abs() <= 0.005).all() and (tilt[-1] <= 0.05).all()):
        raise ValueError("Independent landing leaves the grounded-sole tolerance")
    if intended_final is not None and not landing_errors(trace[-1:], intended_final)[2].all():
        raise ValueError("Independent final stance missed the composed original command chain")
    return (
        js,
        trace[-1],
        {
            "source": evidence.source,
            "phase": "step",
            "foot": request.foot,
            "command": request.command.detach(),
            "issued": ref[0].detach(),
            "achieved": trace[-1].detach(),
            "xy_error_m": xy[0].detach(),
            "yaw_error_rad": yaw[0].detach(),
            "tracking": True,
            "clock": int(step.clock[0]),
            "hold_ticks": n - 1 - int(step.clock[0]),
            "history_source": request.initial.history_source,
            "load_validated": bool(request.held and evidence.load_ok),
            "dt": request.dt,
        },
    )


class SteppedMotion:
    """State of one independent step/arm motion attempt; shared across grounded operators."""

    def __init__(self, skeleton, predictions, adapter, world, validator):
        self.inputs, self.adapter, self.world, self.validator = predictions, adapter, world, validator
        self.by_op = {}
        self.last_state = None
        self.stance = None
        self.records = []
        self.occupied_arm = None
        self.supported_kinematics = None
        self.planned_endpoints = {}
        self.endpoint_collides = None
        gap, offset = 0, 0
        for idx, op in enumerate(skeleton):
            typ = op.operator.metadata.action_type
            if typ == "step":
                self.by_op[idx] = (gap, offset)
                offset += 1
            elif typ in ("pick", "place"):
                offset = 0
                gap += 1

    def check_supported_start(self):
        """The arm planner pins both head joints. Verify their measured handoff before using its geometry."""
        if self.stance is None:
            return
        if self.last_state is None:
            raise TransitionUnavailable("Supported motion requires measured terminal head position and velocity")
        names = ("AAHead_yaw", "Head_pitch")
        columns = [self.adapter.names.index(n) for n in names]
        from cutamp.robots.t1 import t1_curobo_cfg

        q, qd = self.last_state.tensors[4:6]
        pinned = t1_curobo_cfg(self.world.model)["robot_cfg"]["kinematics"]["lock_joints"]
        err = (q[:, columns] - q.new_tensor([pinned[n] for n in names])).abs()
        if not ((err <= 0.05).all() and (qd[:, columns].abs() <= 0.05).all()):
            raise TransitionUnavailable("Measured head state cannot join the supported planner's locked head")

    def initial_velocity(self):
        initial, _, _, *preparations = self.inputs[0]
        source = preparations[0].initial if preparations and preparations[0] is not None else initial
        return self.adapter.planner(source.tensors[:6])[1]

    def step_segments(self, idx, last_js, object_poses, held):
        from cutamp.landing_predictor import _intended_pair

        gap, j = self.by_op[idx]
        initial, commands, prediction, *preparations = self.inputs[gap]
        incoming = last_js.position.reshape(1, -1)
        velocity = last_js.velocity.reshape(1, -1) if last_js.velocity is not None else torch.zeros_like(incoming)
        bound = {
            arm: (obj, torch.linalg.inv(self.world.tool_from_ee[self.world.tool_frame_for_arm(arm)]) @ grasp)
            for arm, (obj, grasp) in held.items()
        }
        segments = []
        if j == 0:
            start = initial
            original = self.adapter.predictor.model.feet(start.tensors[:6])[0]
            for k, step in enumerate(prediction.steps):
                original = _intended_pair(original, "LR".index(step.side), commands[:, k])
            self.intended_final = original
            if preparations and preparations[0] is not None:
                # Independently advance the 50-tick preparatory HOLD from the actual
                # end of the preceding segment. It can move the base/joints appreciably.
                prepared = self.adapter.initialize(
                    incoming, initial.dynamics, velocity, world=self.world, object_poses=object_poses, held=bound
                )
                if not prepared.compatible.all():
                    raise ValueError("Actual arrival cannot execute the supported HOLD preparation")
                hold = prepared.preparation
                req = TransitionRequest(
                    "HOLD",
                    incoming.new_empty(0),
                    hold.initial,
                    hold.steps[0],
                    incoming[0],
                    velocity[0],
                    self.world,
                    object_poses,
                    bound,
                )
                evidence = self.validator(req)
                plan, _, record = verify_transition(req, evidence, self.adapter)
                segments.append((plan, record))
                start = evidence.terminal
                incoming, velocity = plan.position[-1:], plan.velocity[-1:]
        else:
            if self.last_state is None:
                raise TransitionUnavailable(
                    "Adjacent step requires independent terminal observation and action histories"
                )
            start = self.last_state
        # Rerun from the actual issue state and the saved command.
        saved = prediction.steps[j]
        foot = saved.side
        # Preserve the recorded dwell even if the measured state gives a different step clock.
        hold_ticks = int(saved.valid_ticks[0].sum() - saved.clock[0])
        result = self.adapter.predictor.rollout(start, foot, commands[:, j : j + 1], final_hold_ticks=hold_ticks)
        s = result.steps[0]
        if not self.adapter.initialize(incoming, start.dynamics, velocity, observed=start).compatible.all():
            raise ValueError("Previous motion cannot connect to the requested step state and velocity")
        if not validate_step_geometry(self.world, self.adapter, result, object_poses=object_poses, held=bound)[
            "valid"
        ].all():
            raise ValueError("Replayed command violates the model trace constraints")
        request = TransitionRequest(
            foot, commands[0, j], start, s, incoming[0], velocity[0], self.world, object_poses, bound
        )
        evidence = self.validator(request)
        js, stance, record = verify_transition(
            request, evidence, self.adapter, self.intended_final if j == len(prediction.steps) - 1 else None
        )
        self.stance, self.last_state = stance, evidence.terminal
        self.records.append(record)
        # Save the independently checked trace through JointState; retain the complete
        # model prediction separately so command replay and auditor provenance stay explicit.
        entry = dict(record, model_prediction=model_step_record(self.adapter, start, s, commands[:, j]))
        if evidence.terminal is not None:
            entry["measured_terminal"] = {
                "history_source": evidence.terminal.history_source,
                "tensors": [v[0].detach().cpu().tolist() for v in evidence.terminal.tensors],
            }
        return segments + [(js, entry)]

    def approach(self, idx, op, particle, object_poses, cfg, ik_tol):
        from cutamp.com_polygon_cost import margin_from_support_planes, support_region_planes
        from cutamp.grasps import approach_targets, is_vertical
        from cutamp.motion_solver import _hand_poses
        from cutamp.step_ik import solve_stance_ik
        from cutamp.utils.common import action_4dof_to_mat4x4, grasp_7dof_to_mat4x4

        kind, arm = op.operator.metadata.action_type, op.operator.metadata.arm
        seed = particle[op.arg("q")].reshape(1, -1)
        obj = object_poses[op.arg("obj")] if kind == "pick" else action_4dof_to_mat4x4(particle[op.arg("placement")])
        wrist = (
            obj
            @ grasp_7dof_to_mat4x4(particle[op.arg("grasp")])
            @ self.world.tool_from_ee[self.world.tool_frame_for_arm(arm)]
        ).reshape(1, 4, 4)
        s = self.stance
        if s is None:
            s = self.adapter.predictor.model.feet(self.adapter.physical(self.world.q_init.reshape(1, -1)))[0][0]
        s = s.reshape(1, 2, 3)
        fitted = solve_stance_ik(s, wrist, seed, arm, other_arm_reference=self.occupied_arm,
                                 collides=self.endpoint_collides, kinematics=self.supported_kinematics)

        def pose_ok(ik):
            return (
                ik.ok
                & (ik.wrist_pos_error <= ik_tol.pos)
                & (ik.wrist_rot_error <= ik_tol.rot)
                & (ik.sole_pos_error <= ik_tol.sole_pos)
                & (ik.sole_rot_error <= ik_tol.sole_rot)
            )

        if (
            not pose_ok(fitted).item()
            or float(fitted.margin[0]) < cfg.com_comfort_margin_m
            or not (
                margin_from_support_planes(
                    fitted.com[:, :2], *support_region_planes(s.double(), self.world.model.sole_box, True)
                )
                >= 0
            ).all()
        ):
            raise ValueError("Actual landing cannot support the bound full wrist pose with the required margin")
        particle[op.arg("q")] = fitted.q[0].to(seed).detach()
        self.planned_endpoints[idx] = fitted
        hand = _hand_poses(self.world, particle[op.arg("q")][None], arm)[0]
        targets = torch.stack(approach_targets(hand, kind))
        idle = self.world.model.right_arm if arm == "left" else self.world.model.left_arm
        linked = solve_stance_ik(s.expand(2, -1, -1), targets, fitted.q.expand(2, -1), arm,
                                 other_arm_reference=fitted.q[:, idle].expand(2, -1), kinematics=self.supported_kinematics)
        if not pose_ok(linked).all():
            raise ValueError("Actual stance cannot reach manipulation pre/post keyframes")
        return linked.q[0].to(seed).detach(), linked.q[1].to(seed).detach(), is_vertical(hand)
