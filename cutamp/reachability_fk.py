"""Offline, grasp-independent FK sampling for T1 hand reachability on planted soles."""

import copy
import hashlib
import xml.etree.ElementTree as ET
from typing import NamedTuple

import numpy as np
import torch
import yaml
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import DeviceCfg, JointState
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from yourdfpy import URDF

from cutamp.gn_refresh import _sole_residual_and_jacobian
from cutamp.robots.envelope import ENVELOPES_DIR
from cutamp.robots.t1 import (
    GRIPPER_JOINT_NAMES,
    REAL_LEGS,
    SWEEP_VOLUME_PATH,
    TOOL_FRAME_FOR_ARM,
    get_t1_kinematics,
    load_sweep_volume,
    t1_curobo_cfg,
)


class PostureSeed(NamedTuple):
    trunk_pitch: float
    knee_pitch: float
    ankle_pitch: float


# Absolute sagittal angles in radians. Hip flexion in the deep squat moves the pelvis
# back while the trunk pitches forward. Each fitted stance is screened below.
POSTURES = {
    "standing": PostureSeed(0.0, .40, -.20),
    "crouch_0.30": PostureSeed(0.0, 1.00, -.50),
    "crouch_0.60": PostureSeed(0.0, 1.60, -.80),
    "deep_squat_hinge": PostureSeed(.60, 1.97, -.45),
}
COM_MARGIN_M = 0.05
SOLE_POS_TOL_M = 0.0003
SOLE_ANGLE_TOL_RAD = 0.001
BODY_LINKS = {
    "spine": ("Waist", "Trunk", "H2"),
    "left_arm": ("AL1", "AL3", "AL5", "left_base_link"),
    "right_arm": ("AR1", "AR3", "AR5", "right_base_link"),
    "left_leg": ("Waist", "Hip_Pitch_Left", "Shank_Left", "left_sole_link"),
    "right_leg": ("Waist", "Hip_Pitch_Right", "Shank_Right", "right_sole_link"),
}


def _sole_transforms(soles):
    transforms = np.tile(np.eye(4), (2, 1, 1))
    for mat, (x, y, yaw) in zip(transforms, soles):
        c, s = np.cos(yaw), np.sin(yaw)
        mat[:2, :2] = [[c, -s], [s, c]]
        mat[:2, 3] = [x, y]
    return transforms


def _support_planes(transforms, sole_box):
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * (np.asarray(sole_box) / 2)
    xy = corners @ transforms[:, :2, :2].transpose(0, 2, 1) + transforms[:, None, :2, 3]
    # Qhull gives outward unit normals and signed offsets; interior has n·p + offset <= 0.
    return ConvexHull(xy.reshape(-1, 2)).equations


def _gripper_hulls(model, locked_joints):
    """Vertices supporting each locked gripper in every direction, in its palm frame."""
    urdf = URDF.load(str(model.asset_root / "t1_planner.urdf"),
                     filename_handler=lambda fname: str(model.asset_root / fname), load_collision_meshes=False)
    urdf.update_cfg(locked_joints)
    hulls, assets = {}, []
    for arm, palm in TOOL_FRAME_FOR_ARM.items():
        links = [palm, *GRIPPER_JOINT_NAMES[arm]]
        vertices = []
        for link in links:
            nodes = [node for node in urdf.scene.graph.nodes_geometry if urdf.scene.graph.transforms.parents[node] == link]
            if not nodes:
                raise ValueError(f"Missing gripper mesh for {link}")
            for node in nodes:
                _, geometry = urdf.scene.graph[node]
                local = urdf.scene.geometry[geometry].vertices
                pose = urdf.get_transform(node, frame_from=palm)
                vertices.append(local @ pose[:3, :3].T + pose[:3, 3])
            assets.extend(model.asset_root / vis.geometry.mesh.filename for vis in urdf.link_map[link].visuals)
        points = np.concatenate(vertices)
        hulls[arm] = points[ConvexHull(points).vertices]
    return hulls, assets


class T1MapSampler:
    """Sample arm and waist joints about physically planted upright/crouched poses with native cuRobo."""

    def __init__(self, samples=8192, seed=3, device="cuda"):
        if int(samples) != samples or samples < 1:
            raise ValueError("samples must be a positive integer")
        self.samples = int(samples)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Offline T1 FK sampling requires cuRobo on a CUDA device.")
        self.model = REAL_LEGS
        self.cfg = DeviceCfg(device=self.device)
        self.kin = get_t1_kinematics(self.model, self.cfg)
        self.leg_kin = Kinematics(self.kin.config, compute_jacobian=True, compute_spheres=False)
        self._viz_kin = None
        self.self_cost = SelfCollisionCost(SelfCollisionCostCfg(
            weight=self.cfg.to_device([1.0]), device_cfg=self.cfg,
            self_collision_kin_config=self.kin.get_self_collision_config(),
        ))
        self.joint_names = list(self.model.joint_names)
        self.limits = self.kin.get_joint_limits().position.cpu().numpy()
        self.home = np.asarray(self.model.home)
        self.sole_idx = tuple(self.kin.tool_frames.index(link) for link in self.model.sole_frames)
        home_poses = self._feet(self._fk(self.kin, self.cfg.to_device(self.home)))[0].cpu().numpy()
        self.home_soles = np.column_stack((home_poses[:, :2, 3], np.arctan2(home_poses[:, 1, 0], home_poses[:, 0, 0])))
        robot_config = yaml.safe_load(self.model.cfg_path.read_text())
        self.sole_box = robot_config["cutamp"]["sole_box"]
        locked = robot_config["robot_cfg"]["kinematics"]["lock_joints"]
        hulls, mesh_assets = _gripper_hulls(self.model, locked)
        self.gripper_hulls = {arm: torch.as_tensor(points, device=self.device) for arm, points in hulls.items()}
        self.waist_index = self.joint_names.index("Waist")
        urdf = ET.parse(self.model.asset_root / "t1_planner.urdf")
        waist = urdf.find("joint[@name='Waist']")
        foot_box = urdf.find("link[@name='left_foot_link']/collision/geometry/box")
        self.waist_origin = self.cfg.to_device(np.fromstring(waist.find("origin").get("xyz"), sep=" "))
        asset = load_sweep_volume()
        self.contact_offset = self.cfg.to_device((np.array(asset["base_from_G"]) @ [*asset["offset_mid"], 1.0])[:3])
        self.draws = torch.quasirandom.SobolEngine(8, scramble=True, seed=seed).draw(self.samples).to(self.device)
        assets = [self.model.cfg_path, self.model.asset_root / "t1_planner.urdf",
                  self.model.asset_root / "t1_29dof_spheres.yml", ENVELOPES_DIR / "route_a_start.yml", SWEEP_VOLUME_PATH,
                  *mesh_assets]
        self.metadata = {
            "backend": "cuRobo T1 real_legs", "device": torch.cuda.get_device_name(self.device),
            "joint_names": self.joint_names, "seed": seed, "samples_per_arm_posture": self.samples,
            "posture_seed_rad": {name: seed._asdict() for name, seed in POSTURES.items()},
            "gripper_floor_check": "both locked grippers, minimum URDF palm/finger mesh vertex z >= 0 m",
            "gripper_hull_vertices": {arm: len(points) for arm, points in hulls.items()},
            "physical_margin_m": COM_MARGIN_M, "sole_box_m": self.sole_box,
            "foot_height_m": float(foot_box.get("size").split()[2]),
            "sole_position_tolerance_m": SOLE_POS_TOL_M, "sole_angle_tolerance_rad": SOLE_ANGLE_TOL_RAD,
            "asset_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in assets},
        }

    def _fk(self, kin, q):
        return kin.compute_kinematics(JointState.from_position(q.reshape(-1, len(self.joint_names)).contiguous(),
                                                                joint_names=self.joint_names))

    def _feet(self, state):
        return torch.stack([state.tool_poses.get_link_pose(link, make_contiguous=True).get_matrix().reshape(-1, 4, 4)
                            for link in self.model.sole_frames], dim=1)

    def _planted_pose(self, goals, posture):
        # Close the sagittal chain for a flat foot, then refit both legs for the requested split and yaw.
        q0 = self.home.copy()
        q0[3] = posture.trunk_pitch
        hip_pitch = -(posture.trunk_pitch + posture.knee_pitch + posture.ankle_pitch)
        for side in ("Left", "Right"):
            for name, angle in (("Hip_Pitch", hip_pitch), ("Knee_Pitch", posture.knee_pitch),
                                ("Ankle_Pitch", posture.ankle_pitch)):
                q0[self.joint_names.index(f"{side}_{name}")] = angle
        feet = self._feet(self._fk(self.kin, self.cfg.to_device(q0)))[0]
        q0[:3] += (goals[:, :3, 3] - feet[:, :3, 3]).mean(0).cpu().numpy()
        # Keep the trunk pitch and let the floating base translate over both 6-DOF legs.
        active = np.array([i for i, name in enumerate(self.joint_names)
                           if i < 3 or name.startswith(("Left_Hip", "Right_Hip", "Left_Ankle", "Right_Ankle",
                                                       "Left_Knee", "Right_Knee"))])
        lo, hi = self.limits[:, active]
        seed = np.clip(q0[active], lo, hi)
        mask = torch.ones((1, len(q0)), device=self.device)
        cache_x, cache_r, cache_j = None, None, None

        def linearize(x):
            nonlocal cache_x, cache_r, cache_j
            if cache_x is None or not np.array_equal(x, cache_x):
                q = q0.copy()
                q[active] = x
                state = self._fk(self.leg_kin, self.cfg.to_device(q))
                residual, jac = _sole_residual_and_jacobian(state, self.sole_idx, goals, mask)
                cache_x = x.copy()
                # A weak posture prior resolves redundant translations and preserves the requested squat depth.
                cache_r = np.r_[-residual[0].cpu().double().numpy(), .001 * (x - seed)]
                cache_j = np.vstack((jac[0].cpu().double().numpy()[:, active], .001 * np.eye(len(active))))
            return cache_r, cache_j

        out = least_squares(lambda x: linearize(x)[0], seed, jac=lambda x: linearize(x)[1],
                            bounds=(lo, hi), max_nfev=90, gtol=1e-10, ftol=1e-11, xtol=1e-10)
        q0[active] = out.x
        error = linearize(out.x)[0][:12].reshape(2, 6)
        if np.max(np.linalg.norm(error[:, :3], axis=1)) > SOLE_POS_TOL_M:
            return None
        if np.max(np.linalg.norm(error[:, 3:], axis=1)) > SOLE_ANGLE_TOL_RAD:
            return None
        return self.cfg.to_device(q0)

    def _arm_samples(self, posture_q, arm):
        q = posture_q.repeat(self.samples, 1)
        arm_slice = self.model.left_arm if arm == "left" else self.model.right_arm
        lower, upper = self.cfg.to_device(self.limits)
        q[:, arm_slice] = torch.lerp(lower[arm_slice], upper[arm_slice], self.draws[:, :7])
        waist = torch.lerp(lower[self.waist_index], upper[self.waist_index], self.draws[:, 7])
        # Trunk is the URDF root and Waist is its child. Counter-rotate the trunk about the waist pivot
        # to move the arms with waist yaw without changing either leg in world coordinates.
        q[:, self.waist_index] = waist
        q[:, 5] = -waist
        x, y = self.waist_origin[:2]
        c, s = torch.cos(waist), torch.sin(waist)
        # Base angles compose as Ry(pitch) Rx(roll) Rz(yaw), with roll zero in these
        # seeds. Rotate the pivot correction into the world frame to keep the soles fixed.
        dx = x - (c*x + s*y)
        pitch = posture_q[3]
        q[:, 0] += torch.cos(pitch) * dx
        q[:, 1] += y - (-s*x + c*y)
        q[:, 2] -= torch.sin(pitch) * dx
        q[0] = posture_q
        return q

    def _contact(self, state, arm):
        frame = state.tool_poses.get_link_pose(TOOL_FRAME_FOR_ARM[arm], make_contiguous=True)
        mat = frame.get_matrix().reshape(-1, 4, 4)
        return mat[:, :3, :3] @ self.contact_offset + mat[:, :3, 3]

    def _gripper_clearance(self, state, arm):
        frame = state.tool_poses.get_link_pose(TOOL_FRAME_FOR_ARM[arm], make_contiguous=True)
        z_row = frame.get_matrix().reshape(-1, 4, 4)[:, 2].double()
        # The minimum height over a mesh is attained at one of its convex hull vertices.
        return z_row[:, 3] + (z_row[:, :3] @ self.gripper_hulls[arm].T).amin(1)

    @torch.no_grad()
    def map_stance(self, soles):
        """Return accepted witnesses in the supplied stance-local coordinates, one batch per arm and posture."""
        soles = np.asarray(soles, dtype=float)
        if soles.shape != (2, 3) or not np.isfinite(soles).all():
            raise ValueError("soles must contain finite left/right x, y, yaw")
        transforms = _sole_transforms(soles)
        goals = self.cfg.to_device(transforms)
        planes = self.cfg.to_device(_support_planes(transforms, self.sole_box))
        batches = []
        lower, upper = self.cfg.to_device(self.limits)
        for posture, seed in POSTURES.items():
            posture_q = self._planted_pose(goals, seed)
            if posture_q is None:
                continue
            for arm in ("left", "right"):
                chunks = []
                for q in self._arm_samples(posture_q, arm).split(4096):
                    state = self._fk(self.kin, q)
                    spheres = state.robot_spheres
                    self.self_cost.setup_batch_tensors(len(q), 1)
                    collision = self.self_cost.forward(spheres).reshape(len(q), -1).amax(1)
                    min_height = (spheres[..., 2] - spheres[..., 3]).reshape(len(q), -1).amin(1)
                    xy = state.robot_com.reshape(len(q), 4)[:, :2]
                    margin = -(xy @ planes[:, :2].T + planes[:, 2]).amax(1)
                    feet = self._feet(state)
                    pos_error = (feet[:, :, :3, 3] - goals[:, :3, 3]).norm(dim=-1).amax(1)
                    # The Frobenius distance of rotations is 2 sqrt(2) sin(angle/2).
                    rot_error = (feet[:, :, :3, :3] - goals[:, :3, :3]).norm(dim=(-2, -1)).amax(1)
                    valid = ((q >= lower - 1e-7) & (q <= upper + 1e-7)).all(1)
                    valid &= (collision == 0) & (min_height >= 0) & (margin >= COM_MARGIN_M)
                    for hand in self.gripper_hulls:
                        valid &= self._gripper_clearance(state, hand) >= 0
                    valid &= (pos_error <= SOLE_POS_TOL_M) & (rot_error <= 2*np.sqrt(2)*np.sin(SOLE_ANGLE_TOL_RAD/2))
                    points = self._contact(state, arm)
                    chunks.append(tuple(x[valid].cpu().numpy() for x in (points, q, margin)))
                points, configs, margins = (np.concatenate(a) for a in zip(*chunks))
                batches.append({"arm": arm, "posture": posture, "points": points, "q": configs, "margin": margins})
        return batches

    @torch.no_grad()
    def landmarks(self, q):
        """Named polylines for a single saved witness, in the witness's local floor coordinates."""
        if self._viz_kin is None:
            cfg = copy.deepcopy(t1_curobo_cfg(self.model))
            frames = list(dict.fromkeys(list(self.kin.tool_frames) + [link for path in BODY_LINKS.values() for link in path]))
            self._viz_kin = Kinematics(KinematicsCfg.from_robot_yaml_file(cfg, tool_frames=frames, device_cfg=self.cfg),
                                       compute_spheres=False)
            if self._viz_kin.joint_names != self.joint_names:
                raise RuntimeError("Visualization FK joint order differs from the map.")
        state = self._fk(self._viz_kin, self.cfg.to_device(q))
        lines = {}
        for name, path in BODY_LINKS.items():
            xyz = [state.tool_poses.get_link_pose(link).position.reshape(3) for link in path]
            if name.endswith("_arm"):
                xyz.append(self._contact(state, name.split("_")[0])[0])
            lines[name] = torch.stack(xyz).cpu().numpy()
        return lines
