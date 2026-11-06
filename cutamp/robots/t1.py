"""
T1 Robot Module — Single-MotionPlanner Humanoid Support for cuTAMP (cuRobo v0.8)
================================================================================

The planning model of the Booster T1, ``REAL_LEGS`` (a :class:`T1Model`): bfm_wm's pinned 33-DOF T1
(``assets/t1_29dof``, design 7.2), its legs on six massless base joints whose limits are the legs' reach, standing
on the home stance, its limits cut to route A's envelope (``assets/envelopes``, call 22). The 21-DOF simplified
model the paper's ablation arms ran on lives on branch ``ablation-baseline``.

Consumers read names, slices, home and the support region from ``world.model``.

The head (2 DOF) and grippers (4+4 DOF) are construction-time lock_joints in
``t1_29dof.yml``. Both arms expose tool frames ``left_base_link`` and
``right_base_link``; multi-tool-frame planning uses ``GoalToolPose``.

Inactive-arm pinning (during a Pick/Place/Retract that uses one arm) is
handled at planning time in ``cutamp/t1_state.py``, and is a no-op (see
``T1State.pin_for_arm_action``).
"""

import copy
import hashlib
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Dict, Literal, NamedTuple, Tuple, Union

import numpy as np
import rerun as rr
import roma
import torch
import trimesh
from yourdfpy import URDF

from curobo._src.util.config_io import load_yaml
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import DeviceCfg

from cutamp.com_polygon_cost import LOAD_GUARD_A, support_rectangles
from cutamp.robots.envelope import Envelope, load_envelope
from cutamp.utils.rerun_utils import clean_rerun_path, log_scene

# =============================================================================
# Arms, tool frames and grippers
# =============================================================================

LEFT_ARM_JOINT_NAMES: Tuple[str, ...] = (
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw", "Left_Wrist_Pitch",
    "Left_Wrist_Yaw", "Left_Hand_Roll",
)
RIGHT_ARM_JOINT_NAMES: Tuple[str, ...] = (
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw", "Right_Wrist_Pitch",
    "Right_Wrist_Yaw", "Right_Hand_Roll",
)

LEFT_TOOL_FRAME = "left_base_link"
RIGHT_TOOL_FRAME = "right_base_link"
TOOL_FRAMES: Tuple[str, str] = (LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME)
TOOL_FRAME_FOR_ARM: Dict[str, str] = {"left": LEFT_TOOL_FRAME, "right": RIGHT_TOOL_FRAME}

# Gripper (4-bar parallel linkage): 4 joints, 1 effective DOF.
GRIPPER_OPEN: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
GRIPPER_CLOSED: Tuple[float, ...] = (1.0, -1.0, -1.0, 1.0)
# Each hand's four gripper joints, in GRIPPER_OPEN's and GRIPPER_CLOSED's order (t1_planner.urdf names them so).
GRIPPER_JOINT_NAMES: Dict[str, Tuple[str, ...]] = {
    side: tuple(f"{side}_{link}" for link in ("Link1", "Link11", "Link2", "Link22")) for side in ("left", "right")
}

# =============================================================================
# Planning models
# =============================================================================

T1_29DOF_DIR = Path(__file__).parent / "assets" / "t1_29dof"


class Support(NamedTuple):
    """Where the CoM must stay: axis-aligned half-extents about ``center``, in the frame of ``link``."""
    link: str                            # the support frame, z = 0 on the sole plane
    center: Tuple[float, float]          # the region's centre in that frame
    gate_half: Tuple[float, float]       # per-frame trajectory gate
    keyframe_half: Tuple[float, float]   # confs: masks, penalties, GN, trajopt cost


@dataclass(frozen=True, eq=False)        # identity hash: usable in lru_cache keys
class T1Model:
    """The planning model of the T1: its cuRobo config, cspace layout, home and support region."""
    name: str                            # "real_legs", which every saved plan records
    cfg_path: Path                       # the cuRobo robot YAML
    asset_root: Path                     # where its relative asset paths resolve
    joint_names: Tuple[str, ...]         # == world.kinematics.joint_names
    base: slice
    body: slice                          # legs and trunk joints (minimize_body_movement)
    left_arm: slice
    right_arm: slice
    home: Tuple[float, ...]
    support: Support
    sole_frames: Tuple[str, str]         # ("left_sole_link", "right_sole_link")
    envelope: Envelope                   # what the policy may be sent (design 7.10, call 22); its limits are cut to it


def _span(names: Tuple[str, ...], members) -> slice:
    """The slice of ``names`` that ``members`` fill; raises unless they fill it without a gap."""
    idx = sorted(names.index(n) for n in members)
    if not idx or idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError(f"joints {sorted(members)} are not contiguous in {names}")
    return slice(idx[0], idx[-1] + 1)


def _load_real_legs() -> T1Model:
    """The real-leg model (design 7.2), from the ``cutamp`` block that build_planner_model.py writes into
    t1_29dof.yml: cuRobo's joint order for that file's tool frames, home (bfm_wm's default_q, on the base pose
    that stands the soles flat on the floor) and the sole geometry."""
    cfg_path = T1_29DOF_DIR / "t1_29dof.yml"
    block = load_yaml(str(cfg_path))["cutamp"]
    names = tuple(block["joint_order"])
    base = _span(names, [n for n in names if n.startswith("base_j_")])
    left_arm, right_arm = _span(names, LEFT_ARM_JOINT_NAMES), _span(names, RIGHT_ARM_JOINT_NAMES)
    taken = set(names[base] + names[left_arm] + names[right_arm])
    body = _span(names, [n for n in names if n not in taken])
    if (base, left_arm, right_arm, body) != (slice(0, 6), slice(6, 13), slice(13, 20), slice(20, 33)):
        raise ValueError(f"{cfg_path.name}: base, arms and body moved in the joint order {names}")
    home = tuple(float(v) for v in block["home"])
    if len(home) != len(names):
        raise ValueError(f"{cfg_path.name}: {len(home)} home values for {len(names)} joints")
    gate_half, keyframe_half = support_rectangles(tuple(block["sole_box"]), block["sole_y"], LOAD_GUARD_A)
    return T1Model(
        name="real_legs",
        cfg_path=cfg_path,
        asset_root=T1_29DOF_DIR,
        joint_names=names,
        base=base,
        body=body,
        left_arm=left_arm,
        right_arm=right_arm,
        home=home,
        # The stance link sits at the home mid-sole, on the floor: the rectangle is centred on it.
        support=Support("stance", (0.0, 0.0), gate_half, keyframe_half),
        sole_frames=("left_sole_link", "right_sole_link"),
        envelope=load_envelope("route_a_start"),  # by name: route A has no checkpoint to pick it by (design 7.10)
    )


REAL_LEGS = _load_real_legs()


# =============================================================================
# Config loading
# =============================================================================

@lru_cache(maxsize=None)
def t1_curobo_cfg(model: T1Model) -> dict:
    """``model``'s cuRobo robot config, its asset paths resolved to absolutes, without the YAML's ``cutamp``
    block (the model's own data, which cuRobo never reads), its limits cut to ``model.envelope``. Shared:
    deep-copy it before editing."""
    cfg = load_yaml(str(model.cfg_path))
    cfg.pop("cutamp", None)
    kinematics = cfg["robot_cfg"]["kinematics"]

    if isinstance(kinematics.get("urdf_path"), str):
        kinematics["urdf_path"] = str(model.asset_root / kinematics["urdf_path"])
    if isinstance(kinematics.get("collision_spheres"), str):
        kinematics["collision_spheres"] = str(model.asset_root / kinematics["collision_spheres"])
    kinematics["asset_root_path"] = str(model.asset_root)
    _cut_to_envelope(kinematics, model.envelope)
    return cfg


def _cut_to_envelope(kinematics: dict, envelope: Envelope) -> None:
    """Narrow the limits in ``kinematics`` to ``envelope``'s (call 22), in place, never widening one: a Trunk
    height floor, where the envelope sets one, raises ``base_j_z``'s lower limit, and each listed joint's range cuts
    its soft range (the URDF range less ``position_limit_clip`` at both ends, kinematics_loader.py:1119-1120) at
    either end on its own. cuRobo reads a URDF joint's limits only from the URDF, so a cut joint's range is written into
    a copy of it, clip 0, which ``urdf_path`` then names (meshes resolve against ``asset_root_path``). Raises if a range leaves
    nothing of the soft range."""
    if envelope.trunk_height_min is not None:
        base_z = next((link for link in kinematics.get("extra_links", {}).values()
                       if link.get("joint_name") == "base_j_z"), None)
        if base_z is None:
            raise ValueError(f"envelope {envelope.name!r}: no extra link carries base_j_z, whose lower limit is the "
                             "Trunk floor")
        lower, upper = base_z["joint_limits"]
        if envelope.trunk_height_min >= upper:
            raise ValueError(f"envelope {envelope.name!r}: Trunk floor {envelope.trunk_height_min} m >= base_j_z's "
                             f"{upper}")
        base_z["joint_limits"] = [max(lower, envelope.trunk_height_min), upper]

    urdf = ET.parse(kinematics["urdf_path"]).getroot()
    urdf_limits = {joint.get("name"): joint.find("limit") for joint in urdf.findall("joint")}
    cspace = kinematics["cspace"]
    clip = cspace["position_limit_clip"]
    cut = False
    for joint, (lo, hi) in envelope.joint_ranges.items():
        i = cspace["joint_names"].index(joint)
        limit = urdf_limits[joint]
        soft = (float(limit.get("lower")) + clip[i], float(limit.get("upper")) - clip[i])
        new = (max(soft[0], lo), min(soft[1], hi))
        if new[0] >= new[1]:
            raise ValueError(f"envelope {envelope.name!r}: {joint}'s range {(lo, hi)} leaves nothing of its soft range "
                             f"{soft}")
        if new != soft:
            limit.set("lower", repr(float(new[0])))
            limit.set("upper", repr(float(new[1])))
            clip[i] = 0.0
            cut = True
    if cut:  # per user and content, atomically: concurrent runs may write it at once; a file there not ours is replaced
        data = ET.tostring(urdf)
        path = Path(tempfile.gettempdir()) / f"cutamp-{os.getuid()}" / \
            f"{Path(kinematics['urdf_path']).stem}-{envelope.name}-{hashlib.sha1(data).hexdigest()[:16]}.urdf"
        if not path.exists() or path.read_bytes() != data:
            path.parent.mkdir(mode=0o700, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        kinematics["urdf_path"] = str(path)


def _check_joint_order(model: T1Model, joint_names) -> None:
    """Raise unless a build's joints are ``model.joint_names``, in that order (the YAML's lock_joints, the head and
    fingers, are not among them).

    cuRobo orders the cspace by walking the tool frames, then adds ``position_limit_clip`` (the soft limits,
    call 17) without reordering it (``CSpaceParams.inplace_reindex``, cspace_params.py:156-171). A build
    in another order would put the soft limits on the wrong joints, silently, and every consumer indexes q by
    ``model``'s slices."""
    if tuple(joint_names) != model.joint_names:
        raise RuntimeError(f"cuRobo built the {model.name} model's joints as {list(joint_names)}, "
                           f"not {list(model.joint_names)}")


# =============================================================================
# Kinematics, IK, and MotionPlanner factories
# =============================================================================

def get_t1_kinematics(model: T1Model, device_cfg: DeviceCfg = None, compute_jacobian: bool = False) -> Kinematics:
    """``model``'s Kinematics over its whole cspace (both arms and the base, which nothing here locks).

    Always built with ``compute_com=True`` so ``KinematicsState.robot_com``
    is populated on every FK call (COM constraint/gate consumers). Measured
    overhead vs the plain kernel: none (0.126 vs 0.127 ms per 1024-batch FK).

    ``compute_jacobian=True`` additionally populates
    ``KinematicsState.tool_jacobians`` ``[b, h, num_tool_frames, 6, dof]``
    (rows 0-2 linear / 3-5 angular, world frame; verified against finite
    differences). Used by the Gauss-Newton reIK refresh.
    """
    if device_cfg is None:
        device_cfg = DeviceCfg()
    cfg = t1_curobo_cfg(model)
    kin_cfg = KinematicsCfg.from_robot_yaml_file(cfg, device_cfg=device_cfg)
    kin = Kinematics(kin_cfg, compute_com=True, compute_jacobian=compute_jacobian)
    _check_joint_order(model, kin.joint_names)
    return kin


class T1Bodies(NamedTuple):
    """Every massive link of the planning model, for the multibody ZMP (com_polygon_cost.com_zmp_in_support)."""
    kinematics: Kinematics  # tool frames: the support frame (massless), then the links
    mass: torch.Tensor      # [L] kg
    com: torch.Tensor       # [L, 3] in the link frame, m
    inertia: torch.Tensor   # [L, 3, 3] about the CoM, in the link frame, kg m^2


def get_t1_bodies(model: T1Model, device_cfg: DeviceCfg = None) -> T1Bodies:
    """:class:`T1Bodies` of ``model``: its support frame (``model.support.link``), then every massive link. Masses
    and CoMs are cuRobo's, the ones every ``robot_com`` sums; inertias are read from the URDF, and equal the ones
    cuRobo keeps (``test_curobo_link_inertias_match_the_urdf``). A frame the URDF lacks, such as ``stance``, weighs
    nothing."""
    if device_cfg is None:
        device_cfg = DeviceCfg()
    cfg = copy.deepcopy(t1_curobo_cfg(model))  # from_robot_yaml_file writes tool_frames into the dict it is given
    kp = KinematicsCfg.from_robot_yaml_file(cfg, device_cfg=device_cfg).kinematics_config
    support = model.support.link
    names = [support] + [n for n, i in kp.link_name_to_idx_map.items() if kp.link_masses_com[i, 3] > 0 and n != support]
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(cfg, tool_frames=names, device_cfg=device_cfg),
                     compute_spheres=False)
    _check_joint_order(model, kin.joint_names)
    kp = kin.config.kinematics_config
    mass_com = kp.link_masses_com[[kp.link_name_to_idx_map[n] for n in names]]
    urdf = URDF.load(cfg["robot_cfg"]["kinematics"]["urdf_path"], load_meshes=False, build_scene_graph=False)
    inertia = np.zeros((len(names), 3, 3))
    for k, n in enumerate(names):
        inertial = urdf.link_map[n].inertial if n in urdf.link_map else None
        if inertial is not None:
            r = inertial.origin[:3, :3]
            inertia[k] = r @ inertial.inertia @ r.T
    return T1Bodies(kin, mass_com[:, 3], mass_com[:, :3], mass_com.new_tensor(inertia))


def _trajopt_transition_dict_with_compute_com() -> dict:
    """Default trajopt transition YAML with ``compute_com=True`` injected.

    The COM-over-base cost reads ``state.robot_com`` (per-link mass-weighted
    world COM); cuRobo only populates this when the runtime kinematics is
    built with ``compute_com=True``. The kernel template is fixed at planner
    build time, so we have to flip this before MotionPlanner construction.
    """
    from curobo._src.util.config_io import resolve_config, join_path
    from curobo.content import get_task_configs_path

    d = resolve_config(join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml"))
    d["transition_model_cfg"]["compute_com"] = True
    return d


TRAJOPT_POSITION_BOUND_WEIGHT = 1e6
"""Trajopt's joint-position bound cost weight, cuRobo's 1e4 times 100. The bound is a soft cost there (active within
1% of a joint's range of it) and the success test a hard one; home's ankle pitch is the envelope's cap, so every
return to home ends on it, and at 1e4 paths crossed it mid-motion (by up to 0.014 rad) and failed: 12 of 15 blocks_t1
cases planned, 15 of 15 at 1e6 (2026-11-06)."""


def _trajopt_optimizer_dict() -> dict:
    """cuRobo's trajopt optimizer YAML with the position bound's weight at ``TRAJOPT_POSITION_BOUND_WEIGHT``."""
    from curobo._src.util.config_io import resolve_config, join_path
    from curobo.content import get_task_configs_path

    d = resolve_config(join_path(get_task_configs_path(), "trajopt/lbfgs_bspline_trajopt.yml"))
    d["rollout"]["cost_cfg"]["cspace_cfg"]["weight"][0] = TRAJOPT_POSITION_BOUND_WEIGHT  # position, vel, acc, jerk, tau
    return d


def get_t1_motion_planner(
    model: T1Model,
    scene: Scene,
    *,
    use_cuda_graph: bool = True,
    collision_activation_distance: float = 0.01,
    device_cfg: DeviceCfg = None,
) -> MotionPlanner:
    """Single MotionPlanner for the T1, with every joint free, the base included.

    The planner also (design 7.4a and 7.5):

    - runs cuRobo's RNEA in trajopt (``load_dynamics``), whose cspace cost and success test bound every joint's torque
      at the URDF's effort, with the floor's push on the planted soles (``install_floor_wrench``, hooked before any
      plan or warm-up: cuRobo's CUDA graphs keep the hook they were captured with). cuRobo's energy term keeps its
      default weight;
    - holds both soles on the stance at every waypoint (pose tracking at non-terminal scale 1.0, stage 2's verified
      weight), tracks the hands at the terminal only, and leaves the ``stance`` frame untracked (it is world-fixed,
      and the CoM cost's support frame);
    - bounds the Trunk tilt (``TrunkTiltCost``, weight 1e5) when the envelope sets a maximum. Route A's sets none.

    The planner gets a soft cost penalizing trajectories whose mass-weighted
    COM projects outside the model's keyframe rectangle (``model.support``),
    on the kinematics kernel built with ``compute_com=True``.

    The joint limits are bfm_wm's soft limits cut by the envelope, with no further margin (call 17); trajopt bounds
    them at ``TRAJOPT_POSITION_BOUND_WEIGHT``.
    """
    if device_cfg is None:
        device_cfg = DeviceCfg()
    cfg_dict = copy.deepcopy(t1_curobo_cfg(model))
    cfg_dict["robot_cfg"]["load_dynamics"] = True
    cfg = MotionPlannerCfg.create(
        robot=cfg_dict,
        scene_model=scene,
        use_cuda_graph=use_cuda_graph,
        num_ik_seeds=64,
        num_trajopt_seeds=4,
        optimizer_collision_activation_distance=collision_activation_distance,
        max_batch_size=1,
        max_goalset=1,
        device_cfg=device_cfg,
        trajopt_transition_model=_trajopt_transition_dict_with_compute_com(),
        trajopt_optimizer_configs=[_trajopt_optimizer_dict()],
    )
    planner = MotionPlanner(cfg)
    _check_joint_order(model, planner.kinematics.joint_names)
    from curobo.types import ToolPoseCriteria
    from cutamp.floor_wrench import install_floor_wrench

    install_floor_wrench(planner, model.joint_names[model.base], model.sole_frames)
    criteria = {
        frame: ToolPoseCriteria.track_position_and_orientation(non_terminal_scale=0.0)  # the hands
        for frame in planner.kinematics.tool_frames
    }
    criteria.update({
        frame: ToolPoseCriteria.track_position_and_orientation(non_terminal_scale=1.0)
        for frame in model.sole_frames
    })
    criteria[model.support.link] = ToolPoseCriteria.disabled()
    planner.update_tool_pose_criteria(criteria)
    from cutamp._curobo_internals import add_extra_cost
    from cutamp.com_polygon_cost import (
        ComOverBasePolygonCost,
        ComOverBasePolygonCostCfg,
        COM_INSIDE_MARGIN,
        COM_INSIDE_WEIGHT,
    )
    # Tuned on the simplified model (branch ablation-baseline) and kept on the real legs:
    # COM-polygon cost is a SAFETY BARRIER: zero cost anywhere inside
    # the two-foot support rectangle, quadratic-in-distance penalty
    # outside. The joint-level costs are what keep the body near upright
    # — having COM ALSO try to pull toward center produced fighting costs (COM "lean back to recenter" vs body "stay
    # upright"), which manifested as the trunk pitching backward when
    # arms reach forward. Boundary-only shape lets those costs
    # dictate posture; COM only intervenes if the body actually drifts
    # to a tipping configuration. Bumped weight 1e5 → 5e5 to push back
    # harder on excursions without re-introducing the inside barrier.
    # Inside-barrier ON for the planner rollouts too: keyframe-only COM
    # filter (Adam side) is satisfied, but the saved trajectory's mid-
    # waypoint COM was excursing up to 5cm out of polygon (89% inside,
    # ~58/528 frames out, peak penalty 2.4e-3 in place segments). Adding
    # the inside-barrier here gives the trajopt LBFGS a gradient pull
    # from inside the polygon edge band toward center, flattening the
    # mid-trajectory bumps. The earlier concern (this fighting
    # the joint-level costs) is validated by retesting; if trunk pitches
    # back too much we lower inside_weight from 1.0.
    rollout_device_cfg = planner.trajopt_solver.optimizer_rollouts[0].device_cfg
    cost_cfg = ComOverBasePolygonCostCfg(
        weight=[5.0e5],
        device_cfg=rollout_device_cfg,
        half_extents=torch.tensor(model.support.keyframe_half),
        inside_margin=COM_INSIDE_MARGIN,
        inside_weight=COM_INSIDE_WEIGHT,
        base_link_name=model.support.link,
        center=model.support.center,
    )
    add_extra_cost(planner, "com_polygon", ComOverBasePolygonCost(cost_cfg))
    if model.envelope.trunk_tilt_max is not None:
        from cutamp.joint_limit_cost import TrunkTiltCost, TrunkTiltCostCfg

        # The envelope's Trunk tilt bound (call 22) along the trajectory: GN and the Motion value hold it at the confs.
        active_names = list(planner.kinematics.joint_names)
        tilt_cfg = TrunkTiltCostCfg(
            weight=[1.0e5],
            device_cfg=planner.trajopt_solver.optimizer_rollouts[0].device_cfg,
            pitch_index=active_names.index("base_j_pitch"),
            roll_index=active_names.index("base_j_roll"),
            cos_max=math.cos(model.envelope.trunk_tilt_max),
        )
        add_extra_cost(planner, "trunk_tilt", TrunkTiltCost(tilt_cfg))
    return planner


# =============================================================================
# Tool frames and gripper spheres
# =============================================================================

def get_t1_tool_from_ee(
    device_cfg: DeviceCfg = None,
) -> Dict[str, torch.Tensor]:
    """``world_from_ee = world_from_grasp @ tool_from_ee``.

    The T1 EE frames are ``left_base_link`` / ``right_base_link``. In each EE
    frame, +X points toward the fingertips. Ry(+90°) maps EE +X → grasp -Z, so
    a top-down grasp has the gripper pointing DOWN. Both arms share the same
    tool transform.
    """
    if device_cfg is None:
        device_cfg = DeviceCfg()
    device = device_cfg.device
    rotvec_y_pos90 = torch.tensor([[0.0, torch.pi / 2, 0.0]], device=device, dtype=torch.float32)
    grasp_from_ee = roma.rotvec_to_rotmat(rotvec_y_pos90)[0]
    tool_from_ee = torch.eye(4, device=device, dtype=torch.float32)
    tool_from_ee[:3, :3] = grasp_from_ee
    tool_from_ee[:3, 3] = torch.tensor([0.01, 0.0, 0.09], device=device, dtype=torch.float32)
    return {LEFT_TOOL_FRAME: tool_from_ee, RIGHT_TOOL_FRAME: tool_from_ee.clone()}


def get_t1_gripper_spheres(
    arm: Literal["left", "right"],
    device_cfg: DeviceCfg = None,
) -> torch.Tensor:
    """``[n, 4]`` collision spheres of the named gripper, in its base frame: the ``{arm}_base_link`` spheres of
    ``t1_29dof_spheres.yml``, the file cuRobo loads (the finger links hold none)."""
    if arm not in ("left", "right"):
        raise ValueError(f"Invalid arm: {arm}")
    if device_cfg is None:
        device_cfg = DeviceCfg()
    spheres = load_yaml(str(T1_29DOF_DIR / "t1_29dof_spheres.yml"))["collision_spheres"][TOOL_FRAME_FOR_ARM[arm]]
    return torch.tensor([[*s["center"], s["radius"]] for s in spheres], dtype=torch.float32, device=device_cfg.device)


# =============================================================================
# Visualization
# =============================================================================

def trunk_pose_from_q(model: T1Model, q) -> torch.Tensor:
    """``[..., 4, 4]`` float64 world_from_Trunk of confs ``q`` ``[..., len(model.joint_names)]``.

    The floating base's six joints (x, y, z, pitch, roll, yaw) chain ``base_link_z`` -> ``base_link_pitch``
    (Y_ROT) -> ``base_link_roll`` (X_ROT) -> ``base_link_yaw`` (Z_ROT) -> ``Trunk`` with identity offsets
    (t1_29dof.yml), so the Trunk sits at (x, y, z) with R = Ry(pitch) Rx(roll) Rz(yaw). Use to anchor the URDF
    visualization at the planned base pose.
    """
    q = torch.as_tensor(q, dtype=torch.float64)
    base = q[..., [model.joint_names.index(f"base_j_{a}") for a in ("x", "y", "z", "pitch", "roll", "yaw")]]
    pose = torch.eye(4, dtype=torch.float64).repeat(*q.shape[:-1], 1, 1)
    pose[..., :3, :3] = roma.euler_to_rotmat("YXZ", base[..., 3:])  # intrinsic: Ry(pitch) Rx(roll) Rz(yaw)
    pose[..., :3, 3] = base[..., :3]
    return pose


def _get_scene_transforms(
    scene: trimesh.Scene, root_node: str, root_path: Union[str, None] = None
) -> Dict[str, np.ndarray]:
    """Get the transformations for each link in the scene in the expected rerun keys that we use."""
    stack = [(root_node, root_path)]
    path_to_transform = {}

    while stack:
        node, path = stack.pop()
        rerun_path = f"{path}/{node}" if path else node
        rerun_path = clean_rerun_path(rerun_path)

        # Get parent and children for this node
        parent = scene.graph.transforms.parents.get(node)
        children = scene.graph.transforms.children.get(node)
        node_data = scene.graph.get(frame_to=node, frame_from=parent)

        if node_data and parent:
            assert rerun_path not in path_to_transform
            path_to_transform[rerun_path] = node_data[0]

        # Add children to stack
        if children:
            for child in children:
                stack.append((child, rerun_path))

    return path_to_transform


class T1RerunRobot:
    """Rerun URDF robot for the T1 on ``model``, its base pose logged as a ``Transform3D`` on the robot's root
    entity (``name``), so the rendered robot moves with the planner.

    It accepts the model's 33-DOF conf, in ``model.joint_names`` order. The six base joints, cuRobo
    ``extra_links`` the URDF doesn't include, give the Trunk's world pose (:func:`trunk_pose_from_q`). Every URDF
    joint is set by name: the model's from the conf, the fingers from the gripper state (:meth:`set_grippers`), and
    the others the planner locks (the head) at their locked values, 0.
    """

    def __init__(self, name: str, urdf: URDF, model: T1Model, q_neutral=None, load_mesh: bool = True):
        self.name = name
        self.urdf = urdf
        self.model = model
        self._left_gripper: Tuple[float, ...] = GRIPPER_OPEN
        self._right_gripper: Tuple[float, ...] = GRIPPER_OPEN
        self._locked = dict(t1_curobo_cfg(model)["robot_cfg"]["kinematics"].get("lock_joints") or {})
        unset = set(urdf.actuated_joint_names) - set(model.joint_names) - set(self._locked)
        if unset:
            raise ValueError(f"URDF joints {sorted(unset)} are neither in the {model.name} model nor locked")
        rr.log(self.urdf.base_link, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        log_scene(scene=self.urdf.scene, node=self.urdf.base_link, path=self.name, static=False, add_mesh=load_mesh)
        if q_neutral is not None:
            self.set_joint_positions(q_neutral)

    def set_grippers(self, left=GRIPPER_OPEN, right=GRIPPER_OPEN) -> None:
        self._left_gripper = tuple(left)
        self._right_gripper = tuple(right)

    def _log_base_transform(self, q_curobo) -> None:
        """Apply the Trunk pose of the conf ``q_curobo`` to the robot's root entity."""
        m = trunk_pose_from_q(self.model, q_curobo).numpy()
        rr.log(self.name, rr.Transform3D(translation=m[:3, 3], mat3x3=m[:3, :3]))

    def _real_legs_conf(self, joint_positions: list) -> list:
        """``joint_positions``, checked to be a conf: the model's joints, in ``model.joint_names`` order."""
        n = len(self.model.joint_names)
        if len(joint_positions) != n:
            raise ValueError(f"Expected the {self.model.name} model's {n} joints, got {len(joint_positions)} values")
        return joint_positions

    def _real_legs_urdf_cfg(self, q: list) -> list:
        """Every one of the URDF's actuated joints, by name, at the conf ``q``."""
        value = {**self._locked, **dict(zip(self.model.joint_names, q))}
        for side, state in (("left", self._left_gripper), ("right", self._right_gripper)):
            value.update(zip(GRIPPER_JOINT_NAMES[side], state))
        return [value[n] for n in self.urdf.actuated_joint_names]

    def set_joint_positions(self, joint_positions) -> None:
        if hasattr(joint_positions, "tolist"):
            joint_positions = joint_positions.tolist()
        elif hasattr(joint_positions, "__iter__"):
            joint_positions = list(joint_positions)

        q = self._real_legs_conf(joint_positions)
        self._log_base_transform(q)
        self.urdf.update_cfg(self._real_legs_urdf_cfg(q))
        log_scene(scene=self.urdf.scene, node=self.urdf.base_link, path=self.name, static=False, add_mesh=False)

    def get_rr_columns(self, joint_positions: torch.Tensor) -> Dict[str, rr.Transform3D.columns]:
        """
        Gets the Rerun columns for a batch of joint positions for use in columnar logging. This is especially helpful
        for logging joint trajectories, as it's much faster than individual row-oriented logging.

        See rerun documentation for more details: https://rerun.io/docs/howto/logging/send-columns

        Parameters
        ----------
        joint_positions: torch.Tensor
           Batch of joint positions for which to get the rerun components.

        Returns
        -------
        Dict[str, rr.Transform3D.columns]
           Mapping of the rerun path to the columns for each joint position.
        """
        if joint_positions.ndim != 2:
            raise ValueError(f"joint_positions must be a 2D tensor of shape (n, d), got {joint_positions.shape}")

        confs = [self._real_legs_conf(q) for q in joint_positions.tolist()]
        base_mats = trunk_pose_from_q(self.model, confs).numpy()
        base_columns = rr.Transform3D.columns(mat3x3=base_mats[:, :3, :3], translation=base_mats[:, :3, 3])
        joint_positions = torch.tensor([self._real_legs_urdf_cfg(q) for q in confs], dtype=torch.float64)

        path_to_tforms = {}  # map of path to list of transforms

        # Update the URDF with joint position and get the transformations
        for q in joint_positions:
            self.urdf.update_cfg(q.numpy())
            for path, tform in _get_scene_transforms(self.urdf.scene, self.urdf.base_link, self.name).items():
                if path not in path_to_tforms:
                    path_to_tforms[path] = []
                path_to_tforms[path].append(tform)
        path_to_tforms = {k: np.stack(v) for k, v in path_to_tforms.items()}

        # Convert transformations into rerun components
        path_to_components = {}
        for path, tforms in path_to_tforms.items():
            components = rr.Transform3D.columns(mat3x3=tforms[:, :3, :3], translation=tforms[:, :3, 3])
            path_to_components[path] = components
        path_to_components[self.name] = base_columns
        return path_to_components


def load_t1_rerun(model: T1Model, load_mesh: bool = True) -> T1RerunRobot:
    """Load ``model``'s URDF, the one its cuRobo config plans on, for Rerun visualization, at home."""
    urdf_path = Path(t1_curobo_cfg(model)["robot_cfg"]["kinematics"]["urdf_path"])
    if not urdf_path.exists():
        raise FileNotFoundError(f"T1 URDF not found at {urdf_path}")

    def _locate_t1_asset(fname: str) -> str:
        full_path = model.asset_root / fname
        return str(full_path) if full_path.exists() else fname

    urdf = URDF.load(str(urdf_path), filename_handler=_locate_t1_asset)
    return T1RerunRobot("t1", urdf, model, q_neutral=model.home, load_mesh=load_mesh)
