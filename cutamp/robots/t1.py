"""
T1 Robot Module — Single-MotionPlanner Humanoid Support for cuTAMP (cuRobo v0.8)
================================================================================

Booster T1 humanoid as a single 21-DOF MotionPlanner. The planner sees:

    - 3 DOF planar base (base_j_x, base_j_y, base_j_yaw, virtual via extra_links)
    - 2 DOF leg (ankle_pitch, knee_pitch — revolute, mounted on the mobile base)
    - 2 DOF torso (Torso_Pitch — abstracts coordinated Hip_Pitch; Waist_Yaw)
    - 7 DOF left arm (Left_Shoulder_Pitch ... Left_Hand_Roll)
    - 7 DOF right arm (Right_Shoulder_Pitch ... Right_Hand_Roll)

The head (2 DOF) and grippers (4+4 DOF) are construction-time lock_joints in
``t1_planar_base.yml``. Both arms expose tool frames ``left_base_link`` and
``right_base_link``; multi-tool-frame planning uses ``GoalToolPose``.

Inactive-arm pinning (during a Pick/Place/Retract that uses one arm) is
handled at planning time in ``cutamp/t1_state.py``, and is a no-op (see
``T1State.pin_for_arm_action``).
"""

import copy
import logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, Literal, Tuple, Union

import numpy as np
import rerun as rr
import roma
import torch
import trimesh
from yourdfpy import URDF

from curobo._src.util.config_io import load_yaml
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import DeviceCfg

from cutamp.utils.common import action_4dof_to_mat4x4
from cutamp.utils.rerun_utils import clean_rerun_path, log_scene

_log = logging.getLogger(__name__)

# =============================================================================
# DOF layout
# =============================================================================

CUROBO_DOF = 21          # 3 base + 2 leg + 2 torso + 7 left arm + 7 right arm
URDF_DOF = 28            # Total actuated joints in the URDF (no virtual base joints)
BASE_DOF = 3

# Slices into a 21-DOF cspace vector (must match cspace.joint_names ordering in
# t1_planar_base.yml).
BASE_INDICES = slice(0, 3)
BODY_INDICES = slice(3, 7)           # leg + torso (everything between base and arms)
LEFT_ARM_INDICES = slice(7, 14)
RIGHT_ARM_INDICES = slice(14, 21)

# Cspace joint names (must match t1_planar_base.yml order).
JOINT_NAMES_FULL: Tuple[str, ...] = (
    "base_j_x", "base_j_y", "base_j_yaw",
    "ankle_pitch", "knee_pitch",
    "Torso_Pitch", "Waist_Yaw",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
    "Left_Elbow_Yaw", "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
    "Right_Elbow_Yaw", "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
)

LEFT_ARM_JOINT_NAMES: Tuple[str, ...] = JOINT_NAMES_FULL[LEFT_ARM_INDICES]
RIGHT_ARM_JOINT_NAMES: Tuple[str, ...] = JOINT_NAMES_FULL[RIGHT_ARM_INDICES]

LEFT_TOOL_FRAME = "left_base_link"
RIGHT_TOOL_FRAME = "right_base_link"
TOOL_FRAMES: Tuple[str, str] = (LEFT_TOOL_FRAME, RIGHT_TOOL_FRAME)
TOOL_FRAME_FOR_ARM: Dict[str, str] = {"left": LEFT_TOOL_FRAME, "right": RIGHT_TOOL_FRAME}

# Gripper (4-bar parallel linkage): 4 joints, 1 effective DOF.
GRIPPER_OPEN: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
GRIPPER_CLOSED: Tuple[float, ...] = (1.0, -1.0, -1.0, 1.0)

# Home pose: base at origin; leg standing straight (max height); torso pitch +
# yaw neutral; arms in neutral pre-grasp.
t1_home: Tuple[float, ...] = (
    0.0, 0.0, 0.0,                              # base
    0.0, 0.0,                                   # ankle_pitch, knee_pitch
    0.0, 0.0,                                   # Torso_Pitch, Waist_Yaw
    0.5, -1.0, 0.0, -1.4, 0.0, 0.0, 0.0,        # left arm
    0.5,  1.0, 0.0,  1.4, 0.0, 0.0, 0.0,        # right arm
)
assert len(t1_home) == CUROBO_DOF

# Locked head joint values (exposed for the visualizer / URDF expansion).
t1_home_head: Tuple[float, ...] = (0.0, 0.0)

T1_ASSETS_DIR = Path(__file__).parent / "assets" / "t1_description"
T1_CONFIG_PATH = T1_ASSETS_DIR / "t1_planar_base.yml"

# Seed-IK CoM residual weights. Constraints that bound them:
# - Pose/joint-limit weights in the seed-IK solver are O(1); the planner's
#   5e5 rollout weight is NOT transferable here.
# - The fork re-solves the problems the seed LM fails with cuRobo's CoM-blind
#   optimizer, so terms strong enough to hurt seed convergence lose centring
#   on those problems (before known-bugs N7, one failure re-solved them all).
# Sweeps: commits 473ec51 (support weight) and 5b9ae65 (center weight). The
# known-bugs Task 6 re-check (corrected CoM, N7 fixed) solves the 48-pose grid
# in every cell of 0.3-3 x 3e-4-3e-3, so these stay; (1.0, 3e-3) centres to
# 0.11 cm at a similar pose error but is untried end to end.
T1_COM_IK_WEIGHT = 1.0

# Pull-to-center term: cw * sum((com/half)^2), active everywhere (not just
# the edge band). On the 48-pose grid: mean |CoM x| 0.88 cm, against 2.49 cm
# with the CoM residual off (2.8x).
T1_COM_IK_CENTER_WEIGHT = 1e-3

# =============================================================================
# Joint-limit margins
# =============================================================================
# Per-side soft-margin bands (rad) from each joint's position limit, in full
# cspace order (JOINT_NAMES_FULL). Single source of truth for BOTH margin
# costs (spec: docs/superpowers/specs/2026-06-11-joint-limit-margin-design.md):
#   - cutamp soft cost "joint_limit_margin" over Conf particles
#     (segment endpoints — the load-bearing fix: empirically 100% of
#     near-limit danger in saved plans comes from endpoint configs)
#   - JointLimitMarginCost on the motion planner (trajectory insurance)
# 0.0 on the home-side bounds: the standing home posture sits exactly ON the
# ankle_pitch-upper / knee_pitch-lower / Torso_Pitch-upper limits by design
# (straight leg, t1_simplified.urdf:79,116,159), so margining those sides
# would penalize standing still. 0.0 on the virtual base DOFs (locked).
T1_JOINT_LIMIT_MARGIN = 0.1

_M = T1_JOINT_LIMIT_MARGIN
T1_LIMIT_MARGIN_LOWER: Tuple[float, ...] = (
    0.0, 0.0, 0.0,                    # base_j_x, base_j_y, base_j_yaw
    _M, 0.0,                          # ankle_pitch, knee_pitch (home == lower limit)
    _M, _M,                           # Torso_Pitch, Waist_Yaw
    _M, _M, _M, _M, _M, _M, _M,       # left arm
    _M, _M, _M, _M, _M, _M, _M,       # right arm
)
T1_LIMIT_MARGIN_UPPER: Tuple[float, ...] = (
    0.0, 0.0, 0.0,                    # base_j_x, base_j_y, base_j_yaw
    0.0, _M,                          # ankle_pitch (home == upper limit), knee_pitch
    0.0, _M,                          # Torso_Pitch (home == upper limit), Waist_Yaw
    _M, _M, _M, _M, _M, _M, _M,       # left arm
    _M, _M, _M, _M, _M, _M, _M,       # right arm
)
assert len(T1_LIMIT_MARGIN_LOWER) == CUROBO_DOF
assert len(T1_LIMIT_MARGIN_UPPER) == CUROBO_DOF
del _M

# Trajopt-only band on ankle_pitch's upper bound, which the margins above
# leave at 0. With a zero band the COM-polygon cost leans the body back past
# 0 late in the path (seed 1: +0.9 to +5.7 mrad), and cuRobo rejects the
# plan. 0.02 removes that bump: the ankle then reaches its goal from below,
# after a deeper mid-path dip (seed 1: -39 vs -12 mrad). 0.01 still left
# +0.1 mrad. Start and goal are pinned, so the band only shapes the interior.
T1_TRAJOPT_ANKLE_UPPER_MARGIN = 0.02


# =============================================================================
# Config loading
# =============================================================================

@lru_cache(maxsize=1)
def t1_curobo_cfg() -> dict:
    """Load the T1 v0.8 robot config and resolve asset paths to absolutes."""
    cfg = load_yaml(str(T1_CONFIG_PATH))
    kinematics = cfg["robot_cfg"]["kinematics"]

    if isinstance(kinematics.get("urdf_path"), str):
        kinematics["urdf_path"] = str(T1_ASSETS_DIR / kinematics["urdf_path"])
    if isinstance(kinematics.get("collision_spheres"), str):
        kinematics["collision_spheres"] = str(T1_ASSETS_DIR / kinematics["collision_spheres"])
    kinematics["asset_root_path"] = str(T1_ASSETS_DIR)
    return cfg


def _lock_mobile_base(cfg_dict: dict) -> dict:
    """Extend ``lock_joints`` to fix the 3 base DOFs at 0. Mutates in place.

    Locked joints are removed from cspace, so IK and trajopt cannot drift the
    base during arm operations.
    """
    lock_joints = cfg_dict["robot_cfg"]["kinematics"].setdefault("lock_joints", {})
    for name in JOINT_NAMES_FULL[BASE_INDICES]:
        lock_joints[name] = 0.0
    return cfg_dict


# =============================================================================
# Kinematics, IK, and MotionPlanner factories
# =============================================================================

def get_t1_kinematics(device_cfg: DeviceCfg = None, compute_jacobian: bool = False) -> Kinematics:
    """Single 21-DOF Kinematics for T1 (both arms + planar base).

    Always built with ``compute_com=True`` so ``KinematicsState.robot_com``
    is populated on every FK call (COM constraint/gate consumers). Measured
    overhead vs the plain kernel: none (0.126 vs 0.127 ms per 1024-batch FK).

    ``compute_jacobian=True`` additionally populates
    ``KinematicsState.tool_jacobians`` ``[b, h, num_tool_frames, 6, 21]``
    (rows 0-2 linear / 3-5 angular, world frame; verified against finite
    differences). Used by the Gauss-Newton reIK refresh.
    """
    if device_cfg is None:
        device_cfg = DeviceCfg()
    cfg = t1_curobo_cfg()
    kin_cfg = KinematicsCfg.from_robot_yaml_file(cfg, device_cfg=device_cfg)
    return Kinematics(kin_cfg, compute_com=True, compute_jacobian=compute_jacobian)


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


def get_t1_motion_planner(
    scene: Scene,
    *,
    use_cuda_graph: bool = True,
    collision_activation_distance: float = 0.01,
    enable_com_polygon: bool = True,
    enable_joint_limit_margin: bool = True,
    device_cfg: DeviceCfg = None,
) -> MotionPlanner:
    """Single MotionPlanner for the T1 with the mobile base locked.

    Legs (ankle/knee), Torso_Pitch, Waist_Yaw, and both arms remain free.

    When ``enable_com_polygon`` is True (default), the planner gets a soft
    cost penalizing trajectories whose mass-weighted COM projects outside a
    20×30 cm rectangle in ``mobile_base_link`` frame. Note the kinematics
    kernel always builds with ``compute_com=True`` (cheap, useful for any
    future COM-based cost or diagnostic) — the flag only gates registration
    of the cost itself.

    ``enable_joint_limit_margin`` (default True) additionally registers a
    hinge² cost that keeps the trajectory interior a soft 0.1 rad band away
    from joint position limits (zero band on the home-side bounds, except
    ``T1_TRAJOPT_ANKLE_UPPER_MARGIN`` on ankle-upper).
    """
    if device_cfg is None:
        device_cfg = DeviceCfg()
    cfg_dict = _lock_mobile_base(copy.deepcopy(t1_curobo_cfg()))
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
    )
    planner = MotionPlanner(cfg)
    if enable_com_polygon:
        from cutamp._curobo_internals import add_extra_cost
        from cutamp.com_polygon_cost import (
            ComOverBasePolygonCost,
            ComOverBasePolygonCostCfg,
            COM_HALF_EXTENTS,
            COM_INSIDE_MARGIN,
            COM_INSIDE_WEIGHT,
        )
        # COM-polygon cost is a SAFETY BARRIER: zero cost anywhere inside
        # the two-foot support rectangle, quadratic-in-distance penalty
        # outside. ``body_home_posture`` is what keeps the body near upright
        # at the joint level — having COM ALSO try to pull toward center
        # produced fighting costs (COM "lean back to recenter" vs body "stay
        # upright"), which manifested as the trunk pitching backward when
        # arms reach forward. Boundary-only shape lets body_home_posture
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
        # body_home_posture) is validated by retesting; if trunk pitches
        # back too much we lower inside_weight from 1.0.
        rollout_device_cfg = planner.trajopt_solver.optimizer_rollouts[0].device_cfg
        cost_cfg = ComOverBasePolygonCostCfg(
            weight=[5.0e5],
            device_cfg=rollout_device_cfg,
            half_extents=torch.tensor(COM_HALF_EXTENTS),
            inside_margin=COM_INSIDE_MARGIN,
            inside_weight=COM_INSIDE_WEIGHT,
        )
        add_extra_cost(planner, "com_polygon", ComOverBasePolygonCost(cost_cfg))
    if enable_joint_limit_margin:
        from cutamp._curobo_internals import add_extra_cost
        from cutamp.joint_limit_cost import JointLimitMarginCost, JointLimitMarginCostCfg

        # Joint-limit margin: hinge² on entering a per-joint/per-side band
        # inside the position limits (margins: T1_LIMIT_MARGIN_LOWER/UPPER
        # above). Endpoint configs are handled by the cutamp-side soft cost;
        # this keeps the trajectory INTERIOR away from limits between
        # endpoints. Home-side bounds carry zero margin, so the home dwell
        # waypoints that legitimately sit ON the knee/torso bounds cost
        # nothing; ankle-upper gets T1_TRAJOPT_ANKLE_UPPER_MARGIN instead
        # (see there). Optimizer-objective only: extra costs never reach
        # the feasibility/success gate (see _curobo_internals.add_extra_cost).
        rollout = planner.trajopt_solver.optimizer_rollouts[0]
        limits = rollout.transition_model.joint_limits.position  # [2, dof_active]
        active_names = list(planner.kinematics.joint_names)
        assert len(active_names) == limits.shape[1], (active_names, limits.shape)
        name_to_full = {n: i for i, n in enumerate(JOINT_NAMES_FULL)}
        missing = [n for n in active_names if n not in name_to_full]
        assert not missing, f"planner joints missing from JOINT_NAMES_FULL: {missing}"
        full_idx = [name_to_full[n] for n in active_names]
        m_lo = torch.tensor(
            [T1_LIMIT_MARGIN_LOWER[i] for i in full_idx],
            device=limits.device, dtype=limits.dtype,
        )
        m_hi = torch.tensor(
            [T1_LIMIT_MARGIN_UPPER[i] for i in full_idx],
            device=limits.device, dtype=limits.dtype,
        )
        m_hi[active_names.index("ankle_pitch")] = T1_TRAJOPT_ANKLE_UPPER_MARGIN
        jl_weight = 1.0e5  # tuned 1e4→1e5 in verification: kills interior excursions (cuRobo's own bound cost runs 1e4)
        jl_cfg = JointLimitMarginCostCfg(
            weight=[jl_weight],
            device_cfg=rollout.device_cfg,
            shrunk_lower=limits[0] + m_lo,
            shrunk_upper=limits[1] - m_hi,
        )
        add_extra_cost(planner, "joint_limit_margin", JointLimitMarginCost(jl_cfg))
        _log.info(
            "Registered joint_limit_margin trajopt cost (weight %.0e, band %.2f rad)",
            jl_weight,
            T1_JOINT_LIMIT_MARGIN,
        )
    return planner


def get_t1_ik_solver(
    scene: Scene,
    *,
    max_batch_size: int = 64,
    enable_com_aware_ik: bool = True,
    num_seeds: int = 32,
    lbfgs_iters: int = 0,
    use_cuda_graph: bool = False,
    device_cfg: DeviceCfg = None,
) -> InverseKinematics:
    """InverseKinematics solver for T1 with the mobile base locked.

    Always built with both tool frames from ``t1_planar_base.yml``. During
    particle-init IK calls the multi-frame goal (``_build_multi_frame_goal``)
    holds the inactive *hand* where the IK's start conf puts it (home, or a
    CoM-gate retry's perturbed seed), so the inactive arm's joints
    counter-rotate as the body bends (about 1 rad per approach in blocks_t1;
    known-bugs B15). The cspace pin written beside it is a no-op
    (``T1State.pin_for_arm_action``).

    The 3 base DOFs (``base_j_x``, ``base_j_y``, ``base_j_yaw``) are pinned
    at 0 — IK cannot drift the base during particle init. Legs / torso /
    waist / arms remain free.

    When ``enable_com_aware_ik`` is True (default), the seed-IK LM solver
    carries a CoM-over-support-rectangle residual (cuRobo fork) that
    recruits the legs to keep the COM centered while reaching.
    """
    from cutamp.com_polygon_cost import COM_HALF_EXTENTS, COM_INSIDE_MARGIN, COM_INSIDE_WEIGHT

    if device_cfg is None:
        device_cfg = DeviceCfg()
    cfg_dict = _lock_mobile_base(copy.deepcopy(t1_curobo_cfg()))
    cfg = InverseKinematicsCfg.create(
        robot=cfg_dict,
        scene_model=scene,
        num_seeds=num_seeds,
        # The LBFGS polish stage defaults to 100 iterations (lbfgs_ik.yml) —
        # sized for cold global solves. Warm-started refresh solvers pass a
        # smaller count; 0 keeps the cuRobo default schedule.
        override_optimizer_num_iters=(
            {"particle": None, "lbfgs": lbfgs_iters} if lbfgs_iters > 0
            else {"particle": None, "lbfgs": None}
        ),
        self_collision_check=True,
        max_batch_size=max_batch_size,
        device_cfg=device_cfg,
        # CUDA graphs specialize for the call structure captured at first call
        # and can't be re-captured ("CUDA graph reset is not available.").
        # Particle init mixes calls with and without ``current_state``, so the
        # main solver runs without them; the refresh solver always gets the
        # same full-batch warm-seeded call and opts in.
        use_cuda_graph=use_cuda_graph,
        # CoM-aware seed-IK residual (fork): pulls the CoM toward the support
        # rectangle while the LM solver reaches the hand pose. The two weights
        # are the sole on/off gate (the fork derives compute_com from them);
        # geometry params are passed unconditionally.
        seed_com_support_weight=(T1_COM_IK_WEIGHT if enable_com_aware_ik else 0.0),
        seed_com_half_extents=list(COM_HALF_EXTENTS),
        seed_com_inside_margin=COM_INSIDE_MARGIN,
        seed_com_inside_weight=COM_INSIDE_WEIGHT,
        seed_com_center_weight=(T1_COM_IK_CENTER_WEIGHT if enable_com_aware_ik else 0.0),
        seed_com_base_link_name="mobile_base_link",
    )
    ik_solver = InverseKinematics(cfg)
    return ik_solver


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
    """Collision spheres for the named gripper, in the gripper base frame."""
    if arm not in ("left", "right"):
        raise ValueError(f"Invalid arm: {arm}")
    if device_cfg is None:
        device_cfg = DeviceCfg()
    spheres_file = T1_ASSETS_DIR / f"{arm}_gripper_spheres.pt"
    if not spheres_file.exists():
        raise FileNotFoundError(
            f"T1 {arm} gripper spheres not found at {spheres_file}. "
            "Generate via cutamp/scripts/gripper_sphere_editor.py"
        )
    spheres = torch.load(spheres_file, map_location=device_cfg.device, weights_only=True)
    assert spheres.ndim == 2 and spheres.shape[1] == 4, f"Bad shape: {spheres.shape}"
    return spheres[spheres[:, 3] > 0]


# =============================================================================
# URDF expansion for visualization
# =============================================================================

def curobo_to_urdf_joints(
    q_curobo,
    *,
    left_gripper: Tuple[float, ...] = GRIPPER_OPEN,
    right_gripper: Tuple[float, ...] = GRIPPER_OPEN,
) -> Tuple[float, ...]:
    """Map a 21-DOF cuRobo cspace vector to the 28-DOF URDF actuated-joint vector.

    The 3 planar-base DOFs (q_curobo[0:3]) are NOT in the URDF — they're virtual
    cuRobo joints. Visualizers should apply the base SE(2) transform to the
    URDF root externally.

    URDF joint order (28):
        [0-1]   ankle_pitch, knee_pitch
        [2-3]   Torso_Pitch, Waist_Yaw
        [4-5]   AAHead_yaw, Head_pitch (locked)
        [6-12]  Left_Shoulder_Pitch ... Left_Hand_Roll
        [13-16] left_Link{1,11,2,22} (gripper)
        [17-23] Right_Shoulder_Pitch ... Right_Hand_Roll
        [24-27] right_Link{1,11,2,22} (gripper)
    """
    if hasattr(q_curobo, "tolist"):
        q_curobo = q_curobo.tolist()
    if len(q_curobo) != CUROBO_DOF:
        raise ValueError(f"Expected {CUROBO_DOF} DOF, got {len(q_curobo)}")

    leg = tuple(q_curobo[3:5])
    torso = tuple(q_curobo[5:7])    # Torso_Pitch, Waist_Yaw
    left_arm = tuple(q_curobo[7:14])
    right_arm = tuple(q_curobo[14:21])
    urdf = (
        *leg,                   # 0-1 (ankle_pitch, knee_pitch)
        *torso,                 # 2-3 (Torso_Pitch, Waist_Yaw)
        *t1_home_head,          # 4-5 (locked)
        *left_arm,              # 6-12
        *left_gripper,          # 13-16
        *right_arm,             # 17-23
        *right_gripper,         # 24-27
    )
    assert len(urdf) == URDF_DOF, f"Expected {URDF_DOF} URDF joints, got {len(urdf)}"
    return urdf


def base_se3_from_q(q_curobo) -> torch.Tensor:
    """Extract the world-frame SE(3) transform from a 21-DOF cspace vector.

    ``q_curobo[0:3] = (x, y, yaw)`` — z=0, no roll/pitch. Use to anchor the
    URDF visualization at the planned base pose.
    """
    if hasattr(q_curobo, "tolist"):
        q_curobo = q_curobo.tolist()
    x, y, yaw = q_curobo[0], q_curobo[1], q_curobo[2]
    return action_4dof_to_mat4x4(torch.tensor([x, y, 0.0, yaw], dtype=torch.float32))


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
    """Rerun URDF robot for T1 with planar-base awareness.

    Accepts joint vectors of length:
      - 21: cuRobo cspace (base + body). The first 3 DOFs are the planar
        base SE(2) ``(x, y, yaw)``, which the URDF doesn't include — they
        live as cuRobo ``extra_links``. We log them as a ``Transform3D`` on
        the robot's root entity so the rendered robot moves with the planner.
      - 28: URDF actuated joints (planar base ignored; robot stays at origin).
    """

    def __init__(self, name: str, urdf: URDF, q_neutral=None, load_mesh: bool = True):
        self.name = name
        self.urdf = urdf
        self._left_gripper: Tuple[float, ...] = GRIPPER_OPEN
        self._right_gripper: Tuple[float, ...] = GRIPPER_OPEN
        rr.log(self.urdf.base_link, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        log_scene(scene=self.urdf.scene, node=self.urdf.base_link, path=self.name, static=False, add_mesh=load_mesh)
        if q_neutral is not None:
            self.set_joint_positions(q_neutral)

    def set_grippers(self, left=GRIPPER_OPEN, right=GRIPPER_OPEN) -> None:
        self._left_gripper = tuple(left)
        self._right_gripper = tuple(right)

    def _log_base_transform(self, q_curobo) -> None:
        """Apply the SE(2) planar base transform to the robot's root entity."""
        m = base_se3_from_q(q_curobo).numpy()
        rr.log(self.name, rr.Transform3D(translation=m[:3, 3], mat3x3=m[:3, :3]))

    def set_joint_positions(self, joint_positions) -> None:
        if hasattr(joint_positions, "tolist"):
            joint_positions = joint_positions.tolist()
        elif hasattr(joint_positions, "__iter__"):
            joint_positions = list(joint_positions)

        if len(joint_positions) == CUROBO_DOF:
            self._log_base_transform(joint_positions)
            joint_positions = curobo_to_urdf_joints(
                joint_positions, left_gripper=self._left_gripper, right_gripper=self._right_gripper,
            )

        # Append the remaining joint positions, this can happen if we're only set arm joints not gripper
        if len(joint_positions) != self.urdf.num_actuated_joints:
            start_idx = len(joint_positions)
            joint_positions = [*joint_positions, *self.urdf.cfg[start_idx:]]

        self.urdf.update_cfg(joint_positions)
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

        base_columns = None
        if joint_positions.shape[-1] == CUROBO_DOF:
            # Build per-frame SE(2) columns BEFORE collapsing to URDF DOFs.
            base_mats = np.stack([base_se3_from_q(q).numpy() for q in joint_positions])
            base_columns = rr.Transform3D.columns(
                mat3x3=base_mats[:, :3, :3], translation=base_mats[:, :3, 3],
            )
            mapped = []
            for q in joint_positions:
                mapped.append(curobo_to_urdf_joints(
                    q, left_gripper=self._left_gripper, right_gripper=self._right_gripper,
                ))
            joint_positions = torch.tensor(
                mapped, dtype=joint_positions.dtype, device=joint_positions.device,
            )

        # Detach and move to CPU to be safe
        joint_positions = joint_positions.detach().cpu()
        n, d = joint_positions.shape

        # Add remaining joint positions (e.g., for the gripper) if needed
        if d != self.urdf.num_actuated_joints:
            q_remaining = self.urdf.cfg[d:]
            q_remaining = torch.tensor(q_remaining, dtype=joint_positions.dtype)
            # Expand to batch size and concatenate with existing joint positions
            q_remaining = q_remaining.expand((n, len(q_remaining)))
            joint_positions = torch.cat([joint_positions, q_remaining], dim=1)

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
        if base_columns is not None:
            path_to_components[self.name] = base_columns
        return path_to_components


def load_t1_rerun(load_mesh: bool = True) -> T1RerunRobot:
    """Load the T1 URDF for Rerun visualization."""
    urdf_path = T1_ASSETS_DIR / "t1_simplified.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"T1 URDF not found at {urdf_path}")

    def _locate_t1_asset(fname: str) -> str:
        if fname.startswith("package://"):
            return str(T1_ASSETS_DIR / fname.replace("package://", ""))
        full_path = T1_ASSETS_DIR / fname
        return str(full_path) if full_path.exists() else fname

    urdf = URDF.load(str(urdf_path), filename_handler=_locate_t1_asset)
    full_neutral = curobo_to_urdf_joints(t1_home)
    return T1RerunRobot("t1", urdf, q_neutral=full_neutral, load_mesh=load_mesh)
