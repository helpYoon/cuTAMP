"""Grasps (design 7.8). ``grasp_sets`` makes each object shape's grasps with GraspGenX, in this process, when the world
loads. Init draws from them: ``drop_grasps`` removes those the hand can't take, and ``approach_in_cone`` keeps the
approaches from the robot's side. The motion solver's grasp geometry: each pick and place approaches along the grasp's
axis and leaves the same way, or lifts and comes in from above.

``world_from_ee`` is the hand's tool frame (``*_base_link``, ``robots/t1.py``), whose +x is GraspGenX's approach axis
(G's +z). The knobs are search settings, not robot limits (design section 12).
"""

import functools
import hashlib
import importlib.util
import logging
import math
import os
import platform
import random
from typing import Dict, Literal, NamedTuple, Sequence, Tuple, Union

import numpy as np
import torch
import trimesh
from curobo.scene import Obstacle
from curobo.types import DeviceCfg

from cutamp.costs import sphere_to_sphere_overlap
from cutamp.robots.t1 import LEFT_TOOL_FRAME, get_t1_tool_from_ee, load_sweep_volume
from cutamp.utils.common import mat4x4_to_grasp_7dof, transform_spheres

_log = logging.getLogger(__name__)

APPROACH_DISTANCE = 0.05
"""m: how far back along the hand's +x a pick's or a place's straight insert starts, and a back-out ends: upstream
cuTAMP's Franka pre-grasp distance."""

LIFT_HEIGHT = 0.05
"""m: the straight lift after a pick whose approach is not vertical, and the height a non-vertical place comes in
from."""

VERTICAL_CONE = math.pi / 4
"""An approach within this angle of straight down backs out along its own axis, which rises at least as fast as it
moves sideways. Any other approach lifts or comes in from above."""

LINE_TOL = 0.005
"""m: the most a straight segment's palm may stray from its line: the IK's own position tolerance
(``scripts/utils.py``'s ``pos_err``)."""

LINE_BACKTRACK = 0.001
"""m: the most a straight segment's progress along its line may drop."""

LINE_END_TOL = 0.005
"""m: the most a straight segment's last frame may end from the line's end."""

AZIMUTH_CONE = math.pi / 3
"""An approach that isn't near vertical (``VERTICAL_CONE``) must come from the robot's side: its horizontal direction
within this angle of the direction from the stance to the object, at init's picks (``approach_in_cone``) and in its
place yaws."""

GRASP_REDRAW_ROUNDS = 2
"""How many times init draws again for the rows whose IK failed, and solves only those rows: a pick row draws a new
grasp, a place row a new placement. Nothing Adam moves sets a pick's target (grasps are never optimized), so only a
new draw can repair a pick row."""


def near_vertical(approach: torch.Tensor) -> torch.Tensor:
    """``[...]`` bool: whether each approach direction ``approach`` ``[..., 3]`` (a unit vector) is within
    ``VERTICAL_CONE`` of -z."""
    return -approach[..., 2] >= math.cos(VERTICAL_CONE)


def is_vertical(world_from_ee: torch.Tensor) -> bool:
    """Whether the hand's +x (the approach) of ``world_from_ee`` ``[4, 4]`` is within ``VERTICAL_CONE`` of world -z."""
    return bool(near_vertical(world_from_ee[:3, 0]))


def drop_grasps(
    obj_from_G: torch.Tensor, hand_spheres: torch.Tensor, obj_spheres: torch.Tensor, obj_bottom_z: float,
) -> torch.Tensor:
    """``[K]`` bool, True where the hand can take the grasp ``obj_from_G`` ``[K, 4, 4]`` (G in the object frame). With
    the hand's spheres ``hand_spheres`` ``[S, 4]`` (its tool frame, ``*_base_link``) placed at the grasp, it is dropped
    when either holds:

    - a sphere's lowest point (z - r, in the object frame) is below ``obj_bottom_z``, the object's bottom face. Objects
      stand upright on a support at every pick and place, so such a grasp goes into the support;
    - a sphere overlaps one of the object's own spheres ``obj_spheres`` ``[M, 4]`` (object frame). GN's collision row
      would push the hand off such a grasp's target, and ``ik_check`` rejects the conf.
    """
    # In float64: CUDA float32 matmuls run in TF32 here (torch.backends.cuda.matmul.allow_tf32 is on), which moves the
    # spheres by about 0.1 mm and flipped 3 of blocks_t1's 4,100 grasps per hand.
    G_from_base = get_t1_tool_from_ee(DeviceCfg(device=obj_from_G.device))[LEFT_TOOL_FRAME]  # both hands share it
    hand = transform_spheres(hand_spheres.double(), obj_from_G.double() @ G_from_base.double())  # [K, S, 4], obj frame
    below = (hand[..., 2] - hand[..., 3] < obj_bottom_z).any(-1)
    touching = sphere_to_sphere_overlap(obj_spheres.double(), hand) > 0
    return ~(below | touching)


def approach_in_cone(world_from_G: torch.Tensor, stance_xy: torch.Tensor, obj_xy: torch.Tensor) -> torch.Tensor:
    """``[K]`` bool: whether each grasp ``world_from_G`` ``[K, 4, 4]`` approaches the object at ``obj_xy`` from the side
    of the robot standing at ``stance_xy`` (``[2]`` each). Its approach is Z_G. A near-vertical one (``near_vertical``)
    always passes; any other passes when its horizontal direction is within ``AZIMUTH_CONE`` of ``obj_xy -
    stance_xy``. A straight-up approach has no horizontal direction, and fails."""
    approach = world_from_G[..., :3, 2]
    away = torch.nn.functional.normalize(obj_xy - stance_xy, dim=-1)
    cos = (torch.nn.functional.normalize(approach[..., :2], dim=-1) * away).sum(-1)
    return near_vertical(approach) | (cos >= math.cos(AZIMUTH_CONE))


def approach_targets(world_from_ee: torch.Tensor, kind: Literal["pick", "place"]) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(pre, post)``: the hand poses ``[4, 4]`` where a pick's or a place's straight insert to ``world_from_ee``
    starts, and where its leave ends.

    ``back`` is ``world_from_ee`` moved ``APPROACH_DISTANCE`` back along its own +x, and ``up`` is it moved
    ``LIFT_HEIGHT`` up world z, each with its rotation. A pick comes in from ``back`` and leaves to ``back`` when its
    approach is vertical (``is_vertical``), else lifts to ``up``. A place comes in from ``back`` when vertical, else
    from ``up``, and always backs out to ``back``."""
    back = world_from_ee.clone()
    back[:3, 3] -= APPROACH_DISTANCE * world_from_ee[:3, 0]
    up = world_from_ee.clone()
    up[2, 3] += LIFT_HEIGHT
    vertical = is_vertical(world_from_ee)
    if kind == "pick":
        pre, post = back, back if vertical else up
    elif kind == "place":
        pre, post = back if vertical else up, back
    else:
        raise ValueError(f"kind must be 'pick' or 'place', got {kind!r}")
    return pre.clone(), post.clone()


# ---- GraspGenX's grasps, made in-process when the world loads ----------------------------------------------------

GRASP_SEED = 0
"""Seeds each shape's surface sample, and every generator right before GraspGenX runs: every process, and a cold
sampler as well as a warm one, then gives the same grasps."""

_README_FOLDERS = {
    "GRASPGENX_CHECKPOINT_DIR": "~/graspgenx-model/7c834043",
    "GRASPGENX_GRIPPER_CFG_DIR": "~/graspgenx-model/no-gripper-descriptions",
}
"""Each variable's folder in README.md, Installation, the GraspGenX paragraph."""


class GraspSet(NamedTuple):
    """One object shape's GraspGenX grasps, every one kept with no threshold: init weights its draws by score."""

    grasps: torch.Tensor
    """``[K, 7]`` float32 on the device: GraspGenX's gripper frame G in the object frame, ``[x, y, z, qw, qx, qy, qz]``
    (``utils.common.grasp_7dof_to_mat4x4``), with a unit quaternion."""
    scores: torch.Tensor
    """``[K]`` float32 in [0, 1]: GraspGenX's discriminator scores."""


_SETS: Dict[Tuple[str, torch.device], GraspSet] = {}
"""Each shape's grasps by ``(mesh_sha256, device)``, kept for the life of the process: a world on another device gets
tensors of its own."""

_SAMPLER = None
"""GraspGenX's ``GraspGenXSampler`` for the T1 gripper, its model on the GPU (538 MB), kept for the life of the
process. One serves every device: GraspGenX runs on ``.cuda()`` and returns numpy arrays, which ``grasp_sets`` places
on each call's device."""


def mesh_sha256(obj: Obstacle) -> str:
    """The sha256 of the mesh the sphere fitting uses, ``obj.get_trimesh_mesh()`` at the unit pose: its float64
    vertices, then its int64 faces, both C-ordered. Same-shaped objects share it, whatever their poses."""
    mesh = obj.get_trimesh_mesh()
    digest = hashlib.sha256(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _checked_model_dir() -> str:
    """The checked ``GRASPGENX_CHECKPOINT_DIR``. An unset or empty variable is set to its README folder
    (``_README_FOLDERS``), and every value is made absolute in ``os.environ`` (``load_model_cfg`` doesn't expand ``~``),
    so GraspGenX's first-import hook finds both folders and never clones."""
    for var, folder in _README_FOLDERS.items():
        os.environ[var] = os.path.abspath(os.path.expanduser(os.environ.get(var) or folder))
    ckpt, grippers = os.environ["GRASPGENX_CHECKPOINT_DIR"], os.environ["GRASPGENX_GRIPPER_CFG_DIR"]
    has_model = all(os.path.isfile(os.path.join(ckpt, "release", side, "config.yaml")) for side in ("gen", "dis"))
    if not (has_model and os.path.isdir(grippers)):
        raise RuntimeError(
            f"GraspGenX's model folders are missing: GRASPGENX_CHECKPOINT_DIR ({ckpt}) must hold "
            f"release/gen/config.yaml and release/dis/config.yaml, and GRASPGENX_GRIPPER_CFG_DIR ({grippers}) must be "
            "an existing folder. See README.md, Installation, the GraspGenX paragraph."
        )
    return ckpt


def _deterministic(fn):
    """``fn`` with torch's deterministic algorithms on for each call; the caller's mode and warn-only flag come back
    after it."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        mode = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(mode, warn_only=warn_only)

    wrapper.cutamp_deterministic = True
    return wrapper


def _load_sampler(ckpt: str):
    """Import GraspGenX, keeping its logging out of cuTAMP's; make PTv3's segment reduction deterministic; and build
    the sampler on the T1 gripper's sweep volume (``load_sweep_volume``), which loads the model's weights."""
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:  # graspgenx's import sets the root logger to INFO and adds a stdout handler (utils/logging_config.py)
        import graspgenx  # noqa: F401
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.models.ptv3 import ptv3_vanilla
        from graspgenx.serving.types import FLAT_FIELD_ORDER, SweepVolumeParams
        from graspgenx.utils.checkpoint_io import load_model_cfg
    finally:
        root.setLevel(level)
        root.handlers[:] = handlers
    for name in ("graspgenx", "graspgenx.dataset.dataset"):  # dataset.py gives its logger its own INFO level
        logging.getLogger(name).setLevel(logging.WARNING)
    # PTv3's scatter_reduce_ (ptv3_vanilla.py:257-279) is nondeterministic on CUDA: seeded runs differ by up to 1.8 um.
    # Its callers look the name up at each call, so the wrapper reaches all of them.
    if not getattr(ptv3_vanilla.segment_csr_vanilla, "cutamp_deterministic", False):
        ptv3_vanilla.segment_csr_vanilla = _deterministic(ptv3_vanilla.segment_csr_vanilla)
    sv = load_sweep_volume()
    flat = [x for field in FLAT_FIELD_ORDER for x in sv[field]]
    cfg = load_model_cfg(f"{ckpt}/release/gen", f"{ckpt}/release/dis")  # the latest epoch_*.pth: gen 736, dis 1056
    params = SweepVolumeParams.from_flat(flat, fingertip_depth=sv["fingertip_depth"])
    return GraspGenXSampler.from_sweep_volume(cfg, params)


def _z_offsets_cm(mesh: trimesh.Trimesh) -> Tuple[int, ...]:
    """The box grasps' depths into each face: 0, -1, ..., -m cm, with m = min(8, floor(the smallest extent in cm) - 1),
    so the fingertips stop at least 1 cm short of the far side. Upstream's -8 cm would put them below a 5 cm block."""
    m = min(8, math.floor(100 * min(mesh.extents) + 1e-9) - 1)  # 1e-9: 0.29 m is 28.999999999999996 cm in floats
    return tuple(range(0, -m - 1, -1))


def grasp_sets(objs: Sequence[Obstacle], device: Union[torch.device, str]) -> Dict[str, GraspSet]:
    """Each object's GraspGenX grasps, keyed by name, made in this process.

    Same-shaped objects (``mesh_sha256``) share one ``GraspSet``, and each shape is made once per process and device
    (``_SETS``). A new shape samples 3,000 points of its unit-pose mesh, in the object frame, and runs GraspGenX's
    graspmoe planner on them: 500 diffusion grasps and the box grasps on the top and the four sides, all kept.
    GraspGenX always runs on ``.cuda()``; ``device`` only places the returned tensors.

    The grasps repeat bit for bit across calls and processes: each run is seeded with ``GRASP_SEED`` and runs in FP32,
    with PTv3's segment reduction deterministic. The caller's random state (Python's, numpy's, torch's CPU and every
    CUDA generator) and both TF32 flags are restored afterwards.

    Call it only inside ``setup_cutamp``, before the motion planner's warmup thread starts, and in tests: deterministic
    mode and the TF32 pin are process-wide, which is safe only while no other CUDA work runs.

    Every call first checks that GraspGenX is installed, which it is on x86_64 only (``pyproject.toml``), and then its
    model folders (``_checked_model_dir``), before anything imports ``graspgenx`` or touches CUDA. Either check raises
    ``RuntimeError`` naming the README.
    """
    global _SAMPLER
    if importlib.util.find_spec("graspgenx") is None:  # finds the package only: its first-import hook doesn't run
        raise RuntimeError(
            "GraspGenX is not installed, and planning needs it: every world loads its movables' grasps from it "
            f"(cutamp/grasps.py). uv sync installs it on x86_64 only, and this machine is {platform.machine()}. "
            "See README.md, Installation, the GraspGenX paragraph."
        )
    ckpt = _checked_model_dir()
    device = torch.device(device)  # a str and a torch.device of one device share a key
    shape_of = {obj.name: mesh_sha256(obj) for obj in objs}
    new = {shape_of[obj.name]: obj for obj in objs if (shape_of[obj.name], device) not in _SETS}
    if new:
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
        tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            if _SAMPLER is None:
                _SAMPLER = _load_sampler(ckpt)
            from graspgenx.samplers.planner import run_planner_on_object

            for shape, obj in new.items():
                mesh = obj.get_trimesh_mesh()
                points = trimesh.sample.sample_surface(mesh, 3000, seed=GRASP_SEED)[0]  # not centred
                # The model build draws from torch's CPU generator: seeding here makes a cold sampler match a warm one.
                random.seed(GRASP_SEED)
                np.random.seed(GRASP_SEED)
                torch.manual_seed(GRASP_SEED)  # and every CUDA generator
                grasps, scores, tags, _ = run_planner_on_object(
                    points, _SAMPLER, planner="graspmoe", num_grasps=500, grasp_threshold=-1.0, topk_num_grasps=-1,
                    moe_obb_density="dense-topandside", moe_z_offsets_cm=_z_offsets_cm(mesh), moe_num_yaws=36,
                    moe_obb_mode="advanced", moe_skip_obb_rule="auto", moe_obb_position_spacing_cm=1.0,
                    moe_outlier_threshold=0.014, moe_outlier_k=20,
                )
                _SETS[shape, device] = GraspSet(
                    mat4x4_to_grasp_7dof(torch.from_numpy(grasps).to(device)), torch.from_numpy(scores).to(device)
                )
                n_box, names = tags.count("obb"), ", ".join(n for n, s in shape_of.items() if s == shape)
                _log.debug(f"GraspGenX: {len(tags)} grasps for {names}, {len(tags) - n_box} diffusion and {n_box} box")
        finally:
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            torch.cuda.set_rng_state_all(rng[3])
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    return {name: _SETS[shape, device] for name, shape in shape_of.items()}
