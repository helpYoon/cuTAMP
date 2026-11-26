"""Shared pytest fixtures for cuTAMP tests (cuRobo single-MotionPlanner)."""

import dataclasses
import functools
import gc
import importlib.util
import math
import os
import struct
import sys
import weakref
from contextlib import contextmanager

import pytest
import torch

from cutamp.config import TAMPConfiguration
from cutamp.robots.t1 import REAL_LEGS

_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
#: Skips unless a GPU is visible; reads no CUDA state, so collection initializes none.
needs_cuda = pytest.mark.skipif(
    _cvd in ("", "-1") or (_cvd is None and not os.path.exists("/dev/nvidia0")),
    reason="Requires a CUDA device.",
)
#: Whether GraspGenX is installed: it installs on x86_64 only (pyproject.toml), and every picked object's grasps come
#: from it (``grasps.grasp_sets``). find_spec only finds the package, so its first-import hook never runs here.
HAS_GRASPGENX = importlib.util.find_spec("graspgenx") is not None
_NO_GRASPGENX = "Requires GraspGenX, which installs on x86_64 only: every picked object's grasps come from it."
#: Skips unless GraspGenX is installed. ``make_blocks_t1_world`` skips by itself: a mark can't skip a fixture.
needs_graspgenx = pytest.mark.skipif(not HAS_GRASPGENX, reason=_NO_GRASPGENX)


def link_kinematics(world):
    """A Kinematics of ``world``'s model whose tool frames are its support frame, then every massive link, for FK of
    links the planner does not track (cuRobo's FK returns tool frames only). cuRobo orders the joints by walking the
    tool frames; in its own link order they keep the model's. Built on a copy of the robot config, whose tool frames
    every later solver reads."""
    import copy
    from curobo.kinematics import Kinematics, KinematicsCfg
    from cutamp.robots.t1 import _check_joint_order, t1_curobo_cfg
    kp, support = world.kinematics.kinematics_config, world.model.support.link
    tools = [support] + [n for n, i in kp.link_name_to_idx_map.items() if kp.link_masses_com[i, 3] > 0 and n != support]
    cached = t1_curobo_cfg(world.model)["robot_cfg"]["kinematics"]
    before = list(cached["tool_frames"])
    cfg = copy.deepcopy(t1_curobo_cfg(world.model))  # from_robot_yaml_file writes tool_frames into the dict it is given
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(cfg, tool_frames=tools, device_cfg=world.kinematics.device_cfg),
                     compute_spheres=False)
    _check_joint_order(world.model, kin.joint_names)
    assert cached["tool_frames"] == before  # rewritten, later IK solvers and planners in the process find no solution
    return kin


#: blocks_t1's task plan: block3 in the left hand, block2 in the right, both picked before either is placed (the
#: plan the tests' hash-seed notes were measured on, PYTHONHASHSEED 0).
BLOCKS_T1_ACTIONS = (("LeftPick", "block3"), ("RightPick", "block2"), ("LeftPlace", "block3"), ("RightPlace", "block2"))


def skeleton_through(initial, goal, *actions):
    """The T1 task plan from ``initial`` to ``goal`` whose picks and places are ``actions`` in order, each an
    ``(operator name, object)`` pair, with the motions ``ManipulationPrefix`` grounds before, between and after them."""
    from cutamp.t1_domain import all_t1_operators
    from cutamp.task_planning.plan_search import ManipulationPrefix

    prefix = ManipulationPrefix(initial, goal, all_t1_operators)
    for name, obj in actions:
        (prefix,) = [p for p in prefix.choices()
                     if p.action is not None and (p.action.operator.name, p.action.arg("obj")) == (name, obj)]
    if not prefix.complete:
        (prefix,) = prefix.choices()  # the closing motions, when the goal also asks for free hands
    assert prefix.complete
    return prefix.skeleton()


def make_blocks_t1_world():
    """Build the standard blocks_t1 TAMPWorld used by GPU integration tests, on ``REAL_LEGS``, starting at its home,
    with the GraspGenX grasps the planner would request at the default particle count (the first build in a process
    loads the model, which then stays). Without GraspGenX it skips the calling test, or every test of the calling
    fixture."""
    if not HAS_GRASPGENX:
        pytest.skip(_NO_GRASPGENX)
    from curobo.types import DeviceCfg
    from cutamp.config import TAMPConfiguration
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.grasps import grasp_sets
    from cutamp.robots import load_t1_container
    from cutamp.tamp_world import TAMPWorld
    dc = DeviceCfg()
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    robot = load_t1_container(REAL_LEGS, dc)
    q_init = torch.as_tensor(REAL_LEGS.home, dtype=torch.float32, device=dc.device)
    grasps = grasp_sets(env.movables, dc.device, max_grasps=TAMPConfiguration().num_particles)
    return TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=q_init, grasps=grasps)


@pytest.fixture(autouse=True)
def _free_cuda_between_tests():
    """Release GPU memory after heavy tests. Integration tests build full
    TAMPWorld stacks (several GiB each); without reclaiming between tests,
    two heavy tests in one process OOM a 24 GiB card. Gated on reserved
    memory so the dozens of CPU/light tests skip the gc.collect() tax. A
    failed test's exception, which pytest keeps in ``sys.last_*``, is
    dropped first: its traceback holds the test's locals."""
    yield
    if torch.cuda.is_available() and torch.cuda.memory_reserved() > (1 << 30):
        sys.last_type = sys.last_value = sys.last_traceback = None
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def _real_world_slot():
    """Holds ``real_world``'s world, so teardown can free it: ``real_world`` is torn down first, and with it pytest's
    own reference to the world."""
    slot = [make_blocks_t1_world()]
    yield slot
    slot.clear()
    sys.last_type = sys.last_value = sys.last_traceback = None  # a failed test's traceback holds its locals
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def real_world(_real_world_slot):
    """The blocks_t1 world on the real-leg model. A module using it must not build a second world (two OOM a 24 GiB
    card)."""
    return _real_world_slot[0]


@functools.cache
def landing_predictor():
    """The trained landing predictor the planner loads (``LANDING_BUNDLE``), loaded once per process."""
    from cutamp.landing_accuracy import LANDING_BUNDLE
    from cutamp.landing_predictor import LandingPredictor
    return LandingPredictor(LANDING_BUNDLE)


def zero_step_plan(world, num_particles: int = 64, actions=BLOCKS_T1_ACTIONS, seed: int = 0, **config):
    """The planner's plan without steps on ``world``, as ``refine_stepped_plan`` builds it: a ``SteppedPlan`` on the
    task plan through ``actions`` (:func:`skeleton_through`), standing where the robot stands, its starting particles
    drawn after ``torch.manual_seed(seed)``, on a ``TAMPConfiguration`` of ``num_particles`` and ``config``."""
    from cutamp.cost_reduction import CostReducer
    from cutamp.particle_initialization import _pick_pools
    from cutamp.scripts.utils import default_constraint_to_mult
    from cutamp.step_loop import SteppedPlan
    from cutamp.step_planner import StepInputs
    from cutamp.task_planning.insert_steps import with_steps
    from cutamp.task_planning.landing_options import ChosenLandings
    from cutamp.task_planning.plan_search import RankedPlan

    initial, goal = world.initial_state, world.goal_state
    skeleton = with_steps(skeleton_through(initial, goal, *actions), {}, initial, goal)
    proposal = RankedPlan(skeleton, ("none",) * len(actions), ChosenLandings((), (0, 0)), 0.0)
    cfg = TAMPConfiguration(num_particles=num_particles, **config)
    torch.manual_seed(seed)
    return SteppedPlan(proposal, world, cfg, landing_predictor(), StepInputs(), _pick_pools(world, skeleton),
                       default_checker(), reducer=CostReducer(default_constraint_to_mult.copy()))


def plan_info(plan) -> dict:
    """The ``plan_info`` ``ParticleOptimizer.__call__`` reads for ``plan`` (a ``SteppedPlan``), as
    ``refine_stepped_plan`` hands it over."""
    return {"plan_skeleton": plan.arm_skeleton, "particles": plan.particles, "rollout_fn": plan.rollout,
            "cost_fn": plan, "steps": plan}


@pytest.fixture(scope="module")
def real_plan(real_world):
    """``[skeleton, particles]``: blocks_t1's task plan (``BLOCKS_T1_ACTIONS``) without steps and the starting
    particles of its 64-particle plan on ``real_world`` (:func:`zero_step_plan`), without the grasp indices. Read-only:
    tests run on ``clone_particles(particles)``. A list, so teardown can free the particles while pytest still holds
    the yielded object; ``real_world``'s slot then empties the cache."""
    plan = zero_step_plan(real_world)
    held = [plan.arm_skeleton, {k: v for k, v in plan.particles.items() if not k.endswith("_idx")}]
    del plan  # the list holds the only references
    yield held
    held.clear()


def only_schedule(monkeypatch, orders, arm):
    """Pass on only the plan search's plans whose picks and places all use ``arm`` with these foot ``orders``."""
    from cutamp.task_planning import plan_search as search_module

    skeletons = search_module.plan_search

    def matching(*args, **kwargs):
        for proposal in skeletons(*args, **kwargs):
            arms = {op.operator.metadata.arm for op in proposal.skeleton if op.operator.metadata.is_actionable}
            if proposal.orders == tuple(orders) and arms == {arm}:
                yield proposal

    monkeypatch.setattr(search_module, "plan_search", matching)


def standing_skeleton(monkeypatch, *actions):
    """In place of the plan search's plans, propose one plan that takes no step: the world's task plan whose picks and
    places are ``actions`` (:func:`skeleton_through`), with no chosen landings. Filtering the plan search for such a
    plan walks every stepped schedule its reach evidence ranks ahead first: on blocks_t1's floor it was still in
    ``LandingOptions.edges`` after 11 minutes (2026-11-21)."""
    from cutamp.task_planning.landing_options import ChosenLandings
    from cutamp.task_planning import plan_search as search_module
    from cutamp.task_planning.insert_steps import with_steps

    def propose(options, world, operators, incoming, **kwargs):
        initial, goal = world.initial_state, world.goal_state
        skeleton = with_steps(skeleton_through(initial, goal, *actions), {}, initial, goal)
        yield search_module.RankedPlan(skeleton, ("none",) * len(actions), ChosenLandings((), (0, 0)), 0.0)

    monkeypatch.setattr(search_module, "plan_search", propose)


def first_schedules(monkeypatch, count):
    """Pass on only the first ``count`` plans of the plan search (after any earlier :func:`only_schedule`)."""
    from itertools import islice

    from cutamp.task_planning import plan_search as search_module

    skeletons = search_module.plan_search
    monkeypatch.setattr(search_module, "plan_search", lambda *a, **kw: islice(skeletons(*a, **kw), count))


def clone_particles(particles: dict) -> dict:
    """Fresh leaf copies of ``particles``, each keeping its ``requires_grad``."""
    return {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in particles.items()}


def ik_check(world, q, world_from_ee, arms, movables, tol):
    """``(ok, safe, hand_err)``, each ``[N]``, from one FK of ``q`` ``[N, dof]``, with rows in ``arms``' dep-major
    blocks (``gn_project_balanced``'s layout). ``safe``: the conf stands balanced (the CoM inside the support
    rectangle, both soles on the stance within ``tol``), clear of the scene, of ``movables`` ``[N, M, 4]`` and of
    itself. ``ok``: safe, with the hand on ``world_from_ee`` within ``tol``. The GN tests' IK oracle."""
    from curobo.types import Pose

    from cutamp.com_polygon_cost import _fk_state, com_polygon_mask_from_state
    from cutamp.costs import curobo_pose_error
    from cutamp.gn_refresh import _row_layout, _row_tool_pose, collision_penetration, soles_on_stance

    arms = (arms,) if isinstance(arms, str) else tuple(arms)
    _, is_left = _row_layout(world.model, arms, q.shape[0], q.shape[1], q.device, q.dtype)
    state = _fk_state(world, q)
    pos, quat = _row_tool_pose(state, world.tool_frame_for_arm("left"), world.tool_frame_for_arm("right"), is_left)
    pos_err, rot_err = curobo_pose_error(
        Pose(position=pos.contiguous(), quaternion=quat.contiguous()).get_matrix(), world_from_ee,
    )
    safe = (com_polygon_mask_from_state(state, world.model.support)
            & soles_on_stance(world, state, tol.sole_pos, tol.sole_rot)
            & (collision_penetration(world, state.robot_spheres, movables) <= 0))
    return safe & (pos_err <= tol.pos) & (rot_err <= tol.rot), safe, pos_err + rot_err


def fixed_stances(world, particles: dict) -> dict:
    """``CostFunction``'s ``achieved_stances`` for feet that never move: the world's stance, both soles' (x, y, yaw)
    ``[N, 2, 3]``, under every key of ``particles``."""
    import torch

    soles = world.stance_soles
    yaw = torch.atan2(soles[..., 1, 0], soles[..., 0, 0])
    xyyaw = torch.cat((soles[..., :2, 3], yaw.unsqueeze(-1)), -1).reshape(2, 3)
    return {k: xyyaw.expand(v.shape[0], 2, 3) for k, v in particles.items()}


def default_checker():
    """The ``ConstraintChecker`` on the default tolerances, as ``run_cutamp.py`` builds it."""
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.scripts.utils import default_constraint_to_tol
    return ConstraintChecker(default_constraint_to_tol.copy())


def make_optimizer(config):
    """A ``ParticleOptimizer`` on the default multipliers and tolerances, as
    ``run_cutamp.py`` builds it."""
    from cutamp.cost_reduction import CostReducer
    from cutamp.optimize_plan import ParticleOptimizer
    from cutamp.scripts.utils import default_constraint_to_mult
    return ParticleOptimizer(config, CostReducer(default_constraint_to_mult.copy()), default_checker())


def run_optimizer(plan_info: dict, config, optimizer=None, visualizer=None, timer=None) -> tuple:
    """``(opt_metrics, timer)`` of one optimize call."""
    from cutamp.utils.timer import TorchTimer
    from cutamp.utils.visualizer import Visualizer
    timer = timer or TorchTimer()
    _, opt_metrics, _ = (optimizer or make_optimizer(config))(
        plan_info, timer, visualizer or Visualizer(config, None),
    )
    return opt_metrics, timer


class SpyVis:
    """Duck-typed visualizer, so the Adam loop runs its vis block: records each ``opt_*`` time step and each
    logged loss. Not a ``Visualizer`` subclass: visualizer.py imports rerun and cuRobo at module level, and
    collection must not."""

    def __init__(self, config):
        self.config = config
        self.opt_steps, self.losses = [], []

    def set_time_sequence(self, timeline: str, val: int) -> None:
        if timeline.startswith("opt_"):
            self.opt_steps.append(val)

    def log_scalar(self, name: str, value: float) -> None:
        if name == "loss":
            self.losses.append(value)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


def twin_body(body):
    """``body`` on fresh clones of its particles, sharing its FK sites. A site's backward overwrites its
    ``grad_out``, which a ``.grad`` may hold: run the twin before ``body``'s own step, and clone its outputs
    (:func:`step_outputs`) before the next FK call."""
    clones = clone_particles(body.particles)
    clone_of = {id(body.particles[name]): clone for name, clone in clones.items()}
    return dataclasses.replace(
        body, particles=clones, params=[clone_of[id(p)] for p in body.params],
        grad_masks={id(clone_of[i]): m for i, m in body.grad_masks.items()},
    )


def step_outputs(packed, vis, params) -> dict:
    """Clones of one step's ``packed``, vis outputs and params' ``.grad``, which the next FK call or graph
    replay overwrites."""
    def clone(t):
        return None if t is None else t.detach().clone()

    out = {"packed": clone(packed), "grads": [clone(p.grad) for p in params]}
    if vis is not None:
        _, cost_dict, costs, mask = vis
        out["costs"], out["mask"] = clone(costs), clone(mask)
        out["values"] = {
            (cost, name): clone(v) for cost, entry in cost_dict.items() for name, v in entry["values"].items()
        }
    return out


_INT_VIEW = {torch.float32: torch.int32, torch.float64: torch.int64, torch.float16: torch.int16,
             torch.bfloat16: torch.int16, torch.bool: torch.uint8}


#: torch.testing.assert_close's float32 tolerances: what ``close`` comparisons allow. The plan's Adam step is not bit
#: for bit repeatable even eagerly: cuRobo's kinematics backward sums with shared-memory atomicAdd
#: (kinematics_backward_kernel.cuh:180-215), one posture gradient element off by 7.5e-9 on 2026-11-25. The graph and
#: lockstep tests compare at these (the user's ruling, 2026-11-25).
F32_RTOL, F32_ATOL = 1.3e-6, 1e-5


def bitwise_diffs(a, b, raw: bool = True, path: str = "", close: bool = False) -> list:
    """Every difference between ``a`` and ``b`` through dicts, lists and tuples, as ``(path, detail)`` pairs.
    ``raw`` compares tensor bits and float bytes (NaN equals the same NaN, -0.0 differs); else ``torch.equal``.
    ``close`` compares floating values at ``F32_RTOL`` and ``F32_ATOL`` instead (NaN equals NaN); the rest stay exact."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        # First: != promotes dtypes and broadcasts shapes, so f32 could pass against f64 and [1] against [N].
        if (a.dtype, a.shape, a.device) != (b.dtype, b.shape, b.device):
            return [(path, f"{a.dtype} {tuple(a.shape)} {a.device} vs {b.dtype} {tuple(b.shape)} {b.device}")]
        if close and a.is_floating_point():
            ne = ~torch.isclose(a.detach(), b.detach(), rtol=F32_RTOL, atol=F32_ATOL, equal_nan=True)
        elif raw:
            dt = _INT_VIEW.get(a.dtype, a.dtype)
            ne = a.detach().contiguous().view(dt) != b.detach().contiguous().view(dt)
        else:
            ne = a.detach() != b.detach()
        if not ne.any():
            return []
        d = (a.double() - b.double()).abs()[ne].nan_to_num(float("inf")).max().item()
        return [(path, f"{int(ne.sum())} of {a.numel()} elements differ, max abs diff {d}")]
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [(path, f"keys {list(a)} vs {list(b)}")]
        return [d for k in a for d in bitwise_diffs(a[k], b[k], raw, f"{path}[{k!r}]", close)]
    if isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in bitwise_diffs(x, y, raw, f"{path}[{i}]", close)]
    if close and isinstance(a, float) and isinstance(b, float):
        same = math.isclose(a, b, rel_tol=F32_RTOL, abs_tol=F32_ATOL) or (math.isnan(a) and math.isnan(b))
        return [] if same else [(path, f"{a!r} vs {b!r}")]
    if raw and isinstance(a, float) and isinstance(b, float):
        return [] if struct.pack("<d", a) == struct.pack("<d", b) else [(path, f"{a!r} vs {b!r}")]
    nodes = (torch.Tensor, dict, list, tuple)
    if type(a) is type(b) and not isinstance(a, nodes) and a == b:
        return []
    return [(path, f"{a!r:.80} vs {b!r:.80}")]


def assert_bitwise(got, want, where: str, raw: bool = False, close: bool = False) -> None:
    """``got`` has no :func:`bitwise_diffs` from ``want``; the message names ``where`` and the first differences."""
    diffs = bitwise_diffs(got, want, raw, close=close)
    assert not diffs, f"{where}: {len(diffs)} differences, the first {diffs[:5]}"


@contextmanager
def sync_debug_error():
    """Raise on every synchronizing CUDA call in the block; it misses driver-API waits (:func:`cupti_sync_events`)."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(0)


def install_step_replay_spy(monkeypatch) -> list:
    """Check every G_step replay against its eager twin, at float32 tolerance (``F32_RTOL``, ``F32_ATOL``) (:func:`twin_body`), run first. Returns one
    list of compared replay indices per G_step graph, in capture order. Holds graphs weakly, so a release
    still frees them."""
    from cutamp.cuda_graphs import CapturedGraph
    from cutamp.optimize_plan import _StepBody

    orig, replays, of_graph = CapturedGraph.replay, [], weakref.WeakKeyDictionary()

    def spy(self):
        if self.name != "G_step":
            return orig(self)
        if self not in of_graph:  # the graph's first replay opens its list
            of_graph[self] = []
            replays.append(of_graph[self])
        mine = of_graph[self]
        k = len(mine)
        twin = twin_body(self.body)
        want = step_outputs(*_StepBody.__call__(twin), twin.params)
        orig(self)
        assert_bitwise(step_outputs(*self.outputs, self.body.params), want, f"G_step replay {k}", close=True)
        mine.append(k)

    monkeypatch.setattr(CapturedGraph, "replay", spy)
    return replays


def cupti_sync_events() -> dict:
    """``profile`` kwargs enabling CUPTI sync events, which also catch driver-API waits (cuRobo launches FK
    through cuda.core); ``{}`` where this torch lacks the private config."""
    try:
        return {"experimental_config": torch._C._profiler._ExperimentalConfig(enable_cuda_sync_events=True)}
    except (AttributeError, TypeError):
        return {}
