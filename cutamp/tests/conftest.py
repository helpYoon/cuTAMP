"""Shared pytest fixtures for cuTAMP tests (cuRobo v0.8 single-MotionPlanner)."""

import dataclasses
import gc
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


def make_blocks_t1_world(**tamp_world_kwargs):
    """Build the standard blocks_t1 TAMPWorld used by GPU integration tests, on ``REAL_LEGS``, starting at its home."""
    from curobo.types import DeviceCfg
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.robots import load_t1_container
    from cutamp.tamp_world import TAMPWorld
    dc = DeviceCfg()
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    robot = load_t1_container(REAL_LEGS, dc)
    q_init = torch.as_tensor(REAL_LEGS.home, dtype=torch.float32, device=dc.device)
    return TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=q_init, **tamp_world_kwargs)


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


@pytest.fixture(scope="module")
def real_plan(real_world):
    """``[skeleton, particles]``: blocks_t1's first task plan (which follows PYTHONHASHSEED) and its seeded 64-particle
    init on ``real_world``. Read-only: tests run on ``clone_particles(particles)``. A list, so teardown can free the
    particles while pytest still holds the yielded object; ``real_world``'s slot then empties the cache."""
    from cutamp.particle_initialization import ParticleInitializer
    from cutamp.t1_domain import all_t1_operators
    from cutamp.task_planning import task_plan_generator

    skeleton = next(task_plan_generator(real_world.initial_state, real_world.goal_state, all_t1_operators))
    torch.manual_seed(0)
    particles = ParticleInitializer(real_world, TAMPConfiguration(num_particles=64), default_checker())(skeleton)
    plan = [skeleton, particles]
    del skeleton, particles  # the list holds the only references
    yield plan
    plan.clear()


def clone_particles(particles: dict) -> dict:
    """Fresh leaf copies of ``particles``, each keeping its ``requires_grad``."""
    return {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in particles.items()}


def make_plan_info(world, skeleton, particles: dict, config) -> dict:
    """The four ``plan_info`` keys ``ParticleOptimizer.__call__`` reads, with a
    fresh rollout and cost function."""
    from cutamp.cost_function import CostFunction
    from cutamp.rollout import RolloutFunction
    return {
        "plan_skeleton": skeleton,
        "particles": particles,
        "rollout_fn": RolloutFunction(skeleton, world),
        "cost_fn": CostFunction(skeleton, world, config),
    }


def default_checker():
    """The ``ConstraintChecker`` on the default tolerances, as ``run_cutamp.py`` builds it."""
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.scripts.utils import default_constraint_to_tol
    return ConstraintChecker(default_constraint_to_tol.copy())


def make_optimizer(config, **kw):
    """A ``ParticleOptimizer`` on the default multipliers and tolerances, as
    ``run_cutamp.py`` builds it."""
    from cutamp.cost_reduction import CostReducer
    from cutamp.optimize_plan import ParticleOptimizer
    from cutamp.scripts.utils import default_constraint_to_mult
    return ParticleOptimizer(config, CostReducer(default_constraint_to_mult.copy()), default_checker(), **kw)


def run_optimizer(plan_info: dict, config, optimizer=None, visualizer=None, timer=None, **kw) -> tuple:
    """One optimize call: ``(opt_metrics, timer)``; without ``optimizer``, ``kw`` goes to :func:`make_optimizer`."""
    from cutamp.utils.timer import TorchTimer
    from cutamp.utils.visualizer import Visualizer
    timer = timer or TorchTimer()
    _, opt_metrics, _ = (optimizer or make_optimizer(config, **kw))(
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


def bitwise_diffs(a, b, raw: bool = True, path: str = "") -> list:
    """Every difference between ``a`` and ``b`` through dicts, lists and tuples, as ``(path, detail)`` pairs.
    ``raw`` compares tensor bits and float bytes (NaN equals the same NaN, -0.0 differs); else ``torch.equal``."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        # First: != promotes dtypes and broadcasts shapes, so f32 could pass against f64 and [1] against [N].
        if (a.dtype, a.shape, a.device) != (b.dtype, b.shape, b.device):
            return [(path, f"{a.dtype} {tuple(a.shape)} {a.device} vs {b.dtype} {tuple(b.shape)} {b.device}")]
        if raw:
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
        return [d for k in a for d in bitwise_diffs(a[k], b[k], raw, f"{path}[{k!r}]")]
    if isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in bitwise_diffs(x, y, raw, f"{path}[{i}]")]
    if raw and isinstance(a, float) and isinstance(b, float):
        return [] if struct.pack("<d", a) == struct.pack("<d", b) else [(path, f"{a!r} vs {b!r}")]
    nodes = (torch.Tensor, dict, list, tuple)
    if type(a) is type(b) and not isinstance(a, nodes) and a == b:
        return []
    return [(path, f"{a!r:.80} vs {b!r:.80}")]


def assert_bitwise(got, want, where: str, raw: bool = False) -> None:
    """``got`` has no :func:`bitwise_diffs` from ``want``; the message names ``where`` and the first differences."""
    diffs = bitwise_diffs(got, want, raw)
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
    """Check every G_step replay bitwise against its eager twin (:func:`twin_body`), run first. Returns one
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
        assert_bitwise(step_outputs(*self.outputs, self.body.params), want, f"G_step replay {k}")
        mine.append(k)

    monkeypatch.setattr(CapturedGraph, "replay", spy)
    return replays


def install_refresh_spy(monkeypatch) -> list:
    """Check every G_refresh replay bitwise against an eager ``_refresh_balanced`` on clones, run first (it
    shares the gate site and ``kinematics_jac``): covered confs and slot, and every other particle
    unchanged. Returns the replays' steps."""
    from cutamp.optimize_plan import _ReikDriftGate, _refresh_balanced, _StaticRefresh

    orig, eager = _StaticRefresh.__call__, []

    def spy(self, step):
        clones, before = clone_particles(self.particles), clone_particles(self.particles)
        g2 = _ReikDriftGate(self.gate.pos_tol, self.gate.rot_tol)
        _refresh_balanced(clones, self.ik_deps, self.world, g2, step)
        orig(self, step)
        where = f"G_refresh replay at step {step}"
        assert_bitwise({"q": {n: self.particles[n] for n in self.q_names}, "slot": self.slot},
                       {"q": {n: clones[n] for n in self.q_names}, "slot": g2._queued[-1].packed}, where)
        others = [n for n in self.particles if n not in self.q_names]
        assert_bitwise({n: self.particles[n] for n in others}, {n: before[n] for n in others},
                       f"{where}: a particle outside the covered confs")
        eager.append(step)

    monkeypatch.setattr(_StaticRefresh, "__call__", spy)
    return eager


def cupti_sync_events() -> dict:
    """``profile`` kwargs enabling CUPTI sync events, which also catch driver-API waits (cuRobo launches FK
    through cuda.core); ``{}`` where this torch lacks the private config."""
    try:
        return {"experimental_config": torch._C._profiler._ExperimentalConfig(enable_cuda_sync_events=True)}
    except (AttributeError, TypeError):
        return {}
