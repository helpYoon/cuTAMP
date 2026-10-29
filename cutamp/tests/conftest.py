"""Shared pytest fixtures for cuTAMP tests (cuRobo v0.8 single-MotionPlanner)."""

import gc
import os
import sys

import pytest
import torch

_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
#: Shared GPU-required marker: skips when CUDA_VISIBLE_DEVICES hides every GPU
#: ('' or '-1'), or is unset and /dev/nvidia0 is missing. It reads no CUDA
#: state, so collection initializes no CUDA.
needs_cuda = pytest.mark.skipif(
    _cvd in ("", "-1") or (_cvd is None and not os.path.exists("/dev/nvidia0")),
    reason="Requires a CUDA device.",
)


def make_blocks_t1_world(**tamp_world_kwargs):
    """Build the standard blocks_t1 TAMPWorld used by GPU integration tests."""
    import torch
    from curobo.types import DeviceCfg
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.robots import load_t1_container
    from cutamp.robots.t1 import t1_home
    from cutamp.tamp_world import TAMPWorld
    dc = DeviceCfg()
    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    robot = load_t1_container(dc)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=dc.device)
    return TAMPWorld(env=env, device_cfg=dc, robot=robot, q_init=q_init, **tamp_world_kwargs)


@pytest.fixture
def t1_planar_base_config():
    from curobo._src.util.config_io import load_yaml
    return load_yaml(os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "robots", "assets", "t1_description", "t1_planar_base.yml",
    ))


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
def blocks_t1_plan():
    """``[world, skeleton, particles]``: the blocks_t1 world ``algorithm.py``
    builds for 64 particles, the first task plan, and its seeded 64-particle
    init. Tests never write to it: they run on ``clone_particles(particles)``.
    A module that uses it must not build a second world (two OOM a 24 GiB card).
    numpy is seeded over the world build and the init, then restored: it
    samples the block spheres. The plan follows PYTHONHASHSEED.

    A list, so its teardown can free the world: pytest runs a fixture's
    teardown before it drops the cached value, and the module's last test
    still holds the yielded object until after teardown. The autouse
    ``_free_cuda_between_tests`` runs before module teardown, so it can't.
    A failed test's exception in ``sys.last_*`` holds its locals until the
    next test's call phase, so the teardown drops it first. Clearing the
    traceback's frames is not enough: on CPython 3.10, the ``f_locals``
    snapshot pytest reads to render a failure outlives ``frame.clear()``.
    """
    import numpy as np
    from cutamp.config import TAMPConfiguration
    from cutamp.particle_initialization import ParticleInitializer
    from cutamp.t1_domain import all_t1_operators
    from cutamp.task_planning import task_plan_generator

    rng = np.random.get_state()
    np.random.seed(0)
    try:
        world = make_blocks_t1_world(collision_activation_distance=0.0, ik_max_batch_size=64)
        skeleton = next(task_plan_generator(world.initial_state, world.goal_state, all_t1_operators))
        torch.manual_seed(0)
        particles = ParticleInitializer(world, TAMPConfiguration(num_particles=64))(skeleton)
    finally:
        np.random.set_state(rng)
    plan = [world, skeleton, particles]
    del world, skeleton, particles  # the list holds the only references
    yield plan
    plan.clear()
    sys.last_type = sys.last_value = sys.last_traceback = None
    gc.collect()
    torch.cuda.empty_cache()


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
        "rollout_fn": RolloutFunction(skeleton, world, config),
        "cost_fn": CostFunction(skeleton, world, config),
    }


def make_optimizer(config, **kw):
    """A ``ParticleOptimizer`` on the default multipliers and tolerances, as
    ``run_cutamp.py`` builds it."""
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.optimize_plan import ParticleOptimizer
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    return ParticleOptimizer(
        config, CostReducer(default_constraint_to_mult.copy()), ConstraintChecker(default_constraint_to_tol.copy()),
        **kw,
    )


def run_optimizer(plan_info: dict, config, optimizer=None, visualizer=None, timer=None, **kw) -> tuple:
    """One optimize call on ``plan_info``: ``(opt_metrics, timer)``. Without an
    ``optimizer``, ``kw`` goes to :func:`make_optimizer`; the no-op
    ``Visualizer`` stands in for a missing ``visualizer``."""
    from cutamp.utils.timer import TorchTimer
    from cutamp.utils.visualizer import Visualizer
    timer = timer or TorchTimer()
    _, opt_metrics, _ = (optimizer or make_optimizer(config, **kw))(
        plan_info, timer, visualizer or Visualizer(config, None),
    )
    return opt_metrics, timer


def step_outputs(packed, vis, params) -> dict:
    """Clones of one Adam step's outputs, for :func:`assert_bitwise`: the next
    eager call or graph replay overwrites the FK sites' buffers and a graph's
    static outputs. ``packed`` is the step's stacked scalars, ``vis`` its
    ``(rollout, cost_dict, costs, mask)`` or ``None``, and ``params`` the
    optimized tensors, whose ``.grad`` (or ``None``) is kept in order."""
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


def assert_bitwise(got, want, where: str) -> None:
    """``got`` equals ``want`` bitwise: the same dtype, shape and device, then
    ``torch.equal``, on every tensor, through lists, tuples and dicts. Other
    leaves, ``None`` among them, must have the same type and compare equal.
    The ``AssertionError`` names ``where``, the first differing path and its
    max abs diff, ``n/a`` when the dtype, shape, device or structure differs:
    ``torch.equal`` alone promotes, so it passes f32 against f64 or bool."""
    def fail(path, detail):
        raise AssertionError(f"{where}: {path or 'value'} differs, {detail}")

    def kind(x):
        if isinstance(x, torch.Tensor):
            return f"{x.dtype} {tuple(x.shape)} on {x.device}"
        if isinstance(x, dict):
            return f"dict with keys {list(x)}"
        if isinstance(x, (list, tuple)):
            return f"{type(x).__name__} of {len(x)}"
        return type(x).__name__

    nodes = (torch.Tensor, dict, list, tuple)

    def walk(g, w, path):
        if isinstance(w, torch.Tensor) and isinstance(g, torch.Tensor):
            if g.dtype != w.dtype or g.shape != w.shape or g.device != w.device:
                fail(path, f"max abs diff n/a: {kind(g)} vs {kind(w)}")
            if not torch.equal(g, w):
                fail(path, f"max abs diff {(g.double() - w.double()).abs()[g != w].max().item()}")
        elif isinstance(w, dict) and isinstance(g, dict) and g.keys() == w.keys():
            for k in w:
                walk(g[k], w[k], f"{path}[{k!r}]" if path else str(k))
        elif isinstance(w, (list, tuple)) and type(g) is type(w) and len(g) == len(w):
            for i, (gi, wi) in enumerate(zip(g, w)):
                walk(gi, wi, f"{path}[{i}]")
        elif isinstance(g, nodes) or isinstance(w, nodes) or not (type(g) is type(w) and g == w):
            fail(path, f"max abs diff n/a: got {kind(g)}, want {kind(w)}")

    walk(got, want, "")


def cupti_sync_events() -> dict:
    """``torch.profiler.profile`` kwargs that turn on CUPTI's CUDA sync events,
    which also catch driver-API waits (cuRobo launches FK through cuda.core)
    that the runtime-API names and sync debug mode miss. The config is a
    private API, hence the guard: ``{}`` where this torch lacks it."""
    try:
        return {"experimental_config": torch._C._profiler._ExperimentalConfig(enable_cuda_sync_events=True)}
    except (AttributeError, TypeError):
        return {}
