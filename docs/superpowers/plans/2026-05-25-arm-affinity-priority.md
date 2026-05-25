# Arm-affinity priority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bias the BFS plan-skeleton search so the arm closer to each block is preferred for picking, with graceful cross-body fallback when same-side is infeasible.

**Architecture:** Add an optional `ground_op_priority_fn: Callable[[GroundOperator], float]` parameter threaded through `task_plan_generator` → `breadth_first_search` → `get_valid_ground_operators`. In the grounding function, sort the returned ground ops ascending by the priority before yielding. A cuTAMP-layer factory `make_arm_affinity_priority_fn(world)` returns a priority that scores pick operators by 3D Euclidean distance between the arm's home end-effector position and the block's pose; non-pick operators get 0.

**Tech Stack:** Python 3.10+, PyTorch, cuRobo v0.8, pytest. Conda env at `/home/yoonwoo/miniconda3/envs/tamp/bin/python`.

**Spec:** `docs/superpowers/specs/2026-05-24-arm-affinity-priority-design.md`

**Smoke test command** (used across tasks):
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan
```
Pass criteria: ≥1 satisfying solution, motion plans succeed.

**Notes for the implementer:**
- The current branch (`curobo_v2`) has ~100 files of uncommitted prior port work in the working tree. **Do NOT use `git add -A` / `git add .`.** Only stage the EXACT files listed in each task.
- Each commit may carry unrelated prior modifications on the same files (user-accepted; see `docs/superpowers/specs/2026-05-23-code-review-fixes-design.md` for context).
- World tool_frames (`t1_planar_base.yml`) include `left_base_link` and `right_base_link` (palm frames) but NOT `*_hand_link`. The priority function uses palm-frame FK because (a) it's already in tool_frames so no extra config needed, and (b) the exact frame doesn't affect the RANKING between left and right arms — only direction matters for the priority semantics.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `cutamp/tamp_world.py` | Modify | Compute `arm_home_ee_world` dict in `__init__` |
| `cutamp/task_planning/__init__.py` | Modify | Thread `ground_op_priority_fn` through `task_plan_generator` |
| `cutamp/task_planning/search.py` | Modify | Thread `ground_op_priority_fn` through `breadth_first_search` and use it in `get_valid_ground_operators` |
| `cutamp/algorithm.py` | Modify | Add `make_arm_affinity_priority_fn` factory; wire into the `task_plan_generator` call at line 376 |
| `cutamp/tests/test_arm_affinity.py` | Create | Unit tests for priority function + ordering |

Total: ~70 LOC + tests.

---

## Task 1: Compute `arm_home_ee_world` on TAMPWorld

**Files:**
- Modify: `cutamp/tamp_world.py` — append computation at end of `__init__` (after line 110, after `self._obj_to_aabb = {}`)
- Test: `cutamp/tests/test_arm_affinity.py` (NEW)

This task computes the world-frame XYZ position of each arm's palm frame (`*_base_link`) at the home joint configuration. Stored as a dict for later use by the priority function. We use palm frames because they're already in the planner kinematics' `tool_frames` (per `t1_planar_base.yml:33-35`) — no separate kinematics build needed.

### Step 1.1: Write the failing test

- [ ] Create `cutamp/tests/test_arm_affinity.py`:

```python
"""Tests for arm-affinity priority in plan-skeleton search."""
import os
import pytest


needs_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="Requires a CUDA device.",
)


def _make_world():
    """Build a real TAMPWorld for blocks_t1. Used by tests that need real
    kinematics + scene state."""
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from curobo.types import DeviceCfg
    from cutamp.robots.t1 import t1_home
    import torch

    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    device_cfg = DeviceCfg()
    robot = load_robot_container("t1", device_cfg)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=device_cfg.device)
    return TAMPWorld(env=env, device_cfg=device_cfg, robot=robot, q_init=q_init)


@needs_cuda
def test_arm_home_ee_world_populated_after_init():
    """TAMPWorld.__init__ computes arm_home_ee_world for both arms.

    Each value is a 3-vector; left should be on +Y side, right on -Y side
    (T1 stands facing +X so left arm is to its left, which is +Y in world)."""
    import torch
    world = _make_world()
    assert hasattr(world, "arm_home_ee_world"), (
        "TAMPWorld must expose arm_home_ee_world after init"
    )
    assert set(world.arm_home_ee_world.keys()) == {"left", "right"}
    for arm in ("left", "right"):
        v = world.arm_home_ee_world[arm]
        assert isinstance(v, torch.Tensor), f"{arm} value must be a torch.Tensor"
        assert v.shape == (3,), f"{arm} value must be shape [3], got {tuple(v.shape)}"
    # Sanity: left arm is on +Y side, right arm on -Y side (T1 standing forward)
    assert world.arm_home_ee_world["left"][1].item() > 0, (
        "Left arm Y should be positive in world frame at home pose"
    )
    assert world.arm_home_ee_world["right"][1].item() < 0, (
        "Right arm Y should be negative in world frame at home pose"
    )
```

### Step 1.2: Run the test to verify it fails

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_arm_affinity.py::test_arm_home_ee_world_populated_after_init -v 2>&1 | tail -15
```

Expected: FAIL with `AttributeError: 'TAMPWorld' object has no attribute 'arm_home_ee_world'`.

If the test setup itself fails (e.g., `load_robot_container` signature mismatch, missing kwargs), check actual signatures in `cutamp/robots/__init__.py` and `cutamp/tamp_world.py:__init__` and adapt the helper. Don't change the assertions, just the setup.

### Step 1.3: Implement arm_home_ee_world computation in TAMPWorld

- [ ] Edit `cutamp/tamp_world.py`. Add at the END of `__init__` (after the existing `self._obj_to_aabb = {}` line at ~line 110):

```python
        # Arm-home end-effector positions in world frame. Used by the
        # arm-affinity priority function in plan-skeleton search to rank
        # which arm should pick which object. Uses palm frames (*_base_link)
        # because they're already in tool_frames per t1_planar_base.yml;
        # exact frame doesn't affect ranking, only LEFT-vs-RIGHT direction
        # matters.
        from curobo.types import JointState
        home_js = JointState.from_position(
            self._q_init.to(self.kinematics.device_cfg.device).unsqueeze(0)
        )
        home_ks = self.kinematics.compute_kinematics(home_js)
        self.arm_home_ee_world: Dict[str, torch.Tensor] = {
            "left":  home_ks.tool_poses.get_link_pose(
                "left_base_link", make_contiguous=True,
            ).position.flatten().detach().cpu(),
            "right": home_ks.tool_poses.get_link_pose(
                "right_base_link", make_contiguous=True,
            ).position.flatten().detach().cpu(),
        }
```

The `Dict[str, torch.Tensor]` annotation uses imports already present at the top of `tamp_world.py` (`Dict` from typing, `torch`). If `Dict` isn't already imported, add `from typing import Dict` to the imports.

**Adapt if needed**: if `self.kinematics.compute_kinematics(...)` raises a shape error (active vs full cspace), the planner kinematics may have locked DOFs. In that case, reproject `self._q_init` to active cspace using `self.kinematics.get_active_js(...)` first:

```python
        from curobo.types import JointState
        full_js = JointState.from_position(
            self._q_init.to(self.kinematics.device_cfg.device).unsqueeze(0)
        )
        active_js = self.kinematics.get_active_js(full_js)
        home_ks = self.kinematics.compute_kinematics(active_js)
        # ... rest unchanged
```

Use whichever path is needed for the kinematics to accept the JointState. If neither works (e.g., joint_names mismatch), surface as NEEDS_CONTEXT.

### Step 1.4: Run the test to verify it passes

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_arm_affinity.py::test_arm_home_ee_world_populated_after_init -v 2>&1 | tail -15
```

Expected: PASS.

### Step 1.5: Run smoke test (regression-safety)

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan 2>&1 | tail -20
```

Expected: ≥1 satisfying solution, motion plans succeed. The FK call at TAMPWorld init adds <100ms — should not affect runtime.

### Step 1.6: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/tamp_world.py \
  cutamp/tests/test_arm_affinity.py && \
git commit -m "$(cat <<'EOF'
feat: compute arm_home_ee_world on TAMPWorld init

Adds arm_home_ee_world: Dict[str, torch.Tensor] populated at init via FK
on t1_home. Each value is the world-frame XYZ of an arm's palm frame
(*_base_link, already in tool_frames per t1_planar_base.yml).

Used by the arm-affinity priority function (next commit) to rank which
arm should pick which object during plan-skeleton BFS.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Thread `ground_op_priority_fn` through the search layer

**Files:**
- Modify: `cutamp/task_planning/__init__.py:17-31` (`task_plan_generator` signature + forward to `breadth_first_search`)
- Modify: `cutamp/task_planning/search.py:208-272` (`breadth_first_search` signature + forward to `get_valid_ground_operators`)
- Modify: `cutamp/task_planning/search.py:72-205` (`get_valid_ground_operators` accepts `priority_fn`, sorts at end)
- Test: extend `cutamp/tests/test_arm_affinity.py`

This task is pure plumbing — adds an optional kwarg through three functions and uses it in one sort call. No behavior change when `priority_fn=None` (the default).

### Step 2.1: Write the failing test

- [ ] Append to `cutamp/tests/test_arm_affinity.py`:

```python
def test_get_valid_ground_operators_sorts_by_priority_fn():
    """When ground_op_priority_fn is provided, ground operators are returned
    in ascending priority order. Closer-arm picks come first."""
    from cutamp.task_planning.search import get_valid_ground_operators, _Node
    from cutamp.t1_domain import LeftPick, RightPick, all_t1_operators
    from cutamp.task_planning import Atom

    # Stub a _Node with a state where both LeftPick and RightPick can ground
    # on two blocks. We bypass the full pipeline by directly building atoms
    # for preconditions.
    # NOTE: this test is structural — it verifies the sort happens, not the
    # real-scene priority math. Real-scene math is exercised by the smoke
    # test at the end.
    # The fastest way to test the sort: provide a priority_fn that returns
    # a fixed score per ground op based on its serialized name, and assert
    # the returned list is sorted.
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from curobo.types import DeviceCfg
    from cutamp.robots.t1 import t1_home
    import os, torch

    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    device_cfg = DeviceCfg()
    robot = load_robot_container("t1", device_cfg)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=device_cfg.device)
    world = TAMPWorld(env=env, device_cfg=device_cfg, robot=robot, q_init=q_init)
    initial_node = _Node(state=world.initial_state, parent=None, operator=None, depth=0)

    # Priority: rank by reverse alphabetic order of the ground_op's repr.
    # This guarantees a non-trivial reordering vs the un-sorted default.
    def priority(ground_op):
        return -ord(repr(ground_op)[0])  # negate so it sorts reverse-alphabetic

    unsorted = get_valid_ground_operators(initial_node, all_t1_operators)
    sorted_ops = get_valid_ground_operators(
        initial_node, all_t1_operators, priority_fn=priority,
    )
    # Must return the same set
    assert set(map(repr, unsorted)) == set(map(repr, sorted_ops))
    # And the sorted version must be in non-decreasing priority order
    priorities = [priority(op) for op in sorted_ops]
    assert priorities == sorted(priorities), (
        f"ground_ops not sorted by priority: {priorities}"
    )


@needs_cuda
def test_task_plan_generator_accepts_priority_fn():
    """task_plan_generator must accept ground_op_priority_fn without error
    and still yield plans."""
    from cutamp.task_planning import task_plan_generator
    from cutamp.t1_domain import all_t1_operators
    from cutamp.envs.utils import get_env_dir, load_env
    from cutamp.tamp_world import TAMPWorld
    from cutamp.robots import load_robot_container
    from curobo.types import DeviceCfg
    from cutamp.robots.t1 import t1_home
    import os, torch

    env = load_env(os.path.join(get_env_dir(), "blocks_t1.yml"))
    device_cfg = DeviceCfg()
    robot = load_robot_container("t1", device_cfg)
    q_init = torch.as_tensor(t1_home, dtype=torch.float32, device=device_cfg.device)
    world = TAMPWorld(env=env, device_cfg=device_cfg, robot=robot, q_init=q_init)
    gen = task_plan_generator(
        world.initial_state, world.goal_state, all_t1_operators,
        ground_op_priority_fn=lambda op: 0.0,
        max_plan_skeletons=1,
    )
    # Pulling one plan should not raise.
    plan = next(gen)
    assert plan is not None
    assert len(plan) > 0
```

The first test (`test_get_valid_ground_operators_sorts_by_priority_fn`) needs `_Node` import from `search.py`. The `_Node` is private (leading underscore) — that's fine for testing internal behavior.

The `world.initial_state` attribute is the dict of atoms from `TAMPEnvironment.initial_state` (delegated via TAMPWorld).

### Step 2.2: Run the tests to verify they fail

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_arm_affinity.py::test_get_valid_ground_operators_sorts_by_priority_fn \
  cutamp/tests/test_arm_affinity.py::test_task_plan_generator_accepts_priority_fn \
  -v 2>&1 | tail -20
```

Expected: both FAIL — `get_valid_ground_operators` doesn't accept `priority_fn`, and `task_plan_generator` doesn't accept `ground_op_priority_fn`.

### Step 2.3: Modify `get_valid_ground_operators` to accept + use `priority_fn`

- [ ] Edit `cutamp/task_planning/search.py:72-74`. Change the signature and add the sort. Replace:

```python
def get_valid_ground_operators(
    node: _Node, operators: Sequence[Operator], verbose: bool = False
) -> list[GroundOperator]:
    """
    Get all valid ground operators by testing the operators, binding samples for the unspecified variables, and
    checking preconditions are satisfied.
    """
```

with:

```python
def get_valid_ground_operators(
    node: _Node,
    operators: Sequence[Operator],
    verbose: bool = False,
    priority_fn: Optional["Callable[[GroundOperator], float]"] = None,
) -> list[GroundOperator]:
    """
    Get all valid ground operators by testing the operators, binding samples for the unspecified variables, and
    checking preconditions are satisfied.

    When ``priority_fn`` is provided, the returned ground operators are
    sorted ascending by priority (smaller = explored first by BFS). This
    biases BFS exploration without changing search correctness — all valid
    ground ops are still returned, just in a different order.
    """
```

At the END of the function (replacing the existing `return ground_ops` at line 205), add the sort:

```python
    if priority_fn is not None:
        # Stable sort: equal-priority groundings preserve declaration order.
        ground_ops.sort(key=priority_fn)
    return ground_ops
```

Also add `Callable` to the imports at the top of the file (line 15):

```python
from typing import Callable, Generator, List, Optional, Sequence
```

### Step 2.4: Modify `breadth_first_search` to thread the kwarg

- [ ] Edit `cutamp/task_planning/search.py:208-215`. Change the signature:

```python
def breadth_first_search(
    initial_state: State,
    goal_state: State,
    operators: Sequence[Operator],
    continue_branch_after_goal: bool = False,
    explored_state_check: bool = True,
    verbose: bool = False,
    ground_op_priority_fn: Optional[Callable[[GroundOperator], float]] = None,
) -> Generator[List[GroundOperator], None, None]:
```

Then at line 262 (the existing `ground_ops = get_valid_ground_operators(...)` call), pass the priority:

```python
        ground_ops = get_valid_ground_operators(
            node, operators, verbose=verbose, priority_fn=ground_op_priority_fn,
        )
```

### Step 2.5: Modify `task_plan_generator` to thread the kwarg

- [ ] Edit `cutamp/task_planning/__init__.py:17-31`. Replace the function with:

```python
from typing import Callable, Optional, Sequence

from .base_structs import Atom, Fluent, GroundOperator, Operator, OperatorMetadata, Parameter, State
from .tamp_structs import Constraint, Cost, GroundTAMPOperator, PlanSkeleton, TAMPOperator
from .search import breadth_first_search


def task_plan_generator(
    initial: State,
    goal: State,
    operators: Sequence[Operator],
    explored_state_check: bool = True,
    max_plan_skeletons: int = 99999,
    ground_op_priority_fn: Optional[Callable[[GroundOperator], float]] = None,
) -> Sequence[PlanSkeleton]:
    """Iterator that yields task plans.

    When ``ground_op_priority_fn`` is provided, BFS sorts ground operators
    by ascending priority at each expansion step (closer-arm picks first
    for the arm-affinity case). Cross-body groundings are still enumerated,
    just later in the order — BFS naturally backtracks if same-side fails.
    """
    plan_iter = breadth_first_search(
        initial, goal, operators,
        explored_state_check=explored_state_check,
        ground_op_priority_fn=ground_op_priority_fn,
    )
    for _ in range(max_plan_skeletons):
        try:
            plan = next(plan_iter)
            yield plan
        except StopIteration:
            break
```

Update the import line if `Callable, Optional` weren't already imported.

### Step 2.6: Run the tests to verify they pass

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_arm_affinity.py::test_get_valid_ground_operators_sorts_by_priority_fn \
  cutamp/tests/test_arm_affinity.py::test_task_plan_generator_accepts_priority_fn \
  -v 2>&1 | tail -15
```

Expected: both PASS.

### Step 2.7: Smoke test (regression-safety)

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan 2>&1 | tail -20
```

Expected: ≥1 satisfying. No regression because priority_fn defaults to None (no behavior change).

### Step 2.8: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/task_planning/__init__.py \
  cutamp/task_planning/search.py \
  cutamp/tests/test_arm_affinity.py && \
git commit -m "$(cat <<'EOF'
feat: thread ground_op_priority_fn through plan-skeleton search

Adds optional ground_op_priority_fn parameter to task_plan_generator,
breadth_first_search, and get_valid_ground_operators. When provided,
ground operators returned from get_valid_ground_operators are sorted
ascending by priority. Defaults to None (no behavior change).

Enables biased BFS for arm-affinity (wire-up in next commit).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `make_arm_affinity_priority_fn` and wire into the call site

**Files:**
- Modify: `cutamp/algorithm.py` — add factory + pass to `task_plan_generator` at line 376
- Test: extend `cutamp/tests/test_arm_affinity.py`

This task brings the pieces together: a factory builds a priority function bound to a specific TAMPWorld, and the existing `task_plan_generator` call site receives it.

### Step 3.1: Write the failing tests

- [ ] Append to `cutamp/tests/test_arm_affinity.py`:

```python
@needs_cuda
def test_arm_affinity_priority_orders_closer_arm_first():
    """priority(LeftPick(block_on_left)) < priority(LeftPick(block_on_right)).

    For the blocks_t1 env: block2 (orange) is at x=0.35, y=-0.325 (right side
    in world Y); block3 (green) is at x=0.36, y=+0.32 (left side in world Y).
    Wait — let's check actual env values dynamically rather than hardcoding."""
    from cutamp.algorithm import make_arm_affinity_priority_fn
    world = _make_world()

    # Find the two blocks and their world Y values.
    block_ys = {obj.name: obj.pose[1] for obj in world.env.movables}
    # In blocks_t1: blocks are roughly symmetric — one on +Y, one on -Y.
    blocks_pos_y = [n for n, y in block_ys.items() if y > 0]
    blocks_neg_y = [n for n, y in block_ys.items() if y < 0]
    assert blocks_pos_y, "Test env must have a block at +Y for this test"
    assert blocks_neg_y, "Test env must have a block at -Y for this test"

    priority = make_arm_affinity_priority_fn(world)

    # Stub two ground ops: LeftPick(block_on_left), LeftPick(block_on_right).
    # Use the real ground operator API.
    from cutamp.t1_domain import LeftPick
    block_left = blocks_pos_y[0]   # T1 facing +X; +Y is its left
    block_right = blocks_neg_y[0]
    op_left_close  = LeftPick.ground({"obj": block_left, "grasp": "grasp0", "q": "left_q0"})
    op_left_far    = LeftPick.ground({"obj": block_right, "grasp": "grasp0", "q": "left_q0"})

    pri_close = priority(op_left_close)
    pri_far   = priority(op_left_far)
    assert pri_close < pri_far, (
        f"Expected priority(LeftPick(close)) < priority(LeftPick(far)); "
        f"got {pri_close} vs {pri_far}"
    )


@needs_cuda
def test_arm_affinity_priority_zero_for_non_pick():
    """Non-pick operators return 0 priority (preserves original BFS order)."""
    from cutamp.algorithm import make_arm_affinity_priority_fn
    from cutamp.t1_domain import LeftMoveFree
    world = _make_world()
    priority = make_arm_affinity_priority_fn(world)
    # LeftMoveFree(q_start, q_end) — values aren't picks
    op = LeftMoveFree.ground({"q_start": "left_q0", "q_end": "left_q1"})
    assert priority(op) == 0.0


@needs_cuda
def test_arm_affinity_priority_zero_for_missing_block():
    """Pick with an unknown block name returns 0 (graceful degradation)."""
    from cutamp.algorithm import make_arm_affinity_priority_fn
    from cutamp.t1_domain import LeftPick
    world = _make_world()
    priority = make_arm_affinity_priority_fn(world)
    op = LeftPick.ground({"obj": "nonexistent_block", "grasp": "grasp0", "q": "left_q0"})
    assert priority(op) == 0.0
```

If `LeftMoveFree`'s parameter names aren't `q_start`/`q_end`, check `cutamp/t1_domain.py` for the actual names and adapt the test's substitution dict.

### Step 3.2: Run the tests to verify they fail

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_arm_affinity.py -v -k "priority_orders or priority_zero" 2>&1 | tail -20
```

Expected: all three new tests FAIL with `ImportError: cannot import name 'make_arm_affinity_priority_fn' from 'cutamp.algorithm'`.

### Step 3.3: Implement `make_arm_affinity_priority_fn` in `algorithm.py`

- [ ] Edit `cutamp/algorithm.py`. Locate the existing imports section near the top of the file (after the existing `from cutamp...` imports). Add:

```python
from typing import Callable
import numpy as np
```

(If `numpy as np` and `Callable` are already imported, skip.)

Then add this function near the other module-level functions (e.g., right after `heuristic_fn` definition, around line 80):

```python
def make_arm_affinity_priority_fn(world) -> Callable[[object], float]:
    """Build a priority function for BFS sibling ordering.

    For pick operators, returns the 3D Euclidean distance from the arm's
    home end-effector position to the block's world pose. Closer = lower
    priority = explored first by biased BFS. For non-pick operators or
    pick operators whose block doesn't resolve (e.g., placeholder name
    not yet bound to a real scene object), returns 0.0.

    Same-side picks bubble to the top of the BFS exploration order, so
    the first satisfying plan skeleton tends to be the same-side assignment.
    Cross-body groundings are still enumerated (just later); BFS naturally
    backtracks if same-side fails feasibility.
    """
    def priority(ground_op) -> float:
        meta = ground_op.operator.metadata
        if meta.action_type != "pick" or meta.arm is None:
            return 0.0
        # LeftPick/RightPick parameter order is (obj, grasp, q). Block is values[0].
        block_name = ground_op.values[0]
        try:
            obj = world.get_object(block_name)
        except KeyError:
            return 0.0
        if obj is None or obj.pose is None:
            return 0.0
        block_xyz = np.asarray(obj.pose[:3], dtype=np.float64)
        arm_home_xyz = world.arm_home_ee_world[meta.arm].cpu().numpy().astype(np.float64)
        return float(np.linalg.norm(block_xyz - arm_home_xyz))
    return priority
```

### Step 3.4: Wire into the `task_plan_generator` call site

- [ ] Edit `cutamp/algorithm.py:376` (the existing `task_plan_generator(...)` call inside `setup_cutamp`). Find:

```python
    with timer.time("get_plan_generator", log_callback=_log.info):
        plan_gen = task_plan_generator(
            world.initial_state,
            world.goal_state,
            operators=all_operators,
            explored_state_check=config.explored_state_check,
        )
```

Replace with:

```python
    with timer.time("get_plan_generator", log_callback=_log.info):
        arm_affinity_priority = make_arm_affinity_priority_fn(world)
        plan_gen = task_plan_generator(
            world.initial_state,
            world.goal_state,
            operators=all_operators,
            explored_state_check=config.explored_state_check,
            ground_op_priority_fn=arm_affinity_priority,
        )
```

### Step 3.5: Run all three unit tests — should PASS

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_arm_affinity.py -v -k "priority_orders or priority_zero" 2>&1 | tail -15
```

Expected: all three PASS.

### Step 3.6: Smoke test

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan 2>&1 | tail -25
```

Expected: ≥1 satisfying. Look at the printed plan skeleton — the picks should now be same-side (LeftPick(block_on_left), RightPick(block_on_right)).

### Step 3.7: Commit

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP && git add \
  cutamp/algorithm.py \
  cutamp/tests/test_arm_affinity.py && \
git commit -m "$(cat <<'EOF'
feat: bias plan-skeleton BFS to prefer closer arm for picks

Adds make_arm_affinity_priority_fn factory in algorithm.py that returns a
priority function scoring pick operators by 3D Euclidean distance from
the arm's home end-effector to the block's world pose. Wires into the
task_plan_generator call at algorithm.py:376.

Effect: BFS explores closer-arm picks first; first satisfying skeleton
tends to be same-side. Cross-body groundings are still enumerated and
BFS naturally backtracks if same-side fails.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Integration verification (5-run statistical test)

**Files:** none modified — pure verification.

This task confirms the priority function actually changes observed planner behavior. Without it, the bias might be too weak (e.g., IK noise dominates), or some other code path overrides the ordering.

### Step 4.1: Run the smoke test 5 times and tally same-side hits

- [ ] Run:

```bash
cd /home/yoonwoo/cuTAMP
for i in 1 2 3 4 5; do
  echo "=== Run $i ==="
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
    --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan 2>&1 | \
    grep -E "final_plan_skeleton|LeftPick|RightPick" | head -5
done
```

For each run, the plan log should show LeftPick paired with the +Y block and RightPick paired with the -Y block.

### Step 4.2: Verify statistical preference

- [ ] Eyeball the 5 runs' plan skeletons. Count how many run same-side picks.

Expected: ≥4 of 5 runs choose same-side. (Pre-fix baseline was ~50%; post-fix should be much higher.)

If <4 of 5 are same-side: the bias is too weak. Possible causes:
- BFS isn't sorting (check Task 2.6 test outcome).
- Priority function returning 0 unexpectedly (check Task 3.5).
- IK satisfaction noise causes downstream non-skeleton-related variance.

Surface as DONE_WITH_CONCERNS with the actual count if the bias is partially working but below threshold.

### Step 4.3: Manual cross-body fallback test

- [ ] Temporarily edit `cutamp/envs/assets/blocks_t1.yml` to position both blocks on the same side (e.g., both at y=+0.3):

```yaml
    block2:
      pose: [0.35, 0.30, 0.025, 0.03, 0.0, 0.0, 0.99]
    block3:
      pose: [0.36, 0.30, 0.025, -0.43, 0.0, 0.0, 0.90]
```

Run the smoke test once. Expected: plan must use one cross-body pick (both blocks are on left; left arm can only pick one before placing). Verifies BFS still enumerates cross-body groundings.

Revert the env file after this test:

```bash
cd /home/yoonwoo/cuTAMP && git checkout cutamp/envs/assets/blocks_t1.yml
```

### Step 4.4: Final verification — full test suite

- [ ] Run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/ -v 2>&1 | tail -50
```

Expected: all tests PASS, including the new `test_arm_affinity.py` tests from Tasks 1–3.

---

## Self-review checklist (for plan-writer, post-write)

**1. Spec coverage**: Spec requires `arm_home_ee_world` on TAMPWorld (Task 1), `make_arm_affinity_priority_fn` factory (Task 3), threading through 3 search files (Task 2), wire-up at algorithm.py:376 (Task 3), 4 unit tests (Tasks 1+2+3). ✅

**2. Placeholder scan**: No TBD, TODO, no "implement later", no vague guidance. All code blocks complete. ✅

**3. Type consistency**: 
- `ground_op_priority_fn: Optional[Callable[[GroundOperator], float]]` is consistent across `task_plan_generator`, `breadth_first_search`, `get_valid_ground_operators`.
- `priority_fn` is the parameter name inside `get_valid_ground_operators` (matches the spec — the rename mirrors how `verbose` is internal).
- `arm_home_ee_world: Dict[str, torch.Tensor]` is shape [3] per arm, on CPU. The priority function moves it through `.cpu().numpy()`.
- `make_arm_affinity_priority_fn(world) -> Callable[[object], float]` returns the inner function with `(ground_op) -> float` signature. ✅

**4. Adaptation guidance**: Step 1.3 includes a fallback path for JointState shape issues. Step 3.1 includes adaptation guidance for unknown operator parameter names. ✅
