# Arm-affinity priority for TAMP plan-skeleton search — design

**Date**: 2026-05-24
**Scope**: Bias the BFS plan-skeleton enumeration so the closer arm picks each block by default, with graceful cross-body fallback when same-side is infeasible.

## Context

In the current cuTAMP planner, both `LeftPick(block)` and `RightPick(block)` are valid groundings for any block (the only preconditions are `LeftHandEmpty` / `RightHandEmpty` / `LeftCanMove` / `RightCanMove` — none reference block position). BFS at `cutamp/task_planning/search.py:88` iterates operators in their declaration order from `t1_domain.all_t1_operators`, and the heuristic at `cutamp/algorithm.py:40-80` scores skeletons by constraint-failure rate + plan length — **no arm-distance term**.

Consequence: when blocks are clearly closer to one arm in world space (e.g., orange-on-left, green-on-right in `blocks_t1`), the planner still picks the cross-body assignment roughly half the time. Whichever skeleton's particles happen to satisfy first wins.

User goal: usually use the closer arm, but fall back to cross-body when needed (closer arm busy, no IK solution, collision). Soft preference, not a hard rule.

## Decisions (locked in)

- **Soft preference, not hard rule.** BFS still enumerates cross-body groundings — they just get explored AFTER same-side. If same-side fails feasibility, BFS naturally backtracks.
- **Distance metric**: full 3D Euclidean `||arm_home_ee_world - block_world_pose||`.
- **Search structure**: biased BFS (sort siblings within each expansion by priority). NOT full A*. A* would be more globally optimal but requires heap-based search, principled `g(n)+h(n)`, and admissibility analysis — overkill for current envs (`blocks_t1` has 2 blocks; realistic stretch is 6-10). Biased BFS scales fine in that range; the per-skeleton motion-planning cost dominates.
- **A* compatibility**: the `ground_op_priority_fn(ground_op) → float` interface plugs unchanged into a future A* upgrade as `h(n)`. No design lock-in.
- **Non-pick operators**: priority returns 0; preserves existing within-operator-class order.

## Architecture

```
TAMPWorld init
  └─ compute arm_home_ee_world via FK on t1_home  ────────────┐
                                                              ▼
algorithm.setup_cutamp                  make_arm_affinity_priority_fn(world)
  └─ priority_fn ────────► task_plan_generator(..., ground_op_priority_fn)
                            └─ breadth_first_search(..., ground_op_priority_fn)
                                 └─ get_valid_ground_operators(priority_fn=priority_fn)
                                      └─ ground_ops.sort(key=priority_fn)
                                            ▼
                                  BFS explores closer-arm picks first
                                            ▼
                                  first satisfying skeleton tends to be same-side
```

## Components

### 1. `TAMPWorld.arm_home_ee_world` (new attribute)

**File**: `cutamp/tamp_world.py` (modify `__init__`)
**Type**: `Dict[Literal["left", "right"], torch.Tensor]` (each tensor shape `[3]`, world frame)

Computed once at init via FK on `t1_home`:

```python
from cutamp.robots.t1 import t1_home

# After the kinematics is built:
home_js = JointState.from_position(
    torch.as_tensor(t1_home, device=self.kinematics.device_cfg.device).unsqueeze(0)
)
home_ks = self.kinematics.compute_kinematics(home_js)
self.arm_home_ee_world = {
    "left":  home_ks.tool_poses.get_link_pose("left_hand_link", make_contiguous=True)
                .position.flatten().detach().cpu(),
    "right": home_ks.tool_poses.get_link_pose("right_hand_link", make_contiguous=True)
                .position.flatten().detach().cpu(),
}
```

Stored on CPU as a 3-vector each (compared against block poses which are stored on CPU in env YAML).

### 2. `make_arm_affinity_priority_fn(world)` (new factory)

**File**: `cutamp/algorithm.py`
**Signature**: `Callable[[TAMPWorld], Callable[[GroundOperator], float]]`

```python
def make_arm_affinity_priority_fn(world: TAMPWorld) -> Callable[[GroundOperator], float]:
    """Return a priority function for BFS sibling ordering.

    Pick operators get arm-distance to the block as their priority
    (smaller = explored first → same-side wins ties). Non-pick operators
    return 0, preserving their existing relative order in BFS expansion.
    """
    def priority(ground_op) -> float:
        meta = ground_op.operator.metadata
        if meta.action_type != "pick" or meta.arm is None:
            return 0.0
        # LeftPick/RightPick: first parameter is the block name.
        block_name = ground_op.values[0]
        obj = world.get_object(block_name)
        if obj is None:
            return 0.0
        block_xyz = np.asarray(obj.pose[:3], dtype=np.float64)
        arm_home_xyz = world.arm_home_ee_world[meta.arm].cpu().numpy().astype(np.float64)
        return float(np.linalg.norm(block_xyz - arm_home_xyz))
    return priority
```

**Edge cases**:
- `obj is None` (block name not found): priority 0 (no preference, graceful degradation).
- block_xyz dtype mismatch: explicit `astype(np.float64)` for stable comparison.

### 3. Thread `ground_op_priority_fn` through the search layer

**File**: `cutamp/task_planning/__init__.py`

```python
def task_plan_generator(
    initial: State,
    goal: State,
    operators: Sequence[Operator],
    max_plan_skeletons: int = 99999,
    explored_state_check=None,
    ground_op_priority_fn: Optional[Callable[[GroundOperator], float]] = None,  # NEW
):
    plan_iter = breadth_first_search(
        initial, goal, operators,
        explored_state_check=explored_state_check,
        ground_op_priority_fn=ground_op_priority_fn,  # NEW
    )
    for _ in range(max_plan_skeletons):
        ...
```

**File**: `cutamp/task_planning/search.py` (`breadth_first_search` + `get_valid_ground_operators`)

```python
def breadth_first_search(
    initial, goal, operators,
    explored_state_check=None,
    ground_op_priority_fn: Optional[Callable[[GroundOperator], float]] = None,  # NEW
):
    ...
    children = get_valid_ground_operators(
        node, operators, verbose=False, priority_fn=ground_op_priority_fn,  # NEW
    )
    ...

def get_valid_ground_operators(
    node: _Node,
    operators: Sequence[Operator],
    verbose: bool = False,
    priority_fn: Optional[Callable[[GroundOperator], float]] = None,  # NEW
) -> list[GroundOperator]:
    ground_ops = []
    ... (existing grounding logic — unchanged) ...

    if priority_fn is not None:
        # Stable sort: ties preserve declaration order within an operator class.
        ground_ops.sort(key=priority_fn)
    return ground_ops
```

### 4. Wire-up at the call site

**File**: `cutamp/algorithm.py:376` (the single call to `task_plan_generator`)

```python
priority_fn = make_arm_affinity_priority_fn(world)
plan_gen = task_plan_generator(
    initial_state, goal_state, all_t1_operators,
    max_plan_skeletons=config.max_plan_skeletons,
    ground_op_priority_fn=priority_fn,  # NEW
)
```

### 5. Tests

**File**: `cutamp/tests/test_arm_affinity.py` (NEW)

```python
"""Tests for arm-affinity priority function in plan-skeleton search."""
import numpy as np
import torch


def test_arm_affinity_priority_orders_closer_arm_first():
    """priority(LeftPick(block_on_left)) < priority(LeftPick(block_on_right))."""
    # Build a minimal mock world with two blocks at known positions.
    # Confirm the priority function returns smaller values for closer arm-block pairs.
    ...


def test_priority_zero_for_non_pick_ops():
    """priority(LeftMoveFree(...)) == 0 — non-pick operators preserve original order."""
    ...


def test_priority_zero_for_missing_block():
    """priority(LeftPick(?nonexistent)) == 0 — graceful degradation."""
    ...
```

Plus integration test: 5 smoke runs on `blocks_t1`; assert ≥4 runs produce same-side assignment (`LeftPick(block2) ... RightPick(block3)`).

## Falls-back behavior (explicit)

1. **Closer arm already holding**: `LeftHandEmpty` precondition is already false → `LeftPick(block_on_left)` doesn't ground → BFS explores `RightPick(block_on_left)` as the only option. No change to behavior.
2. **Closer arm has no IK**: `LeftPick(block_on_left)` grounds but its particles fail satisfaction → `sample_plan_skeleton` returns `has_solution=False` → outer loop pulls the next skeleton from `plan_gen`, which (by biased BFS ordering) is the cross-body option. Same as current behavior, just a couple iterations later.
3. **Symmetric blocks (Y=0 or equal-distance)**: priorities tie → stable sort preserves declaration order → behavior matches today (coin-flip determined by IK seeds).

## Files / LOC estimate

| File | LOC delta | Notes |
|---|---|---|
| `cutamp/tamp_world.py` | +10 | `arm_home_ee_world` FK at init |
| `cutamp/algorithm.py` | +20 | `make_arm_affinity_priority_fn` + wire-up |
| `cutamp/task_planning/__init__.py` | +5 | thread kwarg through `task_plan_generator` |
| `cutamp/task_planning/search.py` | +5 | thread kwarg + sort call |
| `cutamp/tests/test_arm_affinity.py` | +30 | unit tests (NEW file) |

Total: ~70 LOC.

## Verification

### Smoke test (regression-safety)
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m cutamp.scripts.run_cutamp \
  --env blocks_t1 -n 16 --num_opt_steps 50 --motion_plan
```
Pass: ≥1 satisfying solution, all motion plans succeed. (Same bar as the existing smoke test.)

### Same-side preference test (the actual goal)
Run the same command 5 times. In ≥4 runs, the plan skeleton starts with `LeftPick(block2)` (orange, on left) and includes `RightPick(block3)` (green, on right) — verifying the closer-arm preference holds.

### Cross-body fallback test (manual)
Modify `blocks_t1.yml` to disable the left arm temporarily (e.g., via a precondition hack or by closing the gripper at init). Run the planner. Verify it finds `RightPick(block2)` (cross-body) instead of failing.

### Unit tests
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /home/yoonwoo/miniconda3/envs/tamp/bin/python -m pytest \
  cutamp/tests/test_arm_affinity.py -v
```
Pass: all tests in the new file PASS.

## Risk / open questions

1. **Stale block pose for re-picks**: priority uses the INITIAL pose from `world.get_object(block).pose`. If a block has been moved by an earlier action in the skeleton, subsequent re-picks of the same block see stale data. Current envs don't re-pick → punt. Future fix: thread the BFS node state into the priority function so it can query the block's current pose.
2. **Symmetric envs**: blocks centered on Y=0 → no preference, coin-flip determined by IK seeds. Acceptable.
3. **A* upgrade path**: if globally-optimal skeleton selection becomes important, swap `breadth_first_search` for an A* implementation; the same `ground_op_priority_fn` becomes `h(n)`. Document this in `search.py` so future-you remembers the contract.
4. **`world.get_object` API drift**: the priority function depends on this method returning an object with a `.pose` attribute. If `TAMPWorld` refactors that API, the priority function needs updating. Mitigation: add a one-line assertion in the priority function that surfaces the breakage clearly.

## Out of scope

- A* upgrade (deferred until problem requires it).
- Per-step pose updates in priority (stale-pose limitation accepted).
- Arm-affinity for place / push / push_stick (only pick is addressed; same pattern would extend).
- Soft-cost ranking for same-side preference (already covered by biased BFS — adding a redundant soft cost would be noise).
