# COM-over-base-polygon cost

A soft cuRobo cost that penalizes trajectories whose center of mass projects
outside an axis-aligned support rectangle in `mobile_base_link` frame.

## Motivation

The T1 humanoid sits on a 0.25 × 0.25 m wheeled platform. Body joints
(`ankle_pitch`, `knee_pitch`, `Torso_Pitch`, `Waist_Yaw`) and both arms are all
unlocked during planning, so the optimizer is free to produce postures whose
projected COM leaves the platform footprint — physically these would tip the
robot. The cost discourages those postures while leaving the planner unconstrained
for safe configurations.

## Math

### Center of mass

cuRobo computes the per-timestep mass-weighted world COM via a CUDA kernel
template selected at planner build time:

$$
\mathrm{COM}_\text{world}(t) \;=\;
\frac{\sum_i m_i \cdot \mathbf{T}^{\text{world}}_{L_i}(t)\, \mathbf{c}_i}{\sum_i m_i}
$$

where $i$ ranges over inertia-bearing URDF links, $m_i$ is the link mass,
$\mathbf{c}_i$ is the link-frame COM offset (both from `<inertial>` blocks),
and $\mathbf{T}^{\text{world}}_{L_i}(t)$ is the link's world pose at trajectory
step $t$. The kernel populates `KinematicsState.robot_com` as
$[x, y, z, M_\text{total}]$ per timestep — we use only the $xy$ projection.

### Projecting into base frame

The support rectangle is fixed in `mobile_base_link` frame. Let
$\mathbf{T}^{\text{world}}_{B}(t)$ be the base's world pose (from FK on the same
state). Then

$$
\mathbf{p}_\text{base}(t) \;=\;
\bigl(\mathbf{T}^{\text{world}}_{B}(t)\bigr)^{-1}\,
\begin{bmatrix} \mathrm{COM}_\text{world}(t) \\ 1 \end{bmatrix}
$$

and we keep $(p_x, p_y) \in \mathbb{R}^2$.

### Rectangle distance

For half-extents $\mathbf{h} = (h_x, h_y) = (0.10, 0.15)$ m (20 cm fore/aft × 30 cm
lateral), the per-axis signed distance to the rectangle edge is

$$
\Delta = |\mathbf{p}_\text{base}| - \mathbf{h}
$$

(absolute value taken element-wise; positive on an axis means outside on that
axis). Two clamps separate the regions:

$$
\mathbf{u}_\text{out} = \max(\Delta, \mathbf{0}), \qquad
\mathbf{u}_\text{in}  = \max(\Delta + m, \mathbf{0})
$$

where $m$ is the inside margin (default 0.02 m). $\mathbf{u}_\text{out}$ is
non-zero only when the COM is outside the rectangle on that axis;
$\mathbf{u}_\text{in}$ becomes non-zero starting $m$ inside the edge, then
remains active as the COM moves further out (saturating but not vanishing).

### Final cost

Summed across the two axes and weighted by the user-tunable inside-barrier
strength $w_\text{in}$ (default 1.0), then scaled by the cost-config weight
$w$ (default 50.0):

$$
\mathcal{L}(t) \;=\;
w \cdot \Bigl( \|\mathbf{u}_\text{out}(t)\|^2_2
            \;+\; w_\text{in}\, \|\mathbf{u}_\text{in}(t)\|^2_2 \Bigr)
$$

Three regimes:
- **Strictly inside the inset rectangle** ($|p_{\cdot}| < h_{\cdot} - m$):
  $\mathcal{L} = 0$. Planner is unencumbered.
- **Inside-margin band** ($h_{\cdot} - m < |p_{\cdot}| < h_{\cdot}$): only the
  inside term active — soft barrier nudges COM back toward center.
- **Outside the rectangle** ($|p_{\cdot}| > h_{\cdot}$): both terms active;
  $\mathbf{u}_\text{out}$ grows quadratically with distance, $\mathbf{u}_\text{in}$
  is saturated. Steepest gradient back to feasibility.

The cost returns $[B, H, 1]$ per the `RobotCostManager` convention; it is
summed across the horizon by the optimizer.

## Implementation

### Bundled cuRobo (2-line plumbing)

`compute_com` is a Python kwarg on `Kinematics.__init__` that selects a
heavier compiled CUDA kernel template — it must be set at planner build time
and cannot be flipped post-construction. The runtime kinematics is built at
`curobo/_src/transition/robot_state_transition.py:50` with `compute_com`
defaulting to `False` in the constructor; no upstream config field plumbs it.

Two minimal edits expose it via cfg without changing default behavior:

```python
# curobo/_src/transition/robot_state_transition_cfg.py — RobotStateTransitionCfg
compute_com: bool = False  # cuTAMP local addition
```

```python
# curobo/_src/transition/robot_state_transition.py:50
self.robot_model = Kinematics(
    self.config.robot_config.kinematics,
    compute_jacobian=False,
    compute_com=self.config.compute_com,  # was: compute_jacobian=False only
)
```

Both edits are annotated with `cuTAMP local addition` so the next cuRobo
upgrade audit catches them.

### Tracked link

`mobile_base_link` is added to `tool_frames` in
`cutamp/robots/assets/t1_description/t1_planar_base.yml` so its world pose
shows up in `state.tool_poses` (no extra FK call inside the cost). IK is
unaffected — `t1.py:204` overrides `tool_frames` to a single per-arm frame
when constructing the IK solver; only the planner sees the augmented YAML.

### Cost class — `cutamp/com_polygon_cost.py`

Subclasses `BaseCost` / `BaseCostCfg`. Forward (cleaned of comments here):

```python
def forward(self, state):
    b, h = state.joint_state.position.shape[:2]
    n = b * h
    com_world = state.cuda_robot_model_state.robot_com[..., :3].reshape(n, 3)
    base_T = (
        state.tool_poses
             .get_link_pose(self._base_link_name, make_contiguous=True)
             .get_matrix()
             .reshape(n, 4, 4)
    )
    ones  = torch.ones_like(com_world[:, :1])
    com_h = torch.cat([com_world, ones], dim=-1).unsqueeze(-1)        # [N, 4, 1]
    com_in_base = (torch.linalg.inv(base_T) @ com_h).squeeze(-1)[:, :2]
    offset  = torch.abs(com_in_base) - self._half_extents
    outside = torch.clamp(offset,                       min=0.0)
    inside  = torch.clamp(offset + self._inside_margin, min=0.0)
    cost = (outside ** 2).sum(-1) + self._inside_weight * (inside ** 2).sum(-1)
    return cost.reshape(b, h, 1) * self._weight
```

Two cuRobo quirks the code handles explicitly:
- `state.cuda_robot_model_state.robot_com` is `[B, H, 4]`; `tool_poses` link
  matrices come back flattened to `[B*H, 4, 4]`. We flatten everything to
  `N = B*H`, do the rectangle math in flat space, then reshape to `[B, H, 1]`.
- `state.robot_com` doesn't exist as a `RobotState` property — `RobotState`
  exposes `tool_poses`/`robot_spheres` but COM lives one level deeper on
  `cuda_robot_model_state` (which is the `KinematicsState`).

### Dispatch — `cutamp/_curobo_internals.py`

`RobotCostManager.compute_costs` (in `curobo/_src/rollout/cost_manager/cost_manager_robot.py:195-286`)
hardcodes evaluation of exactly four cost names: `tool_pose`, `cspace`,
`self_collision`, `scene_collision`. `register_cost("com_polygon", …)` adds
to the registry but `compute_costs` never calls the new cost. The cuRobo
custom-cost docs gloss over this in v0.8.

We monkey-patch each manager so any cost in `_extra_costs` is invoked after
the hardcoded set:

```python
def _ensure_extra_cost_dispatch(manager):
    if hasattr(manager, "_extra_costs"):
        return
    manager._extra_costs = {}
    orig = manager.compute_costs

    def wrapped(state, cost_collection=None, goal=None, **kwargs):
        cc = orig(state, cost_collection=cost_collection, goal=goal, **kwargs)
        for name, cost in manager._extra_costs.items():
            if cost.enabled:
                cc.add(cost.forward(state), name)
        return cc

    manager.compute_costs = wrapped


def add_extra_cost(planner, name, cost):
    for rollout in getattr(planner.trajopt_solver, "optimizer_rollouts", []):
        for mgr in (
            getattr(rollout, "cost_manager", None),
            getattr(rollout, "metrics_cost_manager", None),
        ):
            if mgr is None:
                continue
            _ensure_extra_cost_dispatch(mgr)
            mgr._extra_costs[name] = cost
```

Passing the full `state` (instead of pre-extracting fields) lets future extra
costs pick whichever fields they need without widening the wrapper signature.

### Wiring — `cutamp/robots/t1.py` `get_t1_motion_planner`

`MotionPlannerCfg.create` accepts `trajopt_transition_model` as either a path
or a dict. We load the default YAML, inject `compute_com: True`, pass the
dict — that flows through `TrajOptSolverCfg.create → create_solver_core_cfg`
into the `RobotStateTransitionCfg` instance built per rollout:

```python
def _trajopt_transition_dict_with_compute_com():
    d = resolve_config(join_path(get_task_configs_path(),
                                 "trajopt/transition_bspline_trajopt.yml"))
    d["transition_model_cfg"]["compute_com"] = True
    return d

def get_t1_motion_planner(scene, *, enable_com_polygon=True, ...):
    ...
    cfg = MotionPlannerCfg.create(
        ...,
        trajopt_transition_model=_trajopt_transition_dict_with_compute_com(),
    )
    planner = MotionPlanner(cfg)
    if enable_com_polygon:
        rollout_device_cfg = planner.trajopt_solver.optimizer_rollouts[0].device_cfg
        cost_cfg = ComOverBasePolygonCostCfg(weight=[50.0], device_cfg=rollout_device_cfg)
        add_extra_cost(planner, "com_polygon", ComOverBasePolygonCost(cost_cfg))
    return planner
```

`compute_com=True` is on regardless of `enable_com_polygon` (the kernel
overhead is small and the field is useful for future costs); only the cost
registration is gated.

## Knobs

`ComOverBasePolygonCostCfg`:

| field | default | meaning |
| --- | --- | --- |
| `weight` | `[50.0]` | overall multiplier (cost-config convention) |
| `half_extents` | `(0.10, 0.15)` m | rectangle half-sizes (X fore/aft, Y lateral) |
| `inside_margin` | `0.02` m | how far inside the edge the soft barrier starts; `0` disables the barrier |
| `inside_weight` | `1.0` | relative scale of the inside barrier vs the outside quadratic |
| `base_link_name` | `"mobile_base_link"` | tool frame to use as the rectangle's local origin |

## Verification

REPL smoke check after planner construction:

```python
from cutamp.robots.t1 import get_t1_motion_planner
from curobo.scene import Scene

planner = get_t1_motion_planner(Scene())
mgr = planner.trajopt_solver.optimizer_rollouts[0].cost_manager
assert "com_polygon" in mgr._extra_costs

tm = planner.trajopt_solver.optimizer_rollouts[0].transition_model
assert tm.robot_model.compute_com is True
js = tm.robot_model.default_joint_state.clone()
js.position = js.position.view(1, 1, -1)
ks = tm.robot_model.compute_kinematics(js)
assert ks.robot_com is not None  # populated by the heavier kernel template
```

End-to-end smoke test:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python -m cutamp.scripts.run_cutamp \
    --env blocks_t1 --disable_visualizer -n 16 --num_opt_steps 50 --motion_plan
```

Pass criteria: ≥1 satisfying solution, all per-operator trajectories planned,
no `Motion plan failed` warnings tied to shape mismatches.

## Files touched

**New**:
- `cutamp/com_polygon_cost.py` — cost cfg + class.

**Modified (cutamp)**:
- `cutamp/robots/assets/t1_description/t1_planar_base.yml` — add
  `mobile_base_link` to `tool_frames`.
- `cutamp/_curobo_internals.py` — `_ensure_extra_cost_dispatch`,
  `add_extra_cost`.
- `cutamp/robots/t1.py` — `_trajopt_transition_dict_with_compute_com` and the
  `enable_com_polygon` gating in `get_t1_motion_planner`.

**Modified (bundled cuRobo, annotated as `cuTAMP local addition`)**:
- `curobo/_src/transition/robot_state_transition_cfg.py` — `compute_com` field.
- `curobo/_src/transition/robot_state_transition.py:50` — pass `compute_com`
  to `Kinematics(...)`.
