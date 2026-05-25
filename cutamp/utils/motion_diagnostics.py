# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Motion-plan failure diagnostics.

When ``solve_curobo`` raises on a failed plan_cspace / plan_pose attempt, the
outer retry usually finds a different particle that succeeds — so by default
we don't dump per-step diagnostics. Set ``TAMPConfiguration.debug_motion_failures
= True`` to enable these helpers, which walk the failed trajectory and report
the first violated constraint per category (self-collision, world-collision,
joint limit, dynamics) with timestep + sphere indices + depth.

These functions are slow: they do horizon × N forward-kinematics calls in a
Python loop. Diagnostic only — not for the production hot path.
"""

import traceback

import torch

from curobo.types import JointState


def walk_trajectory_collisions(planner, result, log) -> None:
    """Walk each timestep of the failed trajectory and report collisions.

    Goes around the metrics_rollout API entirely. For each timestep:
      1. Compute robot spheres via planner.kinematics.compute_kinematics.
      2. Run a vectorized self-collision check using the planner's
         precomputed `collision_pairs` (which already has the YAML
         self_collision_ignore list applied).
      3. Run a world-collision check by querying the scene_collision_checker
         for sphere distance.

    Reports the worst violations per category with timestep + sphere
    indices + depth in metres. Sphere indices map back to links via the
    YAML ordering (printed as a hint).
    """
    js_sol = getattr(result, "js_solution", None)
    if js_sol is None:
        log.error("  walker: no js_solution to walk")
        return

    # Squeeze trajectory to 2D [horizon, dof]
    pos = js_sol.position
    while pos.dim() > 2:
        pos = pos.squeeze(0) if pos.shape[0] == 1 else pos[0]
    horizon = pos.shape[0]

    # Convert full-DOF (31) to active-DOF (21) for compute_kinematics
    active_dof = len(planner.kinematics.joint_names)
    if pos.shape[-1] != active_dof:
        flat_js = JointState.from_position(pos, joint_names=getattr(js_sol, "joint_names", None))
        active = planner.trajopt_solver.get_active_js(flat_js)
        pos = active.position

    log.error(f"  walker: walking {horizon} timesteps × {pos.shape[-1]} DOF")

    # Pull collision_pairs + sphere_padding from the trajopt's self_collision cost
    coll_pairs = None
    sphere_padding = None
    try:
        for r in getattr(planner.trajopt_solver, "optimizer_rollouts", []):
            cm = getattr(r, "cost_manager", None)
            sc_cost = cm.costs.get("self_collision") if cm and hasattr(cm, "costs") else None
            cfg = getattr(sc_cost, "config", None) if sc_cost else None
            sckc = getattr(cfg, "self_collision_kin_config", None) if cfg else None
            if sckc is not None:
                coll_pairs = getattr(sckc, "collision_pairs", None)
                sphere_padding = getattr(sckc, "sphere_padding", None)
                if coll_pairs is not None:
                    break
    except Exception as e:
        log.error(f"  walker: pulling collision_pairs failed: {e}")

    # Walk the trajectory
    worst_self = []   # list of (depth, t, i, j, sphere_i, sphere_j)
    for t in range(horizon):
        q_t = pos[t:t+1]   # [1, dof]
        js_t = JointState.from_position(q_t)
        try:
            kin = planner.kinematics.compute_kinematics(js_t)
        except Exception as e:
            log.error(f"  walker: compute_kinematics(t={t}) failed: {e}")
            continue
        spheres = kin.robot_spheres   # [batch=1, num_spheres, 4]
        if spheres is None:
            continue
        sph = spheres.view(-1, 4)   # [num_spheres, 4]

        # Self-collision check
        if coll_pairs is not None and coll_pairs.numel() > 0:
            i_idx = coll_pairs[:, 0].long()
            j_idx = coll_pairs[:, 1].long()
            ci = sph[i_idx, :3]
            cj = sph[j_idx, :3]
            ri = sph[i_idx, 3]
            rj = sph[j_idx, 3]
            pi = sphere_padding[i_idx] if sphere_padding is not None else 0.0
            pj = sphere_padding[j_idx] if sphere_padding is not None else 0.0
            d = (ci - cj).norm(dim=-1)
            thresh = ri + rj + pi + pj
            penetration = thresh - d   # positive = collision
            worst = penetration.max()
            if float(worst.item()) > 1e-6:
                top_idx = int(penetration.argmax().item())
                worst_self.append((
                    float(worst.item()), t,
                    int(i_idx[top_idx].item()), int(j_idx[top_idx].item()),
                    sph[i_idx[top_idx]].tolist(),
                    sph[j_idx[top_idx]].tolist(),
                ))

    # Report top self-collision violations
    if worst_self:
        log.error(f"  walker: SELF-COLLISION at {len(worst_self)}/{horizon} timesteps")
        worst_self.sort(reverse=True)
        for depth, t, i, j, si, sj in worst_self[:5]:
            log.error(
                f"    t={t:3d}  spheres ({i},{j})  depth={depth*1000:.2f}mm  "
                f"sph_i=({si[0]:+.3f},{si[1]:+.3f},{si[2]:+.3f}, r={si[3]:.3f})  "
                f"sph_j=({sj[0]:+.3f},{sj[1]:+.3f},{sj[2]:+.3f}, r={sj[3]:.3f})"
            )
    else:
        log.error("  walker: no self-collision detected anywhere in trajectory")

    # World-collision: try via scene_collision_checker.get_sphere_distance
    sc = getattr(planner, "scene_collision_checker", None)
    if sc is None:
        log.error("  walker: no scene_collision_checker — skipping world check")
        return

    worst_world = []
    for t in range(horizon):
        q_t = pos[t:t+1]
        js_t = JointState.from_position(q_t)
        try:
            kin = planner.kinematics.compute_kinematics(js_t)
            spheres = kin.robot_spheres   # [1, num_spheres, 4]
        except Exception:
            continue
        if spheres is None:
            continue
        # get_sphere_distance_raw expects [batch, horizon, num_spheres, 4]
        try:
            qs = spheres.unsqueeze(1)   # [1, 1, num_spheres, 4]
            from curobo._src.geom.collision.buffer_collision import CollisionBuffer
            from curobo._src.types.device_cfg import DeviceCfg
            dev_cfg = DeviceCfg(device=qs.device, dtype=qs.dtype)
            buf = CollisionBuffer.from_shape(qs.shape, dev_cfg)
            weight = torch.ones(1, device=qs.device, dtype=qs.dtype)
            act_dist = torch.zeros(1, device=qs.device, dtype=qs.dtype)
            d = sc.get_sphere_distance_raw(qs, buf, weight, act_dist)
            # d is a cost (positive = penetration). Find max.
            if d is not None and d.numel() > 0:
                m = float(d.max().item())
                if m > 1e-4:
                    worst_world.append((m, t, int(d.view(-1).argmax().item())))
        except Exception as e:
            if t == 0:
                log.error(f"  walker: world distance query failed at t=0: {type(e).__name__}: {e}")
            break

    if worst_world:
        log.error(f"  walker: WORLD-COLLISION at {len(worst_world)}/{horizon} timesteps")
        worst_world.sort(reverse=True)
        for cost, t, sph_idx in worst_world[:5]:
            log.error(f"    t={t:3d}  sphere_idx={sph_idx}  cost={cost:.4f}")
    else:
        log.error("  walker: no world-collision detected (or query failed)")

    # Manual world-collision against scene cuboids (sphere-vs-AABB, ignoring
    # cuboid orientation — works for axis-aligned obstacles which is all we
    # have in blocks_t1). Reports gripper-dipping-through-table or
    # block-vs-block intrusions that the cuRobo API didn't catch.
    try:
        sm = sc.scene_model if sc is not None else None
        if isinstance(sm, list):
            sm = sm[0]
        cuboids = getattr(sm, "cuboid", None) if sm is not None else None
        if cuboids:
            ws_violations = []   # (t, sphere_idx, obstacle_name, depth)
            for t in range(horizon):
                q_t = pos[t:t+1]
                kin = planner.kinematics.compute_kinematics(JointState.from_position(q_t))
                sph = kin.robot_spheres.view(-1, 4)
                centers = sph[:, :3]
                radii = sph[:, 3]
                for cube in cuboids:
                    # cube.pose is [x,y,z, qw, qx, qy, qz]; cube.dims is [x,y,z]
                    cp = torch.tensor(cube.pose[:3], device=centers.device, dtype=centers.dtype)
                    cd = torch.tensor(cube.dims, device=centers.device, dtype=centers.dtype) * 0.5
                    # signed distance from sphere center to cube face (axis aligned)
                    delta = (centers - cp).abs() - cd
                    out = torch.clamp(delta, min=0.0).norm(dim=-1)
                    inner = torch.clamp(delta.max(dim=-1).values, max=0.0)
                    sd = out + inner
                    pene = radii - sd   # positive = sphere intersects cube
                    if (pene > 1e-3).any():
                        worst_idx = int(pene.argmax().item())
                        ws_violations.append((t, worst_idx, cube.name, float(pene[worst_idx].item())))
            if ws_violations:
                log.error(f"  walker: MANUAL WORLD-COLLISION at {len(ws_violations)} (timestep, sphere) hits:")
                ws_violations.sort(key=lambda x: -x[3])
                for t, si, cn, depth in ws_violations[:8]:
                    log.error(f"    t={t:3d}  sphere {si} vs {cn}  depth={depth*1000:.1f}mm")
            else:
                log.error("  walker: manual world-collision check: clean")
    except Exception as e:
        log.error(f"  walker: manual world check failed: {type(e).__name__}: {e}")

    # Joint-limit + velocity check
    try:
        kc = planner.kinematics.config.kinematics_config
        jl = getattr(kc, "joint_limits", None)
        if jl is not None and getattr(jl, "position", None) is not None:
            jp = jl.position   # [2, dof] (low, high)
            jv = getattr(jl, "velocity", None)
            jacc = getattr(jl, "acceleration", None)
            low, high = jp[0].to(pos.device), jp[1].to(pos.device)
            # pos is [horizon, dof]
            below = (pos < low).any(dim=-1)
            above = (pos > high).any(dim=-1)
            limit_violations = (below | above)
            n_viol = int(limit_violations.sum().item())
            if n_viol > 0:
                log.error(f"  walker: JOINT-LIMIT violated at {n_viol}/{horizon} timesteps")
                for t in range(horizon):
                    if below[t] or above[t]:
                        bad = ((pos[t] < low) | (pos[t] > high)).nonzero().view(-1).tolist()
                        log.error(f"    t={t:3d}  DOF idx {bad}  q={pos[t][bad].tolist()}  "
                                  f"low={low[bad].tolist()}  high={high[bad].tolist()}")
                        if t > 5: break
            else:
                log.error("  walker: no joint-limit violations")

            # Use the trajopt-computed velocity/accel/jerk if present —
            # otherwise finite-difference. Compare against active-DOF limits.
            def _flat(t):
                if t is None: return None
                while t.dim() > 2:
                    t = t.squeeze(0) if t.shape[0] == 1 else t[0]
                return t

            traj_vel = _flat(getattr(js_sol, "velocity", None))
            traj_acc = _flat(getattr(js_sol, "acceleration", None))
            traj_jrk = _flat(getattr(js_sol, "jerk", None))

            # Convert full→active DOF if present
            def _active_traj(t):
                if t is None or t.shape[-1] == active_dof: return t
                flat_js2 = JointState.from_position(t.reshape(-1, t.shape[-1]),
                                              joint_names=getattr(js_sol, "joint_names", None))
                return planner.trajopt_solver.get_active_js(flat_js2).position.reshape(*t.shape[:-1], active_dof)

            traj_vel = _active_traj(traj_vel)
            traj_acc = _active_traj(traj_acc)
            traj_jrk = _active_traj(traj_jrk)

            for label, traj_vals, lim in [
                ("velocity",     traj_vel, jv),
                ("acceleration", traj_acc, jacc),
                ("jerk",         traj_jrk, getattr(jl, "jerk", None)),
            ]:
                if traj_vals is None or lim is None:
                    log.error(f"  walker: {label}: no data (traj={traj_vals is not None}, lim={lim is not None})")
                    continue
                lim_high = lim[1].to(pos.device)
                vmax_per_dof = traj_vals.abs().max(dim=0).values
                vio = (vmax_per_dof > lim_high).nonzero().view(-1).tolist()
                if vio:
                    log.error(f"  walker: {label.upper()} exceeds limit on DOF idx {vio}")
                    for d in vio[:5]:
                        log.error(f"    DOF {d}: max={float(vmax_per_dof[d].item()):.3f}  "
                                  f"limit={float(lim_high[d].item()):.3f}  "
                                  f"ratio={float(vmax_per_dof[d].item() / lim_high[d].item()):.2f}x")
                else:
                    log.error(f"  walker: {label} OK (max={float(vmax_per_dof.max().item()):.3f}, "
                              f"limit={float(lim_high.max().item()):.3f})")
    except Exception as e:
        log.error(f"  walker: limit check failed: {type(e).__name__}: {e}")


def log_cspace_failure_diagnostic(planner, start_js, goal_js, result, ground_op, log):
    """Dump everything we can about why plan_cspace returned no successful seed.

    Walks ``result.metrics.costs_and_constraints`` to identify which named
    constraint(s) are violated and at which timesteps along the failed
    trajectory. This is the most direct signal: trajopt success = all
    constraints satisfied at every timestep AND last-step cspace distance
    below tolerance.
    """
    log.error(f"=== Diagnostic for failed plan_cspace ({ground_op.name}) ===")

    # 1. Start vs goal cspace distance
    try:
        d = (goal_js.position - start_js.position).abs().squeeze()
        log.error(f"  cspace |goal - start| per DOF: {[round(x, 3) for x in d.tolist()]}")
        log.error(f"  cspace L2 distance: {float(d.norm().item()):.4f}, max: {float(d.max().item()):.4f}")
    except Exception as e:
        log.error(f"  cspace distance computation failed: {e}")

    if result is None:
        log.error("  result: None (planner returned no result)")
        log.error("=== End diagnostic ===")
        return

    try:
        succ = result.success
        if succ is None:
            log.error("  result.success: None")
        else:
            log.error(f"  result.success.shape={tuple(succ.shape)}, success.tolist()={succ.tolist()}")
    except Exception as e:
        log.error(f"  result.success log failed: {type(e).__name__}: {e}")

    try:
        ce = getattr(result, "cspace_error", None)
        if ce is None:
            log.error("  result.cspace_error: None")
        elif hasattr(ce, "tolist"):
            log.error(f"  result.cspace_error: {ce.tolist()}")
        else:
            log.error(f"  result.cspace_error: {ce}")
    except Exception as e:
        log.error(f"  result.cspace_error log failed: {type(e).__name__}: {e}")

    try:
        log.error(f"  result.position_error: {getattr(result, 'position_error', None)}")
        if getattr(result, "js_solution", None) is not None:
            jp = result.js_solution.position
            log.error(f"  js_solution.position.shape={tuple(jp.shape)}")
            # Trajectory may be in full kinematic DOF (31) vs active cspace (21).
            # Pull through trajopt_solver.get_active_js if shapes mismatch.
            last_pos = jp[..., -1, :]
            if last_pos.shape[-1] != goal_js.position.shape[-1]:
                _full_js = JointState.from_position(
                    last_pos.reshape(-1, last_pos.shape[-1]),
                    joint_names=getattr(result.js_solution, "joint_names", None),
                )
                _active = planner.trajopt_solver.get_active_js(_full_js)
                last_pos = _active.position
            d = (last_pos - goal_js.position).abs()
            log.error(f"  trajectory_end |pos - goal| max={float(d.max().item()):.4f}")
    except Exception as e:
        log.error(f"  trajectory_end check failed: {type(e).__name__}: {e}")

    # 2. Per-constraint, per-timestep violations
    # result.metrics is often None for plan_cspace; recompute by rolling the
    # failed js_solution through the trajopt's metrics_rollout, which evaluates
    # every cost/constraint at every timestep and returns the named breakdown.
    metrics = getattr(result, "metrics", None)
    cnc = getattr(metrics, "costs_and_constraints", None)
    # Trajectory-walker: bypass the metrics_rollout entirely. Walks each
    # timestep of the failed trajectory and checks self-collision via the
    # planner's precomputed collision_pairs (which already has the YAML
    # ignore list baked in), plus a separate world-collision query.
    try:
        walk_trajectory_collisions(planner, result, log)
    except Exception as e:
        log.error(f"  trajectory walker failed: {type(e).__name__}: {e}")
        log.error(f"  traceback: {traceback.format_exc().splitlines()[-3:]}")

    if cnc is None:
        try:
            from curobo._src.state.state_robot import RobotState
            mr = getattr(planner.trajopt_solver, "metrics_rollout", None)
            js_sol = getattr(result, "js_solution", None)
            if mr is not None and js_sol is not None:
                # 1) squeeze seed dim so position is 3D (batch, horizon, dof)
                pos = js_sol.position
                while pos.dim() > 3:
                    if pos.shape[1] == 1:
                        pos = pos.squeeze(1)
                    elif pos.shape[0] == 1:
                        pos = pos.squeeze(0)
                    else:
                        pos = pos[:, 0]
                squeezed_js = JointState.from_position(
                    pos, joint_names=getattr(js_sol, "joint_names", None),
                )
                # 2) trajectory is in full kinematic DOF (31). The rollout
                # is configured for active DOF (21). Convert.
                active_dof = len(planner.kinematics.joint_names)
                if squeezed_js.position.shape[-1] != active_dof:
                    # get_active_js expects 2D, so flatten batch×horizon
                    flat = squeezed_js.position.reshape(-1, squeezed_js.position.shape[-1])
                    flat_js = JointState.from_position(flat, joint_names=getattr(js_sol, "joint_names", None))
                    active = planner.trajopt_solver.get_active_js(flat_js)
                    new_pos = active.position.reshape(*pos.shape[:-1], active_dof)
                    squeezed_js = JointState.from_position(new_pos, joint_names=getattr(active, "joint_names", None))
                state = RobotState(joint_state=squeezed_js)
                metrics = mr.compute_metrics_from_state(state)
                cnc = getattr(metrics, "costs_and_constraints", None)
                if cnc is not None:
                    log.error("  (metrics rolled out via trajopt_solver.metrics_rollout)")
        except Exception as e:
            log.error(f"  metrics_rollout failed: {type(e).__name__}: {e}")
            log.error(f"  traceback: {traceback.format_exc().splitlines()[-3:]}")

    if cnc is None:
        log.error("  metrics.costs_and_constraints not available")
        log.error("=== End diagnostic ===")
        return

    constraints = getattr(cnc, "constraints", None)
    if constraints is None or not getattr(constraints, "names", None):
        log.error("  constraints.names empty; nothing to report")
    else:
        log.error(f"  Per-constraint violation summary (over horizon, summed across the violation axis):")
        for name, vals in zip(constraints.names, constraints.values):
            try:
                if vals.dim() == 3:
                    per_step = vals.sum(dim=-1)  # [batch, horizon]
                else:
                    per_step = vals
                # Take the worst seed (max over batch dim) — we're inspecting WHY no seed succeeded.
                worst_seed = per_step.max(dim=0).values if per_step.dim() > 1 else per_step
                worst_step_idx = int(worst_seed.argmax().item())
                worst_value = float(worst_seed.max().item())
                violating_steps = (worst_seed > 1e-4).sum().item()
                horizon = worst_seed.shape[-1]
                if worst_value > 1e-4:
                    log.error(
                        f"    [VIOLATED] {name:30s}  max={worst_value:.4f} at t={worst_step_idx}/{horizon}, "
                        f"steps_violating={int(violating_steps)}"
                    )
                else:
                    log.error(f"    [ ok ]     {name:30s}  max={worst_value:.4e}")
            except Exception as e:
                log.error(f"    [error]    {name}: {e}")

    # 3. Held-block / sphere bounds at start (sanity check that attach worked)
    try:
        kin_state = planner.kinematics.compute_kinematics(start_js)
        spheres = kin_state.robot_spheres
        if spheres is not None:
            n = spheres.shape[-2]
            log.error(f"  total robot spheres at start: {n}")
    except Exception as e:
        log.error(f"  sphere count failed: {e}")

    log.error("=== End diagnostic ===")
