"""WB-Manifold-2: projected whole-body trajectory optimization.

This is a cuTAMP optimizer, not a successful cuRobo TrajOpt result. Every
iterate minimizes joint acceleration subject to balanced GN projection.
Independent final collision, sole, COM, velocity, effort, ZMP, line and
exact CAD checks remain mandatory. Convergence is never assumed.
"""
from dataclasses import dataclass
import math,time
import torch
from curobo.types import JointState

@dataclass
class ManifoldTrajectory:
    plan: JointState
    geometric_verified: bool
    diagnostics: dict
    trajectory_source: str = 'wb_manifold_v2_projected_optimizer'
    sample_dt: float | None = None
    def get_interpolated_plan(self):
        return self.plan

def _free_hand_motion(world,phase):
    return bool(getattr(world,"manifold_free_hand_motion",False)) and phase in ("pre_reach","approach")

def _geometry(world,planner,q,targets,arm,phase):
    from cutamp.com_polygon_cost import _fk_state,compute_com_polygon_mask,SOLE_POS_TOL,SOLE_ROT_TOL
    from cutamp.gn_refresh import soles_on_stance
    from cutamp.motion_solver import _hand_poses
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    state=_fk_state(world,q)
    soles=soles_on_stance(world,state,SOLE_POS_TOL,SOLE_ROT_TOL)
    com=compute_com_polygon_mask(world,q)
    # Use the planner model/scene, including its real attachment spheres and
    # explicitly enabled/disabled contact obstacles, rather than world-only FK.
    js=JointState.from_position(q.unsqueeze(0),joint_names=list(world.model.joint_names))
    ks=planner.kinematics.compute_kinematics(js)
    spheres=ks.robot_spheres.reshape(1,len(q),-1,4)
    cfg=planner.kinematics.device_cfg
    buffer=CollisionBuffer.from_shape(spheres.shape,cfg)
    scene=planner.scene_collision_checker.get_sphere_distance_raw(spheres,buffer,cfg.to_device([1.]),cfg.to_device([0.]),return_loss=True)
    # Attachment slots change from negative-radius disabled spheres to live
    # payload spheres between phases. Never cache a pre-pick active-pair mask.
    selfcfg=planner.kinematics.get_self_collision_config()
    pad=selfcfg.sphere_padding
    radius=planner.kinematics.robot_spheres[:,3].to(pad.device)+pad
    pairs=selfcfg.collision_pairs.long()
    pairs=pairs[(radius[pairs]>=0).all(-1)]
    i,j=pairs[:,0].contiguous(),pairs[:,1].contiguous()
    flat=spheres.reshape(len(q),-1,4);rad=flat[...,3]+pad
    selfpen=torch.relu(rad[:,i]+rad[:,j]-(flat[:,i,:3]-flat[:,j,:3]).norm(dim=-1)).sum(-1)
    actual=_hand_poses(world,q,arm)
    pos=(actual[:,:3,3]-targets[:,:3,3]).norm(dim=-1)
    rot=(actual[:,:3,:3]-targets[:,:3,:3]).square().sum((-1,-2)).sqrt()/math.sqrt(2)
    path_pos=pos.clone();path_rot=rot.clone()
    if _free_hand_motion(world,phase):pos=pos[[0,-1]];rot=rot[[0,-1]]
    lower,upper=world.kinematics.get_joint_limits().position
    bound_excess=torch.maximum(lower-q,q-upper).clamp_min(0)
    bounds=(bound_excess<=0).all()
    worst=int(bound_excess.argmax())
    bound_report=dict(joint_bound_excess_max=float(bound_excess.max()),joint_bound_excess_joint=list(world.model.joint_names)[worst%q.shape[-1]])
    comfort_ok=True;comfort_report={}
    if hasattr(world,'trajectory_waist_comfort'):
        from cutamp.anatomical_comfort import angles,violations
        angle=angles(world,q);violation=violations(world,q)
        comfort_ok=bool((violation<=1e-6).all())
        comfort_report=dict(anatomical_comfort_valid=comfort_ok,anatomical_angles_min_deg=(angle.min(0).values*180/math.pi).tolist(),anatomical_angles_max_deg=(angle.max(0).values*180/math.pi).tolist(),anatomical_max_violation_deg=float(violation.max()*180/math.pi))
    support_ok=True
    if hasattr(world,'payload_support_guard'):
        support=world.payload_support_guard(q,arm,phase)
        if support.shape != q.shape[:1] or support.dtype != torch.bool:
            raise ValueError('payload_support_guard must return boolean [frames]')
        support_ok=bool(support.all())
        comfort_report['payload_support_guard_valid']=support_ok
    ok=bool(comfort_ok and support_ok and soles.all() & com.all() & (scene<=0).all() & (selfpen<=0).all() & bounds & (pos<=.005).all() & (rot<=.02).all())
    return ok,dict(sole_valid=int(soles.sum()),com_valid=int(com.sum()),frames=len(q),scene_penetration_max=float(scene.max()),self_penetration_max=float(selfpen.max()),hand_error_max_m=float(pos.max()),hand_rotation_chord_max=float(rot.max()),joint_bounds_valid=bool(bounds),hand_constraint_mode="endpoints_only" if _free_hand_motion(world,phase) else "full_path",unconstrained_hand_path_deviation_m=float(path_pos.max()),**bound_report,**comfort_report)

def _cubic_extrema(q,dt):
    """Exact scalar extrema of each joint's clamped C2 cubic, including ends."""
    import numpy as np
    from scipy.interpolate import CubicSpline
    y=q.detach().double().cpu().numpy()
    spline=CubicSpline(np.arange(len(y))*dt,y,axis=0,bc_type='clamped')
    c=spline.c;A=3*c[0];B=2*c[1];C=c[2];D=B*B-4*A*C
    lower=y.min(0);upper=y.max(0)
    with np.errstate(divide='ignore',invalid='ignore'):
        Q=-.5*(B+np.copysign(np.sqrt(np.maximum(D,0)),B))
        # Stable quadratic roots avoid cancellation for nearly quadratic splines.
        roots=[Q/A,C/Q,-C/B]
    for i,t in enumerate(roots):
        valid=(t>0)&(t<dt)&np.isfinite(t)
        valid &= ((abs(A)>1e-20)&(D>=0)) if i<2 else (abs(A)<=1e-20)&(abs(B)>1e-20)
        with np.errstate(invalid='ignore',over='ignore'):
            value=((c[0]*t+c[1])*t+c[2])*t+c[3]
        lower=np.minimum(lower,np.where(valid,value,np.inf).min(0))
        upper=np.maximum(upper,np.where(valid,value,-np.inf).max(0))
    return torch.as_tensor(lower,device=q.device),torch.as_tensor(upper,device=q.device)

def _smooth_cost(q):
    return float((q[2:]-2*q[1:-1]+q[:-2]).square().mean())

def plan_manifold(world,planner,startJS,targetJS,arm,phase):
    from cutamp.gn_refresh import gn_project_balanced
    from cutamp.motion_solver import _hand_poses
    started=time.monotonic();names=list(world.model.joint_names)
    a=startJS.reorder(names).position.reshape(-1,len(names))[0].clone()
    b=targetJS.reorder(names).position.reshape(-1,len(names))[0].clone()
    dt=float(planner.trajopt_solver.config.interpolation_dt)
    duration=float(getattr(world,'manifold_segment_duration',4.))
    n=max(65,math.ceil(duration/dt)+1)
    u=torch.linspace(0,1,n,device=a.device,dtype=a.dtype).reshape(-1,1)
    s=u**3*(10+u*(-15+6*u));q=a+(b-a)*s
    targets=_hand_poses(world,q,arm).clone()
    hand_mask=None
    if _free_hand_motion(world,phase):
        index=torch.arange(n,device=q.device);hand_mask=(index==0)|(index==n-1)
    straight_retract = phase=='retract' and bool(getattr(world,'carry_retract_straight',False))
    cartesian_approach = phase=='approach' and bool(getattr(world,'manifold_cartesian_approach',False))
    if phase in ('insert','lift','back_out') or straight_retract or cartesian_approach:
        targets[:,:3,3]=targets[0,:3,3]*(1-s)+targets[-1,:3,3]*s
        if straight_retract:targets[:,:3,:3]=targets[0,:3,:3].clone()
        if cartesian_approach:
            import roma
            targets[:,:3,:3]=roma.rotmat_slerp(targets[0,:3,:3].clone(),targets[-1,:3,:3].clone(),s.reshape(-1))
        # Endpoint orientations are kept exact; linear task segments have
        # matching orientations by construction in native approach_confs.
    q=gn_project_balanced(world,q,targets,arm,iters=30,collision=True,hand_constraint_mask=hand_mask)
    q[[0,-1]]=torch.stack((a,b));history=[];best=None;bestcost=float('inf');last_report={};bestreport=None;bestkey=None
    lowtask=hasattr(world,'trajectory_waist_comfort')
    max_updates=int(getattr(world,'manifold_max_updates',64 if lowtask else 6))
    velocity_limits=world.segment_torques.velocity.to(q.device)
    for iteration in range(max_updates+1):
        ok,report=_geometry(world,planner,q,targets,arm,phase)
        cost=_smooth_cost(q)
        velocity=torch.gradient(q,spacing=(dt,),dim=(0,))[0]
        ratio=float((velocity.abs()/velocity_limits).max())
        jump=(q[1:]-q[:-1]).abs();flatindex=int(jump.argmax())
        history.append(dict(iteration=iteration,smoothness=cost,geometry_valid=ok,
                            max_adjacent_joint_step=float(jump.max()),
                            max_jump_joint=names[flatindex%len(names)],
                            max_native_velocity_ratio=ratio))
        last_report=report
        key=(ratio>2.,cost if ratio<=2. else ratio) if lowtask else (False,cost)
        if ok and (bestkey is None or key<bestkey):
            best=q.clone();bestcost=cost;bestreport=report.copy();bestkey=key
        if iteration==max_updates:break
        if lowtask and iteration>=6 and ok and ratio<=2.0:break
        # Gradient of squared second differences. Project every optimization
        # update, not a final repair of an already accepted trajectory.
        second=q[2:]-2*q[1:-1]+q[:-2]
        grad=torch.zeros_like(q);grad[:-2]+=second;grad[1:-1]-=2*second;grad[2:]+=second
        proposal=q-.08*grad.clamp(-.10,.10)
        q=gn_project_balanced(world,proposal,targets,arm,iters=8,collision=True,hand_constraint_mask=hand_mask)
        q[[0,-1]]=torch.stack((a,b))
    if best is None:
        print('WB_MANIFOLD_DIAGNOSTIC',dict(version='WB-Manifold-2',phase=phase,failed=True,geometry=last_report,history=history,wall_s=time.monotonic()-started),flush=True)
        raise RuntimeError('WB-Manifold-2 geometry did not converge: '+str(last_report))
    q=best
    if straight_retract:
        from cutamp.motion_solver import check_line
        palm=_hand_poses(world,q,arm)[:,:3,3]
        check_line(palm,palm[0],palm[-1])
    lower,upper=world.joint_limits.to(q.device,q.dtype)
    inset=torch.zeros_like(lower);bound_refinement=[]
    for trial in range(4):
        lo,hi=_cubic_extrema(q,dt)
        excess=torch.maximum(lower.double()-lo,hi-upper.double()).clamp_min(0)
        worst=int(excess.argmax())
        bound_refinement.append(dict(trial=trial,exact_cubic_bound_excess=float(excess.max()),joint=names[worst]))
        if not bool((excess>0).any()):break
        if trial==3:raise RuntimeError('WB-Manifold-2 exact cubic bounds failed: '+str(bound_refinement))
        new_inset=torch.where(excess>0,4*excess.to(q.dtype)+1e-4,torch.zeros_like(inset))
        inset=inset+new_inset
        if bool(((lower+inset)>=(upper-inset)).any()):
            raise RuntimeError('WB-Manifold-2 spline inset exhausted joint range')
        q=gn_project_balanced(world,q,targets,arm,iters=30,collision=True,joint_limit_inset=inset,hand_constraint_mask=hand_mask)
        q[[0,-1]]=torch.stack((a,b))
    v=torch.gradient(q,spacing=(dt,),dim=(0,))[0];acc=torch.gradient(v,spacing=(dt,),dim=(0,))[0]
    # Native operations start/end at rest; exact endpoint positions retained.
    v[[0,-1]]=0;acc[[0,-1]]=0
    plan=JointState(position=q,velocity=v,acceleration=acc,jerk=torch.gradient(acc,spacing=(dt,),dim=(0,))[0],joint_names=names)
    # Fit a clamped C2 curve as part of trajectory generation. Evaluate its
    # actual midpoint positions and analytic derivatives, not finite-difference
    # derivatives of independently projected knots. Preserve physical timing.
    from cutamp.motion_solver import _retime,_zmp_margins,_model_frames
    import roma
    dense=_retime(plan,2.,dt)
    dense.velocity*=2.;dense.acceleration*=4.;dense.jerk*=8.
    dense.position[[0,-1]]=q[[0,-1]]
    dense_targets=targets.new_zeros(2*n-1,4,4)
    dense_targets[::2]=targets
    dense_targets[1::2]=targets[:-1]
    dense_targets[1::2,:3,3]=(targets[:-1,:3,3]+targets[1:,:3,3])/2
    dense_targets[1::2,:3,:3]=roma.rotmat_slerp(targets[:-1,:3,:3],targets[1:,:3,:3],q.new_tensor([.5])).squeeze(0)
    valid,dense_report=_geometry(world,planner,dense.position,dense_targets,arm,phase)
    if not valid:raise RuntimeError('WB-Manifold-2 C2 midpoint geometry failed: '+str(dense_report))
    scales=q.new_tensor([1.,2.,4.,8.,1000.])
    margins=_zmp_margins(world,_model_frames(world,dense),scales)
    acceleration=dense.acceleration.abs();ai=int(acceleration.argmax())
    diagnostics=dict(version='WB-Manifold-2',phase=phase,history=history,geometry=dense_report,projected_node_geometry=bestreport,
                     c2_dense_geometry=dense_report,cubic_bound_refinement=bound_refinement,sample_dt_s=dt/2,
                     zmp_margin_by_timescale_m={str(float(k)):float(v) for k,v in zip(scales,margins)},
                     max_abs_acceleration=float(acceleration.max()),max_acceleration_joint=names[ai%len(names)],
                     duration_s=(n-1)*dt,wall_s=time.monotonic()-started,requires_balance_and_exact_cad=True)
    print('WB_MANIFOLD_DIAGNOSTIC',diagnostics,flush=True)
    return ManifoldTrajectory(dense,True,diagnostics,sample_dt=dt/2)
