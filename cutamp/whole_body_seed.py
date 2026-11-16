"""Version WB-Seed-1: balanced task-conditioned initialization for native TrajOpt.

This creates seeds, not accepted motions. Native optimizer feasibility, full-path
balance/sole/line checks and exact CAD remain independent acceptance requirements.
"""
import torch
from curobo.types import JointState

def balanced_seed(world,planner,start,goal,arm,phase,iters=30):
 from cutamp.gn_refresh import gn_project_balanced
 from cutamp.motion_solver import _hand_poses
 solver=planner.trajopt_solver;fitter=getattr(planner,'graph_seed_fitter',None)
 n=fitter.config.rollout_horizon if fitter is not None else solver.action_horizon
 names=list(planner.kinematics.joint_names);a=start.position.reshape(-1,len(start.joint_names))[0][[list(start.joint_names).index(k) for k in names]];b=goal.position.reshape(-1,len(names))[0]
 if names!=list(world.model.joint_names):raise ValueError('WB-Seed-1 requires full native active configuration ordering')
 t=torch.linspace(0.,1.,n,device=a.device,dtype=a.dtype).reshape(-1,1);s=t**3*(10+t*(-15+6*t));q=a+(b-a)*s
 targets=_hand_poses(world,q,arm).clone()
 if phase in ('insert','lift','back_out'):
  targets[:,:3,3]=targets[0,:3,3]*(1-s)+targets[-1,:3,3]*s
  # Insert/drop-out preserve the desired hand orientation. Any start discrepancy
  # remains visible to strict downstream line/pose checks; exact endpoints retained.
  targets[:,:3,:3]=targets[-1,:3,:3].clone()
 q=gn_project_balanced(world,q,targets,arm,iters=iters,collision=True)
 q[[0,-1]]=torch.stack((a,b))
 seeds=int(solver.config.num_seeds);dense=q.unsqueeze(0).expand(seeds,-1,-1).contiguous()
 if fitter is not None:
  current=start.reorder(names);rep=JointState.from_position(current.position.expand(seeds,-1).contiguous(),joint_names=names)
  for key in ('velocity','acceleration','jerk'):
   value=getattr(current,key,None)
   if value is not None:setattr(rep,key,value.expand(seeds,-1).contiguous())
  controls=fitter.fit(dense,rep,b.expand(seeds,-1).contiguous())
 else:controls=dense
 return controls.reshape(1,seeds,solver.action_horizon,len(names))

def solve_balanced_cspace(world,planner,target,current,arm,phase,max_attempts=2):
 """Native TrajOpt refinement of explicit balanced seeds; never fabricate success."""
 seed=balanced_seed(world,planner,current,target,arm,phase)
 dt=seed.new_full((1,int(planner.trajopt_solver.config.num_seeds)),4.0/max(1,planner.trajopt_solver.horizon-1))
 result=planner.trajopt_solver.solve_cspace(target,current,seed_traj=seed,dt=dt,finetune_attempts=max(1,min(max_attempts,4)),finetune_dt_scale=1.0,time_optimal_iters=0)
 print('WB_SEED_DIAGNOSTIC',dict(version='WB-Seed-1',phase=phase,seed_shape=list(seed.shape),requested_duration_s=4.,success=None if result is None else result.success.detach().cpu().tolist(),feasible=None if result is None or getattr(result,'feasible',None) is None else result.feasible.detach().cpu().tolist()),flush=True)
 if result is not None:
  result.trajectory_source='wb_seed_v1_native_trajopt'
  if getattr(result,'debug_info',None) is not None:result.debug_info['trajectory_source']='wb_seed_v1_native_trajopt'
 return result
