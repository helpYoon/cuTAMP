"""Independent contact replay audit of ONE continuous saved physical execution."""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial import ConvexHull
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--trace',required=True);p.add_argument('--audit',required=True);p.add_argument('--mu',type=float,default=.7);a=p.parse_args();path=Path(a.trace);z=dict(np.load(path));m=(mujoco.MjModel.from_binary_path(a.model) if a.model.endswith(".mjb") else mujoco.MjModel.from_xml_path(a.model));m.opt.timestep=.0005
if m.na:raise ValueError('activation state required for exact contact replay')
d=mujoco.MjData(m);names=[x['joint'] for x in json.loads(Path(a.audit).read_text())['mapping']];acts=[m.actuator(n).id for n in names];gacts=[m.actuator(s+'_gripper/fingers_actuator').id for s in ('left','right')];feet=[m.body(s+'_ankle_roll_link').id for s in ('left','right')];sole=np.array([.042485,0,-.032506]);n=len(z['qpos']);force=np.empty((n,2,3));worldforce=np.empty_like(force);mom=np.empty_like(force);copmargin=np.empty((n,2));friction=np.empty((n,2));slip=np.empty((n,2));nonground=np.zeros(n,int);global_cop_margin=np.empty(n);started=time.monotonic()
for k in range(n):
 d.qpos[:]=z['qpos'][k];d.qvel[:]=z['qvel'][k];d.ctrl[:]=0;d.ctrl[acts]=z['torques'][k];d.ctrl[gacts]=z['gripper'][k];mujoco.mj_forward(m,d);P=np.array([d.xpos[b]+d.xmat[b].reshape(3,3)@sole for b in feet]);R=np.array([d.xmat[b].reshape(3,3) for b in feet]);W=np.zeros((2,6))
 for c in range(d.ncon):
  con=d.contact[c];bodies=[int(m.geom_bodyid[g]) for g in con.geom];isground=any(b==0 or m.body(b).name=='floor' for b in bodies)
  if not any(b in feet for b in bodies):continue
  if not isground:nonground[k]+=1;continue
  f=np.zeros(6);mujoco.mj_contactForce(m,d,c,f);frame=con.frame.reshape(3,3).T;F=frame@f[:3];M=frame@f[3:]
  for i,b in enumerate(feet):
   if b in bodies:
    sign=1 if bodies[1]==b else -1;W[i,:3]+=sign*F;W[i,3:]+=sign*(M+np.cross(con.pos-P[i],F))
 worldforce[k]=W[:,:3]
 F=W[:,:3].sum(0);M=(W[:,3:]+np.cross(P,W[:,:3])).sum(0);global_cop=np.array([-M[1],M[0]])/max(F[2],1e-9);loaded_ids=np.flatnonzero(W[:,2]>1.)
 corners=np.array([[-.10791,-.040886,0],[-.10791,.040886,0],[.10791,.040886,0],[.10791,-.040886,0]])
 vertices=np.concatenate([(corners@R[i].T+P[i])[:,:2] for i in loaded_ids],0) if len(loaded_ids) else np.zeros((0,2))
 if len(vertices)>=3:
  hull=ConvexHull(vertices);global_cop_margin[k]=float((-hull.equations[:,:2]@global_cop-hull.equations[:,2]).min())
 else:global_cop_margin[k]=-np.inf
 for i in range(2):
  localF=R[i].T@W[i,:3];localM=R[i].T@W[i,3:];force[k,i]=localF;mom[k,i]=localM;cop=np.array([-localM[1],localM[0]])/max(localF[2],1e-9);copmargin[k,i]=np.min(np.array([.10791,.040886])-abs(cop)) if localF[2]>5 else np.nan;friction[k,i]=a.mu*max(localF[2],0)-np.linalg.norm(localF[:2]);slip[k,i]=np.linalg.norm(P[i]-z['desired_soles'][k,i])
t=z['time'] if 'time' in z else z['time_s'];contacts=z['contacts'];single=contacts.sum(1)==1;ids=np.flatnonzero(single);groups=np.split(ids,np.where(np.diff(ids)>1)[0]+1) if len(ids) else [];steps=[]
for group in groups:
 if not len(group):continue
 lo,hi=group[0],group[-1];support=int(np.flatnonzero(contacts[lo])[0]);swing=1-support;mask=(t>=t[lo]+.1)&(t<=t[hi]-.1);core=np.flatnonzero(mask)
 if not len(core):continue
 steps.append(dict(start_s=float(t[lo]),end_s=float(t[hi]),swing_foot='left' if swing==0 else 'right',interior_swing_normal_max_N=float(force[core,swing,2].max()),support_normal_min_N=float(force[core,support,2].min()),support_measured_cop_min_margin_m=float(np.nanmin(copmargin[core,support])),support_friction_min_margin_N=float(friction[core,support].min()),max_support_sole_error_m=float(slip[core,support].max()),swing_clearance_m=float(z['soles'][core,swing,2].max()-z['soles'][max(0,lo-1),swing,2])))
loaded=force[...,2]>20;finitecop=copmargin[loaded];recorded=z['measured_foot_forces'] if 'measured_foot_forces' in z else None
replay_delta=None if recorded is None else float(np.linalg.norm(worldforce-recorded,axis=-1).max())
report=dict(recorded_vs_replayed_force_max_delta_N=replay_delta,frames=n,duration_s=float(t[-1]),actual_contact_replay=True,restored_actual_motor_and_gripper_controls=True,mu=a.mu,loaded_threshold_N=20,step_interiors=steps,whole_episode_combined_support_cop_min_margin_m=float(global_cop_margin.min()),whole_episode_loaded_cop_min_margin_m=float(np.nanmin(finitecop)),whole_episode_loaded_friction_min_margin_N=float(friction[loaded].min()),max_active_sole_error_m=float(slip[contacts].max()),nonground_foot_contact_frames=int((nonground>0).sum()),wall_s=time.monotonic()-started,scope='measured contacts in actual sole frames; excludes0.1s phase boundaries for per-step interior stats; no rerun or state correction')
np.savez_compressed(path.parent/'contact-audit.npz',time_s=t,force_world=worldforce,force_local=force,moment_local=mom,cop_margin_m=copmargin,friction_margin_N=friction,sole_error_m=slip,nonground_contacts=nonground,combined_support_cop_margin_m=global_cop_margin);(path.parent/'contact-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
