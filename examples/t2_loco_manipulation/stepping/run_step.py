"""Real torque-controlled prescribed step: dynamic contact QP, no state writes after reset."""
import argparse,json,time,os
from pathlib import Path
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from contact_wbc import ContactWholeBodyQP
class StepReference:
 def __init__(self,m,source,maps):
  source=dict(source)  # Materialize NPZ arrays once, outside the control loop.
  self.z=source;self.ts=source['time_s'];self.duration=float(self.ts[-1]);self.q=CubicSpline(self.ts,source['qpos']);self.root=RotationSpline(self.ts,Rotation.from_quat(source['qpos'][:,[4,5,6,3]]));self.qids=np.array([x[1] for x in maps['active']]);self.feet=CubicSpline(self.ts,source['feet'][...,:3,3]);self.caps=CubicSpline(self.ts,source['normal_caps']);self.bodyq=CubicSpline(self.ts,source['qpos'][:,self.qids]);d=mujoco.MjData(m);hp=[];hr=[]
  for q in source['qpos']:
   d.qpos[:]=q;mujoco.mj_kinematics(m,d);hp.append([d.xpos[m.body(s+'_gripper/base').id].copy() for s in ('left','right')]);hr.append([d.xmat[m.body(s+'_gripper/base').id].reshape(3,3).copy() for s in ('left','right')])
  self.hp=CubicSpline(self.ts,np.array(hp));self.hr=[RotationSpline(self.ts,Rotation.from_matrix(np.array(hr)[:,i])) for i in range(2)]
 def packet(self,t):
  t=float(np.clip(t,0,self.duration));q=self.q(t);R=self.root(t).as_matrix();quat=self.root(t).as_quat()[[3,0,1,2]];x=np.zeros(74);x[:3]=q[:3];x[3:6]=Rotation.from_matrix(R).as_euler('ZYX');x[6:37]=self.bodyq(t);x[37:40]=self.q(t,1)[:3];x[43:74]=self.bodyq(t,1);hands=np.zeros((2,14));hands[:,:3]=self.hp(t);vel=np.zeros((2,6));acc=vel.copy();vel[:,:3]=self.hp(t,1);acc[:,:3]=self.hp(t,2)
  for i in range(2):
   rh=self.hr[i](t).as_matrix();hands[i,3:12]=rh.reshape(9);vel[i,3:]=rh@self.hr[i](t,1);acc[i,3:]=rh@self.hr[i](t,2)
  omega=R@self.root(t,1);alpha=R@self.root(t,2)
  return x,self.bodyq(t,2),hands,vel,acc,R,quat,omega,alpha,self.q(t,2)[:3]
 def feet_packet(self,t):
  t=float(np.clip(t,0,self.duration));i=min(np.searchsorted(self.ts,t,side='right')-1,len(self.ts)-1);poses=self.z['feet'][i].copy();poses[:,:3,3]=self.feet(t);v=np.zeros((2,6));a=v.copy();v[:,:3]=self.feet(t,1);a[:,:3]=self.feet(t,2);return poses,v,a,self.z['contacts'][i],np.clip(self.caps(t),0,1000)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--reference',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);os.chdir(out);m=(mujoco.MjModel.from_binary_path(a.model) if a.model.endswith('.mjb') else mujoco.MjModel.from_xml_path(a.model));m.opt.timestep=.0005;d=mujoco.MjData(m);z=np.load(a.reference);d.qpos[:]=m.qpos0
 if 'payload_rows' in z and np.any(z['payload_rows'][:,0]>0):d.qpos[:]=z['initial_qpos']
 else:
  d.qpos[:7]=z['initial_qpos'][:7]
  for row in json.loads(Path(a.audit).read_text())['mapping']:
   ix=int(m.joint(row['joint']).qposadr[0]);d.qpos[ix]=z['initial_qpos'][ix]
 mujoco.mj_forward(m,d);audit=json.loads(Path(a.audit).read_text());names=[x['joint'] for x in audit['mapping']];active=[(m.joint(n).id,int(m.joint(n).qposadr[0]),int(m.joint(n).dofadr[0])) for n in names];acts=[m.actuator(n).id for n in names];maps=dict(active=active,actuators=acts);ref=StepReference(m,z,maps);ctrl=ContactWholeBodyQP(m,d,maps,ref)
 if 'payload_rows' in z:ctrl.payload_rows=z['payload_rows']
 ctrl.friction_mu=.7;ctrl.friction_facets=8;ctrl.foot_kp=400.;ctrl.foot_kd=40.;root=int(m.jnt_bodyid[0]);start=time.monotonic();rows=[];fault=None
 for k in range(round(ref.duration/.002)):
  t=k*.002
  try:tau=ctrl.step(d,t,state_is_current=True)
  except Exception as exc:fault=str(exc);break
  d.ctrl[acts]=tau
  for i,side in enumerate(('left','right')):d.ctrl[m.actuator(side+'_gripper/fingers_actuator').id]=255. if hasattr(ctrl,'payload_rows') and ctrl.payload_rows[i,0]>0 else 0.
  for _ in range(4):mujoco.mj_step(m,d)
  mujoco.mj_forward(m,d);poses=np.array([d.xpos[b]+d.xmat[b].reshape(3,3)@ctrl.sole for b in ctrl.feet]);desired=ref.feet_packet(t)[0][:,:3,3];forces=np.zeros((2,3));footids=set(ctrl.feet)
  for c in range(d.ncon):
   con=d.contact[c];bs=[int(m.geom_bodyid[g]) for g in con.geom];f=np.zeros(6);mujoco.mj_contactForce(m,d,c,f);world=con.frame.reshape(3,3).T@f[:3]
   for i,b in enumerate(ctrl.feet):
    if b in bs:forces[i]+=world if bs[1]==b else -world
  rows.append((d.qpos.copy(),d.qvel.copy(),poses,desired,ctrl.last_wrenches.copy(),tau.copy(),forces,d.subtree_com[root].copy(),ref.feet_packet(t)[3].copy()))
  if not np.isfinite(d.qpos).all() or d.qpos[2]<z['initial_qpos'][2]-.15:fault='fall';break
  if k%500==0:print(json.dumps(dict(t=t,phase=str(z['phases'][min(np.searchsorted(z['time_s'],t),len(z['time_s'])-1)]),sole_error_m=np.linalg.norm(poses-desired,axis=1).tolist(),forces=forces[:,2].tolist())),flush=True)
 if rows:
  keys=['qpos','qvel','soles','desired_soles','predicted_wrenches','torques','measured_forces','com','contacts'];trace={key:np.array([row[i] for row in rows]) for i,key in enumerate(keys)};trace['time_s']=np.arange(len(rows))*.002;np.savez_compressed(out/'trace.npz',**trace);errors=np.linalg.norm(trace['soles']-trace['desired_soles'],axis=-1);hold=trace['time_s']>=ref.duration-1.;holdmax=errors[hold].max() if hold.any() else None
 else:holdmax=None
 report=dict(fault=fault,simulated_s=len(rows)*.002,reference_s=ref.duration,wall_s=time.monotonic()-start,hold_max_sole_error_m=None if holdmax is None else float(holdmax),success=bool(fault is None and holdmax is not None and holdmax<.01));(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
