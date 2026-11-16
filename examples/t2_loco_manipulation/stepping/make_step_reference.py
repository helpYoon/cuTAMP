"""Prescribed quasi-static whole-body stepping references, native MuJoCo IK."""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear

def smooth(u):
 u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
def make(model,initial,maps,foot=0,dx=.03,dy=0.,clearance=.035,dt=.01,payload_rows=None,terminal_joint_targets=None,swing_knee_bump=.30,support_forward_bias=0.):
 m=model;d=mujoco.MjData(m);d.qpos[:]=initial;mujoco.mj_forward(m,d);dofs=np.array(list(range(6))+[x[2] for x in maps['active']]);qids=np.array([x[1] for x in maps['active']]);feet=[m.body(n).id for n in ('left_ankle_roll_link','right_ankle_roll_link')];sole=np.array([.042485,0.,-.032506]);root=int(m.jnt_bodyid[0]);R0=d.xmat[root].reshape(3,3).copy();home=d.qpos[qids].copy();body_names=[m.joint(x[0]).name for x in maps["active"]];payload_rows=np.zeros((2,4)) if payload_rows is None else np.asarray(payload_rows,dtype=float)
 robotmass=float(m.body_subtreemass[root]);totalmass=robotmass+payload_rows[:,0].sum();hands=[m.body(s+'_gripper/base').id for s in ('left','right')]
 def combined_com_and_jac():
  com=robotmass*d.subtree_com[root].copy();jac=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,jac,root);jac*=robotmass
  for i,hand in enumerate(hands):
   mass=payload_rows[i,0]
   if mass>0:
    point=d.xpos[hand]+d.xmat[hand].reshape(3,3)@(payload_rows[i,1:]/mass);jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,point,hand);com+=mass*point;jac+=mass*jp
  return com/totalmass,jac/totalmass
 initial_com=combined_com_and_jac()[0];P=np.array([d.xpos[b]+d.xmat[b].reshape(3,3)@sole for b in feet]);RR=np.array([d.xmat[b].reshape(3,3).copy() for b in feet]);land=P.copy();land[foot,:2]+=[dx,dy];support=1-foot;supportcom=initial_com.copy();supportcom[:2]=P[support,:2]+RR[support][:2,0]*support_forward_bias
 # settle, shift, unload, swing, touchdown, recenter, hold
 phases=[('settle',1.),('shift',3.),('unload',1.),('swing',2.),('touchdown',1.),('recenter',3.),('hold',2.)];duration=sum(t for _,t in phases);ts=np.arange(0,duration+dt/2,dt);qs=[];cs=[];desired=[];labels=[];caps=[];ik=[]
 boundaries=np.cumsum([0]+[t for _,t in phases]);mass=m.body_subtreemass[root];gravity=9.81*totalmass
 for t in ts:
  idx=min(np.searchsorted(boundaries,t,side='right')-1,len(phases)-1);phase,length=phases[idx];u=(t-boundaries[idx])/length;h=smooth(u);target=P.copy();com=initial_com.copy();contact=np.ones(2,dtype=bool);cap=np.ones(2)*1000
  if phase=='shift':com=initial_com*(1-h)+supportcom*h
  elif phase in ('unload','swing','touchdown'):com=supportcom.copy()
  elif phase in ('recenter','hold'):
   center=supportcom.copy();center[:2]=land[:,:2].mean(0)
   com=supportcom*(1-h)+center*h if phase=='recenter' else center
  if phase=='unload':cap[foot]=max(0.,gravity*(1-h))
  if phase=='swing':
   contact[foot]=False;cap[foot]=0;target[foot]=P[foot]*(1-h)+land[foot]*h;target[foot,2]+=clearance*np.sin(np.pi*u)**2
  elif phase in ('touchdown','recenter','hold'):target=land.copy()
  if phase=='touchdown':cap[foot]=max(0.,gravity*h)
  crouch=smooth((t-1.)/1.)
  knee_initial=[home[body_names.index(side+'_knee_pitch_joint')] for side in ('left','right')]
  knee_increments=np.maximum(.30-np.array(knee_initial),0.)
  com[2]-=.012*crouch*float(knee_increments.max()/.30)
  posture=home.copy()
  body_names=[m.joint(x[0]).name for x in maps['active']]
  for side,increment in zip(('left','right'),knee_increments):
   for joint,offset in [('hip_pitch',-.5*increment),('knee_pitch',increment),('ankle_pitch',-.5*increment)]:
    posture[body_names.index(side+'_'+joint+'_joint')]+=offset*crouch
  if terminal_joint_targets is not None:
   blend=h if phase=='recenter' else 1. if phase=='hold' else 0.
   for name,value in terminal_joint_targets.items():
    col=body_names.index(name);posture[col]=(1-blend)*home[col]+blend*float(value)
  if phase=='swing':posture[body_names.index(('left' if foot==0 else 'right')+'_knee_pitch_joint')]+=swing_knee_bump*np.sin(np.pi*u)**2
  headcols=[i for i,n in enumerate(body_names) if 'head' in n]
  for it in range(40):
   d.qpos[qids[headcols]]=posture[headcols]
   mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);A=[];b=[];err=[]
   for i,f in enumerate(feet):
    point=d.xpos[f]+d.xmat[f].reshape(3,3)@sole;jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,point,f);er=Rotation.from_matrix(RR[i]@d.xmat[f].reshape(3,3).T).as_rotvec();ep=target[i]-point;A.extend([100*jp[:,dofs],100*jr[:,dofs]]);b.extend([100*ep,100*er]);err.append(np.linalg.norm(ep))
   combined,jc=combined_com_and_jac();A.append(25*jc[:,dofs]);b.append(25*(com-combined))
   jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,root);A.append(3*jr[:,dofs]);b.append(3*Rotation.from_matrix(R0@d.xmat[root].reshape(3,3).T).as_rotvec())
   ar=np.zeros((31,len(dofs)));ar[:,6:]=np.eye(31)*.3;A.append(ar);b.append(.3*(posture-d.qpos[qids]));
   knee=np.zeros((2,len(dofs)));kr=[]
   for i,side in enumerate(('left','right')):
    col=body_names.index(side+'_knee_pitch_joint');knee[i,6+col]=5.;kr.append(5.*(posture[col]-d.qpos[qids[col]]))
   A.append(knee);b.append(np.array(kr))
   armcols=[i for i,n in enumerate(body_names) if any(j in n for j in ('shoulder','elbow','wrist'))];armA=np.zeros((len(armcols),len(dofs)));armA[np.arange(len(armcols)),6+np.array(armcols)]=10.;A.append(armA);b.append(10*(posture[armcols]-d.qpos[qids[armcols]]));AA=np.vstack(A);bb=np.concatenate(b);AA[:,6+np.array(headcols,dtype=int)]=0.;ranges=m.jnt_range[[x[0] for x in maps['active']]];lower=np.full(len(dofs),-.08);upper=np.full(len(dofs),.08);lower[6:]=np.maximum(lower[6:],ranges[:,0]-d.qpos[qids]);upper[6:]=np.minimum(upper[6:],ranges[:,1]-d.qpos[qids]);upper=np.maximum(upper,lower+1e-10)
   dq=lsq_linear(AA,bb,bounds=(lower,upper),method='bvls',tol=1e-8,max_iter=50).x;dv=np.zeros(m.nv);dv[dofs]=dq;mujoco.mj_integratePos(m,d.qpos,dv,1.)
   if np.max(np.abs(dq))<1e-7:break
  mujoco.mj_forward(m,d);poses=np.tile(np.eye(4),(2,1,1));poses[:,:3,:3]=RR;poses[:,:3,3]=target;actual=np.array([d.xpos[f]+d.xmat[f].reshape(3,3)@sole for f in feet]);ik.append([float(np.max(np.linalg.norm(actual-target,axis=1))),float(np.linalg.norm(combined_com_and_jac()[0]-com))]);qs.append(d.qpos.copy());cs.append(contact);desired.append(poses);labels.append(phase);caps.append(cap)
 qarray=np.array(qs);qvel=np.zeros((len(qs),m.nv))
 for i in range(len(qs)):
  lo=max(0,i-1);hi=min(len(qs)-1,i+1);mujoco.mj_differentiatePos(m,qvel[i],(hi-lo)*dt,qarray[lo],qarray[hi])
 qacc=np.gradient(qvel,dt,axis=0)
 return dict(time_s=ts,qpos=qarray,qvel=qvel,qacc=qacc,body_joint_names=np.array(body_names),qids=qids,dofs=dofs,feet=np.array(desired),contacts=np.array(cs),normal_caps=np.array(caps),phases=np.array(labels),ik_errors=np.array(ik),initial_qpos=initial,foot=foot,dx=dx,dy=dy,duration=duration,payload_rows=payload_rows)
def initial_robot_state(model, source, index=0):
 """JSON robot-only state or explicitly supplied NPZ component fixture.
 Scene object positions belong to the model/scene, never the robot JSON.
 """
 path=Path(source)
 if path.suffix=='.json':
  state=json.loads(path.read_text());q=model.qpos0.copy()
  root=np.asarray(state['root_qpos_wxyz'],dtype=float)
  if root.shape!=(7,):raise ValueError('root_qpos_wxyz must have seven values')
  q[:7]=root
  positions=dict(state['body_joints']);positions.update(state.get('passive_robot_joints',{}))
  for name,value in positions.items():
   j=model.joint(name)
   if model.body_rootid[model.jnt_bodyid[j.id]] != model.body_rootid[model.jnt_bodyid[0]]:
    raise ValueError('Initial JSON may set robot joints only')
   if model.jnt_type[j.id] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):
    raise ValueError('body_joints must contain scalar robot joints')
   q[int(j.qposadr[0])]=value
  return q
 z=dict(np.load(path));q=z['qpos']
 return (q[index] if q.ndim==2 else q).copy()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True,help='Physical scene XML or version-matched MJB')
 p.add_argument('--initial',required=True,help='Robot-only initial-standing JSON or explicit NPZ fixture')
 p.add_argument('--initial-index',type=int,default=0)
 p.add_argument('--audit',required=True,help='Named body/actuator mapping JSON')
 p.add_argument('--output',required=True,help='Output reference NPZ')
 p.add_argument('--foot',choices=['left','right'],default='left')
 p.add_argument('--dx',type=float,default=.15)
 p.add_argument('--dy',type=float,default=0.)
 p.add_argument('--clearance',type=float,default=.035)
 p.add_argument('--knee-bump',type=float,default=.20)
 p.add_argument('--support-bias',type=float,default=0.,help='Load-point offset along the SUPPORT sole x axis, metres')
 p.add_argument('--waist-upper',type=float,default=.04,help='Planning-only waist-pitch upper restriction, radians')
 p.add_argument('--payload-json',help='Known planned payload rows; no measured object forces')
 p.add_argument('--terminal-joints-json',help='Optional body-joint targets during final double support')
 a=p.parse_args()
 if abs(a.support_bias)>.10791-.015:
  p.error('support-bias must remain inside the unchanged sole with its 15 mm inset')
 m=mujoco.MjModel.from_binary_path(a.model) if a.model.endswith('.mjb') else mujoco.MjModel.from_xml_path(a.model)
 initial=initial_robot_state(m,a.initial,a.initial_index)
 names=[x['joint'] for x in json.loads(Path(a.audit).read_text())['mapping']]
 maps={'active':[(m.joint(n).id,int(m.joint(n).qposadr[0]),int(m.joint(n).dofadr[0])) for n in names]}
 m.jnt_range[m.joint('waist_pitch_joint').id,1]=a.waist_upper
 payload=np.zeros((2,4))
 if a.payload_json:
  value=json.loads(Path(a.payload_json).read_text());payload=np.asarray(value.get('payload_rows') if isinstance(value,dict) else value,dtype=float)
  if payload.shape!=(2,4) or (payload[:,0]<0).any():p.error('payload rows must be [2,4] with nonnegative masses')
 terminal=None if not a.terminal_joints_json else json.loads(Path(a.terminal_joints_json).read_text())
 started=time.monotonic()
 out=make(m,initial,maps,foot=0 if a.foot=='left' else 1,dx=a.dx,dy=a.dy,clearance=a.clearance,payload_rows=payload,terminal_joint_targets=terminal,swing_knee_bump=a.knee_bump,support_forward_bias=a.support_bias)
 # A held step must never silently default to an open gripper.
 out['gripper']=np.tile(np.where(payload[:,0]>0,255.,0.),(len(out['time_s']),1))
 out['expected_holding']=payload[:,0]>0
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**out)
 print(json.dumps(dict(reference=str(output),wall_s=time.monotonic()-started,max_sole_position_error_m=float(out['ik_errors'][:,0].max()),max_combined_com_target_error_m=float(out['ik_errors'][:,1].max()),duration_s=out['duration'],physical_qualification_required=True)))

if __name__=='__main__':main()
