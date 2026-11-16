"""Task-specific stand-up carry endpoint; never a whole-motion height bound."""
from contextlib import contextmanager
import math,torch

def residual(world,q):
 from cutamp.anatomical_comfort import angles
 cfg=world.carry_posture;n=list(world.model.joint_names);k=q[:,[n.index(s+'_knee_pitch_joint') for s in ['left','right']]]
 return (cfg['root_z_min_m']-q[:,2]).clamp_min(0)+.1*(angles(world,q)[:,0]-math.radians(cfg['torso_pitch_max_deg'])).clamp_min(0)+.1*(k-math.radians(cfg['knee_max_deg'])).clamp_min(0).sum(-1)

def valid(world,q,tol=1e-6):return residual(world,q)<=tol

@contextmanager
def project_endpoint(world,mask):
 old=getattr(world,'extra_posture_residual',None)
 def r(q):return residual(world,q)*mask.to(q.device,q.dtype)+(old(q) if old is not None else 0)
 world.extra_posture_residual=r
 try:yield
 finally:
  if old is None:del world.extra_posture_residual
  else:world.extra_posture_residual=old
