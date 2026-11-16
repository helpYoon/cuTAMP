"""Signed anatomical comfort in fixed initial-foot-heading coordinates.

Robot-specific verified T2 chain: trunk Ry(basepitch) Rx(baseroll) Rz(baseyaw),
pelvis trunk Ry(waistpitch) Rx(waistroll) Rz(waistyaw). Yaw at the final
factor does not change up. Bounds are an explicit task policy, not physical
joint limits. Does not prescribe root height or knee bend.
"""
import math
import torch

def _rx(t):
 c,s=t.cos(),t.sin();o=torch.ones_like(t);z=torch.zeros_like(t)
 return torch.stack((o,z,z,z,c,-s,z,s,c),-1).reshape(*t.shape,3,3)
def _ry(t):
 c,s=t.cos(),t.sin();o=torch.ones_like(t);z=torch.zeros_like(t)
 return torch.stack((c,z,s,z,o,z,-s,z,c),-1).reshape(*t.shape,3,3)
def _rz(t):
 c,s=t.cos(),t.sin();o=torch.ones_like(t);z=torch.zeros_like(t)
 return torch.stack((c,-s,z,s,c,z,z,z,o),-1).reshape(*t.shape,3,3)

def angles(world,q):
 names=list(world.model.joint_names)
 R=_ry(q[...,3])@_rx(q[...,4])@_rz(q[...,5])
 P=R@_ry(q[...,names.index('waist_pitch_joint')])@_rx(q[...,names.index('waist_roll_joint')])
 soles=world.stance_soles
 xy=soles[...,:2,0].sum(0)
 forward=torch.cat((xy,torch.zeros_like(xy[:1])))
 forward=forward/forward.norm().clamp_min(1e-9)
 left=torch.stack((-forward[1],forward[0],torch.zeros_like(forward[0])))
 up=R[...,2];pelvisup=P[...,2]
 pitch=torch.atan2((up*forward).sum(-1),up[...,2])
 pelvispitch=torch.atan2((pelvisup*forward).sum(-1),pelvisup[...,2])
 lateral=torch.atan2((up*left).sum(-1),up[...,2])
 return torch.stack((pitch,lateral,pitch-pelvispitch),-1)

def violations(world,q):
 a=angles(world,q)
 bounds=((-5.,60.),(-12.,12.),(-3.,45.))
 return torch.stack([((lo*math.pi/180)-a[...,i]).clamp_min(0)
                     +(a[...,i]-(hi*math.pi/180)).clamp_min(0)
                     for i,(lo,hi) in enumerate(bounds)],-1)

def residual(world,q):
 # Metres-equivalent trust-region row; nonzero exactly when policy violated.
 return .10*violations(world,q).sum(-1)
