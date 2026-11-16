"""Explicit, batched support state. No physical contact transition is implied."""
from dataclasses import dataclass
from typing import Mapping
import torch

@dataclass(frozen=True)
class StanceState:
    soles: torch.Tensor  # [...,2,4,4], world_from_sole; one shared whole-body stance
    contacts: torch.Tensor  # [...,2], boolean

    def __post_init__(self):
        if self.soles.shape[-3:] != (2,4,4): raise ValueError('soles must end in [2,4,4]')
        if self.contacts.shape != self.soles.shape[:-2] or self.contacts.dtype != torch.bool: raise ValueError('contacts shape/dtype')
        if self.contacts.device != self.soles.device: raise ValueError('device mismatch')
        if not bool(torch.isfinite(self.soles).all()): raise ValueError('nonfinite sole transform')
        R=self.soles[...,:3,:3].double();eye=torch.eye(3,device=R.device,dtype=R.dtype)
        if not torch.allclose(R.mT@R,eye.expand_as(R),atol=1e-5,rtol=0) or not bool((torch.linalg.det(R)>0).all()):raise ValueError('invalid rotation')
        row=self.soles[...,3,:];want=row.new_tensor([0,0,0,1]).expand_as(row)
        if not torch.allclose(row,want,atol=1e-6,rtol=0):raise ValueError('invalid homogeneous transform')
        if not bool(self.contacts.any(-1).all()):raise ValueError('flight mode not supported')

    def support_vertices(self,sole_half,load_guard=.1):
        """Flat-ground load-share support region, vertex set with padding.

        Convex hull of a*L+(1-a)*R and (1-a)*L+a*R enforces that
        both contacts carry at least a weight share. Single support uses its sole.
        """
        if not 0<=load_guard<=.5:raise ValueError('invalid load guard')
        R=self.soles[...,:3,:3]
        if not bool((R[...,2,:2].abs()<1e-5).all()) or not bool((self.soles[...,0,2,3]-self.soles[...,1,2,3]).abs().lt(1e-4).all()):raise ValueError('coplanar flat ground required')
        corners=self.soles.new_tensor([[-1,-1],[-1,1],[1,1],[1,-1]])*self.soles.new_tensor(sole_half)
        xy=torch.einsum('...fij,vj->...fvi',R[...,:2,:2],corners)+self.soles[...,:2,3].unsqueeze(-2)
        l,r=xy[...,0,:,:],xy[...,1,:,:];a=load_guard
        lr=(a*l[..., :,None,:]+(1-a)*r[...,None,:,:]).flatten(-3,-2)
        rl=((1-a)*l[..., :,None,:]+a*r[...,None,:,:]).flatten(-3,-2)
        both=torch.cat((lr,rl),-2)
        single=torch.where(self.contacts[...,0,None,None],l,r).repeat_interleave(8,-2)
        return torch.where(self.contacts.all(-1)[...,None,None],both,single)

    def support_halfspaces(self,sole_half,load_guard=.1):
        v=self.support_vertices(sole_half,load_guard);d=v[...,None,:,:]-v[...,:,None,:]
        length=d.norm(dim=-1);normal=torch.stack((-d[...,1],d[...,0]),-1)/length.clamp_min(1e-12)[...,None]
        offsets=v[...,None,None,:,:]-v[...,:,None,None,:]
        edge=((normal[...,None,:]*offsets).sum(-1)>=-1e-6).all(-1)&(length>1e-8)
        offset=(normal*v[...,:,None,:]).sum(-1)
        return normal.flatten(-3,-2),offset.flatten(-2),edge.flatten(-2)

    def support_margin(self,points,sole_half,load_guard=.1,inset=0.):
        """Signed distance to convex support boundary, positive inside. Batched GPU math."""
        normal,offset,edge=self.support_halfspaces(sole_half,load_guard)
        distances=(normal*points[...,None,:]).sum(-1)-offset
        return torch.where(edge,distances,torch.inf).amin(-1)-inset

@dataclass(frozen=True)
class StanceSchedule:
    """Configuration names map to persistent stance IDs; transitions must be explicit."""
    states: Mapping[str,StanceState]
    configurations: Mapping[str,str]

    def for_configuration(self,name): return self.states[self.configurations[name]]

    def for_action(self,start,end,transition=False):
        a,b=self.configurations[start],self.configurations[end]
        if a!=b and not transition:raise ValueError('manipulation action changes stance without transition')
        return self.states[a],self.states[b]

def solve_stance_seed(world, desired_soles, seed=None, iterations=60):
    """Native whole-body sole IK for an explicitly initialized flat-ground study.

    Returns a seed; raises if joint-limited IK misses sole tolerances. This does
    not certify static equilibrium, collision freedom, or ability to step here.
    """
    from curobo.types import JointState
    from cutamp.gn_refresh import _sole_residual_and_jacobian
    from cutamp.costs import curobo_pose_error
    state=StanceState(desired_soles,torch.ones(2,dtype=torch.bool,device=desired_soles.device))
    q=(world.q_init if seed is None else seed).detach().clone().reshape(1,-1)
    lo,hi=world.joint_limits.to(q.device,q.dtype);kin=world.kinematics_jac
    idx=tuple(list(kin.tool_frames).index(f) for f in world.model.sole_frames)
    with torch.no_grad():
        for _ in range(iterations):
            ks=kin.compute_kinematics(JointState.from_position(q))
            r,J=_sole_residual_and_jacobian(ks,idx,state.soles,torch.ones_like(q))
            # Keep the inverse small and solve in float64; cuRobo enables TF32.
            J=J.double();r=r.double()
            dq=J.mT@torch.linalg.solve_ex(J@J.mT+torch.eye(12,device=q.device,dtype=torch.float64)[None]*1e-6,r[...,None])[0]
            q=(q+dq[...,0].to(q.dtype).clamp(-.05,.05)).clamp(lo,hi)
        ks=kin.compute_kinematics(JointState.from_position(q))
        actual=torch.stack([ks.tool_poses.get_link_pose(f,make_contiguous=True).get_matrix().reshape(4,4) for f in world.model.sole_frames])
        p,R=curobo_pose_error(actual,state.soles)
        if not bool((p<=1e-4).all()&(R<=1e-3).all()):
            raise RuntimeError(f'stance seed IK failed: sole position {float(p.max()):.6f}m rotation {float(R.max()):.6f}rad')
    return q[0]
