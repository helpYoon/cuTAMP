"""Exact cuboid SAT for movable-vs-static keyframe collision, allowing physical support contact.
Robot collision queries retain native cuRobo geometry. Cuboids only; 1um contact tolerance.
"""
import torch
from functools import lru_cache

def box_penetration(A,ha,B,hb):
    # 15 separating axes: 3 axes of each OBB and 9 cross products. Parallel cross axes ignored.
    aa=A[...,:3,:3].transpose(-1,-2);bb=B[...,:3,:3].transpose(-1,-2).expand_as(aa)
    cross=torch.cross(aa.unsqueeze(-2).expand(*aa.shape[:-2],3,3,3),bb.unsqueeze(-3).expand(*bb.shape[:-2],3,3,3),dim=-1).flatten(-3,-2)
    axes=torch.cat((aa,bb,cross),-2);length=axes.norm(dim=-1);unit=axes/length.clamp_min(1e-8).unsqueeze(-1)
    ra=(torch.matmul(unit,A[...,:3,:3]).abs()*ha).sum(-1);rb=(torch.matmul(unit,B[...,:3,:3]).abs()*hb).sum(-1)
    center=(A[...,:3,3]-B[...,:3,3]).unsqueeze(-2)
    overlaps=ra+rb-(unit*center).sum(-1).abs();overlaps=torch.where(length>1e-6,overlaps,torch.inf)
    return overlaps.amin(-1).clamp_min(0)

def install():
    from cutamp.utils.common import pose_list_to_mat4x4
    import cutamp.algorithm as algorithm
    import cutamp.tamp_world as tw
    from cutamp.cost_function import CostFunction
    @lru_cache(None)
    def constants(world):
        dev=world.device
        return ({o.name:torch.tensor(o.dims,device=dev,dtype=torch.float32)/2 for o in world.movables},[(pose_list_to_mat4x4(o.pose).to(dev),torch.tensor(o.dims,device=dev,dtype=torch.float32)/2) for o in world.statics])
    def cost(world,poses):
        halves,statics=constants(world)
        out=None
        for obj in world.movables:
            if not hasattr(obj,'dims'):raise TypeError('Exact support-aware collision supports cuboids only')
            A=poses[obj.name];ha=halves[obj.name]
            value=A.new_zeros(A.shape[:-2])
            for B,hb in statics:
                value=value+(box_penetration(A,ha,B,hb)-1e-6).clamp_min(0)
            out=value if out is None else out+value
        return out
    def initial(world,collision_tol=1e-6):
        poses={o.name:pose_list_to_mat4x4(o.pose).to(world.device)[None,None] for o in world.movables}
        penetration=cost(world,poses)
        if bool((penetration>collision_tol).any()):raise ValueError(f'Exact initial box penetration {float(penetration.max())}m')
    original=CostFunction.collision_costs
    def collision(self,rollout,obj_to_spheres):
        result=original(self,rollout,obj_to_spheres);result['values']['movable_to_world']=cost(self.world,rollout['obj_to_pose']);return result
    CostFunction.collision_costs=collision;algorithm.check_tamp_world_not_in_collision=tw.check_tamp_world_not_in_collision=initial
