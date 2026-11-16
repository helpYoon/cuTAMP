"""Yaw-aware full cuboid footprint in the surface frame, with a 5mm execution reserve."""
import itertools,torch

def placement_residuals(T,corners,Tsurf,half,inset=.005,release_gap=.002):
    world_corners=corners[None]@T[:,:3,:3].transpose(-1,-2)+T[:,:3,3][:,None]
    local=(world_corners-Tsurf[:3,3])@Tsurf[:3,:3]
    xy=(local[...,:2].abs()-(half[:2]-inset)).clamp_min(0).norm(dim=-1).sum(-1)
    support=(local[...,2].amin(-1)-half[2]-release_gap).abs()
    return xy,support

def install():
 from cutamp.cost_function import CostFunction
 from cutamp.utils.common import pose_list_to_mat4x4
 original_init=CostFunction.__init__
 def init(self,*args,**kwargs):
  original_init(self,*args,**kwargs);self._exact_corners={};self._exact_surfaces={}
  for surface,placements in self.surface_to_placements.items():
   obj=self.world.get_object(surface);T=pose_list_to_mat4x4(obj.pose).to(self.world.device);half=torch.tensor(obj.dims,device=self.world.device)/2
   self._exact_surfaces[surface]=(T,half)
   for name,_ in placements:
    obj=self.world.get_object(name);self._exact_corners[name]=torch.tensor(list(itertools.product([-1.,1.],repeat=3)),device=self.world.device)*torch.tensor(obj.dims,device=self.world.device)/2
 original=CostFunction.stable_placement_costs
 def stable(self,rollout,obj_to_spheres):
  result=original(self,rollout,obj_to_spheres)
  if result is None:return result
  for surface,placements in self.surface_to_placements.items():
   Tsurf,half=self._exact_surfaces[surface];xy=[];support=[]
   for obj,placement in placements:
    T=rollout['obj_to_pose'][obj][:,rollout['action_to_pose_ts'][placement]]
    x,z=placement_residuals(T,self._exact_corners[obj],Tsurf,half)
    xy.append(x);support.append(z)
   result['values'][surface+'_in_xy']=torch.stack(xy,1);result['values'][surface+'_support']=torch.stack(support,1)
  return result
 CostFunction.__init__=init;CostFunction.stable_placement_costs=stable
