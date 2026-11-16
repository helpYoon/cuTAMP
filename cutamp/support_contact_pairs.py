"""Exact-box support contact replacing only conservative payload/support sphere pairs.
Ordinary robot/support collisions are always retained. All accepted motions still
require independent dense CAD and physical contact qualification.
"""
from contextlib import contextmanager
import torch
from cutamp.utils.common import grasp_7dof_to_mat4x4

def configure(world,planner):
 cfg=getattr(world,'support_contact_geometry',None)
 if cfg is None:return
 for name in cfg['floor_names']:
  obstacle=world.get_object(name)
  if abs(float(obstacle.pose[2]+obstacle.dims[2]/2)-cfg['floor_top_m'])>1e-6:
   raise ValueError('Floor contact pair may only replace a coplanar floor/visual-region obstacle')
 kp=planner.attachment_manager.kinematics_params;kp.sphere_collision_mask.view(torch.int32).fill_(7)
 planner.scene_collision_checker.data.set_obstacle_layers(cfg['source_name'],2)
 for name in cfg['floor_names']:planner.scene_collision_checker.data.set_obstacle_layers(name,4)
 G=grasp_7dof_to_mat4x4(torch.tensor(cfg['object_from_grasp_wxyz'],device=world.device,dtype=torch.float32));dims=torch.tensor(cfg['object_dims_m'],device=world.device,dtype=torch.float32)
 corners=torch.tensor([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]],device=world.device)*dims
 def guard(q,arm,phase):
  from cutamp.motion_solver import _hand_poses,_ATTACH_LINK_FOR_ARM
  indices=kp.get_sphere_index_from_link_name(_ATTACH_LINK_FOR_ARM[arm])
  if not bool((kp.link_spheres[0,indices,3]>0).any()):return torch.ones(q.shape[0],device=q.device,dtype=torch.bool)
  W=_hand_poses(world,q,arm)@torch.linalg.inv(world.tool_from_ee[world.tool_frame_for_arm(arm)])@torch.linalg.inv(G)
  pts=torch.einsum('nij,kj->nki',W[:,:3,:3],corners)+W[:,:3,3].unsqueeze(1)
  height=cfg['source_top_m'] if phase=='lift' and getattr(planner,'_support_operator_kind',None)=='pick' else cfg['floor_top_m']
  return pts[...,2].amin(-1)>=height-1e-6
 world.payload_support_guard=guard
 planner._exact_payload_support_contacts=True

@contextmanager
def contact_phase(world,planner,arm,phase,operator):
 if not getattr(planner,'_exact_payload_support_contacts',False):yield;return
 from cutamp.motion_solver import _ATTACH_LINK_FOR_ARM
 kp=planner.attachment_manager.kinematics_params;link=_ATTACH_LINK_FOR_ARM[arm]
 mask=5 if phase=='lift' else (3 if phase=='insert' and 'Place(' in str(operator) else 7)
 planner._support_operator_kind='place' if 'Place(' in str(operator) else ('pick' if 'Pick(' in str(operator) else 'other')
 kp.set_link_collision_mask(link,mask)
 try:yield
 finally:
  kp.set_link_collision_mask(link,7)
  planner._support_operator_kind=None
