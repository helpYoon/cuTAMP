"""Checked native whole-body configuration for transporting a held object."""
import math,torch
from cutamp.gn_refresh import gn_project_balanced
from cutamp.particle_initialization import ik_check,IKTol
from cutamp.utils.common import grasp_7dof_to_mat4x4,transform_spheres
from cutamp.carry_posture import project_endpoint,valid

def configure(world,grasp,arm='left'):
 O=world.get_object_pose('block').clone();O[:3,3]=O.new_tensor([float(world.stance_soles[:,0,3].mean())+.30,.25,.65]);G=grasp_7dof_to_mat4x4(O.new_tensor(grasp));target=(O@G@world.tool_from_ee[world.tool_frame_for_arm(arm)]).unsqueeze(0).expand(4,-1,-1).contiguous();q=world.q_init.unsqueeze(0).expand(4,-1).clone();names=list(world.model.joint_names);q[:,names.index(arm+'_elbow_pitch_joint')]=q.new_tensor([.35,.6,.9,1.2]);mov=transform_spheres(world.get_collision_spheres('block'),world.get_object_pose('block')).unsqueeze(0).expand(4,-1,-1).contiguous()
 with project_endpoint(world,torch.ones(4,device=world.device,dtype=torch.bool)):
  q=gn_project_balanced(world,q,target,arm,iters=40,collision=False)
  q=gn_project_balanced(world,q,target,arm,iters=120,movables=mov,collision=True)
 ok=ik_check(world,q,target,arm,mov,IKTol(.005,.05,.005,.05))[0]&valid(world,q)
 if not bool(ok.any()):raise RuntimeError('No configuration meets declared comfortable carry endpoint and native whole-body gates')
 cost=(q[:,3:5].square().sum(-1)).masked_fill(~ok,torch.inf);j=int(cost.argmin());world.retract_configuration_hints={arm:q[j]};world.carry_retract_straight=True
 from cutamp.anatomical_comfort import angles
 report=dict(object_target_world=O.cpu().tolist(),q=q[j].cpu().tolist(),joint_names=names,anatomical_angles_deg=(angles(world,q[j:j+1])*180/math.pi).cpu().tolist(),carry_posture=world.carry_posture,accepted_starts=int(ok.sum()),scope='Native endpoint only; motion, dynamics, CAD and physical retention qualification remain mandatory')
 print('CARRY_CONFIGURATION',report,flush=True);return report
