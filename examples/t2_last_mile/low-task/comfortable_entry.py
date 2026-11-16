"""Low manipulation benchmark: declared comfort + intrinsic WB-Manifold-2."""
import json,os,runpy
from pathlib import Path
import torch
import comfort_profile;comfort_profile.install()
import cutamp.algorithm as algorithm
from cutamp.grasps import GraspSet
frozen=json.load(open(os.environ['STUDY_FROZEN_GRASP']))
algorithm.grasp_sets=lambda objects,device:{o.name:GraspSet(torch.tensor([frozen['object_from_grasp_wxyz']],device=device,dtype=torch.float32),torch.ones(1,device=device)) for o in objects}
algorithm.all_t1_operators=[op for op in algorithm.all_t1_operators if op.metadata.arm=='left']
import support_contact;support_contact.install()
import exact_placement;exact_placement.install()
import fixed_pick;fixed_pick.install()
import prescribed_stance;prescribed_stance.install()
if os.environ.get('CONNECTED_HELD_JSON'):
 import held_start;held_start.install()
old=algorithm.setup_cutamp
def setup(*args,**kwargs):
 result=old(*args,**kwargs);w=result[-1];w.manipulation_clearance={'pick_lift_height_m':.35,'place_approach_height_m':.10};w.approach_ik_iters=120;w.trajectory_waist_comfort=(-.61,-.17);w.manifold_cartesian_approach=False;w.manifold_free_hand_motion=False;w.carry_posture=dict(torso_pitch_max_deg=30.,root_z_min_m=.85,knee_max_deg=60.);source=w.get_object('support_block');w.support_contact_geometry=dict(source_name='support_block',floor_names=['floor','goal'],object_from_grasp_wxyz=frozen['object_from_grasp_wxyz'],object_dims_m=list(w.get_object('block').dims),source_top_m=float(source.pose[2]+source.dims[2]/2),floor_top_m=0.)
 if os.environ.get('LOW_TASK_PICK_HINT'):
  hint=json.load(open(os.environ['LOW_TASK_PICK_HINT']));assert hint['joint_names']==list(w.model.joint_names);w.configuration_hints={('block','left'):torch.tensor(hint['q'],device=w.device,dtype=torch.float32)}
 from cutamp.stance import solve_stance_seed
 w.ik_seed_postures=[];names=list(w.model.joint_names)
 for knee in [20.,40.]:
  q=w.q_init.clone();knee=knee*3.141592653589793/180
  for side in ['left','right']:
   q[names.index(side+'_knee_pitch_joint')]=knee;q[names.index(side+'_hip_pitch_joint')]=-knee/2;q[names.index(side+'_ankle_pitch_joint')]=-knee/2
  q[2]-=.05 if knee<.5 else .15;w.ik_seed_postures.append(solve_stance_seed(w,w.stance_soles,q))
 if os.environ.get('LOW_TASK_HELD_CARRY')=='1':
  from cutamp.carry_configuration import configure as configure_carry
  w.carry_goal_report=configure_carry(w,frozen['object_from_grasp_wxyz'])
 print('LOW_TASK_CONFIG',dict(algorithm='WB-Manifold-2',comfort='explicit declared body policy v1',clearance=w.manipulation_clearance,approach_ik_iters=w.approach_ik_iters),flush=True)
 return result
algorithm.setup_cutamp=setup
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
runpy.run_module('cutamp.scripts.run_cutamp',run_name='__main__')
