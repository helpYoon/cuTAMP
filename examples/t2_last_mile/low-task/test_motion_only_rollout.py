from types import SimpleNamespace as NS
import torch
from cutamp.t1_domain import Conf,_ARM_FLUENTS
from cutamp.rollout import RolloutFunction,get_conf_parameters

def op(values,types,actionable=False):
 return NS(values=values,operator=NS(parameters=[NS(type=t) for t in types],metadata=NS(is_actionable=actionable,arm='left',is_motion=not actionable,action_type='retract',name='LeftRetractHolding')))

def test_actionable_configuration_selection_unchanged():
 skeleton=[op(['q0','traj','q1'],[Conf,'traj',Conf]),op(['block','grasp','q1'],['movable','grasp',Conf],True)]
 assert get_conf_parameters(skeleton)==['q1']

def test_motion_only_unique_endpoints():
 skeleton=[op(['q0','traj1','q1'],[Conf,'traj',Conf]),op(['q1','traj2','q2'],[Conf,'traj',Conf])]
 assert get_conf_parameters(skeleton)==['q0','q1','q2']

class Poses:
 def __init__(self,q):self.q=q
 def get_link_pose(self,name,make_contiguous=False):
  T=torch.eye(4).expand(len(self.q),4,4).clone()
  if name.endswith('tool'):
   T[:,0,3]=self.q[:,0]+.1;T[:,1,3]=.2 if name=='left_tool' else -.2;T[:,2,3]=.7
  else:T[:,1,3]=.1 if name=='left_sole' else -.1
  return NS(get_matrix=lambda:T)

def run(held):
 skeleton=[op(['block','held_grasp','left_q0','traj','left_q1'],['movable','grasp',Conf,'traj',Conf])]
 initial=torch.eye(4);initial[:3,3]=torch.tensor([.2,.2,.65])
 def kin(js):return NS(tool_poses=Poses(js.position),robot_spheres=torch.zeros(len(js.position),1,4))
 world=NS(movables=[NS(name='block')],get_object_pose=lambda _:initial,
  model=NS(sole_frames=['left_sole','right_sole']),site_kinematics=lambda _:NS(compute_kinematics=kin),
  tool_frame_for_arm=lambda a: 'left_tool' if a=='left' else 'right_tool',
  tool_from_ee={'left_tool':torch.eye(4),'right_tool':torch.eye(4)},
  initial_state={_ARM_FLUENTS['left']['HoldingWithGrasp'].ground('block','held_grasp')} if held else set())
 a=torch.zeros(2,6);a[1,0]=1.;b=a.clone();b[:,0]+=.2
 particles={'left_q0':a,'right_q0':a.clone(),'left_q1':b,'held_grasp':torch.tensor([-.1,0,.05,1.,0,0,0]).expand(2,-1)}
 return RolloutFunction(skeleton,world)(particles)

def test_held_motion_follows_fk_without_dummy_action():
 r=run(True)
 assert r['action_params']==[] and r['gripper_close']==[True,True]
 expected=torch.tensor([[[.2,.2,.65],[.4,.2,.65]],[[1.2,.2,.65],[1.4,.2,.65]]])
 torch.testing.assert_close(r['obj_to_pose']['block'][...,:3,3],expected)
 torch.testing.assert_close(r['world_from_ee'],r['world_from_ee_desired'])

def test_empty_motion_does_not_move_object_or_close_gripper():
 r=run(False)
 assert r['action_params']==[] and r['gripper_close']==[False,False]
 torch.testing.assert_close(r['obj_to_pose']['block'][...,:3,3],torch.tensor([.2,.2,.65]).expand(2,2,3))
