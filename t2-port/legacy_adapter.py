"""Model-only T2 adapter for unmodified release-based cuTAMP algorithms."""
import json,inspect,hashlib,os
from pathlib import Path
import torch
from curobo.types import DeviceCfg

def install(root):
 from cutamp.robots import t1
 import cutamp.robots as robots
 from cutamp.robots.envelope import Envelope
 data=json.loads((root/'planning/release-model.json').read_text())
 envelope=Envelope('t2_physical',hashlib.sha1((root/'planning/t2_planner.urdf').read_bytes()).hexdigest(),{},None,None,None,{}, {}, {})
 model=t1.T1Model(name='t2_real_legs',cfg_path=root/'planning/t2-release.yml',asset_root=root/'planning',joint_names=tuple(data['names']),base=slice(0,6),left_arm=slice(6,13),right_arm=slice(13,20),body=slice(20,len(data['names'])),home=tuple(data['home']),support=t1.Support('stance',(.09,.13),(.08,.12)),sole_frames=('left_sole_link','right_sole_link'),envelope=envelope)
 hands=json.loads((root/'planning/gripper_spheres.json').read_text())
 def transforms(device_cfg=None):
  dc=device_cfg or DeviceCfg();T=torch.eye(4,device=dc.device,dtype=torch.float32);T[1,1]=-1.;T[2,2]=-1.
  return {'left_base_link':T,'right_base_link':T.clone()}
 def spheres(arm,device_cfg=None):
  dc=device_cfg or DeviceCfg();value=torch.tensor(hands[arm],device=dc.device,dtype=torch.float32)
  value[:,:3]=value[:,:3]@transforms(dc)[arm+'_base_link'][:3,:3].T
  return value
 t1.REAL_LEGS=robots.REAL_LEGS=model
 t1.get_t1_tool_from_ee=robots.get_t1_tool_from_ee=transforms
 t1.get_t1_gripper_spheres=robots.get_t1_gripper_spheres=spheres
 t1.LEFT_ARM_JOINT_NAMES=tuple(data['names'][6:13]);t1.RIGHT_ARM_JOINT_NAMES=tuple(data['names'][13:20])
 # Preserve upstream mirrored leg sampling; translate only its T1 joint names.
 import cutamp.particle_initialization as pi
 aliases={f'{side}_{joint}':f'{side.lower()}_{joint.lower()}_joint' for side in ('Left','Right') for joint in pi._MIRRORED_LEG_JOINTS+pi._HELD_LEG_JOINTS}
 namespace=dict(pi.__dict__);namespace['_t2_aliases']=aliases
 source=inspect.getsource(pi._noisy_start).replace('col = list(model.joint_names).index','col = lambda n: list(model.joint_names).index(_t2_aliases.get(n,n))')
 exec(compile(source,'t2_noisy_start','exec'),namespace);pi._noisy_start=namespace['_noisy_start']
 # Public cuRobo shape knob and scalar cost weights, scoped to T2 only.
 original_create=t1.MotionPlannerCfg.create
 def create(*args,**kwargs):
  if os.environ.get('T2_KNOTS'):kwargs['num_knots']=int(os.environ['T2_KNOTS'])
  return original_create(*args,**kwargs)
 t1.MotionPlannerCfg.create=staticmethod(create)
 original_factory=t1.get_t1_motion_planner
 def factory(*args,**kwargs):
  planner=original_factory(*args,**kwargs)
  if os.environ.get('T2_POSE_WEIGHT'):
   weight=float(os.environ['T2_POSE_WEIGHT'])
   for rollout in planner.trajopt_solver.optimizer_rollouts:
    for manager in (rollout.cost_manager,rollout.metrics_cost_manager):
     term=manager.get_cost('tool_pose');term._weight.fill_(weight);term.config.weight.copy_(term._weight)
  return planner
 t1.get_t1_motion_planner=factory
 import cutamp.motion_solver as motion
 motion.get_t1_motion_planner=factory
 import cutamp.algorithm as alg
 alg.REAL_LEGS=model
 return model

def install_diagnostics(root):
 import __main__,time
 if hasattr(__main__,'deadline'):__main__.deadline=max(__main__.deadline,time.monotonic()+600)
 import cutamp.motion_solver as motion,os
 profile_file=root/'planner-profile.json'
 if profile_file.exists():
  profile=json.loads(profile_file.read_text())
  base_factory=getattr(motion.get_t1_motion_planner,'_port_original',motion.get_t1_motion_planner)
  def profiled_factory(*args,**kwargs):
   planner=base_factory(*args,**kwargs)
   if profile.get('collision_weight'):
    for rollout in planner.trajopt_solver.optimizer_rollouts:
     for name in ('self_collision','scene_collision'):
      for term in rollout.get_cost_component_by_name(name):
       term._weight.fill_(profile['collision_weight']);term.config.weight.copy_(term._weight)
   return planner
  profiled_factory._port_original=base_factory
  motion.get_t1_motion_planner=profiled_factory
 original=motion._failure_reason
 while getattr(original,'__name__','')=='reason' and original.__closure__:
  cells=dict(zip(original.__code__.co_freevars,original.__closure__))
  if 'original' not in cells:break
  original=cells['original'].cell_contents
 index=0
 def reason(world,result):
  nonlocal index
  record={'result_type':str(type(result)),'result_fields':{k:repr(v)[:1500] for k,v in vars(result).items() if k not in ('js_solution','interpolated_trajectory','solution','optimized_seeds','metrics','interpolated_metrics')}} if result is not None else {'result':None}
  for label,js in [('native',getattr(result,'js_solution',None)),('interpolated',result.get_interpolated_plan() if result is not None else None)]:
   if js is None:continue
   from curobo.types import JointState
   from cutamp.costs import curobo_pose_error
   from cutamp.com_polygon_cost import com_polygon_mask_from_state
   q=motion._full_positions(world,js)
   from curobo.kinematics import Kinematics
   diag_kin=Kinematics(world.kinematics.config,compute_com=True)
   state=diag_kin.compute_kinematics(JointState.from_position(q))
   feet=torch.stack([state.tool_poses.get_link_pose(f,make_contiguous=True).get_matrix() for f in world.model.sole_frames],dim=1)
   pe,re=curobo_pose_error(feet,world.stance_soles.expand_as(feet));bounds=world.joint_limits
   record[label+'_checks']=dict(sole_position_max_m=float(pe.max()),sole_rotation_max_rad=float(re.max()),joint_max_violation=float(torch.maximum(bounds[0]-q,q-bounds[1]).max()),com_all=bool(com_polygon_mask_from_state(state,world.model.support).all()))
  for label,metrics in [('native',getattr(result,'metrics',None)),('interpolated',getattr(result,'interpolated_metrics',None))]:
   if metrics is None:continue
   values={}
   costs=getattr(metrics,'costs_and_constraints',None)
   for group in ('costs','constraints','hybrid_costs_constraints'):
    collection=getattr(costs,group,None)
    if collection is not None:
     values[group]={name:dict(min=float(v.min()),max=float(v.max())) for name,v in zip(collection.names,collection.values)}
   values['fields']={k:repr(v)[:700] for k,v in vars(metrics).items() if k not in ('actions','state','costs_and_constraints')}
   record[label]=values
  folder=root/('failure-diagnostics-'+os.environ.get('SLURM_JOB_ID','manual'));folder.mkdir(exist_ok=True)
  if result is not None and result.get_interpolated_plan() is not None:
   js=result.get_interpolated_plan();(folder/f'failed-q-{time.time_ns()}.json').write_text(json.dumps(dict(names=list(js.joint_names),position=js.position.detach().cpu().tolist())))
  (folder/f'failure-{index:03d}.json').write_text(json.dumps(record,indent=2));index+=1
  print('T2_NATIVE_FAILURE',record,flush=True)
  return original(world,result)
 motion._failure_reason=reason
