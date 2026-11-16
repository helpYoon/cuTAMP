"""Native TAMP continuation from a measured, already-held payload; no synthetic pick action."""
import inspect,json,os
from pathlib import Path
import torch

def install():
 record=json.loads(Path(os.environ['CONNECTED_HELD_JSON']).read_text());arm=record['arm'];obj=record['object'];grasp_name='held_grasp';grasp=record['object_from_grasp_wxyz']
 import cutamp.tamp_world as tw
 import cutamp.t1_domain as domain
 from cutamp.particle_initialization import ParticleInitializer
 import cutamp.motion_solver as motion
 import cutamp.algorithm as algorithm
 if 'q_init' in record:
  old_setup=algorithm.setup_cutamp
  def setup(env,config,q_init=None,*,experiment_id):return old_setup(env,config,q_init=record['q_init'],experiment_id=experiment_id)
  algorithm.setup_cutamp=setup
 old_state=tw.TAMPWorld.initial_state.fget
 def state(world):
  s=set(old_state(world));s.discard(domain._ARM_FLUENTS[arm]['HandEmpty'].ground());s.discard(domain.HasNotPickedUp.ground(obj));s.add(domain._ARM_FLUENTS[arm]['Holding'].ground(obj));s.add(domain._ARM_FLUENTS[arm]['HoldingWithGrasp'].ground(obj,grasp_name))
  if record.get('just_picked',False):
   other='right' if arm=='left' else 'left';s.discard(domain._ARM_FLUENTS[arm]['CanMove'].ground());s.discard(domain._ARM_FLUENTS[other]['CanMove'].ground());s.add(domain._ARM_FLUENTS[arm]['JustPicked'].ground())
  return frozenset(s)
 tw.TAMPWorld.initial_state=property(state)
 old_init=ParticleInitializer._initial_particles
 def particles(self,n):
  p=old_init(self,n);p[grasp_name]=torch.tensor(grasp,device=self.world.device,dtype=torch.float32).repeat(n,1);return p
 ParticleInitializer._initial_particles=particles
 old_place=ParticleInitializer._place_step
 def place_step(self,*args,**kwargs):
  step=old_place(self,*args,**kwargs);return step._replace(drawn_from=tuple(n for n in step.drawn_from if n!=grasp_name))
 ParticleInitializer._place_step=place_step
 old_call=ParticleInitializer.__call__
 def initialize(self,skeleton,verbose=True):
  if any(op.operator.metadata.action_type=='pick' for op in skeleton):raise ValueError('Held-state continuation must not invent another pick')
  self.grasp_choice=[];p,_=self._walk(skeleton,self.config.num_particles,{},chosen={});return p
 ParticleInitializer.__call__=initialize
 source=inspect.getsource(motion.solve_curobo);needle='    held = {}  # {arm: (obj_name, grasp_from_obj)}';assert source.count(needle)==1
 source=source.replace(needle,"    held = {_initial_arm: (_initial_obj, torch.inverse(grasp_7dof_to_mat4x4(last_js.position.new_tensor(_initial_grasp))))}\n    _attach_object(planner, _initial_obj, last_js, _initial_arm)")
 namespace=dict(motion.__dict__);namespace.update(_initial_arm=arm,_initial_obj=obj,_initial_grasp=grasp)
 exec(compile(source,'native_held_payload_continuation','exec'),namespace);motion.solve_curobo=algorithm.solve_curobo=namespace['solve_curobo']
