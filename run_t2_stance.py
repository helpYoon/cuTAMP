import os,sys,runpy,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import t2_adapter
t2_adapter.install()
from cutamp.tamp_world import TAMPWorld
from cutamp.stance import StanceState,solve_stance_seed
from curobo.types import JointState
original=TAMPWorld.__init__
def init(self,*a,**kw):
 original(self,*a,**kw)
 if os.environ.get('STANCE_MODE','legacy')=='legacy':return
 q=self.q_init.clone();q[0]+=float(os.environ.get('STANCE_DX','.02'));q[1]+=float(os.environ.get('STANCE_DY','.01'));q[5]+=float(os.environ.get('STANCE_YAW','.08'))
 ks=self.kinematics.compute_kinematics(JointState.from_position(q[None]));soles=torch.stack([ks.tool_poses.get_link_pose(f,make_contiguous=True).get_matrix().reshape(4,4).clone() for f in self.model.sole_frames])
 if os.environ.get('STANCE_MODE')=='staggered':
  soles[0,0,3]+=float(os.environ.get('STANCE_STAGGER','.04'));soles[1,0,3]-=float(os.environ.get('STANCE_STAGGER','.04'))
  q=solve_stance_seed(self,soles,q)
 self.set_planning_stance(StanceState(soles,torch.ones(2,dtype=torch.bool,device=self.device)),q,(.10791,.040886))
TAMPWorld.__init__=init
runpy.run_module('cutamp.scripts.run_cutamp',run_name='__main__')
