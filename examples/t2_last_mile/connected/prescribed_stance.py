"""Bind prescribed planar contact stance from a physical endpoint without editing simulator state."""
import json,os
from pathlib import Path
import torch

def install():
 import cutamp.algorithm as algorithm
 from cutamp.stance import StanceState,solve_stance_seed
 original=algorithm.setup_cutamp
 def setup(*args,**kwargs):
  result=original(*args,**kwargs);world=result[-1];path=Path(os.environ['CONNECTED_STANCE_JSON']);request=json.loads(path.read_text());assert request['joint_names']==list(world.model.joint_names)
  state=StanceState(torch.tensor(request['soles'],device=world.device,dtype=torch.float32),torch.ones(2,device=world.device,dtype=torch.bool));seed=torch.tensor(request['q_seed'],device=world.device,dtype=torch.float32);q=solve_stance_seed(world,state.soles,seed)
  world.set_planning_stance(state,q,sole_half=(.10791,.040886));receipt=dict(q_seed=q.cpu().tolist(),soles=state.soles.cpu().tolist(),seed_correction_max=float((q-seed).abs().max()),joint_names=list(world.model.joint_names))
  path.with_name(path.stem+'-realized.json').write_text(json.dumps(receipt,indent=2));print('PRESCRIBED_STANCE',receipt['seed_correction_max'],flush=True)
  return result
 algorithm.setup_cutamp=setup
