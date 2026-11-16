"""Isolated T2 model binding for upstream cuTAMP 890579f and native GraspGen-X."""
import json
from dataclasses import replace
from pathlib import Path
import torch
from curobo.types import DeviceCfg

def install():
 root=Path(__file__).parent/'t2-port'
 import sys;sys.path.insert(0,str(root))
 import legacy_adapter
 model=legacy_adapter.install(root)
 from cutamp.robots import t1
 import cutamp.robots as robots
 import cutamp.algorithm as alg
 import cutamp.grasps as grasps
 model=replace(model,name='t2_graspgen_real_legs',cfg_path=root/'planning/t2-graspgen.yml')
 def load_sweep_volume():return json.loads((root/'sweep_volume.json').read_text())
 t1.load_sweep_volume=grasps.load_sweep_volume=load_sweep_volume
 # Upstream's implementation takes the inverse of base_from_G, now set from the T2 axes.
 original_transform=__import__('importlib').util.spec_from_file_location
 def transforms(device_cfg=None):
  dc=device_cfg or DeviceCfg();B=torch.tensor(load_sweep_volume()['base_from_G'],device=dc.device,dtype=torch.float32);T=torch.linalg.inv(B)
  return {s+'_base_link':T.clone() for s in ('left','right')}
 def spheres(arm,device_cfg=None):
  dc=device_cfg or DeviceCfg();return torch.tensor(json.loads((root/'gripper_spheres_virtual.json').read_text())[arm],device=dc.device,dtype=torch.float32)
 t1.get_t1_tool_from_ee=robots.get_t1_tool_from_ee=grasps.get_t1_tool_from_ee=transforms
 t1.get_t1_gripper_spheres=robots.get_t1_gripper_spheres=spheres
 robots.REAL_LEGS=t1.REAL_LEGS=alg.REAL_LEGS=model
 return model
