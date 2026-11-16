"""Declared manipulation comfort envelope, distinct from physical feasibility."""
import os,xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import replace
import yaml

def install():
 import t2_adapter
 model=t2_adapter.install();root=Path(os.environ['LOW_TASK_MODEL_DIR']).resolve();root.mkdir(parents=True,exist_ok=True)
 original=Path(model.cfg_path);cfg=yaml.safe_load(original.read_text());k=cfg['robot_cfg']['kinematics'];urdf=(original.parent/k['urdf_path']).resolve();tree=ET.parse(urdf)
 for j in tree.getroot().findall('joint'):
  if j.get('name')=='waist_pitch_joint':j.find('limit').set('upper','.0523598776')
 tree.write(root/'comfort.urdf');k['urdf_path']=str(root/'comfort.urdf');k['collision_spheres']=str((original.parent/k['collision_spheres']).resolve());k['asset_root_path']=str(original.parent)
 for link in k['extra_links'].values():
  if link.get('joint_name') in ['base_j_x','base_j_y']:link['joint_limits']=[-2.,2.]
  if link.get('joint_name')=='base_j_pitch':link['joint_limits']=[-.0872664626,1.0471975512]
  if link.get('joint_name')=='base_j_roll':link['joint_limits']=[-.2094395102,.2094395102]
 (root/'comfort.yml').write_text(yaml.safe_dump(cfg));model=replace(model,name='t2_low_task_comfort_v1',cfg_path=root/'comfort.yml',asset_root=root)
 from cutamp.robots import t1
 import cutamp.robots as robots,cutamp.algorithm as algorithm
 t1.REAL_LEGS=robots.REAL_LEGS=algorithm.REAL_LEGS=model
 return model
