"""Actual saved-state robot contact geometry replay, independent of task success."""
import argparse,json,collections
from pathlib import Path
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--trace',required=True);a=p.parse_args();path=Path(a.trace);m=(mujoco.MjModel.from_binary_path(a.model) if a.model.endswith('.mjb') else mujoco.MjModel.from_xml_path(a.model));d=mujoco.MjData(m);z=dict(np.load(path));root=m.body('trunk').id;robot=set(np.flatnonzero(m.body_rootid==m.body_rootid[root]).tolist());feet={m.body(s+'_ankle_roll_link').id for s in ('left','right')};unexpected=collections.defaultdict(list);intended=collections.defaultdict(list);times=z.get('time',z.get('time_s'))
for k,q in enumerate(z['qpos']):
    d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
    for c in d.contact:
        bs=[int(m.geom_bodyid[g]) for g in c.geom];rs=[b in robot for b in bs]
        if not any(rs):continue
        names=[m.body(b).name for b in bs]
        if any(b in feet for b in bs) and any(b==0 or m.body(b).name=='floor' for b in bs):continue
        key=' | '.join(names[i]+':'+m.geom(c.geom[i]).name for i in range(2))
        fingers=any('gripper/' in names[i] and any(w in names[i] for w in ('follower','finger','pad')) for i in range(2) if rs[i])
        expected=sum(rs)==1 and fingers and 'block' in names
        (intended if expected else unexpected)[key].append((float(times[k]),float(c.dist),k))
def summarize(pairs):
    return {key:dict(contact_frames=len({x[2] for x in v}),contact_points=len(v),start_s=v[0][0],end_s=v[-1][0],min_dist_m=min(x[1] for x in v)) for key,v in pairs.items()}
report=dict(scope='Actual saved-state physical-model robot geometry replay; excludes sole-ground; intended gripper follower/finger-block pairs retained separately. No simulation rerun.',frames=len(z['qpos']),unexpected_pairs=summarize(unexpected),unexpected_pair_count=len(unexpected),unexpected_contact_frames=len({x[2] for v in unexpected.values() for x in v}),expected_finger_contact_pairs=summarize(intended));(path.parent/'whole-body-contact-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
