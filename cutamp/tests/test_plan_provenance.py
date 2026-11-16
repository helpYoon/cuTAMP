import pickle
from types import SimpleNamespace
import torch
from curobo.types import JointState
from cutamp.utils.plan_processor import raw_motion_plan

def test_export_preserves_custom_generator_and_fallback_metadata():
    model=SimpleNamespace(joint_names=['a','b'],name='test')
    js=JointState(position=torch.zeros(3,2),velocity=torch.zeros(3,2),acceleration=torch.zeros(3,2),joint_names=['a','b'])
    e=dict(arm='left',dt=.025,op_idx=0,phase='lift',gripper=None,plan=js,trajectory_generator='wb_manifold_v2_projected_optimizer',trajectory_generator_status='all_planner_gates_passed',trajectory_diagnostics={'version':'WB-Manifold-2','history':[{'valid':True}]},native_fallback_status={'fallback_used':False})
    result=raw_motion_plan([e],model)
    segment=pickle.loads(pickle.dumps(result))['segments'][0]
    for key in ('trajectory_generator','trajectory_generator_status','trajectory_diagnostics','native_fallback_status'):
        assert segment[key]==e[key]
    e['trajectory_diagnostics']['history'][0]['valid']=False
    assert result['segments'][0]['trajectory_diagnostics']['history'][0]['valid']
    legacy={k:v for k,v in e.items() if not k.startswith('trajectory_') and k!='native_fallback_status'}
    assert 'trajectory_generator' not in raw_motion_plan([legacy],model)['segments'][0]
