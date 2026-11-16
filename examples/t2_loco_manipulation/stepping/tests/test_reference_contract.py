"""Asset-free tests of the reference and initialization interfaces."""
import json,sys
from pathlib import Path
import mujoco,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_step import StepReference
from make_step_reference import initial_robot_state
from contact_wbc import ContactWholeBodyQP

def model_and_maps():
    joints=''.join(f'<body name="b{i}" pos="0 0 .01"><joint name="j{i}"/><geom type="sphere" size=".01" mass=".1"/></body>' for i in range(31))
    motors=''.join(f'<motor name="j{i}" joint="j{i}" ctrllimited="true" ctrlrange="-50 50"/>' for i in range(31))
    xml=f'''<mujoco><worldbody><body name="trunk" pos="0 0 1"><freejoint name="world_joint"/><geom type="sphere" size=".1" mass="10"/>{joints}<body name="left_ankle_roll_link" pos="0 .1 -.9"/><body name="right_ankle_roll_link" pos="0 -.1 -.9"/><body name="left_gripper/base" pos=".2 .2 0"/><body name="right_gripper/base" pos=".2 -.2 0"/></body><body name="block" pos=".95 .25 .18"><freejoint name="block_free"/><geom type="box" size=".03 .03 .03" mass=".1"/></body></worldbody><actuator>{motors}</actuator></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml)
    active=[(m.joint(f'j{i}').id,int(m.joint(f'j{i}').qposadr[0]),int(m.joint(f'j{i}').dofadr[0])) for i in range(31)]
    return m,dict(active=active,actuators=list(range(31)))

def test_robot_only_json_preserves_scene_object(tmp_path):
    m,maps=model_and_maps();initial=m.qpos0.copy();oid=int(m.joint('block_free').qposadr[0])
    p=tmp_path/'initial.json';p.write_text(json.dumps(dict(root_qpos_wxyz=[.1,0,1,1,0,0,0],body_joints={'j0':.2},passive_robot_joints={'j1':.03})))
    q=initial_robot_state(m,p)
    assert q[0]==.1 and q[maps['active'][0][1]]==.2 and q[maps['active'][1][1]]==.03
    np.testing.assert_array_equal(q[oid:oid+7],initial[oid:oid+7])

def test_reference_packet_derivatives_and_contact_switch():
    m,maps=model_and_maps();ts=np.linspace(0,1,5);q=np.tile(m.qpos0,(5,1));q[:,0]=ts**3
    feet=np.tile(np.eye(4),(5,2,1,1));feet[:,0,0,3]=ts**3
    contacts=np.ones((5,2),bool);contacts[2:,0]=False
    ref=StepReference(m,dict(time_s=ts,qpos=q,feet=feet,contacts=contacts,normal_caps=np.ones((5,2))*1000),maps)
    packet=ref.packet(.5);assert len(packet)==10 and packet[0].shape==(74,)
    np.testing.assert_allclose(packet[0][0],.125,atol=1e-12)
    np.testing.assert_allclose(packet[0][37],.75,atol=1e-12)
    np.testing.assert_allclose(packet[9][0],3.,atol=1e-12)
    np.testing.assert_array_equal(ref.feet_packet(.5)[3],[False,True])
    np.testing.assert_allclose(ref.feet_packet(.5)[1][0,0],.75,atol=1e-12)

def test_controller_rejects_wrong_body_mapping():
    m,maps=model_and_maps();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    with pytest.raises(ValueError,match='31 named'):
        ContactWholeBodyQP(m,d,dict(active=maps['active'][:-1],actuators=maps['actuators']),None)
