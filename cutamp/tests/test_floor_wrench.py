"""The floor's push on the planted soles (design 7.4a): the split between the soles, the base-joint
map, and RNEA's second pass against virtual work, on a 12-DoF biped hung from a 6-DoF virtual base."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from cutamp.floor_wrench import base_wrench, install_floor_wrench, make_floor_wrench_fn, split_between_soles
from cutamp.tests.conftest import needs_cuda

BASE = [("x", "X_PRISM"), ("y", "Y_PRISM"), ("z", "Z_PRISM"), ("pitch", "Y_ROT"), ("roll", "X_ROT"), ("yaw", "Z_ROT")]
BASE_JOINTS = [f"base_j_{n}" for n, _ in BASE]
SOLES = ("sole_l", "sole_r")
CROUCH = {"hip": -0.4, "knee": 0.8, "ankle": -0.4}
POSES = [({}, {}), (CROUCH, {}), (CROUCH, {"yaw": 0.3, "pitch": 0.2, "roll": -0.1})]
POSE_IDS = ["straight", "crouch", "tilted_crouch"]


def _cross(a, b):
    return torch.cross(a, b, dim=-1)


def test_split_is_exact_in_total():
    g = torch.Generator().manual_seed(0)
    F, M, P = (torch.randn(8, 3, generator=g, dtype=torch.float64) for _ in range(3))
    s_l = torch.tensor([0.02, 0.10, 0.00], dtype=torch.float64).expand(8, 3)
    s_r = torch.tensor([-0.01, -0.10, 0.03], dtype=torch.float64).expand(8, 3)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)[:, None]
    com = (1 - t) * s_l + t * s_r + torch.tensor([0.01, 0.0, 0.55], dtype=torch.float64)
    f_l, m_l, f_r, m_r = split_between_soles(F, M, P, s_l, s_r, com)
    torch.testing.assert_close(f_l + f_r, F, atol=1e-6, rtol=0)
    torch.testing.assert_close(_cross(s_l - P, f_l) + m_l + _cross(s_r - P, f_r) + m_r, M, atol=1e-6, rtol=0)


def test_split_follows_the_com():
    F = torch.tensor([[1.0, -2.0, 120.0]]).expand(3, 3)
    M = torch.tensor([[1.0, -2.0, 0.5]]).expand(3, 3)
    P = torch.tensor([[0.0, 0.0, 0.6]]).expand(3, 3)
    s_l, s_r = torch.tensor([0.0, 0.1, 0.0]).expand(3, 3), torch.tensor([0.0, -0.1, 0.0]).expand(3, 3)
    com = torch.tensor([[0.02, 0.10, 0.5],    # over the left sole
                        [0.02, 0.00, 0.5],    # midway
                        [0.02, 0.15, 0.5]])   # 5 cm beyond the left sole, off the segment
    f_l, _, f_r, _ = split_between_soles(F, M, P, s_l, s_r, com)
    assert torch.all(f_r[0] == 0)
    torch.testing.assert_close(f_l[1], f_r[1])
    assert torch.all(f_r[2] == 0)
    torch.testing.assert_close(f_l[2], F[2])


def test_base_wrench_inverts_the_joint_map():
    # τ_base by hand: the force along x, y, z, then the moment about P along the pitch, roll and yaw
    # axes in the world, from scipy's rotations rather than base_wrench's formulas. Roll, the middle
    # angle, stays inside the base's limit, clear of the singularity.
    rng = np.random.default_rng(0)
    n = 64
    F, M, P = rng.normal(size=(3, n, 3))
    ang = rng.uniform(-np.pi, np.pi, size=(n, 3))
    ang[:, 1] = rng.uniform(-1.0, 1.0, n)
    ang[:2, 1] = (1.0, -1.0)
    a = np.tile([0.0, 1.0, 0.0], (n, 1))
    b = Rotation.from_euler("Y", ang[:, 0]).apply([1.0, 0.0, 0.0])
    c = Rotation.from_euler("YX", ang[:, :2]).apply([0.0, 0.0, 1.0])
    tau = np.concatenate([F, np.stack([(a * M).sum(1), (b * M).sum(1), (c * M).sum(1)], 1)], 1)
    got = base_wrench(torch.tensor(np.concatenate([P, ang], 1)), torch.tensor(tau))
    for g, want in zip(got, (F, M, P)):
        assert torch.isfinite(g).all()
        np.testing.assert_allclose(g.numpy(), want, atol=1e-5)


def _link(name, mass, com, inertia):
    return (f'<link name="{name}"><inertial><origin xyz="{com}" rpy="0 0 0"/><mass value="{mass}"/>'
            f'<inertia ixx="{inertia[0]}" ixy="0" ixz="0" iyy="{inertia[1]}" iyz="0" izz="{inertia[2]}"/>'
            f'</inertial></link>')


def _biped_cfg(path) -> dict:
    links = ['<link name="world"/>', _link("trunk", 6.0, "0.03 0 0.10", (0.10, 0.08, 0.05))]
    joints = ['<joint name="world_trunk" type="fixed"><parent link="world"/><child link="trunk"/></joint>']
    for s, y in (("l", 0.10), ("r", -0.10)):
        for name, mass, com in ((f"thigh_{s}", 1.5, "0 0 -0.125"), (f"shank_{s}", 1.2, "0 0 -0.125"),
                                (f"foot_{s}", 0.6, "0.03 0 -0.03")):
            links.append(_link(name, mass, com, (0.01, 0.01, 0.005)))
        for joint, parent, child, xyz in ((f"hip_{s}", "trunk", f"thigh_{s}", f"0 {y} -0.10"),
                                          (f"knee_{s}", f"thigh_{s}", f"shank_{s}", "0 0 -0.25"),
                                          (f"ankle_{s}", f"shank_{s}", f"foot_{s}", "0 0 -0.25")):
            joints.append(f'<joint name="{joint}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
                          f'<origin xyz="{xyz}" rpy="0 0 0"/><axis xyz="0 1 0"/>'
                          f'<limit lower="-2.5" upper="2.5" effort="100" velocity="10"/></joint>')
    urdf = path / "biped.urdf"
    urdf.write_text('<robot name="biped">' + "".join(links + joints) + "</robot>")
    extra, parent = {}, "world"
    for k, (n, joint_type) in enumerate(BASE):
        extra[f"base_link_{n}"] = dict(parent_link_name=parent, link_name=f"base_link_{n}", joint_name=f"base_j_{n}",
                                       joint_type=joint_type, fixed_transform=[0, 0, 0, 1, 0, 0, 0],
                                       joint_limits=[-2.0, 2.0], link_mass=0.0)
        parent = f"base_link_{n}"
    extra[parent]["child_link_name"] = "trunk"
    for s in "lr":
        extra[f"sole_{s}"] = dict(parent_link_name=f"foot_{s}", link_name=f"sole_{s}", joint_name=f"sole_j_{s}",
                                  joint_type="FIXED", fixed_transform=[0.03, 0, -0.05, 1, 0, 0, 0], link_mass=0.0)
    names = BASE_JOINTS + [f"{j}_{s}" for s in "lr" for j in ("hip", "knee", "ankle")]
    kinematics = dict(urdf_path=str(urdf), base_link="world", tool_frames=list(SOLES), extra_links=extra,
                      collision_spheres=None, cspace=dict(joint_names=names, default_joint_position=[0.0] * 12,
                                  cspace_distance_weight=[1.0] * 12, null_space_weight=[1.0] * 12))
    return {"robot_cfg": {"kinematics": kinematics, "load_dynamics": True}}


@pytest.fixture(scope="module")
def biped(tmp_path_factory):
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import DeviceCfg
    cfg = _biped_cfg(tmp_path_factory.mktemp("biped"))
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg), device_cfg=DeviceCfg()),
                     compute_com=True, compute_jacobian=True)
    kp = kin.config.kinematics_config
    dyn = Dynamics(DynamicsCfg(kinematics_config=kp, device_cfg=DeviceCfg()))
    dyn.setup_batch_size(1, 1)
    names = list(kp.joint_names)
    soles = tuple(kp.link_name_to_idx_map[s] for s in SOLES)
    fn = make_floor_wrench_fn([names.index(j) for j in BASE_JOINTS], (0, 1), soles, kp.num_links)
    return SimpleNamespace(cfg=cfg, kin=kin, dyn=dyn, names=names, soles=soles, n_links=kp.num_links, fn=fn)


def _q(names, legs=None, base=None) -> torch.Tensor:
    q = torch.zeros(len(names))
    for j, v in (legs or {}).items():
        for s in "lr":
            q[names.index(f"{j}_{s}")] = v
    for j, v in (base or {}).items():
        q[names.index(f"base_j_{j}")] = v
    return q.cuda()


def _inputs(biped, legs, base):
    from curobo.types import JointState
    js = JointState.from_position(_q(biped.names, legs, base).view(1, 1, -1), joint_names=biped.names)
    ks = biped.kin.compute_kinematics(js).clone()  # Kinematics and Dynamics hand back reused buffers
    return js, ks, biped.dyn.compute_inverse_dynamics(js).clone()


@needs_cuda
@pytest.mark.parametrize("legs, base", POSES, ids=POSE_IDS)
def test_second_pass_matches_virtual_work_at_rest(biped, legs, base):
    from curobo._src.geom.transform import torch_quaternion_to_matrix
    from curobo.types import JointState
    js, ks, tau_free = _inputs(biped, legs, base)
    f_ext = biped.fn(js, ks, tau_free)
    tau2 = biped.dyn.compute_inverse_dynamics(js, f_ext=f_ext)[0, 0].clone()
    # Statics without RNEA: gravity's generalized force minus the soles' wrenches through their Jacobians.
    n, h = len(biped.names), 1e-2
    qq = js.position.repeat(2 * n, 1, 1)
    k = torch.arange(n)
    qq[2 * k, 0, k] += h
    qq[2 * k + 1, 0, k] -= h
    z = biped.kin.compute_kinematics(JointState.from_position(qq, joint_names=biped.names)).robot_com[:, 0, 2]
    tau_vw = ks.robot_com[0, 0, 3] * 9.81 * (z[0::2] - z[1::2]) / (2 * h)
    for i, link in enumerate(biped.soles):
        R = torch_quaternion_to_matrix(ks.tool_poses.quaternion[0, 0, i])
        m_i, f_i = R @ f_ext[0, 0, link, :3], R @ f_ext[0, 0, link, 3:]
        J = ks.tool_jacobians[0, 0, i]
        tau_vw = tau_vw - J[:3].T @ f_i - J[3:].T @ m_i
    torch.testing.assert_close(tau2, tau_vw, atol=0.05, rtol=0)
    assert tau2[:6].abs().max() <= 0.05


@needs_cuda
def test_second_pass_leaves_the_base_nothing_in_motion(biped):
    from curobo.types import JointState
    g = torch.Generator(device="cuda").manual_seed(0)
    q = _q(biped.names, *POSES[2]).expand(3, 5, -1).clone()
    js = JointState(position=q, velocity=torch.rand(q.shape, generator=g, device="cuda") * 2 - 1,
                    acceleration=torch.rand(q.shape, generator=g, device="cuda") * 2 - 1,
                    jerk=torch.zeros_like(q), joint_names=biped.names)
    ks = biped.kin.compute_kinematics(js).clone()
    f_ext = biped.fn(js, ks, biped.dyn.compute_inverse_dynamics(js).clone())
    assert f_ext.shape == (3, 5, biped.n_links, 6)
    assert biped.dyn.compute_inverse_dynamics(js, f_ext=f_ext)[..., :6].abs().max() <= 0.05


@needs_cuda
def test_floor_wrench_replays_in_a_cuda_graph(biped):
    js, ks, tau = _inputs(biped, *POSES[1])
    js_b, ks_b, tau_b = _inputs(biped, *POSES[2])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        biped.fn(js, ks, tau)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = biped.fn(js, ks, tau)
    js.position.copy_(js_b.position)
    ks.copy_(ks_b)
    tau.copy_(tau_b)
    graph.replay()
    torch.testing.assert_close(out, biped.fn(js_b, ks_b, tau_b), atol=1e-6, rtol=0)


@needs_cuda
def test_install_floor_wrench_needs_compute_com(biped):
    # A planner built without compute_com (cuRobo's default trajopt transitions) is refused. That the hook lands
    # on every trajopt transition and only there is test_real_legs_motion's, on the real planner.
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg

    cfg = MotionPlannerCfg.create(robot=copy.deepcopy(biped.cfg), self_collision_check=False, device_cfg=DeviceCfg())
    with MotionPlanner(cfg) as mp, pytest.raises(ValueError, match="compute_com"):
        install_floor_wrench(mp, BASE_JOINTS, SOLES)
