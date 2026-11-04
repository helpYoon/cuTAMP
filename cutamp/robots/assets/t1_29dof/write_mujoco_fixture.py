"""Write mujoco_fixture.npz: MuJoCo's FK, masses, inertias and soft limits on bfm_wm's T1 (design call 18).

Neither of cuTAMP's Python environments has MuJoCo, so this runs once in bfm_wm's venv, and the tests compare the
planner model with the fixture. It reads bfm_wm's ``mjcf/t1.xml`` (its sha1 must be PROVENANCE.yml's, and the
fixture is stamped with it) and the pinned ``t1_actuators.npz``; it writes nothing into bfm_wm. Run it from cuTAMP:

    env -C <cuTAMP> PYTHONDONTWRITEBYTECODE=1 /home/yoonwoo/bfm_wm/.venv/bin/python \\
        cutamp/robots/assets/t1_29dof/write_mujoco_fixture.py [--bfm_wm DIR]

Keys: ``t1_xml_sha1``; ``body_names`` [30] and ``joint_names`` [29] (T1_JOINT_NAMES order); ``confs`` [22, 29]
(default_q, all-zero, 20 random draws inside the soft limits; the head at 0 in each); ``body_pos`` [22, 30, 3] and
``body_quat`` [22, 30, 4] (wxyz) in the Trunk frame; ``com`` [22, 3]; ``body_mass`` [30]; ``body_inertia``
[30, 3, 3], about each CoM in the body frame; ``soft_lower`` and ``soft_upper`` [29].
"""

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / "mujoco_fixture.npz"
XML = Path("assets/t1/t1_description/mjcf/t1.xml")  # in bfm_wm
HEAD = ("AAHead_yaw", "Head_pitch")
NUM_RANDOM = 20


def _from_t1_file(path: Path) -> tuple[list, float]:
    """T1_JOINT_NAMES and T1_CFG's soft_joint_pos_limit_factor, parsed from bfm_wm's robot/t1.py."""
    names = factor = None
    for node in ast.parse(path.read_text()).body:
        target = node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None
        if target == "T1_JOINT_NAMES":
            names = ast.literal_eval(node.value)
        elif target == "T1_CFG":
            (keyword,) = [k for k in node.value.keywords if k.arg == "soft_joint_pos_limit_factor"]
            factor = ast.literal_eval(keyword.value)
    return names, float(factor)


def _bfm_wm_t1(bfm_wm: Path) -> tuple[list, float, str]:
    """T1_JOINT_NAMES and the soft-limit factor from bfm_wm's ``robot.t1``, imported, or parsed from the file with
    ``ast`` if the import fails; and which of the two it was."""
    sys.path.insert(0, str(bfm_wm))
    try:
        from robot.t1 import T1_CFG, T1_JOINT_NAMES
        return list(T1_JOINT_NAMES), float(T1_CFG.soft_joint_pos_limit_factor), "imported from robot.t1"
    except Exception as e:  # any failure of the Isaac Lab stack it imports
        names, factor = _from_t1_file(bfm_wm / "robot" / "t1.py")
        return names, factor, f"parsed from robot/t1.py (import failed: {type(e).__name__}: {e})"
    finally:
        sys.path.remove(str(bfm_wm))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bfm_wm", type=Path, default=Path("/home/yoonwoo/bfm_wm"), help="bfm_wm's checkout")
    args = parser.parse_args()

    xml_sha1 = hashlib.sha1((args.bfm_wm / XML).read_bytes()).hexdigest()
    pinned = yaml.safe_load((HERE / "PROVENANCE.yml").read_text())["not_copied"]["t1.xml"]["sha1"]
    assert xml_sha1 == pinned, f"bfm_wm's t1.xml is {xml_sha1}, PROVENANCE.yml pins {pinned}"
    t1_joint_names, factor, source = _bfm_wm_t1(args.bfm_wm)
    print(f"soft_joint_pos_limit_factor = {factor!r}, {source}")

    m = mujoco.MjModel.from_xml_path(str(args.bfm_wm / XML))
    d = mujoco.MjData(m)
    joint_names = [m.joint(j).name for j in range(1, m.njnt)]
    assert joint_names == t1_joint_names, joint_names
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and (m.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE).all()
    assert m.nq == 7 + len(joint_names) and (m.jnt_qposadr[1:] == np.arange(7, m.nq)).all()
    body_names = [m.body(b).name for b in range(1, m.nbody)]
    assert body_names[0] == "Trunk" and m.body_jntadr[1] == 0, body_names

    # bfm_wm's soft limits (src/mdp.py:174-177): the MJCF range's midpoint ± factor × its half-width
    lo, hi = m.jnt_range[1:].T
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * factor
    soft_lower, soft_upper = mid - half, mid + half

    default_q = np.load(HERE / "t1_actuators.npz")["default_q"].astype(np.float64)
    rng = np.random.default_rng(0)
    confs = np.concatenate([default_q[None], np.zeros((1, len(joint_names))),
                            rng.uniform(soft_lower, soft_upper, size=(NUM_RANDOM, len(joint_names)))])
    confs[:, [joint_names.index(h) for h in HEAD]] = 0.0

    body_pos = np.zeros((len(confs), len(body_names), 3))
    body_quat = np.zeros((len(confs), len(body_names), 4))
    com = np.zeros((len(confs), 3))
    for k, q in enumerate(confs):
        d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]  # the free joint at the identity: the world frame is the Trunk frame
        d.qpos[7:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        body_pos[k], body_quat[k], com[k] = d.xpos[1:], d.xquat[1:], d.subtree_com[1]
    assert not body_pos[:, 0].any() and (body_quat[:, 0] == [1, 0, 0, 0]).all()

    body_inertia = np.zeros((len(body_names), 3, 3))
    for b in range(1, m.nbody):  # MuJoCo keeps principal moments in a frame turned by body_iquat
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.body_iquat[b])
        rot = rot.reshape(3, 3)
        body_inertia[b - 1] = rot @ np.diag(m.body_inertia[b]) @ rot.T

    np.savez_compressed(
        OUT, t1_xml_sha1=np.array(xml_sha1), body_names=np.array(body_names), joint_names=np.array(joint_names),
        confs=confs, body_pos=body_pos, body_quat=body_quat, com=com, body_mass=m.body_mass[1:].copy(),
        body_inertia=body_inertia, soft_lower=soft_lower, soft_upper=soft_upper)
    print(f"wrote {OUT.name}: {len(confs)} confs, {len(body_names)} bodies, {m.body_mass[1:].sum()!r} kg, "
          f"MuJoCo {mujoco.__version__}")


if __name__ == "__main__":
    main()
