"""Use cuRobo's T1 model to check the saved grasp postures from the paired experiment.

Run with cuTAMP's GPU Python. The checks use empty-hand robot geometry at the
manipulation keyframe and include both locked gripper meshes against the floor.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cutamp.reachability_fk import T1MapSampler, _sole_transforms, _support_planes

DEFAULT_INPUT = ROOT / "data/t1-reachability/grasp-landing/paired-model.json"


def physical_margin(foot_poses, sole_box, com):
    planes = _support_planes(foot_poses, sole_box)
    return float(-(planes[:, :2] @ com[:2] + planes[:, 2]).max())


@torch.no_grad()
def measure(paired):
    robot = T1MapSampler(samples=1)
    rows = [(run, phase, run[phase]) for run in paired["runs"] for phase in ("initial", "final")]
    postures = np.stack([r["posture"] for _, _, r in rows])
    fk = robot._fk(robot.kin, robot.cfg.to_device(postures))
    feet = robot._feet(fk).cpu().numpy()
    robot_com = fk.robot_com.reshape(len(rows), 4)[:, :3].cpu().numpy()
    wrists = (
        fk.tool_poses.get_link_pose("left_base_link", make_contiguous=True).get_matrix().reshape(-1, 4, 4).cpu().numpy()
    )
    spheres = fk.robot_spheres
    robot.self_cost.setup_batch_tensors(len(rows), 1)
    collisions = robot.self_cost.forward(spheres).reshape(len(rows), -1).amax(1)
    floor = (spheres[..., 2] - spheres[..., 3]).reshape(len(rows), -1).amin(1)
    gripper_floor = {side: robot._gripper_clearance(fk, side) for side in ("left", "right")}
    output = []
    for i, (run, phase, row) in enumerate(rows):
        wrist_goal = np.array(run["wrist_world"])
        planar_feet = _sole_transforms(row["soles"])
        q = postures[i]
        joint_bound_violation = float(max(0.0, np.max(q - robot.limits[1]), np.max(robot.limits[0] - q)))
        com = robot_com[i]
        configured_margin = physical_margin(planar_feet, robot.sole_box, com)
        actual_margin = physical_margin(feet[i], robot.sole_box, com)
        floor_mesh = {side: float(height[i]) for side, height in gripper_floor.items()}
        wrist_error = np.linalg.norm(wrists[i, :3, 3] - wrist_goal[:3, 3])
        wrist_angle = Rotation.from_matrix(wrists[i, :3, :3] @ wrist_goal[:3, :3].T).magnitude()
        sole_errors = np.linalg.norm(feet[i, :, :3, 3] - planar_feet[:, :3, 3], axis=1)
        sole_angles = Rotation.from_matrix(feet[i, :, :3, :3] @ planar_feet[:, :3, :3].transpose(0, 2, 1)).magnitude()
        output.append(
            {
                "grasp": run["name"],
                "stage": phase,
                "joint_bound_violation": joint_bound_violation,
                "self_collision_penalty": float(collisions[i]),
                "min_sphere_height_m": float(floor[i]),
                "min_gripper_mesh_height_m": floor_mesh,
                "world_com_m": com.tolist(),
                "com_disagreement_m": float(np.linalg.norm(com - row["com"])),
                "wrist_position_error_m": float(wrist_error),
                "wrist_rotation_error_rad": float(wrist_angle),
                "sole_position_errors_m": sole_errors.tolist(),
                "sole_rotation_errors_rad": sole_angles.tolist(),
                "planned_stance_margin_m": configured_margin,
                "fk_stance_margin_m": actual_margin,
                "kinematic_pose_pass": bool(
                    wrist_error < 0.005
                    and wrist_angle < 0.05
                    and sole_errors.max() < 0.001
                    and sole_angles.max() < 0.01
                    and joint_bound_violation < 1e-7
                ),
                "geometry_pass": bool(
                    float(collisions[i]) == 0 and float(floor[i]) >= 0 and min(floor_mesh.values()) >= 0
                ),
                "physical_margin_pass": bool(actual_margin >= 0.05),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = args.input.read_bytes()
    rows = measure(json.loads(data))
    output = args.output if args.output is not None else args.input.with_name("keyframes.json")
    report = {
        "input_file": args.input.name,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "device": torch.cuda.get_device_name(),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(
            f"{row['grasp']} {row['stage']}: FK margin={1000 * row['fk_stance_margin_m']:.3f} mm, "
            f"geometry_pass={row['geometry_pass']}, pose_pass={row['kinematic_pose_pass']}"
        )


if __name__ == "__main__":
    main()
