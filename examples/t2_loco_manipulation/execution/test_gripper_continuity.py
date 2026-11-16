import tempfile, unittest
from pathlib import Path
import numpy as np
from compose_reference import compose


class GripperContinuity(unittest.TestCase):
    def test_loaded_stage_cannot_silently_open(self):
        with tempfile.TemporaryDirectory() as td:
            stages = []
            for i, command in enumerate([255.0, 0.0]):
                p = Path(td) / f"{i}.npz"
                np.savez(
                    p,
                    time_s=[0.0, 1.0],
                    qpos=np.zeros((2, 1)),
                    feet=np.zeros((2, 2, 4, 4)),
                    contacts=np.ones((2, 2), bool),
                    normal_caps=np.ones((2, 2)),
                    gripper=np.tile([command, 0.0], (2, 1)),
                )
                stages.append(
                    dict(reference_npz=str(p), label=str(i), robot_qpos_indices=[0])
                )
            with self.assertRaisesRegex(AssertionError, "Gripper stage discontinuity"):
                compose(stages)

    def test_declared_held_stage_stays_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.npz"
            np.savez(
                p,
                time_s=[0.0, 1.0],
                qpos=np.zeros((2, 1)),
                feet=np.zeros((2, 2, 4, 4)),
                contacts=np.ones((2, 2), bool),
                normal_caps=np.ones((2, 2)),
                gripper=[[255.0, 0.0], [0.0, 0.0]],
            )
            with self.assertRaisesRegex(AssertionError, "Held stage opens"):
                compose(
                    [
                        dict(
                            reference_npz=str(p),
                            label="loaded",
                            robot_qpos_indices=[0],
                            required_closed_arm=0,
                        )
                    ]
                )


if __name__ == "__main__":
    unittest.main()
