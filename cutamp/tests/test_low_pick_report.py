import json
import subprocess
import sys

import numpy as np


def test_saved_map_can_score_all_orders_without_the_robot_or_grasp_models(tmp_path):
    from cutamp.reachability import ReachMap

    soles = np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])
    atlas = ReachMap(0.02, metadata={"home_soles": soles.tolist()})
    layers = [
        {"posture": posture, "arm": "left", "points": np.array([[0.29, 0.13, z]]),
         "q": np.zeros((1, 33)), "margin": np.array([0.08])}
        for posture, z in (("standing", 0.41), ("crouch_1", 0.21))
    ]
    atlas.add_stance(soles, layers)
    filename = tmp_path / "map.npz"
    atlas.save(filename)
    output = tmp_path / "report"
    driver = """
import runpy, sys
class ForbiddenImport:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'graspgenx', 'mujoco'} or name in {'cutamp.grasps', 'cutamp.reachability_fk', 'cutamp.samplers'}:
            raise AssertionError('The saved-map query loaded ' + name)
sys.meta_path.insert(0, ForbiddenImport())
runpy.run_module('cutamp.scripts.t1_reachability', run_name='__main__')
"""
    subprocess.run(
        [sys.executable, "-c", driver, "score", "--map", str(filename), "--output", str(output),
         "--center", "0.29", "0.13", "0.21", "--size", "0.01", "0.01", "0.01"],
        check=True,
    )
    result = json.loads((output / "ranking.json").read_text())
    assert {row["order"] for row in result["combined"]} == {"none", "L", "R", "LR", "RL"}
    fixed = next(row for row in result["standing"] if row["order"] == "none")
    assert fixed["distance_m"] > 0.15
    assert result["combined"][0]["order"] == "none"
    assert result["combined"][0]["distance_m"] == 0.0
