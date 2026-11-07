"""real_legs_report on hand-built raw plans (cuRobo's RNEA runs, so the GPU)."""
import json

import pytest

from cutamp.tests.conftest import needs_cuda


@needs_cuda
def test_real_legs_report_on_a_still_home_plan(tmp_path):
    # real_legs_report on a raw plan of two still segments at home: cuRobo's torques with the floor's push give
    # design 7.4a's feet-on-the-floor statics (knees -4.57, ankle pitches +3.92 N m), the soles sit on the stance,
    # the Trunk's tilt is home's, and the margins come from overall_metrics. A plan of another schema beside it is
    # not a real-leg one.
    import pickle

    import numpy as np
    import torch

    from cutamp.robots.t1 import REAL_LEGS, trunk_pose_from_q
    from cutamp.scripts import real_legs_report

    names = list(REAL_LEGS.joint_names)
    still = np.tile(np.asarray(REAL_LEGS.home, dtype=np.float32), (3, 1))
    zero = np.zeros_like(still)
    segment = {"arm": None, "dt": 0.025, "position": still, "velocity": zero, "acceleration": zero,
               "planner_timing": None}
    study = tmp_path / "study" / "experiments"
    for exp_id, plan in (("real_legs_h050_s0", {"schema_version": "stage3-raw", "model": "real_legs",
                                                "joint_names": names, "segments": [segment, segment]}),
                         ("other_schema_h050_s0", {"schema_version": 3})):
        (study / exp_id).mkdir(parents=True)
        with open(study / exp_id / "motion_plan.pkl", "wb") as f:
            pickle.dump(plan, f)
    (study / "real_legs_h050_s0" / "overall_metrics.json").write_text(json.dumps(
        {"traj_com_min_margin": 0.03, "traj_zmp_min_margin": 0.02, "retime_max": 1.0}))

    prefix = tmp_path / "out" / "report"
    real_legs_report.main(["--study", str(tmp_path / "study"), "--out", str(prefix)])
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert [r["run"] for r in report["runs"]] == ["real_legs_h050_s0"]
    run, joints = report["runs"][0], {j["joint"]: j for j in report["joints"]}
    assert len(joints) == 27 and not any(j.startswith("base_j_") for j in joints)  # the actuated joints
    for side in ("Left", "Right"):
        assert abs(joints[f"{side}_Knee_Pitch"]["peak_torque"] - 4.57) <= 0.3
        assert abs(joints[f"{side}_Ankle_Pitch"]["peak_torque"] - 3.92) <= 0.3
        assert joints[f"{side}_Knee_Pitch"]["effort"] == pytest.approx(130.5)
        assert joints[f"{side}_Knee_Pitch"]["knee_speed"] == pytest.approx(6.71)
        assert joints[f"{side}_Ankle_Pitch"]["velocity_limit"] == pytest.approx(7.29)  # cuRobo's, not 9.57
    assert joints["Waist"]["knee_speed"] == pytest.approx(1.88)
    assert joints["Waist"]["velocity_limit"] == pytest.approx(14.66)
    assert all(j["peak_speed"] == 0.0 for j in joints.values())
    home_tilt = float(torch.acos(trunk_pose_from_q(REAL_LEGS, REAL_LEGS.home)[2, 2]))
    assert run["max_tilt"] == pytest.approx(home_tilt, abs=1e-4)
    assert run["max_sole_drift_m"] < 1e-5 and run["max_sole_drift_rad"] < 1e-3
    assert (run["segments"], run["slowed"], run["com_min_margin"], run["zmp_min_margin"], run["retime_max"]) == (
        2, 0, 0.03, 0.02, 1.0)
    assert run["unslowed_over_effort"] == []
    md = prefix.with_suffix(".md").read_text()
    assert "| Left_Knee_Pitch |" in md and "| real_legs_h050_s0 |" in md


@needs_cuda
def test_real_legs_report_counts_slowed_segments(tmp_path):
    # Each segment the balance gate slowed keeps the planner's timing, so its s is its frames over the planner's.
    # The report counts them, takes the gate's time scales and binding checks (by check) from overall_metrics, and
    # writes them in its tables.
    import pickle

    import numpy as np

    from cutamp.robots.t1 import REAL_LEGS
    from cutamp.scripts import real_legs_report

    names = list(REAL_LEGS.joint_names)

    def still(n):
        position = np.tile(np.asarray(REAL_LEGS.home, dtype=np.float32), (n, 1))
        return {"position": position, "velocity": np.zeros_like(position), "acceleration": np.zeros_like(position)}

    segments = [{"arm": "left", "dt": 0.025, **still(4), "planner_timing": still(3)},   # slowed, s = 1.5
                {"arm": "left", "dt": 0.025, **still(5), "planner_timing": still(3)},   # slowed, s = 2
                {"arm": "right", "dt": 0.025, **still(3), "planner_timing": None}]       # s = 1
    run = tmp_path / "study" / "real_legs_h000_s0"
    run.mkdir(parents=True)
    with open(run / "motion_plan.pkl", "wb") as f:
        pickle.dump({"schema_version": "stage3-raw", "model": "real_legs", "joint_names": names,
                     "segments": segments}, f)
    (run / "overall_metrics.json").write_text(json.dumps({
        "traj_com_min_margin": 0.03, "traj_zmp_min_margin": 0.02, "retime_min": 1.0, "retime_max": 2.0,
        "retime_binding": {"velocity Waist": 1, "zmp": 1, "torque Left_Knee_Pitch": 1}}))

    prefix = tmp_path / "out" / "report"
    real_legs_report.main(["--study", str(tmp_path / "study"), "--out", str(prefix)])
    report = json.loads(prefix.with_suffix(".json").read_text())
    (r,) = report["runs"]
    assert (r["segments"], r["slowed"], r["retime_min"], r["retime_max"]) == (3, 2, 1.0, 2.0)
    assert r["binding"] == {"torque": 1, "velocity": 1, "zmp": 1}
    assert (report["summary"]["retime_min"], report["summary"]["retime_max"]) == (1.0, 2.0)
    assert r["unslowed_over_effort"] == []  # only the segment at the planner's timing is read for it
    md = prefix.with_suffix(".md").read_text()
    assert "| real_legs_h000_s0 | 3 | 2 | 1.00 | 2.00 | torque 1, velocity 1, zmp 1 |" in md
    assert "time scales from 1.00x to 2.00x" in md
