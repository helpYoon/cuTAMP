# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import sys

from cutamp.task_planning.constraints import (
    ComPolygon,
    KinematicConstraint,
    StablePlacement,
    Collision,
    Motion,
)
from cutamp.task_planning.costs import TrajectoryLength
from cutamp.com_polygon_cost import COM_TOL, SOLE_POS_TOL, SOLE_ROT_TOL


default_constraint_to_mult = {
    # sole_*: the real legs' soles against the stance at every conf (design 7.4), weighted as the hand is.
    KinematicConstraint.type: {"pos_err": 1.0, "rot_err": 5.0, "sole_pos_err": 1.0, "sole_rot_err": 5.0},
    StablePlacement.type: {"goal_support": 2.0},
    TrajectoryLength.type: {"traj_length": 1e-3},
    # ComPolygon penalty values are O(1e-4 to 1e-3) in m²; multiplier 10
    # puts the COM gradient contribution to Adam's loss on par with KinematicConstraint
    # pos_err (mult 1.0 × value O(1e-2)). "default" applies to every
    # per-conf name — relies on CostReducer's (type, "default") fallback.
    ComPolygon.type: {"default": 10.0},
    "soft": {
        "dist_from_origin": 5e-1,
        "min_y": 1e-1,
        "max_y": 1e-1,
        "max_obj_dist": 3e-1,
        "min_obj_dist": 1e-1,
        "align_yaw": 5e-2,
        "retract_close_to_home": 1e-1,
        "minimize_body_movement": 10,  # Penalize leg/torso deviation from home
        # Hinge² band penalties are O(1e-3) rad² (0.05 rad intrusion into the
        # 0.1 band ≈ 2.5e-3). Tuned in the Task-6 verification battery
        # (2026-06-11): 1e2 left endpoint offenders; 1e3 cleared every
        # non-trunk endpoint; 1e4 bought nothing more; 1e5 distorted particle
        # selection (minted a new elbow offender). Keep 1e3.
        "joint_limit_margin": 1e3,
    },
}


default_constraint_to_tol = {
    Collision.type: {"default": 1e-3},
    # Based on cuRobo thresholds: https://github.com/NVlabs/curobo/blob/2fbffc35225398cf9d5f382804faa9de2608753b/src/curobo/wrap/reacher/ik_solver.py#L127-L128
    # Can lower position for high precision, but should be fine for now.
    # sole_*: the real legs' soles against the stance (design 7.4), on the hand's tolerances; the balance gate holds
    # every trajectory frame to the same two.
    KinematicConstraint.type: {
        "pos_err": 0.005, "rot_err": 0.05, "sole_pos_err": SOLE_POS_TOL, "sole_rot_err": SOLE_ROT_TOL,
    },
    # retract_*: the real legs' retract confs, which join the motion constraints (design 7.3).
    # trunk_tilt: the Trunk tilt past the envelope's maximum, at every conf, where the envelope sets one (call 22;
    # route A's sets none), in cosine: 1e-6 is 1.2e-6 rad of tilt at a 1.0 rad bound and 17 float32 ulps of its
    # cosine, so the rounding of cos(1.0) does not reject a conf on the bound.
    Motion.type: {
        "joint_limit": 0.0,
        "self_collision": 0.0,
        "retract_joint_limit": 0.0,
        "retract_self_collision": 0.0,
        "trunk_tilt": 1e-6,
    },
    StablePlacement.type: {
        # You might need to add additional tolerances if your support surfaces have different names
        "goal_in_xy": 1e-3,
        "goal_support": 1e-2,
    },
    # ComPolygon per-conf values are continuous penalty in m² from
    # compute_com_polygon_penalties. With the max-over-axes barrier, the
    # on-boundary penalty (edge OR corner) == COM_TOL == inside_weight *
    # inside_margin² = 4e-4, so "at the boundary or inside" satisfies and any
    # excursion past an edge fails. "default" applies to every per-conf name
    # (left_q0, right_q1, ...).
    ComPolygon.type: {"default": COM_TOL},
}


def setup_logging():
    # Define a custom formatter with ANSI escape codes for colors
    class CustomFormatter(logging.Formatter):
        level_colors = {
            "DEBUG": "\033[34m",  # Blue
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset = "\033[0m"

        def format(self, record):
            levelname_color = self.level_colors.get(record.levelname, "")
            levelname = f"{levelname_color}{record.levelname}{self.reset}"
            record.levelname = levelname
            return super().format(record)

    # Define the log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = CustomFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure the root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # Set levels for specific loggers
    logging.getLogger("curobo").setLevel(logging.WARNING)
    logging.getLogger("cutamp").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.DEBUG)
