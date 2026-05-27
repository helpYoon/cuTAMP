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
    ValidPush,
)
from cutamp.task_planning.costs import TrajectoryLength


default_constraint_to_mult = {
    KinematicConstraint.type: {"pos_err": 1.0, "rot_err": 5.0},
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
        "com_polygon": 10,  # Penalize COM projection outside the base rectangle
    },
}


default_constraint_to_tol = {
    Collision.type: {"default": 1e-3},
    # Based on cuRobo thresholds: https://github.com/NVlabs/curobo/blob/2fbffc35225398cf9d5f382804faa9de2608753b/src/curobo/wrap/reacher/ik_solver.py#L127-L128
    # Can lower position for high precision, but should be fine for now.
    KinematicConstraint.type: {"pos_err": 0.005, "rot_err": 0.05},
    Motion.type: {"joint_limit": 0.0, "self_collision": 0.0},
    StablePlacement.type: {
        # You might need to add additional tolerances if your support surfaces have different names
        "goal_in_xy": 1e-3,
        "goal_support": 1e-2,
        "platform0_in_xy": 1e-3,
        "platform0_support": 1e-2,
        "floor_in_xy": 1e-3,
        "floor_support": 1e-2,
        "table_in_xy": 1e-3,
        "table_support": 1e-2,
        "shelf_in_xy": 1e-3,
        "shelf_support": 1e-2,
        "stove_in_xy": 1e-3,
        "stove_support": 1e-2,
    },
    ValidPush.type: {"dist_from_button": 0.0},
    # ComPolygon per-conf values are now continuous penalty in m² from
    # compute_com_polygon_penalties (inside_margin=0.02, inside_weight=1.0):
    # at the polygon edge, penalty = inside_weight * inside_margin² = 4e-4.
    # tol 4e-4 ⇒ "at the edge or inside" satisfies; any excursion past
    # the edge fails (1mm outside has penalty ≈ 4.4e-4). "default"
    # applies to every per-conf name (left_q0, right_q1, ...).
    ComPolygon.type: {"default": 4e-4},
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
