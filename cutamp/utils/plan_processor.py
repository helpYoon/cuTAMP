"""The plan ``--save_plan`` writes: :func:`raw_motion_plan` saves a ``solve_curobo`` plan as planned, schema
``"stage3-raw"``, until stage 6's schema-4 export (design 7.6). See its docstring for the layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import torch

from curobo.types import JointState

if TYPE_CHECKING:
    from cutamp.robots.t1 import T1Model


# raw_motion_plan's schema_version: a plan as planned, until stage 6's schema 4.
RAW_SCHEMA_VERSION = "stage3-raw"


def raw_motion_plan(curobo_plan: List[Dict[str, Any]], model: "T1Model") -> Dict[str, Any]:
    """A ``solve_curobo`` plan as planned (decision 9), until stage 6's schema-4 export (design 7.6)::

        {
            "schema_version": "stage3-raw",
            "model": model.name,                     # "real_legs"
            "joint_names": list(model.joint_names),  # the base's six joints included
            "segments": [{
                "arm": "left" | "right" | None,
                "dt": float,                         # s between frames: cuRobo's interpolation_dt
                "position": [T, n],                  # float32, in joint_names order
                "velocity": [T, n],
                "acceleration": [T, n],
                # A segment the balance gate retimed (slowed down or sped up: it sets the pace):
                # the same segment at the motion planner's own timing, as the same three arrays ([T', n] at the
                # same dt); else None.
                "planner_timing": {"position", "velocity", "acceleration"} | None,
            }],
        }

    The joint values are the planner's own; the joints it locks (the head, the fingers) are left out. A segment without native velocities or accelerations raises rather than having them fabricated."""
    names = list(model.joint_names)

    def frames(plan_js: JointState) -> Dict[str, np.ndarray]:
        if plan_js.joint_names is None:
            raise ValueError("Motion plan segment has no joint names, so it cannot be put in the model's order")
        cols = [list(plan_js.joint_names).index(n) for n in names]
        out = {}
        for key in ("position", "velocity", "acceleration"):
            x = getattr(plan_js, key)
            if x is None:
                raise RuntimeError(f"Motion plan segment lacks native joint {key}; refusing to fabricate it")
            # One trajectory: any leading dims are 1.
            out[key] = x.detach().reshape(x.shape[-2:])[:, cols].to(torch.float32).cpu().numpy()
        return out

    segments = []
    for entry in curobo_plan:
        fast = entry.get("planner_timing")
        segments.append({"arm": entry.get("arm"), "dt": float(entry["dt"]), **frames(entry["plan"]),
                         "planner_timing": None if fast is None else frames(fast["plan"])})
    return {"schema_version": RAW_SCHEMA_VERSION, "model": model.name, "joint_names": names, "segments": segments}
