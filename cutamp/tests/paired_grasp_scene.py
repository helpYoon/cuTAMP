"""Reproduce a matched grasp, registered standing state and low manipulation scene."""

import json
from pathlib import Path

import torch
from curobo.scene import Cuboid
from curobo.types import DeviceCfg

from cutamp.envs import TAMPEnvironment
from cutamp.grasps import GraspSet
from cutamp.landing_predictor import LandingPredictor
from cutamp.robots.t1 import get_t1_tool_from_ee
from cutamp.step_planner import StepInputs
from cutamp.t1_domain import all_t1_fluents
from cutamp.utils.common import mat4x4_to_grasp_7dof

FIXTURE = Path(__file__).parent / "fixtures/t1_paired_grasps.json"
ROOT = Path(__file__).resolve().parents[2]


def paired_setup(label, *, place=False):
    from cutamp.step_validation import StepStateAdapter

    dc = DeviceCfg()
    inp = json.loads(FIXTURE.read_text())
    ws = dc.to_device(inp["world_object"])
    world_wrist = ws @ dc.to_device(inp["object_to_wrist"][label])
    ws = ws.clone()
    ws[2, 3] = 0.0365  # shipped scenes give the 5 mm object spheres 1 mm floor clearance
    xyz = ws[:3, 3].tolist()
    part = Cuboid(name="part", pose=xyz + [1.0, 0, 0, 0], dims=[0.075, 0.015, 0.061])
    floor = Cuboid(name="floor", pose=[0.0, 0.0, -0.101, 1.0, 0, 0, 0], dims=[3.0, 3.0, 0.20])
    # A destination patch is a surface label; its top coincides with the floor.
    dest = [xyz[0] - 0.03, xyz[1] + 0.025, xyz[2] + 0.001, 0]
    patch = Cuboid(name="pad", pose=[*dest[:2], -0.005, 1.0, 0, 0, 0], dims=[0.19, 0.11, 0.010])
    goal = next(f for f in all_t1_fluents if f.name == ("On" if place else "LeftHolding"))
    statics = [floor, patch] if place else [floor]
    env = TAMPEnvironment(
        "paired_ground_pick",
        [part],
        statics,
        {"Movable": [part], "Surface": [patch] if place else []},
        frozenset([goal.ground("part", "pad") if place else goal.ground("part")]),
    )
    obj_grasp = torch.linalg.inv(ws) @ world_wrist @ torch.linalg.inv(get_t1_tool_from_ee(dc)["left_base_link"])
    grasps = GraspSet(mat4x4_to_grasp_7dof(obj_grasp)[None], dc.to_device([1.0]))
    bundle = ROOT / "data/t1-reachability/landing-accuracy/landing-seed3.pt"
    predictor = LandingPredictor(bundle)
    d = torch.zeros(1, 38, device="cuda")
    d[:, 0] = 0.25 / 0.45
    d[:, -5:] = -1.0
    start = predictor.standing_start(d)
    q, _ = StepStateAdapter(predictor).planner(start.tensors[:6])
    orders = ("L", "none") if place else ("L",)
    inp_args = StepInputs(
        grasps={"part": grasps},
        orders=orders,
        arms=("left",) * len(orders),
        commands=((((0.12, 0.21, 0.0),), ()) if place else (((0.12, 0.21, 0.0),),)),
        initial_state=start,
        posture_seeds=(inp["seed"],) * len(orders),
        placements={"pose1": dest} if place else None,
    )
    return env, q[0].tolist(), inp_args
