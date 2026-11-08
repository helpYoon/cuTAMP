"""Write the cup of ``cup_t1`` (design 7.7), ``cutamp/envs/assets/meshes/cup.obj``: a revolved, watertight solid,
5 cm across like the standing cylinder (the hand spheres open 65.1 mm, and the cup's own 5 mm surface spheres add
1 cm). Its frame is at its centre, z up, like cuRobo's Cylinder.

Usage:
    python -m cutamp.scripts.make_cup_mesh
"""
from pathlib import Path

import trimesh

OUT = Path(__file__).resolve().parents[1] / "envs" / "assets" / "meshes" / "cup.obj"
OUTER_DIAMETER, WALL, HEIGHT, BASE = 0.05, 0.004, 0.10, 0.005  # m


def cup_mesh() -> trimesh.Trimesh:
    """The cup's (radius, z) profile revolved about z, as trimesh makes its cylinder: out along the bottom, up the
    outside, over the rim, down the inside to the base, and in to the axis."""
    r_out, half = OUTER_DIAMETER / 2, HEIGHT / 2
    r_in = r_out - WALL
    return trimesh.creation.revolve(
        [[0, -half], [r_out, -half], [r_out, half], [r_in, half], [r_in, -half + BASE], [0, -half + BASE]]
    )


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cup_mesh().export(OUT)
    print(f"wrote {OUT}")
