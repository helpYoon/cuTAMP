"""Plot the selected particles from two matched public run_cutamp reports."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import yaml


def foot_polygons(stance, dimensions):
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.asarray(dimensions) / 2
    for x, y, psi in stance:
        c, s = np.cos(psi), np.sin(psi)
        # Draw x forward horizontally and y left vertically, in world centimetres.
        yield 100 * ([x, y] + corners @ np.array([[c, s], [-s, c]]))


def plot(near, far, out):
    runs = [json.loads(p.read_text()) for p in (near, far)]
    assert all(r["model_feasible"] and r["found_solution"] for r in runs)
    gaps = [r["gaps"][0] for r in runs]
    assert gaps[0]["seed_commands"] == gaps[1]["seed_commands"]
    np.testing.assert_array_equal(gaps[0]["initial_achieved"], gaps[1]["initial_achieved"])
    out.mkdir(parents=True, exist_ok=True)
    colors = ["#126f7a", "#ba4f2c"]
    cfg = Path(__file__).resolve().parents[1] / "cutamp/robots/assets/t1_29dof/t1_29dof.yml"
    dimensions = yaml.safe_load(cfg.read_text())["cutamp"]["sole_box"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 0.93]})
    fig.suptitle("The grasp changes where the robot steps to center its mass", fontsize=15, weight="bold")
    for ax, label, gap, color in zip(axes, ("Near grasp", "Far grasp"), gaps, colors):
        soles = list(foot_polygons(gap["achieved"], dimensions))
        vertices = np.vstack(soles)
        hull = vertices[ConvexHull(vertices).vertices]
        ax.fill(*hull.T, facecolor=color, alpha=0.11, edgecolor="none")
        ax.plot(*np.vstack([hull, hull[:1]]).T, c=color, lw=1.2)
        for foot, side in zip(soles, ["L", "R"]):
            ax.fill(*foot.T, facecolor=color, alpha=0.19, edgecolor=color)
            mid = foot.mean(0)
            ax.text(*mid, side, ha="center", va="center", color="0.32", fontsize=9)
        center = 100 * np.asarray(gap["hull_centroid_xy"])
        com = 100 * np.asarray(gap["com_xy"])
        ax.plot(*np.stack([center, com]).T, color=color, lw=2)
        ax.scatter(*center, s=100, marker="+", linewidths=2.4, c="0.10", label="Area centroid", zorder=4)
        ax.scatter(*com, s=48, c=color, label="Projected CoM", zorder=5)
        ax.set(aspect="equal", xlim=(-9, 36), ylim=(-19, 23), xlabel="World x (cm)", title=label)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_ylabel("World y (cm)")
    ax = axes[-1]
    for gap, color in zip(gaps, colors):
        d = 100 * np.array([gap["initial_center_distance_m"], gap["center_distance_m"]])
        ax.plot([0, 1], d, "o-", lw=2, color=color)
        for x, y in enumerate(d):
            ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", color=color)
    heights = [100 * g[k] for g in gaps for k in ("initial_center_distance_m", "center_distance_m")]
    ax.set(
        xticks=[0, 1],
        xticklabels=["Same foot seed", "Optimized"],
        xlim=(-0.28, 1.28),
        ylim=(min(heights) - 0.9, max(heights) + 1.25),
        title="Distance to centroid",
        ylabel="Distance (cm)",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(axis.get_title(), fontsize=12, weight="semibold", pad=12)
        axis.grid(lw=0.5, alpha=0.22)
        axis.set_axisbelow(True)
    fig.savefig(out / "run-cutamp-grasps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (out / "run-cutamp-grasps.json").write_text(json.dumps({"near": runs[0], "far": runs[1]}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("near", type=Path)
    parser.add_argument("far", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    plot(args.near, args.far, args.out)
