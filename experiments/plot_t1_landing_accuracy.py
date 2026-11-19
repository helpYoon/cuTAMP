"""Plot the observed accuracy of sampled single steps and ordered pairs from the T1 world model."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutamp.landing_accuracy import handoff_precision


def swing_displacement_cm(stance, side, command):
    """Requested change from the moving sole, expressed forward/outward along the planted sole's axes."""
    swing = int(side == "R")
    support = stance[1 - swing]
    co, si = np.cos(support[2]), np.sin(support[2])
    rotation = np.array([[co, -si], [si, co]])
    previous = rotation.T @ (stance[swing, :2] - support[:2])
    offset = (command[:2] - previous) * [100, -100 if swing else 100]
    stance[swing, :2] = support[:2] + rotation @ command[:2]
    stance[swing, 2] = support[2] + command[2]
    return offset


def main():
    default_dir = Path("data/t1-reachability/landing-accuracy")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=default_dir / "model-training-draws.json")
    parser.add_argument("--output", type=Path, default=default_dir / "model-landing-accuracy.png")
    args = parser.parse_args()
    data = json.loads(args.scores.read_text())
    lags = data["provenance"]["lags"]
    if len(lags) != 3:
        raise ValueError("This figure compares three actuator-delay settings.")

    styles = [
        {"marker": "x", "color": "#929da6", "size": 21, "name": "0 of 3"},
        {"marker": "o", "color": "#3f77aa", "size": 31, "name": "1 of 3"},
        {"marker": "^", "color": "#c56425", "size": 41, "name": "2 of 3"},
        {"marker": "*", "color": "#137757", "size": 105, "name": "3 of 3"},
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.765, bottom=0.14, wspace=0.13, hspace=0.38)
    fig.text(0.11, 0.965, "Which foot commands land within 2 cm and 1°?", fontsize=20, weight="bold")
    fig.text(
        0.11, 0.918, "T1 world model from the home stance · each mark is one sampled command sequence", fontsize=11
    )
    fig.text(0.11, 0.858, "Delay settings passed", fontsize=10, weight="bold")
    legend = [
        Line2D(
            [], [], linestyle="", marker=s["marker"], color=s["color"], markersize=np.sqrt(s["size"]), label=s["name"]
        )
        for s in styles
    ]
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(0.278, 0.866), ncol=4, frameon=False)

    headings = ["L  ·  left step", "R  ·  right step", "LR  ·  left, then right", "RL  ·  right, then left"]
    for ax, order, title in zip(axes.flat, ("L", "R", "LR", "RL"), headings, strict=True):
        rows = [e for e in data["schedules"] if e["order"] == order]
        coords, passed = [], []
        for row in rows:
            stance = np.array(data["home_soles"])
            for side, cmd in zip(order, row["commands"], strict=True):
                delta = swing_displacement_cm(stance, side, np.asarray(cmd))
            coords.append(delta)
            assert [t["lag"] for t in row["trials"]] == lags
            outcomes = [handoff_precision(t.get("stages"), order) for t in row["trials"]]
            if None in outcomes:
                raise ValueError(f"Incomplete recorded handoffs for {order}/{row['id']}")
            passed.append(sum(outcomes))
        xy, n = np.array(coords), np.array(passed)
        for count, s in enumerate(styles):
            x, y = xy[n == count].T
            ax.scatter(
                x,
                y,
                marker=s["marker"],
                s=s["size"],
                color=s["color"],
                linewidths=0.85,
                alpha=0.65 if count == 0 else 0.95,
                zorder=count + 3,
            )
        ax.set_title(title, loc="left", weight="bold", fontsize=12, pad=8)
        ax.set_title(f"{sum(n == 3)}/{len(n)} passed all three", loc="right", fontsize=9, pad=8)
        ax.axvline(0, color="#aab4b9", lw=0.8)
        ax.axhline(0, color="#aab4b9", lw=0.8)
        ax.grid(True, color="#dde2e4", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#bbc4c7")
    for ax in axes[1]:
        ax.set_xlabel("Last swing sole's forward displacement (cm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Outward displacement (cm)")
    fig.text(
        0.11,
        0.036,
        "Both soles must track each issued target and the original final stance. Yaw varies across samples.\n"
        "The lower panels show the second foot’s displacement and score the entire two-step sequence.",
        fontsize=9,
        color="#39434a",
    )
    fig.savefig(args.output, dpi=155, bbox_inches="tight", pad_inches=0.16, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
