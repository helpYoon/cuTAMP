"""Plot the paired T1 grasp-gradient experiment from its recorded model evaluations."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


COLORS = ("#17665f", "#ab4929")
MARKERS = ("o", "D")
INK = "#213834"
MUTED = "#4c615c"
GRID = "#d5dfd9"


def number(value):
    """Three significant figures, keeping small nonzero measurements visible."""
    return f"{value:.3g}".replace("-", "−") if value else "0"


def plot_runs(runs, output):
    if len(runs) != 2:
        raise ValueError("The figure requires two grasps.")
    initial = [r["initial"] for r in runs]
    final = [r["final"] for r in runs]
    names = [r["name"].replace("_", " ") for r in runs]
    start_soles = np.asarray(initial[0]["soles"])
    if not np.allclose(initial[0]["command"], initial[1]["command"], atol=1e-8, rtol=0):
        raise ValueError("The two runs have different initial commands.")
    if not np.allclose(start_soles, initial[1]["soles"], atol=2e-6, rtol=0):
        raise ValueError("The two runs have different initial predicted soles.")

    fig = plt.figure(figsize=(13.2, 9.3), facecolor="white")
    gs = fig.add_gridspec(2, 2, top=.82, bottom=.10, left=.085, right=.96,
                          hspace=.46, wspace=.30, height_ratios=(1.2, .82))
    move = fig.add_subplot(gs[0, 0])
    gradients = fig.add_subplot(gs[0, 1])
    balance = fig.add_subplot(gs[1, 0])
    errors = fig.add_subplot(gs[1, 1])
    fig.text(.085, .962, "How the grasp changes a predicted foot landing", color=INK,
             fontsize=23, weight="bold", va="top")
    cmd = initial[0]["command"]
    fig.text(.085, .912, "Paired T1 left-step test  ·  Frozen landing policy and world model",
             color=INK, fontsize=13, va="top")
    fig.text(.085, .876,
             f"Shared initial foot command in the planted sole frame   "
             f"x = {number(cmd[0]*100)} cm,   y = {number(cmd[1]*100)} cm,   yaw = {number(np.rad2deg(cmd[2]))}°",
             color=MUTED, fontsize=10.5, va="top")

    for ax in (move, balance, errors):
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(labelcolor=MUTED, color=GRID, labelsize=9)
        ax.set_axisbelow(True)

    def heading(ax, title):
        ax.set_title(title, color=INK, fontsize=13, weight="bold", loc="left", pad=11)

    heading(move, "Predicted swing-sole displacement")
    delta = np.asarray([r["soles"][0][:2] for r in final]) - start_soles[0, :2]
    mm = 1000 * delta
    extent = max(np.abs(mm).max(), .01)
    values = np.vstack((np.zeros(2), mm))
    bounds = np.ptp(values, axis=0)
    pad = .18 * extent
    center = (values.min(axis=0) + values.max(axis=0)) / 2
    # Use a common scale so the apparent direction of the command response is preserved.
    span = max(bounds.max() + 2 * pad, .03)
    for j, setter in enumerate((move.set_xlim, move.set_ylim)):
        setter(center[j] - span / 2, center[j] + span / 2)
    move.set_aspect("equal", adjustable="box", anchor="W")
    move.set_xlabel("World forward Δx (mm)", fontsize=10)
    move.set_ylabel("World left Δy (mm)", fontsize=10)
    move.xaxis.set_major_locator(MaxNLocator(nbins=5))
    move.yaxis.set_major_locator(MaxNLocator(nbins=5))
    move.grid(True, color=GRID, lw=.65)
    move.scatter(0, 0, facecolors="white", edgecolors=INK, s=110, linewidths=1.8, zorder=4)
    for i, (x, y) in enumerate(mm):
        move.annotate("", xy=(x, y), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=COLORS[i], lw=1.8, shrinkA=2, shrinkB=2))
        move.scatter(x, y, s=46 if i == 0 else 27, marker=MARKERS[i], color=COLORS[i], zorder=5)
    move.text(1.06, .96, "Relative to shared\ninitial landing", transform=move.transAxes,
              ha="left", va="top", fontsize=9.5, color=MUTED)
    for i, (x, y) in enumerate(mm):
        move.text(1.06, .68 - i * .28, names[i], transform=move.transAxes,
                  fontsize=10, color=COLORS[i], weight="bold", va="top")
        move.text(1.06, .59 - i * .28, f"({number(x)}, {number(y)}) mm",
                  transform=move.transAxes, fontsize=9.5, va="top", color=INK)

    heading(gradients, "Gradient at the shared command")
    gradients.set_axis_off()
    gradients.set_xlim(0, 1)
    gradients.set_ylim(0, 1)
    columns = [.49, .71, .93]
    gradients.text(.01, .95, "Command axis", color=MUTED, fontsize=10, va="center")
    for x, label in zip(columns, ("Forward x", "Lateral y", "Yaw ψ")):
        gradients.text(x, .95, label, color=MUTED, fontsize=10, ha="center", va="center")
    for key, title, units, top in [
        ("dm_du", "CoM margin derivative   ∂m/∂u", "x, y in m/m    ·    ψ in m/rad", .78),
        ("dloss_du", "Balance-loss derivative   ∂L/∂u", "x, y in m²/m    ·    ψ in m²/rad", .32),
    ]:
        gradients.axhline(top + .10, color=GRID, lw=.85)
        gradients.text(.01, top, title, color=INK, fontsize=11, weight="bold", va="center")
        gradients.text(.01, top - .105, units, color=MUTED, fontsize=9, va="center")
        for i in range(2):
            y = top - .22 - i * .095
            gradients.text(.01, y, names[i], color=COLORS[i], weight="bold", fontsize=10, va="center")
            for x, value in zip(columns, initial[i][key], strict=True):
                gradients.text(x, y, number(value), fontsize=11, color=INK,
                               family="DejaVu Sans Mono", va="center", ha="center")

    heading(balance, "CoM clearance at the grasp")
    margins = 100 * np.asarray([[before["margin_m"], after["margin_m"]]
                                for before, after in zip(initial, final, strict=True)])
    low = min(0, margins.min())
    high = max(5, margins.max())
    balance.set_xlim(low - .035*(high-low), high + .13*(high-low))
    balance.set_ylim(-.65, 1.82)
    balance.axvline(5, color=MUTED, ls=(0, (4, 3)), lw=1.1)
    balance.text(5, 1.48, "5 cm margin", color=MUTED, ha="center", fontsize=9)
    for i, (start, end) in enumerate(margins):
        y = 1 - i
        balance.plot([start, end], [y, y], lw=2, color=COLORS[i])
        balance.scatter(start, y, s=105, facecolors="white", edgecolors=COLORS[i],
                        linewidths=1.6, zorder=3)
        balance.scatter(end, y, s=31, marker=MARKERS[i], color=COLORS[i], zorder=4)
        balance.text(.018, y - .30, f"{number(start)} → {number(end)} cm",
                     transform=balance.get_yaxis_transform(), fontsize=9.5, color=COLORS[i])
    balance.set_yticks([1, 0], names)
    balance.grid(axis="x", color=GRID, lw=.55)
    balance.set_xlabel("Distance to nearest support edge (cm)", fontsize=10)

    heading(errors, "Final landing accuracy")
    rows = []
    for i, r in enumerate(final):
        rows += [(i, "XY", max(r["xy_error"])*100, 2, "cm"),
                 (i, "Yaw", max(r["yaw_error_deg"]), 1, "°")]
    max_ratio = max(v / limit for _, _, v, limit, _ in rows)
    errors.set_xlim(-.035, max(1.24, 1.18 * max_ratio))
    errors.set_ylim(-.75, 4.03)
    errors.axvline(1, color=MUTED, ls=(0, (4, 3)), lw=1.1)
    errors.text(1, 3.55, "strict limit", color=MUTED, fontsize=9, ha="center")
    for j, (i, metric, val, limit, unit) in enumerate(rows):
        y = 3 - j
        ratio = val / limit
        errors.hlines(y, 0, ratio, lw=2.4, color=COLORS[i], alpha=.55)
        errors.scatter(ratio, y, marker=MARKERS[i], s=35, color=COLORS[i], zorder=3)
        value_label = f"{val:.5g}" if abs(ratio - 1) < .01 else number(val)
        errors.annotate(f"{value_label}{unit if unit == '°' else ' cm'}", (ratio, y), (6, 0),
                        textcoords="offset points", color=INK, va="center", fontsize=9)
    errors.set_yticks(range(3, -1, -1), [f"{names[i]} · {metric}" for i, metric, *_ in rows])
    errors.set_xlabel("Worst foot error / tolerance    (XY < 2 cm, yaw < 1°)", fontsize=10)
    errors.grid(axis="x", color=GRID, lw=.55)
    fig.text(.085, .029, "Open circle = initial. Colored markers = optimized.  "
             "Accuracy uses the larger of the left and right sole errors at the post-HOLD handoff.", fontsize=9.4, color=MUTED)
    fig.savefig(output, dpi=165, facecolor="white", bbox_inches="tight", pad_inches=.22)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_runs(json.loads(args.scores.read_text())["runs"], args.output)


if __name__ == "__main__":
    main()
