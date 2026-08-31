"""Generate the tracked figures for the IBM OM deployment synthesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "figures" / "ibm_om_deployment_failure"
FIGURES.mkdir(parents=True, exist_ok=True)


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#666666"
LIGHT_GRAY = "#D6D8DB"
VERY_LIGHT_GRAY = "#F0F1F2"


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#111111",
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "grid.color": "#E3E4E6",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.16,
    }
)


def finish(fig: plt.Figure, name: str) -> None:
    output = FIGURES / name
    fig.savefig(output, dpi=210)
    svg_path = output.with_suffix(".svg")
    fig.savefig(svg_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def clean_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="semibold",
        va="top",
    )


# ---------------------------------------------------------------------------
# Figure 1: zero-shot deployment versus endpoint-specific recovery.
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.4, 6.2),
    gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.44},
)
fig.suptitle(
    "Deployment and endpoint recovery answer different questions",
    fontsize=16,
    fontweight="semibold",
    y=1.02,
)

ax = axes[0]
labels = ["Common epoch 0", "Deterministic QAT", "Mean-2 endpoint QAT", "Tail-4 endpoint QAT"]
ideal = np.array([94.88, 94.51, 92.74, 93.01])
persistent = np.array([68.998, 70.51, 77.80, 77.968])
y = np.arange(len(labels))[::-1]
for yy, left, right in zip(y, persistent, ideal, strict=True):
    ax.plot([left, right], [yy, yy], color=LIGHT_GRAY, linewidth=4, zorder=1)
ax.scatter(ideal, y, s=72, color=BLUE, marker="o", zorder=3, label="Ideal mapped")
ax.scatter(
    persistent,
    y,
    s=72,
    color=ORANGE,
    marker="s",
    zorder=3,
    label="Persistent P&V mean",
)
for xx, yy in zip(ideal, y, strict=True):
    ax.text(xx + 0.55, yy, f"{xx:.2f}%", va="center", ha="left", color=BLUE)
for xx, yy in zip(persistent, y, strict=True):
    ax.text(xx - 0.55, yy, f"{xx:.2f}%", va="center", ha="right", color=ORANGE)
ax.set_yticks(y, labels)
ax.set_xlim(63, 99)
ax.set_xlabel("MNIST test accuracy (%)")
ax.set_title("Zero-shot mapping and P&V")
clean_axis(ax)
add_panel_label(ax, "A")

ax = axes[1]
recovery_labels = ["Exact 87003 P0", "Four 87004 P0s (mean)"]
before = np.array([59.20, 63.16])
after = np.array([94.14, 93.065])
y = np.arange(len(recovery_labels))[::-1]
for yy, left, right in zip(y, before, after, strict=True):
    ax.plot([left, right], [yy, yy], color=LIGHT_GRAY, linewidth=4, zorder=1)
ax.scatter(before, y, s=72, color=VERMILLION, marker="X", zorder=3, label="Persistent P0")
ax.scatter(after, y, s=72, color=GREEN, marker="D", zorder=3, label="After Adam")
for xx, yy in zip(before, y, strict=True):
    ax.text(xx, yy + 0.11, f"{xx:.2f}%", va="bottom", ha="center", color=VERMILLION)
for xx, yy in zip(after, y, strict=True):
    ax.text(xx, yy + 0.11, f"{xx:.2f}%", va="bottom", ha="center", color=GREEN)
ax.set_yticks(y, recovery_labels)
ax.set_xlim(55, 99)
ax.set_ylim(-0.28, 1.28)
ax.set_xlabel("MNIST test accuracy (%)")
ax.set_title("Endpoint-specific pulse Adam")
clean_axis(ax)
add_panel_label(ax, "B")

fig.legend(
    handles=[
        Line2D([0], [0], color=BLUE, marker="o", linewidth=0, markersize=8, label="Ideal mapped"),
        Line2D([0], [0], color=ORANGE, marker="s", linewidth=0, markersize=8, label="Persistent P&V mean"),
        Line2D([0], [0], color=VERMILLION, marker="X", linewidth=0, markersize=8, label="Persistent P0"),
        Line2D([0], [0], color=GREEN, marker="D", linewidth=0, markersize=8, label="After Adam"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.91),
    ncols=4,
)
fig.subplots_adjust(bottom=0.18, top=0.77)
fig.text(
    0.5,
    0.025,
    "Left: one held assignment and five endpoint seeds per QAT arm.  "
    "Right: privileged adaptation to separately realized persistent states; not zero-shot deployment.",
    ha="center",
    color=GRAY,
    fontsize=9.5,
)
finish(fig, "deployment-ladder.png")


# ---------------------------------------------------------------------------
# Figure 2: apparent acceptance versus persistent correctness.
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.4, 5.8),
    gridspec_kw={"width_ratios": [1.45, 0.9], "wspace": 0.40},
)
fig.suptitle(
    "Near-perfect verify acceptance hides poor persistent programming",
    fontsize=16,
    fontweight="semibold",
    y=1.02,
)

ax = axes[0]
labels = [
    "Apparent accepted",
    "Cell nearest requested code",
    "Persistent inside verify window",
    "W1 logical contrast code correct",
    "W2 logical contrast code correct",
]
values = np.array([99.9567, 49.6306, 35.1477, 33.83, 32.23])
colors = [BLUE, ORANGE, ORANGE, VERMILLION, VERMILLION]
y = np.arange(len(labels))[::-1]
bars = ax.barh(y, values, color=colors, height=0.62, zorder=3)
for bar, value in zip(bars, values, strict=True):
    ax.text(
        value + 1.1,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}%",
        va="center",
        ha="left",
    )
ax.axvline(90, color=GRAY, linestyle=(0, (3, 3)), linewidth=1.2)
ax.text(89.2, 2.0, "90% progression gate", rotation=90, va="center", ha="right", color=GRAY)
ax.set_yticks(y, labels)
ax.set_xlim(0, 106)
ax.set_xlabel("Fraction of programmed cells or logical weights (%)")
ax.set_title("Pooled alpha=0, h=1: 3 assignments x 5 endpoints")
clean_axis(ax)
add_panel_label(ax, "A")

ax = axes[1]
offset = np.array([0, 1, 2, 3])
accept_probability = np.array([27.69, 21.76, 10.57, 3.17])
bars = ax.bar(offset, accept_probability, color=[GREEN, ORANGE, VERMILLION, VERMILLION], width=0.64, zorder=3)
for bar, value in zip(bars, accept_probability, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.8,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
    )
ax.set_xticks(offset, ["On target", "1 code", "2 codes", "3 codes"])
ax.set_ylim(0, 32)
ax.set_ylabel("Chance one apparent draw is accepted (%)")
ax.set_xlabel("Persistent offset from requested target")
ax.set_title("Single-draw observability")
clean_axis(ax, grid_axis="y")
add_panel_label(ax, "B")

fig.text(
    0.5,
    0.02,
    "The verify half-window is 0.5h, while modeled apparent-write noise is 1.41h standard deviation.",
    ha="center",
    color=GRAY,
    fontsize=9.5,
)
fig.subplots_adjust(bottom=0.16, top=0.80)
finish(fig, "pv-observability-gap.png")


# ---------------------------------------------------------------------------
# Figure 3: spacing tradeoff.
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.4, 5.7),
    gridspec_kw={"width_ratios": [1.55, 0.75], "wspace": 0.38},
)
fig.suptitle(
    "Wider spacing improves code labels by erasing weight resolution",
    fontsize=16,
    fontweight="semibold",
    y=1.02,
)

spacing = np.array([1, 2, 4])
ideal = np.array([94.53, 93.09, 87.89])
pv = np.array([74.79, 75.79, 71.76])
code = np.array([49.63, 71.26, 93.87])

ax = axes[0]
series = [
    (ideal, BLUE, "o", "Ideal mapped accuracy"),
    (pv, ORANGE, "s", "Persistent P&V accuracy"),
    (code, GREEN, "^", "Requested cell code correct"),
]
for values, color, marker, label in series:
    ax.plot(spacing, values, color=color, marker=marker, markersize=7, linewidth=2.2, label=label, zorder=3)
    for x, value in zip(spacing, values, strict=True):
        offset_y = 1.15 if label != "Persistent P&V accuracy" else -2.0
        ax.text(x, value + offset_y, f"{value:.1f}%", ha="center", va="center", color=color)
ax.set_xticks(spacing, ["1 delta", "2 delta", "4 delta"])
ax.set_ylim(45, 100)
ax.set_xlabel("Quantization spacing")
ax.set_ylabel("Accuracy or cell-code correctness (%)")
ax.set_title("Accuracy and programming tradeoff")
clean_axis(ax, grid_axis="y")
add_panel_label(ax, "A")

ax = axes[1]
w1_erased = np.array([25.01, 47.62, 77.65])
bars = ax.bar(spacing.astype(str), w1_erased, color=[SKY, ORANGE, VERMILLION], width=0.62, zorder=3)
for bar, value in zip(bars, w1_erased, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.4,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
    )
ax.set_ylim(0, 86)
ax.set_xlabel("Spacing multiplier")
ax.set_ylabel("W1 weights mapped to zero (%)")
ax.set_title("Deterministic resolution loss")
clean_axis(ax, grid_axis="y")
add_panel_label(ax, "B")

fig.legend(
    handles=[
        Line2D([0], [0], color=BLUE, marker="o", linewidth=2.2, label="Ideal mapped accuracy"),
        Line2D([0], [0], color=ORANGE, marker="s", linewidth=2.2, label="Persistent P&V accuracy"),
        Line2D([0], [0], color=GREEN, marker="^", linewidth=2.2, label="Requested cell code correct"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.065),
    ncols=3,
)
fig.subplots_adjust(bottom=0.22, top=0.80)
fig.text(
    0.5,
    0.015,
    "Nominal-bound-Winsorized predecessor aggregate: alpha=0, three assignments, five P&V endpoints each.",
    ha="center",
    color=GRAY,
    fontsize=9.5,
)
finish(fig, "spacing-tradeoff.png")


# ---------------------------------------------------------------------------
# Figure 4: robust QAT improves accuracy without changing the writer.
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.4, 5.8),
    gridspec_kw={"width_ratios": [1.1, 1.15], "wspace": 0.38},
)
fig.suptitle(
    "Persistent-endpoint QAT changes network tolerance, not write fidelity",
    fontsize=16,
    fontweight="semibold",
    y=1.02,
)

arm_labels = ["Epoch 0", "Deterministic", "Mean-2", "Tail-4"]
means = np.array([68.998, 70.51, 77.80, 77.968])
mins = np.array([55.33, 60.54, 72.86, 73.02])
maxs = np.array([77.83, 78.34, 83.30, 84.35])
colors = [GRAY, BLUE, ORANGE, GREEN]

ax = axes[0]
x = np.arange(len(arm_labels))
ax.bar(x, means, color=colors, width=0.66, zorder=3)
ax.errorbar(
    x,
    means,
    yerr=np.vstack([means - mins, maxs - means]),
    fmt="none",
    ecolor="#333333",
    elinewidth=1.5,
    capsize=5,
    zorder=4,
)
for xx, mean, low, high in zip(x, means, mins, maxs, strict=True):
    ax.text(xx, mean + 1.2, f"{mean:.2f}%", ha="center", va="bottom")
    ax.text(xx, low - 1.0, f"min {low:.2f}", ha="center", va="top", color=GRAY, fontsize=9)
ax.set_xticks(x, arm_labels)
ax.set_ylim(50, 88)
ax.set_ylabel("Final persistent test accuracy (%)")
ax.set_title("Mean with five-seed min/max")
clean_axis(ax, grid_axis="y")
add_panel_label(ax, "A")

epoch0 = {
    "Window success": 35.102,
    "Cell code correct": 49.586,
    "Residual RMSE": 0.0554121,
}
raw = {
    "Epoch 0": [35.102, 49.586, 0.0554121],
    "Deterministic": [35.0985, 49.5919, 0.05539594],
    "Mean-2": [35.0850, 49.5422, 0.05543268],
    "Tail-4": [35.0717, 49.5288, 0.05543663],
}

ax = axes[1]
metric_names = list(epoch0)
markers = ["o", "s", "^"]
metric_colors = [BLUE, ORANGE, VERMILLION]
y = np.arange(len(arm_labels))[::-1]
for metric_idx, (metric, marker, color) in enumerate(zip(metric_names, markers, metric_colors, strict=True)):
    ratios = np.array([raw[arm][metric_idx] / epoch0[metric] * 100 for arm in arm_labels])
    ax.plot(ratios, y, marker=marker, color=color, linewidth=1.5, markersize=7, label=metric, zorder=3)
ax.axvline(100, color=GRAY, linewidth=1.1, linestyle=(0, (3, 3)))
ax.axvspan(99.8, 100.2, color=VERY_LIGHT_GRAY, zorder=0)
ax.set_yticks(y, arm_labels)
ax.set_xlim(99.75, 100.18)
ax.set_xlabel("Programmer metric relative to epoch 0 (%)")
ax.set_title("Physical metrics remain within 0.12%")
clean_axis(ax)
add_panel_label(ax, "B")

fig.legend(
    handles=[
        Line2D([0], [0], color=BLUE, marker="o", linewidth=1.5, label="Window success"),
        Line2D([0], [0], color=ORANGE, marker="s", linewidth=1.5, label="Cell code correct"),
        Line2D([0], [0], color=VERMILLION, marker="^", linewidth=1.5, label="Residual RMSE"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.75, 0.055),
    ncols=3,
)
ax.set_xticks([99.8, 99.9, 100.0, 100.1])
fig.subplots_adjust(bottom=0.22, top=0.80)
fig.text(
    0.5,
    0.015,
    "The robust arms improve endpoint accuracy while persistent-window success, code correctness, and residual RMSE remain effectively fixed.",
    ha="center",
    color=GRAY,
    fontsize=9.5,
)
finish(fig, "robust-qat-tolerance-not-writer.png")


# ---------------------------------------------------------------------------
# Figure 5: W1 lives near the physical noise floor.
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.4, 5.7),
    gridspec_kw={"width_ratios": [0.9, 1.3], "wspace": 0.38},
)
fig.suptitle(
    "Most first-layer weights live at the programming-error scale",
    fontsize=16,
    fontweight="semibold",
    y=1.02,
)

ax = axes[0]
occupancy = np.array([24.64, 39.97, 20.15, 15.24])
occupancy_labels = ["Level 0", "Level 1", "Level 2", "Level >=3"]
bottom = 0.0
stack_colors = [GRAY, BLUE, SKY, ORANGE]
for value, label, color in zip(occupancy, occupancy_labels, stack_colors, strict=True):
    ax.bar([0], [value], bottom=bottom, color=color, width=0.62, label=label, zorder=3)
    ax.text(0, bottom + value / 2, f"{label}\n{value:.1f}%", ha="center", va="center", color="white" if color != SKY else "#222222")
    bottom += value
ax.set_xlim(-0.65, 0.65)
ax.set_ylim(0, 100)
ax.set_xticks([])
ax.set_ylabel("Share of W1 logical weights (%)")
ax.set_title("Mean-2 requested-level occupancy")
clean_axis(ax, grid_axis="y")
add_panel_label(ax, "A")

ax = axes[1]
levels = np.array([0, 1, 2, 3])
intended = levels.astype(float)
error_rms = np.array([0.10315, 0.10919, 0.11442, 0.11809]) / 0.0949
ax.plot(levels, intended, color=BLUE, marker="o", linewidth=2.2, markersize=7, label="Intended contrast magnitude")
ax.plot(levels, error_rms, color=VERMILLION, marker="s", linewidth=2.2, markersize=7, label="Persistent contrast-error RMS")
for xx, value in zip(levels, error_rms, strict=True):
    ax.text(xx, value + 0.12, f"{value:.2f}", ha="center", va="bottom", color=VERMILLION)
ax.axvspan(-0.2, 1.2, color=VERY_LIGHT_GRAY, zorder=0)
ax.text(0.5, 2.72, "64.6% of W1 weights", ha="center", color=GRAY)
ax.set_xticks(levels, ["0", "1", "2", "3"])
ax.set_xlim(-0.25, 3.25)
ax.set_ylim(0, 3.25)
ax.set_xlabel("Requested logical level")
ax.set_ylabel("Magnitude in logical code steps")
ax.set_title("Signal versus persistent error")
ax.legend(loc="upper left")
clean_axis(ax, grid_axis="y")
add_panel_label(ax, "B")

fig.text(
    0.5,
    0.015,
    "Final mean-2 W1 replay on assignment 87003. One logical step is 2h = 0.0949 full-G contrast.",
    ha="center",
    color=GRAY,
    fontsize=9.5,
)
fig.subplots_adjust(bottom=0.16, top=0.80)
finish(fig, "w1-programming-noise-floor.png")


# ---------------------------------------------------------------------------
# Figure 6: both signed contrast and passive loading are perturbed.
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), gridspec_kw={"wspace": 0.34})
fig.suptitle(
    "P&V perturbs both signed contrast and passive loading",
    fontsize=16,
    fontweight="semibold",
    y=1.02,
)

layers = ["W1", "W2"]
x = np.arange(2)
width = 0.34

ax = axes[0]
ideal_contrast = np.array([0.18422, 0.39636])
contrast_error = np.array([0.11089, 0.12128])
bars1 = ax.bar(x - width / 2, ideal_contrast, width, color=BLUE, label="Ideal contrast RMS", zorder=3)
bars2 = ax.bar(x + width / 2, contrast_error, width, color=VERMILLION, label="Contrast-error RMS", zorder=3)
ax.axhline(0.0949, color=GRAY, linestyle=(0, (3, 3)), linewidth=1.2)
ax.text(1.42, 0.098, "one logical step", ha="right", va="bottom", color=GRAY)
for bars in (bars1, bars2):
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom")
for xx, ratio in zip(x, contrast_error / ideal_contrast * 100, strict=True):
    ax.text(xx + width / 2, contrast_error[xx] / 2, f"{ratio:.1f}%\nof signal", ha="center", va="center", color="white", fontsize=9)
ax.set_xticks(x, layers)
ax.set_ylim(0, 0.46)
ax.set_ylabel("Full-G contrast units")
ax.set_title("Signed logical contrast")
ax.legend(loc="upper left")
clean_axis(ax, grid_axis="y")
add_panel_label(ax, "A")

ax = axes[1]
ideal_load = np.array([1.8590, 2.2554])
load_error = np.array([0.2290, 0.2280])
bars1 = ax.bar(x - width / 2, ideal_load, width, color=PURPLE, label="Ideal mean quad load", zorder=3)
bars2 = ax.bar(x + width / 2, load_error, width, color=ORANGE, label="Loading-error RMS", zorder=3)
for bars in (bars1, bars2):
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.065, f"{value:.3f}", ha="center", va="bottom")
for xx, ratio in zip(x, load_error / ideal_load * 100, strict=True):
    ax.text(xx + width / 2, load_error[xx] / 2, f"{ratio:.1f}%", ha="center", va="center", color="#222222", fontsize=9)
ax.set_xticks(x, layers)
ax.set_ylim(0, 2.65)
ax.set_ylabel("Normalized full-G load units")
ax.set_title("Four-cell dissipative loading")
ax.legend(loc="upper left")
clean_axis(ax, grid_axis="y")
add_panel_label(ax, "B")

fig.text(
    0.5,
    0.015,
    "Final mean-2 replay on assignment 87003, averaged across five persistent endpoints. W1 contains 39,200 of 39,700 logical weights.",
    ha="center",
    color=GRAY,
    fontsize=9.5,
)
fig.subplots_adjust(bottom=0.16, top=0.80)
finish(fig, "contrast-and-loading-errors.png")


print(f"wrote 6 PNG/SVG figure pairs to {FIGURES}")
