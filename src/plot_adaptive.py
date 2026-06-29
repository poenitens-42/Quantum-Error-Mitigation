"""
plot_adaptive.py
Plots adaptive ML selector improvement vs linear and richardson baselines.
Reads from results/adaptive_eval.csv (produced by evaluate_adaptive.py).

Generates:
  results/plots/adaptive_vs_baselines_{circuit}.png  — one per circuit
  results/plots/adaptive_summary.png                 — all circuits combined
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/plots", exist_ok=True)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

PAL = {
    "linear":      "#1D9E75",
    "richardson":  "#534AB7",
    "adaptive_ml": "#E91E63",
}
MARKERS = {
    "linear": "^",
    "richardson": "s",
    "adaptive_ml": "*",
}
DASHES = {
    "linear":      (2, 1),
    "richardson":  (1, 0),
    "adaptive_ml": (1, 0),
}
REGIME_BANDS = [
    (0.0,  0.01, "#1D9E75", "low"),
    (0.01, 0.05, "#EF9F27", "mid"),
    (0.05, 1.0,  "#E24B4A", "high"),
]

# ----------------------------
# LOAD
# ----------------------------
df = pd.read_csv("results/adaptive_eval.csv")
circuits = sorted(df["circuit"].unique())

print(f"Loaded {len(df)} rows from adaptive_eval.csv")
print(f"Circuits: {circuits}\n")


def shade_regimes(ax, x_min, x_max):
    y_lo, y_hi = ax.get_ylim()
    label_y = y_lo + (y_hi - y_lo) * 0.96
    for lo, hi, col, lbl in REGIME_BANDS:
        blo = max(lo, x_min)
        bhi = min(hi, x_max)
        if blo < bhi:
            ax.axvspan(blo, bhi, color=col, alpha=0.07, zorder=0)
            ax.text(
                (blo + bhi) / 2, label_y, lbl,
                ha="center", va="top",
                fontsize=8, color=col, alpha=0.8
            )


def save(fig, name):
    path = f"results/plots/{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------
# FIG A — Per-circuit: adaptive vs baselines
# ----------------------------
print("Generating per-circuit adaptive vs baseline plots...")

for circ in circuits:
    sub = df[df["circuit"] == circ].sort_values("p")
    if sub.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 4.5))

    p_vals = sub["p"].values

    # Linear baseline
    ax.plot(p_vals, sub["imp_linear"], color=PAL["linear"],
            marker=MARKERS["linear"], dashes=DASHES["linear"],
            linewidth=1.8, markersize=5, label="Linear ZNE", zorder=3)

    # Richardson baseline
    ax.plot(p_vals, sub["imp_richardson"], color=PAL["richardson"],
            marker=MARKERS["richardson"], dashes=DASHES["richardson"],
            linewidth=1.8, markersize=5, label="Richardson ZNE", zorder=3)

    # Adaptive ML — thicker, prominent
    ax.plot(p_vals, sub["imp_adaptive"], color=PAL["adaptive_ml"],
            marker=MARKERS["adaptive_ml"], dashes=DASHES["adaptive_ml"],
            linewidth=2.8, markersize=8, label="Adaptive ML", zorder=5)

    # Zero line
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

    # Shade regime bands
    ax.set_xlim(p_vals.min() * 0.8, p_vals.max() * 1.1)
    shade_regimes(ax, p_vals.min() * 0.8, p_vals.max() * 1.1)

    ax.set_xlabel("Noise level (p)")
    ax.set_ylabel("Improvement ratio  (err_noisy − err_mit) / err_noisy")
    ax.set_title(f"Adaptive ML vs Baselines — {circ}")
    ax.legend(framealpha=0.4, loc="lower left")

    save(fig, f"adaptive_vs_baselines_{circ}")


# ----------------------------
# FIG B — Summary: all circuits in one figure (2x2 grid)
# ----------------------------
print("\nGenerating summary 2x2 grid...")

n = len(circuits)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
axes = axes.flatten()

for i, circ in enumerate(circuits):
    ax = axes[i]
    sub = df[df["circuit"] == circ].sort_values("p")
    if sub.empty:
        continue

    p_vals = sub["p"].values

    ax.plot(p_vals, sub["imp_linear"],
            color=PAL["linear"], marker=MARKERS["linear"],
            dashes=DASHES["linear"], linewidth=1.6,
            markersize=4, label="Linear ZNE", zorder=3)

    ax.plot(p_vals, sub["imp_richardson"],
            color=PAL["richardson"], marker=MARKERS["richardson"],
            dashes=DASHES["richardson"], linewidth=1.6,
            markersize=4, label="Richardson ZNE", zorder=3)

    ax.plot(p_vals, sub["imp_adaptive"],
            color=PAL["adaptive_ml"], marker=MARKERS["adaptive_ml"],
            dashes=DASHES["adaptive_ml"], linewidth=2.5,
            markersize=7, label="Adaptive ML", zorder=5)

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.set_xlim(p_vals.min() * 0.8, p_vals.max() * 1.1)

    shade_regimes(ax, p_vals.min() * 0.8, p_vals.max() * 1.1)

    ax.set_title(circ)
    ax.set_xlabel("Noise level (p)")
    ax.set_ylabel("Improvement ratio")

    if i == 0:
        ax.legend(framealpha=0.4, fontsize=9)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Adaptive ML Selector vs Individual Baselines — All Circuits",
    fontsize=13, fontweight="bold", y=1.01
)

save(fig, "adaptive_summary")


# ----------------------------
# FIG C — Method selection heatmap
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 3.5))

color_map = {"linear": "#1D9E75", "richardson": "#534AB7"}

for i, circ in enumerate(circuits):
    sub = df[df["circuit"] == circ].sort_values("p")
    for _, row in sub.iterrows():
        color = color_map.get(row["chosen_method"], "#888780")
        ax.scatter(row["p"], i,
                   c=color,
                   s=120, marker="s",
                   edgecolors="white", linewidths=0.3,
                   zorder=3)

ax.set_yticks(range(len(circuits)))
ax.set_yticklabels(circuits)
ax.set_xlabel("Noise level (p)")
ax.set_title("Method Selected by Adaptive ML Selector")

# Fix x-axis to actual data range
p_vals = sorted(df["p"].unique())
ax.set_xlim(min(p_vals) * 0.5, max(p_vals) * 1.15)

# Regime shading — AFTER setting xlim
shade_regimes(ax, min(p_vals) * 0.5, max(p_vals) * 1.15)

# Legend
patches = [
    mpatches.Patch(color="#1D9E75", label="Linear ZNE"),
    mpatches.Patch(color="#534AB7", label="Richardson ZNE"),
]
ax.legend(handles=patches, loc="upper right", framealpha=0.4)

save(fig, "adaptive_selection_heatmap")


# ----------------------------
# SUMMARY STATS
# ----------------------------
print("\n=== ADAPTIVE IMPROVEMENT SUMMARY ===")
cols = ["imp_linear", "imp_richardson", "imp_adaptive"]
summary = df.groupby("circuit")[cols].mean().round(3)
summary["adaptive_beats_linear"]     = summary["imp_adaptive"] > summary["imp_linear"]
summary["adaptive_beats_richardson"] = summary["imp_adaptive"] > summary["imp_richardson"]
print(summary.to_string())

overall_lin  = df["imp_linear"].mean()
overall_rich = df["imp_richardson"].mean()
overall_adap = df["imp_adaptive"].mean()

print(f"\nOverall mean improvement:")
print(f"  Linear     : {overall_lin:.3f}")
print(f"  Richardson : {overall_rich:.3f}")
print(f"  Adaptive ML: {overall_adap:.3f}")

beats_lin  = (df["imp_adaptive"] > df["imp_linear"]).mean()
beats_rich = (df["imp_adaptive"] > df["imp_richardson"]).mean()
beats_both = ((df["imp_adaptive"] > df["imp_linear"]) &
              (df["imp_adaptive"] > df["imp_richardson"])).mean()

print(f"\nAdaptive beats linear     : {beats_lin:.1%} of conditions")
print(f"Adaptive beats richardson : {beats_rich:.1%} of conditions")
print(f"Adaptive beats both       : {beats_both:.1%} of conditions")

print(f"\nAll plots saved to results/plots/")