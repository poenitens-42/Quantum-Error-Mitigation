"""
plot_results.py
Publication-grade figures for regime-aware QEM thesis.

Fixes vs original:
  - shade_regimes now called after data is plotted (correct ylim)
  - depth_map includes depth_d1
  - regime label positioning uses data range not ylim
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

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
    "noisy":       "#888780",
    "richardson":  "#534AB7",
    "linear":      "#1D9E75",
    "cdr":         "#D85A30",
    "adaptive":    "#F39C12",
    "adaptive_ml": "#E91E63",
}
MARKERS = {
    "noisy": "o", "richardson": "s", "linear": "^",
    "cdr": "D", "adaptive": "P", "adaptive_ml": "*",
}
DASHES = {
    "noisy": (4,2), "richardson": (1,0), "linear": (2,1),
    "cdr": (3,1,1,1), "adaptive": (1,1), "adaptive_ml": (1,0),
}
REGIME_BANDS = [
    (0.0,  0.01, "#1D9E75", "low"),
    (0.01, 0.05, "#EF9F27", "mid"),
    (0.05, 1.0,  "#E24B4A", "high"),
]

OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("results/raw/results.csv")
try:
    df_pair = pd.read_csv("results/raw/pairwise.csv")
    HAS_PAIR = True
except FileNotFoundError:
    HAS_PAIR = False

df = df.drop_duplicates()
if "impr" in df.columns:
    df["impr"] = df["impr"].clip(-2, 2)
df["err"] = df["err"].clip(0, 10)
df = df.groupby(["circuit","model","p","method","regime"], as_index=False).mean()

circuits = sorted(df["circuit"].unique())
methods  = [m for m in ["noisy","richardson","linear","cdr","adaptive","adaptive_ml"]
            if m in df["method"].unique()]
models   = df["model"].unique()


def shade_regimes(ax, x_min, x_max):
    """Call AFTER plotting data so ylim is set correctly."""
    y_lo, y_hi = ax.get_ylim()
    label_y = y_lo + (y_hi - y_lo) * 0.95
    for lo, hi, col, lbl in REGIME_BANDS:
        blo = max(lo, x_min)
        bhi = min(hi, x_max)
        if blo < bhi:
            ax.axvspan(blo, bhi, color=col, alpha=0.07, zorder=0)
            ax.text((blo + bhi) / 2, label_y, lbl,
                    ha="center", va="top", fontsize=8, color=col, alpha=0.7)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/{name}.png")


# ── Fig 1 — Error vs Noise ────────────────────────────────────────────────────
print("Generating Fig 1: error vs noise...")
for model in models:
    for circ in circuits:
        sub = df[(df["circuit"] == circ) & (df["model"] == model)].sort_values("p")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        p_vals = sorted(sub["p"].unique())
        for meth in methods:
            ms = sub[sub["method"] == meth].sort_values("p")
            if ms.empty:
                continue
            ax.plot(ms["p"], ms["err"], color=PAL[meth], marker=MARKERS[meth],
                    dashes=DASHES[meth], linewidth=1.6, markersize=5,
                    label=meth.capitalize(), zorder=3)
            ax.fill_between(ms["p"], ms["ci_lo"], ms["ci_hi"],
                            color=PAL[meth], alpha=0.12, zorder=2)
        ax.set_xlim(min(p_vals)*0.8, max(p_vals)*1.1)
        ax.set_xlabel("Noise level (p)")
        ax.set_ylabel("|ideal − mitigated|")
        ax.set_title(f"Error vs noise — {circ} ({model})")
        ax.legend(framealpha=0.4)
        shade_regimes(ax, min(p_vals)*0.8, max(p_vals)*1.1)  # AFTER plotting
        save(fig, f"fig1_error_{circ}_{model}")


# ── Fig 2 — Improvement ratio ─────────────────────────────────────────────────
print("Generating Fig 2: improvement ratio...")
for model in models:
    for circ in circuits:
        sub = df[(df["circuit"]==circ) & (df["model"]==model) & (df["method"]!="noisy")].sort_values("p")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for meth in [m for m in methods if m != "noisy"]:
            ms = sub[sub["method"] == meth].sort_values("p")
            if ms.empty:
                continue
            lw = 2.5 if meth == "adaptive_ml" else 1.6
            ax.plot(ms["p"], ms["impr"], color=PAL[meth], marker=MARKERS[meth],
                    dashes=DASHES[meth], linewidth=lw,
                    markersize=6 if meth=="adaptive_ml" else 5,
                    label=meth.capitalize(),
                    zorder=4 if meth=="adaptive_ml" else 3)
        p_vals = sorted(sub["p"].unique())
        ax.set_xlim(min(p_vals)*0.8, max(p_vals)*1.1)
        ax.axhline(0, color="#888780", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Noise level (p)")
        ax.set_ylabel("Improvement ratio")
        ax.set_title(f"Improvement ratio — {circ} ({model})")
        ax.legend(framealpha=0.4)
        shade_regimes(ax, min(p_vals)*0.8, max(p_vals)*1.1)
        save(fig, f"fig2_improvement_{circ}_{model}")


# ── Fig 3 — Bias–Variance scatter ────────────────────────────────────────────
print("Generating Fig 3: bias-variance...")
for model in models:
    for circ in circuits:
        sub = df[(df["circuit"]==circ) & (df["model"]==model) & (df["method"]!="noisy")]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        for meth in [m for m in methods if m != "noisy"]:
            ms = sub[sub["method"] == meth]
            if ms.empty:
                continue
            ax.scatter(ms["err"], ms["sd"], c=PAL[meth], marker=MARKERS[meth],
                       s=50, label=meth.capitalize(),
                       edgecolors="white", linewidths=0.4, zorder=3)
            for _, row in ms.iterrows():
                ax.annotate(row["regime"][0].upper(), (row["err"], row["sd"]),
                            fontsize=7, color=PAL[meth],
                            xytext=(3,3), textcoords="offset points")
        ax.set_xlabel("Bias  (|ideal − mean|)")
        ax.set_ylabel("Variance  (std dev)")
        ax.set_title(f"Bias–variance — {circ} ({model})")
        ax.legend(framealpha=0.4)
        save(fig, f"fig3_bias_variance_{circ}_{model}")


# ── Fig 4 — Error heatmap ─────────────────────────────────────────────────────
print("Generating Fig 4: error heatmaps...")
for model in models:
    sub = df[df["model"] == model]
    for meth in [m for m in methods if m != "noisy"]:
        ms = sub[sub["method"] == meth]
        if ms.empty:
            continue
        try:
            pivot = ms.pivot_table(index="p", columns="circuit", values="err")
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)*1.2), 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax,
                    linewidths=0.4, linecolor="#e0e0e0",
                    cbar_kws={"label": "|ideal − mitigated|"})
        ax.set_title(f"Error heatmap — {meth.capitalize()} ({model})")
        save(fig, f"fig4_heatmap_{meth}_{model}")


# ── Fig 5 — Regime summary box plot ──────────────────────────────────────────
print("Generating Fig 5: regime summary...")
for model in models:
    sub = df[(df["model"]==model) & (df["method"]!="noisy")]
    if sub.empty:
        continue
    fig, ax = plt.subplots(figsize=(8, 4))
    order    = ["low", "mid", "high"]
    meth_lst = [m for m in methods if m != "noisy"]
    x_pos    = np.arange(len(order))
    w        = 0.22
    for i, meth in enumerate(meth_lst):
        ms = sub[sub["method"] == meth]
        data = [ms[ms["regime"]==r]["impr"].dropna().values for r in order]
        offset = (i - (len(meth_lst)-1)/2) * w
        bp = ax.boxplot(data, positions=x_pos+offset, widths=w*0.85,
                        patch_artist=True,
                        medianprops=dict(color="white", linewidth=1.5),
                        whiskerprops=dict(color=PAL[meth]),
                        capprops=dict(color=PAL[meth]),
                        flierprops=dict(marker="x", color=PAL[meth], markersize=4),
                        boxprops=dict(facecolor=PAL[meth], alpha=0.7, linewidth=0))
    ax.axhline(0, color="#888780", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.capitalize() for r in order])
    ax.set_xlabel("Noise regime")
    ax.set_ylabel("Improvement ratio")
    ax.set_title(f"Improvement by noise regime — {model}")
    patches = [mpatches.Patch(color=PAL[m], label=m.capitalize()) for m in meth_lst]
    ax.legend(handles=patches, framealpha=0.4)
    save(fig, f"fig5_regime_summary_{model}")


# ── Fig 6 — Depth sweep ───────────────────────────────────────────────────────
print("Generating Fig 6: depth sweep...")
depth_circs = [c for c in circuits if c.startswith("depth_")]

# Updated to include depth_d1
depth_map = {
    "depth_d1": 1, "depth_d3": 3, "depth_d5": 5, "depth_d10": 10,
}

if depth_circs:
    for model in models:
        sub = df[(df["model"]==model) & (df["circuit"].isin(depth_circs))]
        if sub.empty:
            continue
        for regime in ["low", "mid", "high"]:
            rs = sub[sub["regime"]==regime].copy()
            if rs.empty:
                continue
            rs["depth"] = rs["circuit"].map(depth_map)
            rs = rs.dropna(subset=["depth"])
            rs_agg = rs.groupby(["depth","method"])["err"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            for meth in methods:
                ms = rs_agg[rs_agg["method"]==meth].sort_values("depth")
                if ms.empty:
                    continue
                ax.plot(ms["depth"], ms["err"], color=PAL[meth],
                        marker=MARKERS[meth], dashes=DASHES[meth],
                        linewidth=1.6, markersize=5, label=meth.capitalize())
            ax.set_xlabel("Circuit depth (layers)")
            ax.set_ylabel("|ideal − mitigated|")
            ax.set_title(f"Error vs depth — {regime} regime ({model})")
            ax.legend(framealpha=0.4)
            save(fig, f"fig6_depth_sweep_{regime}_{model}")


# ── Fig 7 — Significance table ────────────────────────────────────────────────
if HAS_PAIR:
    print("Generating Fig 7: significance table...")
    for model in models:
        sp = df_pair[df_pair["model"]==model] if "model" in df_pair.columns else df_pair
        if sp.empty:
            continue
        pivot_p = sp.pivot_table(index="p", columns="circuit", values="p_val")
        pivot_d = sp.pivot_table(index="p", columns="circuit", values="cohen_d")
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(pivot_p.columns)*1.6), 5))
        sns.heatmap(pivot_p, annot=True, fmt=".3f", cmap="RdYlGn_r",
                    vmin=0, vmax=0.1, ax=axes[0],
                    linewidths=0.4, cbar_kws={"label": "p-value"})
        axes[0].set_title("Welch t-test p-value\n(Richardson vs Linear)")
        sns.heatmap(pivot_d.abs(), annot=True, fmt=".2f", cmap="Blues",
                    ax=axes[1], linewidths=0.4, cbar_kws={"label": "|Cohen's d|"})
        axes[1].set_title("|Cohen's d| effect size\n(Richardson vs Linear)")
        for ax in axes:
            ax.set_xlabel("Circuit")
            ax.set_ylabel("Noise level (p)")
        save(fig, f"fig7_significance_{model}")


# ── Fig 8 — Cross-circuit noisy error ────────────────────────────────────────
print("Generating Fig 8: cross-circuit noisy error...")
for model in models:
    sub = df[(df["model"]==model) & (df["method"]=="noisy")]
    if sub.empty:
        continue
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("tab10", n_colors=len(circuits))
    for i, circ in enumerate(circuits):
        cs = sub[sub["circuit"]==circ].sort_values("p")
        if cs.empty:
            continue
        ax.plot(cs["p"], cs["err"], color=palette[i], marker="o",
                linewidth=1.4, markersize=4, label=circ)
    p_vals = sorted(sub["p"].unique())
    ax.set_xlim(min(p_vals)*0.8, max(p_vals)*1.1)
    ax.set_xlabel("Noise level (p)")
    ax.set_ylabel("|ideal − noisy mean|")
    ax.set_title(f"Noisy error across all circuits ({model})")
    ax.legend(framealpha=0.4, ncol=2)
    shade_regimes(ax, min(p_vals)*0.8, max(p_vals)*1.1)
    save(fig, f"fig8_cross_circuit_{model}")

print(f"\nDone — figures saved to {OUT}/")