import pandas as pd
import numpy as np
import time
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_CSV = "results/raw/results.csv"
OUT_DIR   = Path("results/benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# weights for composite score
ALPHA = 0.5   # penalty for variance
BETA  = 0.0   # set >0 if you add runtime measurements

# =========================
# LOAD + CLEAN
# =========================
df = pd.read_csv(INPUT_CSV)

# basic cleaning
df = df.drop_duplicates()

# clip extreme values (from instability)
if "impr" in df.columns:
    df["impr"] = df["impr"].clip(-2, 2)

if "err" in df.columns:
    df["err"] = df["err"].clip(0, 10)

# aggregate duplicates robustly
df = df.groupby(
    ["circuit","model","p","method","regime"],
    as_index=False
).mean()

print("Loaded rows:", len(df))

# TEST 1 — ACCURACY

acc = df.groupby("method")["impr"].mean().reset_index()
acc.columns = ["method", "avg_improvement"]
acc.to_csv(OUT_DIR / "accuracy.csv", index=False)


# TEST 2 — STABILITY

stab = df.groupby("method")["sd"].mean().reset_index()
stab.columns = ["method", "avg_sd"]
stab.to_csv(OUT_DIR / "stability.csv", index=False)


# TEST 3 — ROBUSTNESS (by regime)

robust = df.groupby(["method","regime"])["impr"].mean().reset_index()
robust.to_csv(OUT_DIR / "robustness_by_regime.csv", index=False)


# TEST 4 — DEPTH SENSITIVITY
depth_df = df[df["circuit"].str.contains("depth", na=False)]
depth = depth_df.groupby(["circuit","method"])["impr"].mean().reset_index()
depth.to_csv(OUT_DIR / "depth_analysis.csv", index=False)


# TEST 5 — GENERALIZATION

# simple proxy: variance of improvement across circuits
gen = df.groupby(["method","circuit"])["impr"].mean().reset_index()
gen_var = gen.groupby("method")["impr"].var().reset_index()
gen_var.columns = ["method", "cross_circuit_variance"]
gen_var.to_csv(OUT_DIR / "generalization.csv", index=False)


# TEST 6 — RANKING CONSISTENCY

# rank methods within each (circuit, model, p)
def rank_block(sub):
    sub = sub.copy()
    sub["rank"] = sub["impr"].rank(ascending=False, method="min")
    return sub

ranked = df.groupby(["circuit","model","p"]).apply(rank_block).reset_index(drop=True)

wins = ranked[ranked["rank"] == 1].groupby("method").size().reset_index(name="wins")
wins.to_csv(OUT_DIR / "ranking_wins.csv", index=False)


# TEST 7 — FAILURE ANALYSIS
fail = df[df["impr"] < 0]
fail_summary = fail.groupby(["method","circuit","regime"]).size().reset_index(name="fail_count")
fail_summary.to_csv(OUT_DIR / "failures.csv", index=False)


# TEST 8 — BIAS–VARIANCE (proxy)

# using err as bias proxy, sd as variance
bv = df.groupby("method").agg(
    bias=("err","mean"),
    variance=("sd","mean")
).reset_index()
bv.to_csv(OUT_DIR / "bias_variance.csv", index=False)


# TEST 9 — EFFICIENCY (placeholder)

# If you have runtime logs, merge here.
eff = pd.DataFrame({
    "method": df["method"].unique(),
    "time": np.nan
})
eff.to_csv(OUT_DIR / "efficiency.csv", index=False)


# TEST 10 — COMPOSITE SCORE

# merge accuracy + stability (+ optional time)
score = acc.merge(stab, on="method", how="left")
score["time"] = 0.0  # replace if measured

score["score"] = score["avg_improvement"] - ALPHA * score["avg_sd"] - BETA * score["time"]
score = score.sort_values("score", ascending=False)

score.to_csv(OUT_DIR / "composite_score.csv", index=False)


# SUMMARY PRINT
print("\n=== BENCHMARK SUMMARY ===")
print("\nAccuracy:\n", acc)
print("\nStability:\n", stab)
print("\nTop Methods (Composite):\n", score.head())

print(f"\nAll benchmark files saved to: {OUT_DIR.resolve()}")