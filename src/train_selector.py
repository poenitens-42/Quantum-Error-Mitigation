"""
train_selector.py
Trains XGBRegressor models to predict improvement per method:
  model_linear.pkl     — predicts ZNE linear improvement
  model_richardson.pkl — predicts ZNE Richardson improvement

Key fixes vs previous version:
  - Richardson blowup filter REMOVED — it was removing too many samples
    and biasing the model toward always predicting richardson as better
  - Both models trained on identical sample sets (same pivot rows)
    so predictions are directly comparable
  - 5-fold CV for honest RMSE estimate
"""

import pandas as pd
import numpy as np
import joblib
import sys

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("results/raw/results.csv")
print(f"Total samples loaded: {len(df)}")

df = df[df["method"].isin(["linear", "richardson"])].copy()
print("\nMethod distribution:")
print(df["method"].value_counts())

# ----------------------------
# PIVOT — one row per (circuit, model, p)
# ----------------------------
pivot = df.pivot_table(
    index=["circuit", "model", "p"],
    columns="method",
    values="impr"
).reset_index()

print(f"\nColumns after pivot: {list(pivot.columns)}")
print(f"Pivot rows before dropna: {len(pivot)}")

# Only keep rows where BOTH methods have data
pivot = pivot.dropna(subset=["linear", "richardson"])
print(f"Pivot rows after dropna: {len(pivot)}")

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
depth_map = {
    "bell": 2, "ghz": 3, "variational": 6, "deep": 22, "qft": 10,
    "depth_d1": 2, "depth_d3": 6, "depth_d5": 10, "depth_d10": 20,
}
qubit_map = {
    "bell": 2, "ghz": 3, "variational": 2, "deep": 2, "qft": 3,
    "depth_d1": 2, "depth_d3": 2, "depth_d5": 2, "depth_d10": 2,
}
gate_map = {
    "bell": 2,  "ghz": 4,  "variational": 20, "deep": 22, "qft": 15,
    "depth_d1": 6, "depth_d3": 18, "depth_d5": 30, "depth_d10": 60,
}


def regime_to_int(p):
    if p < 0.01:  return 0
    if p < 0.05:  return 1
    return 2


def add_features(d):
    d = d.copy()
    d["depth"]      = d["circuit"].map(depth_map)
    d["n_qubits"]   = d["circuit"].map(qubit_map)
    d["gate_count"] = d["circuit"].map(gate_map)
    d["regime_int"] = d["p"].apply(regime_to_int)

    d["noise_depth_ratio"]     = d["p"] / (d["depth"] + 1e-6)
    d["gate_density"]          = d["gate_count"] / (d["depth"] * d["n_qubits"] + 1e-6)
    d["expected_error"]        = d["p"] * d["depth"]
    d["log_noise"]             = np.log10(d["p"] + 1e-6)
    d["depth_squared"]         = d["depth"] ** 2
    d["entanglement_proxy"]    = (d["gate_count"] - d["depth"]) / (d["gate_count"] + 1e-6)
    d["complexity_score"]      = d["gate_count"] * d["depth"] / (d["n_qubits"] + 1)
    d["noise_polynomial_term"] = (d["p"] ** 2) * d["depth"]
    d["zne_feasibility"]       = np.exp(-d["expected_error"])
    d["cdr_cost_proxy"]        = d["gate_count"] * d["n_qubits"]
    d["cdr_applicable"]        = 0  # no CDR circuits in current experiment
    return d


feature_cols = [
    "p", "regime_int", "depth", "n_qubits", "gate_count",
    "noise_depth_ratio", "gate_density", "expected_error",
    "log_noise", "depth_squared", "entanglement_proxy",
    "complexity_score", "noise_polynomial_term",
    "zne_feasibility", "cdr_cost_proxy", "cdr_applicable",
]


def clean_target(s):
    return np.tanh(s.clip(-1, 1))


# ----------------------------
# PREPARE FEATURES
# NOTE: Use SAME rows for both models so scores are comparable
# ----------------------------
data = add_features(pivot)

# Only remove physically impossible values (keep negative improvements)
data = data[data["linear"].abs() < 2]
data = data[data["richardson"].abs() < 2]

print(f"Training samples (after outlier removal): {len(data)}")

X          = data[feature_cols]
# Rank-normalize targets so both methods are on the same scale
# This prevents richardson's high-variance wins from dominating
from scipy.stats import rankdata
y_linear = pd.Series(rankdata(data["linear"]) / len(data)).values
y_rich   = pd.Series(rankdata(data["richardson"]) / len(data)).values

X_tr, X_te, yL_tr, yL_te, yR_tr, yR_te = train_test_split(
    X, y_linear, y_rich, test_size=0.2, random_state=42
)

print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

# ----------------------------
# TRAIN MODELS
# ----------------------------
params = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    min_child_weight=3,   # prevents overfitting on small groups
)

model_linear = XGBRegressor(**params)
model_rich   = XGBRegressor(**params)

model_linear.fit(X_tr, yL_tr)
model_rich.fit(X_tr, yR_tr)

rmse_L = np.sqrt(mean_squared_error(yL_te, model_linear.predict(X_te)))
rmse_R = np.sqrt(mean_squared_error(yR_te, model_rich.predict(X_te)))

# 5-fold CV
cv_L = cross_val_score(XGBRegressor(**params), X, y_linear,
                        cv=5, scoring="neg_root_mean_squared_error")
cv_R = cross_val_score(XGBRegressor(**params), X, y_rich,
                        cv=5, scoring="neg_root_mean_squared_error")

print(f"\nLinear     — test RMSE: {rmse_L:.4f} | CV RMSE: {-cv_L.mean():.4f} ± {cv_L.std():.4f}")
print(f"Richardson — test RMSE: {rmse_R:.4f} | CV RMSE: {-cv_R.mean():.4f} ± {cv_R.std():.4f}")

# ----------------------------
# SAVE
# ----------------------------
joblib.dump(model_linear, "results/model_linear.pkl")
joblib.dump(model_rich,   "results/model_richardson.pkl")

metadata = {
    "feature_columns":      feature_cols,
    "rmse_linear":          rmse_L,
    "rmse_richardson":      rmse_R,
    "rmse_cdr":             None,
    "has_linear_model":     True,
    "has_richardson_model": True,
    "has_cdr_model":        False,
    "cdr_compatible_circuits": [],
}
joblib.dump(metadata, "results/model_metadata.pkl")

print("\n=== TRAINING COMPLETE ===")
print(f"  Linear RMSE     : {rmse_L:.4f}")
print(f"  Richardson RMSE : {rmse_R:.4f}")
print("  CDR model       : not trained (no CDR data)")
print("\nModels saved to results/")