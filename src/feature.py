import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# LOAD MODEL + FEATURE NAMES
# ----------------------------
model = joblib.load("results/selector_model.pkl")

# IMPORTANT: must match training order exactly
feature_names = [
    "p",
    "regime_int",
    "depth",
    "n_qubits",
    "gate_count",
    "noise_depth_ratio",
    "gate_density",
    "expected_error",
    "log_noise",
    "depth_squared",
    "entanglement_proxy",
    "cdr_regime",
    "zne_regime",
    "adaptive_regime",
    "complexity_score",
    "noise_polynomial_term",
    "zne_feasibility",
    "cdr_cost_proxy"
]

# ----------------------------
# GET IMPORTANCE
# ----------------------------
importances = model.feature_importances_

df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# ----------------------------
# FILTER TOP FEATURES
# ----------------------------
df = df.sort_values(by="importance", ascending=False).head(8)

# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(6,4))

plt.barh(df["feature"], df["importance"])
plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.title("Feature Importance (XGBoost)")

plt.tight_layout()
plt.savefig("results/plots/feature_importance.png", dpi=300)
plt.show()