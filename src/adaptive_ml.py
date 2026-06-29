"""
adaptive_ml.py
ML-based adaptive method selector.
Selects between linear and richardson based on predicted improvement score.
Returns (result_value, chosen_method_name) for logging.
"""

import numpy as np
import joblib

from src.mitigation import METHODS, CDR_SKIP

model_linear     = joblib.load("results/model_linear.pkl")
model_richardson = joblib.load("results/model_richardson.pkl")

try:
    model_cdr      = joblib.load("results/model_cdr.pkl")
    HAS_CDR_MODEL  = True
except FileNotFoundError:
    model_cdr      = None
    HAS_CDR_MODEL  = False


def regime_to_int(p):
    if p < 0.01:  return 0
    if p < 0.05:  return 1
    return 2


def build_features(qc, p, circuit_name):
    """
    16-element feature vector — must match train_selector.py feature_cols exactly.
    Uses actual transpiled circuit properties (depth, n_qubits, gate_count)
    so features are consistent with what the model was trained on.
    """
    depth      = qc.depth()
    n_qubits   = qc.num_qubits
    gate_count = qc.size()

    regime_int            = regime_to_int(p)
    noise_depth_ratio     = p / (depth + 1e-6)
    gate_density          = gate_count / (depth * n_qubits + 1e-6)
    expected_error        = p * depth
    log_noise             = np.log10(p + 1e-6)
    depth_squared         = depth ** 2
    entanglement_proxy    = (gate_count - depth) / (gate_count + 1e-6)
    complexity_score      = gate_count * depth / (n_qubits + 1)
    noise_polynomial_term = (p ** 2) * depth
    zne_feasibility       = np.exp(-expected_error)
    cdr_cost_proxy        = gate_count * n_qubits
    cdr_applicable        = int(circuit_name.lower() not in CDR_SKIP)

    return [[
        p, regime_int, depth, n_qubits, gate_count,
        noise_depth_ratio, gate_density, expected_error,
        log_noise, depth_squared, entanglement_proxy,
        complexity_score, noise_polynomial_term,
        zne_feasibility, cdr_cost_proxy, cdr_applicable,
    ]]


def adaptive_ml(qc, noise_model, metric_fn, p, circuit_name):
    """
    Predict improvement score for each method and run the best one.
    Returns (result_value, chosen_method_name).
    """
    X = build_features(qc, p, circuit_name)

    scores = {
        "linear":     float(model_linear.predict(X)[0]),
        "richardson": float(model_richardson.predict(X)[0]),
    }

    # Add CDR if model exists and circuit is compatible
    if HAS_CDR_MODEL and circuit_name.lower() not in CDR_SKIP:
        scores["cdr"] = float(model_cdr.predict(X)[0])

    method = max(scores, key=scores.get)

    score_str = " | ".join(f"{k}={v:.3f}" for k, v in scores.items())
    print(f"[SELECTOR] {circuit_name} p={p:.4f} | {score_str} → {method}")

    try:
        if method == "cdr":
            result = METHODS["cdr"](qc, noise_model, metric_fn, circuit_name)
        else:
            result = METHODS[method](qc, noise_model, metric_fn)
        return result, method
    except Exception as e:
        print(f"[WARN] {method} failed ({e}), falling back to linear")
        result = METHODS["linear"](qc, noise_model, metric_fn)
        return result, "linear"