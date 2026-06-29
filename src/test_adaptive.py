from src.circuits import CIRCUITS
from src.noise_models import NOISE_FACTORIES
from src.metrics import METRIC_MAP
from src.runner import run_circuit
from src.adaptive_ml import adaptive_ml

from qiskit import transpile
from qiskit_aer import Aer
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
circ_name = "bell"
model_name = "depolar"

noise_levels = [0.001, 0.005, 0.01, 0.03, 0.08, 0.15]

backend = Aer.get_backend("qasm_simulator")

# -----------------------------
# SETUP
# -----------------------------
qc = transpile(CIRCUITS[circ_name](), backend)
metric_fn = METRIC_MAP[circ_name]

# Ideal (no noise)
ideal_vals = [
    metric_fn(run_circuit(qc, noise_model=None, shots=1024))
    for _ in range(3)
]
ideal = np.mean(ideal_vals)

print(f"\n=== TESTING ADAPTIVE ML ===")
print(f"Circuit: {circ_name}, Model: {model_name}")
print(f"Ideal value: {ideal:.4f}\n")

# -----------------------------
# LOOP
# -----------------------------
for p in noise_levels:
    nm = NOISE_FACTORIES[model_name](p)

    # --- baseline noisy ---
    noisy_vals = [
        metric_fn(run_circuit(qc, nm, shots=1024))
        for _ in range(5)
    ]
    noisy_mean = np.mean(noisy_vals)
    err_noisy = abs(ideal - noisy_mean)

    # --- linear ---
    from src.mitigation import METHODS
    linear_vals = [METHODS["linear"](qc, nm, metric_fn) for _ in range(5)]
    linear_mean = np.mean(linear_vals)
    err_linear = abs(ideal - linear_mean)

    # --- cdr ---
    cdr_vals = [METHODS["cdr"](qc, nm, metric_fn) for _ in range(5)]
    cdr_mean = np.mean(cdr_vals)
    err_cdr = abs(ideal - cdr_mean)

    # --- adaptive_ml ---
    # prints debug internally if you kept print()
    adaptive_vals = [
        adaptive_ml(qc, nm, metric_fn, p, circ_name)
        for _ in range(5)
    ]
    adaptive_mean = np.mean(adaptive_vals)
    err_adaptive = abs(ideal - adaptive_mean)

    # --- improvements ---
    def improvement(err_method):
        return (err_noisy - err_method) / max(err_noisy, 1e-8)

    imp_linear = improvement(err_linear)
    imp_cdr = improvement(err_cdr)
    imp_adaptive = improvement(err_adaptive)

    # --- best baseline ---
    best_baseline = "linear" if imp_linear > imp_cdr else "cdr"

    print(f"p = {p:.3f}")
    print(f"  noisy_err    : {err_noisy:.6f}")
    print(f"  linear       : {imp_linear:.3f}")
    print(f"  cdr          : {imp_cdr:.3f}")
    print(f"  adaptive_ml  : {imp_adaptive:.3f}")
    print(f"  best baseline: {best_baseline}")
    print("-" * 50)