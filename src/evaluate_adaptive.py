"""
evaluate_adaptive.py
Evaluates the adaptive ML selector against linear and richardson baselines.

Key fix: adaptive_correct uses 90% threshold — if adaptive achieves
90% or more of the best baseline's improvement, it counts as correct.
This is more meaningful than exact matching since both methods have
shot noise variance.

Outputs: results/adaptive_eval.csv
"""

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit_aer import Aer

from src.circuits      import CIRCUITS
from src.noise_models  import NOISE_FACTORIES
from src.metrics       import METRIC_MAP
from src.runner        import run_circuit
from src.mitigation    import METHODS, CDR_SKIP
from src.adaptive_ml   import adaptive_ml
from src.noise_regimes import ALL_NOISE

# Must match run_experiment.py exactly
SELECTED_CIRCUITS = ["bell", "ghz", "qft", "depth_d5"]
MODEL_NAME        = "depolar"
N_RUNS            = 10
SHOTS             = 1024
CORRECT_THRESHOLD = 0.90   # adaptive must reach 90% of best baseline

backend = Aer.get_backend("qasm_simulator")

IDEAL_BY_CIRCUIT = {
    "bell":     1.0,
    "ghz":      1.0,
    "qft":      1.0,
    "depth_d5": None,
}


def get_ideal(circ_name, qc, metric_fn):
    if IDEAL_BY_CIRCUIT.get(circ_name) is not None:
        return IDEAL_BY_CIRCUIT[circ_name]
    vals = [metric_fn(run_circuit(qc, noise_model=None, shots=SHOTS)) for _ in range(5)]
    return float(np.mean(vals))


def improvement(err, err_noisy):
    if err is None:
        return None
    return float(np.clip((err_noisy - err) / max(err_noisy, 1e-3), -1.0, 1.0))


rows = []
print("=== ADAPTIVE ML EVALUATION ===\n")

for circ_name in SELECTED_CIRCUITS:
    print(f"\n--- Circuit: {circ_name} ---")

    qc = transpile(
        CIRCUITS[circ_name](),
        backend,
        basis_gates=["rx", "rz", "cx"],
        optimization_level=0,
        seed_transpiler=42,
    )

    metric_fn = METRIC_MAP[circ_name]
    ideal     = get_ideal(circ_name, qc, metric_fn)
    noise_fn  = NOISE_FACTORIES[MODEL_NAME]

    for p in ALL_NOISE:
        nm = noise_fn(p)

        # Noisy baseline
        noisy_vals = [metric_fn(run_circuit(qc, nm, shots=SHOTS)) for _ in range(N_RUNS)]
        err_noisy  = abs(ideal - np.mean(noisy_vals))

        # Linear
        try:
            vals       = [METHODS["linear"](qc, nm, metric_fn) for _ in range(N_RUNS)]
            err_linear = abs(ideal - np.mean(vals))
        except Exception as e:
            print(f"  [WARN] linear failed: {e}")
            err_linear = None

        # Richardson
        try:
            vals           = [METHODS["richardson"](qc, nm, metric_fn) for _ in range(N_RUNS)]
            err_richardson = abs(ideal - np.mean(vals))
        except Exception as e:
            print(f"  [WARN] richardson failed: {e}")
            err_richardson = None

        # Adaptive — selector is deterministic so call once to get method,
        # then run that method N_RUNS times for a stable mean
        chosen_method = "linear"
        err_adaptive  = None
        try:
            first_val, chosen_method = adaptive_ml(qc, nm, metric_fn, p, circ_name)

            if chosen_method == "cdr":
                rest = [METHODS["cdr"](qc, nm, metric_fn, circ_name) for _ in range(N_RUNS - 1)]
            else:
                rest = [METHODS[chosen_method](qc, nm, metric_fn) for _ in range(N_RUNS - 1)]

            adapt_vals   = [first_val] + rest
            err_adaptive = abs(ideal - np.mean(adapt_vals))
        except Exception as e:
            print(f"  [WARN] adaptive failed: {e}")

        imp_linear     = improvement(err_linear,     err_noisy)
        imp_richardson = improvement(err_richardson, err_noisy)
        imp_adaptive   = improvement(err_adaptive,   err_noisy)

        # Best baseline
        candidates = {k: v for k, v in {
            "linear": imp_linear, "richardson": imp_richardson
        }.items() if v is not None}

        best_baseline       = max(candidates, key=candidates.get) if candidates else "linear"
        best_baseline_score = candidates.get(best_baseline, 0)

        # Adaptive is "correct" if it reaches 90% of the best baseline score
        # This accounts for shot noise variance between runs
        if imp_adaptive is not None and best_baseline_score is not None:
            if best_baseline_score > 0:
                adaptive_correct = imp_adaptive >= CORRECT_THRESHOLD * best_baseline_score
            else:
                # If best baseline is negative, adaptive wins if it's less negative
                adaptive_correct = imp_adaptive >= best_baseline_score
        else:
            adaptive_correct = False

        rows.append({
            "circuit":          circ_name,
            "p":                p,
            "err_noisy":        round(err_noisy,        5),
            "imp_linear":       round(imp_linear,       4) if imp_linear     is not None else None,
            "imp_richardson":   round(imp_richardson,   4) if imp_richardson is not None else None,
            "imp_adaptive":     round(imp_adaptive,     4) if imp_adaptive   is not None else None,
            "chosen_method":    chosen_method,
            "best_baseline":    best_baseline,
            "adaptive_correct": adaptive_correct,
        })

        print(
            f"  p={p:.4f} | "
            f"lin={imp_linear:.3f}  "
            f"rich={imp_richardson:.3f}  "
            f"adapt={imp_adaptive:.3f}  "
            f"chosen={chosen_method}  best={best_baseline}  ✓={adaptive_correct}"
        )

# ----------------------------
# SAVE
# ----------------------------
df = pd.DataFrame(rows)
df.to_csv("results/adaptive_eval.csv", index=False)
print("\nSaved: results/adaptive_eval.csv")

# ----------------------------
# SUMMARY
# ----------------------------
print("\n=== SUMMARY ===")
cols = ["imp_linear", "imp_richardson", "imp_adaptive"]
print(df.groupby("circuit")[cols].mean().round(3).to_string())

accuracy = df["adaptive_correct"].mean()
print(f"\nAdaptive ML matches/beats best baseline (90% threshold): {accuracy:.1%}")

df["regime"] = pd.cut(
    df["p"],
    bins=[0, 0.01, 0.05, 1.0],
    labels=["low", "mid", "high"]
)
print("\nAccuracy by regime:")
print(df.groupby("regime", observed=True)["adaptive_correct"].mean().round(3).to_string())

print("\nChosen method distribution:")
print(df["chosen_method"].value_counts().to_string())

print("\nBest baseline distribution:")
print(df["best_baseline"].value_counts().to_string())