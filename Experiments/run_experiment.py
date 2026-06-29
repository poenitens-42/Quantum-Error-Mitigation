"""
run_experiment.py
Regime-aware evaluation of quantum error mitigation techniques.

Circuits: bell, ghz, qft, depth_d5
  - depth_d1 removed: too shallow, ZNE noise amplification destroys signal
  - CDR not run (no compatible circuit in selected set)

Outputs:
  results/raw/results.csv
  results/raw/pairwise.csv
"""

import warnings
import numpy as np
import pandas as pd
import os
import yaml

warnings.filterwarnings("ignore")

from src.circuits      import CIRCUITS
from src.noise_models  import NOISE_FACTORIES
from src.noise_regimes import ALL_NOISE, classify
from src.runner        import run_circuit
from src.metrics       import METRIC_MAP
from src.mitigation    import METHODS, CDR_SKIP
from src.stats         import summarise, welch_t, cohen_d

from qiskit import transpile
from qiskit_aer import Aer

with open("experiments/config.yaml") as f:
    cfg = yaml.safe_load(f)

n_runs = cfg["num_runs"]
shots  = cfg.get("shots", 1024)

IDEAL_BY_CIRCUIT = {
    "bell":     1.0,
    "ghz":      1.0,
    "qft":      1.0,
    "depth_d5": None,
}

SELECTED_CIRCUITS = ["bell", "ghz", "qft", "depth_d5"]


def get_ideal(circ_name, qc, metric_fn):
    if IDEAL_BY_CIRCUIT.get(circ_name) is not None:
        return IDEAL_BY_CIRCUIT[circ_name]
    vals = [metric_fn(run_circuit(qc, noise_model=None, shots=shots)) for _ in range(5)]
    return float(np.mean(vals))


def run_single_config(args):
    try:
        circ_name, model_name, p = args
        print(f"[RUN] {circ_name} | {model_name} | p={p:.4f}")

        backend   = Aer.get_backend("qasm_simulator")
        metric_fn = METRIC_MAP[circ_name]
        regime    = classify(p)
        nm        = NOISE_FACTORIES[model_name](p)

        qc = transpile(
            CIRCUITS[circ_name](),
            backend,
            basis_gates=["rx", "rz", "cx"],
            optimization_level=0,
            seed_transpiler=42,
        )

        id_val = get_ideal(circ_name, qc, metric_fn)
        raw = {"noisy": [], **{m: [] for m in METHODS}}

        for _ in range(n_runs):
            counts = run_circuit(qc, nm, shots=shots)
            raw["noisy"].append(metric_fn(counts))

            for mname, mfn in METHODS.items():
                # Skip CDR for all circuits in this experiment
                if mname == "cdr":
                    raw[mname].append(None)
                    continue
                try:
                    val = mfn(qc, nm, metric_fn)
                    raw[mname].append(val)
                except Exception as e:
                    print(f"[WARN] {mname} failed for {circ_name} at p={p:.4f}: {e}")
                    raw[mname].append(None)

        rows_local = []
        n_arr      = np.array(raw["noisy"])
        n_stats    = summarise(n_arr, ideal=id_val)
        method_stats = {}

        for mname in METHODS:
            m_arr = np.array([x for x in raw[mname] if x is not None])
            if len(m_arr) == 0:
                continue
            m_stats = summarise(m_arr, ideal=id_val)
            method_stats[mname] = (m_arr, m_stats)

            imp = (n_stats["err"] - m_stats["err"]) / max(n_stats["err"], 1e-3)
            imp = float(np.clip(imp, -1.0, 1.0))

            rows_local.append({
                "circuit":   circ_name, "model": model_name, "p": p,
                "regime":    regime,    "method": mname,     "ideal": id_val,
                "mu":        m_stats["mu"],   "sd":    m_stats["sd"],
                "ci_lo":     m_stats["ci_lo"],"ci_hi": m_stats["ci_hi"],
                "err":       m_stats["err"],  "err_noisy": n_stats["err"],
                "impr":      imp,
            })

        # Pairwise Richardson vs Linear
        if "richardson" in method_stats and "linear" in method_stats:
            r_arr, _ = method_stats["richardson"]
            l_arr, _ = method_stats["linear"]
            t_stat, p_val = welch_t(r_arr, l_arr)
            d = cohen_d(r_arr, l_arr)
        else:
            t_stat, p_val, d = np.nan, np.nan, np.nan

        pair_row = {
            "circuit": circ_name, "model": model_name, "p": p, "regime": regime,
            "t_stat": round(t_stat, 4), "p_val": round(p_val, 4),
            "cohen_d": round(d, 4), "sig": p_val < 0.05,
        }

        return rows_local, [pair_row]

    except Exception as e:
        print(f"[ERROR] {args}: {e}")
        return [], []


if __name__ == "__main__":
    jobs = [
        (circ, model, p)
        for circ  in SELECTED_CIRCUITS
        for model in NOISE_FACTORIES
        for p     in ALL_NOISE
    ]

    print(f"Total jobs: {len(jobs)}")
    print(f"Circuits : {SELECTED_CIRCUITS}")
    print(f"Noise pts: {len(ALL_NOISE)} per circuit/model\n")

    rows, pair_rows = [], []
    for job in jobs:
        r, p = run_single_config(job)
        rows.extend(r)
        pair_rows.extend(p)

    os.makedirs("results/raw", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/raw/results.csv", index=False)
    pd.DataFrame(pair_rows).to_csv("results/raw/pairwise.csv", index=False)

    df = pd.DataFrame(rows)
    print("\nSaved: results/raw/results.csv")
    print(f"Total rows: {len(df)}")
    print("\nMean improvement by circuit/regime/method:")
    print(df.groupby(["circuit", "regime", "method"])["impr"].mean().round(3).to_string())