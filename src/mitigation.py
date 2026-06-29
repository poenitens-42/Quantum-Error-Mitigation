"""
mitigation.py
Three mitigation methods for thesis comparison:
  zne_richardson  — ZNE with Richardson extrapolation
  zne_linear      — ZNE with linear extrapolation
  cdr             — Clifford Data Regression

CDR_SKIP: circuits structurally incompatible with CDR.
  bell, ghz   — too shallow (2-3 gates), regression trivially perfect
  qft         — QasmUGate Cirq conversion failure
  deep        — redundant CX pairs collapse regression to ideal
  depth_d1    — too shallow
  depth_d5    — CDR overhead not justified vs ZNE
  depth_d10   — same as depth_d5

CDR works on: depth_d3 (if used), any circuit with 10+ non-Clifford gates.
"""

import numpy as np

from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import RichardsonFactory, LinearFactory
from mitiq.zne import execute_with_zne

try:
    from mitiq.cdr import execute_with_cdr, generate_training_circuits
    CDR_AVAILABLE = True
except ImportError:
    CDR_AVAILABLE = False

from src.runner import run_circuit

SCALE_FACTORS = [1, 2, 3]
SHOTS = 1024

CDR_SKIP = {
    "bell", "ghz", "qft", "deep",
    "depth_d1", "depth_d5", "depth_d10",
}


def _executor(noise_model, metric_fn, shots=SHOTS):
    def _run(qc):
        counts = run_circuit(qc, noise_model=noise_model, shots=shots)
        return metric_fn(counts)
    return _run


def _ideal_executor(metric_fn, shots=SHOTS):
    def _run(qc):
        counts = run_circuit(qc, noise_model=None, shots=shots)
        return metric_fn(counts)
    return _run


def _run_zne(qc, noise_model, metric_fn, factory):
    return execute_with_zne(
        qc,
        executor=_executor(noise_model, metric_fn),
        factory=factory,
        scale_noise=fold_gates_at_random,
    )


def zne_richardson(qc, noise_model, metric_fn) -> float:
    result = _run_zne(qc, noise_model, metric_fn, RichardsonFactory(SCALE_FACTORS))
    # Richardson can blow up at very low noise — clamp to physical range
    return float(np.clip(result, -1.5, 1.5))


def zne_linear(qc, noise_model, metric_fn) -> float:
    return _run_zne(qc, noise_model, metric_fn, LinearFactory(SCALE_FACTORS))


def cdr(qc, noise_model, metric_fn, circuit_name=None, n_train=10, shots=SHOTS):
    """
    Clifford Data Regression.
    Results near +/-1.0 are VALID — do NOT use abs(result-1.0) as leakage check.
    Only rejects non-finite or |result| > 2.0 (physically impossible).
    """
    if not CDR_AVAILABLE:
        raise ImportError("mitiq CDR not available")

    if circuit_name and circuit_name.lower() in CDR_SKIP:
        raise ValueError(f"CDR skipped: {circuit_name} is in CDR_SKIP")

    from qiskit import transpile
    qc = transpile(qc, basis_gates=["rx", "rz", "cx"], optimization_level=0)

    noisy_exec = _executor(noise_model, metric_fn, shots=shots)
    ideal_exec = _ideal_executor(metric_fn, shots=shots)

    train_circs = generate_training_circuits(
        qc,
        num_training_circuits=n_train,
        fraction_non_clifford=0.3,
    )

    def jittered_executor(q):
        return noisy_exec(q) + np.random.normal(0, 1e-3)

    result = execute_with_cdr(
        qc,
        executor=jittered_executor,
        simulator=ideal_exec,
        training_circuits=train_circs,
    )

    result = float(result)

    if not np.isfinite(result):
        raise ValueError(f"CDR returned non-finite value: {result}")
    if abs(result) > 2.0:
        raise ValueError(f"CDR extrapolation blew up: result={result:.4f}")

    return result


METHODS = {
    "richardson": zne_richardson,
    "linear":     zne_linear,
    "cdr":        cdr,
}