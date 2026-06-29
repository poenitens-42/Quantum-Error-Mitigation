"""
metrics.py
Metric functions for evaluating QEM output.

All metric_fn(counts) -> float, where float is an expectation value
in [-1, 1] or [0, 1] depending on the observable.
"""

import numpy as np


def exp_all_z(counts: dict) -> float:
    """
    <Z^n> — product of single-qubit Z eigenvalues across all qubits.
    Returns +1 for all-|0>, -1 for all-|1>, intermediate otherwise.
    """
    total = sum(counts.values())
    val = sum(
        count * np.prod([1 if b == "0" else -1 for b in bits])
        for bits, count in counts.items()
    )
    return val / total


def exp_ghz_parity(counts: dict) -> float:
    """
    GHZ parity — fraction of shots in |000...> or |111...> minus rest.
    Ideal GHZ state scores +1.0.
    """
    total = sum(counts.values())
    n = len(next(iter(counts)))
    ideal = {"0" * n, "1" * n}
    val = sum(count * (1 if bits in ideal else -1) for bits, count in counts.items())
    return val / total


def tvd(counts_noisy: dict, counts_ideal: dict) -> float:
    """
    Total Variation Distance between two output distributions.
    TVD = 0 means identical distributions; TVD = 1 means no overlap.
    Useful for circuits where no single observable captures full quality.
    """
    keys = set(counts_noisy) | set(counts_ideal)
    total_n = sum(counts_noisy.values())
    total_i = sum(counts_ideal.values())
    return 0.5 * sum(
        abs(counts_noisy.get(k, 0) / total_n - counts_ideal.get(k, 0) / total_i)
        for k in keys
    )


def impr(err_noisy: float, err_mitigated: float) -> float:
    """
    Improvement ratio — how much of the noisy error was recovered.
    Returns fraction in [0, 1]; negative means mitigation made things worse.
    """
    if err_noisy == 0:
        return 0.0
    return (err_noisy - err_mitigated) / err_noisy


# Maps circuit name to the right metric function
METRIC_MAP = {
    "bell":        exp_all_z,
    "ghz":         exp_ghz_parity,
    "variational": exp_all_z,
    "deep":        exp_all_z,
    "qft":         exp_all_z,
    "depth_d1":    exp_all_z,
    "depth_d3":    exp_all_z,
    "depth_d5":    exp_all_z,
    "depth_d10":   exp_all_z,
}