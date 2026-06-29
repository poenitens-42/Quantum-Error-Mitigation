"""
stats.py
Statistical analysis for regime-aware QEM evaluation.

Functions:
  bootstrap_ci  — 95% confidence interval via percentile bootstrap
  welch_t       — Welch's t-test p-value (unequal variance)
  cohen_d       — effect size between two samples
  summarise     — all stats for one array of repeated measurements
"""

import numpy as np
from scipy import stats as sp_stats


def bootstrap_ci(
    vals: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval.
    Returns (ci_lo, ci_hi) at the requested coverage level.
      n_boot : number of bootstrap resamples (2000 is sufficient)
    """
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Welch's t-test between samples a and b (unequal variance assumed).
    Returns (t_stat, p_value).
    """
    t_stat, p_val = sp_stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_val)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d effect size.
    |d| < 0.2 small, 0.2–0.8 medium, > 0.8 large.
    """
    pooled_sd = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def summarise(vals: np.ndarray, ideal: float = 1.0) -> dict:
    """
    Full statistical summary for one array of repeated measurements.
    Returns a flat dict of scalars — ready to append to a results list.
    """
    mu = float(vals.mean())
    sd = float(vals.std())
    ci_lo, ci_hi = bootstrap_ci(vals)
    err = abs(ideal - mu)
    return {
        "mu":    mu,
        "sd":    sd,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "err":   err,
    }