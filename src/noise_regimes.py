"""
noise_regimes.py
21-point high-resolution noise grid for ML selector training.
"""

import numpy as np

REGIMES = {
    "low":  (0.0,  0.01),
    "mid":  (0.01, 0.05),
    "high": (0.05, 1.0),
}

NOISE_BY_REGIME = {
    "low":  list(np.round(np.linspace(0.001, 0.009, 7), 6)),
    "mid":  list(np.round(np.linspace(0.012, 0.048, 7), 6)),
    "high": list(np.round(np.linspace(0.060, 0.150, 7), 6)),
}

ALL_NOISE = [p for ps in NOISE_BY_REGIME.values() for p in ps]


def classify(p: float) -> str:
    for label, (lo, hi) in REGIMES.items():
        if lo <= p <= hi:
            return label
    raise ValueError(f"p={p} outside defined regime range [0, 1]")


def regime_color(regime: str) -> str:
    return {"low": "#1D9E75", "mid": "#EF9F27", "high": "#E24B4A"}[regime]


if __name__ == "__main__":
    print("Noise levels per regime:")
    for k, v in NOISE_BY_REGIME.items():
        print(f"  {k}: {len(v)} values → {[round(x,4) for x in v]}")
    print(f"\nTotal noise points: {len(ALL_NOISE)}")