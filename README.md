# Regime-Aware Adaptive Quantum Error Mitigation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20140750.svg)](https://doi.org/10.5281/zenodo.20140750)

Let me preface this by saying that this is my Undergrad Research Internship project 
All hardware that was used was my laptop alone, So there will be a Limitations  


Code for the preprint: **"Regime-Aware Adaptive Quantum Error Mitigation for NISQ Devices via Machine Learning"** — Arjun Prabhakaran, B.M.S. College of Engineering (2026).

---

## What This Is

Existing error mitigation methods for NISQ quantum devices — Linear ZNE and Richardson ZNE — vary in effectiveness depending on circuit structure and noise regime. No single method is universally optimal.

This project trains two XGBoost regressors to predict the expected improvement of each method given circuit and noise features, then selects the better one at runtime. The result is a data-driven, regime-aware selector that adapts per circuit rather than applying a fixed strategy.

---

## Results

| Metric | Value |
|---|---|
| Selector exact accuracy | **64.3%** |
| Improvement over best fixed baseline | **+11.9 pp** (vs. 52.4% naive) |
| Richardson ZNE recall | 85% |
| Linear ZNE recall | 47% |
| Benchmark circuits | 4 (simulated) |
| Dataset size | 126 samples |

The adaptive selector outperforms Richardson ZNE on all four benchmark circuits and matches or exceeds both baselines on structured, deeper circuits.

---

## Repository Structure

```
src/
├── circuits.py          — Circuit generation (4 benchmark types)
├── noise_models.py      — Noise regime simulation
├── noise_regimes.py     — Regime classification
├── mitigation.py        — Linear ZNE + Richardson ZNE implementations
├── feature.py           — Feature extraction from circuits
├── train_selector.py    — XGBoost selector training
├── evaluate_adaptive.py — Adaptive selector evaluation
├── adaptive_ml.py       — Runtime method selection logic
├── benchmark.py         — Fixed-method baseline benchmarks
├── metrics.py           — Accuracy, recall, improvement metrics
├── stats.py             — Statistical significance testing
├── runner.py            — End-to-end pipeline runner
├── confusion.py         — Confusion matrix generation
└── plot_adaptive.py     — Result visualisation

Experiments/
├── config.yaml          — Experiment configuration
└── run_experiment.py    — Config-driven experiment runner

results/
├── raw/                 — pairwise.csv, results.csv
├── plots/               — Generated figures
└── benchmark/           — Baseline comparison outputs
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python Experiments/run_experiment.py
```

### 3. Reproduce plots

```bash
python src/plot_adaptive.py
```

Pre-trained model artifacts (`.pkl`) and raw results CSVs are included for immediate reproducibility without retraining.

---

## Limitations

dataset simulated(lack of NISQ device) and run on limited hardware [ my laptop :( ] 

- **Dataset size**: 126 samples across 4 simulated circuit types. IC significance and generalisation claims are limited at this scale.
- **Simulated noise only**: All circuits run on classical noise simulators (not real quantum hardware). Results may not transfer directly to physical NISQ devices.
- **Circuit diversity**: Four benchmark circuit types. Selector performance on out-of-distribution circuits is unknown.
- **Linear ZNE recall**: 47% — the selector is biased toward Richardson ZNE. This reflects training data distribution, not a fundamental property.
- **Scope**: Undergraduate research project / preprint. Not peer-reviewed.

---

## Citation

```bibtex
@misc{prabhakaran2026regimeaware,
  title     = {Regime-Aware Adaptive Quantum Error Mitigation for NISQ Devices via Machine Learning},
  author    = {Prabhakaran, Arjun},
  year      = {2026},
  doi       = {10.5281/zenodo.20140750},
  url       = {https://zenodo.org/records/20140750},
  publisher = {Zenodo}
}
```

---

## License

Code: MIT  
Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
