"""
noise_models.py
Three noise model families for regime-aware evaluation:
  depolarizing  — simple, analytically tractable
  thermal       — physically realistic (T1/T2 relaxation)
  combined      — depolarizing + readout error
"""

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)


def make_depolar(p: float) -> NoiseModel:
    """Depolarizing noise on H (1q) and CX (2q) gates."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["h", "rx", "ry"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    return nm


def make_thermal(p: float) -> NoiseModel:
    """
    Thermal relaxation scaled by p.
    p controls effective gate error via T1/T2 scaling:
      p=0.001 → long T1/T2 (low noise)
      p=0.1   → short T1/T2 (high noise)
    """
    # Scale T1/T2 inversely with p — higher p = more relaxation
    t1 = 50e-6 * (0.001 / p) ** 0.5
    t2 = min(2 * t1, 70e-6 * (0.001 / p) ** 0.5)
    tg_1q = 50e-9
    tg_2q = 300e-9

    nm = NoiseModel()
    err_1q = thermal_relaxation_error(t1, t2, tg_1q)
    err_2q = thermal_relaxation_error(t1, t2, tg_2q).expand(
        thermal_relaxation_error(t1, t2, tg_2q)
    )
    nm.add_all_qubit_quantum_error(err_1q, ["h", "rx", "ry", "rz"])
    nm.add_all_qubit_quantum_error(err_2q, ["cx"])
    return nm


def make_combined(p: float, p_ro: float = 0.02) -> NoiseModel:
    """
    Depolarizing gate noise + symmetric readout error.
      p     : depolarizing probability
      p_ro  : readout bit-flip probability per qubit
    """
    nm = make_depolar(p)
    ro_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    nm.add_all_qubit_readout_error(ro_err)
    return nm


# Registry — used by run_experiment to iterate over model families
NOISE_FACTORIES = {
    "depolar":  make_depolar,
    "combined": lambda p: make_combined(p, p_ro=0.02),
    "thermal":  lambda p: make_thermal(), 
}