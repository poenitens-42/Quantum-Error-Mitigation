"""
runner.py
Thin wrapper around Qiskit Aer's qasm_simulator.
"""

from qiskit_aer import Aer
from qiskit.compiler import transpile


def run_circuit(qc, noise_model=None, shots: int = 1024) -> dict:
    """
    Transpile and run qc on the qasm simulator.
    Returns a counts dict {bitstring: count}.
      shots : number of measurement repetitions
    """
    backend = Aer.get_backend("qasm_simulator")
    job = backend.run(qc, shots=shots, noise_model=noise_model)
    return job.result().get_counts()