"""
circuits.py
Circuit library for regime-aware QEM evaluation.

Each function returns a QuantumCircuit with measurements.
Circuits are chosen to span different structural properties:
  bell        — minimal entanglement, depth 2
  ghz         — multi-qubit entanglement, depth n
  variational — parameterized ansatz, tunable depth
  deep        — redundant gates, intentionally high depth (stress test)
  qft         — structured high-entanglement, algorithmically motivated
  depth_sweep — same ansatz at increasing depths (isolates depth effect)

This is needed becausee its very tedious to continue without structure/syntax :( 
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT


def bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def ghz(n_q: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(n_q)
    qc.h(0)
    for i in range(n_q - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def variational(n_q: int = 2, depth: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(n_q)
    for _ in range(depth):
        for q in range(n_q):
            qc.rx(np.pi / 4, q)
            qc.ry(np.pi / 3, q)
        for q in range(n_q - 1):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def deep() -> QuantumCircuit:
    """
    Bell state followed by 10 pairs of cancelling CX gates.
    Logically equivalent to bell() but accumulates noise proportional
    to gate count — used to stress-test mitigation under high depth.
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    for _ in range(10):
        qc.cx(0, 1)
        qc.cx(0, 1)
    qc.measure_all()
    return qc


def qft_circuit(n_q: int = 3) -> QuantumCircuit:
    """
    Quantum Fourier Transform on n_q qubits.
    Represents a structured, algorithmically-motivated high-entanglement
    circuit — tests ZNE on a real subroutine rather than a synthetic one.
    """
    qc = QuantumCircuit(n_q)
    qc.h(range(n_q))                    # initialise in superposition
    qc.compose(QFT(n_q, do_swaps=True), inplace=True)
    qc.measure_all()
    return qc


def depth_sweep(depth: int, n_q: int = 2) -> QuantumCircuit:
    """
    Variational ansatz at a specified depth — used to isolate the
    effect of circuit depth independently of circuit structure.
    depth : number of rotation+entangling layers (1, 3, 5, 10)
    """
    return variational(n_q=n_q, depth=depth)


# Registry — maps name to zero-argument callables for run_experiment
CIRCUITS = {
    "bell":       bell,
    "ghz":        lambda: ghz(3),
    "variational":lambda: variational(2, depth=3),
    "deep":       deep,
    "qft":        lambda: qft_circuit(3),
    "depth_d1":   lambda: depth_sweep(1),
    "depth_d3":   lambda: depth_sweep(3),
    "depth_d5":   lambda: depth_sweep(5),
    "depth_d10":  lambda: depth_sweep(10),
}