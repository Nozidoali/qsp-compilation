"""Gate counting and verification utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qsp.state import State, to_statevector


def gate_counts(qc: QuantumCircuit) -> Dict[str, int]:
    """Return a dict with counts for every gate type in *qc*."""
    return dict(qc.count_ops())


def clifford_t_counts(qc: QuantumCircuit) -> Dict[str, int]:
    """Summarised gate counts useful for Clifford+T cost analysis."""
    ops = qc.count_ops()
    return {
        "t": ops.get("t", 0) + ops.get("tdg", 0),
        "h": ops.get("h", 0),
        "s": ops.get("s", 0) + ops.get("sdg", 0),
        "x": ops.get("x", 0),
        "z": ops.get("z", 0),
        "cx": ops.get("cx", 0),
        "ccx": ops.get("ccx", 0),
        "ry": ops.get("ry", 0),
        "rz": ops.get("rz", 0),
        "other": sum(v for k, v in ops.items()
                     if k not in {"t", "tdg", "h", "s", "sdg", "x", "z",
                                  "cx", "ccx", "ry", "rz"}),
    }


def estimate_t_count(qc: QuantumCircuit) -> int:
    counts = qc.count_ops()
    t = counts.get("t", 0) + counts.get("tdg", 0)
    t += counts.get("ccx", 0) * 4
    return t


def count_cnots(qc: QuantumCircuit) -> int:
    return qc.count_ops().get("cx", 0)


def count_ry(qc: QuantumCircuit) -> int:
    return qc.count_ops().get("ry", 0)


def verify_state_preparation(
    qc: QuantumCircuit,
    target: State,
    n_bits: int,
    tol: float = 1e-6,
) -> float:
    sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
    target_vec = to_statevector(target, n_bits)

    sim_vec = np.real(sv.data[: 1 << n_bits])
    fidelity = float(np.abs(np.dot(sim_vec, target_vec)) ** 2)
    return fidelity
