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


def rotation_fidelity(qc: QuantumCircuit, target: State, n_bits: int) -> float:
    """|<target|sim>|^2 over the first 2^n_bits amplitudes of the evolved state."""
    try:
        sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
        sim = sv.data[: 1 << n_bits]
        tgt = to_statevector(target, n_bits).astype(complex)
        return float(np.abs(np.vdot(tgt, sim)) ** 2)
    except Exception:
        return 0.0


def qrom_marginal_fidelity(qc: QuantumCircuit, target: State, n_bits: int) -> float:
    """Classical (Bhattacharyya) fidelity between the marginal probabilities of
    the first n_bits of the evolved state and the target amplitudes squared.
    """
    try:
        sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
        probs = np.zeros(1 << n_bits)
        mask = (1 << n_bits) - 1
        for idx, amp in enumerate(sv.data):
            probs[idx & mask] += abs(amp) ** 2
        tgt = to_statevector(target, n_bits)
        tgt_probs = tgt ** 2
        return float(np.sum(np.sqrt(probs * tgt_probs)) ** 2)
    except Exception:
        return 0.0


def qrom_alias_analytical_fidelity(
    target: State,
    n_bits: int,
    precision_bits: int,
) -> float:
    """Bhattacharyya fidelity between the distribution an ideal alias-sampling
    QROM produces (given precision_bits rounding) and the target amplitudes
    squared — computed analytically without running a simulator.

    The QROM circuit picks address j uniformly (1/N), then with probability
    keep[j] / 2^b keeps j else swaps to alias[j]. So the marginal probability
    of observing data = i is
        p(i) = (1/N) * keep[i]/2^b + sum_{j: alias[j]=i} (1/N) * (1 - keep[j]/2^b).
    """
    from qsp.qrom.alias_sampling import build_alias_table

    table = build_alias_table(target, n_bits, precision_bits)
    N = 1 << n_bits
    K = 1 << precision_bits
    probs = np.zeros(N)
    for j in range(N):
        p_keep = table.keep_values[j] / K
        probs[j] += p_keep / N
        probs[table.alias_indices[j]] += (1.0 - p_keep) / N

    tgt = to_statevector(target, n_bits)
    tgt_probs = tgt ** 2
    return float(np.sum(np.sqrt(probs * tgt_probs)) ** 2)


def mps_marginal_fidelity(
    qc: QuantumCircuit,
    target: State,
    n_bits: int,
    shots: int = 20000,
    seed: int | None = 0,
) -> float:
    """Bhattacharyya fidelity estimated from MPS-simulator shot counts on the
    first n_bits (data register). Scales to large qubit counts when the circuit
    has bounded entanglement. Uses plug-in estimator on shot histogram and so
    is biased low when many bins receive few shots; see mps_exact_marginal_fidelity
    for the shot-noise-free variant."""
    from qiskit import ClassicalRegister
    from qiskit_aer import AerSimulator

    meas = qc.copy()
    creg = ClassicalRegister(n_bits, "c_data")
    meas.add_register(creg)
    for q in range(n_bits):
        meas.measure(q, creg[q])

    sim = AerSimulator(method="matrix_product_state")
    result = sim.run(meas, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()

    probs = np.zeros(1 << n_bits)
    for bitstring, c in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        probs[idx] += c
    probs /= probs.sum()

    tgt = to_statevector(target, n_bits)
    tgt_probs = tgt ** 2
    return float(np.sum(np.sqrt(probs * tgt_probs)) ** 2)


def aer_rotation_fidelity(qc: QuantumCircuit, target: State, n_bits: int) -> float:
    """|<target|sim>|^2 computed via Aer's C++ statevector backend.

    Drop-in faster replacement for ``rotation_fidelity`` when the circuit fits
    in memory (qc.num_qubits <= ~28). Uses the first 2^n_bits amplitudes of
    the final state vector.

    Note: we feed Aer the circuit directly without a Qiskit ``transpile`` pass.
    On large circuits (e.g. ~400k gates), the default ``transpile`` is itself
    many-minutes slow while Aer ingests basis gates natively in ~seconds.
    """
    from qiskit_aer import AerSimulator

    m = qc.copy()
    m.save_statevector(label="sv")

    sim = AerSimulator(method="statevector")
    result = sim.run(m, shots=1).result()
    sv = np.asarray(result.data(0)["sv"], dtype=complex)
    tgt = to_statevector(target, n_bits).astype(complex)
    return float(np.abs(np.vdot(tgt, sv[: 1 << n_bits])) ** 2)


def mps_exact_marginal_fidelity(
    qc: QuantumCircuit,
    target: State,
    n_bits: int,
) -> float:
    """Bhattacharyya fidelity against *exact* marginal probabilities on the
    first n_bits qubits, computed by the Aer MPS simulator via
    SaveProbabilities. No shot noise; fidelity == 1 up to MPS truncation when
    the circuit is correct. Practical when n_bits is small (<= ~24) because
    the marginal is dense over 2^n_bits."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import save_instructions  # noqa: F401  (registers methods)

    m = qc.copy()
    m.save_probabilities(qubits=list(range(n_bits)), label="p")

    sim = AerSimulator(method="matrix_product_state")
    result = sim.run(m, shots=1).result()
    probs = np.asarray(result.data(0)["p"], dtype=float)

    tgt = to_statevector(target, n_bits)
    tgt_probs = tgt ** 2
    return float(np.sum(np.sqrt(probs * tgt_probs)) ** 2)
