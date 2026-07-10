"""Tests for sparse (cardinality reduction) state preparation."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from state import dicke_state, to_statevector, w_state
from sparse import prepare_state_sparse
#Diyi_test

def _fidelity(qc, target, n_bits):
    sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
    sim = np.real(sv.data[: 1 << n_bits])
    tgt = to_statevector(target, n_bits)
    return float(np.abs(np.dot(sim, tgt)) ** 2)


class TestSparse:
    def test_w_state_3(self):
        state, n = w_state(3)
        qc = prepare_state_sparse(state, n, timeout=30.0)
        assert qc.num_qubits == n
        fid = _fidelity(qc, state, n)
        assert fid > 1 - 1e-6, f"fidelity={fid}"

    def test_w_state_4(self):
        state, n = w_state(4)
        qc = prepare_state_sparse(state, n, timeout=30.0)
        assert qc.num_qubits == n
        fid = _fidelity(qc, state, n)
        assert fid > 1 - 1e-6, f"fidelity={fid}"

    def test_dicke_3_2(self):
        state, n = dicke_state(3, 2)
        qc = prepare_state_sparse(state, n, timeout=30.0)
        assert qc.num_qubits == n
        fid = _fidelity(qc, state, n)
        assert fid > 1 - 1e-6, f"fidelity={fid}"

    def test_two_term_state(self):
        state = {0b00: 0.6, 0b11: 0.8}
        n = 2
        qc = prepare_state_sparse(state, n, timeout=10.0)
        fid = _fidelity(qc, state, n)
        assert fid > 1 - 1e-6, f"fidelity={fid}"

    def test_large_cardinality_above_1000(self):
        """Regression: a state with cardinality > 1000 must not silently return
        a wrong circuit.  Previously max_iter=1000 hard-capped the reduction,
        so fidelity collapsed from 1.0 (card=990) to ~1e-4 (card=2000)."""
        import random
        from state import normalize
        random.seed(0)
        n = 12
        card = 1500
        idxs = random.sample(range(1 << n), card)
        state = normalize({i: random.gauss(0, 1) for i in idxs})
        qc = prepare_state_sparse(state, n, timeout=120.0)
        fid = _fidelity(qc, state, n)
        assert fid > 1 - 1e-6, f"fidelity={fid} (expected ~1.0)"
