"""Tests for Clifford+T decomposition."""
from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qsp.circuit_utils import decompose_to_clifford_t
from qsp.metrics import verify_state_preparation
from qsp.rotation.dense import prepare_state_dense
from qsp.rotation.sparse import prepare_state_sparse
from qsp.state import w_state, dense_uniform


def _has_ry(qc: QuantumCircuit) -> bool:
    ops = qc.count_ops()
    return ops.get("ry", 0) > 0 or ops.get("cry", 0) > 0


def _fidelity(qc_a: QuantumCircuit, qc_b: QuantumCircuit) -> float:
    sv_a = Statevector.from_label("0" * qc_a.num_qubits).evolve(qc_a)
    sv_b = Statevector.from_label("0" * qc_b.num_qubits).evolve(qc_b)
    return float(abs(np.vdot(sv_a.data, sv_b.data)) ** 2)


class TestDecomposeRY:
    """Test that individual RY gates decompose correctly."""

    def test_trivial_zero(self):
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        out = decompose_to_clifford_t(qc, precision_bits=10)
        assert out.size() == 0

    def test_trivial_pi(self):
        qc = QuantumCircuit(1)
        qc.ry(math.pi, 0)
        out = decompose_to_clifford_t(qc, precision_bits=10)
        assert not _has_ry(out)
        ops = out.count_ops()
        assert ops.get("x", 0) == 1

    def test_trivial_neg_pi(self):
        qc = QuantumCircuit(1)
        qc.ry(-math.pi, 0)
        out = decompose_to_clifford_t(qc, precision_bits=10)
        assert not _has_ry(out)
        ops = out.count_ops()
        assert ops.get("x", 0) == 1


class TestSparseDecomposition:
    """Test decomposing circuits from sparse state preparation."""

    def test_w3_fidelity(self):
        state, n_bits = w_state(3)
        qc = prepare_state_sparse(state, n_bits, timeout=30)
        out = decompose_to_clifford_t(qc, precision_bits=10)
        assert not _has_ry(out)
        fid = _fidelity(qc, out)
        assert fid >= 1 - 1e-3, f"fidelity {fid} too low"

    def test_w4_no_ry(self):
        state, n_bits = w_state(4)
        qc = prepare_state_sparse(state, n_bits, timeout=30)
        out = decompose_to_clifford_t(qc, precision_bits=8)
        assert not _has_ry(out)


class TestDenseDecomposition:
    """Test decomposing circuits from dense state preparation."""

    def test_dense_uniform_3_fidelity(self):
        state, n_bits = dense_uniform(3)
        qc = prepare_state_dense(state, n_bits, timeout=30)
        out = decompose_to_clifford_t(qc, precision_bits=10)
        assert not _has_ry(out)
        fid = _fidelity(qc, out)
        assert fid >= 1 - 1e-3, f"fidelity {fid} too low"
