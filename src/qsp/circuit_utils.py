"""Shared circuit primitives: comparator, controlled-swap, Clifford+T decomposition."""
from __future__ import annotations

import math
import warnings
from typing import Dict, Tuple

from qiskit.circuit import QuantumCircuit


def add_comparator(
    qc: QuantumCircuit,
    a_qubits: list[int],
    b_qubits: list[int],
    flag_qubit: int,
) -> int:
    """Compare a < b using carry chain. Sets flag=1 iff a < b.

    Uses Cuccaro-style MAJ gates to propagate the borrow of (a - b).
    flag must start at |0>. Modifies b_qubits in place (treated as scratch).
    """
    bits = len(a_qubits)
    assert len(a_qubits) == len(b_qubits)
    ccx_count = 0

    for i in range(bits):
        qc.x(a_qubits[i])

    for i in range(bits):
        qc.cx(flag_qubit, b_qubits[i])
        qc.cx(flag_qubit, a_qubits[i])
        qc.ccx(a_qubits[i], b_qubits[i], flag_qubit)
        ccx_count += 1

    return ccx_count


def add_controlled_swap(
    qc: QuantumCircuit,
    ctrl: int,
    reg_a: list[int],
    reg_b: list[int],
) -> tuple[int, int]:
    assert len(reg_a) == len(reg_b)
    ccx_count = 0
    cx_count = 0
    for i in range(len(reg_a)):
        qc.cx(reg_a[i], reg_b[i])
        qc.ccx(ctrl, reg_b[i], reg_a[i])
        qc.cx(reg_a[i], reg_b[i])
        ccx_count += 1
        cx_count += 2
    return ccx_count, cx_count


# --- Clifford+T decomposition ---

_gridsynth_cache: Dict[Tuple[int, int], str] = {}


def _emit_clifford_rz(qc: QuantumCircuit, theta: float, qubit: int) -> bool:
    """Emit exact Clifford gates for Rz angles that are multiples of pi/4.

    Returns True if the angle was handled, False otherwise.
    """
    # Normalize to [0, 2π)
    t = theta % (2 * math.pi)
    # Check multiples of π/4
    k = round(t / (math.pi / 4))
    if abs(t - k * math.pi / 4) > 1e-9:
        return False
    k = int(k) % 8
    # k=0: identity, k=1: T, k=2: S, k=3: S·T, k=4: Z, k=5: Z·T, k=6: Sdg, k=7: Tdg
    if k == 0:
        pass
    elif k == 1:
        qc.t(qubit)
    elif k == 2:
        qc.s(qubit)
    elif k == 3:
        qc.s(qubit); qc.t(qubit)
    elif k == 4:
        qc.z(qubit)
    elif k == 5:
        qc.z(qubit); qc.t(qubit)
    elif k == 6:
        qc.sdg(qubit)
    elif k == 7:
        qc.tdg(qubit)
    return True


def _gridsynth_decompose(theta: float, precision_bits: int) -> str:
    """Return gridsynth gate string for Rz(theta), with caching."""
    key = (round(theta * 1e10), precision_bits)
    if key not in _gridsynth_cache:
        warnings.filterwarnings("ignore")
        from pygridsynth import gridsynth_gates
        epsilon = 2.0 ** (-precision_bits)
        _gridsynth_cache[key] = gridsynth_gates(theta, epsilon)
    return _gridsynth_cache[key]


def _emit_gridsynth_gates(qc: QuantumCircuit, gate_str: str, qubit: int) -> None:
    """Emit Clifford+T gates from a gridsynth gate string onto *qubit*."""
    for ch in gate_str:
        if ch == "H":
            qc.h(qubit)
        elif ch == "T":
            qc.t(qubit)
        elif ch == "S":
            qc.s(qubit)
        elif ch == "X":
            qc.x(qubit)
        # 'W' is global phase — skip


def _flatten(qc: QuantumCircuit, max_rounds: int = 10) -> QuantumCircuit:
    """Decompose until no multi-qubit controlled-RY gates remain."""
    basic = {"ry", "cx", "x", "h", "s", "t", "sdg", "tdg", "z", "id",
             "ccx", "rz", "sx", "sxdg", "cry", "measure", "barrier"}
    flat = qc
    for _ in range(max_rounds):
        needs_more = False
        for instr in flat.data:
            if instr.operation.name not in basic:
                needs_more = True
                break
        if not needs_more:
            break
        flat = flat.decompose()
    return flat


def decompose_to_clifford_t(
    qc: QuantumCircuit,
    precision_bits: int,
) -> QuantumCircuit:
    """Replace RY gates with Clifford+T sequences via gridsynth.

    Identity: RY(θ) = S† · H · Rz(θ) · H · S
    Circuit order: sdg, h, [gridsynth Rz gates], h, s

    Other gates (x, cx, ccx, h, s, t, etc.) are copied as-is.
    """
    flat = _flatten(qc)
    out = QuantumCircuit(flat.num_qubits)

    for instr in flat.data:
        name = instr.operation.name
        qubits = [flat.find_bit(q).index for q in instr.qubits]

        if name == "ry":
            theta = float(instr.operation.params[0])
            # Trivial: theta ~ 0 → skip
            if abs(theta) < 1e-12:
                continue
            # Trivial: theta ~ ±π → X gate
            if abs(abs(theta) - math.pi) < 1e-12:
                out.x(qubits[0])
                continue
            # RY(θ) = S† · H · Rz(θ) · H · S
            q = qubits[0]
            out.sdg(q)
            out.h(q)
            gate_str = _gridsynth_decompose(theta, precision_bits)
            _emit_gridsynth_gates(out, gate_str, q)
            out.h(q)
            out.s(q)

        elif name == "cry":
            theta = float(instr.operation.params[0])
            if abs(theta) < 1e-12:
                continue
            # Decompose CRY into: RY(θ/2) on target, CX, RY(-θ/2) on target, CX
            ctrl, tgt = qubits
            half = theta / 2.0
            # Emit RY(θ/2) as Clifford+T
            out.sdg(tgt)
            out.h(tgt)
            gate_str = _gridsynth_decompose(half, precision_bits)
            _emit_gridsynth_gates(out, gate_str, tgt)
            out.h(tgt)
            out.s(tgt)
            out.cx(ctrl, tgt)
            # Emit RY(-θ/2) as Clifford+T
            out.sdg(tgt)
            out.h(tgt)
            gate_str = _gridsynth_decompose(-half, precision_bits)
            _emit_gridsynth_gates(out, gate_str, tgt)
            out.h(tgt)
            out.s(tgt)
            out.cx(ctrl, tgt)

        elif name == "rz":
            theta = float(instr.operation.params[0])
            if abs(theta) < 1e-12:
                continue
            q = qubits[0]
            if not _emit_clifford_rz(out, theta, q):
                gate_str = _gridsynth_decompose(theta, precision_bits)
                _emit_gridsynth_gates(out, gate_str, q)

        elif name == "u":
            # U(θ, φ, λ) = Rz(φ) · RY(θ) · Rz(λ)
            theta = float(instr.operation.params[0])
            phi = float(instr.operation.params[1])
            lam = float(instr.operation.params[2])
            q = qubits[0]
            # Rz(λ)
            if abs(lam) > 1e-12:
                if not _emit_clifford_rz(out, lam, q):
                    gate_str = _gridsynth_decompose(lam, precision_bits)
                    _emit_gridsynth_gates(out, gate_str, q)
            # RY(θ)
            if abs(theta) < 1e-12:
                pass
            elif abs(abs(theta) - math.pi) < 1e-12:
                out.x(q)
            else:
                out.sdg(q)
                out.h(q)
                gate_str = _gridsynth_decompose(theta, precision_bits)
                _emit_gridsynth_gates(out, gate_str, q)
                out.h(q)
                out.s(q)
            # Rz(φ)
            if abs(phi) > 1e-12:
                if not _emit_clifford_rz(out, phi, q):
                    gate_str = _gridsynth_decompose(phi, precision_bits)
                    _emit_gridsynth_gates(out, gate_str, q)

        else:
            # Copy gate as-is
            out.append(instr.operation, qubits)

    return out
