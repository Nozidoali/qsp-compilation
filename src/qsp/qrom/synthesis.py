"""QROM circuit construction: basic Select and SelectSwap.

Ported from ``qrom-synthesis.cpp``.

Basic Select uses unary iteration (binary address -> one-hot encoding via CCX
tree), then CX-based data loading, then uncomputation.

SelectSwap (Babbush et al. arXiv:1812.00954) splits address into quotient and
remainder, uses lambda parallel data registers, and a swap network to route the
selected register to output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from qiskit.circuit import QuantumCircuit

from qsp.circuit_utils import and_compute, and_uncompute


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class QROMResult:
    qc: QuantumCircuit = field(default_factory=lambda: QuantumCircuit(0))
    t_count: int = 0
    ccx_count: int = 0
    cx_count: int = 0
    x_count: int = 0
    ancilla_count: int = 0
    lambda_used: int = 0
    input_qubits: List[int] = field(default_factory=list)
    output_qubits: List[int] = field(default_factory=list)


def _build_unary_iteration(
    qc: QuantumCircuit,
    addr: List[int],
    unary: List[int],
) -> None:
    n = len(addr)
    N = len(unary)
    assert N == (1 << n)

    qc.x(addr[0])
    qc.cx(addr[0], unary[0])
    qc.x(addr[0])
    qc.cx(addr[0], unary[1])

    for level in range(1, n):
        step = 1 << level
        for j in range(step - 1, -1, -1):
            t = j + step
            if t >= N:
                continue
            qc.ccx(unary[j], addr[level], unary[t])
            qc.cx(unary[t], unary[j])


def _uncompute_unary_iteration(
    qc: QuantumCircuit,
    addr: List[int],
    unary: List[int],
) -> None:
    n = len(addr)
    N = len(unary)

    for level in range(n - 1, 0, -1):
        step = 1 << level
        for j in range(step):
            t = j + step
            if t >= N:
                continue
            qc.cx(unary[t], unary[j])
            qc.ccx(unary[j], addr[level], unary[t])

    qc.cx(addr[0], unary[1])
    qc.x(addr[0])
    qc.cx(addr[0], unary[0])
    qc.x(addr[0])


def _build_unary_iteration_ct(
    qc: QuantumCircuit,
    addr: List[int],
    unary: List[int],
    ancillae: List[int],
) -> None:
    """Unary iteration (compute) using AND gates (4T each) instead of CCX."""
    n = len(addr)
    N = len(unary)

    qc.x(addr[0])
    qc.cx(addr[0], unary[0])
    qc.x(addr[0])
    qc.cx(addr[0], unary[1])

    anc_idx = 0
    for level in range(1, n):
        step = 1 << level
        for j in range(step - 1, -1, -1):
            t = j + step
            if t >= N:
                continue
            and_compute(qc, unary[j], addr[level], ancillae[anc_idx])
            qc.cx(ancillae[anc_idx], unary[t])
            qc.cx(unary[t], unary[j])
            anc_idx += 1


def _uncompute_unary_iteration_ct(
    qc: QuantumCircuit,
    addr: List[int],
    unary: List[int],
    ancillae: List[int],
) -> None:
    """Unary iteration (uncompute) using AND† (4T each) to release ancillae."""
    n = len(addr)
    N = len(unary)

    # Walk in reverse order, matching _build_unary_iteration_ct
    anc_pairs = []
    for level in range(1, n):
        step = 1 << level
        for j in range(step - 1, -1, -1):
            t = j + step
            if t >= N:
                continue
            anc_pairs.append((level, j, t))

    for idx, (level, j, t) in enumerate(reversed(anc_pairs)):
        anc_idx = len(anc_pairs) - 1 - idx
        qc.cx(unary[t], unary[j])
        qc.cx(ancillae[anc_idx], unary[t])
        and_uncompute(qc, unary[j], addr[level], ancillae[anc_idx])

    qc.cx(addr[0], unary[1])
    qc.x(addr[0])
    qc.cx(addr[0], unary[0])
    qc.x(addr[0])


def _count_unary_ccx(addr_bits: int, n_unary: int) -> int:
    """Count CCX gates in unary iteration (for ancilla allocation)."""
    count = 0
    for level in range(1, addr_bits):
        step = 1 << level
        for j in range(step - 1, -1, -1):
            if j + step < n_unary:
                count += 1
    return count


def _load_data(
    qc: QuantumCircuit,
    unary: List[int],
    target: List[int],
    data: List[int],
    data_bits: int,
) -> int:
    cx_count = 0
    for i, val in enumerate(data):
        for b in range(data_bits):
            if (val >> b) & 1:
                qc.cx(unary[i], target[b])
                cx_count += 1
    return cx_count


def synthesize_basic_select(
    data: List[int],
    addr_bits: int,
    data_bits: int,
    clean_ancilla: bool = True,
) -> QROMResult:
    N = len(data)
    n = addr_bits
    b = data_bits

    total = n + N + b
    qc = QuantumCircuit(total)

    addr = list(range(n))
    unary = list(range(n, n + N))
    target = list(range(n + N, n + N + b))

    _build_unary_iteration(qc, addr, unary)
    cx_count = _load_data(qc, unary, target, data, b)
    _uncompute_unary_iteration(qc, addr, unary)

    t_per_ccx = 4 if clean_ancilla else 7
    ccx_count = qc.count_ops().get("ccx", 0)

    return QROMResult(
        qc=qc,
        t_count=ccx_count * t_per_ccx,
        ccx_count=ccx_count,
        cx_count=cx_count,
        ancilla_count=N,
        lambda_used=0,
        input_qubits=addr,
        output_qubits=target,
    )


def synthesize_select_swap(
    data: List[int],
    addr_bits: int,
    data_bits: int,
    lam: int,
    clean_ancilla: bool = True,
) -> QROMResult:
    N = len(data)
    n = addr_bits
    b = data_bits

    r_bits = max(1, (lam - 1).bit_length()) if lam > 1 else 1
    q_bits = n - r_bits
    n_quotients = _ceil_div(N, lam)

    nq = 0
    addr = list(range(n)); nq += n
    rem_q = addr[:r_bits]
    quot_q = addr[r_bits:]

    unary = list(range(nq, nq + n_quotients)); nq += n_quotients

    reg_groups: List[List[int]] = []
    for _ in range(lam):
        reg = list(range(nq, nq + b)); nq += b
        reg_groups.append(reg)

    output = list(range(nq, nq + b)); nq += b

    qc = QuantumCircuit(nq)

    if n_quotients > 1:
        _build_unary_iteration(qc, quot_q, unary)
    else:
        qc.x(unary[0])

    cx_count = 0
    for q_idx in range(n_quotients):
        for r_idx in range(lam):
            idx = q_idx * lam + r_idx
            if idx >= N:
                break
            val = data[idx]
            for j in range(b):
                if (val >> j) & 1:
                    qc.cx(unary[q_idx], reg_groups[r_idx][j])
                    cx_count += 1

    if n_quotients > 1:
        _uncompute_unary_iteration(qc, quot_q, unary)
    else:
        qc.x(unary[0])

    swap_ccx = 0
    swap_cx = 0
    for bit in range(r_bits):
        for reg in range(lam):
            if not (reg & (1 << bit)):
                partner = reg | (1 << bit)
                if partner >= lam:
                    continue
                for j in range(b):
                    qc.cx(reg_groups[reg][j], reg_groups[partner][j])
                    qc.ccx(rem_q[bit], reg_groups[partner][j], reg_groups[reg][j])
                    qc.cx(reg_groups[reg][j], reg_groups[partner][j])
                    swap_ccx += 1
                    swap_cx += 2

    for j in range(b):
        qc.cx(reg_groups[0][j], output[j])
        cx_count += 1

    t_per_ccx = 4 if clean_ancilla else 7
    ccx_total = qc.count_ops().get("ccx", 0)

    return QROMResult(
        qc=qc,
        t_count=ccx_total * t_per_ccx,
        ccx_count=ccx_total,
        cx_count=cx_count + swap_cx,
        ancilla_count=n_quotients + lam * b,
        lambda_used=lam,
        input_qubits=addr,
        output_qubits=output,
    )


def synthesize_basic_select_ct(
    data: List[int],
    addr_bits: int,
    data_bits: int,
) -> QROMResult:
    """Basic select QROM with native Clifford+T gates (no CCX)."""
    N = len(data)
    n = addr_bits
    b = data_bits

    n_unary_ccx = _count_unary_ccx(n, N)
    n_anc = n_unary_ccx  # one ancilla per AND gate (shared across compute/uncompute)

    nq = 0
    addr = list(range(nq, nq + n)); nq += n
    unary = list(range(nq, nq + N)); nq += N
    target = list(range(nq, nq + b)); nq += b
    ancillae = list(range(nq, nq + n_anc)); nq += n_anc

    qc = QuantumCircuit(nq)

    _build_unary_iteration_ct(qc, addr, unary, ancillae)
    cx_count = _load_data(qc, unary, target, data, b)
    _uncompute_unary_iteration_ct(qc, addr, unary, ancillae)

    t_count = n_unary_ccx * 2 * 4  # 4T compute + 4T uncompute per CCX

    return QROMResult(
        qc=qc,
        t_count=t_count,
        ccx_count=0,
        cx_count=cx_count,
        ancilla_count=N + n_anc,
        lambda_used=0,
        input_qubits=addr,
        output_qubits=target,
    )


def synthesize_select_swap_ct(
    data: List[int],
    addr_bits: int,
    data_bits: int,
    lam: int,
) -> QROMResult:
    """SelectSwap QROM with native Clifford+T gates (no CCX)."""
    N = len(data)
    n = addr_bits
    b = data_bits

    r_bits = max(1, (lam - 1).bit_length()) if lam > 1 else 1
    n_quotients = _ceil_div(N, lam)
    rem_q_bits = r_bits
    quot_q_bits = n - r_bits

    # Count ancillae needed
    n_unary_ccx = _count_unary_ccx(quot_q_bits, n_quotients) if n_quotients > 1 else 0
    # Count swap CCX
    n_swap_ccx = 0
    for bit in range(r_bits):
        for reg in range(lam):
            if not (reg & (1 << bit)):
                partner = reg | (1 << bit)
                if partner < lam:
                    n_swap_ccx += b
    n_anc = max(n_unary_ccx, n_swap_ccx)  # reuse ancillae between phases

    nq = 0
    addr = list(range(n)); nq += n
    rem_q = addr[:r_bits]
    quot_q = addr[r_bits:]

    unary = list(range(nq, nq + n_quotients)); nq += n_quotients

    reg_groups: List[List[int]] = []
    for _ in range(lam):
        reg = list(range(nq, nq + b)); nq += b
        reg_groups.append(reg)

    output = list(range(nq, nq + b)); nq += b
    ancillae = list(range(nq, nq + n_anc)); nq += n_anc

    qc = QuantumCircuit(nq)

    if n_quotients > 1:
        _build_unary_iteration_ct(qc, quot_q, unary, ancillae[:n_unary_ccx])
    else:
        qc.x(unary[0])

    cx_count = 0
    for q_idx in range(n_quotients):
        for r_idx in range(lam):
            idx = q_idx * lam + r_idx
            if idx >= N:
                break
            val = data[idx]
            for j in range(b):
                if (val >> j) & 1:
                    qc.cx(unary[q_idx], reg_groups[r_idx][j])
                    cx_count += 1

    if n_quotients > 1:
        _uncompute_unary_iteration_ct(qc, quot_q, unary, ancillae[:n_unary_ccx])
    else:
        qc.x(unary[0])

    # Swap network using AND gates
    swap_cx = 0
    swap_t = 0
    anc_idx = 0
    for bit in range(r_bits):
        for reg in range(lam):
            if not (reg & (1 << bit)):
                partner = reg | (1 << bit)
                if partner >= lam:
                    continue
                for j in range(b):
                    qc.cx(reg_groups[reg][j], reg_groups[partner][j])
                    and_compute(qc, rem_q[bit], reg_groups[partner][j], ancillae[anc_idx])
                    qc.cx(ancillae[anc_idx], reg_groups[reg][j])
                    and_uncompute(qc, rem_q[bit], reg_groups[partner][j], ancillae[anc_idx])
                    qc.cx(reg_groups[reg][j], reg_groups[partner][j])
                    anc_idx += 1
                    swap_t += 8
                    swap_cx += 2

    for j in range(b):
        qc.cx(reg_groups[0][j], output[j])
        cx_count += 1

    t_count = n_unary_ccx * 2 * 4 + swap_t

    return QROMResult(
        qc=qc,
        t_count=t_count,
        ccx_count=0,
        cx_count=cx_count + swap_cx,
        ancilla_count=n_quotients + lam * b + n_anc,
        lambda_used=lam,
        input_qubits=addr,
        output_qubits=output,
    )


def compute_optimal_lambda(n_entries: int, data_bits: int) -> int:
    if n_entries <= 2 or data_bits == 0:
        return 1
    best_cost = float("inf")
    best_lam = 1
    max_lam = min(n_entries, 64)
    lam = 1
    while lam <= max_lam:
        cost = 4.0 * _ceil_div(n_entries, lam) + 4.0 * data_bits * lam
        if cost < best_cost:
            best_cost = cost
            best_lam = lam
        lam *= 2
    return best_lam


def select_qrom_t_count(n_entries: int, t_per_ccx: int = 4) -> int:
    if n_entries <= 1:
        return 0
    return 2 * (n_entries - 1) * t_per_ccx


def select_swap_qrom_t_count(
    n_entries: int, data_bits: int, lam: int, t_per_ccx: int = 4
) -> int:
    n_quotients = _ceil_div(n_entries, lam)
    select_ccx = 2 * (n_quotients - 1) if n_quotients > 1 else 0

    r_bits = max(1, (lam - 1).bit_length()) if lam > 1 else 0
    swap_ccx = r_bits * data_bits * (lam // 2)

    return (select_ccx + swap_ccx) * t_per_ccx


def synthesize_qrom(
    data: List[int],
    addr_bits: int,
    data_bits: int,
    use_select_swap: bool = True,
    lam: int = 0,
    clean_ancilla: bool = True,
    clifford_t: bool = False,
) -> QROMResult:
    assert data
    assert addr_bits > 0
    assert data_bits > 0

    N = 1 << addr_bits
    padded = list(data) + [0] * (N - len(data))

    if clifford_t:
        if not use_select_swap:
            return synthesize_basic_select_ct(padded, addr_bits, data_bits)
        if lam == 0:
            lam = compute_optimal_lambda(N, data_bits)
        if lam <= 1 or N <= 2:
            return synthesize_basic_select_ct(padded, addr_bits, data_bits)
        return synthesize_select_swap_ct(padded, addr_bits, data_bits, lam)

    if not use_select_swap:
        return synthesize_basic_select(padded, addr_bits, data_bits, clean_ancilla)

    if lam == 0:
        lam = compute_optimal_lambda(N, data_bits)

    if lam <= 1 or N <= 2:
        return synthesize_basic_select(padded, addr_bits, data_bits, clean_ancilla)

    return synthesize_select_swap(padded, addr_bits, data_bits, lam, clean_ancilla)
