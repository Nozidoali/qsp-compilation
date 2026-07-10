from __future__ import annotations
from dataclasses import dataclass
from typing import List
from qiskit.circuit import QuantumCircuit
from circuit_utils import and_compute, and_uncompute

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

@dataclass
class QROMResult:
    qc: QuantumCircuit
    input_qubits: List[int]
    output_qubits: List[int]

def _build_unary_iteration(qc: QuantumCircuit, addr: List[int], unary: List[int]) -> None:
    n = len(addr)
    N = len(unary)
    assert N == 1 << n
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

def _uncompute_unary_iteration(qc: QuantumCircuit, addr: List[int], unary: List[int]) -> None:
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

def _build_unary_iteration_ct(qc: QuantumCircuit, addr: List[int], unary: List[int], ancillae: List[int]) -> None:
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

def _uncompute_unary_iteration_ct(qc: QuantumCircuit, addr: List[int], unary: List[int], ancillae: List[int]) -> None:
    n = len(addr)
    N = len(unary)
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
    count = 0
    for level in range(1, addr_bits):
        step = 1 << level
        for j in range(step - 1, -1, -1):
            if j + step < n_unary:
                count += 1
    return count

def _load_data(qc: QuantumCircuit, unary: List[int], target: List[int], data: List[int], data_bits: int) -> None:
    for i, val in enumerate(data):
        for b in range(data_bits):
            if val >> b & 1:
                qc.cx(unary[i], target[b])

def synthesize_basic_select(data: List[int], addr_bits: int, data_bits: int) -> QROMResult:
    N = len(data)
    n = addr_bits
    b = data_bits
    qc = QuantumCircuit(n + N + b)
    addr = list(range(n))
    unary = list(range(n, n + N))
    target = list(range(n + N, n + N + b))
    _build_unary_iteration(qc, addr, unary)
    _load_data(qc, unary, target, data, b)
    _uncompute_unary_iteration(qc, addr, unary)
    return QROMResult(qc, addr, target)

def synthesize_select_swap(data: List[int], addr_bits: int, data_bits: int, lam: int) -> QROMResult:
    N = len(data)
    n = addr_bits
    b = data_bits
    r_bits = max(1, (lam - 1).bit_length()) if lam > 1 else 1
    n_quotients = _ceil_div(N, lam)
    nq = 0
    addr = list(range(n))
    nq += n
    rem_q = addr[:r_bits]
    quot_q = addr[r_bits:]
    unary = list(range(nq, nq + n_quotients))
    nq += n_quotients
    reg_groups: List[List[int]] = []
    for _ in range(lam):
        reg = list(range(nq, nq + b))
        nq += b
        reg_groups.append(reg)
    output = list(range(nq, nq + b))
    nq += b
    qc = QuantumCircuit(nq)
    if n_quotients > 1:
        _build_unary_iteration(qc, quot_q, unary)
    else:
        qc.x(unary[0])
    for q_idx in range(n_quotients):
        for r_idx in range(lam):
            idx = q_idx * lam + r_idx
            if idx >= N:
                break
            val = data[idx]
            for j in range(b):
                if val >> j & 1:
                    qc.cx(unary[q_idx], reg_groups[r_idx][j])
    if n_quotients > 1:
        _uncompute_unary_iteration(qc, quot_q, unary)
    else:
        qc.x(unary[0])
    for bit in range(r_bits):
        for reg in range(lam):
            if not reg & 1 << bit:
                partner = reg | 1 << bit
                if partner >= lam:
                    continue
                for j in range(b):
                    qc.cx(reg_groups[reg][j], reg_groups[partner][j])
                    qc.ccx(rem_q[bit], reg_groups[partner][j], reg_groups[reg][j])
                    qc.cx(reg_groups[reg][j], reg_groups[partner][j])
    for j in range(b):
        qc.cx(reg_groups[0][j], output[j])
    return QROMResult(qc, addr, output)

def synthesize_basic_select_ct(data: List[int], addr_bits: int, data_bits: int) -> QROMResult:
    N = len(data)
    n = addr_bits
    b = data_bits
    n_anc = _count_unary_ccx(n, N)
    nq = 0
    addr = list(range(nq, nq + n))
    nq += n
    unary = list(range(nq, nq + N))
    nq += N
    target = list(range(nq, nq + b))
    nq += b
    ancillae = list(range(nq, nq + n_anc))
    nq += n_anc
    qc = QuantumCircuit(nq)
    _build_unary_iteration_ct(qc, addr, unary, ancillae)
    _load_data(qc, unary, target, data, b)
    _uncompute_unary_iteration_ct(qc, addr, unary, ancillae)
    return QROMResult(qc, addr, target)

def synthesize_select_swap_ct(data: List[int], addr_bits: int, data_bits: int, lam: int) -> QROMResult:
    N = len(data)
    n = addr_bits
    b = data_bits
    r_bits = max(1, (lam - 1).bit_length()) if lam > 1 else 1
    n_quotients = _ceil_div(N, lam)
    quot_q_bits = n - r_bits
    n_unary_ccx = _count_unary_ccx(quot_q_bits, n_quotients) if n_quotients > 1 else 0
    n_swap_ccx = 0
    for bit in range(r_bits):
        for reg in range(lam):
            if not reg & 1 << bit:
                partner = reg | 1 << bit
                if partner < lam:
                    n_swap_ccx += b
    n_anc = max(n_unary_ccx, n_swap_ccx)
    nq = 0
    addr = list(range(n))
    nq += n
    rem_q = addr[:r_bits]
    quot_q = addr[r_bits:]
    unary = list(range(nq, nq + n_quotients))
    nq += n_quotients
    reg_groups: List[List[int]] = []
    for _ in range(lam):
        reg = list(range(nq, nq + b))
        nq += b
        reg_groups.append(reg)
    output = list(range(nq, nq + b))
    nq += b
    ancillae = list(range(nq, nq + n_anc))
    nq += n_anc
    qc = QuantumCircuit(nq)
    if n_quotients > 1:
        _build_unary_iteration_ct(qc, quot_q, unary, ancillae[:n_unary_ccx])
    else:
        qc.x(unary[0])
    for q_idx in range(n_quotients):
        for r_idx in range(lam):
            idx = q_idx * lam + r_idx
            if idx >= N:
                break
            val = data[idx]
            for j in range(b):
                if val >> j & 1:
                    qc.cx(unary[q_idx], reg_groups[r_idx][j])
    if n_quotients > 1:
        _uncompute_unary_iteration_ct(qc, quot_q, unary, ancillae[:n_unary_ccx])
    else:
        qc.x(unary[0])
    anc_idx = 0
    for bit in range(r_bits):
        for reg in range(lam):
            if not reg & 1 << bit:
                partner = reg | 1 << bit
                if partner >= lam:
                    continue
                for j in range(b):
                    qc.cx(reg_groups[reg][j], reg_groups[partner][j])
                    and_compute(qc, rem_q[bit], reg_groups[partner][j], ancillae[anc_idx])
                    qc.cx(ancillae[anc_idx], reg_groups[reg][j])
                    and_uncompute(qc, rem_q[bit], reg_groups[partner][j], ancillae[anc_idx])
                    qc.cx(reg_groups[reg][j], reg_groups[partner][j])
                    anc_idx += 1
    for j in range(b):
        qc.cx(reg_groups[0][j], output[j])
    return QROMResult(qc, addr, output)

def compute_optimal_lambda(n_entries: int, data_bits: int) -> int:
    if n_entries <= 2 or data_bits == 0:
        return 1
    best_cost = float('inf')
    best_lam = 1
    lam = 1
    while lam <= n_entries:
        cost = 4.0 * _ceil_div(n_entries, lam) + 4.0 * data_bits * lam
        if cost < best_cost:
            best_cost = cost
            best_lam = lam
        lam *= 2
    return best_lam

def synthesize_qrom(data: List[int], addr_bits: int, data_bits: int, use_select_swap: bool=True, lam: int=0, clifford_t: bool=False) -> QROMResult:
    assert data
    assert addr_bits > 0
    assert data_bits > 0
    N = 1 << addr_bits
    padded = list(data) + [0] * (N - len(data))
    if lam == 0:
        lam = compute_optimal_lambda(N, data_bits)
    use_swap = use_select_swap and lam > 1 and N > 2
    if clifford_t:
        if use_swap:
            return synthesize_select_swap_ct(padded, addr_bits, data_bits, lam)
        return synthesize_basic_select_ct(padded, addr_bits, data_bits)
    if use_swap:
        return synthesize_select_swap(padded, addr_bits, data_bits, lam)
    return synthesize_basic_select(padded, addr_bits, data_bits)
