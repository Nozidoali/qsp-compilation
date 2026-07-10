"""Alias sampling state preparation via QROM.

Walker's algorithm builds an alias table from a probability distribution.
The quantum circuit uses Hadamard superposition, QROM data loads, a comparator,
and controlled-swap to prepare the target state.

Ported from ``alias-sampling.cpp``.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List

from qiskit.circuit import QuantumCircuit

from qsp.circuit_utils import (
    add_comparator, add_comparator_ct,
    add_controlled_swap, add_controlled_swap_ct,
)
from qsp.qrom.synthesis import (
    QROMResult,
    compute_optimal_lambda,
    select_qrom_t_count,
    select_swap_qrom_t_count,
    synthesize_qrom,
)
from qsp.state import State


@dataclass
class AliasTable:
    keep_values: List[int] = field(default_factory=list)
    alias_indices: List[int] = field(default_factory=list)
    n_bins: int = 0
    precision_bits: int = 0


@dataclass
class AliasSamplingResult:
    qc: QuantumCircuit = field(default_factory=lambda: QuantumCircuit(0))
    t_count: int = 0
    cnot_count: int = 0
    qubit_count: int = 0
    keep_qrom_t: int = 0
    alias_qrom_t: int = 0
    compare_t: int = 0
    cswap_t: int = 0
    # Index-mapping QROM cost (sparse alias sampling only; 0 for the dense path).
    index_qrom_t: int = 0
    compile_time_ms: float = 0.0
    # Number of bins / address bits actually used (sparse compresses to the
    # next power of two above the support size; dense uses 2^n_bits).
    n_bins: int = 0
    addr_bits: int = 0
    # SELECT-SWAP block sizes (lambda_used in underlying QROMs). Both 0 for
    # the basic SELECT (non-swap) path, since that doesn't have a block-size
    # parameter.
    keep_lambda: int = 0
    alias_lambda: int = 0
    index_lambda: int = 0


def _amplitudes_to_probabilities(state: State, n_bits: int) -> List[float]:
    dim = 1 << n_bits
    probs = [0.0] * dim
    for idx, amp in state.items():
        probs[idx] = amp * amp
    return probs


def _build_alias_table_from_probs(probs: List[float], precision_bits: int) -> AliasTable:
    """Walker alias-table construction over an explicit probability vector.

    ``len(probs)`` is the number of bins ``N`` (assumed a power of two by the
    callers). Probabilities are scaled by ``N`` so a fair bin has weight 1.0.
    Alias indices reference positions within ``[0, N)``.
    """
    N = len(probs)
    max_val = (1 << precision_bits) - 1

    scaled = [p * N for p in probs]

    small: deque = deque()
    large: deque = deque()
    for i in range(N):
        if scaled[i] < 1.0:
            small.append(i)
        else:
            large.append(i)

    keep_prob = [1.0] * N
    alias = [0] * N

    while small and large:
        s = small.popleft()
        l = large.popleft()
        keep_prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = (scaled[l] + scaled[s]) - 1.0
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    while large:
        keep_prob[large.popleft()] = 1.0
    while small:
        keep_prob[small.popleft()] = 1.0

    table = AliasTable(n_bins=N, precision_bits=precision_bits)
    for i in range(N):
        clamped = max(0.0, min(1.0, keep_prob[i]))
        table.keep_values.append(int(round(clamped * max_val)))
        table.alias_indices.append(alias[i])

    return table


def build_alias_table(state: State, n_bits: int, precision_bits: int) -> AliasTable:
    """Dense alias table: one bin per basis state, ``N = 2^n_bits`` bins."""
    probs = _amplitudes_to_probabilities(state, n_bits)
    return _build_alias_table_from_probs(probs, precision_bits)


def _sparse_addr_bits(card: int) -> int:
    """Number of address bits for ``card`` nonzero entries, rounded up so the
    bin count is the next power of two (>= 1 bit, so SELECT-SWAP is well-formed).
    """
    if card <= 1:
        return 1
    return (card - 1).bit_length()


def build_sparse_alias_table(state: State, precision_bits: int):
    """Alias table over only the nonzero support, compressed to ``2^m`` bins
    where ``m = ceil(log2(card))``.

    Returns ``(table, index_map, m_bits)`` where ``table`` is the alias table
    over the compressed bins, ``index_map[j]`` is the real basis-state index of
    compressed slot ``j`` (padding slots map to 0 but are never kept), and
    ``m_bits`` is the compressed address width. The compressed alias sampling
    must be followed by an index-mapping QROM ``j -> index_map[j]`` to recover
    the real basis state.
    """
    items = sorted((idx, amp) for idx, amp in state.items() if amp != 0.0)
    card = len(items)
    m_bits = _sparse_addr_bits(card)
    n_bins = 1 << m_bits

    total = sum(amp * amp for _, amp in items)
    probs = [0.0] * n_bins
    index_map = [0] * n_bins
    for slot, (idx, amp) in enumerate(items):
        probs[slot] = (amp * amp) / total if total > 0 else 0.0
        index_map[slot] = idx

    table = _build_alias_table_from_probs(probs, precision_bits)
    return table, index_map, m_bits


def synthesize_alias_sampling(
    state: State,
    n_bits: int,
    precision_bits: int = 8,
    use_select_swap: bool = True,
    clean_ancilla: bool = True,
    clifford_t: bool = False,
) -> AliasSamplingResult:
    start = time.monotonic()

    n = n_bits
    b = precision_bits

    table = build_alias_table(state, n, b)

    keep_res = synthesize_qrom(
        table.keep_values, n, b,
        use_select_swap=use_select_swap,
        clean_ancilla=clean_ancilla,
        clifford_t=clifford_t,
    )
    alias_res = synthesize_qrom(
        table.alias_indices, n, n,
        use_select_swap=use_select_swap,
        clean_ancilla=clean_ancilla,
        clifford_t=clifford_t,
    )

    nq = 0
    addr = list(range(nq, nq + n)); nq += n

    keep_start = nq
    nq += keep_res.qc.num_qubits
    keep_out = [keep_start + keep_res.output_qubits[i] for i in range(b)]

    rand_q = list(range(nq, nq + b)); nq += b
    flag = nq; nq += 1

    alias_start = nq
    nq += alias_res.qc.num_qubits
    alias_out = [alias_start + alias_res.output_qubits[i] for i in range(n)]

    # Allocate ancillae for comparator and controlled-swap if Clifford+T mode
    compare_anc = []
    cswap_anc = []
    if clifford_t:
        compare_anc = list(range(nq, nq + b)); nq += b
        cswap_anc = list(range(nq, nq + n)); nq += n

    qc = QuantumCircuit(nq)

    for i in range(n):
        qc.h(addr[i])
    for i in range(b):
        qc.h(rand_q[i])

    keep_map = list(range(keep_start, keep_start + keep_res.qc.num_qubits))
    for i in range(n):
        keep_map[keep_res.input_qubits[i]] = addr[i]
    qc.compose(keep_res.qc, qubits=keep_map, inplace=True)

    if clifford_t:
        compare_t = add_comparator_ct(qc, keep_out, rand_q, flag, compare_anc)
    else:
        compare_ccx = add_comparator(qc, keep_out, rand_q, flag)

    alias_map = list(range(alias_start, alias_start + alias_res.qc.num_qubits))
    for i in range(n):
        alias_map[alias_res.input_qubits[i]] = addr[i]
    qc.compose(alias_res.qc, qubits=alias_map, inplace=True)

    if clifford_t:
        cswap_t, cswap_cx = add_controlled_swap_ct(qc, flag, addr, alias_out, cswap_anc)
    else:
        cswap_ccx, cswap_cx = add_controlled_swap(qc, flag, addr, alias_out)

    elapsed = (time.monotonic() - start) * 1000.0

    if clifford_t:
        total_t = keep_res.t_count + alias_res.t_count + compare_t + cswap_t
        result = AliasSamplingResult(
            qc=qc,
            t_count=total_t,
            cnot_count=qc.count_ops().get("cx", 0),
            qubit_count=nq,
            keep_qrom_t=keep_res.t_count,
            alias_qrom_t=alias_res.t_count,
            compare_t=compare_t,
            cswap_t=cswap_t,
            compile_time_ms=elapsed,
            n_bins=1 << n,
            addr_bits=n,
            keep_lambda=keep_res.lambda_used,
            alias_lambda=alias_res.lambda_used,
        )
    else:
        t_per_ccx = 4 if clean_ancilla else 7
        result = AliasSamplingResult(
            qc=qc,
            t_count=(keep_res.t_count + alias_res.t_count +
                     compare_ccx * t_per_ccx + cswap_ccx * t_per_ccx),
            cnot_count=qc.count_ops().get("cx", 0),
            qubit_count=nq,
            keep_qrom_t=keep_res.t_count,
            alias_qrom_t=alias_res.t_count,
            compare_t=compare_ccx * t_per_ccx,
            cswap_t=cswap_ccx * t_per_ccx,
            compile_time_ms=elapsed,
            n_bins=1 << n,
            addr_bits=n,
            keep_lambda=keep_res.lambda_used,
            alias_lambda=alias_res.lambda_used,
        )
    return result


def estimate_alias_sampling_t_count(
    n_qubits: int,
    precision_bits: int,
    use_select_swap: bool = True,
    clean_ancilla: bool = True,
) -> int:
    N = 1 << n_qubits
    t_per_ccx = 4 if clean_ancilla else 7

    if use_select_swap:
        keep_lam = compute_optimal_lambda(N, precision_bits)
        keep_t = select_swap_qrom_t_count(N, precision_bits, keep_lam, t_per_ccx)
        alias_lam = compute_optimal_lambda(N, n_qubits)
        alias_t = select_swap_qrom_t_count(N, n_qubits, alias_lam, t_per_ccx)
    else:
        keep_t = select_qrom_t_count(N, t_per_ccx)
        alias_t = select_qrom_t_count(N, t_per_ccx)

    compare_ccx = precision_bits
    cswap_ccx = n_qubits

    return keep_t + alias_t + (compare_ccx + cswap_ccx) * t_per_ccx


def synthesize_sparse_alias_sampling(
    state: State,
    n_bits: int,
    precision_bits: int = 8,
    use_select_swap: bool = True,
    clean_ancilla: bool = True,
    clifford_t: bool = False,
) -> AliasSamplingResult:
    """Alias sampling over the compressed support.

    Instead of one bin per basis state (``2^n_bits`` bins), this builds the
    alias table over ``2^m`` bins where ``m = ceil(log2(card))`` covers only the
    nonzero support, then appends an index-mapping QROM that turns the selected
    compressed slot into its real ``n_bits``-wide basis index. Every QROM
    addresses ``2^m`` entries rather than ``2^n_bits``, so for a sparse state
    (``card << 2^n_bits``) the T-cost shrinks accordingly.
    """
    start = time.monotonic()

    b = precision_bits
    table, index_map, m = build_sparse_alias_table(state, b)

    keep_res = synthesize_qrom(
        table.keep_values, m, b,
        use_select_swap=use_select_swap,
        clean_ancilla=clean_ancilla,
        clifford_t=clifford_t,
    )
    alias_res = synthesize_qrom(
        table.alias_indices, m, m,
        use_select_swap=use_select_swap,
        clean_ancilla=clean_ancilla,
        clifford_t=clifford_t,
    )
    index_res = synthesize_qrom(
        index_map, m, n_bits,
        use_select_swap=use_select_swap,
        clean_ancilla=clean_ancilla,
        clifford_t=clifford_t,
    )

    nq = 0
    addr = list(range(nq, nq + m)); nq += m

    keep_start = nq
    nq += keep_res.qc.num_qubits
    keep_out = [keep_start + keep_res.output_qubits[i] for i in range(b)]

    rand_q = list(range(nq, nq + b)); nq += b
    flag = nq; nq += 1

    alias_start = nq
    nq += alias_res.qc.num_qubits
    alias_out = [alias_start + alias_res.output_qubits[i] for i in range(m)]

    index_start = nq
    nq += index_res.qc.num_qubits
    # index_res.output_qubits hold the recovered real basis index after compose.

    compare_anc = []
    cswap_anc = []
    if clifford_t:
        compare_anc = list(range(nq, nq + b)); nq += b
        cswap_anc = list(range(nq, nq + m)); nq += m

    qc = QuantumCircuit(nq)

    for i in range(m):
        qc.h(addr[i])
    for i in range(b):
        qc.h(rand_q[i])

    keep_map = list(range(keep_start, keep_start + keep_res.qc.num_qubits))
    for i in range(m):
        keep_map[keep_res.input_qubits[i]] = addr[i]
    qc.compose(keep_res.qc, qubits=keep_map, inplace=True)

    if clifford_t:
        compare_t = add_comparator_ct(qc, keep_out, rand_q, flag, compare_anc)
    else:
        compare_ccx = add_comparator(qc, keep_out, rand_q, flag)

    alias_map = list(range(alias_start, alias_start + alias_res.qc.num_qubits))
    for i in range(m):
        alias_map[alias_res.input_qubits[i]] = addr[i]
    qc.compose(alias_res.qc, qubits=alias_map, inplace=True)

    if clifford_t:
        cswap_t, cswap_cx = add_controlled_swap_ct(qc, flag, addr, alias_out, cswap_anc)
    else:
        cswap_ccx, cswap_cx = add_controlled_swap(qc, flag, addr, alias_out)

    # Index-mapping QROM: read the selected compressed slot (addr) and write
    # the real basis index into a fresh n_bits register.
    index_map_qubits = list(range(index_start, index_start + index_res.qc.num_qubits))
    for i in range(m):
        index_map_qubits[index_res.input_qubits[i]] = addr[i]
    qc.compose(index_res.qc, qubits=index_map_qubits, inplace=True)

    elapsed = (time.monotonic() - start) * 1000.0

    if clifford_t:
        compare_cost = compare_t
        cswap_cost = cswap_t
    else:
        t_per_ccx = 4 if clean_ancilla else 7
        compare_cost = compare_ccx * t_per_ccx
        cswap_cost = cswap_ccx * t_per_ccx

    total_t = (keep_res.t_count + alias_res.t_count + index_res.t_count +
               compare_cost + cswap_cost)

    return AliasSamplingResult(
        qc=qc,
        t_count=total_t,
        cnot_count=qc.count_ops().get("cx", 0),
        qubit_count=nq,
        keep_qrom_t=keep_res.t_count,
        alias_qrom_t=alias_res.t_count,
        compare_t=compare_cost,
        cswap_t=cswap_cost,
        index_qrom_t=index_res.t_count,
        compile_time_ms=elapsed,
        n_bins=1 << m,
        addr_bits=m,
        keep_lambda=keep_res.lambda_used,
        alias_lambda=alias_res.lambda_used,
        index_lambda=index_res.lambda_used,
    )


def estimate_sparse_alias_sampling_t_count(
    card: int,
    n_bits: int,
    precision_bits: int,
    use_select_swap: bool = True,
    clean_ancilla: bool = True,
) -> int:
    """Analytical T-count for sparse alias sampling with ``card`` nonzero
    amplitudes in an ``n_bits``-qubit register. Mirrors
    :func:`estimate_alias_sampling_t_count` but over the compressed
    ``2^m`` bins plus the index-mapping QROM (``2^m`` addresses, ``n_bits`` data).
    """
    m = _sparse_addr_bits(card)
    Np = 1 << m
    t_per_ccx = 4 if clean_ancilla else 7

    if use_select_swap:
        keep_t = select_swap_qrom_t_count(
            Np, precision_bits, compute_optimal_lambda(Np, precision_bits), t_per_ccx)
        alias_t = select_swap_qrom_t_count(
            Np, m, compute_optimal_lambda(Np, m), t_per_ccx)
        index_t = select_swap_qrom_t_count(
            Np, n_bits, compute_optimal_lambda(Np, n_bits), t_per_ccx)
    else:
        keep_t = select_qrom_t_count(Np, t_per_ccx)
        alias_t = select_qrom_t_count(Np, t_per_ccx)
        index_t = select_qrom_t_count(Np, t_per_ccx)

    compare_ccx = precision_bits
    cswap_ccx = m

    return keep_t + alias_t + index_t + (compare_ccx + cswap_ccx) * t_per_ccx
