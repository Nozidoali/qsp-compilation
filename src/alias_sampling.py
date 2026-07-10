from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import List
from qiskit.circuit import QuantumCircuit
from circuit_utils import add_comparator, add_comparator_ct, add_controlled_swap, add_controlled_swap_ct
from synthesis import synthesize_qrom
from state import State

@dataclass
class AliasTable:
    keep_values: List[int] = field(default_factory=list)
    alias_indices: List[int] = field(default_factory=list)
    n_bins: int = 0
    precision_bits: int = 0

def _amplitudes_to_probabilities(state: State, n_bits: int) -> List[float]:
    probs = [0.0] * (1 << n_bits)
    for idx, amp in state.items():
        probs[idx] = amp * amp
    return probs

def _build_alias_table_from_probs(probs: List[float], precision_bits: int) -> AliasTable:
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
        scaled[l] = scaled[l] + scaled[s] - 1.0
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
    return _build_alias_table_from_probs(_amplitudes_to_probabilities(state, n_bits), precision_bits)

def _sparse_addr_bits(card: int) -> int:
    if card <= 1:
        return 1
    return (card - 1).bit_length()

def build_sparse_alias_table(state: State, precision_bits: int):
    items = sorted(((idx, amp) for idx, amp in state.items() if amp != 0.0))
    card = len(items)
    m_bits = _sparse_addr_bits(card)
    n_bins = 1 << m_bits
    total = sum((amp * amp for _, amp in items))
    probs = [0.0] * n_bins
    index_map = [0] * n_bins
    for slot, (idx, amp) in enumerate(items):
        probs[slot] = amp * amp / total if total > 0 else 0.0
        index_map[slot] = idx
    return (_build_alias_table_from_probs(probs, precision_bits), index_map, m_bits)

def synthesize_alias_sampling(state: State, n_bits: int, precision_bits: int=8, use_select_swap: bool=True, clifford_t: bool=False) -> QuantumCircuit:
    n = n_bits
    b = precision_bits
    table = build_alias_table(state, n, b)
    keep_res = synthesize_qrom(table.keep_values, n, b, use_select_swap=use_select_swap, clifford_t=clifford_t)
    alias_res = synthesize_qrom(table.alias_indices, n, n, use_select_swap=use_select_swap, clifford_t=clifford_t)
    nq = 0
    addr = list(range(nq, nq + n))
    nq += n
    keep_start = nq
    nq += keep_res.qc.num_qubits
    keep_out = [keep_start + keep_res.output_qubits[i] for i in range(b)]
    rand_q = list(range(nq, nq + b))
    nq += b
    flag = nq
    nq += 1
    alias_start = nq
    nq += alias_res.qc.num_qubits
    alias_out = [alias_start + alias_res.output_qubits[i] for i in range(n)]
    compare_anc = []
    cswap_anc = []
    if clifford_t:
        compare_anc = list(range(nq, nq + b))
        nq += b
        cswap_anc = list(range(nq, nq + n))
        nq += n
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
        add_comparator_ct(qc, keep_out, rand_q, flag, compare_anc)
    else:
        add_comparator(qc, keep_out, rand_q, flag)
    alias_map = list(range(alias_start, alias_start + alias_res.qc.num_qubits))
    for i in range(n):
        alias_map[alias_res.input_qubits[i]] = addr[i]
    qc.compose(alias_res.qc, qubits=alias_map, inplace=True)
    if clifford_t:
        add_controlled_swap_ct(qc, flag, addr, alias_out, cswap_anc)
    else:
        add_controlled_swap(qc, flag, addr, alias_out)
    return qc

def synthesize_sparse_alias_sampling(state: State, n_bits: int, precision_bits: int=8, use_select_swap: bool=True, clifford_t: bool=False) -> QuantumCircuit:
    b = precision_bits
    table, index_map, m = build_sparse_alias_table(state, b)
    keep_res = synthesize_qrom(table.keep_values, m, b, use_select_swap=use_select_swap, clifford_t=clifford_t)
    alias_res = synthesize_qrom(table.alias_indices, m, m, use_select_swap=use_select_swap, clifford_t=clifford_t)
    index_res = synthesize_qrom(index_map, m, n_bits, use_select_swap=use_select_swap, clifford_t=clifford_t)
    nq = 0
    addr = list(range(nq, nq + m))
    nq += m
    keep_start = nq
    nq += keep_res.qc.num_qubits
    keep_out = [keep_start + keep_res.output_qubits[i] for i in range(b)]
    rand_q = list(range(nq, nq + b))
    nq += b
    flag = nq
    nq += 1
    alias_start = nq
    nq += alias_res.qc.num_qubits
    alias_out = [alias_start + alias_res.output_qubits[i] for i in range(m)]
    index_start = nq
    nq += index_res.qc.num_qubits
    compare_anc = []
    cswap_anc = []
    if clifford_t:
        compare_anc = list(range(nq, nq + b))
        nq += b
        cswap_anc = list(range(nq, nq + m))
        nq += m
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
        add_comparator_ct(qc, keep_out, rand_q, flag, compare_anc)
    else:
        add_comparator(qc, keep_out, rand_q, flag)
    alias_map = list(range(alias_start, alias_start + alias_res.qc.num_qubits))
    for i in range(m):
        alias_map[alias_res.input_qubits[i]] = addr[i]
    qc.compose(alias_res.qc, qubits=alias_map, inplace=True)
    if clifford_t:
        add_controlled_swap_ct(qc, flag, addr, alias_out, cswap_anc)
    else:
        add_controlled_swap(qc, flag, addr, alias_out)
    index_map_qubits = list(range(index_start, index_start + index_res.qc.num_qubits))
    for i in range(m):
        index_map_qubits[index_res.input_qubits[i]] = addr[i]
    qc.compose(index_res.qc, qubits=index_map_qubits, inplace=True)
    return qc
