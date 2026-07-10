from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
from qiskit.circuit import QuantumCircuit
from qrom_primitives import add_comparator, add_comparator_clifford_t, add_controlled_swap, add_controlled_swap_clifford_t
from qrom import synthesize_qrom, QROMResult
from state import State

@dataclass
class AliasTable:
    keep_values: List[int] = field(default_factory=list)
    alias_indices: List[int] = field(default_factory=list)
    num_bins: int = 0
    precision_bits: int = 0

def _build_alias_table_from_probabilities(probabilities: List[float], precision_bits: int) -> AliasTable:
    num_bins = len(probabilities)
    max_keep_value = (1 << precision_bits) - 1
    scaled = [probability * num_bins for probability in probabilities]
    small: deque = deque()
    large: deque = deque()
    for index in range(num_bins):
        if scaled[index] < 1.0:
            small.append(index)
        else:
            large.append(index)
    keep_probability = [1.0] * num_bins
    alias = [0] * num_bins
    while small and large:
        small_index = small.popleft()
        large_index = large.popleft()
        keep_probability[small_index] = scaled[small_index]
        alias[small_index] = large_index
        scaled[large_index] = scaled[large_index] + scaled[small_index] - 1.0
        if scaled[large_index] < 1.0:
            small.append(large_index)
        else:
            large.append(large_index)
    while large:
        keep_probability[large.popleft()] = 1.0
    while small:
        keep_probability[small.popleft()] = 1.0
    table = AliasTable(num_bins=num_bins, precision_bits=precision_bits)
    for index in range(num_bins):
        clamped = max(0.0, min(1.0, keep_probability[index]))
        table.keep_values.append(int(round(clamped * max_keep_value)))
        table.alias_indices.append(alias[index])
    return table

def build_alias_table(state: State, num_qubits: int, precision_bits: int) -> AliasTable:
    probabilities = [0.0] * (1 << num_qubits)
    for index, amplitude in state.items():
        probabilities[index] = amplitude * amplitude
    return _build_alias_table_from_probabilities(probabilities, precision_bits)

def build_sparse_alias_table(state: State, precision_bits: int):
    entries = sorted(((index, amplitude) for index, amplitude in state.items() if amplitude != 0.0))
    num_address_bits = max(1, (len(entries) - 1).bit_length())
    num_bins = 1 << num_address_bits
    total = sum((amplitude * amplitude for _, amplitude in entries))
    probabilities = [0.0] * num_bins
    index_map = [0] * num_bins
    for slot, (index, amplitude) in enumerate(entries):
        probabilities[slot] = amplitude * amplitude / total if total > 0 else 0.0
        index_map[slot] = index
    return (_build_alias_table_from_probabilities(probabilities, precision_bits), index_map, num_address_bits)

def _compose_qrom(circuit: QuantumCircuit, qrom: QROMResult, start: int, address_qubits: List[int]) -> None:
    qubit_map = list(range(start, start + qrom.circuit.num_qubits))
    for input_position, address_qubit in enumerate(address_qubits):
        qubit_map[qrom.input_qubits[input_position]] = address_qubit
    circuit.compose(qrom.circuit, qubits=qubit_map, inplace=True)

def _build_alias_sampling_circuit(num_address_bits: int, keep_values: List[int], alias_indices: List[int], index_map: Optional[List[int]], num_index_bits: int, precision_bits: int, use_select_swap: bool, clifford_t: bool) -> QuantumCircuit:
    keep_qrom = synthesize_qrom(keep_values, num_address_bits, precision_bits, use_select_swap=use_select_swap, clifford_t=clifford_t)
    alias_qrom = synthesize_qrom(alias_indices, num_address_bits, num_address_bits, use_select_swap=use_select_swap, clifford_t=clifford_t)
    index_qrom = synthesize_qrom(index_map, num_address_bits, num_index_bits, use_select_swap=use_select_swap, clifford_t=clifford_t) if index_map is not None else None

    qubit_count = 0

    def allocate(width: int) -> int:
        nonlocal qubit_count
        start = qubit_count
        qubit_count += width
        return start

    address_qubits = list(range(allocate(num_address_bits), qubit_count))
    keep_start = allocate(keep_qrom.circuit.num_qubits)
    keep_output_qubits = [keep_start + keep_qrom.output_qubits[i] for i in range(precision_bits)]
    random_qubits = list(range(allocate(precision_bits), qubit_count))
    flag_qubit = allocate(1)
    alias_start = allocate(alias_qrom.circuit.num_qubits)
    alias_output_qubits = [alias_start + alias_qrom.output_qubits[i] for i in range(num_address_bits)]
    index_start = allocate(index_qrom.circuit.num_qubits) if index_qrom is not None else 0
    comparator_ancilla = list(range(allocate(precision_bits), qubit_count)) if clifford_t else []
    swap_ancilla = list(range(allocate(num_address_bits), qubit_count)) if clifford_t else []

    circuit = QuantumCircuit(qubit_count)
    for qubit in address_qubits + random_qubits:
        circuit.h(qubit)
    _compose_qrom(circuit, keep_qrom, keep_start, address_qubits)
    if clifford_t:
        add_comparator_clifford_t(circuit, keep_output_qubits, random_qubits, flag_qubit, comparator_ancilla)
    else:
        add_comparator(circuit, keep_output_qubits, random_qubits, flag_qubit)
    _compose_qrom(circuit, alias_qrom, alias_start, address_qubits)
    if clifford_t:
        add_controlled_swap_clifford_t(circuit, flag_qubit, address_qubits, alias_output_qubits, swap_ancilla)
    else:
        add_controlled_swap(circuit, flag_qubit, address_qubits, alias_output_qubits)
    if index_qrom is not None:
        _compose_qrom(circuit, index_qrom, index_start, address_qubits)
    return circuit

def synthesize_alias_sampling(state: State, num_qubits: int, precision_bits: int=8, use_select_swap: bool=True, clifford_t: bool=False) -> QuantumCircuit:
    table = build_alias_table(state, num_qubits, precision_bits)
    return _build_alias_sampling_circuit(num_qubits, table.keep_values, table.alias_indices, None, num_qubits, precision_bits, use_select_swap, clifford_t)

def synthesize_sparse_alias_sampling(state: State, num_qubits: int, precision_bits: int=8, use_select_swap: bool=True, clifford_t: bool=False) -> QuantumCircuit:
    table, index_map, num_address_bits = build_sparse_alias_table(state, precision_bits)
    return _build_alias_sampling_circuit(num_address_bits, table.keep_values, table.alias_indices, index_map, num_qubits, precision_bits, use_select_swap, clifford_t)
