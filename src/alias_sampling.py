from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import List
from qiskit.circuit import QuantumCircuit
from qrom_primitives import add_comparator, add_comparator_clifford_t, add_controlled_swap, add_controlled_swap_clifford_t
from qrom import synthesize_qrom
from state import State

@dataclass
class AliasTable:
    keep_values: List[int] = field(default_factory=list)
    alias_indices: List[int] = field(default_factory=list)
    num_bins: int = 0
    precision_bits: int = 0

def _amplitudes_to_probabilities(state: State, num_qubits: int) -> List[float]:
    probabilities = [0.0] * (1 << num_qubits)
    for index, amplitude in state.items():
        probabilities[index] = amplitude * amplitude
    return probabilities

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
    return _build_alias_table_from_probabilities(_amplitudes_to_probabilities(state, num_qubits), precision_bits)

def _num_address_bits_for_cardinality(cardinality_count: int) -> int:
    if cardinality_count <= 1:
        return 1
    return (cardinality_count - 1).bit_length()

def build_sparse_alias_table(state: State, precision_bits: int):
    entries = sorted(((index, amplitude) for index, amplitude in state.items() if amplitude != 0.0))
    cardinality_count = len(entries)
    num_address_bits = _num_address_bits_for_cardinality(cardinality_count)
    num_bins = 1 << num_address_bits
    total = sum((amplitude * amplitude for _, amplitude in entries))
    probabilities = [0.0] * num_bins
    index_map = [0] * num_bins
    for slot, (index, amplitude) in enumerate(entries):
        probabilities[slot] = amplitude * amplitude / total if total > 0 else 0.0
        index_map[slot] = index
    return (_build_alias_table_from_probabilities(probabilities, precision_bits), index_map, num_address_bits)

def synthesize_alias_sampling(state: State, num_qubits: int, precision_bits: int=8, use_select_swap: bool=True, clifford_t: bool=False) -> QuantumCircuit:
    table = build_alias_table(state, num_qubits, precision_bits)
    keep_qrom = synthesize_qrom(table.keep_values, num_qubits, precision_bits, use_select_swap=use_select_swap, clifford_t=clifford_t)
    alias_qrom = synthesize_qrom(table.alias_indices, num_qubits, num_qubits, use_select_swap=use_select_swap, clifford_t=clifford_t)
    qubit_count = 0
    address_qubits = list(range(qubit_count, qubit_count + num_qubits))
    qubit_count += num_qubits
    keep_start = qubit_count
    qubit_count += keep_qrom.circuit.num_qubits
    keep_output_qubits = [keep_start + keep_qrom.output_qubits[i] for i in range(precision_bits)]
    random_qubits = list(range(qubit_count, qubit_count + precision_bits))
    qubit_count += precision_bits
    flag_qubit = qubit_count
    qubit_count += 1
    alias_start = qubit_count
    qubit_count += alias_qrom.circuit.num_qubits
    alias_output_qubits = [alias_start + alias_qrom.output_qubits[i] for i in range(num_qubits)]
    comparator_ancilla = []
    swap_ancilla = []
    if clifford_t:
        comparator_ancilla = list(range(qubit_count, qubit_count + precision_bits))
        qubit_count += precision_bits
        swap_ancilla = list(range(qubit_count, qubit_count + num_qubits))
        qubit_count += num_qubits
    circuit = QuantumCircuit(qubit_count)
    for i in range(num_qubits):
        circuit.h(address_qubits[i])
    for i in range(precision_bits):
        circuit.h(random_qubits[i])
    keep_qubit_map = list(range(keep_start, keep_start + keep_qrom.circuit.num_qubits))
    for i in range(num_qubits):
        keep_qubit_map[keep_qrom.input_qubits[i]] = address_qubits[i]
    circuit.compose(keep_qrom.circuit, qubits=keep_qubit_map, inplace=True)
    if clifford_t:
        add_comparator_clifford_t(circuit, keep_output_qubits, random_qubits, flag_qubit, comparator_ancilla)
    else:
        add_comparator(circuit, keep_output_qubits, random_qubits, flag_qubit)
    alias_qubit_map = list(range(alias_start, alias_start + alias_qrom.circuit.num_qubits))
    for i in range(num_qubits):
        alias_qubit_map[alias_qrom.input_qubits[i]] = address_qubits[i]
    circuit.compose(alias_qrom.circuit, qubits=alias_qubit_map, inplace=True)
    if clifford_t:
        add_controlled_swap_clifford_t(circuit, flag_qubit, address_qubits, alias_output_qubits, swap_ancilla)
    else:
        add_controlled_swap(circuit, flag_qubit, address_qubits, alias_output_qubits)
    return circuit

def synthesize_sparse_alias_sampling(state: State, num_qubits: int, precision_bits: int=8, use_select_swap: bool=True, clifford_t: bool=False) -> QuantumCircuit:
    table, index_map, num_address_bits = build_sparse_alias_table(state, precision_bits)
    keep_qrom = synthesize_qrom(table.keep_values, num_address_bits, precision_bits, use_select_swap=use_select_swap, clifford_t=clifford_t)
    alias_qrom = synthesize_qrom(table.alias_indices, num_address_bits, num_address_bits, use_select_swap=use_select_swap, clifford_t=clifford_t)
    index_qrom = synthesize_qrom(index_map, num_address_bits, num_qubits, use_select_swap=use_select_swap, clifford_t=clifford_t)
    qubit_count = 0
    address_qubits = list(range(qubit_count, qubit_count + num_address_bits))
    qubit_count += num_address_bits
    keep_start = qubit_count
    qubit_count += keep_qrom.circuit.num_qubits
    keep_output_qubits = [keep_start + keep_qrom.output_qubits[i] for i in range(precision_bits)]
    random_qubits = list(range(qubit_count, qubit_count + precision_bits))
    qubit_count += precision_bits
    flag_qubit = qubit_count
    qubit_count += 1
    alias_start = qubit_count
    qubit_count += alias_qrom.circuit.num_qubits
    alias_output_qubits = [alias_start + alias_qrom.output_qubits[i] for i in range(num_address_bits)]
    index_start = qubit_count
    qubit_count += index_qrom.circuit.num_qubits
    comparator_ancilla = []
    swap_ancilla = []
    if clifford_t:
        comparator_ancilla = list(range(qubit_count, qubit_count + precision_bits))
        qubit_count += precision_bits
        swap_ancilla = list(range(qubit_count, qubit_count + num_address_bits))
        qubit_count += num_address_bits
    circuit = QuantumCircuit(qubit_count)
    for i in range(num_address_bits):
        circuit.h(address_qubits[i])
    for i in range(precision_bits):
        circuit.h(random_qubits[i])
    keep_qubit_map = list(range(keep_start, keep_start + keep_qrom.circuit.num_qubits))
    for i in range(num_address_bits):
        keep_qubit_map[keep_qrom.input_qubits[i]] = address_qubits[i]
    circuit.compose(keep_qrom.circuit, qubits=keep_qubit_map, inplace=True)
    if clifford_t:
        add_comparator_clifford_t(circuit, keep_output_qubits, random_qubits, flag_qubit, comparator_ancilla)
    else:
        add_comparator(circuit, keep_output_qubits, random_qubits, flag_qubit)
    alias_qubit_map = list(range(alias_start, alias_start + alias_qrom.circuit.num_qubits))
    for i in range(num_address_bits):
        alias_qubit_map[alias_qrom.input_qubits[i]] = address_qubits[i]
    circuit.compose(alias_qrom.circuit, qubits=alias_qubit_map, inplace=True)
    if clifford_t:
        add_controlled_swap_clifford_t(circuit, flag_qubit, address_qubits, alias_output_qubits, swap_ancilla)
    else:
        add_controlled_swap(circuit, flag_qubit, address_qubits, alias_output_qubits)
    index_qubit_map = list(range(index_start, index_start + index_qrom.circuit.num_qubits))
    for i in range(num_address_bits):
        index_qubit_map[index_qrom.input_qubits[i]] = address_qubits[i]
    circuit.compose(index_qrom.circuit, qubits=index_qubit_map, inplace=True)
    return circuit
