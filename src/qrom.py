from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from qiskit.circuit import QuantumCircuit
from qrom_primitives import and_compute, and_uncompute

def _divide_rounding_up(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator

@dataclass
class QROMResult:
    circuit: QuantumCircuit
    input_qubits: List[int]
    output_qubits: List[int]

def _emit_toffoli(circuit: QuantumCircuit, control_a: int, control_b: int, target: int, ancilla: Optional[int]=None) -> None:
    if ancilla is None:
        circuit.ccx(control_a, control_b, target)
    else:
        and_compute(circuit, control_a, control_b, ancilla)
        circuit.cx(ancilla, target)
        and_uncompute(circuit, control_a, control_b, ancilla)

def _unary_iteration_triples(num_address_bits: int, num_unary: int) -> List[tuple]:
    triples = []
    for level in range(1, num_address_bits):
        step = 1 << level
        for node in range(step - 1, -1, -1):
            sibling = node + step
            if sibling < num_unary:
                triples.append((level, node, sibling))
    return triples

def _build_unary_iteration(circuit: QuantumCircuit, address_qubits: List[int], unary_qubits: List[int], ancilla_qubits: Optional[List[int]]=None) -> None:
    circuit.x(address_qubits[0])
    circuit.cx(address_qubits[0], unary_qubits[0])
    circuit.x(address_qubits[0])
    circuit.cx(address_qubits[0], unary_qubits[1])
    for ancilla_index, (level, node, sibling) in enumerate(_unary_iteration_triples(len(address_qubits), len(unary_qubits))):
        if ancilla_qubits is None:
            circuit.ccx(unary_qubits[node], address_qubits[level], unary_qubits[sibling])
        else:
            and_compute(circuit, unary_qubits[node], address_qubits[level], ancilla_qubits[ancilla_index])
            circuit.cx(ancilla_qubits[ancilla_index], unary_qubits[sibling])
        circuit.cx(unary_qubits[sibling], unary_qubits[node])

def _uncompute_unary_iteration(circuit: QuantumCircuit, address_qubits: List[int], unary_qubits: List[int], ancilla_qubits: Optional[List[int]]=None) -> None:
    triples = _unary_iteration_triples(len(address_qubits), len(unary_qubits))
    for ancilla_index in range(len(triples) - 1, -1, -1):
        level, node, sibling = triples[ancilla_index]
        circuit.cx(unary_qubits[sibling], unary_qubits[node])
        if ancilla_qubits is None:
            circuit.ccx(unary_qubits[node], address_qubits[level], unary_qubits[sibling])
        else:
            circuit.cx(ancilla_qubits[ancilla_index], unary_qubits[sibling])
            and_uncompute(circuit, unary_qubits[node], address_qubits[level], ancilla_qubits[ancilla_index])
    circuit.cx(address_qubits[0], unary_qubits[1])
    circuit.x(address_qubits[0])
    circuit.cx(address_qubits[0], unary_qubits[0])
    circuit.x(address_qubits[0])

def _load_data_into_registers(circuit: QuantumCircuit, unary_qubits: List[int], target_qubits: List[int], data: List[int], num_data_bits: int) -> None:
    for entry_index, value in enumerate(data):
        for bit in range(num_data_bits):
            if value >> bit & 1:
                circuit.cx(unary_qubits[entry_index], target_qubits[bit])

def synthesize_basic_select(data: List[int], num_address_bits: int, num_data_bits: int, clifford_t: bool=False) -> QROMResult:
    num_entries = len(data)
    num_ancilla = len(_unary_iteration_triples(num_address_bits, num_entries)) if clifford_t else 0
    qubit_count = 0
    address_qubits = list(range(qubit_count, qubit_count + num_address_bits))
    qubit_count += num_address_bits
    unary_qubits = list(range(qubit_count, qubit_count + num_entries))
    qubit_count += num_entries
    target_qubits = list(range(qubit_count, qubit_count + num_data_bits))
    qubit_count += num_data_bits
    ancilla_qubits = list(range(qubit_count, qubit_count + num_ancilla))
    qubit_count += num_ancilla
    unary_ancilla = ancilla_qubits if clifford_t else None
    circuit = QuantumCircuit(qubit_count)
    _build_unary_iteration(circuit, address_qubits, unary_qubits, unary_ancilla)
    _load_data_into_registers(circuit, unary_qubits, target_qubits, data, num_data_bits)
    _uncompute_unary_iteration(circuit, address_qubits, unary_qubits, unary_ancilla)
    return QROMResult(circuit, address_qubits, target_qubits)

def synthesize_select_swap(data: List[int], num_address_bits: int, num_data_bits: int, block_size: int, clifford_t: bool=False) -> QROMResult:
    num_entries = len(data)
    num_remainder_bits = max(1, (block_size - 1).bit_length()) if block_size > 1 else 1
    num_quotients = _divide_rounding_up(num_entries, block_size)
    num_unary_toffolis = len(_unary_iteration_triples(num_address_bits - num_remainder_bits, num_quotients)) if num_quotients > 1 else 0
    num_swap_toffolis = num_remainder_bits * num_data_bits * (block_size // 2)
    num_ancilla = max(num_unary_toffolis, num_swap_toffolis) if clifford_t else 0
    qubit_count = 0
    address_qubits = list(range(num_address_bits))
    qubit_count += num_address_bits
    remainder_qubits = address_qubits[:num_remainder_bits]
    quotient_qubits = address_qubits[num_remainder_bits:]
    unary_qubits = list(range(qubit_count, qubit_count + num_quotients))
    qubit_count += num_quotients
    register_groups: List[List[int]] = []
    for _ in range(block_size):
        register = list(range(qubit_count, qubit_count + num_data_bits))
        qubit_count += num_data_bits
        register_groups.append(register)
    output_qubits = list(range(qubit_count, qubit_count + num_data_bits))
    qubit_count += num_data_bits
    ancilla_qubits = list(range(qubit_count, qubit_count + num_ancilla))
    qubit_count += num_ancilla
    unary_ancilla = ancilla_qubits[:num_unary_toffolis] if clifford_t else None
    circuit = QuantumCircuit(qubit_count)
    if num_quotients > 1:
        _build_unary_iteration(circuit, quotient_qubits, unary_qubits, unary_ancilla)
    else:
        circuit.x(unary_qubits[0])
    for quotient_index in range(num_quotients):
        for remainder_index in range(block_size):
            entry_index = quotient_index * block_size + remainder_index
            if entry_index >= num_entries:
                break
            value = data[entry_index]
            for bit in range(num_data_bits):
                if value >> bit & 1:
                    circuit.cx(unary_qubits[quotient_index], register_groups[remainder_index][bit])
    if num_quotients > 1:
        _uncompute_unary_iteration(circuit, quotient_qubits, unary_qubits, unary_ancilla)
    else:
        circuit.x(unary_qubits[0])
    swap_ancilla_index = 0
    for bit_position in range(num_remainder_bits):
        for register_index in range(block_size):
            if register_index & 1 << bit_position:
                continue
            partner_index = register_index | 1 << bit_position
            if partner_index >= block_size:
                continue
            for bit in range(num_data_bits):
                register = register_groups[register_index][bit]
                partner = register_groups[partner_index][bit]
                circuit.cx(register, partner)
                if clifford_t:
                    _emit_toffoli(circuit, remainder_qubits[bit_position], partner, register, ancilla_qubits[swap_ancilla_index])
                    swap_ancilla_index += 1
                else:
                    _emit_toffoli(circuit, remainder_qubits[bit_position], partner, register)
                circuit.cx(register, partner)
    for bit in range(num_data_bits):
        circuit.cx(register_groups[0][bit], output_qubits[bit])
    return QROMResult(circuit, address_qubits, output_qubits)

def compute_optimal_block_size(num_entries: int, num_data_bits: int) -> int:
    if num_entries <= 2 or num_data_bits == 0:
        return 1
    best_cost = float('inf')
    best_block_size = 1
    block_size = 1
    while block_size <= num_entries:
        cost = 4.0 * _divide_rounding_up(num_entries, block_size) + 4.0 * num_data_bits * block_size
        if cost < best_cost:
            best_cost = cost
            best_block_size = block_size
        block_size *= 2
    return best_block_size

def synthesize_qrom(data: List[int], num_address_bits: int, num_data_bits: int, use_select_swap: bool=True, block_size: int=0, clifford_t: bool=False) -> QROMResult:
    num_entries = 1 << num_address_bits
    padded_data = list(data) + [0] * (num_entries - len(data))
    if block_size == 0:
        block_size = compute_optimal_block_size(num_entries, num_data_bits)
    if use_select_swap and block_size > 1 and num_entries > 2:
        return synthesize_select_swap(padded_data, num_address_bits, num_data_bits, block_size, clifford_t)
    return synthesize_basic_select(padded_data, num_address_bits, num_data_bits, clifford_t)
