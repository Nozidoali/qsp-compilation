from __future__ import annotations
from dataclasses import dataclass
from typing import List
from qiskit.circuit import QuantumCircuit
from qrom_primitives import and_compute, and_uncompute

def _divide_rounding_up(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator

@dataclass
class QROMResult:
    circuit: QuantumCircuit
    input_qubits: List[int]
    output_qubits: List[int]

def _build_unary_iteration(circuit: QuantumCircuit, address_qubits: List[int], unary_qubits: List[int]) -> None:
    num_address_bits = len(address_qubits)
    num_unary = len(unary_qubits)
    circuit.x(address_qubits[0])
    circuit.cx(address_qubits[0], unary_qubits[0])
    circuit.x(address_qubits[0])
    circuit.cx(address_qubits[0], unary_qubits[1])
    for level in range(1, num_address_bits):
        step = 1 << level
        for node in range(step - 1, -1, -1):
            sibling = node + step
            if sibling >= num_unary:
                continue
            circuit.ccx(unary_qubits[node], address_qubits[level], unary_qubits[sibling])
            circuit.cx(unary_qubits[sibling], unary_qubits[node])

def _uncompute_unary_iteration(circuit: QuantumCircuit, address_qubits: List[int], unary_qubits: List[int]) -> None:
    num_address_bits = len(address_qubits)
    num_unary = len(unary_qubits)
    for level in range(num_address_bits - 1, 0, -1):
        step = 1 << level
        for node in range(step):
            sibling = node + step
            if sibling >= num_unary:
                continue
            circuit.cx(unary_qubits[sibling], unary_qubits[node])
            circuit.ccx(unary_qubits[node], address_qubits[level], unary_qubits[sibling])
    circuit.cx(address_qubits[0], unary_qubits[1])
    circuit.x(address_qubits[0])
    circuit.cx(address_qubits[0], unary_qubits[0])
    circuit.x(address_qubits[0])

def _build_unary_iteration_clifford_t(circuit: QuantumCircuit, address_qubits: List[int], unary_qubits: List[int], ancilla_qubits: List[int]) -> None:
    num_address_bits = len(address_qubits)
    num_unary = len(unary_qubits)
    circuit.x(address_qubits[0])
    circuit.cx(address_qubits[0], unary_qubits[0])
    circuit.x(address_qubits[0])
    circuit.cx(address_qubits[0], unary_qubits[1])
    ancilla_index = 0
    for level in range(1, num_address_bits):
        step = 1 << level
        for node in range(step - 1, -1, -1):
            sibling = node + step
            if sibling >= num_unary:
                continue
            and_compute(circuit, unary_qubits[node], address_qubits[level], ancilla_qubits[ancilla_index])
            circuit.cx(ancilla_qubits[ancilla_index], unary_qubits[sibling])
            circuit.cx(unary_qubits[sibling], unary_qubits[node])
            ancilla_index += 1

def _uncompute_unary_iteration_clifford_t(circuit: QuantumCircuit, address_qubits: List[int], unary_qubits: List[int], ancilla_qubits: List[int]) -> None:
    num_address_bits = len(address_qubits)
    num_unary = len(unary_qubits)
    node_triples = []
    for level in range(1, num_address_bits):
        step = 1 << level
        for node in range(step - 1, -1, -1):
            sibling = node + step
            if sibling >= num_unary:
                continue
            node_triples.append((level, node, sibling))
    for position, (level, node, sibling) in enumerate(reversed(node_triples)):
        ancilla_index = len(node_triples) - 1 - position
        circuit.cx(unary_qubits[sibling], unary_qubits[node])
        circuit.cx(ancilla_qubits[ancilla_index], unary_qubits[sibling])
        and_uncompute(circuit, unary_qubits[node], address_qubits[level], ancilla_qubits[ancilla_index])
    circuit.cx(address_qubits[0], unary_qubits[1])
    circuit.x(address_qubits[0])
    circuit.cx(address_qubits[0], unary_qubits[0])
    circuit.x(address_qubits[0])

def _count_unary_iteration_toffolis(num_address_bits: int, num_unary: int) -> int:
    count = 0
    for level in range(1, num_address_bits):
        step = 1 << level
        for node in range(step - 1, -1, -1):
            if node + step < num_unary:
                count += 1
    return count

def _load_data_into_registers(circuit: QuantumCircuit, unary_qubits: List[int], target_qubits: List[int], data: List[int], num_data_bits: int) -> None:
    for entry_index, value in enumerate(data):
        for bit in range(num_data_bits):
            if value >> bit & 1:
                circuit.cx(unary_qubits[entry_index], target_qubits[bit])

def synthesize_basic_select(data: List[int], num_address_bits: int, num_data_bits: int) -> QROMResult:
    num_entries = len(data)
    circuit = QuantumCircuit(num_address_bits + num_entries + num_data_bits)
    address_qubits = list(range(num_address_bits))
    unary_qubits = list(range(num_address_bits, num_address_bits + num_entries))
    target_qubits = list(range(num_address_bits + num_entries, num_address_bits + num_entries + num_data_bits))
    _build_unary_iteration(circuit, address_qubits, unary_qubits)
    _load_data_into_registers(circuit, unary_qubits, target_qubits, data, num_data_bits)
    _uncompute_unary_iteration(circuit, address_qubits, unary_qubits)
    return QROMResult(circuit, address_qubits, target_qubits)

def synthesize_select_swap(data: List[int], num_address_bits: int, num_data_bits: int, block_size: int) -> QROMResult:
    num_entries = len(data)
    num_remainder_bits = max(1, (block_size - 1).bit_length()) if block_size > 1 else 1
    num_quotients = _divide_rounding_up(num_entries, block_size)
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
    circuit = QuantumCircuit(qubit_count)
    if num_quotients > 1:
        _build_unary_iteration(circuit, quotient_qubits, unary_qubits)
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
        _uncompute_unary_iteration(circuit, quotient_qubits, unary_qubits)
    else:
        circuit.x(unary_qubits[0])
    for bit_position in range(num_remainder_bits):
        for register_index in range(block_size):
            if not register_index & 1 << bit_position:
                partner_index = register_index | 1 << bit_position
                if partner_index >= block_size:
                    continue
                for bit in range(num_data_bits):
                    circuit.cx(register_groups[register_index][bit], register_groups[partner_index][bit])
                    circuit.ccx(remainder_qubits[bit_position], register_groups[partner_index][bit], register_groups[register_index][bit])
                    circuit.cx(register_groups[register_index][bit], register_groups[partner_index][bit])
    for bit in range(num_data_bits):
        circuit.cx(register_groups[0][bit], output_qubits[bit])
    return QROMResult(circuit, address_qubits, output_qubits)

def synthesize_basic_select_clifford_t(data: List[int], num_address_bits: int, num_data_bits: int) -> QROMResult:
    num_entries = len(data)
    num_ancilla = _count_unary_iteration_toffolis(num_address_bits, num_entries)
    qubit_count = 0
    address_qubits = list(range(qubit_count, qubit_count + num_address_bits))
    qubit_count += num_address_bits
    unary_qubits = list(range(qubit_count, qubit_count + num_entries))
    qubit_count += num_entries
    target_qubits = list(range(qubit_count, qubit_count + num_data_bits))
    qubit_count += num_data_bits
    ancilla_qubits = list(range(qubit_count, qubit_count + num_ancilla))
    qubit_count += num_ancilla
    circuit = QuantumCircuit(qubit_count)
    _build_unary_iteration_clifford_t(circuit, address_qubits, unary_qubits, ancilla_qubits)
    _load_data_into_registers(circuit, unary_qubits, target_qubits, data, num_data_bits)
    _uncompute_unary_iteration_clifford_t(circuit, address_qubits, unary_qubits, ancilla_qubits)
    return QROMResult(circuit, address_qubits, target_qubits)

def synthesize_select_swap_clifford_t(data: List[int], num_address_bits: int, num_data_bits: int, block_size: int) -> QROMResult:
    num_entries = len(data)
    num_remainder_bits = max(1, (block_size - 1).bit_length()) if block_size > 1 else 1
    num_quotients = _divide_rounding_up(num_entries, block_size)
    num_quotient_bits = num_address_bits - num_remainder_bits
    num_unary_toffolis = _count_unary_iteration_toffolis(num_quotient_bits, num_quotients) if num_quotients > 1 else 0
    num_swap_toffolis = 0
    for bit_position in range(num_remainder_bits):
        for register_index in range(block_size):
            if not register_index & 1 << bit_position:
                if register_index | 1 << bit_position < block_size:
                    num_swap_toffolis += num_data_bits
    num_ancilla = max(num_unary_toffolis, num_swap_toffolis)
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
    circuit = QuantumCircuit(qubit_count)
    if num_quotients > 1:
        _build_unary_iteration_clifford_t(circuit, quotient_qubits, unary_qubits, ancilla_qubits[:num_unary_toffolis])
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
        _uncompute_unary_iteration_clifford_t(circuit, quotient_qubits, unary_qubits, ancilla_qubits[:num_unary_toffolis])
    else:
        circuit.x(unary_qubits[0])
    ancilla_index = 0
    for bit_position in range(num_remainder_bits):
        for register_index in range(block_size):
            if not register_index & 1 << bit_position:
                partner_index = register_index | 1 << bit_position
                if partner_index >= block_size:
                    continue
                for bit in range(num_data_bits):
                    circuit.cx(register_groups[register_index][bit], register_groups[partner_index][bit])
                    and_compute(circuit, remainder_qubits[bit_position], register_groups[partner_index][bit], ancilla_qubits[ancilla_index])
                    circuit.cx(ancilla_qubits[ancilla_index], register_groups[register_index][bit])
                    and_uncompute(circuit, remainder_qubits[bit_position], register_groups[partner_index][bit], ancilla_qubits[ancilla_index])
                    circuit.cx(register_groups[register_index][bit], register_groups[partner_index][bit])
                    ancilla_index += 1
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
    use_swap = use_select_swap and block_size > 1 and num_entries > 2
    if clifford_t:
        if use_swap:
            return synthesize_select_swap_clifford_t(padded_data, num_address_bits, num_data_bits, block_size)
        return synthesize_basic_select_clifford_t(padded_data, num_address_bits, num_data_bits)
    if use_swap:
        return synthesize_select_swap(padded_data, num_address_bits, num_data_bits, block_size)
    return synthesize_basic_select(padded_data, num_address_bits, num_data_bits)
