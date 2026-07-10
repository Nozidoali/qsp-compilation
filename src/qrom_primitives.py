from __future__ import annotations
from typing import List
from qiskit.circuit import QuantumCircuit

def and_compute(circuit: QuantumCircuit, control_a: int, control_b: int, ancilla: int) -> None:
    circuit.h(ancilla)
    circuit.cx(control_a, ancilla)
    circuit.tdg(ancilla)
    circuit.cx(control_b, ancilla)
    circuit.t(ancilla)
    circuit.cx(control_a, ancilla)
    circuit.tdg(ancilla)
    circuit.cx(control_b, ancilla)
    circuit.t(ancilla)
    circuit.h(ancilla)

def and_uncompute(circuit: QuantumCircuit, control_a: int, control_b: int, ancilla: int) -> None:
    circuit.h(ancilla)
    circuit.tdg(ancilla)
    circuit.cx(control_b, ancilla)
    circuit.t(ancilla)
    circuit.cx(control_a, ancilla)
    circuit.tdg(ancilla)
    circuit.cx(control_b, ancilla)
    circuit.t(ancilla)
    circuit.cx(control_a, ancilla)
    circuit.h(ancilla)

def emit_multi_controlled_ry_gidney(circuit: QuantumCircuit, angle: float, controls: List[int], target: int, ancilla_qubits: List[int]) -> None:
    num_controls = len(controls)
    if num_controls == 0:
        circuit.ry(angle, target)
        return
    if num_controls == 1:
        circuit.cry(angle, controls[0], target)
        return
    and_compute(circuit, controls[0], controls[1], ancilla_qubits[0])
    for i in range(2, num_controls):
        and_compute(circuit, ancilla_qubits[i - 2], controls[i], ancilla_qubits[i - 1])
    circuit.cry(angle, ancilla_qubits[num_controls - 2], target)
    for i in range(num_controls - 1, 1, -1):
        and_uncompute(circuit, ancilla_qubits[i - 2], controls[i], ancilla_qubits[i - 1])
    and_uncompute(circuit, controls[0], controls[1], ancilla_qubits[0])

def add_comparator_clifford_t(circuit: QuantumCircuit, left_qubits: List[int], right_qubits: List[int], flag_qubit: int, ancilla_qubits: List[int]) -> None:
    num_bits = len(left_qubits)
    for i in range(num_bits):
        circuit.x(left_qubits[i])
    for i in range(num_bits):
        circuit.cx(flag_qubit, right_qubits[i])
        circuit.cx(flag_qubit, left_qubits[i])
        and_compute(circuit, left_qubits[i], right_qubits[i], ancilla_qubits[i])
        circuit.cx(ancilla_qubits[i], flag_qubit)
        and_uncompute(circuit, left_qubits[i], right_qubits[i], ancilla_qubits[i])

def add_controlled_swap_clifford_t(circuit: QuantumCircuit, control: int, register_a: List[int], register_b: List[int], ancilla_qubits: List[int]) -> None:
    for i in range(len(register_a)):
        circuit.cx(register_a[i], register_b[i])
        and_compute(circuit, control, register_b[i], ancilla_qubits[i])
        circuit.cx(ancilla_qubits[i], register_a[i])
        and_uncompute(circuit, control, register_b[i], ancilla_qubits[i])
        circuit.cx(register_a[i], register_b[i])

def add_comparator(circuit: QuantumCircuit, left_qubits: List[int], right_qubits: List[int], flag_qubit: int) -> None:
    num_bits = len(left_qubits)
    for i in range(num_bits):
        circuit.x(left_qubits[i])
    for i in range(num_bits):
        circuit.cx(flag_qubit, right_qubits[i])
        circuit.cx(flag_qubit, left_qubits[i])
        circuit.ccx(left_qubits[i], right_qubits[i], flag_qubit)

def add_controlled_swap(circuit: QuantumCircuit, control: int, register_a: List[int], register_b: List[int]) -> None:
    for i in range(len(register_a)):
        circuit.cx(register_a[i], register_b[i])
        circuit.ccx(control, register_b[i], register_a[i])
        circuit.cx(register_a[i], register_b[i])
