from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np
from qiskit.circuit import QuantumCircuit
from state import State
from rotation_helpers import apply_controlled_x, apply_multi_controlled_ry, split_on_most_unbalanced_qubit

def reduce_cardinality_by_one(state: State, num_qubits: int) -> Tuple[State, List[dict]]:
    index_array = np.fromiter(state.keys(), dtype=np.int64, count=len(state))
    amplitude_array = np.fromiter(state.values(), dtype=np.float64, count=len(state))
    remaining_indices = index_array.copy()
    fixed_values: Dict[int, bool] = {}
    split_qubit = 0
    while len(remaining_indices) > 1:
        split_qubit, remaining_indices = split_on_most_unbalanced_qubit(num_qubits, remaining_indices, fixed_values)
    first_index = int(remaining_indices[0])
    fixed_values.pop(split_qubit, None)
    candidate_mask = index_array != first_index
    for qubit, value in fixed_values.items():
        qubit_bit = (index_array >> qubit & 1).astype(bool)
        candidate_mask &= qubit_bit if value else ~qubit_bit
    candidate_indices = index_array[candidate_mask]
    while len(candidate_indices) > 1:
        _, candidate_indices = split_on_most_unbalanced_qubit(num_qubits, candidate_indices, fixed_values)
    second_index = int(candidate_indices[0])
    gates: List[dict] = []
    split_value = bool(first_index >> split_qubit & 1)
    for qubit in range(num_qubits):
        if first_index >> qubit & 1 == second_index >> qubit & 1:
            continue
        if qubit == split_qubit:
            continue
        gates.append({'type': 'cx', 'control': split_qubit, 'ctrl_val': split_value, 'target': qubit})
        index_array, amplitude_array = apply_controlled_x(index_array, amplitude_array, split_qubit, split_value, qubit)
    controls = list(fixed_values.keys())
    phases = [fixed_values[qubit] for qubit in controls]
    aligned_zero = second_index & ~(1 << split_qubit)
    aligned_one = second_index | 1 << split_qubit
    position_zero = np.where(index_array == aligned_zero)[0]
    position_one = np.where(index_array == aligned_one)[0]
    weight_zero = float(amplitude_array[position_zero[0]])
    weight_one = float(amplitude_array[position_one[0]])
    angle = 2.0 * math.atan2(weight_zero, weight_one)
    if second_index >> split_qubit & 1:
        angle = -math.pi + angle
    gates.append({'type': 'mcry', 'ctrls': controls, 'phases': phases, 'theta': angle, 'target': split_qubit})
    index_array, amplitude_array = apply_multi_controlled_ry(index_array, amplitude_array, controls, phases, angle, split_qubit)
    new_state = dict(zip(index_array.tolist(), amplitude_array.tolist()))
    return (new_state, gates)

def prepare_state_by_cardinality_reduction(state: State, num_qubits: int, use_gidney: bool=False) -> QuantumCircuit:
    current_state = dict(state)
    gate_list: List[dict] = []
    for _ in range(len(state)):
        if len(current_state) <= 1:
            break
        current_state, gates = reduce_cardinality_by_one(current_state, num_qubits)
        gate_list.extend(gates)
    if current_state:
        index = next(iter(current_state.keys()))
        for qubit in range(num_qubits):
            if index >> qubit & 1:
                gate_list.append({'type': 'x', 'target': qubit})
    gate_list.reverse()
    if use_gidney:
        max_controls = max((len(gate['ctrls']) for gate in gate_list if gate['type'] == 'mcry'), default=0)
        num_ancilla = max(0, max_controls - 1)
    else:
        num_ancilla = 0
    circuit = QuantumCircuit(num_qubits + num_ancilla)
    ancilla_qubits = list(range(num_qubits, num_qubits + num_ancilla))
    emit_gate_list(circuit, gate_list, ancilla_qubits=ancilla_qubits, use_gidney=use_gidney)
    return circuit

def emit_gate_list(circuit: QuantumCircuit, gates: List[dict], ancilla_qubits: List[int] | None=None, use_gidney: bool=False) -> None:
    ancilla_qubits = ancilla_qubits or []
    for gate in gates:
        if gate['type'] == 'x':
            circuit.x(gate['target'])
        elif gate['type'] == 'cx':
            control = gate['control']
            target = gate['target']
            control_value = gate['ctrl_val']
            if not control_value:
                circuit.x(control)
            circuit.cx(control, target)
            if not control_value:
                circuit.x(control)
        elif gate['type'] == 'mcry':
            angle = -gate['theta']
            controls = gate['ctrls']
            phases = gate['phases']
            target = gate['target']
            if not controls:
                circuit.ry(angle, target)
            else:
                for i, phase in enumerate(phases):
                    if not phase:
                        circuit.x(controls[i])
                if use_gidney:
                    from qrom_primitives import emit_multi_controlled_ry_gidney
                    emit_multi_controlled_ry_gidney(circuit, angle, controls, target, ancilla_qubits[:max(0, len(controls) - 1)])
                elif len(controls) == 1:
                    circuit.cry(angle, controls[0], target)
                else:
                    from qiskit.circuit.library import RYGate
                    circuit.append(RYGate(angle).control(len(controls)), controls + [target])
                for i, phase in enumerate(phases):
                    if not phase:
                        circuit.x(controls[i])
