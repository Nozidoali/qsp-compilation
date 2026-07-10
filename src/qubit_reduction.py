from __future__ import annotations
import math
from typing import List, Tuple
from qiskit.circuit import QuantumCircuit
from state import EPSILON, State, get_support_qubits
from rotation_helpers import optimize_rotation_angles, select_informative_qubit
from cardinality_reduction import prepare_state_by_cardinality_reduction, emit_gate_list

def reduce_one_qubit(state: State, num_qubits: int) -> Tuple[State, List[dict]]:
    support_qubits = get_support_qubits(state, num_qubits)
    if len(support_qubits) <= 1 or len(state) < 2:
        return (state, [])
    pivot_qubit = select_informative_qubit(state, num_qubits, support_qubits)
    control_indices = [qubit for qubit in support_qubits if qubit != pivot_qubit]
    num_controls = len(control_indices)
    rotation_table = [[0.0, 0.0] for _ in range(1 << num_controls)]
    for index, amplitude in state.items():
        table_index = 0
        for control_position, control in enumerate(control_indices):
            if index >> control & 1:
                table_index |= 1 << control_position
        if index >> pivot_qubit & 1:
            rotation_table[table_index][1] += amplitude
        else:
            rotation_table[table_index][0] += amplitude
    angles: List[float] = []
    for weight_zero, weight_one in rotation_table:
        if abs(weight_zero) < EPSILON and abs(weight_one) < EPSILON:
            angles.append(0.0)
        else:
            angles.append(2.0 * math.atan2(weight_one, weight_zero))
    optimized_angles, optimized_controls = optimize_rotation_angles(angles, control_indices)
    gates: List[dict] = []
    if len(optimized_angles) == 1:
        gates.append({'type': 'ry', 'target': pivot_qubit, 'theta': optimized_angles[0]})
    else:
        gates.append({'type': 'ucry', 'controls': optimized_controls, 'angles': optimized_angles, 'target': pivot_qubit})
    reduced_state: State = {}
    for index in state:
        index_zero = index & ~(1 << pivot_qubit)
        index_one = index | 1 << pivot_qubit
        weight_zero = state.get(index_zero, 0.0)
        weight_one = state.get(index_one, 0.0)
        if index_zero not in reduced_state:
            reduced_state[index_zero] = math.sqrt(weight_zero * weight_zero + weight_one * weight_one)
    reduced_state = {index: amplitude for index, amplitude in reduced_state.items() if abs(amplitude) > EPSILON}
    return (reduced_state, gates)

def prepare_state_by_qubit_reduction(state: State, num_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    current_state = dict(state)
    gate_list: List[dict] = []
    support_qubits = get_support_qubits(current_state, num_qubits)
    for _ in range(num_qubits):
        if len(support_qubits) <= 1:
            break
        current_state, gates = reduce_one_qubit(current_state, num_qubits)
        gate_list.extend(gates)
        support_qubits = get_support_qubits(current_state, num_qubits)
    if len(current_state) == 1:
        index = next(iter(current_state.keys()))
        if index != 0 or abs(current_state[index] - 1.0) > 1e-06:
            for target in range(num_qubits):
                if index >> target & 1:
                    gate_list.append({'type': 'x', 'target': target})
    elif len(current_state) > 1:
        fallback_circuit = prepare_state_by_cardinality_reduction(current_state, num_qubits)
        gate_list.reverse()
        circuit.compose(fallback_circuit, inplace=True)
        _emit_dense_gate_list(circuit, gate_list)
        return circuit
    gate_list.reverse()
    _emit_dense_gate_list(circuit, gate_list)
    return circuit
_DEMUX_TOLERANCE = 1e-10

def _emit_uniformly_controlled_ry(circuit: QuantumCircuit, angles: List[float], target: int, controls: List[int]) -> None:
    if not controls:
        if abs(angles[0]) > _DEMUX_TOLERANCE:
            circuit.ry(angles[0], target)
        return
    num_controls = len(controls)
    num_angles = 1 << num_controls
    if num_controls == 1:
        average_angle = (angles[0] + angles[1]) / 2.0
        difference_angle = (angles[0] - angles[1]) / 2.0
        if abs(average_angle) > _DEMUX_TOLERANCE:
            circuit.ry(average_angle, target)
        circuit.cx(controls[0], target)
        if abs(difference_angle) > _DEMUX_TOLERANCE:
            circuit.ry(difference_angle, target)
        circuit.cx(controls[0], target)
        return
    half = num_angles // 2
    even_angles = [(angles[i] + angles[i + half]) / 2.0 for i in range(half)]
    odd_angles = [(angles[i] - angles[i + half]) / 2.0 for i in range(half)]
    sub_controls = controls[:-1]
    _emit_uniformly_controlled_ry(circuit, even_angles, target, sub_controls)
    circuit.cx(controls[-1], target)
    _emit_uniformly_controlled_ry(circuit, odd_angles, target, sub_controls)
    circuit.cx(controls[-1], target)

def _emit_dense_gate_list(circuit: QuantumCircuit, gates: List[dict]) -> None:
    for gate in gates:
        if gate['type'] == 'x':
            circuit.x(gate['target'])
        elif gate['type'] == 'ry':
            circuit.ry(gate['theta'], gate['target'])
        elif gate['type'] == 'ucry':
            _emit_uniformly_controlled_ry(circuit, gate['angles'], gate['target'], gate['controls'])
        elif gate['type'] == 'mcry':
            emit_gate_list(circuit, [gate])
