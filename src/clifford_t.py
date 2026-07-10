from __future__ import annotations
import math
import warnings
from typing import Dict, Tuple
from qiskit.circuit import QuantumCircuit
_gridsynth_cache: Dict[Tuple[int, int], str] = {}

def _emit_rz_as_clifford(circuit: QuantumCircuit, angle: float, qubit: int) -> bool:
    normalized_angle = angle % (2 * math.pi)
    eighth = round(normalized_angle / (math.pi / 4))
    if abs(normalized_angle - eighth * math.pi / 4) > 1e-09:
        return False
    eighth = int(eighth) % 8
    if eighth == 0:
        pass
    elif eighth == 1:
        circuit.t(qubit)
    elif eighth == 2:
        circuit.s(qubit)
    elif eighth == 3:
        circuit.s(qubit)
        circuit.t(qubit)
    elif eighth == 4:
        circuit.z(qubit)
    elif eighth == 5:
        circuit.z(qubit)
        circuit.t(qubit)
    elif eighth == 6:
        circuit.sdg(qubit)
    elif eighth == 7:
        circuit.tdg(qubit)
    return True

def _gridsynth_key(angle: float, precision_bits: int) -> Tuple[int, int]:
    epsilon = 2.0 ** (-precision_bits)
    return (round(angle / (epsilon / 10.0)), precision_bits)

def _gridsynth_decompose(angle: float, precision_bits: int) -> str:
    key = _gridsynth_key(angle, precision_bits)
    if key not in _gridsynth_cache:
        epsilon = 2.0 ** (-precision_bits)
        warnings.filterwarnings('ignore')
        from pygridsynth import gridsynth_gates
        _gridsynth_cache[key] = gridsynth_gates(angle, epsilon)
    return _gridsynth_cache[key]

def _gridsynth_worker(args):
    angle, epsilon = args
    import warnings as local_warnings
    local_warnings.filterwarnings('ignore')
    from pygridsynth import gridsynth_gates
    return gridsynth_gates(angle, epsilon)

def prewarm_gridsynth_cache(angles, precision_bits: int, max_workers: int | None=None) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    epsilon = 2.0 ** (-precision_bits)
    missing: Dict[Tuple[int, int], float] = {}
    for angle in angles:
        key = _gridsynth_key(angle, precision_bits)
        if key in _gridsynth_cache or key in missing:
            continue
        missing[key] = angle
    if not missing:
        return
    items = list(missing.items())
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_gridsynth_worker, (angle, epsilon)): key for key, angle in items}
        for future in as_completed(futures):
            _gridsynth_cache[futures[future]] = future.result()

def _emit_gridsynth_gates(circuit: QuantumCircuit, gate_string: str, qubit: int) -> None:
    for character in gate_string:
        if character == 'H':
            circuit.h(qubit)
        elif character == 'T':
            circuit.t(qubit)
        elif character == 'S':
            circuit.s(qubit)
        elif character == 'X':
            circuit.x(qubit)

def _flatten_to_basic_gates(circuit: QuantumCircuit, max_rounds: int=10) -> QuantumCircuit:
    basic_gates = {'ry', 'cx', 'x', 'h', 's', 't', 'sdg', 'tdg', 'z', 'id', 'ccx', 'rz', 'sx', 'sxdg', 'cry', 'measure', 'barrier'}
    flattened = circuit
    for _ in range(max_rounds):
        needs_more = False
        for instruction in flattened.data:
            if instruction.operation.name not in basic_gates:
                needs_more = True
                break
        if not needs_more:
            break
        flattened = flattened.decompose()
    return flattened

def _collect_gridsynth_angles(flattened: QuantumCircuit):
    for instruction in flattened.data:
        name = instruction.operation.name
        if name == 'ry':
            angle = float(instruction.operation.params[0])
            if abs(angle) < 1e-12:
                continue
            if abs(abs(angle) - math.pi) < 1e-12:
                continue
            yield angle
        elif name == 'cry':
            angle = float(instruction.operation.params[0])
            if abs(angle) < 1e-12:
                continue
            half_angle = angle / 2.0
            yield half_angle
            yield (-half_angle)
        elif name == 'rz':
            angle = float(instruction.operation.params[0])
            if abs(angle) < 1e-12:
                continue
            normalized_angle = angle % (2 * math.pi)
            eighth = round(normalized_angle / (math.pi / 4))
            if abs(normalized_angle - eighth * math.pi / 4) <= 1e-09:
                continue
            yield angle
        elif name == 'u':
            theta = float(instruction.operation.params[0])
            phi = float(instruction.operation.params[1])
            lambda_angle = float(instruction.operation.params[2])
            for value in (lambda_angle, theta, phi):
                if abs(value) < 1e-12:
                    continue
                if value is theta and abs(abs(value) - math.pi) < 1e-12:
                    continue
                normalized_angle = value % (2 * math.pi)
                eighth = round(normalized_angle / (math.pi / 4))
                if abs(normalized_angle - eighth * math.pi / 4) <= 1e-09:
                    continue
                yield value

def _lower_ry_rotation(output: QuantumCircuit, angle: float, qubit: int, precision_bits: int) -> None:
    output.sdg(qubit)
    output.h(qubit)
    _emit_gridsynth_gates(output, _gridsynth_decompose(angle, precision_bits), qubit)
    output.h(qubit)
    output.s(qubit)

def _lower_rz_rotation(output: QuantumCircuit, angle: float, qubit: int, precision_bits: int) -> None:
    if not _emit_rz_as_clifford(output, angle, qubit):
        _emit_gridsynth_gates(output, _gridsynth_decompose(angle, precision_bits), qubit)

def decompose_to_clifford_t(circuit: QuantumCircuit, precision_bits: int, parallel: bool=True, max_workers: int | None=None) -> QuantumCircuit:
    flattened = _flatten_to_basic_gates(circuit)
    if parallel:
        prewarm_gridsynth_cache(_collect_gridsynth_angles(flattened), precision_bits, max_workers=max_workers)
    output = QuantumCircuit(flattened.num_qubits)
    for instruction in flattened.data:
        name = instruction.operation.name
        qubits = [flattened.find_bit(qubit).index for qubit in instruction.qubits]
        if name == 'ry':
            angle = float(instruction.operation.params[0])
            if abs(angle) < 1e-12:
                continue
            if abs(abs(angle) - math.pi) < 1e-12:
                output.x(qubits[0])
            else:
                _lower_ry_rotation(output, angle, qubits[0], precision_bits)
        elif name == 'cry':
            angle = float(instruction.operation.params[0])
            if abs(angle) < 1e-12:
                continue
            control, target = qubits
            half_angle = angle / 2.0
            _lower_ry_rotation(output, half_angle, target, precision_bits)
            output.cx(control, target)
            _lower_ry_rotation(output, -half_angle, target, precision_bits)
            output.cx(control, target)
        elif name == 'rz':
            angle = float(instruction.operation.params[0])
            if abs(angle) < 1e-12:
                continue
            _lower_rz_rotation(output, angle, qubits[0], precision_bits)
        elif name == 'u':
            theta = float(instruction.operation.params[0])
            phi = float(instruction.operation.params[1])
            lambda_angle = float(instruction.operation.params[2])
            qubit = qubits[0]
            if abs(lambda_angle) > 1e-12:
                _lower_rz_rotation(output, lambda_angle, qubit, precision_bits)
            if abs(theta) < 1e-12:
                pass
            elif abs(abs(theta) - math.pi) < 1e-12:
                output.x(qubit)
            else:
                _lower_ry_rotation(output, theta, qubit, precision_bits)
            if abs(phi) > 1e-12:
                _lower_rz_rotation(output, phi, qubit, precision_bits)
        else:
            output.append(instruction.operation, qubits)
    return output
