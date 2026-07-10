from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
State = Dict[int, float]
EPSILON = 1e-06

def cardinality(state: State) -> int:
    return len(state)

def get_support_qubits(state: State, num_qubits: int) -> List[int]:
    support = set()
    for index in state:
        for qubit in range(num_qubits):
            if index >> qubit & 1:
                support.add(qubit)
    return sorted(support)

def state_hash(state: State, num_qubits: int) -> int:
    hash_value = 1469598103934665603
    bit_mask = (1 << 64) - 1

    def mix(value: int) -> None:
        nonlocal hash_value
        hash_value = (hash_value ^ value & bit_mask) & bit_mask
        hash_value = hash_value * 1099511628211 & bit_mask
    mix(num_qubits)
    for index in sorted(state.keys()):
        mix(index)
        mix(round(state[index] / EPSILON) & bit_mask)
    return hash_value

def _keys_are_binary_strings(amplitudes: dict, num_qubits: int) -> bool:
    if not amplitudes:
        return False
    first_key = next(iter(amplitudes))
    return len(first_key) == num_qubits and all((character in '01' for character in first_key))

def state_from_json(json_object: dict) -> Tuple[State, int]:
    num_qubits = json_object['n_bits']
    amplitudes = json_object['amplitudes']
    if _keys_are_binary_strings(amplitudes, num_qubits):
        state: State = {int(key, 2): float(value) for key, value in amplitudes.items()}
    else:
        state = {int(key): float(value) for key, value in amplitudes.items()}
    return (state, num_qubits)

def load_state(path: str | Path) -> Tuple[State, int]:
    return state_from_json(json.loads(Path(path).read_text()))
