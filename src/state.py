from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
State = Dict[int, float]
EPSILON = 1e-06

def get_support_qubits(state: State, num_qubits: int) -> List[int]:
    support = set()
    for index in state:
        for qubit in range(num_qubits):
            if index >> qubit & 1:
                support.add(qubit)
    return sorted(support)

def _keys_are_binary_strings(amplitudes: dict, num_qubits: int) -> bool:
    if not amplitudes:
        return False
    first_key = next(iter(amplitudes))
    return len(first_key) == num_qubits and all((character in '01' for character in first_key))

def load_state(path: str | Path) -> Tuple[State, int]:
    data = json.loads(Path(path).read_text())
    num_qubits = data['n_bits']
    amplitudes = data['amplitudes']
    if _keys_are_binary_strings(amplitudes, num_qubits):
        state: State = {int(key, 2): float(value) for key, value in amplitudes.items()}
    else:
        state = {int(key): float(value) for key, value in amplitudes.items()}
    return (state, num_qubits)
