from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
State = Dict[int, float]
EPS = 1e-06

def cardinality(state: State) -> int:
    return len(state)

def get_supports(state: State, n_bits: int) -> List[int]:
    support = set()
    for idx in state:
        for q in range(n_bits):
            if idx >> q & 1:
                support.add(q)
    return sorted(support)

def state_hash(state: State, n_bits: int) -> int:
    h = 1469598103934665603
    mask = (1 << 64) - 1

    def mix(x: int) -> None:
        nonlocal h
        h = (h ^ x & mask) & mask
        h = h * 1099511628211 & mask
    mix(n_bits)
    for idx in sorted(state.keys()):
        mix(idx)
        q = round(state[idx] / EPS)
        mix(q & mask)
    return h

def _is_binary_format(amplitudes: dict, n_bits: int) -> bool:
    if not amplitudes:
        return False
    first = next(iter(amplitudes))
    return len(first) == n_bits and all((c in '01' for c in first))

def state_from_json(obj: dict) -> Tuple[State, int]:
    n_bits = obj['n_bits']
    amps = obj['amplitudes']
    if _is_binary_format(amps, n_bits):
        state: State = {int(k, 2): float(v) for k, v in amps.items()}
    else:
        state = {int(k): float(v) for k, v in amps.items()}
    return (state, n_bits)

def load_state(path: str | Path) -> Tuple[State, int]:
    obj = json.loads(Path(path).read_text())
    return state_from_json(obj)
