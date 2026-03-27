"""Sparse quantum state representation and generators.

A quantum state is represented as ``(state, n_bits)`` where *state* is a
``dict[int, float]`` mapping basis-state indices to real amplitudes and
*n_bits* is the number of qubits.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

State = Dict[int, float]

EPS = 1e-6


def normalize(state: State) -> State:
    norm_sq = sum(v * v for v in state.values())
    if norm_sq < 1e-10:
        return state
    inv = 1.0 / math.sqrt(norm_sq)
    return {k: v * inv for k, v in state.items()}


def cardinality(state: State) -> int:
    return len(state)


def get_supports(state: State, n_bits: int) -> List[int]:
    support = set()
    for idx in state:
        for q in range(n_bits):
            if (idx >> q) & 1:
                support.add(q)
    return sorted(support)


def to_statevector(state: State, n_bits: int) -> np.ndarray:
    dim = 1 << n_bits
    sv = np.zeros(dim)
    for idx, amp in state.items():
        sv[idx] = amp
    return sv


def state_hash(state: State, n_bits: int) -> int:
    h = 1469598103934665603
    mask = (1 << 64) - 1

    def mix(x: int) -> None:
        nonlocal h
        h = (h ^ (x & mask)) & mask
        h = (h * 1099511628211) & mask

    mix(n_bits)
    for idx in sorted(state.keys()):
        mix(idx)
        q = round(state[idx] / EPS)
        mix(q & mask)
    return h


# --- Generators ---

def ground_state(n: int) -> Tuple[State, int]:
    return {0: 1.0}, n


def w_state(n: int) -> Tuple[State, int]:
    return dicke_state(n, 1)


def dicke_state(n: int, k: int) -> Tuple[State, int]:
    state: State = {}
    for i in range(1 << n):
        if bin(i).count("1") == k:
            state[i] = 1.0
    return normalize(state), n


def dense_uniform(n: int) -> Tuple[State, int]:
    m = 1 << (n - 1)
    amp = 1.0 / math.sqrt(m)
    state = {i: amp for i in range(m)}
    return state, n


def sparse_uniform(n: int, seed: int = 42) -> Tuple[State, int]:
    rng = random.Random(seed)
    indices = list(range(1 << n))
    rng.shuffle(indices)
    amp = 1.0 / math.sqrt(n)
    state = {indices[i]: amp for i in range(n)}
    return state, n


def sparse_random(n: int, seed: int = 42) -> Tuple[State, int]:
    return random_sparse_state(n, seed)


def random_state(n_bits: int, card: int, seed: int = 42) -> Tuple[State, int]:
    rng = random.Random(seed)
    dim = 1 << n_bits
    chosen: set = set()
    while len(chosen) < card:
        chosen.add(rng.randrange(dim))

    state: State = {}
    norm_sq = 0.0
    for idx in chosen:
        w = rng.gauss(0, 1)
        state[idx] = w
        norm_sq += w * w

    if norm_sq > 0:
        inv = 1.0 / math.sqrt(norm_sq)
        state = {k: v * inv for k, v in state.items()}
    else:
        first = next(iter(state))
        state[first] = 1.0

    return state, n_bits


def random_sparse_state(n_bits: int, seed: int = 42) -> Tuple[State, int]:
    return random_state(n_bits, n_bits, seed)


def random_dense_state(n_bits: int, seed: int = 42) -> Tuple[State, int]:
    return random_state(n_bits, 1 << (n_bits - 1), seed)


# --- JSON serialization ---

def state_to_json(state: State, n_bits: int) -> dict:
    """Serialize a state to a JSON-compatible dict."""
    return {"n_bits": n_bits, "amplitudes": {str(k): v for k, v in state.items()}}


def state_from_json(obj: dict) -> Tuple[State, int]:
    """Deserialize a state from a JSON-compatible dict."""
    n_bits = obj["n_bits"]
    state: State = {int(k): float(v) for k, v in obj["amplitudes"].items()}
    return state, n_bits


def save_state(path: str | Path, state: State, n_bits: int) -> None:
    """Write a single state to a JSON file."""
    Path(path).write_text(json.dumps(state_to_json(state, n_bits), indent=2))


def load_state(path: str | Path) -> Tuple[State, int]:
    """Read a state from a JSON file."""
    obj = json.loads(Path(path).read_text())
    return state_from_json(obj)


# --- Benchmark suite generation ---

FAMILIES = {
    "w":              lambda n, seed: w_state(n),
    "dicke_2":        lambda n, seed: dicke_state(n, 2),
    "dicke_3":        lambda n, seed: dicke_state(n, 3),
    "sparse_uniform": lambda n, seed: sparse_uniform(n, seed),
    "sparse_random":  lambda n, seed: sparse_random(n, seed),
    "dense_random":   lambda n, seed: random_dense_state(n, seed),
}


def dump_benchmarks(
    out_dir: str | Path,
    n_min: int = 3,
    n_max: int = 8,
    seeds: int = 3,
    families: List[str] | None = None,
) -> List[Path]:
    """Generate benchmark states and save them as JSON files.

    Returns the list of written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fams = families or list(FAMILIES.keys())
    written: List[Path] = []

    for fam in fams:
        gen = FAMILIES[fam]
        is_random = "random" in fam
        max_n = min(n_max, 12) if "dense" in fam else n_max

        for n in range(n_min, max_n + 1):
            num_seeds = seeds if is_random else 1
            for s in range(1, num_seeds + 1):
                state, n_bits = gen(n, s)
                tag = f"{fam}_n{n}" + (f"_s{s}" if is_random else "")
                p = out / f"{tag}.json"
                save_state(p, state, n_bits)
                written.append(p)

    return written
