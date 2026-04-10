"""T-friendly state generation.

Generate quantum states whose amplitudes arise from circuits built
entirely from T-friendly rotation angles — angles achievable with a
known number of T gates (via Clifford+T decomposition).

The circuit template uses qubit-reduction (demux) decomposition:
  qubit 0: Ry            (1 Ry gate)
  qubit 1: CRY           (2 Ry gates + 2 CNOTs)
  qubit j: j-ctrl RY     (2^j Ry gates + 2^j CNOTs)

Each individual Ry angle is sampled from the T-friendly angle set,
so the rotation T-cost is exact (no gridsynth approximation).

Usage
-----
    python -m qsp.t_friendly --help
    python -m qsp.t_friendly -o benchmarks/t_friendly/ --n-min 3 --n-max 6
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from qsp.state import State, normalize, save_state


# ---------------------------------------------------------------------------
# Single-qubit Clifford group (24 elements)
# ---------------------------------------------------------------------------

def _single_qubit_cliffords() -> List[np.ndarray]:
    I = np.eye(2, dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)

    generators = [I, H, S]
    seen: Set[bytes] = set()
    group: List[np.ndarray] = []

    current = [I]
    for _ in range(5):
        next_round = []
        for u in current:
            for g in generators:
                prod = g @ u
                for val in prod.flat:
                    if abs(val) > 1e-10:
                        prod = prod * np.conj(val) / abs(val)
                        break
                key = np.round(prod, 8).tobytes()
                if key not in seen:
                    seen.add(key)
                    group.append(prod)
                    next_round.append(prod)
        current = next_round
        if len(group) >= 24:
            break

    return group[:24]


def _t_gate() -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


# ---------------------------------------------------------------------------
# T-friendly angle enumeration
# ---------------------------------------------------------------------------

def _enumerate_angles_for_k(k: int) -> Set[float]:
    """Find distinct RY angles achievable with exactly *k* T gates."""
    cliffords = _single_qubit_cliffords()
    T = _t_gate()
    angles: Set[float] = set()

    clifford_sets = [cliffords] * (k + 1)
    for combo in itertools.product(*clifford_sets):
        U = combo[0].copy()
        for i in range(k):
            U = U @ T @ combo[i + 1]

        p0 = abs(U[0, 0]) ** 2
        p0 = max(0.0, min(1.0, p0))

        if p0 < 1e-12 or p0 > 1 - 1e-12:
            continue

        theta = 2.0 * math.acos(math.sqrt(p0))
        theta = round(theta, 8)
        if 0 < theta < math.pi:
            angles.add(theta)

    return angles


def enumerate_exact_angles(
    k_max: int = 5, verbose: bool = False,
) -> Dict[int, List[float]]:
    """Enumerate exact RY angles for k=1..k_max T gates.

    Returns dict mapping k -> list of *new* angles at that level.
    """
    all_angles: Set[float] = set()
    result: Dict[int, List[float]] = {}

    for k in range(1, k_max + 1):
        if verbose:
            print(f"Enumerating angles for k={k} T gates...", file=sys.stderr)
        angles_k = _enumerate_angles_for_k(k)
        new = sorted(angles_k - all_angles)
        result[k] = new
        all_angles |= angles_k
        if verbose:
            print(f"  k={k}: {len(new)} new angles, {len(all_angles)} total",
                  file=sys.stderr)

    return result


def build_angle_lookup(exact_angles: Dict[int, List[float]]) -> Dict[float, int]:
    """Build lookup: rounded angle -> minimum k T-gates needed."""
    lookup: Dict[float, int] = {0.0: 0, round(math.pi, 6): 0}
    for k, angles in exact_angles.items():
        for theta in angles:
            key = round(theta, 6)
            if key not in lookup or k < lookup[key]:
                lookup[key] = k
    return lookup


def collect_all_angles(exact_angles: Dict[int, List[float]]) -> List[Tuple[float, int]]:
    """Collect all T-friendly angles with their T-cost as (angle, k) pairs."""
    lookup = build_angle_lookup(exact_angles)
    return [(theta, k) for theta, k in sorted(lookup.items()) if k > 0]


# ---------------------------------------------------------------------------
# Statevector simulation helpers
# ---------------------------------------------------------------------------

def _apply_ry_sv(sv: np.ndarray, target: int, theta: float) -> None:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    N = len(sv)
    all_idx = np.arange(N)
    mask = 1 << target
    idx0 = all_idx[all_idx & mask == 0]
    idx1 = idx0 | mask
    a0 = sv[idx0].copy()
    a1 = sv[idx1].copy()
    sv[idx0] = c * a0 - s * a1
    sv[idx1] = s * a0 + c * a1


def _apply_cx_sv(sv: np.ndarray, control: int, target: int) -> None:
    N = len(sv)
    all_idx = np.arange(N)
    ctrl_mask = 1 << control
    tgt_mask = 1 << target
    sel = all_idx[(all_idx & ctrl_mask != 0) & (all_idx & tgt_mask == 0)]
    partner = sel | tgt_mask
    sv[sel], sv[partner] = sv[partner].copy(), sv[sel].copy()


def _simulate_demux_ry(
    sv: np.ndarray,
    target: int,
    controls: List[int],
    angle_sampler,
) -> int:
    """Simulate demux Ry decomposition on statevector.

    Returns total T-cost for this UCRY decomposition.
    """
    if not controls:
        theta, t_cost = angle_sampler()
        if abs(theta) > 1e-10:
            _apply_ry_sv(sv, target, theta)
        return t_cost

    if len(controls) == 1:
        theta_a, t_a = angle_sampler()
        if abs(theta_a) > 1e-10:
            _apply_ry_sv(sv, target, theta_a)
        _apply_cx_sv(sv, controls[0], target)
        theta_b, t_b = angle_sampler()
        if abs(theta_b) > 1e-10:
            _apply_ry_sv(sv, target, theta_b)
        _apply_cx_sv(sv, controls[0], target)
        return t_a + t_b

    sub_controls = controls[:-1]
    t1 = _simulate_demux_ry(sv, target, sub_controls, angle_sampler)
    _apply_cx_sv(sv, controls[-1], target)
    t2 = _simulate_demux_ry(sv, target, sub_controls, angle_sampler)
    _apply_cx_sv(sv, controls[-1], target)
    return t1 + t2


# ---------------------------------------------------------------------------
# State generation
# ---------------------------------------------------------------------------

def generate_t_friendly_state(
    n_bits: int,
    available_angles: List[Tuple[float, int]],
    rng: np.random.Generator,
) -> Tuple[State, int, int, int]:
    """Build a random T-friendly state using the qubit-reduction template.

    Returns (state, total_t_cost, total_ry_count, total_cnot_count).
    """
    N = 1 << n_bits
    sv = np.zeros(N)
    sv[0] = 1.0

    total_t = 0
    total_ry = 0
    total_cnot = 0

    for target in range(n_bits):
        controls = list(range(target))
        n_demux_ry = 1 << len(controls)

        def angle_sampler():
            nonlocal total_ry
            theta, k = available_angles[rng.integers(len(available_angles))]
            total_ry += 1
            return theta, k

        t_cost = _simulate_demux_ry(sv, target, controls, angle_sampler)
        total_t += t_cost
        total_cnot += n_demux_ry

    state: State = {}
    for idx, amp in enumerate(sv):
        if abs(amp) > 1e-15:
            state[idx] = float(amp)

    state = normalize(state)
    return state, total_t, total_ry, total_cnot


# ---------------------------------------------------------------------------
# Benchmark generation
# ---------------------------------------------------------------------------

def dump_benchmarks(
    out_dir: str | Path,
    n_min: int = 3,
    n_max: int = 8,
    k_max: int = 2,
    samples: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> List[Path]:
    """Generate T-friendly states and save as JSON files.

    Returns the list of written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    exact_angles = enumerate_exact_angles(k_max, verbose=verbose)
    available = collect_all_angles(exact_angles)
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"{len(available)} non-trivial T-friendly angles available",
              file=sys.stderr)

    written: List[Path] = []
    for n in range(n_min, n_max + 1):
        for s in range(1, samples + 1):
            state, t_cost, ry_count, cnot_count = generate_t_friendly_state(
                n, available, rng,
            )
            tag = f"t_friendly_n{n}_s{s}"
            p = out / f"{tag}.json"
            save_state(p, state, n)
            written.append(p)

            if verbose:
                print(f"  {tag}: m={len(state)}, t_cost={t_cost}, "
                      f"ry={ry_count}, cnot={cnot_count}", file=sys.stderr)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate T-friendly benchmark states as JSON"
    )
    parser.add_argument("-o", "--out-dir", type=Path,
                        default=Path("benchmarks/t_friendly"),
                        help="Output directory (default: benchmarks/t_friendly/)")
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--k-max", type=int, default=2,
                        help="Max T gates per angle (default: 2)")
    parser.add_argument("-s", "--samples", type=int, default=5,
                        help="Random states per n (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    written = dump_benchmarks(
        args.out_dir, args.n_min, args.n_max,
        args.k_max, args.samples, args.seed, args.verbose,
    )
    print(f"Wrote {len(written)} benchmark files to {args.out_dir}")


if __name__ == "__main__":
    main()
