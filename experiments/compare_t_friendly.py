"""Compare T-cost of rotation vs QROM for T-friendly angles.

Enumerate exact RY angles achievable with k T gates (k=1..5),
build product states, and compare rotation T-cost (n*k, exact)
against alias sampling QROM T-cost at various precision levels.

Usage
-----
    python -m experiments.compare_t_friendly [options]

Examples
--------
    python -m experiments.compare_t_friendly --n-max 8 -v
    python -m experiments.compare_t_friendly --k-max 3 -o results.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from qsp.qrom.alias_sampling import estimate_alias_sampling_t_count
from qsp.state import State, normalize


# --- Single-qubit Clifford group (24 elements as 2x2 unitary matrices) ---

def _single_qubit_cliffords() -> List[np.ndarray]:
    """Generate the 24 single-qubit Clifford unitaries."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)

    # Generate via products of H and S (they generate single-qubit Cliffords)
    generators = [I, H, S]
    seen: Set[bytes] = set()
    group: List[np.ndarray] = []

    # BFS over products up to length 5
    current = [I]
    for _ in range(5):
        next_round = []
        for u in current:
            for g in generators:
                prod = g @ u
                # Normalize global phase: make first nonzero entry positive real
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


def _enumerate_angles_for_k(k: int) -> Set[float]:
    """Find distinct RY angles achievable with exactly k T gates.

    Canonical form: C0 · T · C1 · T · ... · T · Ck
    where Ci are single-qubit Cliffords.
    """
    cliffords = _single_qubit_cliffords()
    T = _t_gate()
    angles: Set[float] = set()

    # For k T gates, we need k+1 Cliffords
    # To keep enumeration tractable, sample intelligently
    if k <= 2:
        clifford_sets = [cliffords] * (k + 1)
    else:
        # For larger k, still enumerate but accept it takes longer
        clifford_sets = [cliffords] * (k + 1)

    for combo in itertools.product(*clifford_sets):
        # Build U = C0 · T · C1 · T · ... · T · Ck
        U = combo[0].copy()
        for i in range(k):
            U = U @ T @ combo[i + 1]

        # Extract |<0|U|0>|^2 → effective RY angle
        p0 = abs(U[0, 0]) ** 2
        p0 = max(0.0, min(1.0, p0))

        if p0 < 1e-12 or p0 > 1 - 1e-12:
            # trivial angles (0, pi) — skip
            continue

        theta = 2.0 * math.acos(math.sqrt(p0))

        # Canonicalize to (0, pi) since RY(θ) and RY(π-θ) give same |<0|U|0>|^2
        # but we keep θ in (0, π) for uniqueness
        theta = round(theta, 8)
        if theta > 0 and theta < math.pi:
            angles.add(theta)

    return angles


def enumerate_exact_angles(k_max: int = 5, verbose: bool = False) -> Dict[int, List[float]]:
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


def build_product_state(n_bits: int, theta: float) -> Tuple[State, int]:
    """Build n-qubit product state with RY(theta) on each qubit.

    |ψ⟩ = ⊗ᵢ (cos(θ/2)|0⟩ + sin(θ/2)|1⟩)

    Amplitude of |b₁b₂...bₙ⟩ = ∏ᵢ (cos(θ/2) if bᵢ=0, sin(θ/2) if bᵢ=1)
    """
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)

    state: State = {}
    for idx in range(1 << n_bits):
        amp = 1.0
        for q in range(n_bits):
            if (idx >> q) & 1:
                amp *= s
            else:
                amp *= c
        if abs(amp) > 1e-15:
            state[idx] = amp

    return normalize(state), n_bits


@dataclass
class TFriendlyRecord:
    k: int = 0
    theta: float = 0.0
    n_bits: int = 0
    precision_bits: int = 0
    rotation_t: int = 0
    qrom_t: int = 0
    overhead_ratio: float = 0.0


CSV_HEADER = [
    "k", "theta", "n_bits", "precision_bits",
    "rotation_t", "qrom_t", "overhead_ratio",
]


def run_experiment(
    k_max: int = 5,
    n_bits_list: List[int] | None = None,
    precision_list: List[int] | None = None,
    verbose: bool = False,
) -> List[TFriendlyRecord]:
    if n_bits_list is None:
        n_bits_list = [3, 4, 5, 6, 7, 8, 10, 12]
    if precision_list is None:
        precision_list = [4, 6, 8, 10, 12, 14, 16]

    exact_angles = enumerate_exact_angles(k_max, verbose=verbose)
    records: List[TFriendlyRecord] = []

    for k in range(1, k_max + 1):
        angles = exact_angles.get(k, [])
        if not angles:
            continue

        if verbose:
            print(f"\nk={k}: {len(angles)} angles", file=sys.stderr)

        for theta in angles:
            for n_bits in n_bits_list:
                rotation_t = n_bits * k

                for prec in precision_list:
                    qrom_t = estimate_alias_sampling_t_count(n_bits, prec)
                    ratio = qrom_t / rotation_t if rotation_t > 0 else float("inf")
                    rec = TFriendlyRecord(
                        k=k,
                        theta=theta,
                        n_bits=n_bits,
                        precision_bits=prec,
                        rotation_t=rotation_t,
                        qrom_t=qrom_t,
                        overhead_ratio=round(ratio, 4),
                    )
                    records.append(rec)

                    if verbose:
                        winner = "ROT" if rotation_t <= qrom_t else "QROM"
                        print(
                            f"  θ={theta/math.pi:.4f}π n={n_bits:2d} b={prec:2d}"
                            f"  rot_T={rotation_t:6d} qrom_T={qrom_t:6d}"
                            f"  ratio={ratio:7.2f}  [{winner}]",
                            file=sys.stderr,
                        )

    return records


def write_csv(records: List[TFriendlyRecord], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in records:
            w.writerow([
                r.k, f"{r.theta:.10f}", r.n_bits, r.precision_bits,
                r.rotation_t, r.qrom_t, f"{r.overhead_ratio:.4f}",
            ])


def print_comparison_table(records: List[TFriendlyRecord]) -> None:
    print("\n=== T-Friendly Angles: Rotation vs QROM ===\n")
    header = (
        f"{'k':>3s} {'theta/pi':>10s} {'n':>4s} {'prec':>5s}"
        f" {'rot_T':>8s} {'qrom_T':>8s} {'ratio':>8s} {'winner':>6s}"
    )
    print(header)
    print("-" * len(header))

    prev_k = None
    for r in records:
        if prev_k is not None and r.k != prev_k:
            print()
        prev_k = r.k
        winner = "ROT" if r.rotation_t <= r.qrom_t else "QROM"
        print(
            f"{r.k:3d} {r.theta/math.pi:10.4f} {r.n_bits:4d} {r.precision_bits:5d}"
            f" {r.rotation_t:8d} {r.qrom_t:8d} {r.overhead_ratio:8.2f} {winner:>6s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare T-cost: T-friendly rotation angles vs QROM"
    )
    parser.add_argument("--k-max", type=int, default=5,
                        help="Maximum number of T gates per rotation (default: 5)")
    parser.add_argument("--n-min", type=int, default=3,
                        help="Minimum number of qubits (default: 3)")
    parser.add_argument("--n-max", type=int, default=8,
                        help="Maximum number of qubits (default: 8)")
    parser.add_argument("-b", "--precision", default="4,6,8,10,12,14,16",
                        help="Comma-separated precision bits (default: 4,6,8,10,12,14,16)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    n_bits_list = list(range(args.n_min, args.n_max + 1))
    # Add larger sizes if max allows
    for n in [10, 12]:
        if n > args.n_max and n not in n_bits_list:
            pass  # only use requested range
    precision_list = [int(x) for x in args.precision.split(",")]

    records = run_experiment(
        k_max=args.k_max,
        n_bits_list=n_bits_list,
        precision_list=precision_list,
        verbose=args.verbose,
    )

    out_path = Path(args.output) if args.output else Path("t_friendly_results.csv")
    write_csv(records, out_path)
    print(f"Wrote {len(records)} rows to {out_path}")

    print_comparison_table(records)


if __name__ == "__main__":
    main()
