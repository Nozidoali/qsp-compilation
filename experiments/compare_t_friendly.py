"""Compare T-cost of T-friendly rotation circuits vs QROM-based methods.

Build circuits using the qubit-reduction template (Ry, CRY, CCRY, ...),
where each multi-controlled RY is decomposed via demux (Gray-code
splitting) into CNOT + Ry gates. Each individual Ry gate in the demux
uses an angle randomly sampled from T-friendly angles (exact angles
achievable with k T gates). Since every rotation is T-friendly, the
rotation T-cost is exact (no gridsynth approximation needed).

Compare against sparse rotation, QROM (basic select), and select-swap.

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
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from qsp.circuit_utils import decompose_to_clifford_t
from qsp.metrics import count_ry, estimate_t_count, rotation_fidelity
from qsp.qrom.alias_sampling import (
    build_alias_table, estimate_alias_sampling_t_count,
)
from qsp.rotation.sparse import prepare_state_sparse
from qsp.state import State
from qsp.t_friendly import (
    collect_all_angles,
    enumerate_exact_angles,
    generate_t_friendly_state as build_t_friendly_circuit,
)


def _qrom_analytical_fidelity(state: State, n_bits: int, precision_bits: int) -> float:
    """Compute analytical fidelity of alias-sampling QROM."""
    table = build_alias_table(state, n_bits, precision_bits)
    N = 1 << n_bits
    mx = (1 << precision_bits) - 1
    p_approx = np.zeros(N)
    for i in range(N):
        kf = table.keep_values[i] / mx if mx > 0 else 1.0
        p_approx[i] += kf / N
        p_approx[table.alias_indices[i]] += (1.0 - kf) / N
    p_target = np.zeros(N)
    for idx, amp in state.items():
        p_target[idx] = amp * amp
    return float(np.sum(np.sqrt(p_target * p_approx)) ** 2)


def synthesize_sparse_clifford_t(
    state: State, n_bits: int, precision_bits: int,
) -> Tuple[int, int, float]:
    """Build sparse circuit, decompose to Clifford+T, return (t_count, ry_count, fidelity).

    Uses actual gridsynth decomposition for accurate T-count.
    """
    try:
        qc = prepare_state_sparse(state, n_bits, timeout=5.0)
        n_ry = count_ry(qc)
        qc_ct = decompose_to_clifford_t(qc, precision_bits)
        t = estimate_t_count(qc_ct)
        fid = rotation_fidelity(qc_ct, state, n_bits) if n_bits <= 10 else -1.0
        return t, n_ry, fid
    except Exception:
        return -1, 0, 0.0


@dataclass
class TFriendlyRecord:
    n_bits: int = 0
    sample_id: int = 0
    precision_bits: int = 0
    cardinality: int = 0
    # T-friendly rotation via qubit reduction template (exact, no precision needed)
    dense_t: int = 0
    dense_ry_count: int = 0
    dense_cnot_count: int = 0
    dense_infidelity: float = 0.0  # always 0 (exact)
    # Sparse rotation (Clifford+T via gridsynth)
    sparse_t: int = 0
    sparse_ry_count: int = 0
    sparse_infidelity: float = 0.0
    # QROM methods
    qrom_t: int = 0
    qrom_infidelity: float = 0.0
    selectswap_t: int = 0
    selectswap_infidelity: float = 0.0
    # Best method
    best_method: str = ""
    best_t: int = 0


CSV_HEADER = [
    "n_bits", "sample_id", "precision_bits", "cardinality",
    "dense_t", "dense_ry", "dense_cnot", "dense_infidelity",
    "sparse_t", "sparse_ry", "sparse_infidelity",
    "qrom_t", "qrom_infidelity",
    "selectswap_t", "selectswap_infidelity",
    "best_method", "best_t",
]


def run_experiment(
    k_max: int = 2,
    n_bits_list: List[int] | None = None,
    n_samples: int = 5,
    precision_list: List[int] | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> List[TFriendlyRecord]:
    if n_bits_list is None:
        n_bits_list = [3, 4, 5, 6, 7, 8]
    if precision_list is None:
        precision_list = [1, 2, 4, 6, 8, 10]

    exact_angles = enumerate_exact_angles(k_max, verbose=verbose)
    available = collect_all_angles(exact_angles)
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\n{len(available)} non-trivial T-friendly angles available",
              file=sys.stderr)

    records: List[TFriendlyRecord] = []

    for n_bits in n_bits_list:
        if verbose:
            print(f"\nn={n_bits} (2^n-1 = {(1 << n_bits) - 1} demux Ry gates):",
                  file=sys.stderr)

        for sid in range(n_samples):
            state, dense_t, dense_ry, dense_cnot = build_t_friendly_circuit(
                n_bits, available, rng)

            # Build sparse circuit once (reused across precisions)
            sparse_qc = None
            try:
                sparse_qc = prepare_state_sparse(state, n_bits, timeout=5.0)
            except Exception as e:
                if verbose:
                    print(f"    sparse synthesis failed: {e}", file=sys.stderr)

            for prec in precision_list:
                rec = TFriendlyRecord(
                    n_bits=n_bits,
                    sample_id=sid,
                    precision_bits=prec,
                    cardinality=len(state),
                    dense_t=dense_t,
                    dense_ry_count=dense_ry,
                    dense_cnot_count=dense_cnot,
                    dense_infidelity=0.0,  # exact
                )

                # Sparse rotation: Clifford+T via gridsynth
                if sparse_qc is not None:
                    try:
                        qc_ct = decompose_to_clifford_t(sparse_qc, prec)
                        rec.sparse_t = estimate_t_count(qc_ct)
                        rec.sparse_ry_count = count_ry(sparse_qc)
                        if n_bits <= 10:
                            fid = rotation_fidelity(qc_ct, state, n_bits)
                            rec.sparse_infidelity = max(1 - fid, 0.0)
                    except Exception as e:
                        if verbose:
                            print(f"    sparse decompose failed: {e}",
                                  file=sys.stderr)
                        rec.sparse_t = -1
                else:
                    rec.sparse_t = -1

                # QROM (basic select)
                rec.qrom_t = estimate_alias_sampling_t_count(
                    n_bits, prec, use_select_swap=False)
                qrom_fid = _qrom_analytical_fidelity(state, n_bits, prec)
                rec.qrom_infidelity = max(1 - qrom_fid, 0.0)

                # Select-swap
                rec.selectswap_t = estimate_alias_sampling_t_count(
                    n_bits, prec, use_select_swap=True)
                rec.selectswap_infidelity = rec.qrom_infidelity  # same alias table

                # Determine best method
                candidates = {
                    "dense_tfriendly": rec.dense_t,
                    "sparse": rec.sparse_t if rec.sparse_t >= 0 else 10**9,
                    "qrom": rec.qrom_t,
                    "selectswap": rec.selectswap_t,
                }
                rec.best_method = min(candidates, key=candidates.get)
                rec.best_t = candidates[rec.best_method]

                records.append(rec)

            if verbose:
                # Print summary for last precision
                r = records[-1]
                sp_str = f"{r.sparse_t:6d}" if r.sparse_t >= 0 else "  FAIL"
                print(
                    f"  [{sid}] m={len(state):4d}"
                    f"  dense_T={r.dense_t:4d}(ry={dense_ry},cx={dense_cnot})"
                    f"  sparse_T={sp_str}(ry={r.sparse_ry_count})"
                    f"  qrom_T={r.qrom_t:6d}"
                    f"  sswap_T={r.selectswap_t:6d}"
                    f"  [{r.best_method}]",
                    file=sys.stderr,
                )

    return records


def write_csv(records: List[TFriendlyRecord], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in records:
            w.writerow([
                r.n_bits, r.sample_id, r.precision_bits, r.cardinality,
                r.dense_t, r.dense_ry_count, r.dense_cnot_count,
                f"{r.dense_infidelity:.6e}",
                r.sparse_t, r.sparse_ry_count, f"{r.sparse_infidelity:.6e}",
                r.qrom_t, f"{r.qrom_infidelity:.6e}",
                r.selectswap_t, f"{r.selectswap_infidelity:.6e}",
                r.best_method, r.best_t,
            ])


def print_comparison_table(records: List[TFriendlyRecord]) -> None:
    print("\n=== T-Friendly Circuits: Dense vs Sparse vs QROM vs Select-Swap ===\n")
    header = (
        f"{'n':>4s} {'sid':>4s} {'m':>5s}"
        f" {'dense_T':>8s} {'sparse_T':>8s}"
        f" {'qrom_T':>8s} {'sswap_T':>8s} {'best':>16s}"
    )
    print(header)
    print("-" * len(header))

    prev_n = None
    for r in records:
        if prev_n is not None and r.n_bits != prev_n:
            print()
        prev_n = r.n_bits
        sparse_str = f"{r.sparse_t:8d}" if r.sparse_t >= 0 else "    FAIL"
        print(
            f"{r.n_bits:4d} {r.sample_id:4d} {r.cardinality:5d}"
            f" {r.dense_t:8d} {sparse_str}"
            f" {r.qrom_t:8d} {r.selectswap_t:8d} {r.best_method:>16s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare T-cost: T-friendly rotation vs sparse vs QROM vs select-swap"
    )
    parser.add_argument("--k-max", type=int, default=2,
                        help="Maximum T gates per angle (default: 2)")
    parser.add_argument("--n-min", type=int, default=3,
                        help="Minimum number of qubits (default: 3)")
    parser.add_argument("--n-max", type=int, default=8,
                        help="Maximum number of qubits (default: 8)")
    parser.add_argument("-s", "--samples", type=int, default=5,
                        help="Random circuits per n (default: 5)")
    parser.add_argument("-b", "--precision", default="1,2,4,6,8,10",
                        help="Comma-separated precision bits (default: 1,2,4,6,8,10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    n_bits_list = list(range(args.n_min, args.n_max + 1))

    precision_list = [int(x) for x in args.precision.split(",")]

    records = run_experiment(
        k_max=args.k_max,
        n_bits_list=n_bits_list,
        n_samples=args.samples,
        precision_list=precision_list,
        seed=args.seed,
        verbose=args.verbose,
    )

    out_path = Path(args.output) if args.output else Path("t_friendly_results.csv")
    write_csv(records, out_path)
    print(f"Wrote {len(records)} rows to {out_path}")

    print_comparison_table(records)


if __name__ == "__main__":
    main()
