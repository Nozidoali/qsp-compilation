"""Compare dense vs sparse alias-sampling state preparation.

Dense alias sampling uses one bin per basis state (``2^n`` bins). For a state
with only ``card`` nonzero amplitudes that is wasteful: the sparse variant
compresses the support to ``2^m`` bins (``m = ceil(log2(card))``) and recovers
the real basis index with an extra index-mapping QROM. Every QROM then
addresses ``2^m`` entries instead of ``2^n``.

This experiment sweeps ``n`` (qubit count) and ``card`` (sparsity) over random
sparse states and reports the T-count and qubit savings at matched fidelity.

Usage
-----
    python -m experiments.compare_sparse_alias
    python -m experiments.compare_sparse_alias --n 8,12,16 --card 4,8,16 -b 6
    python -m experiments.compare_sparse_alias --csv out.csv --no-swap
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

from qsp.metrics import (
    qrom_alias_analytical_fidelity,
    sparse_qrom_alias_analytical_fidelity,
)
from qsp.qrom.alias_sampling import (
    estimate_alias_sampling_t_count,
    estimate_sparse_alias_sampling_t_count,
    synthesize_alias_sampling,
    synthesize_sparse_alias_sampling,
)
from qsp.state import random_state


@dataclass
class Row:
    n_bits: int
    card: int
    precision_bits: int
    dense_bins: int
    sparse_bins: int
    dense_t: int
    sparse_t: int
    t_ratio: float
    dense_qubits: int
    sparse_qubits: int
    q_ratio: float
    dense_fidelity: float
    sparse_fidelity: float


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_point(
    n: int,
    card: int,
    b: int,
    seeds: int,
    use_select_swap: bool,
    measure_qubits: bool,
) -> Row:
    dense_t = estimate_alias_sampling_t_count(n, b, use_select_swap=use_select_swap)
    sparse_t = estimate_sparse_alias_sampling_t_count(
        card, n, b, use_select_swap=use_select_swap)

    dense_fids: List[float] = []
    sparse_fids: List[float] = []
    dense_q = 0
    sparse_q = 0
    for s in range(1, seeds + 1):
        state, n_bits = random_state(n, card, seed=s)
        dense_fids.append(qrom_alias_analytical_fidelity(state, n_bits, b))
        sparse_fids.append(sparse_qrom_alias_analytical_fidelity(state, n_bits, b))
        if measure_qubits and s == 1:
            dense_q = synthesize_alias_sampling(
                state, n_bits, b, use_select_swap=use_select_swap).qubit_count
            sparse_q = synthesize_sparse_alias_sampling(
                state, n_bits, b, use_select_swap=use_select_swap).qubit_count

    sparse_bins = 1 << max(1, (card - 1).bit_length()) if card > 1 else 2
    return Row(
        n_bits=n,
        card=card,
        precision_bits=b,
        dense_bins=1 << n,
        sparse_bins=sparse_bins,
        dense_t=dense_t,
        sparse_t=sparse_t,
        t_ratio=dense_t / sparse_t if sparse_t else float("inf"),
        dense_qubits=dense_q,
        sparse_qubits=sparse_q,
        q_ratio=dense_q / sparse_q if sparse_q else 0.0,
        dense_fidelity=_avg(dense_fids),
        sparse_fidelity=_avg(sparse_fids),
    )


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", default="8,10,12,14",
                   help="Comma-separated qubit counts (default: 8,10,12,14)")
    p.add_argument("--card", default="4,8,16",
                   help="Comma-separated nonzero-amplitude counts (default: 4,8,16)")
    p.add_argument("-b", "--precision", type=int, default=6,
                   help="Precision bits for keep probabilities (default: 6)")
    p.add_argument("--seeds", type=int, default=3,
                   help="Random states averaged per point (default: 3)")
    p.add_argument("--no-swap", action="store_true",
                   help="Use basic SELECT instead of SELECT-SWAP")
    p.add_argument("--measure-qubits", action="store_true", default=True,
                   help="Build circuits (seed 1) to report exact qubit counts")
    p.add_argument("--no-measure-qubits", dest="measure_qubits", action="store_false")
    p.add_argument("--csv", type=str, default=None, help="Write rows to CSV path")
    args = p.parse_args()

    ns = _parse_int_list(args.n)
    cards = _parse_int_list(args.card)
    use_select_swap = not args.no_swap

    rows: List[Row] = []
    for n in ns:
        for card in cards:
            if card > (1 << n):
                continue
            # Building the 2^n dense circuit for qubit counts gets heavy past
            # ~14 qubits; fall back to estimate-only there.
            measure = args.measure_qubits and n <= 14
            rows.append(run_point(n, card, args.precision, args.seeds,
                                  use_select_swap, measure))

    header = (f"{'n':>3} {'card':>4} {'b':>3} | "
              f"{'dense_bins':>10} {'sp_bins':>7} | "
              f"{'dense_T':>9} {'sparse_T':>9} {'T x':>6} | "
              f"{'dQ':>6} {'sQ':>5} {'Q x':>5} | "
              f"{'dense_fid':>9} {'sparse_fid':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        qx = f"{r.q_ratio:5.1f}" if r.q_ratio else "    -"
        dq = f"{r.dense_qubits:6d}" if r.dense_qubits else "     -"
        sq = f"{r.sparse_qubits:5d}" if r.sparse_qubits else "    -"
        print(f"{r.n_bits:3d} {r.card:4d} {r.precision_bits:3d} | "
              f"{r.dense_bins:10d} {r.sparse_bins:7d} | "
              f"{r.dense_t:9d} {r.sparse_t:9d} {r.t_ratio:6.1f} | "
              f"{dq} {sq} {qx} | "
              f"{r.dense_fidelity:9.5f} {r.sparse_fidelity:10.5f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        print(f"\nWrote {len(rows)} rows to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
