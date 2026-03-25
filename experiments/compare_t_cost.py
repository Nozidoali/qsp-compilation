"""Compare T-cost of rotation-based vs QROM-based state preparation.

Usage
-----
    python -m experiments.compare_t_cost [options]

Examples
--------
    python -m experiments.compare_t_cost --n-max 5 -v
    python -m experiments.compare_t_cost -f w --precision 4,8
    python -m experiments.compare_t_cost --benchmarks-dir benchmarks/
    python -m experiments.compare_t_cost --dump-benchmarks benchmarks/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from qiskit.quantum_info import Statevector

from qsp.metrics import clifford_t_counts, count_cnots, count_ry, estimate_t_count
from qsp.qrom.alias_sampling import estimate_alias_sampling_t_count, synthesize_alias_sampling
from qsp.rotation.dense import prepare_state_dense
from qsp.rotation.sparse import prepare_state_sparse
from qsp.state import (
    FAMILIES,
    State,
    dump_benchmarks,
    load_state,
    state_to_json,
    to_statevector,
)


@dataclass
class Result:
    # --- state ---
    state_tag: str = ""
    family: str = ""
    n: int = 0
    m: int = 0
    # --- method ---
    approach: str = ""
    precision_bits: int = 0
    # --- primary metrics ---
    t_count: int = 0
    gs_t_count: int = 0
    qubit_count: int = 0
    gate_count: int = 0
    # --- detailed gate counts ---
    ry_count: int = 0
    cnot_count: int = 0
    h_count: int = 0
    s_count: int = 0
    x_count: int = 0
    z_count: int = 0
    ccx_count: int = 0
    rz_count: int = 0
    # --- quality ---
    fidelity: float = 0.0
    time_ms: float = 0.0
    success: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        # compact: drop zero counts
        for k in list(d):
            if isinstance(d[k], (int, float)) and d[k] == 0 and k not in ("n", "m", "precision_bits"):
                del d[k]
        return d


# ---------------------------------------------------------------------------
# State generation
# ---------------------------------------------------------------------------

def generate_state(family: str, n: int, seed: int) -> tuple[State, int]:
    gen = FAMILIES.get(family)
    if gen is not None:
        return gen(n, seed)
    # legacy aliases
    from qsp.state import dense_uniform
    if family == "dense_uniform":
        return dense_uniform(n)
    return {0: 1.0}, n


# ---------------------------------------------------------------------------
# Fidelity helpers
# ---------------------------------------------------------------------------

def _fidelity_rotation(qc, target: State, n_bits: int) -> float:
    try:
        sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
        sim = sv.data[: 1 << n_bits]
        tgt = to_statevector(target, n_bits).astype(complex)
        return float(np.abs(np.vdot(tgt, sim)) ** 2)
    except Exception:
        return 0.0


def _fidelity_qrom(qc, target: State, n_bits: int) -> float:
    try:
        sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
        probs = np.zeros(1 << n_bits)
        for idx, amp in enumerate(sv.data):
            probs[idx & ((1 << n_bits) - 1)] += abs(amp) ** 2
        tgt = to_statevector(target, n_bits)
        tgt_probs = tgt ** 2
        return float(np.sum(np.sqrt(probs * tgt_probs)) ** 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# GridSynth T-count
# ---------------------------------------------------------------------------

_gs_cache: Dict[tuple, int] = {}


def _gridsynth_t_count(qc, precision_bits: int) -> int:
    epsilon = 2.0 ** (-precision_bits)
    total = 0
    for instr in qc.data:
        name = instr.operation.name
        if name == "ry":
            theta = float(instr.operation.params[0])
            if abs(theta) < 1e-12 or abs(abs(theta) - math.pi) < 1e-12:
                continue
            key = (round(theta * 1e10), precision_bits)
            if key not in _gs_cache:
                warnings.filterwarnings("ignore")
                from pygridsynth import gridsynth_gates
                _gs_cache[key] = gridsynth_gates(theta, epsilon).count("T")
            total += _gs_cache[key]
        elif name == "ccx":
            total += 4
        elif name in ("t", "tdg"):
            total += 1
    return total


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _fill_gate_counts(r: Result, qc) -> None:
    ct = clifford_t_counts(qc)
    r.h_count = ct["h"]
    r.s_count = ct["s"]
    r.x_count = ct["x"]
    r.z_count = ct["z"]
    r.ccx_count = ct["ccx"]
    r.rz_count = ct["rz"]


def run_rotation(
    state: State, n_bits: int, tag: str, family: str, strategy: str,
    timeout: float, precision_bits: int = 0,
) -> Result:
    r = Result(state_tag=tag, family=family, n=n_bits, m=len(state),
               approach=f"rotation_{strategy}", precision_bits=precision_bits)
    start = time.monotonic()
    try:
        if strategy == "sparse":
            qc = prepare_state_sparse(state, n_bits, timeout)
        else:
            qc = prepare_state_dense(state, n_bits, timeout)
        r.time_ms = (time.monotonic() - start) * 1000.0
        r.t_count = estimate_t_count(qc)
        r.ry_count = count_ry(qc)
        r.cnot_count = count_cnots(qc)
        r.qubit_count = qc.num_qubits
        r.gate_count = qc.size()
        _fill_gate_counts(r, qc)
        if n_bits <= 10:
            r.fidelity = _fidelity_rotation(qc, state, n_bits)
        if precision_bits > 0 and r.ry_count > 0:
            r.gs_t_count = _gridsynth_t_count(qc, precision_bits)
    except Exception:
        r.success = False
        r.time_ms = (time.monotonic() - start) * 1000.0
    return r


def run_qrom(
    state: State, n_bits: int, tag: str, family: str,
    precision_bits: int, use_select_swap: bool,
) -> Result:
    approach = "selectswap" if use_select_swap else "qrom"
    r = Result(state_tag=tag, family=family, n=n_bits, m=len(state),
               approach=approach, precision_bits=precision_bits)
    start = time.monotonic()
    try:
        res = synthesize_alias_sampling(state, n_bits, precision_bits, use_select_swap)
        r.time_ms = res.compile_time_ms
        r.t_count = res.t_count
        r.cnot_count = res.cnot_count
        r.qubit_count = res.qubit_count
        r.gate_count = res.qc.size()
        _fill_gate_counts(r, res.qc)
        if res.qc.num_qubits <= 25:
            r.fidelity = _fidelity_qrom(res.qc, state, n_bits)
    except Exception:
        r.success = False
        r.time_ms = (time.monotonic() - start) * 1000.0
    return r


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def _state_iter(args):
    """Yield (tag, family, state, n_bits) tuples from either --benchmarks-dir or generators."""
    if args.benchmarks_dir:
        bdir = Path(args.benchmarks_dir)
        for p in sorted(bdir.glob("*.json")):
            state, n_bits, meta = load_state(p)
            family = meta.get("family", p.stem)
            yield p.stem, family, state, n_bits
    else:
        families = (
            list(FAMILIES.keys()) if args.family == "all" else [args.family]
        )
        for family in families:
            max_n = min(args.n_max, 12) if "dense" in family else args.n_max
            is_random = "random" in family
            for n in range(args.n_min, max_n + 1):
                num_seeds = args.seeds if is_random else 1
                for seed in range(1, num_seeds + 1):
                    state, n_bits = generate_state(family, n, seed)
                    tag = f"{family}_n{n}" + (f"_s{seed}" if is_random else "")
                    yield tag, family, state, n_bits


def run_benchmarks(args) -> List[Result]:
    precisions = [int(x) for x in args.precision.split(",")]
    results: List[Result] = []

    for tag, family, state, n_bits in _state_iter(args):
        if args.verbose:
            print(f"  {tag} (m={len(state)})", file=sys.stderr)

        for b in precisions:
            for strat in ["sparse", "dense"]:
                r = run_rotation(state, n_bits, tag, family, strat,
                                 args.timeout, precision_bits=b)
                results.append(r)
                if args.verbose:
                    _print_result(r)

            r_basic = run_qrom(state, n_bits, tag, family, b, False)
            results.append(r_basic)
            if args.verbose:
                _print_result(r_basic)
            r_swap = run_qrom(state, n_bits, tag, family, b, True)
            results.append(r_swap)
            if args.verbose:
                _print_result(r_swap)

    return results


def _print_result(r: Result) -> None:
    status = "" if r.success else " [FAILED]"
    fid_str = f" fid={r.fidelity:.4f}" if r.fidelity > 0 else ""
    ry_str = f" Ry={r.ry_count:4d}" if r.ry_count > 0 else ""
    gs_str = f" GS_T={r.gs_t_count:6d}" if r.gs_t_count > 0 else ""
    print(
        f"    {r.approach:16s} b={r.precision_bits:2d}"
        f" T={r.t_count:8d}{ry_str}{gs_str} CNOT={r.cnot_count:6d}"
        f" q={r.qubit_count:4d} t={r.time_ms:8.1f}ms{fid_str}{status}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_json(results: List[Result], path: Path) -> None:
    data = [r.to_dict() for r in results]
    path.write_text(json.dumps(data, indent=2))


def write_csv(results: List[Result], path: Path) -> None:
    import csv
    header = [
        "state_tag", "family", "n", "m", "approach", "precision_bits",
        "t_count", "gs_t_count", "ry_count", "h_count", "s_count",
        "x_count", "z_count", "cnot_count", "ccx_count", "rz_count",
        "qubit_count", "gate_count", "time_ms", "fidelity", "success",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in results:
            d = asdict(r)
            w.writerow([d[k] for k in header])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QSP T-cost comparison benchmark")
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("-f", "--family", default="all")
    parser.add_argument("-b", "--precision", default="1,2,4,8")
    parser.add_argument("-s", "--seeds", type=int, default=3)
    parser.add_argument("-t", "--timeout", type=float, default=120.0)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    # JSON / benchmarks
    parser.add_argument("--benchmarks-dir", default=None,
                        help="Load states from JSON files in this directory")
    parser.add_argument("--dump-benchmarks", default=None, metavar="DIR",
                        help="Generate benchmark states as JSON and exit")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON instead of CSV")
    args = parser.parse_args()

    # Dump benchmarks and exit
    if args.dump_benchmarks:
        written = dump_benchmarks(
            args.dump_benchmarks, args.n_min, args.n_max, args.seeds,
        )
        print(f"Wrote {len(written)} benchmark files to {args.dump_benchmarks}")
        return

    results = run_benchmarks(args)

    suffix = ".json" if args.json else ".csv"
    out_path = Path(args.output) if args.output else Path(f"qsp_t_cost_results{suffix}")
    if args.json:
        write_json(results, out_path)
    else:
        write_csv(results, out_path)
    print(f"Wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
