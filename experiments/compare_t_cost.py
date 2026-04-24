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
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from qsp.metrics import (
    aer_rotation_fidelity,
    clifford_t_counts,
    count_cnots,
    count_ry,
    estimate_t_count,
    qrom_alias_analytical_fidelity,
    qrom_marginal_fidelity,
    rotation_fidelity,
)
from qsp.qrom.alias_sampling import estimate_alias_sampling_t_count, synthesize_alias_sampling
from qsp.rotation.dense import prepare_state_dense
from qsp.rotation.sparse import prepare_state_sparse
from qsp.state import (
    FAMILIES,
    State,
    dump_benchmarks,
    load_state,
)


# Canonical approach names. Used by all experiment scripts and recorded in CSV/JSON.
SPARSE = "sparse"
DENSE = "dense"
QROM = "qrom"
SELECTSWAP = "selectswap"
ROTATION_APPROACHES = (SPARSE, DENSE)
QROM_APPROACHES = (QROM, SELECTSWAP)
ALL_APPROACHES = ROTATION_APPROACHES + QROM_APPROACHES


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


def _run_rotation(state: State, n_bits: int, r: Result, timeout: float, precision_bits: int):
    from qsp.circuit_utils import decompose_to_clifford_t
    start = time.monotonic()
    try:
        prep = prepare_state_sparse if r.approach == SPARSE else prepare_state_dense
        qc = prep(state, n_bits, timeout)
        r.ry_count = count_ry(qc)

        # Measure T count, CNOT count, and fidelity on the Clifford+T circuit
        # you would actually run at this precision (so (T(b), 1-F(b)) is a
        # coherent Pareto point). We keep `qc` (raw Ry/CRY) as the returned
        # circuit so run.py's original-vs-clifford_t QASM split still makes
        # sense; the decompose result is used only for measurement.
        qc_measure = qc
        if precision_bits > 0 and r.ry_count > 0:
            qc_measure = decompose_to_clifford_t(qc, precision_bits)

        r.time_ms = (time.monotonic() - start) * 1000.0
        r.t_count = estimate_t_count(qc_measure)
        r.gs_t_count = r.t_count  # kept for plotting backwards-compat
        r.cnot_count = count_cnots(qc_measure)
        r.qubit_count = qc_measure.num_qubits
        r.gate_count = qc_measure.size()
        _fill_gate_counts(r, qc_measure)
        if qc_measure.num_qubits <= 28:
            r.fidelity = aer_rotation_fidelity(qc_measure, state, n_bits)
        return qc
    except Exception as e:
        r.success = False
        print(f"Error in rotation: {e}")
        traceback.print_exc()
        r.time_ms = (time.monotonic() - start) * 1000.0
        return None


def _run_qrom(state: State, n_bits: int, r: Result, precision_bits: int):
    start = time.monotonic()
    try:
        res = synthesize_alias_sampling(
            state, n_bits, precision_bits, use_select_swap=(r.approach == SELECTSWAP),
        )
        r.time_ms = res.compile_time_ms
        r.t_count = res.t_count
        r.cnot_count = res.cnot_count
        r.qubit_count = res.qubit_count
        r.gate_count = res.qc.size()
        _fill_gate_counts(r, res.qc)
        # Analytical Bhattacharyya over the alias table — no simulator needed,
        # works at any qubit count / n_entries.
        r.fidelity = qrom_alias_analytical_fidelity(state, n_bits, precision_bits)
        return res.qc
    except Exception:
        r.success = False
        r.time_ms = (time.monotonic() - start) * 1000.0
        return None


def run_approach(
    approach: str, state: State, n_bits: int, tag: str, family: str,
    precision_bits: int, timeout: float = 120.0,
) -> tuple[Result, "QuantumCircuit | None"]:
    """Run a single approach. Returns (Result, compiled circuit or None on failure)."""
    if approach not in ALL_APPROACHES:
        raise ValueError(f"Unknown approach {approach!r}; choose from {ALL_APPROACHES}")
    r = Result(state_tag=tag, family=family, n=n_bits, m=len(state),
               approach=approach, precision_bits=precision_bits)
    if approach in ROTATION_APPROACHES:
        qc = _run_rotation(state, n_bits, r, timeout, precision_bits)
    else:
        qc = _run_qrom(state, n_bits, r, precision_bits)
    return r, qc


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def _state_iter(args):
    """Yield (tag, family, state, n_bits) tuples from either --benchmarks-dir or generators."""
    if args.benchmarks_dir:
        bdir = Path(args.benchmarks_dir)
        for p in sorted(bdir.glob("*.json")):
            state, n_bits = load_state(p)
            # infer family from filename: e.g. "w_n3" -> "w", "sparse_random_n5_s1" -> "sparse_random"
            parts = p.stem.rsplit("_n", 1)
            family = parts[0] if len(parts) > 1 else p.stem
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
            for approach in ALL_APPROACHES:
                r, _ = run_approach(approach, state, n_bits, tag, family,
                                    precision_bits=b, timeout=args.timeout)
                results.append(r)
                if args.verbose:
                    _print_result(r)

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


CSV_COLUMNS = [
    "state_tag", "family", "n", "m", "approach", "precision_bits",
    "t_count", "gs_t_count", "ry_count", "h_count", "s_count",
    "x_count", "z_count", "cnot_count", "ccx_count", "rz_count",
    "qubit_count", "gate_count", "time_ms", "fidelity", "success",
]


def write_csv(results: List[Result], path: Path) -> None:
    from experiments.csv_utils import write_dataclass_csv
    write_dataclass_csv(results, path, CSV_COLUMNS)


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
