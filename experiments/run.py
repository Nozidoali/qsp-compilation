"""Compile a single quantum state to circuits and export QASM + stats.

Usage
-----
    python -m experiments.run state.json -o output_dir/
    python -m experiments.run state.json -o output_dir/ -b 4,8 -a rotation_sparse
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from qiskit.qasm2 import dumps as qasm2_dumps

from experiments.compare_t_cost import (
    Result,
    _fill_gate_counts,
    _fidelity_qrom,
    _fidelity_rotation,
    _gridsynth_t_count,
)
from qsp.metrics import count_cnots, count_ry, estimate_t_count
from qsp.qrom.alias_sampling import synthesize_alias_sampling
from qsp.rotation.dense import prepare_state_dense
from qsp.rotation.sparse import prepare_state_sparse
from qsp.state import load_state

APPROACHES = ["rotation_sparse", "rotation_dense", "qrom", "selectswap"]


def compile_state(
    state_path: Path,
    precisions: list[int],
    approaches: list[str],
    timeout: float,
    verbose: bool,
) -> list[dict]:
    """Compile one state with given approaches/precisions.

    Returns list of result dicts, each with an extra ``qasm`` key.
    """
    state, n_bits = load_state(state_path)
    tag = state_path.stem
    parts = tag.rsplit("_n", 1)
    family = parts[0] if len(parts) > 1 else tag

    outputs = []

    for b in precisions:
        for approach in approaches:
            r = Result(state_tag=tag, family=family, n=n_bits, m=len(state),
                       approach=approach, precision_bits=b)
            qasm_str = None
            start = time.monotonic()

            try:
                if approach.startswith("rotation"):
                    strategy = approach.split("_", 1)[1]
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
                    if b > 0 and r.ry_count > 0:
                        r.gs_t_count = _gridsynth_t_count(qc, b)
                    qasm_str = qasm2_dumps(qc)
                else:
                    use_select_swap = approach == "selectswap"
                    res = synthesize_alias_sampling(state, n_bits, b, use_select_swap)
                    r.time_ms = res.compile_time_ms
                    r.t_count = res.t_count
                    r.cnot_count = res.cnot_count
                    r.qubit_count = res.qubit_count
                    r.gate_count = res.qc.size()
                    _fill_gate_counts(r, res.qc)
                    if res.qc.num_qubits <= 25:
                        r.fidelity = _fidelity_qrom(res.qc, state, n_bits)
                    qasm_str = qasm2_dumps(res.qc)
            except Exception:
                r.success = False
                r.time_ms = (time.monotonic() - start) * 1000.0

            d = r.to_dict()
            d["qasm"] = qasm_str
            outputs.append(d)

            if verbose:
                status = "" if r.success else " [FAILED]"
                fid = f" fid={r.fidelity:.4f}" if r.fidelity > 0 else ""
                print(
                    f"  {r.approach:16s} b={r.precision_bits:2d}"
                    f" T={r.t_count:8d} q={r.qubit_count:4d}"
                    f" t={r.time_ms:8.1f}ms{fid}{status}",
                    file=sys.stderr,
                )

    return outputs


def write_outputs(outputs: list[dict], out_dir: Path, tag: str) -> None:
    """Write QASM files and a combined stats JSON to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for entry in outputs:
        qasm_str = entry.pop("qasm", None)
        approach = entry.get("approach", "unknown")
        b = entry.get("precision_bits", 0)
        suffix = f"{approach}_b{b}"

        if qasm_str:
            qasm_path = out_dir / f"{tag}_{suffix}.qasm"
            qasm_path.write_text(qasm_str)

        stats.append(entry)

    stats_path = out_dir / f"{tag}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile a single state JSON to QASM circuits + stats"
    )
    parser.add_argument("state", type=Path, help="Path to state JSON file")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./output"),
                        help="Directory for output files (default: ./output/)")
    parser.add_argument("-b", "--precision", default="4",
                        help="Comma-separated precision bits (default: 4)")
    parser.add_argument("-a", "--approach", default=None,
                        help=f"Comma-separated approaches (default: all). "
                             f"Options: {', '.join(APPROACHES)}")
    parser.add_argument("-t", "--timeout", type=float, default=120.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.state.exists():
        print(f"Error: {args.state} not found", file=sys.stderr)
        sys.exit(1)

    precisions = [int(x) for x in args.precision.split(",")]
    approaches = (
        [a.strip() for a in args.approach.split(",")]
        if args.approach else APPROACHES
    )

    tag = args.state.stem
    if args.verbose:
        print(f"Compiling {tag} ...", file=sys.stderr)

    outputs = compile_state(args.state, precisions, approaches, args.timeout, args.verbose)
    write_outputs(outputs, args.output_dir, tag)

    print(f"Wrote {len(outputs)} results to {args.output_dir}/")


if __name__ == "__main__":
    main()
