"""Compile a single quantum state to circuits and export QASM + stats.

Usage
-----
    python -m experiments.run state.json -o output_dir/
    python -m experiments.run state.json -o output_dir/ -b 4,8 -a sparse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qiskit.qasm2 import dumps as qasm2_dumps

from experiments.compare_t_cost import (
    ALL_APPROACHES,
    ROTATION_APPROACHES,
    Result,
    SELECTSWAP,
    run_approach,
)
from qsp.circuit_utils import decompose_to_clifford_t
from qsp.qrom.alias_sampling import synthesize_alias_sampling
from qsp.state import load_state

APPROACHES = list(ALL_APPROACHES)


def _emit_clifford_t_qasm(approach, qc, state, n_bits, precision_bits):
    """QASM of the Clifford+T variant. For rotation approaches we lower via
    pygridsynth; for QROM we re-synthesize with native AND gates instead of CCX.
    """
    if qc is None or precision_bits <= 0:
        return None
    if approach in ROTATION_APPROACHES:
        return qasm2_dumps(decompose_to_clifford_t(qc, precision_bits))
    res_ct = synthesize_alias_sampling(
        state, n_bits, precision_bits,
        use_select_swap=(approach == SELECTSWAP),
        clifford_t=True,
    )
    return qasm2_dumps(res_ct.qc)


def compile_state(
    state_path: Path,
    precisions: list[int],
    approaches: list[str],
    timeout: float,
    verbose: bool,
    on_result=None,
) -> tuple[list[dict], list[Result]]:
    """Compile one state with given approaches/precisions.

    Returns ``(outputs, results)`` where ``outputs`` is the legacy list of
    dicts (with ``qasm`` / ``qasm_ct`` strings) used by write_outputs, and
    ``results`` is the raw list of Result dataclasses used by write_csv.

    If ``on_result`` is given, it is called as ``on_result(r)`` after every
    (approach, precision) combination completes — useful for streaming a
    CSV row to disk before the full sweep finishes (crash-safe progress).
    """
    state, n_bits = load_state(state_path)
    tag = state_path.stem
    parts = tag.rsplit("_n", 1)
    family = parts[0] if len(parts) > 1 else tag

    outputs: list[dict] = []
    results: list[Result] = []

    for b in precisions:
        for approach in approaches:
            r, qc = run_approach(
                approach, state, n_bits, tag, family,
                precision_bits=b, timeout=timeout,
            )
            results.append(r)
            qasm_str = qasm2_dumps(qc) if qc is not None else None
            qasm_ct_str = _emit_clifford_t_qasm(approach, qc, state, n_bits, b)

            d = r.to_dict()
            d["qasm"] = qasm_str
            d["qasm_ct"] = qasm_ct_str
            outputs.append(d)
            if on_result is not None:
                on_result(r)

            if verbose:
                status = "" if r.success else " [FAILED]"
                fid = f" fid={r.fidelity:.4f}" if r.fidelity > 0 else ""
                print(
                    f"  {r.approach:16s} b={r.precision_bits:2d}"
                    f" T={r.t_count:8d} q={r.qubit_count:4d}"
                    f" t={r.time_ms:8.1f}ms{fid}{status}",
                    file=sys.stderr,
                )

    return outputs, results


def write_outputs(outputs: list[dict], out_dir: Path, tag: str) -> None:
    """Write QASM files and a combined stats JSON to out_dir.

    Original circuits go to out_dir/original/, Clifford+T circuits
    go to out_dir/clifford_t/.
    """
    orig_dir = out_dir / "original"
    ct_dir = out_dir / "clifford_t"
    orig_dir.mkdir(parents=True, exist_ok=True)
    ct_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for entry in outputs:
        qasm_str = entry.pop("qasm", None)
        qasm_ct_str = entry.pop("qasm_ct", None)
        approach = entry.get("approach", "unknown")
        b = entry.get("precision_bits", 0)
        suffix = f"{approach}_b{b}"

        if qasm_str:
            (orig_dir / f"{tag}_{suffix}.qasm").write_text(qasm_str)

        if qasm_ct_str:
            (ct_dir / f"{tag}_{suffix}.qasm").write_text(qasm_ct_str)

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

    outputs, _ = compile_state(args.state, precisions, approaches, args.timeout, args.verbose)
    write_outputs(outputs, args.output_dir, tag)

    print(f"Wrote {len(outputs)} results to {args.output_dir}/")


if __name__ == "__main__":
    main()
