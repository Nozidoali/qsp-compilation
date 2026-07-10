"""Compile a single state JSON to a single QASM circuit.

The minimal JSON-in / QASM-out entry point: pick one approach and one
precision, get one OpenQASM 2 string (stdout by default, or ``-o file``).

Usage
-----
    python -m experiments.compile state.json
    python -m experiments.compile state.json -a selectswap -b 8 -o out.qasm
    python -m experiments.compile state.json -a sparse_alias -b 6 --clifford-t

The state JSON format matches ``qsp.state.save_state`` (``n_bits`` plus an
``amplitudes`` map of binary-string keys to real amplitudes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qiskit.qasm2 import dumps as qasm2_dumps

from experiments.compare_t_cost import ALL_APPROACHES, SPARSE, run_approach
from experiments.run import _emit_clifford_t_qasm
from qsp.state import load_state


def compile_to_qasm(
    state_path: Path,
    approach: str = SPARSE,
    precision_bits: int = 8,
    clifford_t: bool = False,
    timeout: float = 120.0,
) -> str:
    """Compile ``state_path`` with one approach and return an OpenQASM 2 string.

    When ``clifford_t`` is set, the returned QASM is the Clifford+T lowering
    (rotations via gridsynth, QROM via native AND gates); otherwise it is the
    raw circuit (Ry/CRY/CCX-level).
    """
    state, n_bits = load_state(state_path)
    tag = state_path.stem
    family = tag.rsplit("_n", 1)[0]

    r, qc = run_approach(
        approach, state, n_bits, tag, family,
        precision_bits=precision_bits, timeout=timeout,
    )
    if qc is None or not r.success:
        raise RuntimeError(f"compilation failed for approach {approach!r}")

    if clifford_t:
        qasm = _emit_clifford_t_qasm(approach, qc, state, n_bits, precision_bits)
        if qasm is None:
            raise RuntimeError(
                f"no Clifford+T QASM available for approach {approach!r} "
                f"at precision {precision_bits}"
            )
        return qasm
    return qasm2_dumps(qc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile one state JSON to one QASM circuit"
    )
    parser.add_argument("state", type=Path, help="Path to state JSON file")
    parser.add_argument("-a", "--approach", default=SPARSE,
                        choices=ALL_APPROACHES,
                        help=f"Synthesis approach (default: {SPARSE})")
    parser.add_argument("-b", "--precision", type=int, default=8,
                        help="Precision bits (default: 8)")
    parser.add_argument("--clifford-t", action="store_true",
                        help="Emit the Clifford+T lowering instead of the raw circuit")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Write QASM to this file (default: stdout)")
    parser.add_argument("-t", "--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if not args.state.exists():
        print(f"Error: {args.state} not found", file=sys.stderr)
        sys.exit(1)

    qasm = compile_to_qasm(
        args.state,
        approach=args.approach,
        precision_bits=args.precision,
        clifford_t=args.clifford_t,
        timeout=args.timeout,
    )

    if args.output:
        args.output.write_text(qasm)
        print(f"Wrote QASM to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(qasm)


if __name__ == "__main__":
    main()
