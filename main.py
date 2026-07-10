import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qiskit.qasm2 import dumps as qasm2_dumps
from state import load_state

ROTATION_METHODS = ("sparse", "dense")
QROM_METHODS = ("qrom", "selectswap", "sparse_alias")
METHODS = ROTATION_METHODS + QROM_METHODS


def synthesize(state, n_bits, method, precision_bits, clifford_t, timeout):
    if method in ROTATION_METHODS:
        if method == "sparse":
            from sparse import prepare_state_sparse
            qc = prepare_state_sparse(state, n_bits, timeout)
        else:
            from dense import prepare_state_dense
            qc = prepare_state_dense(state, n_bits, timeout)
        if clifford_t:
            from circuit_utils import decompose_to_clifford_t
            qc = decompose_to_clifford_t(qc, precision_bits)
        return qc
    from alias_sampling import synthesize_alias_sampling, synthesize_sparse_alias_sampling
    if method == "sparse_alias":
        return synthesize_sparse_alias_sampling(state, n_bits, precision_bits, use_select_swap=True, clifford_t=clifford_t)
    return synthesize_alias_sampling(state, n_bits, precision_bits, use_select_swap=(method == "selectswap"), clifford_t=clifford_t)


def compile_to_qasm(state_path, method="sparse", precision_bits=8, clifford_t=False, timeout=120.0):
    state, n_bits = load_state(state_path)
    return qasm2_dumps(synthesize(state, n_bits, method, precision_bits, clifford_t, timeout))


def main():
    parser = argparse.ArgumentParser(description="Compile a state JSON to a QASM circuit")
    parser.add_argument("state", type=Path)
    parser.add_argument("-m", "--method", default="sparse", choices=METHODS)
    parser.add_argument("-b", "--precision", type=int, default=8)
    parser.add_argument("--clifford-t", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-t", "--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if not args.state.exists():
        print(f"Error: {args.state} not found", file=sys.stderr)
        sys.exit(1)

    qasm = compile_to_qasm(args.state, args.method, args.precision, args.clifford_t, args.timeout)

    if args.output:
        args.output.write_text(qasm)
        print(f"Wrote QASM to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(qasm)


if __name__ == "__main__":
    main()
