import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qiskit.qasm2 import dumps as qasm2_dumps
from state import load_state

ROTATION_METHODS = ("sparse", "dense")
QROM_METHODS = ("qrom", "selectswap", "sparse_alias")
METHODS = ROTATION_METHODS + QROM_METHODS


def synthesize(state, num_qubits, method, precision_bits, clifford_t):
    if method in ROTATION_METHODS:
        if method == "sparse":
            from cardinality_reduction import prepare_state_by_cardinality_reduction
            circuit = prepare_state_by_cardinality_reduction(state, num_qubits)
        else:
            from qubit_reduction import prepare_state_by_qubit_reduction
            circuit = prepare_state_by_qubit_reduction(state, num_qubits)
        if clifford_t:
            from clifford_t import decompose_to_clifford_t
            circuit = decompose_to_clifford_t(circuit, precision_bits)
        return circuit
    from alias_sampling import synthesize_alias_sampling, synthesize_sparse_alias_sampling
    if method == "sparse_alias":
        return synthesize_sparse_alias_sampling(state, num_qubits, precision_bits, use_select_swap=True, clifford_t=clifford_t)
    return synthesize_alias_sampling(state, num_qubits, precision_bits, use_select_swap=(method == "selectswap"), clifford_t=clifford_t)


def compile_to_qasm(state_path, method="sparse", precision_bits=8, clifford_t=False):
    state, num_qubits = load_state(state_path)
    return qasm2_dumps(synthesize(state, num_qubits, method, precision_bits, clifford_t))


def main():
    parser = argparse.ArgumentParser(description="Compile a state JSON to a QASM circuit")
    parser.add_argument("state", type=Path)
    parser.add_argument("-m", "--method", default="sparse", choices=METHODS)
    parser.add_argument("-b", "--precision", type=int, default=8)
    parser.add_argument("--clifford-t", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.state.exists():
        print(f"Error: {args.state} not found", file=sys.stderr)
        sys.exit(1)

    qasm = compile_to_qasm(args.state, args.method, args.precision, args.clifford_t)

    if args.output:
        args.output.write_text(qasm)
        print(f"Wrote QASM to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(qasm)


if __name__ == "__main__":
    main()
