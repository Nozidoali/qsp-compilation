import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alias_sampling import dump_alias_table_csv
from state import load_state


def main():
    parser = argparse.ArgumentParser(description="Dump the alias sampling table of a state JSON as CSV")
    parser.add_argument("state", type=Path)
    parser.add_argument("-m", "--mode", default="sparse", choices=("sparse", "dense"))
    parser.add_argument("-b", "--precision", type=int, default=8)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.state.exists():
        print(f"Error: {args.state} not found", file=sys.stderr)
        sys.exit(1)

    state, num_qubits = load_state(args.state)
    table_csv, probabilities_csv = dump_alias_table_csv(state, num_qubits, args.precision, sparse=(args.mode == "sparse"), state_file=str(args.state))

    if args.output:
        prefix = args.output.with_suffix("") if args.output.suffix == ".csv" else args.output
        for suffix, text in ((".table.csv", table_csv), (".probabilities.csv", probabilities_csv)):
            Path(str(prefix) + suffix).write_text(text)
            print(f"Wrote {prefix}{suffix}", file=sys.stderr)
    else:
        sys.stdout.write(table_csv + "\n" + probabilities_csv)


if __name__ == "__main__":
    main()
