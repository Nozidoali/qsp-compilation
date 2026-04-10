"""Compile all state JSONs in a directory to QASM circuits + stats.

Usage
-----
    python -m experiments.run_all benchmarks/
    python -m experiments.run_all benchmarks/ -o results/ -b 4,8
    python -m experiments.run_all benchmarks/ -a rotation_sparse,qrom -v
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.run import APPROACHES, compile_state, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile all state JSONs in a directory"
    )
    parser.add_argument("input_dir", type=Path,
                        help="Directory containing state JSON files")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./output"),
                        help="Root output directory (default: ./output/)")
    parser.add_argument("-b", "--precision", default="4",
                        help="Comma-separated precision bits (default: 4)")
    parser.add_argument("-a", "--approach", default=None,
                        help=f"Comma-separated approaches (default: all). "
                             f"Options: {', '.join(APPROACHES)}")
    parser.add_argument("-t", "--timeout", type=float, default=120.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--summary", type=Path, default=None,
                        help="Write combined summary JSON to this path")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    state_files = sorted(args.input_dir.glob("*.json"))
    if not state_files:
        print(f"No JSON files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    precisions = [int(x) for x in args.precision.split(",")]
    approaches = (
        [a.strip() for a in args.approach.split(",")]
        if args.approach else APPROACHES
    )

    all_stats = []
    for i, state_path in enumerate(state_files, 1):
        tag = state_path.stem
        if args.verbose:
            print(f"[{i}/{len(state_files)}] {tag}", file=sys.stderr)

        outputs = compile_state(
            state_path, precisions, approaches, args.timeout, args.verbose,
        )
        write_outputs(outputs, args.output_dir / tag, tag)
        all_stats.extend(outputs)

    # write combined summary
    summary_path = args.summary or (args.output_dir / "summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_stats, indent=2))

    print(f"Processed {len(state_files)} states, {len(all_stats)} results")
    print(f"Per-state outputs: {args.output_dir}/<state_tag>/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
