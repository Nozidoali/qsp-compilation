"""Compile all state JSONs in a directory to QASM circuits + stats.

Usage
-----
    python -m experiments.run_all benchmarks/
    python -m experiments.run_all benchmarks/ -o results/ -b 4,8
    python -m experiments.run_all benchmarks/ -a sparse,qrom -v
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from experiments.compare_t_cost import CSV_COLUMNS, write_csv
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
    parser.add_argument("--csv", type=Path, default=None,
                        help="Also write a combined CSV (default: alongside summary JSON).")
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

    # Open the combined CSV for streaming writes: one row per (state, approach, b)
    # is appended + flushed as soon as the point is computed, so partial data
    # survives a crash / Ctrl-C mid-sweep.
    summary_path = args.summary or (args.output_dir / "summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or summary_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_stats = []
    all_results = []
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_COLUMNS)

        def _stream_row(r):
            d = asdict(r)
            csv_writer.writerow([d.get(col, "") for col in CSV_COLUMNS])
            csv_file.flush()

        for i, state_path in enumerate(state_files, 1):
            tag = state_path.stem
            if args.verbose:
                print(f"[{i}/{len(state_files)}] {tag}", file=sys.stderr)

            outputs, results = compile_state(
                state_path, precisions, approaches, args.timeout, args.verbose,
                on_result=_stream_row,
            )
            write_outputs(outputs, args.output_dir / tag, tag)
            all_stats.extend(outputs)
            all_results.extend(results)

    # summary.json written at end (full sweep). Per-point progress is already in CSV.
    summary_path.write_text(json.dumps(all_stats, indent=2))

    print(f"Processed {len(state_files)} states, {len(all_stats)} results")
    print(f"Per-state outputs: {args.output_dir}/<state_tag>/")
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")


if __name__ == "__main__":
    main()
