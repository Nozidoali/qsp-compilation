"""Compare THC benchmark states by total gate cost.

Total gate cost is defined as:

    total_gate_cost = CNOT + T + H + S

where T is the estimated T-count used in compare_t_cost.py.

Usage
-----
    python -m experiments.compare_thc_total_gate_cost
    python -m experiments.compare_thc_total_gate_cost --family li --M 350 500 -v
    python -m experiments.compare_thc_total_gate_cost --approaches qrom,selectswap --precision 8
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from experiments.compare_t_cost import ALL_APPROACHES, run_approach
from qsp.state import load_state


DEFAULT_BENCHMARKS_DIR = Path("states/THC_Fomoco/data")
APPROACH_CHOICES = ALL_APPROACHES
FAMILY_CHOICES = ("li", "reiher")
SOURCE_CHOICES = ("approximated", "exact")

THC_FILE_RE = re.compile(
    r"^thc_(?P<family>li|reiher)_(?P<source>[a-zA-Z0-9_]+)_M(?P<M>\d+)_nonorthogonal_prepare$"
)


@dataclass
class THCCostRecord:
    state_tag: str
    file_name: str
    family: str
    source: str
    M: int
    n: int
    m: int
    approach: str
    precision_bits: int
    total_gate_cost: int
    t_count: int
    cnot_count: int
    h_count: int
    s_count: int
    qubit_count: int
    gate_count: int
    time_ms: float
    success: bool


def parse_precisions(precision: str) -> List[int]:
    vals = [x.strip() for x in precision.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one precision value is required.")
    return [int(x) for x in vals]


def parse_approaches(value: str) -> List[str]:
    approaches = [x.strip() for x in value.split(",") if x.strip()]
    if not approaches:
        raise ValueError("At least one approach is required.")
    bad = [x for x in approaches if x not in APPROACH_CHOICES]
    if bad:
        raise ValueError(f"Unsupported approaches: {bad}. Choices: {APPROACH_CHOICES}")
    return approaches


def iter_thc_files(
    benchmarks_dir: Path,
    family_filter: Optional[Sequence[str]],
    source_filter: Optional[Sequence[str]],
    m_filter: Optional[Sequence[int]],
) -> List[tuple[Path, str, str, int]]:
    files: List[tuple[Path, str, str, int]] = []
    for path in sorted(benchmarks_dir.glob("*.json")):
        match = THC_FILE_RE.match(path.stem)
        if not match:
            continue
        family = match.group("family")
        source = match.group("source")
        rank_m = int(match.group("M"))

        if family_filter is not None and family not in family_filter:
            continue
        if source_filter is not None and source not in source_filter:
            continue
        if m_filter is not None and rank_m not in m_filter:
            continue

        files.append((path, family, source, rank_m))
    return files


CSV_COLUMNS = [
    "state_tag", "file_name", "family", "source", "M", "n", "m",
    "approach", "precision_bits", "total_gate_cost",
    "t_count", "cnot_count", "h_count", "s_count",
    "qubit_count", "gate_count", "time_ms", "success",
]


def write_csv(records: List[THCCostRecord], out_path: Path) -> None:
    from experiments.csv_utils import write_dataclass_csv
    write_dataclass_csv(
        records, out_path, CSV_COLUMNS,
        formatters={"time_ms": lambda v: f"{v:.3f}"},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute THC total gate cost where total = CNOT + T + H + S."
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=DEFAULT_BENCHMARKS_DIR,
        help=f"Directory with THC benchmark JSON files (default: {DEFAULT_BENCHMARKS_DIR}).",
    )
    parser.add_argument(
        "--family",
        choices=(*FAMILY_CHOICES, "all"),
        default="all",
        help="Which FeMoco family to include.",
    )
    parser.add_argument(
        "--source",
        choices=(*SOURCE_CHOICES, "all"),
        default="approximated",
        help="Which THC source type to include.",
    )
    parser.add_argument(
        "--M",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of THC ranks M to run.",
    )
    parser.add_argument(
        "--approaches",
        default="sparse,qrom,selectswap",
        help="Comma-separated approaches among: sparse,dense,qrom,selectswap.",
    )
    parser.add_argument(
        "--precision",
        default="8",
        help="Comma-separated precision bits, e.g. 2,4,8.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout (seconds) for rotation-based synthesis.",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="Optional cap on number of THC files to process (useful for smoke tests).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("thc_total_gate_cost.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-run progress.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    approaches = parse_approaches(args.approaches)
    precisions = parse_precisions(args.precision)

    family_filter = None if args.family == "all" else [args.family]
    source_filter = None if args.source == "all" else [args.source]
    m_filter = args.M

    files = iter_thc_files(
        benchmarks_dir=args.benchmarks_dir,
        family_filter=family_filter,
        source_filter=source_filter,
        m_filter=m_filter,
    )
    if args.max_states is not None:
        files = files[: max(args.max_states, 0)]

    if not files:
        raise FileNotFoundError(
            f"No THC benchmark files matched filters in {args.benchmarks_dir}."
        )

    records: List[THCCostRecord] = []
    total_runs = len(files) * len(approaches) * len(precisions)
    run_id = 0

    for path, family, source, rank_m in files:
        state, n_bits = load_state(path)
        state_tag = path.stem
        for precision_bits in precisions:
            for approach in approaches:
                run_id += 1
                if args.verbose:
                    print(
                        f"[{run_id}/{total_runs}] {state_tag} "
                        f"({approach}, b={precision_bits}, m={len(state)})",
                        file=sys.stderr,
                    )
                result, _ = run_approach(
                    approach,
                    state,
                    n_bits,
                    state_tag,
                    family,
                    precision_bits=precision_bits,
                    timeout=args.timeout,
                )
                total = result.cnot_count + result.t_count + result.h_count + result.s_count

                records.append(
                    THCCostRecord(
                        state_tag=state_tag,
                        file_name=path.name,
                        family=family,
                        source=source,
                        M=rank_m,
                        n=n_bits,
                        m=len(state),
                        approach=approach,
                        precision_bits=precision_bits,
                        total_gate_cost=total,
                        t_count=result.t_count,
                        cnot_count=result.cnot_count,
                        h_count=result.h_count,
                        s_count=result.s_count,
                        qubit_count=result.qubit_count,
                        gate_count=result.gate_count,
                        time_ms=result.time_ms,
                        success=result.success,
                    )
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(records, args.output)
    print(f"Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
