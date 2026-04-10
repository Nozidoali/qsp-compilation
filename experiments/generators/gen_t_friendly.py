"""Generate T-friendly benchmark states.

Usage
-----
    python -m experiments.generators.gen_t_friendly
    python -m experiments.generators.gen_t_friendly -o benchmarks/t_friendly/ --n-max 6 -s 3 -v
"""
from __future__ import annotations

import argparse
from pathlib import Path

from qsp.t_friendly import dump_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate T-friendly benchmark states as JSON"
    )
    parser.add_argument("-o", "--out-dir", type=Path,
                        default=Path("benchmarks/t_friendly"),
                        help="Output directory (default: benchmarks/t_friendly/)")
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--k-max", type=int, default=2,
                        help="Max T gates per angle (default: 2)")
    parser.add_argument("-s", "--samples", type=int, default=5,
                        help="Random states per n (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    written = dump_benchmarks(
        args.out_dir, args.n_min, args.n_max,
        args.k_max, args.samples, args.seed, args.verbose,
    )
    print(f"Wrote {len(written)} benchmark files to {args.out_dir}")


if __name__ == "__main__":
    main()
