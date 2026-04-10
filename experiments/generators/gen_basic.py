"""Generate basic benchmark states (W, Dicke, sparse/dense random).

Usage
-----
    python -m experiments.generators.gen_basic
    python -m experiments.generators.gen_basic -o benchmarks/basic/ --n-max 6 -f w,dicke_2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from qsp.state import FAMILIES, dump_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate basic benchmark states as JSON"
    )
    parser.add_argument("-o", "--out-dir", type=Path,
                        default=Path("benchmarks/basic"),
                        help="Output directory (default: benchmarks/basic/)")
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("-s", "--seeds", type=int, default=3,
                        help="Random seeds per (family, n) (default: 3)")
    parser.add_argument("-f", "--families", default=None,
                        help=f"Comma-separated families (default: all). "
                             f"Options: {', '.join(FAMILIES)}")
    args = parser.parse_args()

    families = (
        [f.strip() for f in args.families.split(",")]
        if args.families else None
    )

    written = dump_benchmarks(
        args.out_dir, args.n_min, args.n_max, args.seeds, families,
    )
    print(f"Wrote {len(written)} benchmark files to {args.out_dir}")


if __name__ == "__main__":
    main()
