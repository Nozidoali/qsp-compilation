"""Generate THC (tensor hypercontraction) benchmark states.

Wraps the export script in states/THC_Fomoco/. Requires Lee-code
FeMoco data files to be present.

Usage
-----
    python -m experiments.generators.gen_thc
    python -m experiments.generators.gen_thc --family li --M 50 100
"""
from __future__ import annotations

import sys
from pathlib import Path

# The THC export lives alongside its data files; add it to sys.path.
_THC_DIR = Path(__file__).resolve().parents[2] / "states" / "THC_Fomoco"
if str(_THC_DIR) not in sys.path:
    sys.path.insert(0, str(_THC_DIR))

from export_thc_benchmarks import export_family, FAMILIES, SOURCES  # noqa: E402

import argparse  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate THC benchmark states as JSON"
    )
    parser.add_argument("-o", "--out-dir", type=Path,
                        default=Path("benchmarks/thc"),
                        help="Output directory (default: benchmarks/thc/)")
    parser.add_argument("--family", nargs="+", choices=FAMILIES,
                        default=list(FAMILIES),
                        help="Which FeMoco Hamiltonian family to export")
    parser.add_argument("--source", choices=SOURCES, default="approximated",
                        help="Use approximated or exact THC factors")
    parser.add_argument("--M", dest="ranks", type=int, nargs="*", default=None,
                        help="THC ranks to export (default: all available)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for family in args.family:
        written.extend(export_family(family, args.source, args.ranks, args.out_dir))

    for path in written:
        print(path)
    print(f"Wrote {len(written)} benchmark files to {args.out_dir}")


if __name__ == "__main__":
    main()
