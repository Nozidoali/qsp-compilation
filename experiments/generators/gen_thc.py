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

from export_thc_benchmarks import main  # noqa: E402

if __name__ == "__main__":
    main()
