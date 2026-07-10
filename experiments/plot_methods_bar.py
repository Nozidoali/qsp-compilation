"""Grouped bar plot of T-count per method across state families.

One figure, one subplot per state family. Within a subplot, the x-axis groups
by qubit count ``n`` and each method is a colored bar (log-scale T-count). This
compares the previous methods (dense / sparse rotation, and the two original
alias-sampling variants: basic SELECT = ``qrom`` and SELECT-SWAP) against the
improved sparse alias sampling (``sparse_alias``).

Runs the benchmark in-process, so a single command produces the plot:

    python -m experiments.plot_methods_bar
    python -m experiments.plot_methods_bar --families w,dicke_2,sparse_random \\
        --n-min 4 --n-max 10 -b 6 -o methods_bar.png
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.compare_t_cost import (
    DENSE, QROM, SELECTSWAP, SPARSE, SPARSEALIAS,
    generate_state, run_approach,
)

# Method display order, labels, and colors. Okabe-Ito palette (colorblind-safe);
# the improved method gets the standout bluish-green.
METHODS = [
    (DENSE,       "dense rotation",   "#CC79A7"),  # reddish purple
    (SPARSE,      "sparse rotation",  "#56B4E9"),  # sky blue
    (QROM,        "alias: SELECT",    "#D55E00"),  # vermillion
    (SELECTSWAP,  "alias: SELECT-SWAP", "#E69F00"),  # orange
    (SPARSEALIAS, "alias: sparse (improved)", "#009E73"),  # bluish green
]

# Families that are genuinely sparse (where the improved method is meant to win).
DEFAULT_FAMILIES = ["w", "dicke_2", "sparse_uniform", "sparse_random"]


def collect(families, n_min, n_max, precision, seeds, timeout):
    """Return data[family][method][n] = mean T-count over seeds (None if failed)."""
    data = {f: defaultdict(dict) for f in families}
    for family in families:
        is_random = "random" in family
        max_n = min(n_max, 12) if "dense" in family else n_max
        num_seeds = seeds if is_random else 1
        for n in range(n_min, max_n + 1):
            for method, _, _ in METHODS:
                ts = []
                for seed in range(1, num_seeds + 1):
                    state, n_bits = generate_state(family, n, seed)
                    r, _ = run_approach(method, state, n_bits, f"{family}_n{n}",
                                        family, precision_bits=precision, timeout=timeout)
                    # T-count is valid even if the optional fidelity simulator
                    # (qiskit_aer) is unavailable and flipped success to False.
                    if r.t_count > 0:
                        ts.append(r.t_count)
                if ts:
                    data[family][method][n] = sum(ts) / len(ts)
            print(f"  {family} n={n} done", file=sys.stderr)
    return data


def plot(data, families, precision, output):
    n_fam = len(families)
    ncols = min(n_fam, 2)
    nrows = (n_fam + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for idx, family in enumerate(families):
        ax = axes[idx]
        fam_data = data[family]
        ns = sorted({n for m in fam_data.values() for n in m})
        if not ns:
            ax.set_visible(False)
            continue

        x = np.arange(len(ns))
        n_methods = len(METHODS)
        width = 0.8 / n_methods

        for k, (method, label, color) in enumerate(METHODS):
            vals = [fam_data.get(method, {}).get(n, np.nan) for n in ns]
            offset = (k - (n_methods - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=label, color=color,
                   edgecolor="black", linewidth=0.3)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("n (qubits)", fontsize=10)
        ax.set_ylabel("T-count (log)", fontsize=10)
        ax.set_title(family, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, which="major")
        ax.grid(True, axis="y", alpha=0.1, which="minor")

    for idx in range(len(families), len(axes)):
        axes[idx].set_visible(False)

    # Single shared legend at the top.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"State-preparation T-count by method (precision b={precision})",
                 fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved plot to {output}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                   help=f"Comma-separated families (default: {','.join(DEFAULT_FAMILIES)})")
    p.add_argument("--n-min", type=int, default=4)
    p.add_argument("--n-max", type=int, default=10)
    p.add_argument("-b", "--precision", type=int, default=6)
    p.add_argument("--seeds", type=int, default=2,
                   help="Seeds averaged for random families (default: 2)")
    p.add_argument("-t", "--timeout", type=float, default=120.0)
    p.add_argument("-o", "--output", default="methods_bar.png")
    args = p.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    print(f"Running {families}, n={args.n_min}..{args.n_max}, b={args.precision}",
          file=sys.stderr)
    data = collect(families, args.n_min, args.n_max, args.precision,
                   args.seeds, args.timeout)
    plot(data, families, args.precision, args.output)


if __name__ == "__main__":
    main()
