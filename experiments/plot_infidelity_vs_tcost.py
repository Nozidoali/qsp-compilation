"""Plot infidelity vs T-count from pre-computed CSV results.

Reads CSV files produced by compare_t_friendly and/or compare_t_cost,
then plots infidelity vs T-count for each method.

Usage:
    # First run the experiment (does gridsynth, may take a few minutes):
    python -m experiments.compare_t_friendly -v

    # Then plot (instant):
    python -m experiments.plot_infidelity_vs_tcost
    python -m experiments.plot_infidelity_vs_tcost -i my_results.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_t_friendly_csv(path: Path):
    """Read CSV from compare_t_friendly, return list of row dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def pareto_filter(data):
    """Keep only Pareto-optimal points (lower T and lower infidelity)."""
    if not data:
        return data
    pts = sorted(data, key=lambda p: p[1])
    filtered = [pts[0]]
    best_inf = pts[0][0]
    for inf, t in pts[1:]:
        if inf < best_inf:
            filtered.append((inf, t))
            best_inf = inf
    return filtered


def plot_t_friendly(rows, ax, n_filter=None):
    """Plot infidelity vs T-count for T-friendly experiment results."""
    methods = {
        "dense_tfriendly": {"color": "#7B1FA2", "marker": "*", "ls": "none",
                            "label": "T-friendly (exact)", "ms": 14, "zorder": 10},
        "sparse":          {"color": "#1565C0", "marker": "o", "ls": "-",
                            "label": "sparse rotation", "ms": 5, "zorder": 5},
        "qrom":            {"color": "#C62828", "marker": "^", "ls": "--",
                            "label": "QROM", "ms": 5, "zorder": 5},
        "selectswap":      {"color": "#E65100", "marker": "D", "ls": "--",
                            "label": "select-swap", "ms": 5, "zorder": 5},
    }

    data = defaultdict(list)

    for row in rows:
        n = int(row["n_bits"])
        if n_filter is not None and n != n_filter:
            continue

        dense_t = int(row["dense_t"])
        dense_inf = float(row["dense_infidelity"])
        data["dense_tfriendly"].append((max(dense_inf, 1e-14), dense_t))

        sparse_t = int(row["sparse_t"])
        if sparse_t >= 0:
            sparse_inf = float(row["sparse_infidelity"])
            data["sparse"].append((max(sparse_inf, 1e-14), sparse_t))

        qrom_t = int(row["qrom_t"])
        qrom_inf = float(row["qrom_infidelity"])
        data["qrom"].append((max(qrom_inf, 1e-14), qrom_t))

        sswap_t = int(row["selectswap_t"])
        sswap_inf = float(row["selectswap_infidelity"])
        data["selectswap"].append((max(sswap_inf, 1e-14), sswap_t))

    for method, style in methods.items():
        pts = data.get(method, [])
        if not pts:
            continue
        if method == "dense_tfriendly":
            # Show each sample as a star (no pareto filtering)
            infids, tcounts = zip(*pts)
            ax.plot(infids, tcounts, marker=style["marker"], color=style["color"],
                    label=style["label"], linewidth=0, markersize=style["ms"],
                    alpha=0.85, linestyle="none", markeredgecolor="black",
                    markeredgewidth=0.5, zorder=style["zorder"])
        else:
            pts = pareto_filter(pts)
            if not pts:
                continue
            infids, tcounts = zip(*pts)
            ax.plot(infids, tcounts, marker=style["marker"], color=style["color"],
                    label=style["label"], linewidth=1.8, markersize=style["ms"],
                    alpha=0.85, linestyle=style["ls"], zorder=style["zorder"])


def main():
    parser = argparse.ArgumentParser(
        description="Plot infidelity vs T-count from CSV results"
    )
    parser.add_argument("-i", "--input", default="t_friendly_results.csv",
                        help="Input CSV (default: t_friendly_results.csv)")
    parser.add_argument("-o", "--output", default="infidelity_vs_tcost.png",
                        help="Output plot (default: infidelity_vs_tcost.png)")
    parser.add_argument("-n", "--n-bits", type=int, default=None,
                        help="Filter to specific n (default: all)")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run compare_t_friendly first:")
        print("  python -m experiments.compare_t_friendly -v")
        return

    rows = read_t_friendly_csv(csv_path)
    print(f"Read {len(rows)} rows from {csv_path}")

    # Get unique n values
    n_values = sorted(set(int(r["n_bits"]) for r in rows))
    if args.n_bits is not None:
        n_values = [args.n_bits]

    ncols = min(len(n_values), 3)
    nrows = (len(n_values) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if len(n_values) == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, n in enumerate(n_values):
        if idx >= len(axes):
            break
        ax = axes[idx]
        plot_t_friendly(rows, ax, n_filter=n)

        ax.set_xscale("log")
        ax.set_xlabel("Infidelity (1 - F)", fontsize=10)
        ax.set_ylabel("T-count", fontsize=10)
        ax.set_title(f"n = {n} qubits", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25, which="major")
        ax.grid(True, alpha=0.1, which="minor")
        ax.invert_xaxis()
        ax.legend(fontsize=8, loc="upper left")

    # Hide unused axes
    for idx in range(len(n_values), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Infidelity vs T-count: T-Friendly Rotation vs QROM",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
