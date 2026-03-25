"""Plot infidelity vs T-count for all methods and state families.

Usage: python -m experiments.plot_infidelity_vs_tcost
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from qsp.state import (
    w_state, dicke_state, dense_uniform, sparse_uniform,
    sparse_random, random_dense_state, normalize, to_statevector,
)
from qsp.rotation.sparse import prepare_state_sparse
from qsp.rotation.dense import prepare_state_dense
from qsp.circuit_utils import decompose_to_clifford_t
from qsp.qrom.alias_sampling import (
    build_alias_table, synthesize_alias_sampling, estimate_alias_sampling_t_count,
)
from qiskit.quantum_info import Statevector


# --- helpers ---

def generate_state(family: str, n: int, seed: int = 1):
    if family == "w": return w_state(n)
    if family == "dicke_2": return dicke_state(n, 2)
    if family == "dense_uniform": return dense_uniform(n)
    if family == "dense_random": return random_dense_state(n, seed)
    if family == "sparse_uniform": return sparse_uniform(n, seed)
    if family == "sparse_random": return sparse_random(n, seed)
    if family.startswith("t_friendly"):
        theta = float(family.split("_")[-1])
        return build_product_state(n, theta)
    raise ValueError(family)


def build_product_state(n_bits: int, theta: float):
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    state = {}
    for idx in range(1 << n_bits):
        amp = 1.0
        for q in range(n_bits):
            amp *= s if ((idx >> q) & 1) else c
        if abs(amp) > 1e-15:
            state[idx] = amp
    return normalize(state), n_bits


def sim_fidelity(qc, state, n_bits: int) -> float:
    sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
    sim = sv.data[: 1 << n_bits]
    tgt = to_statevector(state, n_bits).astype(complex)
    return float(np.abs(np.vdot(tgt, sim)) ** 2)


def count_t_gates(qc) -> int:
    total = 0
    for instr in qc.data:
        name = instr.operation.name
        if name in ("t", "tdg"):
            total += 1
        elif name == "ccx":
            total += 4
    return total


def qrom_analytical_fidelity(state, n: int, b: int) -> float:
    table = build_alias_table(state, n, b)
    N = 1 << n
    mx = (1 << b) - 1
    p_approx = np.zeros(N)
    for i in range(N):
        kf = table.keep_values[i] / mx if mx > 0 else 1.0
        p_approx[i] += kf / N
        p_approx[table.alias_indices[i]] += (1.0 - kf) / N
    p_target = np.zeros(N)
    for idx, amp in state.items():
        p_target[idx] = amp * amp
    return float(np.sum(np.sqrt(p_target * p_approx)) ** 2)


# --- data collection ---

def collect_rotation_data(state, n_bits, method, precisions):
    """Returns list of (infidelity, t_count) for each precision."""
    try:
        if method == "sparse":
            qc = prepare_state_sparse(state, n_bits, timeout=60)
        else:
            qc = prepare_state_dense(state, n_bits, timeout=60)
    except Exception:
        return []

    results = []
    for b in precisions:
        try:
            qc_ct = decompose_to_clifford_t(qc, b)
            fid = sim_fidelity(qc_ct, state, n_bits)
            t = count_t_gates(qc_ct)
            inf = max(1 - fid, 1e-14)  # floor for log scale
            results.append((inf, t))
        except Exception:
            pass
    return results


def collect_qrom_data(state, n_bits, use_selectswap, precisions):
    """Returns list of (infidelity, t_count) for each precision."""
    results = []
    for b in precisions:
        try:
            fid = qrom_analytical_fidelity(state, n_bits, b)
            t = estimate_alias_sampling_t_count(n_bits, b, use_selectswap)
            inf = max(1 - fid, 1e-14)
            results.append((inf, t))
        except Exception:
            pass
    return results


def collect_t_friendly_rotation(n_bits, k):
    """T-friendly product state: exact T-count = n*k, infidelity = 0."""
    return [(1e-14, n_bits * k)]


def pareto_filter(data):
    """Keep only Pareto-optimal points: no other point has both lower T and lower infidelity.

    Sort by T-count ascending, then keep points where infidelity is strictly
    decreasing — this traces the efficient frontier.
    """
    if not data:
        return data
    # Sort by T-count ascending
    pts = sorted(data, key=lambda p: p[1])
    filtered = [pts[0]]
    best_inf = pts[0][0]
    for inf, t in pts[1:]:
        if inf < best_inf:
            filtered.append((inf, t))
            best_inf = inf
    return filtered


# --- main ---

def main():
    N = 5  # representative qubit count
    precisions = list(range(1, 21))

    # State families including T-friendly product states
    # T-friendly angles: pi/4 (k=1 T per qubit) and a k=2 angle
    t_theta_k1 = math.pi / 4
    t_theta_k2 = 2.0 * math.acos(math.sqrt((2 + math.sqrt(2)) / 4))  # k=2 angle

    families = {
        "W state": ("w", None),
        "Dicke (k=2)": ("dicke_2", None),
        "Dense random": ("dense_random", None),
        "Sparse uniform": ("sparse_uniform", None),
        "Sparse random": ("sparse_random", None),
        r"T-friendly product ($\theta=\pi/4$)": (f"t_friendly_{t_theta_k1}", 1),
    }

    methods = {
        "rotation_sparse": {"color": "#1565C0", "marker": "o", "ls": "-"},
        "rotation_dense":  {"color": "#2E7D32", "marker": "s", "ls": "-"},
        "qrom":            {"color": "#C62828", "marker": "^", "ls": "--"},
        "selectswap":      {"color": "#E65100", "marker": "D", "ls": "--"},
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (fam_label, (fam_key, k_exact)) in enumerate(families.items()):
        ax = axes[idx]
        state, nb = generate_state(fam_key, N)
        is_t_friendly = fam_key.startswith("t_friendly")

        for method_name, style in methods.items():
            # Skip sparse rotation for T-friendly (dense state, not meaningful)
            if is_t_friendly and method_name == "rotation_sparse":
                continue
            if method_name in ("rotation_sparse", "rotation_dense"):
                strategy = method_name.split("_")[1]
                data = collect_rotation_data(state, nb, strategy, precisions)
            elif method_name == "qrom":
                data = collect_qrom_data(state, nb, False, precisions)
            else:  # selectswap
                data = collect_qrom_data(state, nb, True, precisions)

            if not data:
                continue

            data = pareto_filter(data)
            infids, tcounts = zip(*data)
            ax.plot(infids, tcounts, marker=style["marker"], color=style["color"],
                    label=method_name, linewidth=1.8, markersize=5, alpha=0.85,
                    linestyle=style["ls"])

        # For T-friendly, add the exact rotation point
        if is_t_friendly and k_exact is not None:
            exact_t = N * k_exact
            ax.plot([1e-14], [exact_t], marker="*", color="#7B1FA2",
                    markersize=16, zorder=10, label=f"exact rotation (k={k_exact})",
                    markeredgecolor="black", markeredgewidth=0.5)

        ax.set_xscale("log")
        ax.set_xlabel("Infidelity (1 − F)", fontsize=10)
        ax.set_ylabel("T-count", fontsize=10)
        ax.set_title(fam_label, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.25, which="major")
        ax.grid(True, alpha=0.1, which="minor")
        ax.invert_xaxis()  # lower infidelity (better) on right

        # Add arrow showing "better" direction
        if idx == 0:
            ax.annotate("", xy=(1e-13, ax.get_ylim()[0] + 10),
                        xytext=(1e-1, ax.get_ylim()[0] + 10),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
            ax.text(1e-7, ax.get_ylim()[0] + 30, "← better fidelity",
                    fontsize=7, color="gray", ha="center")

    # Shared legend
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in all_labels:
                all_handles.append(hi)
                all_labels.append(li)

    fig.legend(all_handles, all_labels, loc="lower center", ncol=5,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01),
               edgecolor="gray", fancybox=True)

    fig.suptitle("Infidelity vs T-count Tradeoff (n = 5 qubits)",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = "infidelity_vs_tcost.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved plot to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
