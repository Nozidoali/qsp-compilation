"""Timing harness for fidelity verification on THC_water.

Runs three preparation approaches end-to-end and times both synthesis and
verification:

* dense rotation (Qiskit StatePreparation-style reduction tree)
* sparse rotation (iterative cardinality reduction)
* QROM alias sampling (verified ANALYTICALLY, no simulation)

For the rotation paths we measure two verification methods:
  - `rotation_fidelity` : dense |<target|sim>|^2 via Qiskit Statevector
  - `mps_marginal_fidelity` : Aer MPS sim + Bhattacharyya on shots
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from qsp import prepare_state_dense, prepare_state_sparse
from qsp.qrom.alias_sampling import synthesize_alias_sampling
from qsp.state import load_state
from qsp.metrics import (
    mps_marginal_fidelity,
    qrom_alias_analytical_fidelity,
    rotation_fidelity,
)


def timeit(label, fn):
    t0 = time.monotonic()
    try:
        out = fn()
        dt = time.monotonic() - t0
        print(f"  [{dt:7.2f}s] {label}", flush=True)
        return out, dt, None
    except Exception as e:
        dt = time.monotonic() - t0
        print(f"  [{dt:7.2f}s] {label} FAILED: {e}", flush=True)
        return None, dt, e


def bench_rotation(name, prep_fn, state, n_bits, shots, synth_timeout, run_exact):
    print(f"\n=== {name} rotation prep ===", flush=True)
    qc, t_syn, err = timeit(f"synthesize ({name})", lambda: prep_fn(state, n_bits, synth_timeout))
    if err is not None:
        return
    print(f"     qubits={qc.num_qubits}, gates={qc.size()}, depth={qc.depth()}", flush=True)

    if run_exact:
        fid_exact, t_exact, _ = timeit(
            "rotation_fidelity (Statevector.evolve)",
            lambda: rotation_fidelity(qc, state, n_bits),
        )
        if fid_exact is not None:
            print(f"     fidelity = {fid_exact:.6f}", flush=True)

    fid_mps, t_mps, _ = timeit(
        f"mps_marginal_fidelity (shots={shots})",
        lambda: mps_marginal_fidelity(qc, state, n_bits, shots=shots),
    )
    if fid_mps is not None:
        print(f"     fidelity = {fid_mps:.6f}", flush=True)


def bench_qrom(state, n_bits, precision_bits):
    print(f"\n=== QROM alias sampling (analytical, b={precision_bits}) ===")
    _, t_syn, _ = timeit(
        "synthesize_alias_sampling (SELECT-SWAP, for gate-count info)",
        lambda: synthesize_alias_sampling(state, n_bits, precision_bits, use_select_swap=True),
    )
    fid, t_ana, _ = timeit(
        "qrom_alias_analytical_fidelity",
        lambda: qrom_alias_analytical_fidelity(state, n_bits, precision_bits),
    )
    if fid is not None:
        print(f"     fidelity = {fid:.8f}  (1-f = {1-fid:.2e})")


def main():
    path = REPO / "benchmarks" / "thc" / "THC_water.json"
    print(f"[load] {path}", flush=True)
    state, n_bits = load_state(str(path))
    print(f"  n_bits={n_bits}, nonzeros={len(state)}, 2^n={1 << n_bits}", flush=True)

    shots = int(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else 20000
    synth_timeout = float(sys.argv[sys.argv.index("--timeout") + 1]) if "--timeout" in sys.argv else 600.0
    precision_bits = 8
    run_exact = "--exact" in sys.argv  # skip Statevector.evolve by default on water
    skip_sparse = "--no-sparse" in sys.argv

    bench_qrom(state, n_bits, precision_bits)
    bench_rotation("dense", prepare_state_dense, state, n_bits, shots, synth_timeout, run_exact)
    if not skip_sparse:
        bench_rotation("sparse", prepare_state_sparse, state, n_bits, shots, synth_timeout, run_exact)


if __name__ == "__main__":
    main()
