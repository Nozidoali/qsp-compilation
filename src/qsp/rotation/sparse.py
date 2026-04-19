"""Cardinality reduction strategy for state preparation.

Iteratively reduces the number of nonzero amplitudes (cardinality) by one
at each step, using controlled rotations that merge pairs of basis states.

Ported from ``strategy-sparse.cpp``.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Set, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit

from qsp.state import EPS, State, cardinality, state_hash
from qsp.rotation.common import (
    apply_controlled_cx,
    apply_controlled_cx_arr,
    apply_controlled_cx_np,
    apply_mcry,
    apply_mcry_arr,
    apply_mcry_np,
    apply_x,
    maximize_difference_once,
    maximize_difference_once_np,
)


def cardinality_reduction_by_one(
    state: State, n_bits: int
) -> Tuple[State, List[dict]]:
    # Convert to arrays once; stay in array land until the final return.
    idx_arr = np.fromiter(state.keys(), dtype=np.int64, count=len(state))
    amp_arr = np.fromiter(state.values(), dtype=np.float64, count=len(state))

    # --- Pivot selection: reduce index set to a single element ---
    indices_np = idx_arr.copy()
    diff_values: Dict[int, bool] = {}
    diff_qubit = 0
    while len(indices_np) > 1:
        diff_qubit, indices_np = maximize_difference_once_np(
            n_bits, indices_np, diff_values,
        )
    index0 = int(indices_np[0])
    diff_values.pop(diff_qubit, None)

    # --- Candidate filter (vectorized on idx_arr) ---
    mask = idx_arr != index0
    for q, v in diff_values.items():
        bit_q = ((idx_arr >> q) & 1).astype(bool)
        mask &= bit_q if v else ~bit_q
    candidates_np = idx_arr[mask]
    while len(candidates_np) > 1:
        _, candidates_np = maximize_difference_once_np(
            n_bits, candidates_np, diff_values,
        )
    index1 = int(candidates_np[0])

    gates: List[dict] = []
    diff_val = bool((index0 >> diff_qubit) & 1)

    # --- CX alignment: flip target bits of index1 to match index0 ---
    for qubit in range(n_bits):
        if ((index0 >> qubit) & 1) == ((index1 >> qubit) & 1):
            continue
        if qubit == diff_qubit:
            continue
        gates.append({"type": "cx", "control": diff_qubit,
                      "ctrl_val": diff_val, "target": qubit})
        idx_arr, amp_arr = apply_controlled_cx_arr(
            idx_arr, amp_arr, diff_qubit, diff_val, qubit,
        )

    ctrls = list(diff_values.keys())
    phases = [diff_values[q] for q in ctrls]

    idx0 = index1 & ~(1 << diff_qubit)
    idx1 = index1 | (1 << diff_qubit)
    pos0 = np.where(idx_arr == idx0)[0]
    pos1 = np.where(idx_arr == idx1)[0]
    assert len(pos0) == 1 and len(pos1) == 1, \
        f"expected idx0={idx0} and idx1={idx1} in state after CX alignment"
    w0 = float(amp_arr[pos0[0]])
    w1 = float(amp_arr[pos1[0]])
    theta = 2.0 * math.atan2(w0, w1)
    if (index1 >> diff_qubit) & 1:
        theta = -math.pi + theta

    gates.append({"type": "mcry", "ctrls": ctrls, "phases": phases,
                  "theta": theta, "target": diff_qubit})
    idx_arr, amp_arr = apply_mcry_arr(
        idx_arr, amp_arr, ctrls, phases, theta, diff_qubit,
    )

    new_state = dict(zip(idx_arr.tolist(), amp_arr.tolist()))
    return new_state, gates


def prepare_state_sparse(
    state: State,
    n_bits: int,
    timeout: float = 120.0,
    use_gidney: bool = False,
) -> QuantumCircuit:
    """Cardinality-reduction sparse state preparation.

    By default multi-controlled Ry gates are emitted via Qiskit's
    ``RYGate.control(k)`` (keeps the circuit width at ``n_bits`` qubits but
    decomposes exponentially in k). Pass ``use_gidney=True`` to expand MCRY
    via a Gidney-AND V-chain with up to ``n_bits - 2`` clean ancillas,
    matching the Gleinig-Hoefler O(n) CX per reduction bound at the cost of
    ~k wider circuit width.
    """
    curr = dict(state)
    all_gates: List[dict] = []

    # Each iteration reduces cardinality by at most 1, so |state| iterations
    # is a tight upper bound. Small slack covers non-reducing iterations
    # before the seen-hash / no_reduction guards fire.
    max_iter = len(state) + 16
    seen: set = set()
    no_reduction = 0
    prev_card = cardinality(curr)
    start = time.monotonic()
    timed_out = False

    for _ in range(max_iter):
        if cardinality(curr) <= 1:
            break
        if time.monotonic() - start >= timeout:
            timed_out = True
            break

        h = state_hash(curr, n_bits)
        if h in seen:
            break
        seen.add(h)

        new_state, gates = cardinality_reduction_by_one(curr, n_bits)
        new_card = cardinality(new_state)
        if new_card >= prev_card:
            no_reduction += 1
            if no_reduction > 3:
                break
        else:
            all_gates.extend(gates)
            curr = new_state
            no_reduction = 0
        prev_card = cardinality(curr)

    if cardinality(curr) > 1:
        elapsed = time.monotonic() - start
        reason = "timed out" if timed_out else "stalled (cycle/non-reducing)"
        raise RuntimeError(
            f"prepare_state_sparse {reason} with {cardinality(curr)} residual "
            f"amplitudes (start card={len(state)}, n_bits={n_bits}, "
            f"elapsed={elapsed:.1f}s, timeout={timeout}s)"
        )

    if curr:
        index = next(iter(curr.keys()))
        for qubit in range(n_bits):
            if (index >> qubit) & 1:
                all_gates.append({"type": "x", "target": qubit})

    all_gates.reverse()

    if use_gidney:
        max_ctrls = max(
            (len(g["ctrls"]) for g in all_gates if g["type"] == "mcry"),
            default=0,
        )
        n_anc = max(0, max_ctrls - 1)
    else:
        n_anc = 0
    qc = QuantumCircuit(n_bits + n_anc)
    ancillas = list(range(n_bits, n_bits + n_anc))
    _emit_gates(qc, all_gates, ancillas=ancillas, use_gidney=use_gidney)
    return qc


def _emit_gates(
    qc: QuantumCircuit,
    gates: List[dict],
    ancillas: List[int] | None = None,
    use_gidney: bool = False,
) -> None:
    ancillas = ancillas or []
    for g in gates:
        if g["type"] == "x":
            qc.x(g["target"])
        elif g["type"] == "cx":
            ctrl = g["control"]
            target = g["target"]
            ctrl_val = g["ctrl_val"]
            if not ctrl_val:
                qc.x(ctrl)
            qc.cx(ctrl, target)
            if not ctrl_val:
                qc.x(ctrl)
        elif g["type"] == "mcry":
            theta = -g["theta"]
            ctrls = g["ctrls"]
            phases = g["phases"]
            target = g["target"]
            if not ctrls:
                qc.ry(theta, target)
            else:
                for i, p in enumerate(phases):
                    if not p:
                        qc.x(ctrls[i])
                if use_gidney:
                    from qsp.circuit_utils import mcry_gidney
                    mcry_gidney(qc, theta, ctrls, target,
                                ancillas[: max(0, len(ctrls) - 1)])
                elif len(ctrls) == 1:
                    qc.cry(theta, ctrls[0], target)
                else:
                    from qiskit.circuit.library import RYGate
                    gate = RYGate(theta).control(len(ctrls))
                    qc.append(gate, ctrls + [target])
                for i, p in enumerate(phases):
                    if not p:
                        qc.x(ctrls[i])
