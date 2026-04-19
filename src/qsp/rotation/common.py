"""Shared helpers for rotation-based state preparation.

Includes pivot selection, don't-care optimization, and sparse state simulation
functions that manipulate ``dict[int, float]`` to track the evolving state
during iterative reduction.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from qsp.state import EPS, State


# --- State simulation on dict[int, float] ---

def apply_x(state: State, qubit: int) -> State:
    return {idx ^ (1 << qubit): amp for idx, amp in state.items()}


def apply_cx(state: State, control: int, target: int) -> State:
    new: State = {}
    for idx, amp in state.items():
        if (idx >> control) & 1:
            new[idx ^ (1 << target)] = amp
        else:
            new[idx] = amp
    return new


def apply_controlled_cx(state: State, control: int, ctrl_val: bool, target: int) -> State:
    new: State = {}
    for idx, amp in state.items():
        bit = (idx >> control) & 1
        if bool(bit) == ctrl_val:
            new[idx ^ (1 << target)] = amp
        else:
            new[idx] = amp
    return new


def apply_controlled_cx_np(
    state: State, control: int, ctrl_val: bool, target: int,
) -> State:
    """Vectorized ``apply_controlled_cx``.

    CNOT is a bijection on basis states, so flipping the target bit on the
    matching subset cannot collide with the passed-through subset — we can
    build the output dict directly from a ``np.where`` on the index array.
    """
    if not state:
        return {}
    idx_arr = np.fromiter(state.keys(), dtype=np.int64, count=len(state))
    amp_arr = np.fromiter(state.values(), dtype=np.float64, count=len(state))
    new_idx, new_amp = apply_controlled_cx_arr(
        idx_arr, amp_arr, control, ctrl_val, target,
    )
    return dict(zip(new_idx.tolist(), new_amp.tolist()))


def apply_controlled_cx_arr(
    idx_arr: np.ndarray, amp_arr: np.ndarray,
    control: int, ctrl_val: bool, target: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Array-in / array-out variant for hot-loop use without dict conversion."""
    bit = ((idx_arr >> control) & 1).astype(bool)
    match = bit == bool(ctrl_val)
    new_idx = np.where(match, idx_arr ^ (np.int64(1) << target), idx_arr)
    return new_idx, amp_arr


def apply_ry(state: State, qubit: int, theta: float) -> State:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    new: State = {}
    for idx, amp in state.items():
        idx0 = idx & ~(1 << qubit)
        idx1 = idx | (1 << qubit)
        if (idx >> qubit) & 1:
            new[idx0] = new.get(idx0, 0.0) - s * amp
            new[idx1] = new.get(idx1, 0.0) + c * amp
        else:
            new[idx0] = new.get(idx0, 0.0) + c * amp
            new[idx1] = new.get(idx1, 0.0) + s * amp
    return _prune(new)


def apply_mcry(state: State, ctrls: List[int], phases: List[bool],
               theta: float, target: int) -> State:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    new: State = {}
    for idx, amp in state.items():
        match = all(bool((idx >> q) & 1) == p for q, p in zip(ctrls, phases))
        if match:
            idx0 = idx & ~(1 << target)
            idx1 = idx | (1 << target)
            if (idx >> target) & 1:
                new[idx0] = new.get(idx0, 0.0) - s * amp
                new[idx1] = new.get(idx1, 0.0) + c * amp
            else:
                new[idx0] = new.get(idx0, 0.0) + c * amp
                new[idx1] = new.get(idx1, 0.0) + s * amp
        else:
            new[idx] = new.get(idx, 0.0) + amp
    return _prune(new)


def apply_mcry_np(
    state: State,
    ctrls: List[int],
    phases: List[bool],
    theta: float,
    target: int,
) -> State:
    """Vectorized multi-controlled Ry (dict wrapper over apply_mcry_arr)."""
    if not state:
        return {}
    idx_arr = np.fromiter(state.keys(), dtype=np.int64, count=len(state))
    amp_arr = np.fromiter(state.values(), dtype=np.float64, count=len(state))
    new_idx, new_amp = apply_mcry_arr(
        idx_arr, amp_arr, ctrls, phases, theta, target,
    )
    return dict(zip(new_idx.tolist(), new_amp.tolist()))


def apply_mcry_arr(
    idx_arr: np.ndarray,
    amp_arr: np.ndarray,
    ctrls: List[int],
    phases: List[bool],
    theta: float,
    target: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Array-in / array-out MCRY.

    Matching entries (ctrl bits == phases) rotate as pairs over the target
    bit; non-matching entries pass through. Since idx_arr has unique values,
    each (base, tbit) pair in the matching subset appears at most once, so a
    plain searchsorted-scatter suffices (no np.add.at needed).
    """
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    tgt_mask = np.int64(1) << target

    match = np.ones(len(idx_arr), dtype=bool)
    for q, p in zip(ctrls, phases):
        bit_q = ((idx_arr >> q) & 1).astype(bool)
        match &= (bit_q == bool(p))

    non_match_idx = idx_arr[~match]
    non_match_amp = amp_arr[~match]

    match_idx = idx_arr[match]
    match_amp = amp_arr[match]

    if len(match_idx) == 0:
        keep = np.abs(non_match_amp) > EPS
        return non_match_idx[keep], non_match_amp[keep]

    match_base = match_idx & ~tgt_mask
    match_tbit_1 = ((match_idx >> target) & 1).astype(bool)

    unique_bases = np.unique(match_base)
    pos = np.searchsorted(unique_bases, match_base)

    amp0 = np.zeros(len(unique_bases))
    amp1 = np.zeros(len(unique_bases))
    amp0[pos[~match_tbit_1]] = match_amp[~match_tbit_1]
    amp1[pos[match_tbit_1]] = match_amp[match_tbit_1]

    new_amp0 = c * amp0 - s * amp1
    new_amp1 = s * amp0 + c * amp1

    keep0 = np.abs(new_amp0) > EPS
    keep1 = np.abs(new_amp1) > EPS
    keep_nm = np.abs(non_match_amp) > EPS

    result_idx = np.concatenate([
        non_match_idx[keep_nm],
        unique_bases[keep0],
        (unique_bases | tgt_mask)[keep1],
    ])
    result_amp = np.concatenate([
        non_match_amp[keep_nm],
        new_amp0[keep0],
        new_amp1[keep1],
    ])
    return result_idx, result_amp


def apply_ucry(state: State, control_indices: List[int],
               angles: List[float], target: int) -> State:
    new: State = {}
    for idx, amp in state.items():
        rot_idx = 0
        for i, q in enumerate(control_indices):
            if (idx >> q) & 1:
                rot_idx |= 1 << i
        theta = angles[rot_idx]
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        idx0 = idx & ~(1 << target)
        idx1 = idx | (1 << target)
        if (idx >> target) & 1:
            new[idx0] = new.get(idx0, 0.0) - s * amp
            new[idx1] = new.get(idx1, 0.0) + c * amp
        else:
            new[idx0] = new.get(idx0, 0.0) + c * amp
            new[idx1] = new.get(idx1, 0.0) + s * amp
    return _prune(new)


def _prune(state: State) -> State:
    return {k: v for k, v in state.items() if abs(v) > EPS}


# --- Pivot selection helpers ---

def maximize_difference_once(
    n_bits: int,
    indices: Set[int],
    diff_values: Dict[int, bool],
) -> Tuple[int, bool]:
    max_diff = -1
    max_diff_indices_1: Set[int] = set()
    max_diff_qubit = 0
    max_diff_value = False
    length = len(indices)

    for qubit in range(n_bits):
        if qubit in diff_values:
            continue
        indices_1 = {idx for idx in indices if (idx >> qubit) & 1}
        diff = abs(length - 2 * len(indices_1))
        if diff == length:
            continue
        if diff > max_diff:
            max_diff = diff
            max_diff_indices_1 = indices_1
            max_diff_qubit = qubit
            max_diff_value = length > 2 * len(indices_1)
        if max_diff == length - 1:
            break

    if max_diff_value:
        indices.clear()
        indices.update(max_diff_indices_1)
    else:
        indices -= max_diff_indices_1

    diff_values[max_diff_qubit] = max_diff_value
    return max_diff_qubit, max_diff_value


def maximize_difference_once_np(
    n_bits: int,
    indices: np.ndarray,
    diff_values: Dict[int, bool],
) -> Tuple[int, np.ndarray]:
    """Vectorized pivot selection.

    Equivalent to :func:`maximize_difference_once` but takes / returns a
    ``numpy.int64`` array of basis-state indices (immutable-friendly). Picks
    the qubit whose 0/1 split is most unbalanced, records the minority value
    in ``diff_values``, and returns (qubit, filtered-indices) where
    filtered-indices is the minority subset.
    """
    length = len(indices)
    best_diff = -1
    best_q = 0
    best_val = False
    best_mask = None
    for q in range(n_bits):
        if q in diff_values:
            continue
        mask1 = ((indices >> q) & 1).astype(bool)
        c1 = int(mask1.sum())
        diff = abs(length - 2 * c1)
        if diff == length:
            continue
        if diff > best_diff:
            best_diff = diff
            best_q = q
            best_val = (length > 2 * c1)
            best_mask = mask1
        if best_diff == length - 1:
            break

    if best_val:
        new_indices = indices[best_mask]
    else:
        new_indices = indices[~best_mask]
    diff_values[best_q] = best_val
    return best_q, new_indices


def find_pair_fast_np(
    idx_arr: np.ndarray,
    n_bits: int,
) -> Optional[Tuple[int, int, int, Dict[int, bool]]]:
    """Fast basis-pair selector for dense states.

    Takes the first idx as anchor (index0) and looks for any single-qubit
    neighbor ``index0 ^ (1<<q)`` already in the state. Returns
    ``(diff_qubit, index0, index1, diff_values)`` where diff_values contains
    index0's bits at every qubit except diff_qubit (the MCRY controls that
    isolate {index0, index1}). Returns ``None`` if no neighbor is present
    (caller should fall back to the full bisection path).

    O(n) in time (amortized): a single scan over qubits with O(1) set lookup.
    For dense states (fill >= ~25%), the pigeonhole principle makes failure
    extremely unlikely.
    """
    index0 = int(idx_arr[0])
    idx_set = set(int(x) for x in idx_arr)
    for q in range(n_bits):
        partner = index0 ^ (1 << q)
        if partner in idx_set:
            diff_values: Dict[int, bool] = {}
            for qq in range(n_bits):
                if qq != q:
                    diff_values[qq] = bool((index0 >> qq) & 1)
            return q, index0, partner, diff_values
    return None


def minimize_controls_np(
    idx_arr: np.ndarray,
    index0: int,
    index1: int,
    diff_values: Dict[int, bool],
) -> Dict[int, bool]:
    """Drop controls that aren't needed to keep the MCRY match set = {i0, i1}.

    A control at qubit q can be dropped iff neither ``index0 ^ (1<<q)`` nor
    ``index1 ^ (1<<q)`` is in the state. Otherwise dropping q would make the
    MCRY also match that extra pair, rotating it with the wrong angle.
    """
    idx_set = set(int(x) for x in idx_arr)
    result: Dict[int, bool] = {}
    for q, v in diff_values.items():
        neighbor0 = index0 ^ (1 << q)
        neighbor1 = index1 ^ (1 << q)
        if neighbor0 in idx_set or neighbor1 in idx_set:
            result[q] = v  # keep: neighbor pair would leak in without this control
    return result


def select_informative_qubit(state: State, n_bits: int, supports: List[int]) -> int:
    best_qubit = 0
    min_diff = float("inf")
    length = len(state)
    assert length >= 2

    for qubit in supports:
        count0 = sum(1 for idx in state if not ((idx >> qubit) & 1))
        diff = abs(length - 2 * count0)
        if diff < min_diff:
            min_diff = diff
            best_qubit = qubit

    return best_qubit


def rotation_angles_optimization(
    angles: List[float],
    control_indices: List[int],
) -> Tuple[List[float], List[int]]:
    n_ctrls = len(control_indices)
    assert len(angles) == (1 << n_ctrls)

    min_sep = 1e-6
    dont_cares: Set[int] = set()

    for i in range(n_ctrls):
        is_dc = True
        for rot_idx in range(len(angles)):
            rev = rot_idx ^ (1 << i)
            if abs(angles[rev] - angles[rot_idx]) > min_sep:
                is_dc = False
                break
        if is_dc:
            dont_cares.add(i)

    if not dont_cares:
        return angles, control_indices

    new_ctrls: List[int] = []
    old_positions: List[int] = []
    for old_i in range(n_ctrls):
        if old_i not in dont_cares:
            new_ctrls.append(control_indices[old_i])
            old_positions.append(old_i)

    new_angles: List[float] = []
    for new_idx in range(1 << len(new_ctrls)):
        old_idx = 0
        for i, old_pos in enumerate(old_positions):
            old_idx |= ((new_idx >> i) & 1) << old_pos
        new_angles.append(angles[old_idx])

    return new_angles, new_ctrls


# --- Qubit-signature helpers for dense strategy ---

def get_qubit_signatures(state: State, n_bits: int) -> List[int]:
    sigs = [0] * n_bits
    for idx in sorted(state.keys()):
        for j in range(n_bits):
            sigs[j] = (sigs[j] << 1) | ((idx >> j) & 1)
    return sigs


def get_const1_signature(state: State) -> int:
    return (1 << len(state)) - 1


def get_ap_ry_angle(state: State, n_bits: int, qubit: int) -> Optional[float]:
    theta: Optional[float] = None
    for idx in state:
        rev = idx ^ (1 << qubit)
        if rev not in state:
            return None
        idx0 = idx & ~(1 << qubit)
        idx1 = idx | (1 << qubit)
        if idx0 not in state or idx1 not in state:
            return None
        w0 = state[idx0]
        w1 = state[idx1]
        t = 2.0 * math.atan(w1 / w0) if abs(w0) > EPS else math.pi
        if theta is None:
            theta = t
        elif abs(theta - t) > 1e-10:
            return None
    return theta
