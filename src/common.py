from __future__ import annotations
import math
from typing import Dict, List, Set, Tuple
import numpy as np
from state import EPS, State

def apply_controlled_cx_arr(idx_arr: np.ndarray, amp_arr: np.ndarray, control: int, ctrl_val: bool, target: int) -> Tuple[np.ndarray, np.ndarray]:
    bit = (idx_arr >> control & 1).astype(bool)
    match = bit == bool(ctrl_val)
    new_idx = np.where(match, idx_arr ^ np.int64(1) << target, idx_arr)
    return (new_idx, amp_arr)

def apply_mcry_arr(idx_arr: np.ndarray, amp_arr: np.ndarray, ctrls: List[int], phases: List[bool], theta: float, target: int) -> Tuple[np.ndarray, np.ndarray]:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    tgt_mask = np.int64(1) << target
    match = np.ones(len(idx_arr), dtype=bool)
    for q, p in zip(ctrls, phases):
        bit_q = (idx_arr >> q & 1).astype(bool)
        match &= bit_q == bool(p)
    non_match_idx = idx_arr[~match]
    non_match_amp = amp_arr[~match]
    match_idx = idx_arr[match]
    match_amp = amp_arr[match]
    if len(match_idx) == 0:
        keep = np.abs(non_match_amp) > EPS
        return (non_match_idx[keep], non_match_amp[keep])
    match_base = match_idx & ~tgt_mask
    match_tbit_1 = (match_idx >> target & 1).astype(bool)
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
    result_idx = np.concatenate([non_match_idx[keep_nm], unique_bases[keep0], (unique_bases | tgt_mask)[keep1]])
    result_amp = np.concatenate([non_match_amp[keep_nm], new_amp0[keep0], new_amp1[keep1]])
    return (result_idx, result_amp)

def maximize_difference_once_np(n_bits: int, indices: np.ndarray, diff_values: Dict[int, bool]) -> Tuple[int, np.ndarray]:
    length = len(indices)
    best_diff = -1
    best_q = 0
    best_val = False
    best_mask = None
    for q in range(n_bits):
        if q in diff_values:
            continue
        mask1 = (indices >> q & 1).astype(bool)
        c1 = int(mask1.sum())
        diff = abs(length - 2 * c1)
        if diff == length:
            continue
        if diff > best_diff:
            best_diff = diff
            best_q = q
            best_val = length > 2 * c1
            best_mask = mask1
        if best_diff == length - 1:
            break
    if best_val:
        new_indices = indices[best_mask]
    else:
        new_indices = indices[~best_mask]
    diff_values[best_q] = best_val
    return (best_q, new_indices)

def select_informative_qubit(state: State, n_bits: int, supports: List[int]) -> int:
    best_qubit = 0
    min_diff = float('inf')
    length = len(state)
    assert length >= 2
    for qubit in supports:
        count0 = sum((1 for idx in state if not idx >> qubit & 1))
        diff = abs(length - 2 * count0)
        if diff < min_diff:
            min_diff = diff
            best_qubit = qubit
    return best_qubit

def rotation_angles_optimization(angles: List[float], control_indices: List[int]) -> Tuple[List[float], List[int]]:
    n_ctrls = len(control_indices)
    assert len(angles) == 1 << n_ctrls
    min_sep = 1e-06
    dont_cares: Set[int] = set()
    for i in range(n_ctrls):
        is_dc = True
        for rot_idx in range(len(angles)):
            rev = rot_idx ^ 1 << i
            if abs(angles[rev] - angles[rot_idx]) > min_sep:
                is_dc = False
                break
        if is_dc:
            dont_cares.add(i)
    if not dont_cares:
        return (angles, control_indices)
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
            old_idx |= (new_idx >> i & 1) << old_pos
        new_angles.append(angles[old_idx])
    return (new_angles, new_ctrls)
